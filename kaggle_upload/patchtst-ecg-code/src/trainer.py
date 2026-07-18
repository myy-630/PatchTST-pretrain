"""
PatchTST SSL Trainer.
======================
Handles train/val loops, checkpoint I/O, resume, AMP, logging.
Does NOT parse YAML or CLI — receives a resolved config dict.
"""

import csv
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.dataset.ecg_dataset import collate_ssl
from src.models.loss import masked_reconstruction_loss

log = logging.getLogger("trainer")


class PatchTSTTrainer:
    """Minimal but complete SSL pretraining trainer."""

    def __init__(
        self,
        model: nn.Module,
        recon_head: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None,
        cfg: dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ):
        self.model = model
        self.recon_head = recon_head
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.use_amp = cfg["training"].get("mixed_precision", "fp32") == "fp16"
        self.grad_clip = cfg["training"].get("grad_clip", 1.0)
        self.amp_device = device.type if self.use_amp else "cpu"
        self.scaler = GradScaler(self.amp_device, enabled=self.use_amp)

        # State
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.metrics: list[dict] = []

        # Output dir
        self.save_dir = Path(cfg["checkpoint"]["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # ── train epoch ──────────────────────────────────
    def train_epoch(self) -> float:
        self.model.train()
        self.recon_head.train()
        total_loss = 0.0
        total_grad_norm = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in self.train_loader:
            masked = batch["masked_signal"].to(self.device)
            target = batch["target_signal"].to(self.device)
            mask   = batch["mask"].to(self.device)

            with autocast(self.amp_device, enabled=self.use_amp):
                encoded = self.model(masked)                       # (B, C, N, d)
                B, C, N, D = encoded.shape
                flat = encoded.reshape(B * C, N, D)               # (B*C, N, d)
                recon = self.recon_head(flat).reshape(B, C, N, -1) # (B, C, N, P)
                loss = masked_reconstruction_loss(recon, target, mask)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # Grad norm (before clipping)
            params = list(self.model.parameters()) + list(self.recon_head.parameters())
            gn = 0.0
            for p in params:
                if p.grad is not None:
                    gn += p.grad.data.norm(2).item() ** 2
            gn = gn ** 0.5

            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(params, self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            total_grad_norm += gn
            n_batches += 1
            self.global_step += 1

        elapsed = time.time() - t0
        avg_loss = total_loss / max(n_batches, 1)
        avg_gn = total_grad_norm / max(n_batches, 1)
        return avg_loss, avg_gn, elapsed

    # ── val epoch ────────────────────────────────────
    @torch.no_grad()
    def val_epoch(self) -> float:
        self.model.eval()
        self.recon_head.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            masked = batch["masked_signal"].to(self.device)
            target = batch["target_signal"].to(self.device)
            mask   = batch["mask"].to(self.device)

            with autocast(self.amp_device, enabled=self.use_amp):
                encoded = self.model(masked)
                B, C, N, D = encoded.shape
                flat = encoded.reshape(B * C, N, D)
                recon = self.recon_head(flat).reshape(B, C, N, -1)
                loss = masked_reconstruction_loss(recon, target, mask)

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    # ── full epoch ───────────────────────────────────
    def run_epoch(self) -> None:
        train_loss, grad_norm, elapsed = self.train_epoch()
        val_loss = self.val_epoch()

        lr = self.optimizer.param_groups[0]["lr"]
        wps = len(self.train_loader.dataset) / elapsed if elapsed > 0 else 0

        gpu_mem = ""
        if torch.cuda.is_available():
            gpu_mem = f"{torch.cuda.max_memory_allocated() / 1024**3:.1f}G"

        log.info(
            "epoch %3d | train_loss=%.4f val_loss=%.4f lr=%.2e grad_norm=%.2f "
            "time=%d.0fs wps=%.0f gpu=%s",
            self.epoch, train_loss, val_loss, lr, grad_norm,
            elapsed, wps, gpu_mem,
        )

        self.metrics.append({
            "epoch": self.epoch, "train_loss": train_loss,
            "val_loss": val_loss, "lr": lr, "grad_norm": grad_norm,
            "elapsed_s": elapsed, "windows_per_sec": wps,
        })

        # Best model
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss

        # Increment epoch BEFORE saving so checkpoint reflects next epoch to run
        self.epoch += 1
        self._save_checkpoint(is_best=is_best)

    # ── checkpoint ───────────────────────────────────
    def _save_checkpoint(self, is_best: bool = False) -> None:
        ckpt = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "encoder_state_dict": self.model.state_dict(),
            "recon_head_state_dict": self.recon_head.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.cfg,
        }
        torch.save(ckpt, self.save_dir / "last.pt")
        if is_best:
            torch.save(ckpt, self.save_dir / "best.pt")
            # Also export encoder-only weights for downstream AF fine-tuning
            torch.save(self.model.state_dict(), self.save_dir / "pretrained_encoder.pt")

    # ── resume ───────────────────────────────────────
    @classmethod
    def from_resume(
        cls, resume_path: str | Path,
        model: nn.Module, recon_head: nn.Module,
        optimizer: torch.optim.Optimizer, scheduler: Any | None,
        cfg: dict, train_loader: DataLoader, val_loader: DataLoader,
        device: torch.device,
    ) -> "PatchTSTTrainer":
        """Create Trainer, restore state from checkpoint."""
        trainer = cls(model, recon_head, optimizer, scheduler, cfg,
                      train_loader, val_loader, device)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)

        trainer.model.load_state_dict(ckpt["model_state_dict"])
        trainer.recon_head.load_state_dict(ckpt["recon_head_state_dict"])
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if trainer.scheduler and ckpt.get("scheduler_state_dict"):
            trainer.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        trainer.scaler.load_state_dict(ckpt["scaler_state_dict"])
        trainer.epoch = ckpt["epoch"]
        trainer.global_step = ckpt["global_step"]
        trainer.best_val_loss = ckpt["best_val_loss"]
        trainer.metrics = ckpt.get("metrics", [])

        log.info("Resumed from %s (epoch=%d, step=%d, best_val=%.4f)",
                 resume_path, trainer.epoch, trainer.global_step, trainer.best_val_loss)
        return trainer

    # ── CSV logging ──────────────────────────────────
    def write_metrics_csv(self) -> None:
        if not self.metrics:
            return
        path = self.save_dir / "metrics.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.metrics[0].keys())
            w.writeheader()
            w.writerows(self.metrics)


# ── helpers ───────────────────────────────────────────

def build_optimizer(cfg: dict, model: nn.Module, recon_head: nn.Module) -> torch.optim.Optimizer:
    t = cfg["training"]
    params = list(model.parameters()) + list(recon_head.parameters())
    if t.get("optimizer", "adamw") == "adamw":
        return torch.optim.AdamW(params, lr=t["lr"],
                                 weight_decay=t.get("weight_decay", 0.0))
    return torch.optim.Adam(params, lr=t["lr"])


def build_scheduler(cfg: dict, optimizer: torch.optim.Optimizer,
                    steps_per_epoch: int) -> Any | None:
    t = cfg["training"]
    name = t.get("scheduler", "none")
    if name == "none":
        return None
    if name == "cosine":
        total_steps = t["epochs"] * steps_per_epoch
        warmup = t.get("warmup_epochs", 0) * steps_per_epoch
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        if warmup > 0:
            warmup_sch = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                                  total_iters=warmup)
            cosine_sch = CosineAnnealingLR(optimizer, T_max=total_steps - warmup)
            return SequentialLR(optimizer, schedulers=[warmup_sch, cosine_sch],
                                milestones=[warmup])
        return CosineAnnealingLR(optimizer, T_max=total_steps)
    return None


def env_check(cfg: dict, data_root: str | None, output_dir: str | None) -> None:
    """Print environment info; exit if critical checks fail."""
    import platform
    import sys

    log.info("=" * 60)
    log.info("Environment Check")
    log.info("  Python:     %s", platform.python_version())
    log.info("  PyTorch:    %s", torch.__version__)
    log.info("  CUDA avail: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        log.info("  GPU:        %s", torch.cuda.get_device_name(0))
    log.info("  CWD:        %s", Path.cwd())
    log.info("  data-root:  %s", data_root or "(YAML)")
    log.info("  output-dir: %s", output_dir or "(YAML)")

    # Check data files exist
    for key in ["train_csv", "val_csv", "data_dir"]:
        p = Path(cfg["data"][key])
        if not p.exists():
            raise FileNotFoundError(f"{key} not found: {p}")
    log.info("  train_csv:  %s  (exists)", cfg["data"]["train_csv"])
    log.info("  val_csv:    %s  (exists)", cfg["data"]["val_csv"])
    log.info("  data_dir:   %s  (exists)", cfg["data"]["data_dir"])
    log.info("=" * 60)
