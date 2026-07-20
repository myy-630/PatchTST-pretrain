"""Trainer for contrastive ECG self-supervised pretraining."""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from src.models import mean_pool_patchtst, nt_xent_loss

log = logging.getLogger("contrastive_trainer")


class ContrastiveTrainer:
    def __init__(
        self,
        encoder: nn.Module,
        projector: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None,
        cfg: dict,
        train_loader,
        val_loader,
        device: torch.device,
    ):
        self.encoder = encoder
        self.projector = projector
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.temperature = cfg.get("contrastive", {}).get("temperature", 0.1)
        self.grad_clip = cfg["training"].get("grad_clip", 1.0)
        self.use_amp = cfg["training"].get("mixed_precision", "fp32") == "fp16"
        self.amp_device = device.type if self.use_amp else "cpu"
        self.scaler = GradScaler(self.amp_device, enabled=self.use_amp)
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.metrics: list[dict] = []
        self.save_dir = Path(cfg["checkpoint"]["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _forward_loss(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        h1 = mean_pool_patchtst(self.encoder(view1))
        h2 = mean_pool_patchtst(self.encoder(view2))
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        return nt_xent_loss(z1, z2, temperature=self.temperature)

    def train_epoch(self) -> tuple[float, float, float]:
        self.encoder.train()
        self.projector.train()
        total_loss = 0.0
        total_grad_norm = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in self.train_loader:
            view1 = batch["view1"].to(self.device)
            view2 = batch["view2"].to(self.device)
            with autocast(self.amp_device, enabled=self.use_amp):
                loss = self._forward_loss(view1, view2)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            params = list(self.encoder.parameters()) + list(self.projector.parameters())
            grad_norm = 0.0
            for p in params:
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5

            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(params, self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            total_grad_norm += grad_norm
            n_batches += 1
            self.global_step += 1

        elapsed = time.time() - t0
        return total_loss / max(n_batches, 1), total_grad_norm / max(n_batches, 1), elapsed

    @torch.no_grad()
    def val_epoch(self) -> float:
        self.encoder.eval()
        self.projector.eval()
        total_loss = 0.0
        n_batches = 0
        for batch in self.val_loader:
            view1 = batch["view1"].to(self.device)
            view2 = batch["view2"].to(self.device)
            with autocast(self.amp_device, enabled=self.use_amp):
                loss = self._forward_loss(view1, view2)
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    def run_epoch(self) -> None:
        train_loss, grad_norm, elapsed = self.train_epoch()
        val_loss = self.val_epoch()
        lr = self.optimizer.param_groups[0]["lr"]
        wps = len(self.train_loader.dataset) / elapsed if elapsed > 0 else 0.0
        gpu_mem = ""
        if torch.cuda.is_available():
            gpu_mem = f"{torch.cuda.max_memory_allocated() / 1024**3:.1f}G"

        log.info(
            "epoch %3d | train_loss=%.4f val_loss=%.4f lr=%.2e grad_norm=%.2f "
            "time=%d.0fs wps=%.0f gpu=%s",
            self.epoch, train_loss, val_loss, lr, grad_norm, elapsed, wps, gpu_mem,
        )
        self.metrics.append({
            "epoch": self.epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr,
            "grad_norm": grad_norm,
            "elapsed_s": elapsed,
            "windows_per_sec": wps,
        })

        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
        self.epoch += 1
        self._save_checkpoint(is_best=is_best)

    def _save_checkpoint(self, is_best: bool = False) -> None:
        ckpt = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "encoder_state_dict": self.encoder.state_dict(),
            "model_state_dict": self.encoder.state_dict(),
            "projector_state_dict": self.projector.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.cfg,
        }
        torch.save(ckpt, self.save_dir / "last.pt")
        if is_best:
            torch.save(ckpt, self.save_dir / "best.pt")
            torch.save(self.encoder.state_dict(), self.save_dir / "pretrained_encoder.pt")

    def write_metrics_csv(self) -> None:
        if not self.metrics:
            return
        with (self.save_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.metrics[0].keys())
            writer.writeheader()
            writer.writerows(self.metrics)


def build_contrastive_optimizer(cfg: dict, encoder: nn.Module, projector: nn.Module) -> torch.optim.Optimizer:
    t = cfg["training"]
    params = list(encoder.parameters()) + list(projector.parameters())
    if t.get("optimizer", "adamw") == "adamw":
        return torch.optim.AdamW(params, lr=t["lr"], weight_decay=t.get("weight_decay", 0.0))
    return torch.optim.Adam(params, lr=t["lr"])
