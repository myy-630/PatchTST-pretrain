#!/usr/bin/env python3
"""
PatchTST SSL Pretraining — Entry Point.
==========================================
Usage:
  python run.py --config configs/patchtst_ssl_pilot.yaml
  python run.py --config configs/patchtst_ssl_pilot.yaml --data-root /kaggle/input/...
  python run.py --config configs/patchtst_ssl_full.yaml --resume outputs/last.pt
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config, resolve_paths, save_resolved_config
from src.dataset import MaskedPretrainDataset
from src.dataset.ecg_dataset import collate_ssl
from src.models import PatchTSTBackbone, ReconstructionHead
from src.trainer import (
    PatchTSTTrainer,
    build_optimizer,
    build_scheduler,
    env_check,
)

log = logging.getLogger("run")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


# ═══════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PatchTST SSL Pretraining")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--data-root", default=None,
                   help="Override data directory root")
    p.add_argument("--output-dir", default=None,
                   help="Override output/checkpoint directory")
    p.add_argument("--resume", default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--dry-run", action="store_true",
                   help="Single batch forward+backward, then exit")
    return p.parse_args()


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # 1. Load + resolve config
    cfg = load_config(args.config)
    cfg = resolve_paths(
        cfg,
        data_root=args.data_root,
        output_dir=args.output_dir,
        config_path=args.config,
    )

    # 2. Environment
    env_check(cfg, args.data_root, args.output_dir)

    # 3. Seed
    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)
    import random; random.seed(seed)
    import numpy as np; np.random.seed(seed)

    # 4. Device
    device_str = cfg["training"]["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        if args.dry_run:
            log.warning("CUDA unavailable — dry-run on CPU despite YAML device=cuda")
            device_str = "cpu"
        else:
            raise RuntimeError("Config requires device=cuda but CUDA unavailable. "
                               "Use --dry-run for local CPU verification.")
    device = torch.device("cuda" if device_str == "cuda" and torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # 5. Datasets
    log.info("Building datasets …")
    train_ds = MaskedPretrainDataset(
        cfg["data"]["train_csv"],
        data_dir=cfg["data"]["data_dir"],
        patch_len=cfg["pretrain"]["patch_len"],
        stride=cfg["pretrain"]["stride"],
        mask_ratio=cfg["pretrain"]["mask_ratio"],
        mask_type=cfg["pretrain"].get("mask_type", "random"),
        revin_norm=cfg["pretrain"].get("revin_norm", True),
        seed=seed,
        mode="train",
        backend=cfg["data"].get("backend", "wfdb"),
        npy_dir=cfg["data"].get("npy_dir", None),
    )
    val_ds = MaskedPretrainDataset(
        cfg["data"]["val_csv"],
        data_dir=cfg["data"]["data_dir"],
        patch_len=cfg["pretrain"]["patch_len"],
        stride=cfg["pretrain"]["stride"],
        mask_ratio=cfg["pretrain"]["mask_ratio"],
        mask_type=cfg["pretrain"].get("mask_type", "random"),
        revin_norm=cfg["pretrain"].get("revin_norm", True),
        seed=seed,
        mode="val",
        backend=cfg["data"].get("backend", "wfdb"),
        npy_dir=cfg["data"].get("npy_dir", None),
    )
    log.info("Train: %d windows  |  Val: %d windows", len(train_ds), len(val_ds))

    # Optional subset for dry-run
    if args.dry_run:
        train_ds = Subset(train_ds, range(min(64, len(train_ds))))
        val_ds   = Subset(val_ds,   range(min(32, len(val_ds))))
        log.info("DRY-RUN: train=%d  val=%d", len(train_ds), len(val_ds))

    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=cfg["training"].get("num_workers", 0),
                              collate_fn=collate_ssl)
    val_loader   = DataLoader(val_ds, batch_size=bs, shuffle=False,
                              num_workers=0, collate_fn=collate_ssl)

    # 6. Model
    log.info("Building model …")
    model = PatchTSTBackbone(
        patch_len=cfg["pretrain"]["patch_len"],
        n_vars=cfg["data"]["n_channels"],
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        n_layers=cfg["model"]["n_layers"],
        d_ff=cfg["model"]["d_ff"],
        dropout=cfg["model"]["dropout"],
        max_patches=cfg["model"]["max_patches"],
    ).to(device)
    recon_head = ReconstructionHead(
        cfg["model"]["d_model"], cfg["pretrain"]["patch_len"]
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_params += sum(p.numel() for p in recon_head.parameters())
    log.info("Total parameters: %s", f"{n_params:,}")

    # 7. Optimizer + scheduler
    optimizer = build_optimizer(cfg, model, recon_head)
    steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch)

    # 8. Trainer
    trainer = PatchTSTTrainer(
        model, recon_head, optimizer, scheduler, cfg,
        train_loader, val_loader, device,
    )

    # Resume
    if args.resume:
        trainer = PatchTSTTrainer.from_resume(
            args.resume, model, recon_head, optimizer, scheduler, cfg,
            train_loader, val_loader, device,
        )

    # 9. Save resolved config
    save_resolved_config(cfg, Path(cfg["checkpoint"]["save_dir"]) / "resolved_config.yaml")

    # 10. Dry-run
    if args.dry_run:
        log.info("DRY-RUN: single forward+backward …")
        batch = next(iter(train_loader))
        masked = batch["masked_signal"].to(device)
        target = batch["target_signal"].to(device)
        mask   = batch["mask"].to(device)
        encoded = model(masked)
        B, C, N, D = encoded.shape
        flat = encoded.reshape(B*C, N, D)
        recon = recon_head(flat).reshape(B, C, N, -1)
        from src.models.loss import masked_reconstruction_loss
        loss = masked_reconstruction_loss(recon, target, mask)
        loss.backward()
        log.info("  loss=%.4f  backward OK  — dry-run passed.", loss.item())
        log.info("Dry-run complete. Exiting.")
        return

    # 11. Training loop
    log.info("Starting training (%d epochs) …", cfg["training"]["epochs"])
    # Resolve underlying dataset for set_epoch (handles Subset wrapper)
    ds_for_epoch = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
    for epoch in range(trainer.epoch, cfg["training"]["epochs"]):
        if hasattr(ds_for_epoch, "set_epoch"):
            ds_for_epoch.set_epoch(epoch)
        trainer.run_epoch()

    trainer.write_metrics_csv()
    log.info("Training complete. Best val_loss=%.4f", trainer.best_val_loss)


if __name__ == "__main__":
    main()
