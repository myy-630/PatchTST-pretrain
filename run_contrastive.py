#!/usr/bin/env python3
"""PatchTST contrastive self-supervised pretraining entry point."""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config, resolve_paths, save_resolved_config
from src.contrastive_trainer import ContrastiveTrainer, build_contrastive_optimizer
from src.dataset.contrastive_dataset import ContrastivePretrainDataset, collate_contrastive
from src.models import PatchTSTBackbone, ProjectionHead
from src.trainer import build_scheduler, env_check


log = logging.getLogger("run_contrastive")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PatchTST contrastive pretraining")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_encoder(cfg: dict, device: torch.device) -> PatchTSTBackbone:
    return PatchTSTBackbone(
        patch_len=cfg["pretrain"]["patch_len"],
        n_vars=cfg["data"]["n_channels"],
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        n_layers=cfg["model"]["n_layers"],
        d_ff=cfg["model"]["d_ff"],
        dropout=cfg["model"]["dropout"],
        max_patches=cfg["model"]["max_patches"],
    ).to(device)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = resolve_paths(cfg, data_root=args.data_root, output_dir=args.output_dir, config_path=args.config)
    env_check(cfg, args.data_root, args.output_dir)

    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device_str = cfg["training"]["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        if args.dry_run:
            log.warning("CUDA unavailable; running dry-run on CPU.")
            device_str = "cpu"
        else:
            raise RuntimeError("Config requires CUDA but CUDA is unavailable.")
    device = torch.device("cuda" if device_str == "cuda" and torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    contrast_cfg = cfg.get("contrastive", {})
    train_ds = ContrastivePretrainDataset(
        cfg["data"]["train_csv"],
        data_dir=cfg["data"]["data_dir"],
        patch_len=cfg["pretrain"]["patch_len"],
        stride=cfg["pretrain"]["stride"],
        revin_norm=cfg["pretrain"].get("revin_norm", True),
        seed=seed,
        mode="train",
        backend=cfg["data"].get("backend", "wfdb"),
        npy_dir=cfg["data"].get("npy_dir"),
        aug_strength=contrast_cfg.get("augmentation_strength", "medium"),
    )
    val_ds = ContrastivePretrainDataset(
        cfg["data"]["val_csv"],
        data_dir=cfg["data"]["data_dir"],
        patch_len=cfg["pretrain"]["patch_len"],
        stride=cfg["pretrain"]["stride"],
        revin_norm=cfg["pretrain"].get("revin_norm", True),
        seed=seed,
        mode="val",
        backend=cfg["data"].get("backend", "wfdb"),
        npy_dir=cfg["data"].get("npy_dir"),
        aug_strength=contrast_cfg.get("augmentation_strength", "medium"),
    )
    log.info("Train: %d windows | Val: %d windows", len(train_ds), len(val_ds))

    if args.dry_run:
        train_ds = Subset(train_ds, range(min(64, len(train_ds))))
        val_ds = Subset(val_ds, range(min(32, len(val_ds))))
        log.info("DRY-RUN: train=%d val=%d", len(train_ds), len(val_ds))

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg["training"].get("num_workers", 0),
        collate_fn=collate_contrastive,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_contrastive,
    )

    encoder = build_encoder(cfg, device)
    projector = ProjectionHead(
        in_dim=cfg["model"]["d_model"] * cfg["data"]["n_channels"],
        hidden_dim=contrast_cfg.get("projection_hidden_dim", 256),
        out_dim=contrast_cfg.get("projection_dim", 128),
        dropout=cfg["model"].get("dropout", 0.1),
    ).to(device)
    n_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in projector.parameters())
    log.info("Total parameters: %s", f"{n_params:,}")

    optimizer = build_contrastive_optimizer(cfg, encoder, projector)
    scheduler = build_scheduler(cfg, optimizer, len(train_loader))
    trainer = ContrastiveTrainer(encoder, projector, optimizer, scheduler, cfg, train_loader, val_loader, device)

    save_resolved_config(cfg, Path(cfg["checkpoint"]["save_dir"]) / "resolved_config.yaml")

    if args.dry_run:
        batch = next(iter(train_loader))
        loss = trainer._forward_loss(batch["view1"].to(device), batch["view2"].to(device))
        loss.backward()
        log.info("DRY-RUN loss=%.4f backward OK", loss.item())
        return

    log.info("Starting contrastive training (%d epochs)", cfg["training"]["epochs"])
    ds_for_epoch = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
    for epoch in range(trainer.epoch, cfg["training"]["epochs"]):
        if hasattr(ds_for_epoch, "set_epoch"):
            ds_for_epoch.set_epoch(epoch)
        trainer.run_epoch()

    trainer.write_metrics_csv()
    log.info("Contrastive training complete. Best val_loss=%.4f", trainer.best_val_loss)


if __name__ == "__main__":
    main()
