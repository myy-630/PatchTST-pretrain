#!/usr/bin/env python3
"""Linear probe evaluation for PatchTST ECG SSL representations.

This script freezes a pretrained PatchTST encoder and trains only a linear AF
classification head on labeled train/val windows. It never reads test.csv.
Default label policy is strict: AF is positive, Normal is negative, and
Unlabeled/Mixed/Other windows are excluded from supervised probe training.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.config import load_config, resolve_paths, save_resolved_config
from src.dataset.ecg_dataset import BaseECGDataset, REVIN_EPS
from src.models import PatchTSTBackbone


log = logging.getLogger("linear_probe")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


class LinearProbeDataset(BaseECGDataset):
    """ECG dataset for frozen-encoder linear probe."""

    def __init__(
        self,
        csv_path: str | Path,
        data_dir: str | Path,
        patch_len: int,
        stride: int,
        backend: str = "wfdb",
        npy_dir: str | Path | None = None,
        revin_norm: bool = True,
        label_policy: str = "af_normal_only",
    ):
        super().__init__(csv_path, data_dir=data_dir, backend=backend, npy_dir=npy_dir)
        self.patch_len = patch_len
        self.stride = stride
        self.revin_norm = revin_norm
        self.label_policy = label_policy

        if label_policy == "af_normal_only":
            self.df = self.df[self.df["af_label"].isin(["AF", "Normal"])].reset_index(drop=True)
        elif label_policy == "af_vs_all":
            self.df = self.df.copy().reset_index(drop=True)
        else:
            raise ValueError(f"Unsupported label_policy: {label_policy}")

        counts = self.df["af_label"].value_counts(dropna=False).to_dict()
        log.info("LinearProbeDataset: %d labeled windows after %s filter: %s",
                 len(self.df), label_policy, counts)

    def __getitem__(self, idx: int) -> dict:
        sample = super().__getitem__(idx)
        signal = sample["signal"]

        if self.revin_norm:
            mean = signal.mean(dim=-1, keepdim=True)
            std = signal.std(dim=-1, keepdim=True, unbiased=False)
            signal = (signal - mean) / (std + REVIN_EPS)

        patches = signal.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        label = 1.0 if sample["af_label"] == "AF" else 0.0
        return {
            "window_id": sample["window_id"],
            "record_id": sample["record_id"],
            "patches": patches,
            "label": torch.tensor(label, dtype=torch.float32),
        }


def collate_probe(batch: list[dict]) -> dict:
    return {
        "window_id": [x["window_id"] for x in batch],
        "record_id": [x["record_id"] for x in batch],
        "patches": torch.stack([x["patches"] for x in batch], dim=0),
        "label": torch.stack([x["label"] for x in batch], dim=0),
    }


class LinearProbeClassifier(nn.Module):
    """Frozen PatchTST encoder + linear binary classifier."""

    def __init__(self, encoder: PatchTSTBackbone, d_model: int, n_channels: int):
        super().__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(d_model * n_channels, 1)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        self.encoder.eval()
        with torch.no_grad():
            encoded = self.encoder(patches)       # (B, C, N, D)
            pooled = encoded.mean(dim=2)          # (B, C, D)
            features = pooled.flatten(start_dim=1) # (B, C*D)
        return self.classifier(features).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linear probe for pretrained PatchTST encoder")
    parser.add_argument("--config", required=True, help="SSL YAML config used to define encoder architecture")
    parser.add_argument("--pretrained", required=True, help="Path to pretrained_encoder.pt or SSL checkpoint")
    parser.add_argument("--data-root", default=None, help="Override ECG data root")
    parser.add_argument("--train-csv", default=None, help="Override supervised train CSV; test CSV is never used")
    parser.add_argument("--val-csv", default=None, help="Override supervised validation CSV; test CSV is never used")
    parser.add_argument("--output-dir", required=True, help="Output directory for probe metrics/checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-policy", choices=["af_normal_only", "af_vs_all"], default="af_normal_only")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_encoder_weights(path: str | Path, device: torch.device) -> dict[str, Any]:
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict):
        for key in ["encoder_state_dict", "model_state_dict"]:
            if key in obj:
                return obj[key]
    return obj


def build_encoder(cfg: dict, device: torch.device) -> PatchTSTBackbone:
    encoder = PatchTSTBackbone(
        patch_len=cfg["pretrain"]["patch_len"],
        n_vars=cfg["data"]["n_channels"],
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        n_layers=cfg["model"]["n_layers"],
        d_ff=cfg["model"]["d_ff"],
        dropout=cfg["model"]["dropout"],
        max_patches=cfg["model"]["max_patches"],
    )
    return encoder.to(device)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    metrics: dict[str, float] = {}
    metrics["auroc"] = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")
    metrics["auprc"] = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall_sensitivity"] = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    metrics["tp"] = float(tp)
    metrics["fp"] = float(fp)
    metrics["tn"] = float(tn)
    metrics["fn"] = float(fn)
    return metrics


def run_eval(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> dict:
    model.eval()
    losses = []
    probs = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            x = batch["patches"].to(device)
            y = batch["label"].to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            losses.append(loss.item())
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(y.cpu().numpy())

    y_prob = np.concatenate(probs)
    y_true = np.concatenate(labels).astype(int)
    metrics = compute_metrics(y_true, y_prob)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_config(args.config)
    cfg = resolve_paths(cfg, data_root=args.data_root, output_dir=None, config_path=args.config)
    if args.train_csv:
        cfg["data"]["train_csv"] = str(Path(args.train_csv).resolve())
    if args.val_csv:
        cfg["data"]["val_csv"] = str(Path(args.val_csv).resolve())
    save_resolved_config(cfg, out_dir / "resolved_ssl_config.yaml")

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Linear probe requested cuda but CUDA is unavailable.")
    device = torch.device(device_str)

    train_ds = LinearProbeDataset(
        cfg["data"]["train_csv"],
        data_dir=cfg["data"]["data_dir"],
        patch_len=cfg["pretrain"]["patch_len"],
        stride=cfg["pretrain"]["stride"],
        backend=cfg["data"].get("backend", "wfdb"),
        npy_dir=cfg["data"].get("npy_dir"),
        revin_norm=cfg["pretrain"].get("revin_norm", True),
        label_policy=args.label_policy,
    )
    val_ds = LinearProbeDataset(
        cfg["data"]["val_csv"],
        data_dir=cfg["data"]["data_dir"],
        patch_len=cfg["pretrain"]["patch_len"],
        stride=cfg["pretrain"]["stride"],
        backend=cfg["data"].get("backend", "wfdb"),
        npy_dir=cfg["data"].get("npy_dir"),
        revin_norm=cfg["pretrain"].get("revin_norm", True),
        label_policy=args.label_policy,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_probe,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_probe,
    )

    encoder = build_encoder(cfg, device)
    state = load_encoder_weights(args.pretrained, device)
    encoder.load_state_dict(state)
    model = LinearProbeClassifier(
        encoder,
        d_model=cfg["model"]["d_model"],
        n_channels=cfg["data"]["n_channels"],
    ).to(device)

    pos = float((train_ds.df["af_label"] == "AF").sum())
    neg = float((train_ds.df["af_label"] != "AF").sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log.info("Linear probe: train=%d val=%d pos_weight=%.3f", len(train_ds), len(val_ds), pos_weight.item())
    log.info("Train labels: %s", train_ds.df["af_label"].value_counts().to_dict())
    log.info("Val labels:   %s", val_ds.df["af_label"].value_counts().to_dict())

    metrics_rows = []
    best_auroc = -float("inf")

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            x = batch["patches"].to(device)
            y = batch["label"].to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_metrics = run_eval(model, val_loader, loss_fn, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        metrics_rows.append(row)

        log.info(
            "epoch %03d | train_loss=%.4f val_loss=%.4f auroc=%.4f auprc=%.4f "
            "f1=%.4f sens=%.4f spec=%.4f",
            epoch,
            row["train_loss"],
            row["val_loss"],
            row["val_auroc"],
            row["val_auprc"],
            row["val_f1"],
            row["val_recall_sensitivity"],
            row["val_specificity"],
        )

        if row["val_auroc"] > best_auroc:
            best_auroc = row["val_auroc"]
            torch.save({
                "classifier_state_dict": model.classifier.state_dict(),
                "config": cfg,
                "args": vars(args),
                "epoch": epoch,
                "val_metrics": val_metrics,
            }, out_dir / "best_linear_probe.pt")

        with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metrics_rows)

    log.info("Linear probe complete. Best val AUROC=%.4f", best_auroc)


if __name__ == "__main__":
    main()
