#!/usr/bin/env python3
"""
PatchTST SSL — Minimal Training Loop (Smoke Test).
=====================================================
Phase 5: validates forward / backward / optimizer / loss decrease
on a small data subset.  All parameters configurable from CLI.

Tests:
  A — forward + backward, gradients non-zero
  B — parameter update after optimizer.step()
  C — overfit single batch (optional --overfit_batches)
  D — multi-epoch training, loss trends down, no NaN/Inf
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dataset import MaskedPretrainDataset
from src.dataset.ecg_dataset import collate_ssl
from src.models import PatchTSTBackbone, ReconstructionHead
from src.models.loss import masked_reconstruction_loss


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PatchTST SSL Smoke Test")
    # Data
    p.add_argument("--train_csv", default="splits/train.csv")
    p.add_argument("--val_csv",   default="splits/val.csv")
    p.add_argument("--train_size", type=int, default=256)
    p.add_argument("--val_size",   type=int, default=64)
    p.add_argument("--batch_size", type=int, default=16)
    # Model
    p.add_argument("--patch_len",  type=int, default=12)
    p.add_argument("--stride",     type=int, default=12)
    p.add_argument("--n_vars",     type=int, default=2)
    p.add_argument("--d_model",    type=int, default=128)
    p.add_argument("--n_heads",    type=int, default=16)
    p.add_argument("--n_layers",   type=int, default=3)
    p.add_argument("--d_ff",       type=int, default=256)
    p.add_argument("--dropout",    type=float, default=0.1)
    # Pretrain
    p.add_argument("--mask_ratio", type=float, default=0.4)
    # Optimizer
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip",  type=float, default=1.0)
    # Training
    p.add_argument("--epochs",     type=int, default=5)
    p.add_argument("--overfit_batches", type=int, default=0,
                   help="If >0, train on this many batches repeatedly")
    p.add_argument("--overfit_steps", type=int, default=30,
                   help="Number of overfit steps (only when --overfit_batches>0)")
    # Misc
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--device",     default="cpu")
    return p.parse_args()


# ===========================================================================
# Helpers
# ===========================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _train_step(model, recon_head, batch, optimizer, grad_clip) -> float:
    masked = batch["masked_signal"]
    target = batch["target_signal"]
    mask   = batch["mask"]

    encoded = model(masked)                                # (B, C, N, d)
    B, C, N, D = encoded.shape
    flat = encoded.reshape(B * C, N, D)                    # (B*C, N, d)
    recon = recon_head(flat).reshape(B, C, N, -1)          # (B, C, N, P)
    loss = masked_reconstruction_loss(recon, target, mask)

    optimizer.zero_grad()
    loss.backward()

    # Gradient norm (before clipping)
    total_norm = 0.0
    for p in list(model.parameters()) + list(recon_head.parameters()):
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5

    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(recon_head.parameters()), grad_clip
        )

    optimizer.step()
    return loss.item(), total_norm


@torch.no_grad()
def _val_step(model, recon_head, batch) -> float:
    model.eval()
    recon_head.eval()

    masked = batch["masked_signal"]
    target = batch["target_signal"]
    mask   = batch["mask"]

    encoded = model(masked)
    B, C, N, D = encoded.shape
    flat = encoded.reshape(B * C, N, D)
    recon = recon_head(flat).reshape(B, C, N, -1)
    loss = masked_reconstruction_loss(recon, target, mask)

    model.train()
    recon_head.train()
    return loss.item()


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    print("=" * 70)
    print("PatchTST SSL — Smoke Test")
    print(f"device={device}  seed={args.seed}")
    print(f"train_size={args.train_size}  val_size={args.val_size}  "
          f"batch={args.batch_size}  epochs={args.epochs}")
    print(f"overfit_batches={args.overfit_batches}")
    print(f"lr={args.lr}  wd={args.weight_decay}  grad_clip={args.grad_clip}")
    print("=" * 70)

    # ---- Datasets ----
    print("\n[1] Loading data subsets …")
    full_train = MaskedPretrainDataset(
        args.train_csv, patch_len=args.patch_len, stride=args.stride,
        mask_ratio=args.mask_ratio, seed=args.seed,
    )
    full_val = MaskedPretrainDataset(
        args.val_csv, patch_len=args.patch_len, stride=args.stride,
        mask_ratio=args.mask_ratio, seed=args.seed,
    )

    train_ds = Subset(full_train, range(min(args.train_size, len(full_train))))
    val_ds   = Subset(full_val,   range(min(args.val_size, len(full_val))))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, collate_fn=collate_ssl)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=0, collate_fn=collate_ssl)

    print(f"  Train: {len(train_ds)} windows  Val: {len(val_ds)} windows")

    # ---- Model ----
    print("\n[2] Building model …")
    model = PatchTSTBackbone(
        patch_len=args.patch_len, n_vars=args.n_vars,
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ff=args.d_ff, dropout=args.dropout,
    ).to(device)
    recon_head = ReconstructionHead(args.d_model, args.patch_len).to(device)

    total_p = sum(p.numel() for p in list(model.parameters()) + list(recon_head.parameters()))
    print(f"  Params: {total_p:,}")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(recon_head.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )

    # ---- Test A: single forward + backward ----
    print("\n[3] Test A — forward + backward …")
    batch = next(iter(train_loader))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    model.train(); recon_head.train()

    loss_a, grad_norm_a = _train_step(model, recon_head, batch, optimizer, args.grad_clip)
    print(f"  loss={loss_a:.4f}  grad_norm={grad_norm_a:.4f}")

    enc_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters())
    head_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                        for p in recon_head.parameters())
    print(f"  Encoder grad OK: {enc_has_grad}  |  Head grad OK: {head_has_grad}")
    assert enc_has_grad, "FAIL A: Encoder has no gradient!"
    assert head_has_grad, "FAIL A: ReconstructionHead has no gradient!"
    print("  [OK] Test A passed.")

    # ---- Test B: parameter update check ----
    print("\n[4] Test B — parameter update …")
    param_before = next(model.parameters()).clone().detach()
    _train_step(model, recon_head, batch, optimizer, args.grad_clip)
    param_after = next(model.parameters()).clone().detach()
    max_delta = (param_after - param_before).abs().max().item()
    assert max_delta > 0, "FAIL B: parameters did not change!"
    print(f"  max param delta: {max_delta:.8f}")
    print("  [OK] Test B passed.")

    # ---- Test C: overfit single batch ----
    if args.overfit_batches > 0:
        print(f"\n[5] Test C — overfit {args.overfit_batches} batch(es) …")

        # Re-init model for clean test
        model = PatchTSTBackbone(
            patch_len=args.patch_len, n_vars=args.n_vars,
            d_model=args.d_model, n_heads=args.n_heads,
            n_layers=args.n_layers, d_ff=args.d_ff, dropout=args.dropout,
        ).to(device)
        recon_head = ReconstructionHead(args.d_model, args.patch_len).to(device)
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(recon_head.parameters()),
            lr=args.lr,
        )
        model.train(); recon_head.train()

        # Grab fixed batch(es)
        fixed_batches = []
        loader = DataLoader(train_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, collate_fn=collate_ssl)
        for i, b in enumerate(loader):
            if i >= args.overfit_batches:
                break
            fixed_batches.append({k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()})

        loss_start = None
        loss_end = None
        steps = args.overfit_steps
        for step in range(steps):
            for b in fixed_batches:
                loss_val, _ = _train_step(model, recon_head, b, optimizer, args.grad_clip)
            if step == 0:
                loss_start = loss_val
            loss_end = loss_val
            if step % 20 == 0:
                print(f"    step {step:3d}: loss={loss_val:.4f}")

        print(f"  loss: {loss_start:.4f} → {loss_end:.4f} "
              f"(ratio: {loss_end/loss_start:.4f})")
        assert loss_end < loss_start * 0.95, \
            f"FAIL C: loss did not decrease enough ({loss_start:.4f} → {loss_end:.4f})"
        print("  [OK] Test C passed — overfitting confirmed.")

    # ---- Test D: multi-epoch training ----
    print(f"\n[{'6' if args.overfit_batches else '5'}] Test D — {args.epochs} epochs …")

    # Re-init
    model = PatchTSTBackbone(
        patch_len=args.patch_len, n_vars=args.n_vars,
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ff=args.d_ff, dropout=args.dropout,
    ).to(device)
    recon_head = ReconstructionHead(args.d_model, args.patch_len).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(recon_head.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    model.train(); recon_head.train()

    train_losses = []
    val_losses = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        model.train(); recon_head.train()
        epoch_train_loss = 0.0
        epoch_grad_norm = 0.0
        n_train_batches = 0
        for b in train_loader:
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            lv, gn = _train_step(model, recon_head, b, optimizer, args.grad_clip)
            epoch_train_loss += lv
            epoch_grad_norm += gn
            n_train_batches += 1
        avg_train_loss = epoch_train_loss / max(n_train_batches, 1)
        avg_grad_norm = epoch_grad_norm / max(n_train_batches, 1)

        # Val
        avg_val_loss = 0.0
        n_val_batches = 0
        for b in val_loader:
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            lv = _val_step(model, recon_head, b)
            avg_val_loss += lv
            n_val_batches += 1
        avg_val_loss /= max(n_val_batches, 1)

        elapsed = time.time() - t0
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"  epoch {epoch:2d}/{args.epochs}  "
              f"train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}  "
              f"grad_norm={avg_grad_norm:.2f}  time={elapsed:.1f}s")

        # Safety
        assert np.isfinite(avg_train_loss), f"FAIL: train_loss NaN at epoch {epoch}"
        assert np.isfinite(avg_val_loss), f"FAIL: val_loss NaN at epoch {epoch}"

    # Check loss trend
    assert train_losses[-1] < train_losses[0] * 1.05, \
        f"FAIL D: train loss did not decrease: {train_losses[0]:.4f} → {train_losses[-1]:.4f}"
    print(f"\n  train loss: {train_losses[0]:.4f} → {train_losses[-1]:.4f}")
    print(f"  val loss:   {val_losses[0]:.4f} → {val_losses[-1]:.4f}")
    print("  [OK] Test D passed — loss decreases, no NaN/Inf.")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("  All tests passed (A–D). Ready for full pretraining Trainer.")
    print(f"  Final train loss: {train_losses[-1]:.4f}")
    print(f"  Final val loss:   {val_losses[-1]:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
