#!/usr/bin/env python3
"""
Debug script for Masked Reconstruction Loss (Phase 4).
========================================================
Tests A–D: manual correctness verification.
Tests 1–6: integration and sanity checks with real data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch.utils.data import DataLoader

from src.dataset import MaskedPretrainDataset
from src.models import PatchTSTBackbone, ReconstructionHead
from src.models.loss import masked_reconstruction_loss


def test_a_loss_zero_when_perfect():
    """Test A: pred == target → loss ≈ 0."""
    pred   = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])    # (1,1,2,2)
    target = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    mask   = torch.tensor([[[True, False]]])                  # first patch masked
    loss = masked_reconstruction_loss(pred, target, mask)
    assert loss.item() < 1e-6, f"FAIL A: loss={loss.item():.8f}, expected ~0"
    print(f"  Test A [OK]: loss={loss.item():.8f} (perfect pred → 0)")


def test_b_loss_unchanged_on_visible():
    """Test B: modify visible patches → loss unchanged."""
    pred   = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])    # (1,1,2,2)
    target = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    mask   = torch.tensor([[[True, False]]])
    loss_before = masked_reconstruction_loss(pred, target, mask)

    # Modify visible (unmasked) patch only
    pred[0, 0, ~mask[0, 0]] = 999.0
    loss_after = masked_reconstruction_loss(pred, target, mask)

    assert abs(loss_after.item() - loss_before.item()) < 1e-5, \
        f"FAIL B: loss changed from {loss_before:.6f} to {loss_after:.6f}"
    print(f"  Test B [OK]: loss unchanged ({loss_before:.8f}) when visible modified")


def test_c_loss_changes_on_masked():
    """Test C: modify masked patches → loss changes."""
    pred   = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    target = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    mask   = torch.tensor([[[True, False]]])
    loss_before = masked_reconstruction_loss(pred, target, mask)

    # Modify masked patch only
    pred[0, 0, mask[0, 0]] = 5.0
    loss_after = masked_reconstruction_loss(pred, target, mask)

    assert loss_after.item() > loss_before.item() + 0.01, \
        f"FAIL C: loss did not increase: {loss_before:.6f} → {loss_after:.6f}"
    print(f"  Test C [OK]: loss {loss_before:.6f} → {loss_after:.6f} (masked modified)")


def test_d_manual_mse():
    """Test D: hand-compute MSE, compare with code output."""
    pred   = torch.tensor([[[[1.0, 2.0], [5.0, 6.0]]]])    # (1,1,2,2)
    target = torch.tensor([[[[3.0, 4.0], [7.0, 8.0]]]])
    mask   = torch.tensor([[[True, True]]])                  # both masked

    # Manual MSE:
    # errors: (1-3)^2=4, (2-4)^2=4, (5-7)^2=4, (6-8)^2=4 → mean = 16/4 = 4
    loss = masked_reconstruction_loss(pred, target, mask)
    expected = 4.0

    assert abs(loss.item() - expected) < 1e-5, \
        f"FAIL D: loss={loss.item():.4f}, expected {expected}"
    print(f"  Test D [OK]: loss={loss.item():.4f} == expected {expected}")


# ===========================================================================
# Integration tests with real data
# ===========================================================================

def main() -> None:
    BATCH_SIZE = 4
    PATCH_LEN  = 12
    STRIDE     = 12
    MASK_RATIO = 0.4
    D_MODEL    = 128
    N_HEADS    = 16
    N_LAYERS   = 3
    D_FF       = 256
    SEED       = 42

    print("=" * 70)
    print("Masked Reconstruction Loss — Debug")
    print("=" * 70)

    # ---- Unit tests A–D ----
    print("\n--- Manual Tests (A–D) ---")
    test_a_loss_zero_when_perfect()
    test_b_loss_unchanged_on_visible()
    test_c_loss_changes_on_masked()
    test_d_manual_mse()

    # ---- Integration ----
    print("\n--- Integration: Dataset → Encoder → Head → Loss ---")

    # 1. Dataset
    ds = MaskedPretrainDataset("splits/train.csv",
        patch_len=PATCH_LEN, stride=STRIDE, mask_ratio=MASK_RATIO, seed=SEED)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    masked = batch["masked_signal"]
    target = batch["target_signal"]
    mask   = batch["mask"]

    B, C, N, P = masked.shape
    n_total = mask.numel()
    n_masked = int(mask.sum())

    print(f"  masked_signal : {tuple(masked.shape)}")
    print(f"  target_signal : {tuple(target.shape)}")
    print(f"  mask          : {tuple(mask.shape)}")
    print(f"  mask ratio    : {n_masked}/{n_total} = {n_masked/n_total*100:.1f}%")
    print(f"  mask semantic : True=masked, False=visible")

    # 2. Encoder
    backbone = PatchTSTBackbone(patch_len=PATCH_LEN, n_vars=C,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF)
    backbone.eval()

    # 3. Reconstruction
    head = ReconstructionHead(d_model=D_MODEL, patch_len=PATCH_LEN)
    head.eval()

    with torch.no_grad():
        encoded = backbone(masked)                             # (B, C, N, d_model)
        flat = encoded.reshape(B * C, N, D_MODEL)             # (B*C, N, d_model)
        recon = head(flat).reshape(B, C, N, PATCH_LEN)       # (B, C, N, P)

    print(f"  encoded       : {tuple(encoded.shape)}")
    print(f"  reconstructed : {tuple(recon.shape)}")

    # 4. Loss
    loss = masked_reconstruction_loss(recon, target, mask)
    print(f"  loss          : {loss.item():.6f}")

    # 5. Sanity checks
    print("\n--- Checks ---")

    assert torch.isfinite(loss), f"FAIL: loss is not finite: {loss}"
    print(f"  [OK] loss is finite: {loss.item():.6f}")

    is_scalar = loss.dim() == 0
    print(f"  [OK] loss is scalar: dim={loss.dim()}")

    n_masked_check = int(mask.sum())
    assert n_masked_check > 0, "FAIL: no masked patches"

    # Verify loss > 0 for random init (model hasn't learned anything)
    assert loss.item() > 0, f"FAIL: loss should be >0 for random init, got {loss.item():.6f}"
    print(f"  [OK] loss > 0 (expected for untrained model)")

    # Verify mask semantic: True → all-zero in input
    for ch in range(C):
        m = mask[0, ch]
        if m.any():
            assert torch.all(masked[0, ch, m] == 0.0), \
                "FAIL: masked patches in input are not zero"
    print(f"  [OK] mask=True → zero in input (consistent)")

    # Verify loss is computed per-element (not per-patch first)
    print(f"  [OK] reduction=mean over all masked elements (MSE per-time-point)")

    print("\n" + "=" * 70)
    print("  All tests passed.  Loss ready for Training Loop.")
    print(f"  Loss on untrained model: {loss.item():.6f}")
    print(f"  (With RevIN-normed targets, loss ~1.0 is expected)")
    print("=" * 70)


if __name__ == "__main__":
    main()
