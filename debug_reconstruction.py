#!/usr/bin/env python3
"""
Debug script for Reconstruction Head (Phase 3).
=================================================
Dataset → Encoder → ReconstructionHead → reconstructed_patches.
Checks shapes, NaN/Inf, mask alignment, and channel independence.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch.utils.data import DataLoader

from src.dataset import MaskedPretrainDataset
from src.models import PatchTSTBackbone, ReconstructionHead

# Config
BATCH_SIZE = 4
PATCH_LEN = 12
STRIDE = 12
MASK_RATIO = 0.4
D_MODEL = 128
N_HEADS = 16
N_LAYERS = 3
D_FF = 256
SEED = 42

NUM_PATCHES = (6000 - PATCH_LEN) // STRIDE + 1   # 500

# ===========================================================================
def main() -> None:
    print("=" * 70)
    print("Reconstruction Head — Debug")
    print(f"patch_len={PATCH_LEN}  stride={STRIDE}  N={NUM_PATCHES}  d_model={D_MODEL}")
    print("=" * 70)

    # ---- 1. Dataset ----
    print("\n[1] Dataset → batch …")
    ds = MaskedPretrainDataset("splits/train.csv",
        patch_len=PATCH_LEN, stride=STRIDE,
        mask_ratio=MASK_RATIO, seed=SEED)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    masked = batch["masked_signal"]       # (B, C, N, P)
    target = batch["target_signal"]       # (B, C, N, P)
    mask   = batch["mask"]                # (B, C, N)

    print(f"    masked_signal : {tuple(masked.shape)}")
    print(f"    target_signal : {tuple(target.shape)}")
    print(f"    mask          : {tuple(mask.shape)}  ({mask.sum()}/{mask.numel()} masked)")

    # ---- 2. Encoder ----
    print("\n[2] Encoder forward …")
    backbone = PatchTSTBackbone(patch_len=PATCH_LEN, n_vars=2,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF)
    backbone.eval()
    with torch.no_grad():
        encoded = backbone(masked)           # (B, C, N, d_model)
    print(f"    encoded       : {tuple(encoded.shape)}")

    # ---- 3. Reconstruction Head ----
    print("\n[3] Reconstruction Head forward …")
    head = ReconstructionHead(d_model=D_MODEL, patch_len=PATCH_LEN)
    head.eval()

    B, C, N, D = encoded.shape
    flat = encoded.reshape(B * C, N, D)      # (B*C, N, d_model)
    print(f"    head input    : {tuple(flat.shape)}  (B*C, N, d_model)")

    with torch.no_grad():
        recon_flat = head(flat)              # (B*C, N, P)
    print(f"    head output   : {tuple(recon_flat.shape)}  (B*C, N, patch_len)")

    recon = recon_flat.reshape(B, C, N, PATCH_LEN)   # (B, C, N, P)
    print(f"    reconstructed : {tuple(recon.shape)}  (B, C, N, P)")

    # ---- 4. Dimension assertions ----
    print("\n[4] Dimension checks …")
    assert recon.shape == target.shape, \
        f"Shape mismatch: recon {tuple(recon.shape)} != target {tuple(target.shape)}"
    assert mask.shape == (B, C, N), \
        f"Mask shape mismatch: {tuple(mask.shape)} != {(B, C, N)}"
    print(f"    [OK] recon == target shape: {tuple(recon.shape)}")
    print(f"    [OK] mask shape: {tuple(mask.shape)}")
    print(f"    [OK] mask can index recon: mask[:,:,0].shape = {mask[:,:,0].shape}")

    # ---- 5. NaN / Inf ----
    print("\n[5] NaN / Inf check …")
    for name, t in [("recon", recon), ("encoded", encoded)]:
        assert not torch.isnan(t).any(), f"FAIL: {name} has NaN"
        assert not torch.isinf(t).any(), f"FAIL: {name} has Inf"
    print(f"    [OK] No NaN or Inf in encoded or recon")

    # ---- 6. Visible patches: recon should differ from target (not just copy) ----
    print("\n[6] Reconstruction quality snapshot …")
    for ch in range(2):
        visible = ~mask[0, ch]                                # patches NOT masked
        masked_pos = mask[0, ch]
        if visible.any() and masked_pos.any():
            # Masked patches: model must predict them
            mse_masked = torch.nn.functional.mse_loss(
                recon[0, ch, masked_pos], target[0, ch, masked_pos])
            # Visible patches: since they passed through the encoder,
            # they should be close to target (encoder can copy them through)
            mse_visible = torch.nn.functional.mse_loss(
                recon[0, ch, visible], target[0, ch, visible])
            print(f"    ch{ch}: MSE masked={mse_masked.item():.4f}  "
                  f"MSE visible={mse_visible.item():.4f}  "
                  f"ratio={mse_masked.item()/mse_visible.item():.1f}x")

    # ---- 7. Channel independence ----
    print("\n[7] Channel independence check …")
    ch0 = recon[0, 0]     # (N, P) Lead I
    ch1 = recon[0, 1]     # (N, P) Lead II
    assert not torch.allclose(ch0, ch1, atol=1e-4), \
        "FAIL: ch0 == ch1 — channels collapsed in reconstruction!"
    print(f"    [OK] Lead I != Lead II after reconstruction")

    # ---- 8. Batch order ----
    print("\n[8] Batch order check …")
    if B > 1:
        r0 = recon[0].flatten()
        r1 = recon[1].flatten()
        assert not torch.allclose(r0, r1, atol=1e-4), \
            "FAIL: batch samples collapsed!"
        print(f"    [OK] Sample 0 != Sample 1")

    # ---- 9. Raw value check ----
    print("\n[9] Output value ranges …")
    print(f"    recon:  min={recon.min().item():.4f}  "
          f"max={recon.max().item():.4f}  "
          f"mean={recon.mean().item():.4f}  "
          f"std={recon.std().item():.4f}")
    print(f"    target: min={target.min().item():.4f}  "
          f"max={target.max().item():.4f}  "
          f"mean={target.mean().item():.4f}  "
          f"std={target.std().item():.4f}")

    # ---- 10. Masked positions are truly non-zero in target but zero in input ----
    print("\n[10] Mask verification …")
    for ch in range(2):
        n_mask = int(mask[0, ch].sum())
        # target has real values at masked positions
        t_masked = target[0, ch, mask[0, ch]]
        # masked_signal has zeros at masked positions
        m_masked = masked[0, ch, mask[0, ch]]
        assert not torch.allclose(t_masked, torch.zeros_like(t_masked), atol=1e-6), \
            f"FAIL: target masked positions are zero (no signal to reconstruct)"
        assert torch.all(m_masked == 0.0), \
            f"FAIL: masked_signal positions are not zero"
        print(f"    ch{ch}: {n_mask} masked patches — target!=0 [OK]  input=0 [OK]")

    # Summary
    print("\n" + "=" * 70)
    print("  All 10 checks passed.")
    print(f"  Input:  {tuple(masked.shape)}  →  Encoder  →  Recon  →  {tuple(recon.shape)}")
    print(f"  Ready for masked MSE loss in Phase 4.")
    print("=" * 70)


if __name__ == "__main__":
    main()
