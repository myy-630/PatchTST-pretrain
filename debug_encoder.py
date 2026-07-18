#!/usr/bin/env python3
"""
Debug script for PatchTST Encoder (Phase 2).
=============================================
Loads a batch from MaskedPretrainDataset, passes through PatchTSTBackbone,
prints every tensor shape, and runs sanity checks.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch.utils.data import DataLoader

from src.dataset import MaskedPretrainDataset
from src.models import PatchTSTBackbone

# ===========================================================================
# Config — all tunable from here
# ===========================================================================

BATCH_SIZE = 4
SEQ_LEN = 6000
N_VARS = 2
PATCH_LEN = 12
STRIDE = 12
NUM_PATCHES = (SEQ_LEN - PATCH_LEN) // STRIDE + 1          # 500
MASK_RATIO = 0.4

D_MODEL = 128
N_HEADS = 16
N_LAYERS = 3
D_FF = 256
DROPOUT = 0.1

SEED = 42

# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print("=" * 70)
    print("PatchTST Encoder — Debug")
    print(f"Batch={BATCH_SIZE}  C={N_VARS}  N={NUM_PATCHES}  "
          f"P={PATCH_LEN}  d_model={D_MODEL}")
    print("=" * 70)

    # ---- 1. Dataset & DataLoader ----
    print("\n[1] Loading MaskedPretrainDataset + DataLoader …")
    ds = MaskedPretrainDataset(
        "splits/train.csv",
        patch_len=PATCH_LEN,
        stride=STRIDE,
        mask_ratio=MASK_RATIO,
        seed=SEED,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0)

    batch = next(iter(loader))
    masked = batch["masked_signal"]         # (B, C, N, P)

    print(f"    masked_signal  shape: {masked.shape}")
    print(f"                    dtype: {masked.dtype}")
    print(f"                    min: {masked.min().item():.4f}  "
          f"max: {masked.max().item():.4f}")

    # ---- 2. Build model ----
    print("\n[2] Building PatchTSTBackbone …")
    model = PatchTSTBackbone(
        patch_len=PATCH_LEN,
        n_vars=N_VARS,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Total parameters: {total_params:,}")
    print(f"    Trainable:        {trainable:,}")

    # ---- 3. Forward pass ----
    print("\n[3] Forward pass …")
    model.eval()
    with torch.no_grad():
        output = model(masked)

    print(f"    Input  shape:  {masked.shape}     ← (B, C, N, P)")
    print(f"    Output shape:  {output.shape}     ← (B, C, N, d_model)")

    # ---- 4. Shape checks ----
    print("\n[4] Shape assertions …")
    B, C, N, P = masked.shape
    B_out, C_out, N_out, D_out = output.shape

    assert B_out == B, f"Batch mismatch: {B_out} != {B}"
    assert C_out == N_VARS, f"Channel mismatch: {C_out} != {N_VARS}"
    assert N_out == NUM_PATCHES, f"Patch count mismatch: {N_out} != {NUM_PATCHES}"
    assert D_out == D_MODEL, f"d_model mismatch: {D_out} != {D_MODEL}"
    print(f"    [OK] B={B_out}  C={C_out}  N={N_out}  d_model={D_out}")

    # ---- 5. Value checks ----
    print("\n[5] NaN / Inf check …")
    has_nan = torch.isnan(output).any().item()
    has_inf = torch.isinf(output).any().item()
    assert not has_nan, "FAIL: Output contains NaN!"
    assert not has_inf, "FAIL: Output contains Inf!"
    print(f"    [OK] No NaN: {not has_nan}, No Inf: {not has_inf}")
    print(f"    Output stats — "
          f"min={output.min().item():.4f}  "
          f"max={output.max().item():.4f}  "
          f"mean={output.mean().item():.4f}  "
          f"std={output.std().item():.4f}")

    # ---- 6. Channel independence check ----
    print("\n[6] Channel independence check …")
    # Channel 0 and channel 1 of the same sample should have
    # different representations (they are different leads).
    # But they go through the same encoder with shared weights — verify.
    ch0 = output[0, 0, :, :]   # sample 0, lead 0
    ch1 = output[0, 1, :, :]   # sample 0, lead 1
    assert not torch.allclose(ch0, ch1, atol=1e-6), \
        "FAIL: ch0 == ch1 — channels collapsed!"
    print(f"    [OK] ch0 != ch1 — leads are independently processed")
    print(f"    ch0 mean={ch0.mean().item():.4f}  "
          f"ch1 mean={ch1.mean().item():.4f}")

    # ---- 7. Batch independence check ----
    print("\n[7] Batch independence check …")
    # Sample 0 vs sample 1, same lead — should differ
    if B > 1:
        s0 = output[0, 0, :, :]
        s1 = output[1, 0, :, :]
        assert not torch.allclose(s0, s1, atol=1e-6), \
            "FAIL: sample 0 == sample 1 — samples collapsed!"
        print(f"    [OK] Samples are independent")

    # ---- 8. Per-layer shape trace ----
    print("\n[8] Per-layer shape trace (single sample):")
    x_single = masked[0:1]                            # (1, 2, 500, 12)
    B1, C1, N1, P1 = x_single.shape
    x_flat = x_single.reshape(B1 * C1, N1, P1)         # (2, 500, 12)
    print(f"    masked_signal               {tuple(x_single.shape)}")
    print(f"    channel-flatten             {tuple(x_flat.shape)}")

    # Manual step-through
    proj = model.encoder.patch_proj(x_flat)            # (2, 500, 128)
    print(f"    patch_proj (Linear P→d)      {tuple(proj.shape)}")

    dropped = model.encoder.dropout(proj)
    pe = dropped + model.encoder.pos_embed[:, :N1, :]  # (2, 500, 128)
    print(f"    + pos_embed                  {tuple(pe.shape)}")

    tf_out = model.encoder.transformer(pe)              # (2, 500, 128)
    print(f"    transformer output           {tuple(tf_out.shape)}")

    final = tf_out.reshape(B1, C1, N1, D_MODEL)         # (1, 2, 500, 128)
    print(f"    reshape → (B,C,N,d_model)    {tuple(final.shape)}")

    # ---- 9. Summary ----
    print("\n" + "=" * 70)
    print("  All checks passed.")
    print(f"  B={BATCH_SIZE}  C={N_VARS}  N={NUM_PATCHES}  "
          f"P={PATCH_LEN}  d_model={D_MODEL}")
    print(f"  Input:  {tuple(masked.shape)}")
    print(f"  Output: {tuple(output.shape)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
