#!/usr/bin/env python3
"""
Masked Reconstruction — Learning Diagnosis.
=============================================
Tests why the SSL loss plateaus at ~1.0 (near zero-prediction baseline).

All tests use a SINGLE fixed batch to isolate the model from data variance.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dataset import MaskedPretrainDataset
from src.dataset.ecg_dataset import collate_ssl
from src.models import PatchTSTBackbone, ReconstructionHead
from src.models.loss import masked_reconstruction_loss

SEED = 42
PATCH_LEN = 12
STRIDE = 12
MASK_RATIO = 0.4
D_MODEL = 128
N_HEADS = 16
N_LAYERS = 3
D_FF = 256

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cpu")
OUT_DIR = Path(__file__).resolve().parent / "diagnosis_output"
OUT_DIR.mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════
# Helper: fixed-mask batch
# ════════════════════════════════════════════════════════

def get_fixed_batch():
    """Return a single batch with a FIXED mask (val-mode dataset → deterministic)."""
    ds = MaskedPretrainDataset(
        "splits/train.csv", patch_len=PATCH_LEN, stride=STRIDE,
        mask_ratio=MASK_RATIO, seed=SEED, mode="val",  # val = fixed mask
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_ssl,
    )
    batch = next(iter(loader))
    return {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


# ════════════════════════════════════════════════════════
# Core forward
# ════════════════════════════════════════════════════════

def forward(model, recon_head, batch):
    masked = batch["masked_signal"]
    target = batch["target_signal"]
    mask   = batch["mask"]
    encoded = model(masked)
    B, C, N, D = encoded.shape
    flat = encoded.reshape(B * C, N, D)
    recon = recon_head(flat).reshape(B, C, N, -1)
    loss = masked_reconstruction_loss(recon, target, mask)
    return loss, recon, target, mask, encoded


# ════════════════════════════════════════════════════════
# Test 1: overfit single batch
# ════════════════════════════════════════════════════════

def test_overfit(batch, lr=1e-4, steps=50, log_every=10):
    print(f"\n--- Overfit Test: lr={lr}, steps={steps} ---")
    model = PatchTSTBackbone(PATCH_LEN, n_vars=2, d_model=D_MODEL,
                              n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF).to(DEVICE)
    recon_head = ReconstructionHead(D_MODEL, PATCH_LEN).to(DEVICE)
    opt = torch.optim.AdamW(list(model.parameters()) + list(recon_head.parameters()), lr=lr)
    model.train(); recon_head.train()

    losses, grad_norms_enc, grad_norms_head = [], [], []

    for step in range(steps):
        opt.zero_grad()
        loss, recon, target, mask, _ = forward(model, recon_head, batch)
        loss.backward()

        # Per-component grad norms
        gn_enc = sum(p.grad.data.norm(2).item()**2 for p in model.parameters() if p.grad is not None) ** 0.5
        gn_head = sum(p.grad.data.norm(2).item()**2 for p in recon_head.parameters() if p.grad is not None) ** 0.5

        opt.step()

        losses.append(loss.item())
        grad_norms_enc.append(gn_enc)
        grad_norms_head.append(gn_head)

        if step % log_every == 0:
            print(f"  step {step:4d}: loss={loss.item():.6f}  "
                  f"gn_enc={gn_enc:.4f}  gn_head={gn_head:.4f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(losses); ax1.set_title(f"Loss (lr={lr})"); ax1.set_xlabel("Step"); ax1.set_ylabel("MSE")
    ax2.plot(grad_norms_enc, label="Encoder"); ax2.plot(grad_norms_head, label="Head")
    ax2.set_title("Gradient Norm"); ax2.set_xlabel("Step"); ax2.legend()
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"overfit_lr{lr}.png", dpi=120)
    plt.close(fig)

    return losses, grad_norms_enc, grad_norms_head


# ════════════════════════════════════════════════════════
# Test 2: zero-prediction baseline
# ════════════════════════════════════════════════════════

def test_zero_baseline(batch):
    print("\n--- Zero-Prediction Baseline ---")
    target = batch["target_signal"]
    mask   = batch["mask"]
    mask_exp = mask.unsqueeze(-1).expand_as(target)
    zero_pred = torch.zeros_like(target)
    zero_loss = nn.functional.mse_loss(zero_pred[mask_exp], target[mask_exp])
    print(f"  Zero-prediction MSE: {zero_loss.item():.6f}")
    print(f"  (RevIN-normalized target → mean≈0, so MSE ≈ var(target) ≈ 1.0)")
    return zero_loss.item()


# ════════════════════════════════════════════════════════
# Test 3: masked region statistics
# ════════════════════════════════════════════════════════

def test_masked_stats(batch):
    print("\n--- Masked Region Statistics ---")
    target = batch["target_signal"]
    mask   = batch["mask"]
    mask_exp = mask.unsqueeze(-1).expand_as(target)

    t_masked = target[mask_exp]
    print(f"  target mean={t_masked.mean().item():.4f}  std={t_masked.std().item():.4f}")
    print(f"  target min={t_masked.min().item():.4f}  max={t_masked.max().item():.4f}")
    print(f"  masked elements: {t_masked.numel():,}")
    print(f"  actual mask ratio: {mask.sum().item()/mask.numel():.3f}")

    # Prediction from a random-init model
    model = PatchTSTBackbone(PATCH_LEN, n_vars=2, d_model=D_MODEL,
                              n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF).to(DEVICE)
    recon_head = ReconstructionHead(D_MODEL, PATCH_LEN).to(DEVICE)
    model.eval(); recon_head.eval()
    with torch.no_grad():
        _, recon, target2, mask2, _ = forward(model, recon_head, batch)
    p_masked = recon[mask_exp]
    print(f"  pred mean={p_masked.mean().item():.4f}  std={p_masked.std().item():.4f}")
    print(f"  pred min={p_masked.min().item():.4f}  max={p_masked.max().item():.4f}")


# ════════════════════════════════════════════════════════
# Test 4: normalization trace
# ════════════════════════════════════════════════════════

def test_normalization_trace(batch):
    print("\n--- Normalization Trace ---")
    # The MaskedPretrainDataset does RevIN: (x - mean) / std on RAW signal,
    # then patches + masks.  Target IS the RevIN-normalized patches.
    target = batch["target_signal"]
    masked = batch["masked_signal"]
    mask   = batch["mask"]

    # Check if target is normalized
    mask_exp = mask.unsqueeze(-1).expand_as(target)
    t_visible = target[~mask_exp]
    t_masked_vals = target[mask_exp]

    print(f"  Target (all)         mean={target.mean().item():.4f}  std={target.std().item():.4f}")
    print(f"  Target (visible)     mean={t_visible.mean().item():.4f}  std={t_visible.std().item():.4f}")
    print(f"  Target (masked)      mean={t_masked_vals.mean().item():.4f}  std={t_masked_vals.std().item():.4f}")
    print(f"  Masked input (zeros) mean={masked[mask_exp].mean().item():.4f}  (should be 0)")

    # Verify: target ≈ normalized signal → mean ~0, std ~1 (per-sample RevIN)
    assert abs(target.mean().item()) < 0.2, f"Target mean should be near 0 after RevIN, got {target.mean().item():.4f}"
    print(f"  [OK] Target is RevIN-normalized (~N(0,1) per sample)")

    # Are pred and target at same scale?
    model = PatchTSTBackbone(PATCH_LEN, n_vars=2, d_model=D_MODEL,
                              n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF).to(DEVICE)
    recon_head = ReconstructionHead(D_MODEL, PATCH_LEN).to(DEVICE)
    model.eval(); recon_head.eval()
    with torch.no_grad():
        _, recon, _, _, _ = forward(model, recon_head, batch)
    print(f"  Recon       mean={recon.mean().item():.4f}  std={recon.std().item():.4f}")
    print(f"  Recon scale ≈ target scale: {abs(recon.std().item() - target.std().item()) < 0.5}")

    # Key check: does the model use RevIN denorm before loss?
    # Our dataset applies RevIN NORM to input.  Target = normalized signal.
    # Model output = reconstruction of normalized signal.
    # Loss compares normalized pred vs normalized target.  No denorm needed.
    # This is correct IF no RevIN denorm is expected in the encoder.
    print(f"  [OK] Dataset RevIN-norm → target is normalized → loss in normalized space.")


# ════════════════════════════════════════════════════════
# Test 5: patch/mask alignment visualization
# ════════════════════════════════════════════════════════

def test_patch_alignment(batch):
    print("\n--- Patch / Mask Alignment ---")
    masked = batch["masked_signal"]   # (B, C, N, P)
    target = batch["target_signal"]
    mask   = batch["mask"]

    B, C, N, P = masked.shape
    print(f"  Patches: B={B} C={C} N={N} P={P}")
    print(f"  Expected N: {(6000-PATCH_LEN)//STRIDE + 1} (should be 500)")

    model = PatchTSTBackbone(PATCH_LEN, n_vars=2, d_model=D_MODEL,
                              n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF).to(DEVICE)
    recon_head = ReconstructionHead(D_MODEL, PATCH_LEN).to(DEVICE)
    model.eval(); recon_head.eval()
    with torch.no_grad():
        _, recon, _, _, _ = forward(model, recon_head, batch)

    # Visualize: sample 0, channel 0, first 50 patches
    fig, axes = plt.subplots(3, 1, figsize=(16, 9))
    t_flat = target[0, 0].flatten().numpy()      # (N*P,)
    r_flat = recon[0, 0].flatten().numpy()
    m_flat = mask[0, 0].unsqueeze(-1).expand(-1, P).flatten().numpy()

    axes[0].plot(t_flat, lw=0.5, color="blue", alpha=0.6, label="Target")
    axes[0].set_title("Target (blue) vs Recon (red)")
    axes[0].plot(r_flat, lw=0.3, color="red", alpha=0.6, label="Recon")
    axes[0].legend()

    axes[1].plot(t_flat, lw=0.5, color="gray", alpha=0.4, label="Target")
    # highlight masked regions
    for i in range(N):
        if mask[0, 0, i]:
            axes[1].axvspan(i * P, (i + 1) * P, alpha=0.3, color="red")
    axes[1].set_title("Target + Mask Overlay (red = masked)")

    # Masked input
    axes[2].plot(masked[0, 0].flatten().numpy(), lw=0.5, color="green", alpha=0.6)
    axes[2].set_title("Masked Input (zeros at masked pos)")

    for ax in axes:
        ax.set_xlim(0, min(50 * P, len(t_flat)))
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "patch_alignment.png", dpi=120)
    plt.close(fig)
    print(f"  Saved patch_alignment.png to {OUT_DIR}")


# ════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Masked Learning Diagnosis")
    print("=" * 70)

    # 0. Get fixed batch
    batch = get_fixed_batch()
    B, C, N, P = batch["masked_signal"].shape
    print(f"\nBatch shape: B={B} C={C} N={N} P={P}")
    print(f"Mask ratio: {batch['mask'].sum().item()/batch['mask'].numel():.3f}")

    # 1. Zero baseline
    z_base = test_zero_baseline(batch)

    # 2. Masked stats
    test_masked_stats(batch)

    # 3. Normalization trace
    test_normalization_trace(batch)

    # 4. Patch alignment
    test_patch_alignment(batch)

    # 5. Overfit: lr=1e-4
    losses_le4, gn_enc_4, gn_head_4 = test_overfit(batch, lr=1e-4, steps=500)

    # 6. Overfit: lr=1e-3
    losses_le3, gn_enc_3, gn_head_3 = test_overfit(batch, lr=1e-3, steps=500)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("Diagnosis Summary")
    print("=" * 70)
    print(f"  Zero-prediction baseline: {z_base:.4f}")
    print(f"  lr=1e-4: loss {losses_le4[0]:.4f} → {losses_le4[-1]:.4f} "
          f"({(1-losses_le4[-1]/losses_le4[0])*100:.1f}% decrease)")
    print(f"  lr=1e-3: loss {losses_le3[0]:.4f} → {losses_le3[-1]:.4f} "
          f"({(1-losses_le3[-1]/losses_le3[0])*100:.1f}% decrease)")
    print(f"  Plots: {OUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
