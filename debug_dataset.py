#!/usr/bin/env python3
"""
Debug script for BaseECGDataset + MaskedPretrainDataset.
==========================================================
Randomly samples windows, prints shapes, and visualises:
  - Original ECG (Lead I + Lead II)
  - Masked signal (reconstructed from masked patches)
  - Mask overlay
"""

import sys
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

from src.dataset import BaseECGDataset, MaskedPretrainDataset

# ===========================================================================
# Configuration
# ===========================================================================

N_SAMPLES = 5
SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parent / "debug_output"
PATCH_LEN = 12
STRIDE = 12
MASK_RATIO = 0.4

# ===========================================================================
# Helpers
# ===========================================================================

def plot_one(
    sample: dict,
    patch_len: int,
    idx: int,
    out_dir: Path,
) -> None:
    """Plot original + masked signal for one sample, both leads."""

    masked_patches = sample["masked_signal"]   # (C, N, P)
    target_patches = sample["target_signal"]   # (C, N, P)
    mask = sample["mask"]                      # (C, N)

    C, N, P = masked_patches.shape
    L = N * P  # total length (should be 6000 when stride=patch_len)

    # Flatten patches back to timeseries: (C, N, P) → (C, L)
    masked_flat = masked_patches.reshape(C, L)   # (C, L)
    target_flat = target_patches.reshape(C, L)   # (C, L)

    # Mask overlay as timeseries: 1.0 where masked
    mask_ts = mask.float().unsqueeze(-1).expand(-1, -1, P).reshape(C, L)

    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)

    leads = ["Lead I", "Lead II"]
    colors_target = ["#1f77b4", "#d62728"]
    colors_masked = ["#aec7e8", "#ff9896"]

    t = np.arange(L) / 200.0    # seconds

    for ch in range(min(C, 2)):
        ax = axes[ch]

        # Original (faint)
        ax.plot(t, target_flat[ch].numpy(), color=colors_target[ch],
                lw=0.5, alpha=0.5, label="Original")

        # Masked overlay (bright where unmasked)
        unmasked = target_flat[ch].clone()
        unmasked[mask_ts[ch] > 0.5] = np.nan
        ax.plot(t, unmasked.numpy(), color=colors_target[ch],
                lw=0.8, label="Visible (unmasked)")

        # Mask highlight
        for n in range(N):
            if mask[ch, n]:
                start = n * P / 200.0
                end = (n + 1) * P / 200.0
                ax.axvspan(start, end, alpha=0.25, color="red", linewidth=0)

        ax.set_ylabel(f"{leads[ch]}  (norm)", fontsize=10)
        ax.grid(True, alpha=0.3, lw=0.5)
        ax.set_xlim(0, t[-1])

    axes[0].legend(loc="upper right", fontsize=8,
                   handles=[
                       mpatches.Patch(color=colors_target[0], alpha=0.5,
                                      label="Original"),
                       mpatches.Patch(color=colors_target[0], alpha=1.0,
                                      label="Visible"),
                       mpatches.Patch(color="red", alpha=0.25,
                                      label="Masked"),
                   ])
    axes[1].set_xlabel("Time (seconds)", fontsize=10)

    n_masked = int(mask.sum().item())
    fig.suptitle(
        f"{sample['window_id']}  |  record={sample['record_id']}  "
        f"|  {n_masked}/{C*N} patches masked ({n_masked/(C*N)*100:.0f}%)  "
        f"|  af_label={sample['af_label']}",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fname = f"sample_{idx:02d}_{sample['window_id']}.png"
    fig.savefig(out_dir / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("=" * 60)
    print("Debug: BaseECGDataset + MaskedPretrainDataset")
    print(f"patch_len={PATCH_LEN}  stride={STRIDE}  mask_ratio={MASK_RATIO}")
    print("=" * 60)

    # ---- Base Dataset ----
    print("\n--- BaseECGDataset ---")
    base = BaseECGDataset("splits/train.csv")
    s0 = base[0]
    print(f"  signal shape: {s0['signal'].shape}  (expected: (2, 6000))")
    print(f"  signal dtype: {s0['signal'].dtype}")
    print(f"  signal min/max: {s0['signal'].min():.3f} / {s0['signal'].max():.3f}")
    print(f"  window_id: {s0['window_id']}")
    print(f"  af_label:  {s0['af_label']}")

    # ---- Masked Dataset ----
    print(f"\n--- MaskedPretrainDataset ---")
    mds = MaskedPretrainDataset(
        "splits/train.csv",
        patch_len=PATCH_LEN, stride=STRIDE,
        mask_ratio=MASK_RATIO, seed=SEED,
    )
    print(f"  num_patches: {mds.num_patches}")

    # Check a few samples
    indices = rng.choice(len(mds), size=N_SAMPLES, replace=False)
    print(f"\n--- Sampling {N_SAMPLES} random windows ---")
    for i, idx in enumerate(indices):
        s = mds[int(idx)]
        C, N, P = s["masked_signal"].shape
        n_mask = int(s["mask"].sum())
        print(f"\n  [{i+1}] {s['window_id']}  record={s['record_id']}  "
              f"label={s['af_label']}")
        print(f"      masked: {s['masked_signal'].shape}  "
              f"dtype={s['masked_signal'].dtype}")
        print(f"      target: {s['target_signal'].shape}")
        print(f"      mask:   {s['mask'].shape}  dtype={s['mask'].dtype}")
        print(f"      masked: {n_mask}/{C*N} patches ({n_mask/(C*N)*100:.0f}%)")
        print(f"      masked_patches min/max: "
              f"{s['masked_signal'].min():.4f} / {s['masked_signal'].max():.4f}")
        print(f"      target_patches min/max: "
              f"{s['target_signal'].min():.4f} / {s['target_signal'].max():.4f}")

        plot_one(s, PATCH_LEN, i + 1, OUTPUT_DIR)

    # ---- Consistency checks ----
    print(f"\n--- Consistency Checks ---")
    all_ok = True
    for i in range(min(100, len(mds))):
        s = mds[i]
        # Check that unmasked patches == target patches
        mask = s["mask"]                                     # (C, N)
        m_patches = s["masked_signal"]                       # (C, N, P)
        t_patches = s["target_signal"]                       # (C, N, P)
        for ch in range(mask.shape[0]):
            visible = ~mask[ch]
            if visible.any():
                if not torch.allclose(m_patches[ch, visible],
                                      t_patches[ch, visible]):
                    print(f"  FAIL at {s['window_id']} ch{ch}: "
                          "visible patch mismatch!")
                    all_ok = False
            masked = mask[ch]
            if masked.any():
                if not torch.all(m_patches[ch, masked] == 0.0):
                    print(f"  FAIL at {s['window_id']} ch{ch}: "
                          "masked patch not zero!")
                    all_ok = False
    if all_ok:
        print("  [OK] All checks passed — masked=zero, visible=original")

    print(f"\n  Plots: {OUTPUT_DIR}/")
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
