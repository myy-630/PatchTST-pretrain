#!/usr/bin/env python3
"""
Test script: MaskedPretrainDataset randomness guarantees.
============================================================
Verifies:
  T1 — train: same idx, same epoch → same mask (within-epoch consistency)
  T2 — train: same idx, different epoch → different mask
  T3 — val:   same idx, always same mask
  T4 — reproducibility: two runs with same seed → identical masks
  T5 — mask ratio stays close to configured value
  T6 — multi-worker isolation (different idx → different masks)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from src.dataset import MaskedPretrainDataset

SEED = 42
PATCH_LEN = 12
STRIDE = 12
MASK_RATIO = 0.4


def test_t1_train_same_epoch_same_mask():
    """T1: train mode — same idx in same epoch → same mask (deterministic)."""
    print("T1 — train: same (epoch, idx) → same mask …", end=" ")
    ds = MaskedPretrainDataset("splits/train.csv", seed=SEED, mode="train")
    ds.set_epoch(0)

    m1 = ds[0]["mask"]
    m2 = ds[0]["mask"]          # same idx, same epoch
    assert torch.equal(m1, m2), "FAIL"
    print("OK  (epoch=0, idx=0, masks equal)")


def test_t2_train_different_epoch_different_mask():
    """T2: train mode — same idx, different epoch → different mask."""
    print("T2 — train: same idx, different epoch → different mask …", end=" ")
    ds = MaskedPretrainDataset("splits/train.csv", seed=SEED, mode="train")

    ds.set_epoch(0)
    m_epoch0 = ds[0]["mask"]

    ds.set_epoch(1)
    m_epoch1 = ds[0]["mask"]

    assert not torch.equal(m_epoch0, m_epoch1), "FAIL"
    print("OK  (epoch 0 vs epoch 1: masks differ)")


def test_t3_val_deterministic():
    """T3: val mode — same idx always gives same mask."""
    print("T3 — val: same idx always same mask …", end=" ")

    ds = MaskedPretrainDataset("splits/val.csv", seed=SEED, mode="val")
    m1 = ds[0]["mask"]
    m2 = ds[0]["mask"]
    m3 = ds[0]["mask"]
    assert torch.equal(m1, m2), "FAIL: m1 != m2"
    assert torch.equal(m1, m3), "FAIL: m1 != m3"

    # Even after crossing epochs (set_epoch should NOT affect val)
    ds.set_epoch(99)
    m4 = ds[0]["mask"]
    assert torch.equal(m1, m4), "FAIL: val mask changed after set_epoch"

    print("OK  (4 reads, all identical; set_epoch ignored)")


def test_t4_reproducibility():
    """T4: two datasets with same seed → same masks for same idx."""
    print("T4 — reproducibility: same seed → same masks …", end=" ")

    ds1 = MaskedPretrainDataset("splits/train.csv", seed=123, mode="train")
    ds1.set_epoch(0)

    ds2 = MaskedPretrainDataset("splits/train.csv", seed=123, mode="train")
    ds2.set_epoch(0)

    for idx in [0, 5, 10, 50, 100]:
        m1 = ds1[idx]["mask"]
        m2 = ds2[idx]["mask"]
        assert torch.equal(m1, m2), f"FAIL at idx={idx}"

    print("OK  (5 indices, all match across independent dataset instances)")


def test_t5_mask_ratio():
    """T5: mask ratio stays close to configured value over many samples."""
    print("T5 — mask ratio ≈ configured value …", end=" ")

    ds = MaskedPretrainDataset("splits/train.csv", seed=SEED, mode="train",
                               mask_ratio=MASK_RATIO)
    ds.set_epoch(0)

    n_total = 0
    n_masked = 0
    for idx in range(200):
        mask = ds[idx]["mask"]
        n_total += mask.numel()
        n_masked += int(mask.sum())

    actual_ratio = n_masked / n_total
    # Allow ±2% tolerance
    assert abs(actual_ratio - MASK_RATIO) < 0.02, \
        f"FAIL: actual ratio={actual_ratio:.4f}, expected {MASK_RATIO}"
    print(f"OK  (actual={actual_ratio:.3f}, expected={MASK_RATIO})")


def test_t6_different_idx_different_mask():
    """T6: different indices → different masks (worker isolation)."""
    print("T6 — different idx → different mask …", end=" ")

    ds = MaskedPretrainDataset("splits/train.csv", seed=SEED, mode="train")
    ds.set_epoch(0)

    m0 = ds[0]["mask"]
    m1 = ds[1]["mask"]
    m10 = ds[10]["mask"]

    # Flatten to compare
    assert not torch.equal(m0, m1), "FAIL: idx 0 == idx 1"
    assert not torch.equal(m0, m10), "FAIL: idx 0 == idx 10"
    print("OK  (idx 0 != 1 != 10)")


def test_t7_train_different_epochs_different_but_valid():
    """T7: sanity — masks across epochs all have correct shape and ratio."""
    print("T7 — train masks valid across 5 epochs …", end=" ")

    ds = MaskedPretrainDataset("splits/train.csv", seed=SEED, mode="train",
                               mask_ratio=MASK_RATIO)

    masks_by_epoch = []
    for epoch in range(5):
        ds.set_epoch(epoch)
        m = ds[0]["mask"]
        assert m.shape == (2, 500), f"FAIL shape at epoch {epoch}"
        ratio = m.sum().item() / m.numel()
        assert abs(ratio - MASK_RATIO) < 0.02, \
            f"FAIL ratio={ratio:.4f} at epoch {epoch}"
        masks_by_epoch.append(m)

    # All 5 epochs should produce different masks
    for i in range(5):
        for j in range(i + 1, 5):
            assert not torch.equal(masks_by_epoch[i], masks_by_epoch[j]), \
                f"FAIL: epoch {i} == epoch {j}"

    print(f"OK  (5 epochs, all valid shapes, all different)")


# ===========================================================================
def main() -> None:
    print("=" * 60)
    print("Mask Randomness Tests")
    print(f"seed={SEED}  mode=train/val  patch_len={PATCH_LEN}  "
          f"mask_ratio={MASK_RATIO}")
    print("=" * 60)

    test_t1_train_same_epoch_same_mask()
    test_t2_train_different_epoch_different_mask()
    test_t3_val_deterministic()
    test_t4_reproducibility()
    test_t5_mask_ratio()
    test_t6_different_idx_different_mask()
    test_t7_train_different_epochs_different_but_valid()

    print("\n" + "=" * 60)
    print("  All 7 tests passed. Dataset ready for Pilot.")
    print("=" * 60)


if __name__ == "__main__":
    main()
