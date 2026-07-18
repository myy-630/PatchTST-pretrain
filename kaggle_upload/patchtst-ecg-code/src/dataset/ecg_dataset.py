"""
ECG Dataset classes for PatchTST Self-Supervised Pretraining.
===============================================================

  BaseECGDataset         — reads 30 s windows from WFDB, returns (C, L)
  MaskedPretrainDataset  — adds PatchTST-style patching + random masking

Masking follows the official PatchTST implementation:
  - Non-overlapping patches (stride = patch_len for SSL)
  - Patch-level masking (entire patch set to zero)
  - Mask ratio defaults to 0.4
  - RevIN applied per-instance before patching

Reference: PatchTST_self_supervised/src/callback/patch_mask.py (PatchMaskCB)
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import wfdb


def collate_ssl(batch: list[dict]) -> dict:
    """Custom collate for MaskedPretrainDataset.

    Stacks tensor fields; keeps scalar/str fields as lists.
    """
    keys = batch[0].keys()
    out = {}
    for k in keys:
        val = [s[k] for s in batch]
        if isinstance(val[0], torch.Tensor):
            out[k] = torch.stack(val, dim=0)
        else:
            out[k] = val   # keep as list (window_id, record_id, …)
    return out

log = logging.getLogger(__name__)

# ===========================================================================
# Configuration defaults — overridable at init time
# ===========================================================================

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "AF"

DEFAULT_PATCH_LEN = 12    # samples per patch
DEFAULT_STRIDE = 12       # stride = patch_len for non-overlap (SSL)
DEFAULT_MASK_RATIO = 0.4  # fraction of patches to mask
DEFAULT_MASK_TYPE = "random"
REVIN_EPS = 1e-5


# ===========================================================================
# BaseECGDataset
# ===========================================================================

class BaseECGDataset(Dataset):
    """Loads 30 s ECG windows from WFDB or .npy files on-the-fly.

    Parameters
    ----------
    csv_path : str or Path
    data_dir : str or Path  — WFDB root (if backend='wfdb')
    backend : 'wfdb' | 'npy_mmap'
    npy_dir : str or Path  — directory with {record_id}.npy files (if backend='npy_mmap')
    """

    def __init__(self, csv_path: str | Path, data_dir: str | Path = DATA_DIR,
                 backend: str = "wfdb", npy_dir: str | Path | None = None):
        self.backend = backend
        self.data_dir = Path(data_dir)
        self.df = pd.read_csv(csv_path, encoding="utf-8-sig",
                              dtype={"record_id": str})
        self._original_cwd = os.getcwd()

        # npy_mmap: cache mmap handles per record
        self._npy_cache: dict[str, np.ndarray] = {}
        self.npy_dir = Path(npy_dir) if npy_dir else None

        log.info("BaseECGDataset: backend=%s  %d windows from %s",
                 backend, len(self.df), csv_path)

    def _get_signal_wfdb(self, record_id: str, ss: int, es: int):
        os.chdir(str(self.data_dir.parent))
        sig, _meta = wfdb.rdsamp(str(self.data_dir / record_id),
                                  sampfrom=ss, sampto=es, channels=[0, 1])
        os.chdir(self._original_cwd)
        return sig

    def _get_signal_npy(self, record_id: str, ss: int, es: int):
        if record_id not in self._npy_cache:
            path = self.npy_dir / f"{record_id}.npy"
            self._npy_cache[record_id] = np.load(str(path), mmap_mode="r")
        seg = self._npy_cache[record_id][:, ss:es]          # (C, window)
        return seg.T.copy()                                   # (window, C)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        rid = row["record_id"]
        ss = int(row["start_sample"])
        es = int(row["end_sample"]) + 1

        if self.backend == "npy_mmap":
            sig = self._get_signal_npy(rid, ss, es)
        else:
            sig = self._get_signal_wfdb(rid, ss, es)

        signal = torch.from_numpy(sig.astype(np.float32)).T    # (C, L)

        return {
            "window_id": row["window_id"],
            "record_id": rid,
            "signal": signal,
            "af_label": row["af_label"],
            "mixed_label": row["mixed_label"],
        }


# ===========================================================================
# MaskedPretrainDataset
# ===========================================================================

class MaskedPretrainDataset(BaseECGDataset):
    """BaseECGDataset + PatchTST masking for self-supervised pretraining.

    Masking strategy
    ----------------
      train mode  — each (epoch, idx) pair produces a different mask.
                    Call ``ds.set_epoch(e)`` at the start of each epoch.
                    Internally: seed = base_seed + epoch * 1_000_000 + idx.
                    If base_seed is None, a system-entropy RNG is used.

      val mode    — each idx ALWAYS produces the same mask.
                    Internally: seed = base_seed + idx.
                    If base_seed is None, falls back to seed = 42 + idx.

    Multi-worker safety: each worker processes different ``idx`` values,
    so seeds are naturally distinct regardless of worker_id.

    Returns (per sample):
        masked_signal  : (C, N, patch_len)  — signal with masked patches zeroed
        target_signal  : (C, N, patch_len)  — original patched signal
        mask           : (C, N)              — True where patch is masked
    """

    def __init__(
        self,
        csv_path: str | Path,
        data_dir: str | Path = DATA_DIR,
        patch_len: int = DEFAULT_PATCH_LEN,
        stride: int = DEFAULT_STRIDE,
        mask_ratio: float = DEFAULT_MASK_RATIO,
        mask_type: str = DEFAULT_MASK_TYPE,
        revin_norm: bool = True,
        seed: Optional[int] = None,
        mode: str = "train",
        backend: str = "wfdb",           # "wfdb" | "npy_mmap"
        npy_dir: str | Path | None = None,
    ):
        super().__init__(csv_path, data_dir, backend=backend, npy_dir=npy_dir)
        self.patch_len = patch_len
        self.stride = stride
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        self.revin_norm = revin_norm
        self.base_seed = seed
        self.mode = mode
        self.epoch = 0

        # Number of patches per channel
        L = 6000  # 30 s × 200 Hz
        self.num_patches = (L - patch_len) // stride + 1

        # Fallback seed for val mode when base_seed is None
        self._val_fallback_seed = 42

        log.info(
            "MaskedPretrainDataset: mode=%s  patch_len=%d  stride=%d  "
            "num_patches=%d  mask_ratio=%.1f  mask_type=%s  revin=%s  base_seed=%s",
            mode, patch_len, stride, self.num_patches, mask_ratio,
            mask_type, revin_norm, seed,
        )

    def set_epoch(self, epoch: int) -> None:
        """Call at the start of each training epoch for dynamic masks."""
        self.epoch = epoch

    def _rng_for_sample(self, idx: int) -> np.random.Generator:
        """Return a per-sample RNG following the train/val seed policy."""
        if self.mode == "train":
            if self.base_seed is not None:
                # seed = base + epoch*1e6 + idx  → reproducible per (epoch, idx)
                s = self.base_seed + self.epoch * 1_000_000 + idx
            else:
                # No seed provided → fully random (system entropy)
                s = None
        else:  # val mode
            if self.base_seed is not None:
                # seed = base + idx  → same mask every time for this idx
                s = self.base_seed + idx
            else:
                # Fallback: deterministic per idx even without explicit seed
                s = self._val_fallback_seed + idx

        return np.random.default_rng(s)

    def _sample_mask(self, n_channels: int, rng: np.random.Generator) -> torch.Tensor:
        """Sample a patch mask according to the configured mask strategy."""
        n_mask = max(1, int(self.num_patches * self.mask_ratio))
        mask = torch.zeros((n_channels, self.num_patches), dtype=torch.bool)
        mask_type = self.mask_type.lower().replace("-", "_")

        if mask_type in {"random", "random_independent", "independent"}:
            for ch in range(n_channels):
                mask_idx = rng.choice(self.num_patches, size=n_mask, replace=False)
                mask[ch, mask_idx] = True
            return mask

        if mask_type in {"random_synchronized", "synchronized", "random_sync", "sync"}:
            mask_idx = rng.choice(self.num_patches, size=n_mask, replace=False)
            mask[:, mask_idx] = True
            return mask

        if mask_type in {"block", "block_mask"}:
            start_max = self.num_patches - n_mask
            start = int(rng.integers(0, start_max + 1)) if start_max > 0 else 0
            mask[:, start:start + n_mask] = True
            return mask

        raise ValueError(f"Unsupported mask_type: {self.mask_type}")

    def __getitem__(self, idx: int) -> dict:
        sample = super().__getitem__(idx)
        signal = sample["signal"]            # (C, L), torch.float32

        # ---- RevIN: per-instance normalization ----
        if self.revin_norm:
            mean = signal.mean(dim=-1, keepdim=True)
            std = signal.std(dim=-1, keepdim=True, unbiased=False)
            signal = (signal - mean) / (std + REVIN_EPS)

        # ---- Patching (non-overlapping, following official PatchTST) ----
        patches = signal.unfold(dimension=-1, size=self.patch_len,
                                step=self.stride)                  # (C, N, P)

        # Per-sample RNG (train: epoch-dependent; val: deterministic)
        rng = self._rng_for_sample(idx)

        # ---- Patch-level masking ----
        mask = self._sample_mask(signal.shape[0], rng)

        # Create masked version
        masked_patches = patches.clone()
        for ch in range(signal.shape[0]):
            masked_patches[ch, mask[ch]] = 0.0

        return {
            "window_id": sample["window_id"],
            "record_id": sample["record_id"],
            "masked_signal": masked_patches,    # (C, N, P)
            "target_signal": patches,           # (C, N, P), original
            "mask": mask,                       # (C, N), bool
            "af_label": sample["af_label"],
        }


# ===========================================================================
# Quick sanity test (run directly)
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== BaseECGDataset test ===")
    ds = BaseECGDataset("splits/train.csv")
    s = ds[0]
    print(f"  window_id: {s['window_id']}")
    print(f"  signal:    {s['signal'].shape}  dtype={s['signal'].dtype}")
    print(f"  af_label:  {s['af_label']}")

    print("\n=== MaskedPretrainDataset test ===")
    mds = MaskedPretrainDataset("splits/train.csv",
                                patch_len=12, stride=12, mask_ratio=0.4, seed=42)
    m = mds[0]
    print(f"  masked_signal: {m['masked_signal'].shape}")
    print(f"  target_signal: {m['target_signal'].shape}")
    print(f"  mask:          {m['mask'].shape}")
    print(f"  masked patches: {m['mask'].sum().item()} / "
          f"{m['mask'].numel()}  ({m['mask'].sum().item()/m['mask'].numel()*100:.1f}%)")
    print(f"  num_patches:    {mds.num_patches}")
