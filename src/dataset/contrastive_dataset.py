"""Datasets for ECG contrastive self-supervised pretraining."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.dataset.ecg_dataset import BaseECGDataset, REVIN_EPS


def collate_contrastive(batch: list[dict]) -> dict:
    return {
        "window_id": [x["window_id"] for x in batch],
        "record_id": [x["record_id"] for x in batch],
        "view1": torch.stack([x["view1"] for x in batch], dim=0),
        "view2": torch.stack([x["view2"] for x in batch], dim=0),
        "af_label": [x["af_label"] for x in batch],
    }


class ContrastivePretrainDataset(BaseECGDataset):
    """Return two augmented ECG views from the same window.

    Augmentations are intentionally conservative so the AF/Normal rhythm label
    should be preserved. Returned tensors are already RevIN-normalized and
    patched to ``(C, N, patch_len)``.
    """

    def __init__(
        self,
        csv_path: str | Path,
        data_dir: str | Path,
        patch_len: int,
        stride: int,
        revin_norm: bool = True,
        seed: Optional[int] = None,
        mode: str = "train",
        backend: str = "wfdb",
        npy_dir: str | Path | None = None,
        aug_strength: str = "medium",
    ):
        super().__init__(csv_path, data_dir=data_dir, backend=backend, npy_dir=npy_dir)
        self.patch_len = patch_len
        self.stride = stride
        self.revin_norm = revin_norm
        self.base_seed = seed
        self.mode = mode
        self.epoch = 0
        self.aug_strength = aug_strength

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _rng(self, idx: int, view: int) -> np.random.Generator:
        if self.mode == "train":
            seed = None if self.base_seed is None else self.base_seed + self.epoch * 1_000_000 + idx * 2 + view
        else:
            seed = 42 + idx * 2 + view if self.base_seed is None else self.base_seed + idx * 2 + view
        return np.random.default_rng(seed)

    def _params(self) -> dict[str, float]:
        if self.aug_strength == "weak":
            return {"scale": 0.05, "noise": 0.02, "wander": 0.03, "drop": 0.02}
        if self.aug_strength == "strong":
            return {"scale": 0.15, "noise": 0.08, "wander": 0.10, "drop": 0.08}
        return {"scale": 0.10, "noise": 0.05, "wander": 0.06, "drop": 0.05}

    def _augment(self, signal: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        params = self._params()
        out = signal.clone()
        c, l = out.shape

        scale = rng.uniform(1.0 - params["scale"], 1.0 + params["scale"], size=(c, 1))
        out = out * torch.tensor(scale, dtype=out.dtype)

        noise_std = params["noise"] * float(out.std().item() + REVIN_EPS)
        out = out + torch.randn_like(out) * noise_std

        t = torch.linspace(0.0, 1.0, l, dtype=out.dtype)
        for ch in range(c):
            amp = rng.uniform(-params["wander"], params["wander"])
            freq = rng.uniform(0.2, 0.8)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            out[ch] = out[ch] + amp * torch.sin(2.0 * torch.pi * freq * t + phase)

        drop_len = int(l * params["drop"])
        if drop_len > 0:
            start = int(rng.integers(0, max(1, l - drop_len + 1)))
            out[:, start:start + drop_len] = 0.0
        return out

    def _normalize_and_patch(self, signal: torch.Tensor) -> torch.Tensor:
        if self.revin_norm:
            mean = signal.mean(dim=-1, keepdim=True)
            std = signal.std(dim=-1, keepdim=True, unbiased=False)
            signal = (signal - mean) / (std + REVIN_EPS)
        return signal.unfold(dimension=-1, size=self.patch_len, step=self.stride)

    def __getitem__(self, idx: int) -> dict:
        sample = super().__getitem__(idx)
        signal = sample["signal"]
        if self.revin_norm:
            mean = signal.mean(dim=-1, keepdim=True)
            std = signal.std(dim=-1, keepdim=True, unbiased=False)
            signal = (signal - mean) / (std + REVIN_EPS)

        view1 = self._augment(signal, self._rng(idx, 0))
        view2 = self._augment(signal, self._rng(idx, 1))
        return {
            "window_id": sample["window_id"],
            "record_id": sample["record_id"],
            "view1": view1.unfold(dimension=-1, size=self.patch_len, step=self.stride),
            "view2": view2.unfold(dimension=-1, size=self.patch_len, step=self.stride),
            "af_label": sample["af_label"],
        }
