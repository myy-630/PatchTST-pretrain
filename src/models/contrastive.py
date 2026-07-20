"""Projection head and NT-Xent loss for contrastive ECG pretraining."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Small MLP projection head used only during contrastive pretraining."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def mean_pool_patchtst(encoded: torch.Tensor) -> torch.Tensor:
    """Pool PatchTST output from (B, C, N, D) to (B, C*D)."""
    pooled = encoded.mean(dim=2)
    return pooled.flatten(start_dim=1)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """SimCLR/NT-Xent loss for paired batches.

    ``z1[i]`` and ``z2[i]`` are positives. All other examples in the 2B batch are
    negatives.
    """
    if z1.shape != z2.shape:
        raise ValueError(f"z1 shape {tuple(z1.shape)} != z2 shape {tuple(z2.shape)}")
    batch_size = z1.size(0)
    if batch_size < 2:
        raise ValueError("NT-Xent requires batch_size >= 2")

    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)
    logits = z @ z.T / temperature
    logits.fill_diagonal_(-float("inf"))

    labels = torch.arange(2 * batch_size, device=z.device)
    labels = (labels + batch_size) % (2 * batch_size)
    return F.cross_entropy(logits, labels)
