"""
PatchTST Reconstruction Head — Phase 3
========================================
Official reference:
  PatchTST_self_supervised/src/models/PatchTST_backbone.py → PretrainHead

Structure:
  A single Linear layer: d_model → patch_len.
  Applied independently to each patch position.
  Channel-independent: same weights across all leads.

For ECG (C=2):
  Encoder output  (B, C, N, d_model)
  → reshape      (B*C, N, d_model)
  → Linear       (B*C, N, patch_len)
  → reshape      (B, C, N, patch_len)

That's it — no activation, no LayerNorm, no extra layers.
This matches the official PretrainHead exactly.
"""

import torch
import torch.nn as nn


class ReconstructionHead(nn.Module):
    """Patch-level reconstruction: d_model → patch_len per patch.

    Official analogue: PretrainHead(nn.Module) — a single nn.Linear.

    Parameters
    ----------
    d_model : int
        Encoder embedding dimension.
    patch_len : int
        Number of time steps to reconstruct per patch.
    """

    def __init__(self, d_model: int, patch_len: int):
        super().__init__()
        # Official: nn.Linear(d_model, patch_len), no bias for simplicity
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B_total, N, d_model) — Encoder output, channel-flattened.
                B_total = batch_size × n_channels.

        Returns:
            (B_total, N, patch_len) — reconstructed patches.
        """
        return self.linear(x)          # (B*, N, P)


# ===========================================================================
# Shape reference (ECG: B=4, C=2, N=500, P=12, d_model=128)
# ===========================================================================
#
#   Encoder output       (B, C, N, d_model)   (4, 2, 500, 128)
#   channel-flatten      (B*C, N, d_model)    (8, 500, 128)
#   ReconstructionHead   (B*C, N, patch_len)  (8, 500, 12)
#   reshape back         (B, C, N, patch_len) (4, 2, 500, 12)
#
#   target_signal        (B, C, N, patch_len) (4, 2, 500, 12)
#   mask                 (B, C, N)            (4, 2, 500)
# ===========================================================================
