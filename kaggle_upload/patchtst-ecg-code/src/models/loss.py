"""
Masked Reconstruction Loss — Phase 4
======================================
Official reference:
  PatchTST self-supervised pretraining — MSE computed ONLY on masked patches.
  The loss is mean-reduced over all masked time-points (not per-patch first).

Mask semantic (consistent across Dataset / Encoder / Loss):
  mask = True  → patch IS masked (values set to zero in input)
  mask = False → patch is visible (original values preserved)

This matches the official PatchMaskCB where masked indices are set to zero
and the loss is computed exclusively on those positions.
"""

import torch
import torch.nn as nn


def masked_reconstruction_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MSE loss computed only on masked patches.

    Parameters
    ----------
    predicted : (B, C, N, P)
        Reconstructed patches from ReconstructionHead.
    target : (B, C, N, P)
        Original patches (pre-masking) — the ground truth.
    mask : (B, C, N)
        Boolean tensor. True = masked (should contribute to loss).

    Returns
    -------
    loss : scalar Tensor
        Mean MSE over all masked time-points across batch.
    """
    # ---- Validate shapes ----
    B, C, N, P = predicted.shape
    assert target.shape == (B, C, N, P), \
        f"target shape {tuple(target.shape)} != predicted {(B, C, N, P)}"
    assert mask.shape == (B, C, N), \
        f"mask shape {tuple(mask.shape)} != {(B, C, N)}"

    n_masked = mask.sum().item()
    if n_masked == 0:
        raise ValueError(
            "No masked patches in this batch. "
            "Increase mask_ratio or check Dataset masking logic."
        )

    # ---- Expand mask to patch_len dimension ----
    # mask: (B, C, N) → (B, C, N, 1) → expand → (B, C, N, P)
    mask_exp = mask.unsqueeze(-1).expand(-1, -1, -1, P)

    # ---- MSE on masked positions only ----
    # Extract masked elements from pred and target
    pred_masked = predicted[mask_exp]       # (n_masked * P,)
    target_masked = target[mask_exp]        # (n_masked * P,)

    loss = nn.functional.mse_loss(pred_masked, target_masked, reduction="mean")

    return loss


# ===========================================================================
# Manual computation reference for testing
# ===========================================================================
#
#   pred  = [[[1,2],[3,4]]]            # (1,1,2,2)
#   target= [[[1,2],[3,4]]]            # same → loss=0
#   mask  = [[[True, False]]]          # first patch masked
#
#   mask_exp = [[[[T,T],[F,F]]]]       # (1,1,2,2)
#   pred_masked  = [1, 2]              # only first patch
#   target_masked= [1, 2]
#   loss = MSE([1,2], [1,2]) = 0.0
# ===========================================================================
