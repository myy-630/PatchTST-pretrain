"""
PatchTST Encoder — Phase 2
============================
Strictly follows the PatchTST official self-supervised backbone.

Official reference:
  PatchTST_self_supervised/src/models/PatchTST_backbone.py
    → TSTEncoder
    → PatchTST_backbone.forward()

Channel-independent design:
  Each ECG lead is treated as an independent univariate series.
  (B, C, N, P) → reshape → (B*C, N, P) → shared Encoder → (B*C, N, d_model)

Adaptations for 2-lead ECG (vs official univariate forecasting):
  - C=2 instead of C=1; channel-independent architecture handles this natively.
  - No structural changes to the Transformer or patching logic.
"""

import torch
import torch.nn as nn


# ===========================================================================
# PatchTST Encoder — core Transformer over patches
# ===========================================================================

class PatchTSTEncoder(nn.Module):
    """Patch-level Transformer Encoder (channel-independent).

    Official analogue: TSTEncoder in PatchTST_backbone.py

    Parameters
    ----------
    patch_len : int
        Number of time steps per patch (P).
    d_model : int
        Embedding dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of Transformer encoder layers.
    d_ff : int
        Feed-forward hidden dimension.
    dropout : float
        Dropout rate.
    max_patches : int
        Maximum number of patches (for positional encoding buffer).
    """

    def __init__(
        self,
        patch_len: int,
        d_model: int = 128,
        n_heads: int = 16,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_patches: int = 1024,
    ):
        super().__init__()

        # ---- Patch Projection: P → d_model ----
        # Official: nn.Linear(patch_len, d_model) applied to each patch
        self.patch_proj = nn.Linear(patch_len, d_model)

        # ---- Learnable Positional Encoding ----
        # Official: nn.Parameter of shape (1, max_patches, d_model)
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_patches, d_model) * 0.02
        )

        # ---- Transformer Encoder ----
        # Official: nn.TransformerEncoder with batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",            # official uses GELU
            batch_first=True,
            norm_first=True,             # Pre-LN (official default)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B_total, N, P)  — patched & masked signal, channel-independent.
                B_total = batch_size × n_channels  (each lead is a "sample").

        Returns:
            (B_total, N, d_model)  — patch-level representations.
        """
        # ---- 1. Patch projection ----
        # (B*C, N, P) → (B*C, N, d_model)
        x = self.patch_proj(x)

        # ---- 2. Dropout (official applies after projection) ----
        x = self.dropout(x)

        # ---- 3. Add positional encoding ----
        # (B*C, N, d_model) + (1, N, d_model) → (B*C, N, d_model)
        x = x + self.pos_embed[:, : x.size(1), :]

        # ---- 4. Transformer Encoder ----
        # (B*C, N, d_model) → (B*C, N, d_model)
        x = self.transformer(x)

        return x


# ===========================================================================
# PatchTST Backbone — channel-independent wrapper
# ===========================================================================

class PatchTSTBackbone(nn.Module):
    """Full PatchTST backbone: channel handling + encoder.

    Official analogue: PatchTST_backbone.forward()

    Accepts pre-patched input from MaskedPretrainDataset:
      (B, C, N, P)  where P = patch_len, N = num_patches.

    Reshapes to (B*C, N, P), passes through shared Encoder,
    then reshapes back to (B, C, N, d_model).
    """

    def __init__(
        self,
        patch_len: int,
        n_vars: int = 2,               # ECG channels
        d_model: int = 128,
        n_heads: int = 16,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_patches: int = 1024,
    ):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model

        self.encoder = PatchTSTEncoder(
            patch_len=patch_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_patches=max_patches,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, N, P)  — patched + masked signal from MaskedPretrainDataset.

        Returns:
            (B, C, N, d_model) — patch-level representations.
        """
        B, C, N, P = x.shape

        # ---- Channel-independent: flatten B and C ----
        # (B, C, N, P) → (B*C, N, P)
        # Each lead is an independent univariate series; Encoder weights are shared.
        x = x.reshape(B * C, N, P)            # (B*C, N, P)

        # ---- Shared Encoder ----
        x = self.encoder(x)                    # (B*C, N, d_model)

        # ---- Reshape back to per-channel ----
        # (B*C, N, d_model) → (B, C, N, d_model)
        x = x.reshape(B, C, N, self.d_model)   # (B, C, N, d_model)

        return x


# ===========================================================================
# Shape reference (ECG config: B=4, C=2, N=500, P=12, d_model=128)
# ===========================================================================
#
#   Dataset output:
#     masked_signal  (B, C, N, P)      e.g. (4, 2, 500, 12)
#
#   Backbone.forward:
#     x              (B, C, N, P)      (4, 2, 500, 12)
#     reshape        (B*C, N, P)       (8, 500, 12)
#
#   Encoder.forward:
#     patch_proj     (B*C, N, P)       (8, 500, 12)
#     →              (B*C, N, d_model)  (8, 500, 128)
#     + pos_embed    (1, N, d_model)   (1, 500, 128)
#     transformer    (B*C, N, d_model)  (8, 500, 128)
#
#   Backbone output:
#     reshape        (B, C, N, d_model) (4, 2, 500, 128)
# ===========================================================================
