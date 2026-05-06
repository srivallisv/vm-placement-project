"""
patchtst.py
PatchTST: Channel-Independent Patch Transformer for time-series forecasting.

Reference: Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting
with Transformers", ICLR 2023.
"""

import math
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Split a 1-D time series into overlapping patches and project to d_model."""

    def __init__(self, seq_len: int, patch_len: int, stride: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = max(1, (seq_len - patch_len) // stride + 1)
        self.projection = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, 1)  — single channel
        Returns:
            (batch, n_patches, d_model)
        """
        x = x.squeeze(-1)  # (batch, seq_len)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # patches: (batch, n_patches, patch_len)
        return self.projection(patches)


class LearnedPositionalEncoding(nn.Module):
    """Learnable positional encoding."""

    def __init__(self, n_patches: int, d_model: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embed[:, : x.size(1), :]


class TransformerEncoderBlock(nn.Module):
    """Standard transformer encoder layer."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class PatchTSTBackbone(nn.Module):
    """
    Backbone for a single channel: patch → embed → transformer → flatten head.
    """

    def __init__(
        self,
        seq_len: int = 48,
        pred_len: int = 6,
        patch_len: int = 8,
        stride: int = 4,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(seq_len, patch_len, stride, d_model)
        n_patches = self.patch_embed.n_patches
        self.pos_enc = LearnedPositionalEncoding(n_patches, d_model)
        self.encoder = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(e_layers)
        ])
        self.flatten_head = nn.Linear(n_patches * d_model, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, 1) — single channel
        Returns:
            (batch, pred_len) — predictions for this channel
        """
        x = self.patch_embed(x)
        x = self.pos_enc(x)
        x = self.dropout(x)
        for layer in self.encoder:
            x = layer(x)
        # Flatten all patches
        x = x.flatten(start_dim=1)  # (batch, n_patches * d_model)
        return self.flatten_head(x)


class PatchTSTModel(nn.Module):
    """
    Channel-Independent PatchTST.

    Each feature channel (CPU, Mem, Storage) is processed independently
    through a shared transformer backbone, then stacked.

    Input:  (batch, seq_len, n_features)
    Output: (batch, pred_len, n_features)
    """

    def __init__(
        self,
        n_features: int = 3,
        seq_len: int = 48,
        pred_len: int = 6,
        patch_len: int = 8,
        stride: int = 4,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_features = n_features

        # Shared backbone across all channels
        self.backbone = PatchTSTBackbone(
            seq_len=seq_len,
            pred_len=pred_len,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            (batch, pred_len, n_features)
        """
        outputs = []
        for i in range(self.n_features):
            xi = x[:, :, i : i + 1]         # (batch, seq_len, 1)
            out_i = self.backbone(xi)        # (batch, pred_len)
            outputs.append(out_i)
        return torch.stack(outputs, dim=-1)  # (batch, pred_len, n_features)
