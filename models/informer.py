"""
informer.py
Informer model with ProbSparse self-attention for efficient long-sequence
time-series forecasting.

Reference: Zhou et al., "Informer: Beyond Efficient Transformer for Long
Sequence Time-Series Forecasting", AAAI 2021.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


# ---------------------------------------------------------------------------
# ProbSparse self-attention
# ---------------------------------------------------------------------------

class ProbSparseAttention(nn.Module):
    """
    ProbSparse self-attention: selects top-u dominant queries instead of
    computing full O(L²) attention, achieving O(L log L) complexity.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 factor: int = 5):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.factor = factor

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        """Compute ProbSparse top-u query selection."""
        B, H, L_Q, D = Q.shape
        _, _, L_K, _ = K.shape

        # Randomly sample keys for sparsity measure
        K_sample_idx = torch.randint(0, L_K, (sample_k,), device=K.device)
        K_sample = K[:, :, K_sample_idx, :]  # (B, H, sample_k, D)

        # Compute query sparsity measurement: M(qi, K)
        Q_K_sample = torch.matmul(
            Q, K_sample.transpose(-2, -1)
        ) / math.sqrt(D)  # (B, H, L_Q, sample_k)

        M = Q_K_sample.max(dim=-1).values - Q_K_sample.mean(dim=-1)  # (B, H, L_Q)

        # Select top-u queries
        M_top = M.topk(n_top, sorted=False).indices  # (B, H, n_top)

        # Gather top queries
        Q_reduce = torch.gather(
            Q, 2,
            M_top.unsqueeze(-1).expand(-1, -1, -1, D)
        )  # (B, H, n_top, D)

        # Full attention only on selected queries
        attn = torch.matmul(Q_reduce, K.transpose(-2, -1)) / math.sqrt(D)
        return attn, M_top

    def forward(self, queries, keys, values, attn_mask=None):
        B, L_Q, _ = queries.shape
        _, L_K, _ = keys.shape
        H = self.n_heads
        D = self.d_k

        Q = self.W_Q(queries).view(B, L_Q, H, D).transpose(1, 2)
        K = self.W_K(keys).view(B, L_K, H, D).transpose(1, 2)
        V = self.W_V(values).view(B, L_K, H, D).transpose(1, 2)

        U = max(1, int(self.factor * math.ceil(math.log(L_K + 1))))
        u = max(1, int(self.factor * math.ceil(math.log(L_Q + 1))))
        U = min(U, L_K)
        u = min(u, L_Q)

        scores_top, top_idx = self._prob_QK(Q, K, sample_k=U, n_top=u)

        if attn_mask is not None:
            scores_top = scores_top.masked_fill(
                attn_mask[:, :, :u, :L_K] == 0, float("-inf")
            )

        attn_weights = self.dropout(torch.softmax(scores_top, dim=-1))
        attn_out = torch.matmul(attn_weights, V)  # (B, H, u, D)

        # Fill context with mean of V for non-selected queries
        context = V.mean(dim=2, keepdim=True).expand(-1, -1, L_Q, -1).clone()
        context.scatter_(
            2,
            top_idx.unsqueeze(-1).expand(-1, -1, -1, D),
            attn_out,
        )

        context = context.transpose(1, 2).contiguous().view(B, L_Q, H * D)
        return self.out_proj(context)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class ConvLayer(nn.Module):
    """Distilling layer: Conv1d + ELU + MaxPool to halve sequence length."""

    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(d_model)
        self.activation = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model) → (B, d_model, L)
        x = x.transpose(1, 2)
        x = self.pool(self.activation(self.norm(self.conv(x))))
        return x.transpose(1, 2)


class EncoderLayer(nn.Module):
    """Single Informer encoder layer with ProbSparse attention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attention = ProbSparseAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn))
        ff = self.ff(x)
        x = self.norm2(x + self.dropout(ff))
        return x


class Encoder(nn.Module):
    """Informer encoder stack with distilling layers between attention layers."""

    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.conv_layers = nn.ModuleList([
            ConvLayer(d_model) for _ in range(n_layers - 1)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.conv_layers):
                x = self.conv_layers[i](x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    """Informer decoder layer with cross-attention."""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = ProbSparseAttention(d_model, n_heads, dropout)
        self.cross_attn = ProbSparseAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out)))
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x


class Decoder(nn.Module):
    """Informer decoder stack."""

    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, enc_out):
        for layer in self.layers:
            x = layer(x, enc_out)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Full Informer model
# ---------------------------------------------------------------------------

class InformerModel(nn.Module):
    """
    Informer: Encoder-Decoder architecture with ProbSparse attention.

    Input:  (batch, seq_len, n_features)
    Output: (batch, pred_len, n_features)
    """

    def __init__(
        self,
        n_features: int = 3,
        seq_len: int = 48,
        pred_len: int = 6,
        label_len: int = 24,
        d_model: int = 256,
        n_heads: int = 8,
        e_layers: int = 3,
        d_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.label_len = label_len

        # Embedding projections
        self.enc_embedding = nn.Linear(n_features, d_model)
        self.dec_embedding = nn.Linear(n_features, d_model)
        self.enc_pos = PositionalEncoding(d_model, max_len=seq_len + 100)
        self.dec_pos = PositionalEncoding(d_model, max_len=label_len + pred_len + 100)

        self.encoder = Encoder(d_model, n_heads, d_ff, e_layers, dropout)
        self.decoder = Decoder(d_model, n_heads, d_ff, d_layers, dropout)

        self.projection = nn.Linear(d_model, n_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            (batch, pred_len, n_features)
        """
        # Encoder
        enc_inp = self.dropout(self.enc_pos(self.enc_embedding(x)))
        enc_out = self.encoder(enc_inp)

        # Decoder input: last label_len steps + zeros for prediction
        dec_start = x[:, -self.label_len:, :]
        dec_zeros = torch.zeros(
            x.size(0), self.pred_len, x.size(2), device=x.device
        )
        dec_inp = torch.cat([dec_start, dec_zeros], dim=1)
        dec_inp = self.dropout(self.dec_pos(self.dec_embedding(dec_inp)))

        dec_out = self.decoder(dec_inp, enc_out)
        out = self.projection(dec_out[:, -self.pred_len:, :])
        return out
