"""
gru.py
Bidirectional GRU model for multi-step, multi-resource time-series forecasting.
"""

import torch
import torch.nn as nn


class GRUModel(nn.Module):
    """
    Bidirectional GRU with fully-connected output head.

    Architecture:
        Input (batch, seq_len, n_features)
        → Bidirectional GRU (2 layers, 128 hidden, dropout=0.2)
        → FC → Output (batch, horizon, n_features)
    """

    def __init__(
        self,
        n_features: int = 3,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_horizon: int = 6,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.output_horizon = output_horizon

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_horizon * n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            (batch, output_horizon, n_features)
        """
        # gru_out: (batch, seq_len, hidden*2)
        gru_out, _ = self.gru(x)

        # Use last timestep output
        last = gru_out[:, -1, :]              # (batch, hidden*2)
        last = self.dropout(last)
        out = self.fc(last)                    # (batch, horizon * n_features)
        out = out.view(-1, self.output_horizon, self.n_features)
        return out
