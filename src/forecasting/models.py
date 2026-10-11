"""PyTorch forecasting models.

Primary candidate: CNN-LSTM (Conv1d local temporal features -> LSTM temporal
dependencies -> scalar soil-moisture forecast at a given horizon). A plain LSTM
is provided as an architecture ablation. All models share the same input/ouput
contract (batch, history, n_features) -> (batch, 1) so evaluation is identical.
"""
from __future__ import annotations

import torch
from torch import nn


class CNNLSTM(nn.Module):
    """Conv1d layers over the time axis, then an LSTM, then a scalar head."""

    def __init__(self, history: int, n_features: int, hidden: int = 64,
                 channels: int = 32, dropout: float = 0.1, kernel: int = 5,
                 num_layers: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, channels, kernel_size=kernel, padding=kernel // 2),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=kernel, padding=kernel // 2),
            nn.ReLU(),
        )
        lstm_drop = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(channels, hidden, num_layers=num_layers, batch_first=True,
                            dropout=lstm_drop)
        self.head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.transpose(1, 2))          # (B, C, T)
        h = h.transpose(1, 2)                     # (B, T, C)
        out, _ = self.lstm(h)
        return self.head(out[:, -1])


class LSTMForecaster(nn.Module):
    """Standard LSTM baseline (no convolutional front-end)."""

    def __init__(self, history: int, n_features: int, hidden: int = 64,
                 dropout: float = 0.1, num_layers: int = 1):
        super().__init__()
        lstm_drop = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(n_features, hidden, num_layers=num_layers, batch_first=True,
                            dropout=lstm_drop)
        self.head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


class QuantileCNNLSTM(CNNLSTM):
    """CNN-LSTM with three scalar heads: location, lower and upper quantiles.

    Trained on a joint pinball loss, so the network directly returns calibrated
    quantile forecasts of soil moisture (an alternative to the deep ensemble).
    """

    def __init__(self, history: int, n_features: int, hidden: int = 64,
                 channels: int = 32, dropout: float = 0.1, kernel: int = 5,
                 alpha: float = 0.10, num_layers: int = 1):
        super().__init__(history, n_features, hidden, channels, dropout, kernel, num_layers)
        self.history = history
        self.n_features = n_features
        self.alpha = alpha
        self.head = nn.Sequential(
            nn.Linear(hidden, 16), nn.ReLU(),
            nn.Linear(16, 3),  # loc, q_low, q_high
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.lstm(h)
        return self.head(out[:, -1])


def pinball_loss(pred: torch.Tensor, y: torch.Tensor, alpha: float = 0.10,
                 tau_match: torch.Tensor | None = None) -> torch.Tensor:
    """Joint loss: MSE on the location head + pinball on the two quantile heads."""
    loc, q_lo, q_hi = pred[:, 0], pred[:, 1], pred[:, 2]
    loc_loss = (loc - y).pow(2).mean()
    err_lo = y - q_lo
    err_hi = y - q_hi
    pin_lo = torch.maximum(alpha * err_lo, (alpha - 1) * err_lo).mean()
    pin_hi = torch.maximum((1 - alpha) * err_hi, -alpha * err_hi).mean()
    return loc_loss + pin_lo + pin_hi


def build_model(kind: str, history: int, n_features: int, hidden: int = 64,
                channels: int = 32, dropout: float = 0.1, num_layers: int = 1) -> nn.Module:
    kind = kind.lower()
    if kind == "cnn_lstm":
        return CNNLSTM(history, n_features, hidden, channels, dropout, num_layers=num_layers)
    if kind == "lstm":
        return LSTMForecaster(history, n_features, hidden, dropout, num_layers=num_layers)
    if kind == "quantile_cnn_lstm":
        return QuantileCNNLSTM(history, n_features, hidden, channels, dropout, num_layers=num_layers)
    raise ValueError(f"unknown model kind {kind!r}")