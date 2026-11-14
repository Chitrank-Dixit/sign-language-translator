"""Neural network sequence classification models for sign language gestures.

Implements exchangeable LSTM and Transformer backbones adhering to unified
tensor input/output contracts.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from sign_translator.schemas import ModelArchitecture, PipelineConfig


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence tokens."""

    def __init__(self, d_model: int, max_len: int = 500) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to sequence tensor of shape (B, T, d_model)."""
        return x + self.pe[:, : x.size(1)]


class LSTMClassifier(nn.Module):
    """Multi-layer LSTM classification backbone."""

    def __init__(
        self,
        input_dim: int = 126,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(p=dropout)
        linear_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(linear_input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass mapping (B, T, D) to (B, num_classes)."""
        _, (h_n, _) = self.lstm(x)
        if self.bidirectional:
            h_fwd = h_n[-2, :, :]
            h_bwd = h_n[-1, :, :]
            h_rep = torch.cat([h_fwd, h_bwd], dim=-1)
        else:
            h_rep = h_n[-1, :, :]
        return self.fc(self.dropout(h_rep))


class TransformerClassifier(nn.Module):
    """Self-attention Transformer classification backbone."""

    def __init__(
        self,
        input_dim: int = 126,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 5,
        nhead: int = 4,
        dropout: float = 0.2,
        dim_feedforward: int | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.nhead = nhead

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(d_model=hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward or (hidden_dim * 4),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass mapping (B, T, D) to (B, num_classes)."""
        h = self.input_proj(x)
        h = self.pos_encoder(h)
        out = self.transformer_encoder(h)
        pooled = out.mean(dim=1)
        return self.fc(self.dropout(self.norm(pooled)))


def build_model(config: PipelineConfig) -> nn.Module:
    """Instantiate a classification model from pipeline configuration.

    Args:
        config: PipelineConfig containing architecture selection and hyperparameters.

    Returns:
        Instantiated nn.Module ready for training or inference.

    Raises:
        ValueError: If config specifies an unsupported architecture.
    """
    num_classes = len(config.vocabulary)
    if config.architecture == ModelArchitecture.LSTM:
        return LSTMClassifier(
            input_dim=config.feature_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_classes=num_classes,
            dropout=config.dropout,
            bidirectional=config.bidirectional,
        )
    if config.architecture == ModelArchitecture.TRANSFORMER:
        return TransformerClassifier(
            input_dim=config.feature_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_classes=num_classes,
            nhead=config.transformer_heads,
            dropout=config.dropout,
        )
    raise ValueError(f"Unsupported model architecture: {config.architecture}")
