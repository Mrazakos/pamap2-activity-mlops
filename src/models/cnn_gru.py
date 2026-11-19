"""CNN front-end followed by a GRU over the downsampled time axis."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from src.models.cnn import conv_stack


class CNNGRU(nn.Module):
    """Conv blocks extract local motion patterns; a GRU models their temporal order.

    The classifier reads the GRU's last hidden state.
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        conv_channels: Sequence[int] = (64, 128, 128),
        kernel_size: int = 7,
        dropout: float = 0.3,
        gru_hidden: int = 128,
        gru_layers: int = 1,
    ) -> None:
        """Create the network.

        Args:
            n_channels: Sensor channels in the input.
            n_classes: Number of activity classes.
            conv_channels: Width of each conv block.
            kernel_size: Temporal kernel size.
            dropout: Dropout before the classifier (and between GRU layers).
            gru_hidden: GRU hidden size.
            gru_layers: Number of stacked GRU layers.
        """
        super().__init__()
        if not conv_channels:
            raise ValueError("conv_channels must not be empty")
        self.features = conv_stack(n_channels, conv_channels, kernel_size)
        self.gru = nn.GRU(
            input_size=conv_channels[-1],
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(gru_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits.

        Args:
            x: Windows of shape ``(batch, channels, time)``.

        Returns:
            Logits of shape ``(batch, n_classes)``.
        """
        h = self.features(x).transpose(1, 2)  # (batch, time, features)
        _, last = self.gru(h)  # last: (layers, batch, hidden)
        return self.head(self.dropout(last[-1]))
