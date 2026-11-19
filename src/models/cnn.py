"""1D convolutional network for multichannel sensor windows."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    """Conv1d -> BatchNorm -> ReLU -> MaxPool(2), with 'same' padding."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        """Create the block.

        Args:
            in_channels: Input feature maps.
            out_channels: Output feature maps.
            kernel_size: Temporal kernel size (odd values keep the length exact).
        """
        super().__init__(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )


def conv_stack(in_channels: int, channels: Sequence[int], kernel_size: int) -> nn.Sequential:
    """Stack :class:`ConvBlock` s with the given widths.

    Args:
        in_channels: Number of sensor channels.
        channels: Output width of each block.
        kernel_size: Temporal kernel size.

    Returns:
        The feature extractor.
    """
    blocks: list[nn.Module] = []
    previous = in_channels
    for width in channels:
        blocks.append(ConvBlock(previous, width, kernel_size))
        previous = width
    return nn.Sequential(*blocks)


class CNN1D(nn.Module):
    """Conv blocks, global average pooling over time, dropout and a linear classifier.

    Global *mean* pooling is used instead of ``AdaptiveAvgPool1d`` because its
    backward pass is deterministic on CUDA.
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        conv_channels: Sequence[int] = (64, 128, 128),
        kernel_size: int = 7,
        dropout: float = 0.3,
    ) -> None:
        """Create the network.

        Args:
            n_channels: Sensor channels in the input.
            n_classes: Number of activity classes.
            conv_channels: Width of each conv block.
            kernel_size: Temporal kernel size.
            dropout: Dropout before the classifier.
        """
        super().__init__()
        if not conv_channels:
            raise ValueError("conv_channels must not be empty")
        self.features = conv_stack(n_channels, conv_channels, kernel_size)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(conv_channels[-1], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits.

        Args:
            x: Windows of shape ``(batch, channels, time)``.

        Returns:
            Logits of shape ``(batch, n_classes)``.
        """
        h = self.features(x).mean(dim=-1)
        return self.head(self.dropout(h))
