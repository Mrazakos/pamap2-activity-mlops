"""Model definitions and the factory that builds them from the config."""

from __future__ import annotations

from torch import nn

from src.config import ModelConfig
from src.models.cnn import CNN1D
from src.models.cnn_gru import CNNGRU


def build_model(cfg: ModelConfig, n_channels: int, n_classes: int) -> nn.Module:
    """Instantiate the deep model named in the config.

    Args:
        cfg: Model settings.
        n_channels: Sensor channels per window.
        n_classes: Number of activity classes.

    Returns:
        An untrained ``nn.Module`` producing logits.
    """
    if cfg.name == "cnn":
        return CNN1D(n_channels, n_classes, cfg.conv_channels, cfg.kernel_size, cfg.dropout)
    return CNNGRU(
        n_channels,
        n_classes,
        cfg.conv_channels,
        cfg.kernel_size,
        cfg.dropout,
        cfg.gru_hidden,
        cfg.gru_layers,
    )


__all__ = ["CNN1D", "CNNGRU", "build_model"]
