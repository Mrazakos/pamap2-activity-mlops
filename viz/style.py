"""Shared figure style: reference palette (light mode) and matplotlib defaults."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

# Categorical slots 1-3 (validated all-pairs, light surface) and chart chrome.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = (BLUE, ORANGE, AQUA)
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, CONTEXT = "#e1e0d9", "#c3c2b7", "#d9d8d2"

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#86b6ef", "#f0efec", "#f19a99", "#b22d2d"]
)

FIGURES = Path("reports/figures")


def apply() -> None:
    """Set the matplotlib defaults used by every figure."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": ["Segoe UI", "DejaVu Sans"],
            "font.size": 9,
            "text.color": INK,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "axes.edgecolor": AXIS,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
        }
    )


def title(fig: Figure, text: str) -> None:
    """Left-aligned figure title.

    Args:
        fig: Figure.
        text: Title text.
    """
    fig.suptitle(text, x=0.01, ha="left", fontsize=11, fontweight="bold")


def save(fig: Figure, path: Path) -> None:
    """Write a figure as PNG and close it.

    Args:
        fig: Figure.
        path: Output file; parent folders are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)
