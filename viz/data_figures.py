"""Data figures that complement the MLflow EDA run: signals, class balance, embeddings, subjects.

Usage (repository root, after ``dvc pull``)::

    python -m viz.data_figures

Reads ``data/clean``, ``data/processed`` and ``data/features`` (the checked-out data version)
and writes ``reports/figures/data/``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from src.constants import ACTIVITY_NAMES, PROTOCOL_ACTIVITIES
from src.utils import read_json
from viz import style

OUT = style.FIGURES / "data"
NAMES = [ACTIVITY_NAMES[a].replace("_", " ") for a in PROTOCOL_ACTIVITIES]
SPLIT_OF = {101: "test", 102: "val"}
SENSORS = (("hand", style.BLUE), ("chest", style.ORANGE), ("ankle", style.AQUA))


def signals_per_activity(
    windows: pd.DataFrame, acc: dict[str, npt.NDArray], length: int, rate: float
) -> None:
    """One window per activity: acceleration magnitude of each sensor (subject 105).

    Args:
        windows: Window index (``start`` = row of the clean data).
        acc: Acceleration magnitude per sensor, one value per clean row.
        length: Window length in samples.
        rate: Sample rate in Hz.
    """
    fig, axes = plt.subplots(4, 3, figsize=(11, 9.5), sharex=True)
    t = np.arange(length) / rate
    for k, ax in enumerate(axes.flat):
        w = windows[(windows.subject_id == 105) & (windows.label == k)]
        if w.empty:
            w = windows[(windows.label == k) & (windows.split == "train")]
        start = int(w.start.iloc[len(w) // 2])
        for sensor, color in SENSORS:
            ax.plot(t, acc[sensor][start : start + length], color=color, lw=1.2)
        ax.set_title(NAMES[k], loc="left")
        ax.set_ylim(0, 60)
        ax.axhline(9.81, color=style.MUTED, lw=0.8, ls=(0, (2, 2)))
    for ax in axes[-1]:
        ax.set_xlabel("time in window (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("|acc| (m/s²)")
    handles = [plt.Line2D([], [], color=c, lw=2) for _, c in SENSORS]
    fig.legend(
        handles,
        ["hand (wrist)", "chest", "ankle"],
        loc="upper right",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(1, 1.01),
    )
    style.title(
        fig,
        f"One {length / rate:.2f} s window per activity (subject 105, ±16 g accelerometer magnitude; "
        "dashed = 1 g, y capped at 60)",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    style.save(fig, OUT / "01_signals_per_activity.png")


def windows_per_split(windows: pd.DataFrame) -> None:
    """Windows per activity in each split.

    Args:
        windows: Window index.
    """
    counts = windows.groupby(["split", "label"]).size().unstack(0).fillna(0).astype(int)
    order = counts["train"].sort_values().index
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.6), sharey=True)
    for ax, split in zip(axes, ("train", "val", "test"), strict=True):
        vals = counts.loc[order, split]
        ax.barh(range(len(order)), vals, color=style.BLUE, height=0.7)
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v}", va="center", fontsize=7.5, color=style.INK2)
        subj = sorted(windows[windows.split == split].subject_id.unique())
        who = f"subject {subj[0]}" if len(subj) == 1 else f"{len(subj)} subjects"
        ax.set_title(f"{split} · {int(vals.sum())} windows · {who}", loc="left", fontsize=8.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, vals.max() * 1.18)
        ax.set_xlabel("windows")
    axes[0].set_yticks(range(len(order)), [NAMES[i] for i in order])
    style.title(fig, "Windows per activity and split")
    fig.tight_layout()
    style.save(fig, OUT / "02_windows_per_split_class.png")


def embedding_grid(
    emb: npt.NDArray,
    groups: list[int],
    mask_of: Callable[[int], npt.NDArray[np.bool_]],
    title_of: Callable[[int, int], str],
    shape: tuple[int, int],
    suptitle: str,
    name: str,
) -> None:
    """Small multiples of one embedding, each panel highlighting one group.

    Args:
        emb: ``(n, 2)`` embedding.
        groups: Group keys, one panel each.
        mask_of: Group key -> boolean mask of its points.
        title_of: (group key, count) -> panel title.
        shape: Grid rows and columns.
        suptitle: Figure title.
        name: Output file name.
    """
    rows, cols = shape
    fig, axes = plt.subplots(rows, cols, figsize=(2.9 * cols, 2.7 * rows))
    for ax, g in zip(axes.flat, groups, strict=False):
        m = mask_of(g)
        ax.scatter(emb[~m, 0], emb[~m, 1], s=2, color=style.CONTEXT, lw=0, rasterized=True)
        ax.scatter(emb[m, 0], emb[m, 1], s=4, color=style.BLUE, lw=0, rasterized=True)
        ax.set_title(title_of(g, int(m.sum())), loc="left", fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in list(axes.flat)[len(groups) :]:
        ax.set_visible(False)
    style.title(fig, suptitle)
    fig.tight_layout()
    style.save(fig, OUT / name)


def feature_embeddings(features: pd.DataFrame) -> None:
    """t-SNE of the window features, by activity and by subject.

    Args:
        features: Feature table (``f__*`` columns, ``label``, ``subject_id``).
    """
    x = StandardScaler().fit_transform(features.filter(like="f__").to_numpy())
    x = PCA(n_components=50, random_state=42).fit_transform(x)
    emb = TSNE(n_components=2, perplexity=40, init="pca", random_state=42).fit_transform(x)
    lab, sub = features.label.to_numpy(), features.subject_id.to_numpy()
    n_feat = features.filter(like="f__").shape[1]
    embedding_grid(
        emb,
        list(range(len(NAMES))),
        lambda k: lab == k,
        lambda k, n: f"{NAMES[k]}  (n={n})",
        (3, 4),
        f"t-SNE of the {n_feat} window features, one panel per activity (gray = all other windows)",
        "03_tsne_by_activity.png",
    )
    embedding_grid(
        emb,
        sorted(set(sub.tolist())),
        lambda s: sub == s,
        lambda s, n: f"subject {s} · {SPLIT_OF.get(s, 'train')}  (n={n})",
        (3, 3),
        "Same embedding, one panel per subject: do people form their own clusters?",
        "04_tsne_by_subject.png",
    )


def subject_spread(windows: pd.DataFrame, ankle: npt.NDArray, heart_rate: npt.NDArray, length: int) -> None:
    """Per subject and activity: movement intensity and heart rate.

    Args:
        windows: Window index.
        ankle: Ankle acceleration magnitude per clean row.
        heart_rate: Heart rate per clean row.
        length: Window length in samples.
    """
    idx = windows.start.to_numpy()[:, None] + np.arange(length)
    per = (
        windows.assign(intensity=ankle[idx].std(axis=1), hr=heart_rate[idx].mean(axis=1))
        .groupby(["label", "subject_id"])[["intensity", "hr"]]
        .median()
        .reset_index()
    )
    order = per.groupby("label").intensity.median().sort_values().index
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    panels = (
        (axes[0], "intensity", "median std of ankle |acc| per window (m/s²)"),
        (axes[1], "hr", "median heart rate per window (bpm)"),
    )
    for ax, col, xlabel in panels:
        for i, k in enumerate(order):
            for _, r in per[per.label == k].iterrows():
                s = int(r.subject_id)
                ax.scatter(
                    r[col],
                    i + rng.uniform(-0.12, 0.12),
                    s=34 if s in SPLIT_OF else 16,
                    color={101: style.ORANGE, 102: style.BLUE}.get(s, style.MUTED),
                    edgecolor=style.SURFACE,
                    lw=1.2,
                    zorder=3 if s in SPLIT_OF else 2,
                )
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(order)), [NAMES[k] for k in order])
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=c, markersize=ms)
        for c, ms in ((style.MUTED, 5), (style.BLUE, 7), (style.ORANGE, 7))
    ]
    fig.legend(
        handles,
        ["training subjects", "102 (validation)", "101 (test)"],
        loc="upper right",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(1, 1.01),
    )
    style.title(fig, "How much do people differ? One dot per subject and activity")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    style.save(fig, OUT / "05_subject_spread.png")


def main() -> None:
    """Build all data figures."""
    style.apply()
    meta = read_json(Path("data/processed/meta.json"))
    length, rate = int(meta["window_length"]), float(meta["sample_rate_hz"])
    windows = pd.read_parquet("data/processed/windows.parquet")
    columns = [f"{s}_acc16_{a}" for s, _ in SENSORS for a in "xyz"] + ["heart_rate"]
    clean = pd.read_parquet("data/clean/clean.parquet", columns=columns)
    acc = {s: np.linalg.norm(clean[[f"{s}_acc16_{a}" for a in "xyz"]].to_numpy(), axis=1) for s, _ in SENSORS}

    signals_per_activity(windows, acc, length, rate)
    windows_per_split(windows)
    feature_embeddings(pd.read_parquet("data/features/features.parquet"))
    subject_spread(windows, acc["ankle"], clean["heart_rate"].to_numpy(), length)


if __name__ == "__main__":
    main()
