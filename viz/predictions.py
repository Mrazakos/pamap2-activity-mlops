"""What a trained model does: example inputs, a prediction timeline, confusions, per-class F1.

Usage (repository root; bundles from ``dvc pull`` or a training run)::

    python -m viz.predictions models/baseline                  # one model
    python -m viz.predictions models/baseline models/deep      # side by side

Figures go to ``reports/figures/model/``. The default split is **validation**: the test
subject is evaluated once, for the final models, so ``--split test`` additionally needs
``--final``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from src.config import Config
from src.data_versions import resolve_clean_dir
from src.models.bundle import load_bundle
from src.windows import load_clean, prepare_data
from viz import style

OUT = style.FIGURES / "model"
GROUPS = ("hand", "chest", "ankle")


@dataclass
class ModelPredictions:
    """Predictions of one model on one split."""

    name: str
    proba: npt.NDArray[np.float64]
    y: npt.NDArray[np.int64]
    x: npt.NDArray[np.float32]
    channels: list[str]
    class_names: list[str]
    minutes: npt.NDArray[np.float64]
    window_seconds: float
    subjects: list[int]

    @property
    def pred(self) -> npt.NDArray[np.int64]:
        """Predicted class indices."""
        return self.proba.argmax(axis=1)

    @property
    def macro_f1(self) -> float:
        """Macro-F1 over all classes."""
        return float(f1_score(self.y, self.pred, average="macro", labels=range(len(self.class_names))))


def model_name(meta_config: dict, kind: str, run_id: str) -> str:
    """Readable model name.

    Args:
        meta_config: Config stored in the bundle.
        kind: Bundle kind (``sklearn`` or ``torch``).
        run_id: MLflow run id.

    Returns:
        E.g. ``random_forest (4dd27f1a)``.
    """
    name = meta_config["baseline"]["model"] if kind == "sklearn" else meta_config["model"]["name"]
    return f"{name} ({run_id[:8]})" if run_id else name


def predict_split(model_dir: Path, split: str) -> ModelPredictions:
    """Rebuild a split exactly as in training and predict it.

    Args:
        model_dir: Bundle folder.
        split: ``val`` or ``test``.

    Returns:
        Predictions plus the inputs and their recording times.

    Raises:
        RuntimeError: If the normalisation statistics differ from the bundle's.
    """
    predictor = load_bundle(model_dir)
    cfg = Config.model_validate(predictor.meta.config)
    clean, meta = load_clean(resolve_clean_dir(cfg))
    data = prepare_data(cfg, clean, meta["channels"], splits_needed=(split,))  # type: ignore[arg-type]
    if not (np.array_equal(data.norm_mean, predictor.mean) and np.array_equal(data.norm_std, predictor.std)):
        raise RuntimeError("normalisation statistics differ from the bundle: data or split changed")
    part = data.splits[split]

    # Window start rows index the signal store, which is the clean frame (decimated per segment).
    timestamps = clean["timestamp"]
    if cfg.windows.downsample > 1:
        position = clean.groupby("segment_id", sort=False).cumcount().to_numpy()
        timestamps = timestamps[position % cfg.windows.downsample == 0]
    t = timestamps.to_numpy()[part.index["start"].to_numpy()]
    window_seconds = data.window_length / data.sample_rate_hz

    return ModelPredictions(
        name=model_name(predictor.meta.config, predictor.meta.kind, predictor.meta.run_id),
        proba=predictor.predict_proba(part.x),
        y=part.y,
        x=part.x,
        channels=data.channels,
        class_names=[c.replace("_", " ") for c in data.class_names],
        minutes=(t - t.min()) / 60.0,
        window_seconds=window_seconds,
        subjects=data.split_subjects[split],
    )


def channel_ticks(channels: list[str]) -> tuple[list[float], list[str]]:
    """One y tick per sensor group (e.g. ``ankle acc16``) at the middle of its rows.

    Args:
        channels: Channel names in model order.

    Returns:
        Tick positions and labels.
    """
    groups: dict[str, list[int]] = {}
    for i, c in enumerate(channels):
        key = c if c == "heart_rate" else c.rsplit("_", 1)[0].replace("_", " ")
        groups.setdefault(key, []).append(i)
    return [float(np.mean(v)) for v in groups.values()], list(groups)


def task_examples(m: ModelPredictions, split: str) -> None:
    """Two windows through the model: a confident hit and the most confident mistake.

    Args:
        m: Predictions.
        split: Split name for the title.
    """
    pred, conf = m.pred, m.proba.max(axis=1)
    right = np.flatnonzero(pred == m.y)
    wrong = np.flatnonzero(pred != m.y)
    # A typical hit (median confidence among correct windows), not the easiest one.
    hit = right[np.argsort(conf[right])[len(right) // 2]]
    rows = [("correct", hit)]
    if len(wrong):
        rows.append(("most confident mistake", wrong[np.argmax(conf[wrong])]))

    fig, axes = plt.subplots(
        len(rows),
        2,
        figsize=(12, 4.0 * len(rows) + 0.8),
        gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.08, "hspace": 0.12},
        squeeze=False,
        layout="constrained",
    )
    seconds = np.arange(m.x.shape[2]) * m.window_seconds / m.x.shape[2]
    ticks, labels = channel_ticks(m.channels)
    for (kind, i), (ax_in, ax_out) in zip(rows, axes, strict=True):
        image = ax_in.imshow(
            m.x[i],
            aspect="auto",
            cmap=style.DIVERGING,
            vmin=-3,
            vmax=3,
            interpolation="nearest",
            extent=(0, seconds[-1], len(m.channels) - 0.5, -0.5),
        )
        ax_in.set_yticks(ticks, labels, fontsize=7.5)
        ax_in.grid(False)
        ax_in.set_xlabel("time in window (s)")
        ax_in.set_title(f"input: {len(m.channels)} channels x {m.x.shape[2]} samples", loc="left")

        order = np.arange(len(m.class_names))[::-1]
        colors = [style.CONTEXT] * len(m.class_names)
        colors[m.y[i]] = style.BLUE
        if pred[i] != m.y[i]:
            colors[pred[i]] = style.ORANGE
        ax_out.barh(order, m.proba[i], color=colors, height=0.7)
        ax_out.set_yticks(order, m.class_names, fontsize=8)
        ax_out.set_xlim(0, 1)
        ax_out.grid(axis="y", visible=False)
        ax_out.set_xlabel("predicted probability")
        verdict = (
            f"true {m.class_names[m.y[i]]} → predicted {m.class_names[pred[i]]} ({conf[i]:.0%})"
            if pred[i] != m.y[i]
            else f"{m.class_names[m.y[i]]}, {conf[i]:.0%} confident"
        )
        ax_out.set_title(f"output ({kind}): {verdict}", loc="left", fontsize=8.5)
    cbar = fig.colorbar(image, ax=axes[:, 0].tolist(), location="bottom", shrink=0.4, aspect=30)
    cbar.set_label("normalised value (z-score, training statistics)", color=style.INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (style.BLUE, style.ORANGE)]
    fig.legend(handles, ["true activity", "wrong prediction"], loc="upper right", ncol=2, frameon=False)
    style.title(
        fig, f"What the model sees and says: {m.name}, {split} subject {', '.join(map(str, m.subjects))}"
    )
    style.save(fig, OUT / f"task_examples_{slug(m.name)}_{split}.png")


def timeline(models: list[ModelPredictions], split: str) -> None:
    """True activity vs prediction over the recording, one lane per activity.

    Args:
        models: Predictions of one or more models on the same split.
        split: Split name for the title.
    """
    names = models[0].class_names
    fig, axes = plt.subplots(len(models), 1, figsize=(12, 3.9 * len(models)), sharex=True, squeeze=False)
    for m, ax in zip(models, axes[:, 0], strict=True):
        correct = m.pred == m.y
        ax.barh(m.y, m.window_seconds / 60.0, left=m.minutes, height=0.8, color=style.CONTEXT, lw=0)
        ax.scatter(m.minutes[correct] + m.window_seconds / 120, m.pred[correct], s=7, color=style.BLUE, lw=0)
        ax.scatter(
            m.minutes[~correct] + m.window_seconds / 120,
            m.pred[~correct],
            s=18,
            color=style.ORANGE,
            edgecolor=style.SURFACE,
            lw=0.8,
            zorder=3,
        )
        ax.set_yticks(range(len(names)), names, fontsize=8)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_title(
            f"{m.name}: macro-F1 {m.macro_f1:.3f}, {int((~correct).sum())} of {len(m.y)} windows wrong",
            loc="left",
        )
    axes[-1, 0].set_xlabel("minutes since the start of the recording (gaps = transient periods removed)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=style.CONTEXT),
        plt.Line2D([], [], marker="o", ls="", color=style.BLUE, markersize=4),
        plt.Line2D([], [], marker="o", ls="", color=style.ORANGE, markersize=6),
    ]
    fig.legend(
        handles,
        ["true activity", "correct prediction", "wrong prediction (in the predicted lane)"],
        loc="upper right",
        ncol=3,
        frameon=False,
    )
    style.title(fig, f"Predictions over time, {split} subject {', '.join(map(str, models[0].subjects))}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    style.save(fig, OUT / f"timeline_{split}.png")


def confusions(models: list[ModelPredictions], split: str) -> None:
    """Row-normalised confusion matrices, one per model.

    Args:
        models: Predictions.
        split: Split name for the title.
    """
    names = models[0].class_names
    k = len(names)
    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 5.8), squeeze=False)
    for m, ax in zip(models, axes[0], strict=True):
        cm = confusion_matrix(m.y, m.pred, labels=range(k)).astype(float)
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(cm, cmap=style.SEQUENTIAL, vmin=0, vmax=1)
        for r, c in zip(*np.nonzero(cm >= 0.02), strict=True):
            ax.text(
                c,
                r,
                f"{cm[r, c]:.0%}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="#ffffff" if cm[r, c] > 0.55 else style.INK2,
            )
        ax.set_xticks(range(k), names, rotation=50, ha="right", fontsize=7.5)
        ax.set_yticks(range(k), names, fontsize=7.5)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.grid(False)
        ax.set_title(f"{m.name}: macro-F1 {m.macro_f1:.3f}", loc="left")
    style.title(fig, f"Which activities get confused ({split}; rows sum to 100 %)")
    fig.tight_layout()
    style.save(fig, OUT / f"confusion_{split}.png")


def per_class_f1(models: list[ModelPredictions], split: str) -> None:
    """Per-class F1 as a dot plot, one series per model.

    Args:
        models: Predictions (at most three, one palette slot each).
        split: Split name for the title.
    """
    names = models[0].class_names
    scores = pd.DataFrame(
        {
            m.name: f1_score(m.y, m.pred, average=None, labels=range(len(names)), zero_division=0)
            for m in models
        },
        index=names,
    ).sort_values(models[0].name)
    fig, ax = plt.subplots(figsize=(7.5, 0.33 * len(names) + 1.4))
    for (name, col), color in zip(scores.items(), style.SERIES, strict=False):
        ax.scatter(
            col, range(len(col)), s=40, color=color, edgecolor=style.SURFACE, lw=1.5, label=name, zorder=3
        )
    ax.set_yticks(range(len(scores)), scores.index, fontsize=8)
    ax.set_xlim(min(0.5, float(scores.min().min()) - 0.05), 1.0)
    ax.set_xlabel("F1")
    ax.grid(axis="y", visible=False)
    if len(models) > 1:
        ax.legend(frameon=False, loc="lower right", fontsize=8)
    style.title(fig, f"F1 per activity ({split})" + ("" if len(models) > 1 else f": {models[0].name}"))
    fig.tight_layout()
    style.save(fig, OUT / f"per_class_f1_{split}.png")


def slug(name: str) -> str:
    """File-name-safe model name.

    Args:
        name: Display name.

    Returns:
        Lower-case name without spaces or brackets.
    """
    return name.split(" ")[0].lower()


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model_dirs", nargs="+", type=Path, help="bundle folders (at most three)")
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--final", action="store_true", help="required for --split test")
    args = parser.parse_args()
    if args.split == "test" and not args.final:
        parser.error("the test subject is evaluated once, for the final models: add --final")
    if len(args.model_dirs) > 3:
        parser.error("at most three models (one colour each)")

    style.apply()
    models = [predict_split(d, args.split) for d in args.model_dirs]
    for m in models:
        print(f"{m.name}: macro-F1 {m.macro_f1:.4f} on {args.split} ({len(m.y)} windows)")
        task_examples(m, args.split)
    timeline(models, args.split)
    confusions(models, args.split)
    per_class_f1(models, args.split)


if __name__ == "__main__":
    main()
