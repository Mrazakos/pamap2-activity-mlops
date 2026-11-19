"""Metrics, reports and figures shared by the baseline and the deep models.

The same functions and metric names are used for every model, so runs are
directly comparable in the MLflow UI (primary metric: macro-F1).
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.tracking import RunTracker

IntArray = npt.NDArray[np.int64]


@dataclass
class EvalResult:
    """Everything computed for one split."""

    metrics: dict[str, float]
    report: dict[str, object]
    per_subject: pd.DataFrame
    confusion: npt.NDArray[np.int64]


def core_metrics(y_true: IntArray, y_pred: IntArray, n_classes: int) -> dict[str, float]:
    """Accuracy, balanced accuracy, macro-F1 and weighted-F1.

    ``macro_f1`` (the primary metric) averages F1 over the classes that occur
    in ``y_true``, so a split in which a subject never performed an activity is
    not penalised for it. ``macro_f1_all_classes`` averages over all configured
    classes (absent ones count as 0) for reference.

    Args:
        y_true: True class indices.
        y_pred: Predicted class indices.
        n_classes: Number of classes.

    Returns:
        Metric name -> value.
    """
    labels = list(range(n_classes))
    present = sorted(np.unique(y_true).tolist())
    with warnings.catch_warnings():
        # Predicting a class that is absent from y_true is legitimate here.
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
            "macro_f1_all_classes": float(
                f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
            ),
            "weighted_f1": float(
                f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
            ),
        }


def per_subject_scores(y_true: IntArray, y_pred: IntArray, subjects: IntArray) -> pd.DataFrame:
    """Macro-F1 and accuracy per subject (generalisation to individual people).

    Each subject's macro-F1 averages over the classes that subject performed.

    Args:
        y_true: True class indices.
        y_pred: Predicted class indices.
        subjects: Subject id per window.

    Returns:
        DataFrame ``subject_id, windows, accuracy, macro_f1`` sorted by subject.
    """
    rows = []
    for subject in sorted(np.unique(subjects).tolist()):
        mask = subjects == subject
        present = sorted(np.unique(y_true[mask]).tolist())
        rows.append(
            {
                "subject_id": int(subject),
                "windows": int(mask.sum()),
                "accuracy": float(accuracy_score(y_true[mask], y_pred[mask])),
                "macro_f1": float(
                    f1_score(y_true[mask], y_pred[mask], labels=present, average="macro", zero_division=0)
                ),
            }
        )
    return pd.DataFrame(rows, columns=["subject_id", "windows", "accuracy", "macro_f1"])


def top_confusions(
    cm: npt.NDArray[np.int64], class_names: Sequence[str], k: int = 10
) -> list[dict[str, object]]:
    """The most frequent off-diagonal confusions (for the error analysis).

    Args:
        cm: Confusion matrix (rows = true, columns = predicted).
        class_names: Class names.
        k: How many pairs to return.

    Returns:
        List of ``{"true", "predicted", "count", "share_of_true"}`` sorted by count.
    """
    off = cm.copy()
    np.fill_diagonal(off, 0)
    order = np.argsort(off, axis=None, kind="stable")[::-1][:k]
    totals = cm.sum(axis=1)
    pairs = []
    for flat in order:
        i, j = np.unravel_index(flat, cm.shape)
        if off[i, j] == 0:
            break
        pairs.append(
            {
                "true": class_names[i],
                "predicted": class_names[j],
                "count": int(off[i, j]),
                "share_of_true": float(off[i, j] / max(totals[i], 1)),
            }
        )
    return pairs


def evaluate_predictions(
    y_true: IntArray, y_pred: IntArray, subjects: IntArray, class_names: Sequence[str]
) -> EvalResult:
    """Compute all metrics and tables for one split.

    Args:
        y_true: True class indices.
        y_pred: Predicted class indices.
        subjects: Subject id per window.
        class_names: Class names in index order.

    Returns:
        The evaluation result.
    """
    n = len(class_names)
    metrics = core_metrics(y_true, y_pred, n)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(n)),
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n))).astype(np.int64)
    per_subject = per_subject_scores(y_true, y_pred, subjects)
    for row in per_subject.itertuples():
        metrics[f"subject_{row.subject_id}_macro_f1"] = float(row.macro_f1)
    for name in class_names:
        metrics[f"f1_{name}"] = float(report[name]["f1-score"])  # type: ignore[index]
    return EvalResult(
        metrics=metrics,
        report={"classes": report, "top_confusions": top_confusions(cm, class_names)},
        per_subject=per_subject,
        confusion=cm,
    )


def confusion_figure(cm: npt.NDArray[np.int64], class_names: Sequence[str], title: str) -> Figure:
    """Row-normalised confusion matrix heat map (recall on the diagonal).

    Args:
        cm: Confusion matrix.
        class_names: Class names.
        title: Figure title.

    Returns:
        The figure.
    """
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(
        norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "share of true class"},
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def learning_curve_figure(history: Sequence[Mapping[str, float]]) -> Figure:
    """Train/validation loss and validation macro-F1 per epoch.

    Args:
        history: One mapping per epoch with ``epoch``, ``train_loss``, ``val_loss``,
            ``train_macro_f1`` and ``val_macro_f1``.

    Returns:
        The figure.
    """
    frame = pd.DataFrame(list(history))
    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(frame["epoch"], frame["train_loss"], label="train")
    ax_loss.plot(frame["epoch"], frame["val_loss"], label="validation")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("cross-entropy")
    ax_loss.legend()
    ax_f1.plot(frame["epoch"], frame["train_macro_f1"], label="train")
    ax_f1.plot(frame["epoch"], frame["val_macro_f1"], label="validation")
    ax_f1.set_xlabel("epoch")
    ax_f1.set_ylabel("macro-F1")
    ax_f1.legend()
    fig.tight_layout()
    return fig


def log_evaluation(
    tracker: RunTracker, result: EvalResult, split: str, class_names: Sequence[str], step: int | None = None
) -> None:
    """Log metrics (prefixed with the split) and artifacts of one evaluation.

    Args:
        tracker: Active run.
        result: Evaluation result.
        split: ``val`` or ``test`` (used as metric prefix).
        class_names: Class names.
        step: Optional metric step.
    """
    tracker.log_metrics({f"{split}_{k}": v for k, v in result.metrics.items()}, step=step)
    tracker.log_dict(result.report, f"reports/classification_report_{split}.json")
    tracker.log_dict(result.per_subject.to_dict(orient="records"), f"reports/per_subject_{split}.json")
    tracker.log_text(result.per_subject.to_string(index=False), f"reports/per_subject_{split}.txt")
    tracker.log_figure(
        confusion_figure(result.confusion, class_names, f"Confusion matrix ({split})"),
        f"figures/confusion_{split}.png",
    )
