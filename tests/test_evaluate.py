"""Unit tests for the shared metric helpers."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluate import core_metrics, evaluate_predictions, per_subject_scores, top_confusions


def test_perfect_predictions() -> None:
    y = np.array([0, 1, 2, 2])
    metrics = core_metrics(y, y, n_classes=3)
    assert metrics["accuracy"] == metrics["macro_f1"] == metrics["macro_f1_all_classes"] == 1.0


def test_macro_f1_ignores_absent_classes_but_all_classes_variant_does_not() -> None:
    y = np.array([0, 0, 1, 1])
    metrics = core_metrics(y, y, n_classes=4)
    assert metrics["macro_f1"] == 1.0
    assert metrics["macro_f1_all_classes"] == pytest.approx(0.5)


def test_per_subject_scores() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    table = per_subject_scores(y_true, y_pred, np.array([101, 101, 102, 102]))
    assert table["subject_id"].tolist() == [101, 102]
    assert table.loc[0, "macro_f1"] == 1.0
    assert table.loc[1, "accuracy"] == 0.5


def test_top_confusions() -> None:
    cm = np.array([[5, 3, 0], [1, 4, 0], [0, 0, 2]])
    pairs = top_confusions(cm, ["a", "b", "c"], k=5)
    assert pairs[0] == {"true": "a", "predicted": "b", "count": 3, "share_of_true": 0.375}
    assert len(pairs) == 2


def test_evaluate_predictions_adds_subject_and_class_metrics() -> None:
    y = np.array([0, 1, 1, 0])
    result = evaluate_predictions(y, y, np.array([101, 101, 102, 102]), ["lying", "sitting"])
    assert result.metrics["subject_101_macro_f1"] == 1.0
    assert result.metrics["f1_sitting"] == 1.0
    assert result.confusion.trace() == 4
