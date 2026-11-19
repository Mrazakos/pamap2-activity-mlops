"""Classical baseline: RandomForest (or LogisticRegression) on hand-crafted features."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import BaselineConfig


def build_baseline(cfg: BaselineConfig, seed: int) -> ClassifierMixin:
    """Create an unfitted baseline estimator with a fixed ``random_state``.

    RandomForest results do not depend on ``n_jobs`` when ``random_state`` is fixed,
    so the model is reproducible on any core count.

    Args:
        cfg: Baseline settings.
        seed: Random seed.

    Returns:
        A scikit-learn classifier.
    """
    if cfg.model == "random_forest":
        return RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            min_samples_leaf=cfg.min_samples_leaf,
            class_weight=cfg.class_weight,
            n_jobs=cfg.n_jobs,
            random_state=seed,
        )
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=cfg.lr_c,
                    max_iter=cfg.lr_max_iter,
                    class_weight=cfg.class_weight,
                    random_state=seed,
                ),
            ),
        ]
    )


def feature_importances(model: ClassifierMixin, names: Sequence[str]) -> pd.DataFrame:
    """Return feature importances sorted descending.

    Uses impurity importances for forests and mean absolute coefficients for
    logistic regression.

    Args:
        model: A fitted estimator from :func:`build_baseline`.
        names: Feature names in column order.

    Returns:
        DataFrame with ``feature`` and ``importance``.
    """
    if isinstance(model, RandomForestClassifier):
        values = model.feature_importances_
    elif isinstance(model, Pipeline):
        values = np.abs(model.named_steps["clf"].coef_).mean(axis=0)
    else:
        raise TypeError(f"unsupported model type {type(model).__name__}")
    table = pd.DataFrame({"feature": list(names), "importance": values})
    return table.sort_values("importance", ascending=False, kind="mergesort").reset_index(drop=True)
