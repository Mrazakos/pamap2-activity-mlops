"""Pandera schemas used as quality gates between pipeline stages.

* ``raw``      - converted ``.dat`` files (structure only; NaNs allowed).
* ``clean``    - after the data-quality rules (no NaN, known labels, ordered time).
* ``windows``  - the window index (one row per window) incl. split leakage checks.
* ``features`` - hand-crafted feature table (finite values only).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
import pandera.pandas as pa

from src.constants import (
    ACC16_LIMIT,
    ACTIVITY_NAMES,
    GYRO_LIMIT,
    HR_RANGE,
    RAW_SENSOR_COLUMNS,
    SESSIONS,
    SUBJECT_IDS,
)

SPLITS: tuple[str, ...] = ("train", "val", "test")
FEATURE_PREFIX = "f__"


def _increasing_per_recording(strict: bool) -> pa.Check:
    """Check that timestamps increase within every (subject, session) recording."""

    def check(df: pd.DataFrame) -> bool:
        diffs = df.groupby(["subject_id", "session"], sort=False)["timestamp"].diff().dropna()
        return bool((diffs > 0).all() if strict else (diffs >= 0).all())

    kind = "strictly increasing" if strict else "non-decreasing"
    return pa.Check(check, error=f"timestamps must be {kind} per subject/session")


def _no_subject_in_two_splits() -> pa.Check:
    """Check that every subject belongs to exactly one split (leakage guard)."""

    def check(df: pd.DataFrame) -> bool:
        return bool((df.groupby("subject_id")["split"].nunique() <= 1).all())

    return pa.Check(check, error="a subject appears in more than one split (leakage)")


def _meta_columns() -> dict[str, pa.Column]:
    """Columns that identify the source recording."""
    return {
        "subject_id": pa.Column("int64", pa.Check.isin(list(SUBJECT_IDS))),
        "session": pa.Column(str, pa.Check.isin(list(SESSIONS))),
    }


RAW_SCHEMA = pa.DataFrameSchema(
    {
        **_meta_columns(),
        "timestamp": pa.Column("float64", pa.Check.ge(0), nullable=False),
        "activity_id": pa.Column("int64", pa.Check.isin(sorted(ACTIVITY_NAMES))),
        **{col: pa.Column("float64", nullable=True) for col in RAW_SENSOR_COLUMNS},
    },
    strict=True,
    name="raw",
)


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a converted raw recording.

    Args:
        df: Raw frame.

    Returns:
        The same frame (validated).
    """
    return RAW_SCHEMA.validate(df, lazy=True)


def clean_schema(
    channels: Sequence[str],
    activities: Sequence[int],
    check_ranges: bool,
    strict_time: bool,
    single_label_segments: bool,
) -> pa.DataFrameSchema:
    """Build the schema for cleaned data.

    Args:
        channels: Sensor columns kept by the cleaning profile.
        activities: Allowed activity IDs.
        check_ranges: Enforce physical sensor ranges (after clipping).
        strict_time: Require strictly increasing timestamps (no duplicates).
        single_label_segments: Require one activity per segment.

    Returns:
        The schema.
    """

    def channel_checks(col: str) -> list[pa.Check]:
        checks = [pa.Check(lambda s: np.isfinite(s).all(), error="non-finite value")]
        if not check_ranges:
            return checks
        if "_acc16_" in col:
            checks.append(pa.Check.in_range(-ACC16_LIMIT, ACC16_LIMIT))
        elif "_gyro_" in col:
            checks.append(pa.Check.in_range(-GYRO_LIMIT, GYRO_LIMIT))
        elif col == "heart_rate":
            checks.append(pa.Check.in_range(*HR_RANGE))
        return checks

    frame_checks = [_increasing_per_recording(strict_time)]
    if single_label_segments:
        frame_checks.append(
            pa.Check(
                lambda df: bool(
                    (df.groupby(["subject_id", "session", "segment_id"])["activity_id"].nunique() == 1).all()
                ),
                error="segment contains more than one activity",
            )
        )
    return pa.DataFrameSchema(
        {
            **_meta_columns(),
            "timestamp": pa.Column("float64", nullable=False),
            "activity_id": pa.Column("int64", pa.Check.isin(list(activities))),
            "segment_id": pa.Column("int64", pa.Check.ge(0)),
            **{col: pa.Column("float32", channel_checks(col), nullable=False) for col in channels},
        },
        checks=frame_checks,
        strict=True,
        name="clean",
    )


def windows_schema(n_classes: int, min_purity: float = 0.0) -> pa.DataFrameSchema:
    """Build the schema for the window index.

    Args:
        n_classes: Number of target classes.
        min_purity: Minimal allowed label purity of a window.

    Returns:
        The schema.
    """
    return pa.DataFrameSchema(
        {
            **_meta_columns(),
            "segment_id": pa.Column("int64", pa.Check.ge(0)),
            "start": pa.Column("int64", pa.Check.ge(0)),
            "label": pa.Column("int64", pa.Check.in_range(0, n_classes - 1)),
            "purity": pa.Column("float64", pa.Check.in_range(min_purity, 1.0)),
            "split": pa.Column(str, pa.Check.isin(list(SPLITS))),
        },
        checks=[_no_subject_in_two_splits()],
        strict=True,
        name="windows",
    )


def features_schema() -> pa.DataFrameSchema:
    """Build the schema for the feature table (feature columns start with ``f__``).

    Returns:
        The schema.
    """
    return pa.DataFrameSchema(
        {
            "subject_id": pa.Column("int64", pa.Check.isin(list(SUBJECT_IDS))),
            "label": pa.Column("int64", pa.Check.ge(0)),
            "split": pa.Column(str, pa.Check.isin(list(SPLITS))),
            f"^{FEATURE_PREFIX}.*": pa.Column(
                "float32",
                pa.Check(lambda s: np.isfinite(s).all(), error="non-finite feature"),
                regex=True,
            ),
        },
        checks=[_no_subject_in_two_splits()],
        strict=True,
        name="features",
    )


def check_window_array(x: npt.NDArray[np.float32], n_channels: int, length: int) -> None:
    """Validate a window tensor of shape ``(n_windows, n_channels, length)``.

    Args:
        x: Array to check.
        n_channels: Expected channel count.
        length: Expected window length in samples.

    Raises:
        ValueError: On wrong rank, shape, dtype or non-finite values.
    """
    if x.ndim != 3 or x.shape[1:] != (n_channels, length):
        raise ValueError(f"windows must have shape (n, {n_channels}, {length}), got {x.shape}")
    if x.dtype != np.float32:
        raise ValueError(f"windows must be float32, got {x.dtype}")
    if not np.isfinite(x).all():
        raise ValueError("windows contain non-finite values")
