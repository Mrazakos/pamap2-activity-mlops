"""Pandera schema tests: valid frames pass, each kind of invalid frame fails."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pandera.errors as pa_errors
import pytest

from src.clean import clean_recording, profile_channels
from src.config import load_config
from src.constants import ACC16_LIMIT, PROTOCOL_ACTIVITIES
from src.schemas import (
    RAW_SCHEMA,
    check_window_array,
    clean_schema,
    features_schema,
    validate_raw,
    windows_schema,
)
from tests.conftest import make_recording

V2 = load_config().clean["v2"]
CHANNELS = profile_channels(V2)
SCHEMA_ERRORS = (pa_errors.SchemaError, pa_errors.SchemaErrors)


def _clean_schema() -> object:
    """The strict v2 clean schema."""
    return clean_schema(
        CHANNELS, PROTOCOL_ACTIVITIES, check_ranges=True, strict_time=True, single_label_segments=True
    )


@pytest.fixture
def clean_frame() -> pd.DataFrame:
    """A valid v2 clean frame built from a synthetic recording."""
    clean, _ = clean_recording(make_recording(101), V2, PROTOCOL_ACTIVITIES)
    return clean


def _windows() -> pd.DataFrame:
    """A valid window index."""
    return pd.DataFrame(
        {
            "subject_id": [101, 101, 105, 106],
            "session": ["protocol"] * 4,
            "segment_id": [0, 0, 1, 2],
            "start": [0, 64, 0, 0],
            "label": [0, 1, 2, 3],
            "purity": [1.0, 1.0, 1.0, 1.0],
            "split": ["train", "train", "val", "test"],
        }
    )


def test_raw_schema_accepts_synthetic_recording() -> None:
    validate_raw(make_recording(101, defects=True))


def test_clean_schema_accepts_clean_frame(clean_frame: pd.DataFrame) -> None:
    _clean_schema().validate(clean_frame)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param(
            lambda df: df.assign(activity_id=df["activity_id"].where(df.index != 0, 99)),
            id="unknown-activity",
        ),
        pytest.param(
            lambda df: df.assign(activity_id=df["activity_id"].where(df.index != 0, 0)), id="transient-left"
        ),
        pytest.param(
            lambda df: df.assign(hand_acc16_x=df["hand_acc16_x"].where(df.index != 5, np.nan)),
            id="nan-after-cleaning",
        ),
        pytest.param(
            lambda df: df.assign(timestamp=df["timestamp"].where(df.index != 10, 0.0)),
            id="non-monotonic-time",
        ),
        pytest.param(
            lambda df: df.assign(
                hand_acc16_x=df["hand_acc16_x"].where(df.index != 3, np.float32(2 * ACC16_LIMIT))
            ),
            id="out-of-range-acc",
        ),
        pytest.param(lambda df: df.assign(heart_rate=df["heart_rate"].astype("float64")), id="wrong-dtype"),
        pytest.param(lambda df: df.drop(columns="chest_gyro_x"), id="missing-column"),
        pytest.param(lambda df: df.assign(extra=1.0), id="unexpected-column"),
        pytest.param(
            lambda df: df.assign(subject_id=df["subject_id"].where(df.index != 0, 999)), id="unknown-subject"
        ),
    ],
)
def test_clean_schema_rejects_invalid(
    clean_frame: pd.DataFrame, corrupt: Callable[[pd.DataFrame], pd.DataFrame]
) -> None:
    with pytest.raises(SCHEMA_ERRORS):
        _clean_schema().validate(corrupt(clean_frame), lazy=True)  # type: ignore[attr-defined]


def test_clean_schema_rejects_mixed_segment(clean_frame: pd.DataFrame) -> None:
    broken = clean_frame.assign(segment_id=0)  # every activity in one segment
    with pytest.raises(SCHEMA_ERRORS):
        _clean_schema().validate(broken, lazy=True)  # type: ignore[attr-defined]


def test_raw_schema_rejects_missing_column() -> None:
    with pytest.raises(SCHEMA_ERRORS):
        RAW_SCHEMA.validate(make_recording(101).drop(columns="heart_rate"), lazy=True)


def test_windows_schema_valid_and_subject_overlap() -> None:
    schema = windows_schema(n_classes=12, min_purity=1.0)
    schema.validate(_windows())
    leaked = _windows()
    leaked.loc[2, "subject_id"] = 101  # subject 101 now in train and val
    with pytest.raises(SCHEMA_ERRORS, match="leakage"):
        schema.validate(leaked, lazy=True)


@pytest.mark.parametrize(
    ("column", "value"),
    [("label", 12), ("purity", 0.5), ("split", "holdout"), ("start", -1)],
)
def test_windows_schema_rejects_bad_values(column: str, value: object) -> None:
    bad = _windows()
    bad.loc[0, column] = value
    with pytest.raises(SCHEMA_ERRORS):
        windows_schema(n_classes=12, min_purity=1.0).validate(bad, lazy=True)


def test_features_schema() -> None:
    table = pd.DataFrame(
        {
            "subject_id": [101, 106],
            "label": [0, 1],
            "split": ["train", "test"],
            "f__a": np.array([0.1, 0.2], np.float32),
            "f__b": np.array([1.0, 2.0], np.float32),
        }
    )
    features_schema().validate(table)
    table.loc[0, "f__a"] = np.inf
    with pytest.raises(SCHEMA_ERRORS):
        features_schema().validate(table, lazy=True)


def test_check_window_array() -> None:
    check_window_array(np.zeros((2, 3, 8), np.float32), 3, 8)
    with pytest.raises(ValueError, match="shape"):
        check_window_array(np.zeros((2, 8, 3), np.float32), 3, 8)
    with pytest.raises(ValueError, match="float32"):
        check_window_array(np.zeros((2, 3, 8)), 3, 8)
    bad = np.zeros((2, 3, 8), np.float32)
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        check_window_array(bad, 3, 8)
