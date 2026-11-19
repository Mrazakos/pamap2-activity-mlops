"""Unit tests for the data-quality rules (on synthetic recordings)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.clean import (
    assign_segments,
    clean_dataset,
    clean_recording,
    clip_to_sensor_range,
    drop_duplicate_timestamps,
    fill_heart_rate,
    interpolate_short_gaps,
    nan_run_lengths,
    profile_channels,
)
from src.config import CleanProfile, load_config
from src.constants import ACC16_LIMIT, PROTOCOL_ACTIVITIES
from tests.conftest import make_recording

V1 = load_config().clean["v1"]
V2 = load_config().clean["v2"]


def test_nan_run_lengths() -> None:
    s = pd.Series([1.0, np.nan, np.nan, 2.0, np.nan, 3.0])
    assert nan_run_lengths(s).tolist() == [0, 2, 2, 0, 1, 0]


@pytest.mark.parametrize(
    ("values", "max_len", "expected"),
    [
        ([0.0, np.nan, 2.0], 1, [0.0, 1.0, 2.0]),
        (
            [0.0, np.nan, np.nan, 3.0],
            1,
            [0.0, np.nan, np.nan, 3.0],
        ),  # too long: untouched, not partially filled
        ([0.0, np.nan, np.nan, 3.0], 2, [0.0, 1.0, 2.0, 3.0]),
        ([np.nan, 1.0, np.nan], 5, [np.nan, 1.0, np.nan]),  # edges are never extrapolated
        ([np.nan, 1.0, np.nan, 3.0], None, [1.0, 1.0, 2.0, 3.0]),  # v1: fill everything
    ],
)
def test_interpolate_short_gaps(values: list[float], max_len: int | None, expected: list[float]) -> None:
    out = interpolate_short_gaps(pd.Series(values), max_len)
    np.testing.assert_allclose(out.to_numpy(), np.array(expected), equal_nan=True)


def test_heart_rate_forward_fill_respects_limit() -> None:
    s = pd.Series([70.0, np.nan, np.nan, np.nan, 80.0])
    out = fill_heart_rate(s, limit=2)
    np.testing.assert_allclose(out.to_numpy(), [70, 70, 70, np.nan, 80], equal_nan=True)


def test_drop_duplicate_timestamps_keeps_first() -> None:
    df = pd.DataFrame({"timestamp": [0.0, 0.01, 0.01, 0.02], "v": [1, 2, 3, 4]})
    out, dropped = drop_duplicate_timestamps(df)
    assert dropped == 1
    assert out["v"].tolist() == [1, 2, 4]


def test_clip_to_sensor_range_counts() -> None:
    df = pd.DataFrame(
        {"hand_acc16_x": [0.0, 2 * ACC16_LIMIT, -2 * ACC16_LIMIT], "heart_rate": [10.0, 80.0, 300.0]}
    )
    out, clipped = clip_to_sensor_range(df, ["hand_acc16_x", "heart_rate"])
    assert clipped == 4
    assert out["hand_acc16_x"].abs().max() == pytest.approx(ACC16_LIMIT)


def test_assign_segments_on_gap_and_label_change() -> None:
    df = pd.DataFrame({"timestamp": [0.0, 0.01, 0.02, 1.0, 1.01, 1.02], "activity_id": [1, 1, 2, 2, 2, 2]})
    assert assign_segments(df, 0.05, True).tolist() == [0, 0, 1, 2, 2, 2]
    assert assign_segments(df, None, False).tolist() == [0] * 6


def test_profile_channels() -> None:
    assert "hand_orient_0" in profile_channels(V1)
    motion = profile_channels(V2)
    assert "heart_rate" in motion
    assert not any("acc6" in c or "orient" in c or "temp" in c for c in motion)
    assert len(motion) == 1 + 3 * 9


def test_clean_recording_v2_fixes_all_defects() -> None:
    clean, counts = clean_recording(make_recording(101), V2, PROTOCOL_ACTIVITIES)
    assert counts["duplicate_timestamps_dropped"] == 1
    assert counts["transient_rows_dropped"] > 0
    assert counts["imu_values_interpolated"] >= 3
    assert counts["rows_with_long_nan_gaps_dropped"] >= 50
    assert not clean[profile_channels(V2)].isna().any().any()
    assert 0 not in set(clean["activity_id"])
    # one label per segment, strictly increasing time inside each segment
    assert (clean.groupby("segment_id")["activity_id"].nunique() == 1).all()
    assert (clean.groupby("segment_id")["timestamp"].diff().dropna() > 0).all()


def test_clean_recording_v1_is_naive() -> None:
    clean, counts = clean_recording(make_recording(101), V1, PROTOCOL_ACTIVITIES)
    assert clean["segment_id"].nunique() == 1  # not gap-aware
    assert counts["rows_with_long_nan_gaps_dropped"] == 0  # everything interpolated
    assert not clean.isna().any().any()


def test_clean_dataset_makes_segment_ids_globally_unique() -> None:
    recordings = [make_recording(101), make_recording(102)]
    clean, report = clean_dataset(recordings, V2, PROTOCOL_ACTIVITIES)
    per_subject = clean.groupby("subject_id")["segment_id"].agg(["min", "max"])
    assert per_subject.loc[101, "max"] < per_subject.loc[102, "min"]
    assert report["totals"]["rows_out"] == len(clean)  # type: ignore[index]


def test_clean_profile_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        CleanProfile(min_label_purity=1.5)
