"""Unit tests for windowing, labels and train-only normalisation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.clean import clean_dataset, profile_channels
from src.config import Config, WindowConfig
from src.constants import PROTOCOL_ACTIVITIES
from src.windows import (
    build_store,
    build_window_index,
    even_subsample,
    extract_windows,
    fit_normalization,
    plan_windows,
    prepare_data,
    segment_bounds,
    select_channels,
    window_geometry,
    window_labels,
)
from tests.conftest import make_recording


def _clean(cfg: Config, subjects: tuple[int, ...] = (101, 102, 103, 104)) -> pd.DataFrame:
    """Synthetic clean data for the given subjects."""
    clean, _ = clean_dataset([make_recording(s) for s in subjects], cfg.clean_profile, cfg.dataset.activities)
    return clean


def test_window_geometry() -> None:
    assert window_geometry(WindowConfig(window_seconds=5.12, overlap=0.5)) == (512, 256, 100.0)
    assert window_geometry(WindowConfig(window_seconds=5.12, overlap=0.5, downsample=2)) == (256, 128, 50.0)


def test_segment_bounds() -> None:
    assert segment_bounds(np.array([0, 0, 1, 1, 1, 2])) == [(0, 2), (2, 5), (5, 6)]
    assert segment_bounds(np.array([], dtype=np.int64)) == []


def test_window_labels_majority_and_purity() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    majority, purity = window_labels(labels, np.array([0, 2, 4]), 4, n_classes=2)
    assert majority.tolist() == [0, 1, 1]
    np.testing.assert_allclose(purity, [0.75, 0.75, 1.0])


def test_windows_never_cross_segments(tiny_cfg: Config) -> None:
    clean = _clean(tiny_cfg)
    plan = plan_windows(tiny_cfg, clean, profile_channels(tiny_cfg.clean_profile))
    seg = plan.store.segments
    starts = plan.index["start"].to_numpy()
    assert (seg[starts] == seg[starts + plan.length - 1]).all()
    assert (plan.index["purity"] == 1.0).all()


def test_mixed_windows_dropped_only_with_purity_threshold() -> None:
    store_df = pd.DataFrame(
        {
            "subject_id": 101,
            "session": "protocol",
            "segment_id": 0,
            "activity_id": [1] * 6 + [2] * 6,
            "a": np.arange(12, dtype=np.float32),
        }
    )
    store = build_store(store_df, ["a"], [1, 2])
    strict = build_window_index(store, 4, 2, 2, 1.0, {101: "train"})
    loose = build_window_index(store, 4, 2, 2, 0.0, {101: "train"})
    assert len(loose) == 5
    assert len(strict) == 4  # the window straddling the label change is dropped
    assert set(loose["label"]) == {0, 1}


def test_normalization_is_fit_on_train_only(tiny_cfg: Config) -> None:
    clean = _clean(tiny_cfg)
    channels = profile_channels(tiny_cfg.clean_profile)
    before = plan_windows(tiny_cfg, clean, channels)
    shifted = clean.copy()
    test_rows = shifted["subject_id"].isin(tiny_cfg.split.test_subjects + tiny_cfg.split.val_subjects)
    shifted.loc[test_rows, channels] += 1000.0  # changing val/test data must not move the stats
    after = plan_windows(tiny_cfg, shifted, channels)
    np.testing.assert_array_equal(before.mean, after.mean)
    np.testing.assert_array_equal(before.std, after.std)


def test_fit_normalization_requires_rows() -> None:
    with pytest.raises(ValueError):
        fit_normalization(np.zeros((3, 2), np.float32), np.zeros(3, bool))


def test_extract_windows_shape_and_content() -> None:
    values = np.arange(20, dtype=np.float32).reshape(10, 2)
    out = extract_windows(values, np.array([0, 3]), 4)
    assert out.shape == (2, 2, 4)
    np.testing.assert_array_equal(out[1, 0], [6, 8, 10, 12])


def test_prepare_data_shapes_and_disjoint_subjects(tiny_cfg: Config) -> None:
    data = prepare_data(tiny_cfg, _clean(tiny_cfg), profile_channels(tiny_cfg.clean_profile))
    for name, arrays in data.splits.items():
        assert arrays.x.shape[1:] == (len(data.channels), data.window_length), name
        assert arrays.x.dtype == np.float32
        assert len(arrays.x) == len(arrays.y) == len(arrays.subjects)
    subjects = {n: set(a.subjects.tolist()) for n, a in data.splits.items()}
    assert not subjects["train"] & subjects["val"]
    assert not subjects["train"] & subjects["test"]
    assert subjects["test"] == {104}


def test_sensor_subsets_select_channels() -> None:
    channels = ["heart_rate", "hand_acc16_x", "chest_acc16_x", "ankle_acc16_x"]
    assert select_channels(channels, ("hand", "hr")) == ["heart_rate", "hand_acc16_x"]
    assert select_channels(channels, ("hand",)) == ["hand_acc16_x"]
    with pytest.raises(ValueError):
        select_channels(["heart_rate"], ("hand",))


def test_even_subsample_is_deterministic() -> None:
    assert even_subsample(10, None).tolist() == list(range(10))
    a, b = even_subsample(1000, 7), even_subsample(1000, 7)
    assert a.tolist() == b.tolist() and len(a) == 7 and a[0] == 0 and a[-1] == 999


def test_protocol_activity_ids_are_the_class_list(tiny_cfg: Config) -> None:
    assert tiny_cfg.class_ids == sorted(PROTOCOL_ACTIVITIES)
