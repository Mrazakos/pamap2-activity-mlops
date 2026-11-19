"""Unit tests for the subject-wise split (leakage prevention)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import SplitConfig
from src.split import (
    LeakageError,
    activity_coverage,
    assert_no_overlap,
    assign_splits,
    loso_folds,
    subjects_by_split,
)

SPLIT = SplitConfig(val_subjects=[105], test_subjects=[106])


def test_assign_splits_is_deterministic_and_complete() -> None:
    subjects = [109, 101, 106, 105, 102]
    first, second = assign_splits(subjects, SPLIT), assign_splits(list(reversed(subjects)), SPLIT)
    assert first == second
    assert subjects_by_split(first) == {"train": [101, 102, 109], "val": [105], "test": [106]}


def test_no_subject_overlap_between_splits() -> None:
    groups = subjects_by_split(assign_splits(range(101, 110), SPLIT))
    seen = [s for subjects in groups.values() for s in subjects]
    assert len(seen) == len(set(seen)) == 9


def test_assert_no_overlap_detects_leak() -> None:
    with pytest.raises(LeakageError, match="102"):
        assert_no_overlap({"train": [101, 102], "test": [102]})


def test_activity_coverage_reports_missing() -> None:
    df = pd.DataFrame({"subject_id": [101, 105, 106], "activity_id": [1, 1, 2]})
    missing = activity_coverage(df, assign_splits([101, 105, 106], SPLIT), [1, 2])
    assert missing == {"train": [2], "val": [2], "test": [1]}


def test_loso_folds_rotate_every_subject() -> None:
    folds = loso_folds([103, 101, 102])
    assert [f.test_subjects for f in folds] == [[101], [102], [103]]
    assert [f.val_subjects for f in folds] == [[102], [103], [101]]
    with pytest.raises(ValueError):
        loso_folds([101, 102])
