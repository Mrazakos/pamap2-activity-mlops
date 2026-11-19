"""Subject-wise train/validation/test assignment (the leakage-prevention core).

All windows of one person end up in exactly one split. The validation and test
subjects are fixed in ``configs/params.yaml`` before any training; every other
subject is used for training.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import pandas as pd

from src.config import SplitConfig


class LeakageError(ValueError):
    """Raised when a subject would appear in more than one split."""


def assign_splits(subjects: Iterable[int], split: SplitConfig) -> dict[int, str]:
    """Map every subject to ``train``, ``val`` or ``test``.

    Args:
        subjects: Subjects present in the data.
        split: Fixed validation/test subjects.

    Returns:
        ``{subject_id: split_name}`` in sorted subject order.

    Raises:
        LeakageError: If a subject is listed in both val and test.
    """
    val, test = set(split.val_subjects), set(split.test_subjects)
    if val & test:
        raise LeakageError(f"subjects in both val and test: {sorted(val & test)}")
    mapping: dict[int, str] = {}
    for subject in sorted(set(subjects)):
        mapping[subject] = "test" if subject in test else "val" if subject in val else "train"
    return mapping


def subjects_by_split(mapping: Mapping[int, str]) -> dict[str, list[int]]:
    """Invert a subject->split mapping.

    Args:
        mapping: Output of :func:`assign_splits`.

    Returns:
        ``{"train": [...], "val": [...], "test": [...]}`` with sorted subjects.
    """
    result: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for subject, name in sorted(mapping.items()):
        result[name].append(subject)
    return result


def assert_no_overlap(groups: Mapping[str, Sequence[int]]) -> None:
    """Fail if any subject appears in two groups.

    Args:
        groups: Split name -> subjects.

    Raises:
        LeakageError: On overlap.
    """
    seen: dict[int, str] = {}
    for name, subjects in groups.items():
        for subject in subjects:
            if subject in seen and seen[subject] != name:
                raise LeakageError(f"subject {subject} is in both {seen[subject]} and {name}")
            seen[subject] = name


def activity_coverage(
    df: pd.DataFrame, mapping: Mapping[int, str], activities: Sequence[int]
) -> dict[str, list[int]]:
    """List target activities that are *missing* from each split.

    Used to justify the choice of validation/test subjects (they should cover
    every activity, otherwise per-class metrics are undefined).

    Args:
        df: Frame with ``subject_id`` and ``activity_id``.
        mapping: Subject -> split.
        activities: Target activity IDs.

    Returns:
        Split name -> sorted missing activity IDs.
    """
    split_col = df["subject_id"].map(mapping)
    missing: dict[str, list[int]] = {}
    for name in ("train", "val", "test"):
        present = set(df.loc[split_col == name, "activity_id"].unique().tolist())
        missing[name] = sorted(set(activities) - present)
    return missing


def rank_evaluation_candidates(
    seconds: Mapping[int, Mapping[int, float]], activities: Sequence[int], min_seconds: float
) -> list[tuple[int, float]]:
    """Rank subjects by how well they can serve as validation/test subjects.

    The rule used to fix the split before training: a subject is eligible only
    if it performed **every** target activity for at least ``min_seconds``
    (otherwise per-class metrics on that subject are undefined or rest on a
    handful of windows). Eligible subjects are ranked by their *smallest*
    per-activity duration, largest first, so the chosen subjects give every
    class as many evaluation windows as possible. Ties break on subject id.

    Args:
        seconds: ``{subject: {activity_id: seconds of data}}``.
        activities: Target activity IDs.
        min_seconds: Minimum duration per activity to be eligible.

    Returns:
        ``(subject, smallest per-activity seconds)`` for eligible subjects, best first.
    """
    ranked = []
    for subject, per_activity in seconds.items():
        smallest = min(float(per_activity.get(a, 0.0)) for a in activities)
        if smallest >= min_seconds:
            ranked.append((int(subject), smallest))
    return sorted(ranked, key=lambda item: (-item[1], item[0]))


def loso_folds(subjects: Sequence[int]) -> list[SplitConfig]:
    """Leave-one-subject-out folds.

    Fold *i* tests on subject *i* and validates (early stopping) on the next
    subject in sorted cyclic order, so every fold is deterministic.

    Args:
        subjects: Subjects to rotate through.

    Returns:
        One :class:`SplitConfig` per subject.
    """
    ordered = sorted(set(subjects))
    if len(ordered) < 3:
        raise ValueError("LOSO needs at least 3 subjects")
    return [
        SplitConfig(val_subjects=[ordered[(i + 1) % len(ordered)]], test_subjects=[s])
        for i, s in enumerate(ordered)
    ]
