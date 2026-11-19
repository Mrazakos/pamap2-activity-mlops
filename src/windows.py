"""Segmentation into fixed-length windows, labels and train-only normalisation.

The clean data is kept as one continuous ``(rows, channels)`` array ("signal
store"). A window is just a start row; windows never cross a segment boundary
(segments are contiguous, gap-free pieces from the cleaning step). Window
labels are the majority label; windows below the profile's minimum label
purity (mixed windows) are dropped.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.config import Config, SplitConfig, WindowConfig
from src.constants import IMU_RATE_HZ
from src.split import assert_no_overlap, assign_splits, subjects_by_split
from src.utils import get_logger, kv, read_json, write_json

logger = get_logger(__name__)

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]


@dataclass
class SignalStore:
    """Clean data as aligned arrays (row order = recording, then time)."""

    values: FloatArray
    labels: IntArray
    subjects: IntArray
    sessions: npt.NDArray[np.str_]
    segments: IntArray
    channels: list[str]
    sample_rate_hz: float


@dataclass
class SplitArrays:
    """Model-ready windows of one split."""

    x: FloatArray
    y: IntArray
    subjects: IntArray
    index: pd.DataFrame


@dataclass
class PreparedData:
    """Everything a model needs: windows per split plus their provenance."""

    splits: dict[str, SplitArrays]
    channels: list[str]
    class_ids: list[int]
    class_names: list[str]
    norm_mean: FloatArray
    norm_std: FloatArray
    window_length: int
    sample_rate_hz: float
    split_subjects: dict[str, list[int]] = field(default_factory=dict)


def sensor_group(channel: str) -> str:
    """Return the sensor group of a column (``hr`` or an IMU location).

    Args:
        channel: Column name, e.g. ``hand_acc16_x``.

    Returns:
        ``"hr"`` for heart rate, otherwise the IMU location prefix.
    """
    return "hr" if channel == "heart_rate" else channel.split("_", 1)[0]


def select_channels(available: Sequence[str], groups: Sequence[str]) -> list[str]:
    """Keep the channels whose sensor group is selected, in their original order.

    Args:
        available: Channels present in the clean data.
        groups: Selected sensor groups (e.g. ``("hand", "hr")``).

    Returns:
        Selected channel names.

    Raises:
        ValueError: If nothing is selected.
    """
    chosen = [c for c in available if sensor_group(c) in set(groups)]
    if not chosen:
        raise ValueError(f"no channels left for sensor groups {list(groups)}")
    return chosen


def window_geometry(cfg: WindowConfig, rate_hz: float = IMU_RATE_HZ) -> tuple[int, int, float]:
    """Compute window length and step in samples after downsampling.

    Args:
        cfg: Window configuration.
        rate_hz: Original sampling rate.

    Returns:
        ``(length, step, effective_rate_hz)``.
    """
    rate = rate_hz / cfg.downsample
    length = round(cfg.window_seconds * rate)
    step = max(1, round(length * (1.0 - cfg.overlap)))
    if length < 2:
        raise ValueError("window must contain at least 2 samples")
    return length, step, rate


def build_store(
    clean: pd.DataFrame, channels: Sequence[str], class_ids: Sequence[int], downsample: int = 1
) -> SignalStore:
    """Turn the clean frame into aligned arrays, optionally decimating each segment.

    Args:
        clean: Clean data (ordered by recording, segment and time).
        channels: Channels to keep.
        class_ids: Activity IDs in class-index order.
        downsample: Keep every ``downsample``-th row of each segment.

    Returns:
        The signal store.
    """
    if downsample > 1:
        position = clean.groupby("segment_id", sort=False).cumcount().to_numpy()
        clean = clean.loc[position % downsample == 0]
    lookup = {a: i for i, a in enumerate(class_ids)}
    return SignalStore(
        values=clean[list(channels)].to_numpy(dtype=np.float32, copy=True),
        labels=clean["activity_id"].map(lookup).to_numpy(dtype=np.int64),
        subjects=clean["subject_id"].to_numpy(dtype=np.int64),
        sessions=clean["session"].to_numpy(dtype=str),
        segments=clean["segment_id"].to_numpy(dtype=np.int64),
        channels=list(channels),
        sample_rate_hz=IMU_RATE_HZ / downsample,
    )


def segment_bounds(segments: IntArray) -> list[tuple[int, int]]:
    """Return ``[start, end)`` row ranges of consecutive equal segment ids.

    Args:
        segments: Segment id per row.

    Returns:
        List of half-open ranges.
    """
    if len(segments) == 0:
        return []
    cuts = np.flatnonzero(np.diff(segments) != 0) + 1
    starts = np.concatenate([[0], cuts])
    ends = np.concatenate([cuts, [len(segments)]])
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def window_labels(
    labels: IntArray, starts: IntArray, length: int, n_classes: int
) -> tuple[IntArray, npt.NDArray[np.float64]]:
    """Majority label and its share (purity) for each window.

    Uses cumulative one-hot counts, so it is O(rows * classes) instead of
    O(windows * length). Ties go to the lowest class index.

    Args:
        labels: Class index per row of one segment.
        starts: Window starts relative to that segment.
        length: Window length.
        n_classes: Number of classes.

    Returns:
        ``(majority_label, purity)`` per window.
    """
    one_hot = np.zeros((len(labels) + 1, n_classes), dtype=np.int64)
    one_hot[np.arange(1, len(labels) + 1), labels] = 1
    cumulative = one_hot.cumsum(axis=0)
    counts = cumulative[starts + length] - cumulative[starts]
    majority = counts.argmax(axis=1).astype(np.int64)
    purity = counts.max(axis=1) / float(length)
    return majority, purity


def build_window_index(
    store: SignalStore,
    length: int,
    step: int,
    n_classes: int,
    min_purity: float,
    split_of_subject: Mapping[int, str],
) -> pd.DataFrame:
    """List all windows that fit inside a segment and are pure enough.

    Args:
        store: Signal store.
        length: Window length in samples.
        step: Hop between window starts.
        n_classes: Number of classes.
        min_purity: Minimal share of the majority label.
        split_of_subject: Subject -> split.

    Returns:
        DataFrame with ``subject_id, session, segment_id, start, label, purity, split``;
        ``start`` is a row offset into the store.
    """
    columns: dict[str, list[npt.NDArray[np.generic]]] = {
        k: [] for k in ("subject_id", "session", "segment_id", "start", "label", "purity")
    }
    for begin, end in segment_bounds(store.segments):
        if end - begin < length:
            continue
        rel = np.arange(0, end - begin - length + 1, step, dtype=np.int64)
        majority, purity = window_labels(store.labels[begin:end], rel, length, n_classes)
        keep = purity >= min_purity - 1e-12
        if not keep.any():
            continue
        n = int(keep.sum())
        columns["subject_id"].append(np.full(n, store.subjects[begin], dtype=np.int64))
        columns["session"].append(np.full(n, store.sessions[begin]))
        columns["segment_id"].append(np.full(n, store.segments[begin], dtype=np.int64))
        columns["start"].append(rel[keep] + begin)
        columns["label"].append(majority[keep])
        columns["purity"].append(purity[keep])
    if not columns["start"]:
        raise ValueError("no window fits into any segment; check window length and cleaning")
    index = pd.DataFrame({k: np.concatenate(v) for k, v in columns.items()})
    index["session"] = index["session"].astype(str)
    index["split"] = index["subject_id"].map(split_of_subject).astype(str)
    return index


def fit_normalization(values: FloatArray, row_mask: npt.NDArray[np.bool_]) -> tuple[FloatArray, FloatArray]:
    """Per-channel mean and standard deviation over the selected (training) rows.

    Args:
        values: ``(rows, channels)`` signal.
        row_mask: Rows to use (training subjects only).

    Returns:
        ``(mean, std)`` as float32; std is floored at 1e-6.
    """
    if not row_mask.any():
        raise ValueError("normalisation needs at least one training row")
    selected = values[row_mask].astype(np.float64)
    mean = selected.mean(axis=0)
    std = np.maximum(selected.std(axis=0), 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def extract_windows(values: FloatArray, starts: IntArray, length: int) -> FloatArray:
    """Cut windows out of the signal as ``(n, channels, length)`` float32.

    Args:
        values: ``(rows, channels)`` signal.
        starts: Start rows.
        length: Window length.

    Returns:
        Contiguous array of windows.
    """
    view = np.lib.stride_tricks.sliding_window_view(values, length, axis=0)  # (rows-L+1, C, L)
    return np.ascontiguousarray(view[starts], dtype=np.float32)


def even_subsample(n: int, limit: int | None) -> IntArray:
    """Deterministically pick at most ``limit`` evenly spaced indices out of ``n``.

    Args:
        n: Number of items.
        limit: Maximum to keep (``None`` keeps all).

    Returns:
        Sorted indices.
    """
    if limit is None or n <= limit:
        return np.arange(n, dtype=np.int64)
    return np.unique(np.linspace(0, n - 1, limit).round().astype(np.int64))


@dataclass
class WindowPlan:
    """Split assignment, window index and train-only normalisation for one config."""

    store: SignalStore
    split_subjects: dict[str, list[int]]
    index: pd.DataFrame
    mean: FloatArray
    std: FloatArray
    length: int
    step: int


def plan_windows(
    cfg: Config,
    clean: pd.DataFrame,
    available_channels: Sequence[str],
    split: SplitConfig | None = None,
) -> WindowPlan:
    """Assign subjects to splits, index windows and fit normalisation on train only.

    This is the single code path shared by the ``split_window`` stage, training,
    feature extraction and LOSO, so they can never disagree.

    Args:
        cfg: Resolved config (windows, sensor subset, profile, classes).
        clean: Clean data.
        available_channels: Channels present in the clean data.
        split: Split to use (default: ``cfg.split``; LOSO passes its own folds).

    Returns:
        The window plan.
    """
    channels = select_channels(available_channels, cfg.data.sensor_groups)
    length, step, _ = window_geometry(cfg.windows)
    store = build_store(clean, channels, cfg.class_ids, cfg.windows.downsample)
    mapping = assign_splits(np.unique(store.subjects).tolist(), split or cfg.split)
    groups = subjects_by_split(mapping)
    assert_no_overlap(groups)
    index = build_window_index(
        store, length, step, len(cfg.class_ids), cfg.clean_profile.min_label_purity, mapping
    )
    mean, std = fit_normalization(store.values, np.isin(store.subjects, groups["train"]))
    return WindowPlan(store, groups, index, mean, std, length, step)


def prepare_data(
    cfg: Config,
    clean: pd.DataFrame,
    available_channels: Sequence[str],
    split: SplitConfig | None = None,
    splits_needed: Sequence[str] = ("train", "val", "test"),
) -> PreparedData:
    """Build normalised windows per split (normalisation fitted on train only).

    Args:
        cfg: Resolved config.
        clean: Clean data.
        available_channels: Channels present in the clean data.
        split: Split to use (default: ``cfg.split``).
        splits_needed: Which splits to materialise as arrays.

    Returns:
        Prepared data.
    """
    plan = plan_windows(cfg, clean, available_channels, split)
    normalized = (plan.store.values - plan.mean) / plan.std
    n_channels = len(plan.store.channels)

    splits: dict[str, SplitArrays] = {}
    for name in splits_needed:
        part = plan.index.loc[plan.index["split"] == name].reset_index(drop=True)
        keep = even_subsample(len(part), cfg.train.max_windows_per_split)
        part = part.iloc[keep].reset_index(drop=True)
        starts = part["start"].to_numpy(dtype=np.int64)
        x = (
            extract_windows(normalized, starts, plan.length)
            if len(part)
            else np.empty((0, n_channels, plan.length), np.float32)
        )
        splits[name] = SplitArrays(
            x=x,
            # copy=True: pandas 3 (copy-on-write) hands out read-only arrays; torch needs writable ones
            y=part["label"].to_numpy(dtype=np.int64, copy=True),
            subjects=part["subject_id"].to_numpy(dtype=np.int64, copy=True),
            index=part,
        )
        logger.info(
            "windows ready", extra=kv(split=name, windows=len(part), subjects=plan.split_subjects[name])
        )
    return PreparedData(
        splits=splits,
        channels=plan.store.channels,
        class_ids=cfg.class_ids,
        class_names=cfg.class_names,
        norm_mean=plan.mean,
        norm_std=plan.std,
        window_length=plan.length,
        sample_rate_hz=plan.store.sample_rate_hz,
        split_subjects=plan.split_subjects,
    )


def load_clean(clean_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read ``clean.parquet`` and its ``meta.json``.

    Args:
        clean_dir: Folder written by the ``clean`` stage.

    Returns:
        ``(clean_frame, meta)``.
    """
    clean_dir = Path(clean_dir)
    return pd.read_parquet(clean_dir / "clean.parquet"), read_json(clean_dir / "meta.json")


def run_preprocess(cfg: Config) -> dict[str, object]:
    """Pipeline stage ``split_window``: split, normalisation stats and window index.

    Writes ``data/processed/{windows.parquet, norm_stats.json, meta.json}``.
    The window tensors themselves are rebuilt in memory by consumers with the
    same functions (cheap, and avoids storing overlapping copies of the data).

    Args:
        cfg: Resolved config.

    Returns:
        The written metadata.
    """
    from src.schemas import windows_schema
    from src.split import activity_coverage

    clean, meta = load_clean(cfg.paths.clean_dir)
    if meta["data_version"] != cfg.data.version:
        raise ValueError(f"clean data is {meta['data_version']}, config wants {cfg.data.version}")
    plan = plan_windows(cfg, clean, meta["channels"])  # type: ignore[arg-type]
    schema = windows_schema(len(cfg.class_ids), cfg.clean_profile.min_label_purity)
    index = schema.validate(plan.index, lazy=True)

    out = Path(cfg.paths.processed_dir)
    out.mkdir(parents=True, exist_ok=True)
    index.to_parquet(out / "windows.parquet", index=False)
    write_json(
        out / "norm_stats.json",
        {"channels": plan.store.channels, "mean": plan.mean.tolist(), "std": plan.std.tolist()},
    )
    mapping = {s: name for name, subjects in plan.split_subjects.items() for s in subjects}
    counts = index.groupby(["split", "label"]).size().unstack(fill_value=0)
    per_split = {str(k): int(v) for k, v in index["split"].value_counts().sort_index().items()}
    processed_meta: dict[str, object] = {
        "data_version": cfg.data.version,
        "sensor_subset": cfg.data.sensor_subset,
        "split_subjects": plan.split_subjects,
        "missing_activities": activity_coverage(clean, mapping, cfg.dataset.activities),
        "window_length": plan.length,
        "window_step": plan.step,
        "sample_rate_hz": plan.store.sample_rate_hz,
        "windows_per_split": per_split,
        "windows_per_split_and_class": {
            str(s): {cfg.class_names[int(c)]: int(n) for c, n in row.items()} for s, row in counts.iterrows()
        },
    }
    write_json(out / "meta.json", processed_meta)
    logger.info("processed data written", extra=kv(windows=len(index), **per_split))
    return processed_meta
