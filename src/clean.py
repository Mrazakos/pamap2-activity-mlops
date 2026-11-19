"""Data-quality rules. Each rule is a small pure function that reports what it changed.

Rules are applied per recording (one subject + session = one continuous file) in
this order, so that filling uses only time-contiguous neighbours:

1. sort by time, drop duplicated timestamps
2. keep the profile's channels (``all`` or ``motion``)
3. heart rate: forward-fill (causal) with a limit, ~9 Hz -> 100 Hz
4. IMU channels: linear interpolation of *short* NaN bursts only
5. clip spikes to the physical sensor range
6. drop transient (0) and non-target activities, then rows still containing NaN
7. assign ``segment_id``: a new segment at each time gap / label change

The ``v1`` profile disables most of this (interpolate everything, no gap
awareness) and serves as the naive data version.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.config import CleanProfile, Config
from src.constants import (
    ACC6_LIMIT,
    ACC16_LIMIT,
    GYRO_LIMIT,
    HR_RANGE,
    IMU_LOCATIONS,
    RAW_SENSOR_COLUMNS,
)
from src.utils import get_logger, kv, write_json

logger = get_logger(__name__)

RECORDING_KEYS: list[str] = ["subject_id", "session"]


def profile_channels(profile: CleanProfile) -> list[str]:
    """Return the sensor columns kept by a cleaning profile.

    ``motion`` keeps the +-16 g accelerometer, gyroscope and magnetometer of each IMU
    plus heart rate; it drops the saturating +-6 g accelerometer, the invalid
    orientation columns and the slowly drifting temperature.

    Args:
        profile: Cleaning profile.

    Returns:
        Ordered list of column names.
    """
    if profile.channels == "all":
        return list(RAW_SENSOR_COLUMNS)
    motion_fields = ("acc16_x", "acc16_y", "acc16_z", "gyro_x", "gyro_y", "gyro_z", "mag_x", "mag_y", "mag_z")
    return ["heart_rate"] + [f"{loc}_{f}" for loc in IMU_LOCATIONS for f in motion_fields]


def drop_duplicate_timestamps(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Sort a recording by time and keep the first row of each duplicated timestamp.

    Args:
        df: One recording.

    Returns:
        The de-duplicated frame and the number of dropped rows.
    """
    df = df.sort_values("timestamp", kind="mergesort")
    dup = df.duplicated(subset="timestamp", keep="first")
    return df.loc[~dup].reset_index(drop=True), int(dup.sum())


def nan_run_lengths(series: pd.Series) -> npt.NDArray[np.int64]:
    """Return, for every element, the length of the NaN run it belongs to (0 if not NaN).

    Args:
        series: Input series.

    Returns:
        Integer array of the same length.
    """
    isna = series.isna().to_numpy()
    if not isna.any():
        return np.zeros(len(series), dtype=np.int64)
    # Label consecutive runs, then count members per run.
    change = np.diff(isna.astype(np.int8), prepend=0) != 0
    run_id = np.cumsum(change)
    counts = np.bincount(run_id)
    lengths = counts[run_id]
    return np.where(isna, lengths, 0).astype(np.int64)


def interpolate_short_gaps(series: pd.Series, max_len: int | None) -> pd.Series:
    """Linearly interpolate NaN runs of at most ``max_len`` samples.

    Unlike ``Series.interpolate(limit=...)``, a longer run stays *completely*
    NaN instead of being partially filled. With ``max_len=None`` every NaN is
    filled, including leading/trailing ones (naive ``v1`` behaviour).

    Args:
        series: Input series (index order = time order).
        max_len: Maximum NaN-run length to fill, or ``None`` for all.

    Returns:
        The interpolated series.
    """
    if max_len is None:
        return series.interpolate(method="linear", limit_direction="both")
    filled = series.interpolate(method="linear", limit_area="inside")
    too_long = nan_run_lengths(series) > max_len
    filled[too_long] = np.nan
    return filled


def fill_heart_rate(series: pd.Series, limit: int | None) -> pd.Series:
    """Upsample the ~9 Hz heart rate to every row.

    With a limit, the last measurement is carried forward for at most ``limit``
    rows (causal, no look-ahead). Without a limit, it is interpolated like
    every other channel (``v1``).

    Args:
        series: Heart-rate column of one recording.
        limit: Maximum number of rows to forward-fill, or ``None``.

    Returns:
        The filled series.
    """
    if limit is None:
        return interpolate_short_gaps(series, None)
    return series.ffill(limit=limit)


def clip_to_sensor_range(df: pd.DataFrame, channels: Sequence[str]) -> tuple[pd.DataFrame, int]:
    """Clip values to the physical range of their sensor.

    Args:
        df: Frame to clip (modified copy is returned).
        channels: Columns to consider.

    Returns:
        The clipped frame and the number of clipped values.
    """
    df = df.copy()
    clipped = 0
    for col in channels:
        if "_acc16_" in col:
            low, high = -ACC16_LIMIT, ACC16_LIMIT
        elif "_acc6_" in col:
            low, high = -ACC6_LIMIT, ACC6_LIMIT
        elif "_gyro_" in col:
            low, high = -GYRO_LIMIT, GYRO_LIMIT
        elif col == "heart_rate":
            low, high = HR_RANGE
        else:
            continue
        values = df[col]
        clipped += int(((values < low) | (values > high)).sum())
        df[col] = values.clip(low, high)
    return df, clipped


def assign_segments(
    df: pd.DataFrame, max_gap_seconds: float | None, split_on_label_change: bool
) -> pd.Series:
    """Number contiguous segments inside one recording.

    A new segment starts at the first row, after a timestamp step larger than
    ``max_gap_seconds`` (if set) and at every label change (if enabled).

    Args:
        df: One recording, sorted by time, after rows were dropped.
        max_gap_seconds: Largest step that still counts as contiguous, or ``None``.
        split_on_label_change: Start a new segment when ``activity_id`` changes.

    Returns:
        Integer segment ids (0, 1, 2, ...) aligned with ``df``.
    """
    boundary = np.zeros(len(df), dtype=bool)
    if len(df):
        boundary[0] = True
    if max_gap_seconds is not None:
        boundary |= df["timestamp"].diff().to_numpy() > max_gap_seconds
    if split_on_label_change:
        boundary |= df["activity_id"].diff().fillna(0).to_numpy() != 0
    return pd.Series(np.cumsum(boundary) - 1, index=df.index, dtype="int64")


def clean_recording(
    df: pd.DataFrame, profile: CleanProfile, activities: Sequence[int]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply all quality rules of a profile to one recording.

    Args:
        df: One raw recording (single subject and session).
        profile: Cleaning profile.
        activities: Target activity IDs; all other rows are dropped.

    Returns:
        The cleaned frame (with ``segment_id``) and per-rule counts.
    """
    counts: dict[str, int] = {"rows_in": len(df)}
    channels = profile_channels(profile)

    if profile.drop_duplicates:
        df, counts["duplicate_timestamps_dropped"] = drop_duplicate_timestamps(df)
    else:
        df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    df = df[[*RECORDING_KEYS, "timestamp", "activity_id", *channels]].copy()
    counts["nan_values_before"] = int(df[channels].isna().sum().sum())

    hr_before = int(df["heart_rate"].isna().sum())
    df["heart_rate"] = fill_heart_rate(df["heart_rate"], profile.hr_ffill_limit)
    counts["heart_rate_filled"] = hr_before - int(df["heart_rate"].isna().sum())

    imu_cols = [c for c in channels if c != "heart_rate"]
    imu_before = int(df[imu_cols].isna().sum().sum())
    for col in imu_cols:
        df[col] = interpolate_short_gaps(df[col], profile.interpolate_limit)
    counts["imu_values_interpolated"] = imu_before - int(df[imu_cols].isna().sum().sum())

    if profile.clip_to_sensor_range:
        df, counts["values_clipped"] = clip_to_sensor_range(df, channels)

    if profile.drop_transient:
        transient = df["activity_id"] == 0
        counts["transient_rows_dropped"] = int(transient.sum())
        df = df.loc[~transient]
    other = ~df["activity_id"].isin(activities)
    counts["non_target_rows_dropped"] = int(other.sum())
    df = df.loc[~other]

    still_nan = df[channels].isna().any(axis=1)
    counts["rows_with_long_nan_gaps_dropped"] = int(still_nan.sum())
    df = df.loc[~still_nan].reset_index(drop=True)

    df["segment_id"] = assign_segments(df, profile.max_gap_seconds, profile.split_on_label_change)
    df[channels] = df[channels].astype("float32")
    counts["segments"] = int(df["segment_id"].nunique())
    counts["rows_out"] = len(df)
    return df, counts


def acc6_saturation_count(df: pd.DataFrame, tolerance: float = 0.98) -> int:
    """Count +-6 g accelerometer values at (or near) the sensor's saturation limit.

    Args:
        df: Raw recording.
        tolerance: Fraction of the limit that counts as saturated.

    Returns:
        Number of saturated values over all ``*_acc6_*`` columns.
    """
    cols = [c for c in df.columns if "_acc6_" in c]
    return int((df[cols].abs() >= tolerance * ACC6_LIMIT).sum().sum())


def load_raw(raw_dir: Path, sessions: Sequence[str]) -> list[pd.DataFrame]:
    """Load the raw Parquet recordings of the selected sessions, in sorted order.

    Args:
        raw_dir: ``data/raw``.
        sessions: Sessions to load.

    Returns:
        One frame per recording.
    """
    files = sorted(p for s in sorted(sessions) for p in (Path(raw_dir) / s).glob("subject*.parquet"))
    if not files:
        raise FileNotFoundError(f"no raw parquet files for sessions {list(sessions)} in {raw_dir}")
    return [pd.read_parquet(p) for p in files]


def clean_dataset(
    recordings: Sequence[pd.DataFrame], profile: CleanProfile, activities: Sequence[int]
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Clean every recording and combine them.

    Segment ids are made globally unique, so ``segment_id`` alone identifies a
    contiguous piece of signal.

    Args:
        recordings: Raw recordings.
        profile: Cleaning profile.
        activities: Target activity IDs.

    Returns:
        Combined clean frame and a quality report (totals and per recording).
    """
    parts: list[pd.DataFrame] = []
    per_recording: dict[str, dict[str, int]] = {}
    offset = 0
    for raw in recordings:
        subject, session = int(raw["subject_id"].iloc[0]), str(raw["session"].iloc[0])
        sat = acc6_saturation_count(raw)
        clean, counts = clean_recording(raw, profile, activities)
        counts["acc6_saturated_values"] = sat
        clean["segment_id"] += offset
        offset += counts["segments"]
        parts.append(clean)
        per_recording[f"{session}/{subject}"] = counts
        logger.info("cleaned recording", extra=kv(subject=subject, session=session, **counts))

    combined = pd.concat(parts, ignore_index=True)
    totals: dict[str, int] = {}
    for counts in per_recording.values():
        for key, value in counts.items():
            totals[key] = totals.get(key, 0) + value
    report: dict[str, object] = {"totals": totals, "per_recording": per_recording}
    return combined, report


def run_clean(cfg: Config) -> dict[str, object]:
    """Pipeline stage ``clean``: raw Parquet -> ``data/clean/clean.parquet`` + report.

    Args:
        cfg: Resolved config.

    Returns:
        The quality report (also written to ``data/clean/meta.json``).
    """
    from src.schemas import clean_schema

    profile = cfg.clean_profile
    clean, report = clean_dataset(
        load_raw(cfg.paths.raw_dir, cfg.dataset.sessions), profile, cfg.dataset.activities
    )
    channels = profile_channels(profile)
    schema = clean_schema(
        channels,
        cfg.dataset.activities,
        check_ranges=profile.clip_to_sensor_range,
        strict_time=profile.drop_duplicates,
        single_label_segments=profile.split_on_label_change,
    )
    clean = schema.validate(clean, lazy=True)

    out_dir = Path(cfg.paths.clean_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(out_dir / "clean.parquet", index=False, compression="zstd")
    meta = {
        "data_version": cfg.data.version,
        "profile": profile.model_dump(),
        "channels": channels,
        "sessions": cfg.dataset.sessions,
        "activities": cfg.dataset.activities,
        "rows": len(clean),
        "quality": report,
    }
    write_json(out_dir / "meta.json", meta)
    logger.info("clean data written", extra=kv(version=cfg.data.version, rows=len(clean)))
    return meta
