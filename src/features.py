"""Hand-crafted window features for the classical baseline.

Per channel: mean, std, min, max, percentiles, RMS, zero-crossing rate, dominant
frequency and relative spectral energy in fixed bands (spectral features are
skipped for heart rate). Per sensor triad (x/y/z): pairwise correlations and the
mean/std of the vector magnitude (orientation-invariant).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.config import Config, FeatureConfig, SplitConfig
from src.schemas import FEATURE_PREFIX
from src.utils import get_logger, kv, write_json

logger = get_logger(__name__)

FloatArray = npt.NDArray[np.float32]
_EPS = 1e-12
_MIN_VARIANCE = 1e-12
"""Below this variance a window counts as constant (no spectrum)."""


def time_features(x: FloatArray, percentiles: Sequence[float]) -> tuple[FloatArray, list[str]]:
    """Statistical features per channel.

    Args:
        x: Windows ``(n, channels, length)``.
        percentiles: Percentiles to compute (0-100).

    Returns:
        ``(features (n, channels, k), feature suffixes of length k)``.
    """
    mean = x.mean(axis=-1)
    centered = x - mean[..., None]
    sign_changes = np.diff(np.signbit(centered), axis=-1) != 0
    stats = [
        mean,
        x.std(axis=-1),
        x.min(axis=-1),
        x.max(axis=-1),
        *np.percentile(x, percentiles, axis=-1),
        np.sqrt((x**2).mean(axis=-1)),
        sign_changes.mean(axis=-1),
    ]
    names = ["mean", "std", "min", "max", *[f"p{p:g}" for p in percentiles], "rms", "zcr"]
    return np.stack(stats, axis=-1).astype(np.float32), names


def spectral_features(
    x: FloatArray, rate_hz: float, bands: Sequence[tuple[float, float]]
) -> tuple[FloatArray, list[str]]:
    """Dominant frequency and relative band energies per channel.

    The mean is removed (in float64) and the DC bin is excluded, so band shares
    sum to at most 1; a (near-)constant signal has dominant frequency 0 and
    zero energy in every band.

    Args:
        x: Windows ``(n, channels, length)``.
        rate_hz: Sampling rate.
        bands: ``[low, high)`` frequency bands in Hz.

    Returns:
        ``(features (n, channels, 1 + len(bands)), suffixes)``.
    """
    centered = x.astype(np.float64) - x.astype(np.float64).mean(axis=-1, keepdims=True)
    ac_power = (np.abs(np.fft.rfft(centered, axis=-1)) ** 2)[..., 1:]
    ac_freqs = np.fft.rfftfreq(x.shape[-1], d=1.0 / rate_hz)[1:]
    total = ac_power.sum(axis=-1)
    moving = centered.var(axis=-1) > _MIN_VARIANCE
    safe_total = np.where(moving, total, 1.0)
    dominant = np.where(moving, ac_freqs[ac_power.argmax(axis=-1)], 0.0)
    shares = [
        np.where(moving, ac_power[..., (ac_freqs >= lo) & (ac_freqs < hi)].sum(axis=-1) / safe_total, 0.0)
        for lo, hi in bands
    ]
    names = ["dom_freq", *[f"band_{lo:g}_{hi:g}hz" for lo, hi in bands]]
    return np.stack([dominant, *shares], axis=-1).astype(np.float32), names


def triads(channels: Sequence[str]) -> list[tuple[str, tuple[int, int, int]]]:
    """Find x/y/z channel triples, e.g. ``hand_acc16_{x,y,z}``.

    Args:
        channels: Channel names.

    Returns:
        ``(prefix, (ix, iy, iz))`` for each complete triple, in channel order.
    """
    position = {c: i for i, c in enumerate(channels)}
    found: list[tuple[str, tuple[int, int, int]]] = []
    for c in channels:
        if c.endswith("_x"):
            prefix = c[:-2]
            if f"{prefix}_y" in position and f"{prefix}_z" in position:
                found.append((prefix, (position[c], position[f"{prefix}_y"], position[f"{prefix}_z"])))
    return found


def triad_features(x: FloatArray, channels: Sequence[str]) -> tuple[FloatArray, list[str]]:
    """Inter-axis correlations and magnitude statistics per sensor triad.

    Args:
        x: Windows ``(n, channels, length)``.
        channels: Channel names.

    Returns:
        ``(features (n, k), full feature names)``.
    """
    columns: list[npt.NDArray[np.floating]] = []
    names: list[str] = []
    for prefix, (ix, iy, iz) in triads(channels):
        axes = {"x": x[:, ix], "y": x[:, iy], "z": x[:, iz]}
        for a, b in (("x", "y"), ("x", "z"), ("y", "z")):
            da = axes[a] - axes[a].mean(axis=-1, keepdims=True)
            db = axes[b] - axes[b].mean(axis=-1, keepdims=True)
            denom = np.sqrt((da**2).sum(axis=-1) * (db**2).sum(axis=-1))
            columns.append(np.clip((da * db).sum(axis=-1) / (denom + _EPS), -1.0, 1.0))
            names.append(f"{prefix}__corr_{a}{b}")
        magnitude = np.sqrt(axes["x"] ** 2 + axes["y"] ** 2 + axes["z"] ** 2)
        columns += [magnitude.mean(axis=-1), magnitude.std(axis=-1)]
        names += [f"{prefix}__mag_mean", f"{prefix}__mag_std"]
    if not columns:
        return np.empty((x.shape[0], 0), dtype=np.float32), []
    return np.stack(columns, axis=-1).astype(np.float32), names


def window_features(
    x: FloatArray, channels: Sequence[str], rate_hz: float, cfg: FeatureConfig
) -> tuple[FloatArray, list[str]]:
    """All features for a batch of windows.

    Args:
        x: Windows ``(n, channels, length)``.
        channels: Channel names.
        rate_hz: Sampling rate.
        cfg: Feature settings.

    Returns:
        ``(features (n, n_features), names)`` with names prefixed by ``f__``.
    """
    stats, stat_names = time_features(x, cfg.percentiles)
    blocks = [stats.reshape(len(x), -1)]
    names = [f"{c}__{s}" for c in channels for s in stat_names]

    motion_idx = [i for i, c in enumerate(channels) if c != "heart_rate"]
    if motion_idx:
        spec, spec_names = spectral_features(x[:, motion_idx], rate_hz, cfg.fft_bands_hz)
        blocks.append(spec.reshape(len(x), -1))
        names += [f"{channels[i]}__{s}" for i in motion_idx for s in spec_names]

    tri, tri_names = triad_features(x, channels)
    blocks.append(tri)
    names += tri_names
    return np.concatenate(blocks, axis=1).astype(np.float32), [FEATURE_PREFIX + n for n in names]


def features_frame(
    x: FloatArray,
    channels: Sequence[str],
    rate_hz: float,
    cfg: FeatureConfig,
) -> tuple[FloatArray, list[str]]:
    """Compute features in chunks (bounded memory for long splits).

    Args:
        x: Windows ``(n, channels, length)``.
        channels: Channel names.
        rate_hz: Sampling rate.
        cfg: Feature settings (``chunk_size``).

    Returns:
        ``(features, names)``.
    """
    parts: list[FloatArray] = []
    names: list[str] = []
    for start in range(0, len(x), cfg.chunk_size):
        feats, names = window_features(x[start : start + cfg.chunk_size], channels, rate_hz, cfg)
        parts.append(feats)
    if not parts:
        _, names = window_features(np.zeros((1, *x.shape[1:]), np.float32), channels, rate_hz, cfg)
        return np.empty((0, len(names)), np.float32), names
    return np.concatenate(parts, axis=0), names


def compute_feature_table(
    cfg: Config, split: SplitConfig | None = None
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Window the clean data, compute features per split and validate the table.

    Shared by the ``features`` stage (default split) and LOSO (custom folds).

    Args:
        cfg: Resolved config.
        split: Optional split override.

    Returns:
        ``(feature_table, metadata)``; the table has ``subject_id, label, split``
        plus ``f__*`` feature columns.
    """
    from src.data_versions import resolve_clean_dir
    from src.schemas import features_schema
    from src.windows import load_clean, prepare_data

    clean, meta = load_clean(resolve_clean_dir(cfg))
    data = prepare_data(cfg, clean, meta["channels"], split)  # type: ignore[arg-type]
    frames: list[pd.DataFrame] = []
    names: list[str] = []
    for split_name, arrays in data.splits.items():
        feats, names = features_frame(arrays.x, data.channels, data.sample_rate_hz, cfg.features)
        frame = pd.DataFrame(feats, columns=names)
        frame.insert(0, "split", split_name)
        frame.insert(0, "label", arrays.y)
        frame.insert(0, "subject_id", arrays.subjects)
        frames.append(frame)
        logger.info("features computed", extra=kv(split=split_name, windows=len(frame), features=len(names)))
    table = features_schema().validate(pd.concat(frames, ignore_index=True), lazy=True)
    feature_meta: dict[str, object] = {
        "data_version": cfg.data.version,
        "sensor_subset": cfg.data.sensor_subset,
        "channels": data.channels,
        "n_features": len(names),
        "feature_names": names,
        "sample_rate_hz": data.sample_rate_hz,
        "window_length": data.window_length,
        "class_ids": data.class_ids,
        "split_subjects": data.split_subjects,
        "norm_mean": data.norm_mean.tolist(),
        "norm_std": data.norm_std.tolist(),
    }
    return table, feature_meta


def run_features(cfg: Config) -> dict[str, object]:
    """Pipeline stage ``features``: windows -> ``data/features/features.parquet``.

    Args:
        cfg: Resolved config.

    Returns:
        Feature metadata (also written to ``data/features/meta.json``).
    """
    table, feature_meta = compute_feature_table(cfg)
    out = Path(cfg.paths.features_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / "features.parquet", index=False)
    write_json(out / "meta.json", feature_meta)
    return feature_meta


def load_feature_table(cfg: Config) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read the feature table written by the ``features`` stage and check it matches the config.

    Args:
        cfg: Resolved config.

    Returns:
        ``(feature_table, metadata)``.

    Raises:
        ValueError: If the stored features were built for another data version,
            sensor subset or split.
    """
    from src.utils import read_json
    from src.windows import window_geometry

    out = Path(cfg.paths.features_dir)
    meta = read_json(out / "meta.json")
    expected = {
        "data_version": cfg.data.version,
        "sensor_subset": cfg.data.sensor_subset,
        "window_length": window_geometry(cfg.windows)[0],
    }
    for key, value in expected.items():
        if meta[key] != value:
            raise ValueError(
                f"features were built with {key}={meta[key]!r}, config has {value!r}; run `dvc repro`"
            )
    if meta["split_subjects"]["val"] != sorted(cfg.split.val_subjects) or meta["split_subjects"][
        "test"
    ] != sorted(cfg.split.test_subjects):
        raise ValueError("features were built for a different split; run `dvc repro`")
    return pd.read_parquet(out / "features.parquet"), meta
