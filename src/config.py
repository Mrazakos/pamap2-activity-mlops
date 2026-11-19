"""Typed configuration loaded from ``configs/params.yaml`` plus CLI overrides.

The YAML file is the single source of truth (it is also the DVC params file).
Overrides use dotted keys, e.g. ``train.lr=3e-4`` or ``split.test_subjects=[101]``,
and are applied *before* validation so they are type-checked like the file itself.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.constants import ACTIVITY_NAMES, SENSOR_GROUPS, SUBJECT_IDS

DEFAULT_CONFIG_PATH = Path("configs/params.yaml")


class _Strict(BaseModel):
    """Base model that rejects unknown keys, so typos in configs fail loudly."""

    model_config = ConfigDict(extra="forbid")


class DatasetConfig(_Strict):
    """Where the dataset comes from and which part of it is used."""

    url: str
    sha256: str = ""
    sessions: list[Literal["protocol", "optional"]] = ["protocol"]
    activities: list[int]

    @model_validator(mode="after")
    def _known_activities(self) -> DatasetConfig:
        unknown = [a for a in self.activities if a not in ACTIVITY_NAMES or a == 0]
        if unknown:
            raise ValueError(f"unknown or transient activity ids: {unknown}")
        if len(set(self.activities)) != len(self.activities):
            raise ValueError("duplicate activity ids")
        return self


class DataConfig(_Strict):
    """Which data version and sensor subset a run uses."""

    version: str = "v2"
    sensor_subset: Literal["all", "wrist_hr", "wrist", "chest_hr"] = "all"

    @property
    def sensor_groups(self) -> tuple[str, ...]:
        """Sensor groups (IMU locations and/or ``hr``) of the selected subset."""
        return SENSOR_GROUPS[self.sensor_subset]


class PathsConfig(_Strict):
    """Filesystem layout (all relative to the project root)."""

    zip_path: Path
    extract_dir: Path
    raw_dir: Path
    clean_dir: Path
    processed_dir: Path
    features_dir: Path
    versions_dir: Path
    models_dir: Path
    reports_dir: Path
    runs_dir: Path


class CleanProfile(_Strict):
    """One data-quality profile (``v1`` = naive, ``v2`` = full pipeline)."""

    drop_transient: bool = True
    drop_duplicates: bool = True
    channels: Literal["all", "motion"] = "motion"
    clip_to_sensor_range: bool = True
    interpolate_limit: int | None = Field(default=10, ge=1)
    hr_ffill_limit: int | None = Field(default=200, ge=1)
    max_gap_seconds: float | None = Field(default=0.05, gt=0)
    split_on_label_change: bool = True
    min_label_purity: float = Field(default=1.0, ge=0.0, le=1.0)


class SplitConfig(_Strict):
    """Subject-wise split. Every subject not listed is a training subject."""

    val_subjects: list[int]
    test_subjects: list[int]

    @model_validator(mode="after")
    def _disjoint(self) -> SplitConfig:
        overlap = sorted(set(self.val_subjects) & set(self.test_subjects))
        if overlap:
            raise ValueError(f"subjects in both val and test: {overlap}")
        unknown = sorted(set(self.val_subjects + self.test_subjects) - set(SUBJECT_IDS))
        if unknown:
            raise ValueError(f"unknown subjects: {unknown}")
        return self


class WindowConfig(_Strict):
    """Sliding-window segmentation."""

    window_seconds: float = Field(default=5.12, gt=0)
    overlap: float = Field(default=0.5, ge=0.0, lt=1.0)
    downsample: int = Field(default=1, ge=1)


class FeatureConfig(_Strict):
    """Hand-crafted features for the baseline."""

    percentiles: list[float] = [10, 25, 50, 75, 90]
    fft_bands_hz: list[tuple[float, float]] = [(0, 1), (1, 3), (3, 5), (5, 10), (10, 20)]
    chunk_size: int = Field(default=256, ge=1)


class BaselineConfig(_Strict):
    """Classical model on hand-crafted features."""

    model: Literal["random_forest", "logistic_regression"] = "random_forest"
    n_estimators: int = Field(default=300, ge=1)
    max_depth: int | None = None
    min_samples_leaf: int = Field(default=1, ge=1)
    class_weight: Literal["balanced"] | None = "balanced"
    lr_c: float = Field(default=1.0, gt=0)
    lr_max_iter: int = Field(default=3000, ge=1)
    n_jobs: int = 4


class ModelConfig(_Strict):
    """Deep model architecture."""

    name: Literal["cnn", "cnn_gru"] = "cnn"
    conv_channels: list[int] = [64, 128, 128]
    kernel_size: int = Field(default=7, ge=1)
    dropout: float = Field(default=0.3, ge=0.0, lt=1.0)
    gru_hidden: int = Field(default=128, ge=1)
    gru_layers: int = Field(default=1, ge=1)


class TrainConfig(_Strict):
    """Optimisation settings for the deep models."""

    lr: float = Field(default=1e-3, gt=0)
    batch_size: int = Field(default=64, ge=1)
    epochs: int = Field(default=40, ge=1)
    weight_decay: float = Field(default=1e-4, ge=0)
    class_weighting: bool = True
    patience: int = Field(default=8, ge=1)
    scheduler: Literal["none", "cosine"] = "none"
    num_workers: int = Field(default=2, ge=0)
    log_every_n_steps: int = Field(default=20, ge=1)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    max_windows_per_split: int | None = Field(default=None, ge=1)
    eval_test: bool = False


class TrackingConfig(_Strict):
    """MLflow naming (the tracking URI itself only ever comes from the environment)."""

    experiment: str = "shmqak-pamap2-har"
    run_name: str | None = None


class Config(_Strict):
    """Root of the resolved configuration."""

    seed: int = 42
    dataset: DatasetConfig
    data: DataConfig
    paths: PathsConfig
    clean: dict[str, CleanProfile]
    split: SplitConfig
    windows: WindowConfig
    features: FeatureConfig
    baseline: BaselineConfig
    model: ModelConfig
    train: TrainConfig
    tracking: TrackingConfig

    @model_validator(mode="after")
    def _version_has_profile(self) -> Config:
        if self.data.version not in self.clean:
            raise ValueError(
                f"data.version={self.data.version!r} has no profile in clean: {sorted(self.clean)}"
            )
        return self

    @property
    def clean_profile(self) -> CleanProfile:
        """The cleaning profile selected by ``data.version``."""
        return self.clean[self.data.version]

    @property
    def class_ids(self) -> list[int]:
        """Activity IDs in class-index order (sorted, so the mapping is stable)."""
        return sorted(self.dataset.activities)

    @property
    def class_names(self) -> list[str]:
        """Human-readable class names in class-index order."""
        return [ACTIVITY_NAMES[a] for a in self.class_ids]


def parse_override(override: str) -> tuple[list[str], Any]:
    """Parse one ``dotted.key=value`` override.

    The value is parsed as YAML, so ``3e-4``, ``true``, ``null`` and ``[101, 102]``
    get their natural types.

    Args:
        override: String of the form ``a.b.c=value``.

    Returns:
        The key path and the parsed value.

    Raises:
        ValueError: If the string has no ``=`` or an empty key.
    """
    key, sep, raw = override.partition("=")
    if not sep or not key.strip():
        raise ValueError(f"override must look like key=value, got {override!r}")
    # YAML 1.1 reads "1e-3" as a string; pydantic's lax mode turns it into a float later.
    value = yaml.safe_load(raw) if raw.strip() else None
    return key.strip().split("."), value


def apply_overrides(raw: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Return a copy of ``raw`` with dotted overrides applied.

    Args:
        raw: Nested dictionary as loaded from YAML.
        overrides: Strings like ``train.lr=3e-4``.

    Returns:
        A new dictionary; ``raw`` is not modified.

    Raises:
        KeyError: If an override addresses a key that does not exist (except
            inside the ``clean`` profiles, which may be added).
    """
    result = copy.deepcopy(raw)
    for override in overrides:
        path, value = parse_override(override)
        node = result
        for i, part in enumerate(path[:-1]):
            if part not in node or not isinstance(node[part], dict):
                raise KeyError(f"unknown config key {'.'.join(path[: i + 1])!r} in {override!r}")
            node = node[part]
        if path[-1] not in node and path[0] != "clean":
            raise KeyError(f"unknown config key {'.'.join(path)!r} in {override!r}")
        node[path[-1]] = value
    return result


def load_raw_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Read the YAML config file without validation.

    Args:
        path: Path to the YAML file.

    Returns:
        The parsed YAML mapping.
    """
    with Path(path).open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a mapping")
    return data


def load_config(path: Path = DEFAULT_CONFIG_PATH, overrides: Sequence[str] = ()) -> Config:
    """Load, override and validate the configuration.

    Args:
        path: YAML file to read.
        overrides: Dotted ``key=value`` overrides applied before validation.

    Returns:
        The validated :class:`Config`.
    """
    raw = apply_overrides(load_raw_config(path), overrides)
    return Config.model_validate(raw)


def config_to_dict(cfg: Config) -> dict[str, Any]:
    """Return a JSON-serialisable dictionary of the resolved config.

    Args:
        cfg: Validated config.

    Returns:
        Plain dict (paths as strings, tuples as lists).
    """
    return json.loads(cfg.model_dump_json())


def config_hash(cfg: Config) -> str:
    """Return a short stable hash of the resolved config.

    Args:
        cfg: Validated config.

    Returns:
        First 12 hex characters of the SHA-256 of the canonical JSON.
    """
    canonical = json.dumps(config_to_dict(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]
