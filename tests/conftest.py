"""Shared fixtures. All data here is synthetic (sines, noise, steps) with the real column schema.

No real PAMAP2 rows are ever used in or committed with the tests.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import Config, load_config
from src.constants import DAT_COLUMNS, GRAVITY, IMU_LOCATIONS

RATE = 100
ACTIVITY_PROFILE: dict[int, tuple[float, float, float]] = {
    # activity: (frequency Hz, acceleration amplitude m/s^2, heart rate bpm)
    1: (0.2, 0.05, 62.0),
    2: (0.2, 0.08, 68.0),
    3: (0.3, 0.10, 75.0),
    4: (1.8, 3.0, 100.0),
    5: (2.8, 9.0, 150.0),
    24: (2.2, 12.0, 165.0),
}
SYNTH_ACTIVITIES: tuple[int, ...] = tuple(ACTIVITY_PROFILE)


def make_recording(
    subject_id: int,
    activities: Sequence[int] = SYNTH_ACTIVITIES,
    seconds_per_activity: float = 12.0,
    transient_seconds: float = 2.0,
    seed: int = 0,
    session: str = "protocol",
    defects: bool = True,
) -> pd.DataFrame:
    """Build one synthetic recording in the raw ``.dat`` column layout.

    Each activity is a block of sinusoidal motion with an activity-specific
    frequency/amplitude, separated by transient (label 0) blocks. Heart rate
    is present only every 11th row, like the real ~9 Hz monitor.

    Args:
        subject_id: Subject id to stamp on the rows.
        activities: Activity blocks in order.
        seconds_per_activity: Duration of each block.
        transient_seconds: Duration of the transient blocks in between.
        seed: Noise seed.
        session: ``protocol`` or ``optional``.
        defects: Inject a short NaN burst, a long NaN burst, a timestamp gap and
            a duplicated timestamp.

    Returns:
        DataFrame with ``subject_id``, ``session`` and the 54 raw columns.
    """
    rng = np.random.default_rng(seed + subject_id)
    labels: list[np.ndarray] = []
    for i, activity in enumerate(activities):
        if i:
            labels.append(np.zeros(int(transient_seconds * RATE), dtype=np.int64))
        labels.append(np.full(int(seconds_per_activity * RATE), activity, dtype=np.int64))
    label = np.concatenate(labels)
    n = len(label)
    t = np.arange(n) / RATE
    freq = np.array([ACTIVITY_PROFILE.get(int(a), (0.5, 1.0, 90.0))[0] for a in label])
    amp = np.array([ACTIVITY_PROFILE.get(int(a), (0.5, 1.0, 90.0))[1] for a in label])
    hr_level = np.array([ACTIVITY_PROFILE.get(int(a), (0.5, 1.0, 90.0))[2] for a in label])
    phase = 2 * np.pi * freq * t

    data: dict[str, np.ndarray] = {
        "timestamp": 5.0 + t,
        "activity_id": label,
        "heart_rate": np.where(np.arange(n) % 11 == 0, hr_level + rng.normal(0, 1, n), np.nan),
    }
    for k, loc in enumerate(IMU_LOCATIONS):
        wave = amp * np.sin(phase + k)
        data[f"{loc}_temp"] = 32.0 + 0.01 * t
        for axis, offset in zip("xyz", (0.0, 0.0, GRAVITY), strict=True):
            noise = rng.normal(0, 0.02, n)
            data[f"{loc}_acc16_{axis}"] = offset + wave + noise
            data[f"{loc}_acc6_{axis}"] = np.clip(offset + wave + noise, -6 * GRAVITY, 6 * GRAVITY)
            data[f"{loc}_gyro_{axis}"] = 0.3 * amp * np.cos(phase + k) + rng.normal(0, 0.01, n)
            data[f"{loc}_mag_{axis}"] = 20.0 + rng.normal(0, 0.5, n)
        for j in range(4):
            data[f"{loc}_orient_{j}"] = np.full(n, 1.0 if j == 0 else 0.0)
    df = pd.DataFrame(data)[DAT_COLUMNS]

    if defects:
        start = int(seconds_per_activity * RATE // 2)
        df.loc[start : start + 2, "hand_acc16_x"] = np.nan  # 3-sample burst: interpolated in v2
        df.loc[start + 300 : start + 349, "chest_gyro_y"] = np.nan  # 50-sample burst: dropped in v2
        gap_at = n - int(seconds_per_activity * RATE // 2)
        df.loc[gap_at:, "timestamp"] += 0.5  # 0.5 s gap: splits the segment in v2
        dup = df.iloc[[100]].copy()
        df = pd.concat([df.iloc[:101], dup, df.iloc[101:]], ignore_index=True)  # duplicated timestamp

    df.insert(0, "session", session)
    df.insert(0, "subject_id", subject_id)
    df["activity_id"] = df["activity_id"].astype("int64")
    df["subject_id"] = df["subject_id"].astype("int64")
    return df


def write_raw(raw_dir: Path, subjects: Sequence[int] = (101, 102, 103, 104)) -> list[Path]:
    """Write synthetic raw Parquet files in the ``data/raw`` layout.

    Args:
        raw_dir: Target ``raw`` folder.
        subjects: Subjects to generate.

    Returns:
        Written files.
    """
    paths = []
    for subject in subjects:
        out = raw_dir / "protocol" / f"subject{subject}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        make_recording(subject).to_parquet(out, index=False)
        paths.append(out)
    return paths


TINY_OVERRIDES: tuple[str, ...] = (
    "split.val_subjects=[103]",
    "split.test_subjects=[104]",
    "windows.window_seconds=1.28",
    "model.conv_channels=[8,8]",
    "model.kernel_size=3",
    "model.gru_hidden=8",
    "train.epochs=2",
    "train.batch_size=16",
    "train.patience=2",
    "train.num_workers=0",
    "train.device=cpu",
    "train.log_every_n_steps=2",
    "baseline.n_estimators=10",
    "baseline.n_jobs=1",
    "features.chunk_size=64",
)


def path_overrides(root: Path) -> list[str]:
    """Point every data/model/run path at a temporary folder.

    Args:
        root: Temporary root.

    Returns:
        Override strings.
    """
    names = [
        "raw_dir",
        "clean_dir",
        "processed_dir",
        "features_dir",
        "versions_dir",
        "models_dir",
        "reports_dir",
        "runs_dir",
        "extract_dir",
    ]
    overrides = [f"paths.{name}={(root / name).as_posix()}" for name in names]
    return [*overrides, f"paths.zip_path={(root / 'pamap2.zip').as_posix()}"]


@pytest.fixture(scope="session")
def mlflow_store(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Point MLflow at a throwaway SQLite store for the whole test session.

    Args:
        tmp_path_factory: pytest factory for temporary folders.

    Yields:
        The store folder.
    """
    root = tmp_path_factory.mktemp("mlflow")
    patcher = pytest.MonkeyPatch()
    patcher.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(root / 'mlflow.db').as_posix()}")
    patcher.setenv("MLFLOW_ARTIFACT_ROOT", (root / "artifacts").as_uri())
    patcher.setenv("MLFLOW_EXPERIMENT", "pytest")
    patcher.delenv("SLURM_LOG_FILE", raising=False)
    yield root
    patcher.undo()


@pytest.fixture
def tiny_overrides(tmp_path: Path) -> list[str]:
    """Overrides for a fast end-to-end run on synthetic data in ``tmp_path``.

    Args:
        tmp_path: pytest temporary folder.

    Returns:
        Override strings.
    """
    return [*TINY_OVERRIDES, *path_overrides(tmp_path)]


@pytest.fixture
def tiny_cfg(tiny_overrides: list[str]) -> Config:
    """Config for fast tests (small windows/models, temporary paths).

    Args:
        tiny_overrides: Override strings.

    Returns:
        The validated config.
    """
    return load_config(overrides=tiny_overrides)


@pytest.fixture
def recording() -> pd.DataFrame:
    """One synthetic raw recording with injected defects.

    Returns:
        Raw frame.
    """
    return make_recording(101)
