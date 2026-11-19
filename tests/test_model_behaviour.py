"""CheckList-style behaviour tests of the *trained* model (INV, MFT, DIR).

They load the bundle produced by the DVC ``train`` stage (``MODEL_DIR``, default
``models/deep``) and the real clean data for the invariance tests. When either is
missing (e.g. in CI, which has no data), they are skipped with a reason.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from src.config import Config
from src.constants import ACTIVITY_NAMES, HIGH_INTENSITY_ACTIVITIES, STATIC_ACTIVITIES
from src.models.bundle import Predictor, load_bundle

pytestmark = pytest.mark.model

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models/deep"))
CONFIDENT = 0.9
AGREEMENT = 0.95


@pytest.fixture(scope="module")
def predictor() -> Predictor:
    """The trained model bundle (skips the module if absent)."""
    if not (MODEL_DIR / "bundle.json").exists():
        pytest.skip(f"no trained model bundle at {MODEL_DIR} (run `dvc repro train` or set MODEL_DIR)")
    return load_bundle(MODEL_DIR)


@pytest.fixture(scope="module")
def confident_windows(predictor: Predictor) -> npt.NDArray[np.float32]:
    """Validation windows the model classifies with probability >= 0.9 (real data, never committed)."""
    from src.data_versions import resolve_clean_dir
    from src.windows import load_clean, prepare_data

    cfg = Config.model_validate(predictor.meta.config)
    cfg = cfg.model_copy(update={"train": cfg.train.model_copy(update={"max_windows_per_split": 400})})
    try:
        clean, meta = load_clean(resolve_clean_dir(cfg))
    except (FileNotFoundError, RuntimeError) as exc:
        pytest.skip(f"clean data not available: {exc}")
    x = prepare_data(cfg, clean, meta["channels"], splits_needed=("val",)).splits["val"].x  # type: ignore[arg-type]
    keep = predictor.predict_proba(x).max(axis=1) >= CONFIDENT
    if keep.sum() < 20:
        pytest.skip("fewer than 20 confident validation windows")
    return x[keep]


def _class_index(predictor: Predictor, activity_ids: tuple[int, ...]) -> list[int]:
    """Class indices of the given activity ids."""
    return [predictor.meta.class_ids.index(a) for a in activity_ids if a in predictor.meta.class_ids]


def _motionless(predictor: Predictor, rng: np.random.Generator) -> npt.NDArray[np.float32]:
    """A synthetic resting window in raw units: average posture, zero rotation, resting heart rate."""
    channels, length = predictor.meta.channels, predictor.meta.window_length
    raw = np.repeat(predictor.mean[:, None], length, axis=1).astype(np.float32)
    for i, name in enumerate(channels):
        if "_gyro_" in name:
            raw[i] = 0.0
        elif name == "heart_rate":
            raw[i] = 70.0
    raw += rng.normal(0, 0.01, raw.shape).astype(np.float32)
    return predictor.normalize(raw[None])


def test_inv_small_noise_keeps_prediction(
    predictor: Predictor, confident_windows: npt.NDArray[np.float32]
) -> None:
    rng = np.random.default_rng(0)
    noisy = confident_windows + rng.normal(0, 0.05, confident_windows.shape).astype(np.float32)
    agreement = (predictor.predict(confident_windows) == predictor.predict(noisy)).mean()
    assert agreement >= AGREEMENT, f"only {agreement:.1%} unchanged under Gaussian noise"


@pytest.mark.parametrize("shift", [-5, 5])
def test_inv_small_time_shift_keeps_prediction(
    predictor: Predictor, confident_windows: npt.NDArray[np.float32], shift: int
) -> None:
    shifted = np.roll(confident_windows, shift, axis=-1)
    agreement = (predictor.predict(confident_windows) == predictor.predict(shifted)).mean()
    assert agreement >= AGREEMENT, f"only {agreement:.1%} unchanged under a {shift}-sample shift"


def test_mft_motionless_window_is_static(predictor: Predictor) -> None:
    window = _motionless(predictor, np.random.default_rng(0))
    predicted = predictor.meta.class_ids[int(predictor.predict(window)[0])]
    assert predicted not in HIGH_INTENSITY_ACTIVITIES, (
        f"motionless window classified as {ACTIVITY_NAMES[predicted]}"
    )
    assert predicted in STATIC_ACTIVITIES, f"motionless window classified as {ACTIVITY_NAMES[predicted]}"


def test_dir_more_movement_raises_high_intensity_probability(predictor: Predictor) -> None:
    rng = np.random.default_rng(0)
    base = (
        _motionless(predictor, rng)[0] * predictor.std[:, None] + predictor.mean[:, None]
    )  # back to raw units
    t = np.arange(predictor.meta.window_length) / predictor.meta.sample_rate_hz
    acc_rows = [i for i, c in enumerate(predictor.meta.channels) if "_acc" in c]
    high = _class_index(predictor, HIGH_INTENSITY_ACTIVITIES)
    scores = []
    for amplitude in (0.0, 20.0):
        raw = base.copy()
        raw[acc_rows] += amplitude * np.sin(2 * np.pi * 2.8 * t + np.arange(len(acc_rows))[:, None])
        scores.append(
            predictor.predict_proba(predictor.normalize(raw[None].astype(np.float32)))[0, high].sum()
        )
    assert scores[1] > scores[0], f"P(high intensity) did not increase: {scores}"
