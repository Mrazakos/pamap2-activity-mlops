"""Unit tests for the hand-crafted features on signals with known properties."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import FeatureConfig
from src.features import features_frame, spectral_features, time_features, triad_features, triads

RATE = 100.0
T = np.arange(512) / RATE


def _batch(*signals: np.ndarray) -> np.ndarray:
    """Stack 1-D signals into a single window of shape (1, channels, length)."""
    return np.stack(signals)[None].astype(np.float32)


@pytest.mark.parametrize("freq", [1.5625, 3.125, 6.25])
def test_dominant_frequency_of_sine(freq: float) -> None:
    feats, names = spectral_features(_batch(np.sin(2 * np.pi * freq * T)), RATE, [(0, 5), (5, 20)])
    assert names[0] == "dom_freq"
    assert feats[0, 0, 0] == pytest.approx(freq, abs=RATE / 512)


def test_band_shares_locate_energy() -> None:
    feats, _ = spectral_features(_batch(np.sin(2 * np.pi * 2.0 * T)), RATE, [(0, 1), (1, 3), (3, 20)])
    shares = feats[0, 0, 1:]
    assert shares[1] > 0.95
    assert shares.sum() <= 1.0 + 1e-5


def test_constant_signal_has_no_spectrum() -> None:
    feats, _ = spectral_features(_batch(np.full_like(T, 9.81)), RATE, [(0, 5)])
    assert np.allclose(feats, 0.0)


def test_time_features_known_values() -> None:
    feats, names = time_features(_batch(np.sin(2 * np.pi * 1.0 * T) * 2.0), [50])
    f = dict(zip(names, feats[0, 0], strict=True))
    assert f["rms"] == pytest.approx(np.sqrt(2.0), rel=0.02)
    assert f["max"] == pytest.approx(2.0, rel=0.01)
    assert f["zcr"] == pytest.approx(2 * 1.0 / RATE, rel=0.15)  # two zero crossings per period


def test_triad_correlation_and_magnitude() -> None:
    s = np.sin(2 * np.pi * T)
    x = _batch(s, s, -s)
    found = triads(["acc_x", "acc_y", "acc_z"])
    assert found == [("acc", (0, 1, 2))]
    feats, names = triad_features(x, ["acc_x", "acc_y", "acc_z"])
    f = dict(zip(names, feats[0], strict=True))
    assert f["acc__corr_xy"] == pytest.approx(1.0, abs=1e-5)
    assert f["acc__corr_xz"] == pytest.approx(-1.0, abs=1e-5)


def test_features_frame_chunks_consistently() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(10, 4, 128)).astype(np.float32)
    channels = ["heart_rate", "hand_acc16_x", "hand_acc16_y", "hand_acc16_z"]
    whole, names = features_frame(x, channels, RATE, FeatureConfig(chunk_size=100))
    chunked, names2 = features_frame(x, channels, RATE, FeatureConfig(chunk_size=3))
    assert names == names2 and whole.shape == (10, len(names))
    np.testing.assert_allclose(whole, chunked, rtol=1e-6)
    assert all(n.startswith("f__") for n in names)
    assert not any("heart_rate__dom_freq" in n for n in names)  # no spectrum for heart rate
    assert np.isfinite(whole).all()
