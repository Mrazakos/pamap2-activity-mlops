"""Always-on model tests on tiny untrained networks (no trained model needed)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from src.config import BaselineConfig, ModelConfig
from src.models import build_model
from src.models.baseline import build_baseline, feature_importances
from src.models.bundle import state_dict_sha256
from src.train import class_weights, make_loader
from src.utils import set_seed

N_CHANNELS, N_CLASSES, LENGTH = 6, 4, 64


@pytest.fixture(params=["cnn", "cnn_gru"])
def model(request: pytest.FixtureRequest) -> nn.Module:
    """A tiny CNN or CNN-GRU."""
    set_seed(0)
    cfg = ModelConfig(name=request.param, conv_channels=[8, 16], kernel_size=3, dropout=0.0, gru_hidden=8)
    return build_model(cfg, N_CHANNELS, N_CLASSES)


def test_output_shape(model: nn.Module) -> None:
    logits = model(torch.randn(5, N_CHANNELS, LENGTH))
    assert logits.shape == (5, N_CLASSES)


def test_probabilities_sum_to_one(model: nn.Module) -> None:
    model.eval()
    proba = torch.softmax(model(torch.randn(7, N_CHANNELS, LENGTH)), dim=1)
    torch.testing.assert_close(proba.sum(dim=1), torch.ones(7))


def test_loss_decreases_when_overfitting_one_batch(model: nn.Module) -> None:
    set_seed(0)
    x, y = torch.randn(16, N_CHANNELS, LENGTH), torch.arange(16) % N_CLASSES
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    first = loss_fn(model(x), y).item()
    for _ in range(60):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.3 * first


def test_same_seed_same_weights() -> None:
    cfg = ModelConfig(conv_channels=[8], kernel_size=3)
    set_seed(1)
    a = build_model(cfg, N_CHANNELS, N_CLASSES)
    set_seed(1)
    b = build_model(cfg, N_CHANNELS, N_CLASSES)
    assert state_dict_sha256(a) == state_dict_sha256(b)


def test_seeded_loader_order_is_reproducible() -> None:
    x, y = np.zeros((20, 2, 4), np.float32), np.arange(20)
    order = [torch.cat([b[1] for b in make_loader(x, y, 5, True, seed, 0)]).tolist() for seed in (3, 3, 4)]
    assert order[0] == order[1] != order[2]


def test_class_weights_inverse_frequency() -> None:
    w = class_weights(np.array([0, 0, 0, 1]), n_classes=3)
    assert w[2] == 0  # absent class
    assert w[1] == pytest.approx(3 * w[0])


@pytest.mark.parametrize("kind", ["random_forest", "logistic_regression"])
def test_baseline_is_deterministic(kind: str) -> None:
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(60, 5)), np.arange(60) % 3
    cfg = BaselineConfig(model=kind, n_estimators=10, n_jobs=2)  # type: ignore[arg-type]
    p1 = build_baseline(cfg, seed=7).fit(x, y).predict_proba(x)
    p2 = build_baseline(cfg, seed=7).fit(x, y).predict_proba(x)
    np.testing.assert_array_equal(p1, p2)
    table = feature_importances(build_baseline(cfg, seed=7).fit(x, y), list("abcde"))
    assert list(table.columns) == ["feature", "importance"] and len(table) == 5
