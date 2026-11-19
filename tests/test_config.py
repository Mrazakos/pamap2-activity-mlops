"""Unit tests for config loading, overrides and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import apply_overrides, config_hash, load_config, parse_override


def test_default_config_loads() -> None:
    cfg = load_config()
    assert cfg.data.version in cfg.clean
    assert len(cfg.class_ids) == 12
    assert cfg.class_ids == sorted(cfg.class_ids)


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ("train.lr=3e-4", 3e-4),
        ("train.lr=0.01", 0.01),
        ("train.batch_size=128", 128),
        ("train.class_weighting=false", False),
        ("baseline.max_depth=null", None),
        ("split.test_subjects=[101, 102]", [101, 102]),
    ],
)
def test_overrides_are_typed(override: str, expected: object) -> None:
    cfg = load_config(overrides=[override, "split.val_subjects=[105]"])
    section, key = override.split("=")[0].split(".")
    assert getattr(getattr(cfg, section), key) == expected


def test_unknown_override_key_fails() -> None:
    with pytest.raises(KeyError):
        load_config(overrides=["train.learning_rate=0.1"])


def test_parse_override_requires_equals() -> None:
    with pytest.raises(ValueError, match="key=value"):
        parse_override("train.lr")


def test_apply_overrides_does_not_mutate_input() -> None:
    raw = {"train": {"lr": 1.0}}
    out = apply_overrides(raw, ["train.lr=2.0"])
    assert raw["train"]["lr"] == 1.0
    assert out["train"]["lr"] == 2.0


def test_split_overlap_is_rejected() -> None:
    with pytest.raises(ValidationError, match="both val and test"):
        load_config(overrides=["split.val_subjects=[101]", "split.test_subjects=[101]"])


def test_unknown_data_version_is_rejected() -> None:
    with pytest.raises(ValidationError, match="no profile"):
        load_config(overrides=["data.version=v9"])


def test_typo_in_section_is_rejected() -> None:
    with pytest.raises(ValidationError):
        load_config(overrides=["clean.v2.interpolate_limt=3"])


def test_config_hash_is_stable_and_sensitive() -> None:
    a, b = load_config(), load_config()
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash(load_config(overrides=["seed=7"]))
