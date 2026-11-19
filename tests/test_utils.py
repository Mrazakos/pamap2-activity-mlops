"""Unit tests for seeding, hashing, flattening and structured logging."""

from __future__ import annotations

import json
import logging
import random

import numpy as np
import torch

from src.utils import JsonFormatter, KeyValueFormatter, flatten_dict, kv, log_context, set_seed, sha256_json


def _draws() -> tuple[float, float, list[float]]:
    """One draw from each RNG."""
    return random.random(), float(np.random.rand()), torch.rand(3).tolist()


def test_set_seed_makes_all_rngs_repeatable() -> None:
    set_seed(123)
    first = _draws()
    set_seed(123)
    assert _draws() == first
    set_seed(124)
    assert _draws() != first


def test_set_seed_enables_deterministic_torch() -> None:
    set_seed(0)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark


def test_flatten_dict() -> None:
    nested = {"a": {"b": 1, "c": {"d": [1, 2]}}, "e": None, "f": {}}
    assert flatten_dict(nested) == {"a.b": 1, "a.c.d": [1, 2], "e": None, "f": {}}


def test_sha256_json_is_key_order_independent() -> None:
    assert sha256_json({"a": 1, "b": 2}) == sha256_json({"b": 2, "a": 1})
    assert sha256_json({"a": 1}) != sha256_json({"a": 2})


def _record(**ctx: object) -> logging.LogRecord:
    """A log record carrying per-call context."""
    record = logging.makeLogRecord({"name": "t", "levelname": "INFO", "msg": "hello"})
    record.ctx = ctx
    return record


def test_key_value_formatter_appends_context() -> None:
    with log_context(stage="train"):
        line = KeyValueFormatter("%(message)s").format(_record(**kv(subject=101)["ctx"]))
    assert line == "hello | stage=train subject=101"


def test_json_formatter_is_valid_json() -> None:
    payload = json.loads(JsonFormatter().format(_record(run_id="abc")))
    assert payload["message"] == "hello" and payload["run_id"] == "abc"
