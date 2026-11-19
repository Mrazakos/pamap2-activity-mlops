"""Tests for the MLflow helpers (against a throwaway SQLite store)."""

from __future__ import annotations

from pathlib import Path

import mlflow
import pytest

from src import tracking
from src.config import Config
from src.tracking import (
    DirtyWorkingTreeError,
    get_or_create_experiment,
    make_run_name,
    start_run,
    tracking_uri,
    with_retry,
)


def test_with_retry_recovers_from_locked_database(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tracking.time, "sleep", lambda _: None)
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("(sqlite3.OperationalError) database is locked")
        return "ok"

    assert with_retry(flaky) == "ok"
    assert calls["n"] == 3


def test_with_retry_does_not_hide_other_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tracking.time, "sleep", lambda _: None)

    def broken() -> None:
        raise ValueError("bad input")

    with pytest.raises(ValueError, match="bad input"):
        with_retry(broken)


def test_tracking_uri_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "")
    monkeypatch.setattr(tracking, "load_dotenv", lambda **_: None)
    with pytest.raises(RuntimeError, match="MLFLOW_TRACKING_URI"):
        tracking_uri()


def test_run_names_are_descriptive(tiny_cfg: Config) -> None:
    assert make_run_name(tiny_cfg, "cnn").startswith("cnn_lr0.001_bs16_do0.3_w1.28_all_data-v2_s42")
    assert make_run_name(tiny_cfg, "baseline").startswith("baseline-rf_n10")


def test_get_or_create_experiment_is_idempotent(mlflow_store: Path) -> None:
    assert get_or_create_experiment("idempotent") == get_or_create_experiment("idempotent")


def test_start_run_logs_config_and_lineage(mlflow_store: Path, tiny_cfg: Config) -> None:
    with start_run(tiny_cfg, stage="unit", run_name="unit-run", allow_dirty=True) as tracker:
        tracker.log_metrics({"m": 1.0})
        run_id = tracker.run_id
    run = mlflow.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.params["train.lr"] == "0.001"
    assert run.data.params["split.test_subjects"] == "[104]"
    for tag in ("git.commit", "git.dirty", "dvc.data_version", "seed", "reproduce.command", "config.hash"):
        assert tag in run.data.tags
    artifacts = {a.path for a in mlflow.MlflowClient().list_artifacts(run_id, "config")}
    assert "config/resolved_config.json" in artifacts


def test_failed_block_marks_run_failed(mlflow_store: Path, tiny_cfg: Config) -> None:
    with (
        pytest.raises(ZeroDivisionError),
        start_run(tiny_cfg, "unit", "failing", allow_dirty=True) as tracker,
    ):
        run_id = tracker.run_id
        _ = 1 / 0
    assert mlflow.get_run(run_id).info.status == "FAILED"


def test_dirty_tree_is_refused(monkeypatch: pytest.MonkeyPatch, mlflow_store: Path, tiny_cfg: Config) -> None:
    monkeypatch.setattr(tracking, "git_info", lambda: {"commit": "abc", "branch": "x", "dirty": "true"})
    with pytest.raises(DirtyWorkingTreeError), start_run(tiny_cfg, "unit", "dirty"):
        pass
