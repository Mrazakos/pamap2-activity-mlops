"""Integration tests: the Typer CLI end to end on synthetic data and a throwaway MLflow store."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pytest
from typer.testing import CliRunner

from src.main import app
from src.models.bundle import load_bundle
from src.utils import read_json
from tests.conftest import write_raw

pytestmark = pytest.mark.integration
runner = CliRunner()


def _invoke(*args: str) -> dict[str, object]:
    """Run a CLI command, assert success and parse its JSON output."""
    result = runner.invoke(app, list(args), catch_exceptions=False)
    assert result.exit_code == 0, result.output
    text = result.stdout.strip()
    return json.loads(text[text.index("{") :]) if "{" in text else {"output": text}


def _sets(overrides: list[str]) -> list[str]:
    """Turn override strings into repeated ``--set`` options."""
    return [arg for o in overrides for arg in ("--set", o)]


def test_pipeline_end_to_end(tmp_path: Path, tiny_overrides: list[str], mlflow_store: Path) -> None:
    write_raw(tmp_path / "raw_dir")
    sets = _sets(tiny_overrides)

    dq = _invoke("clean", *sets, "--allow-dirty")
    assert dq["transient_rows_dropped"] > 0 and dq["duplicate_timestamps_dropped"] == 4  # type: ignore[operator]

    pre = _invoke("preprocess", *sets)
    assert pre["split_subjects"] == {"train": [101, 102], "val": [103], "test": [104]}
    assert (tmp_path / "processed_dir" / "windows.parquet").exists()

    feats = _invoke("features", *sets)
    assert feats["n_features"] > 100  # type: ignore[operator]

    base = _invoke(
        "baseline", *sets, "--allow-dirty", "--output-dir", str(tmp_path / "models_dir" / "baseline")
    )
    assert 0.0 <= base["best_val_macro_f1"] <= 1.0  # type: ignore[operator]

    trained = _invoke("train", *sets, "--allow-dirty", "--output-dir", str(tmp_path / "models_dir" / "cnn"))
    run = mlflow.get_run(str(trained["run_id"]))
    assert run.data.params["model.name"] == "cnn"
    assert {"train_loss", "val_loss", "val_macro_f1", "best_val_macro_f1", "train_loss_step"} <= set(
        run.data.metrics
    )
    assert run.data.tags["model.weights_sha256"] == trained["weights_sha256"]
    artifacts = {a.path for a in mlflow.MlflowClient().list_artifacts(run.info.run_id)}
    assert {"bundle", "figures", "reports", "config"} <= artifacts
    history = mlflow.MlflowClient().get_metric_history(run.info.run_id, "val_macro_f1")
    assert [m.step for m in history] == [0, 1]

    report = tmp_path / "cnn_test.json"
    test_metrics = _invoke(
        "evaluate",
        "--model-dir",
        str(tmp_path / "models_dir" / "cnn"),
        "--report",
        str(report),
        "--allow-dirty",
    )
    assert "test_macro_f1" in test_metrics
    assert read_json(report)["run_id"] == trained["run_id"]
    assert "test_macro_f1" in mlflow.get_run(str(trained["run_id"])).data.metrics

    bundle = load_bundle(tmp_path / "models_dir" / "cnn")
    assert bundle.meta.run_id == trained["run_id"]


def test_training_is_bit_reproducible(tmp_path: Path, tiny_overrides: list[str], mlflow_store: Path) -> None:
    write_raw(tmp_path / "raw_dir")
    sets = _sets(tiny_overrides)
    _invoke("clean", *sets, "--no-track")
    first = _invoke(
        "train", *sets, "--allow-dirty", "--set", "model.name=cnn_gru", "--output-dir", str(tmp_path / "a")
    )
    # The completed checkpoint folder is discarded, so this is a genuine re-run.
    second = _invoke(
        "train", *sets, "--allow-dirty", "--set", "model.name=cnn_gru", "--output-dir", str(tmp_path / "b")
    )
    assert first["run_id"] != second["run_id"]
    assert first["weights_sha256"] == second["weights_sha256"]
    assert first["best_val_macro_f1"] == second["best_val_macro_f1"]


def test_sweep_parent_and_child(tmp_path: Path, tiny_overrides: list[str], mlflow_store: Path) -> None:
    write_raw(tmp_path / "raw_dir")
    sets = _sets(tiny_overrides)
    _invoke("clean", *sets, "--no-track")
    parent = _invoke("sweep-init", "--name", "round-test", "--hypothesis", "smoke", *sets, "--allow-dirty")[
        "output"
    ]
    child = _invoke(
        "train", *sets, "--allow-dirty", "--parent-run-id", str(parent), "--set", "train.epochs=1"
    )
    assert mlflow.get_run(str(child["run_id"])).data.tags["mlflow.parentRunId"] == parent


def test_eda_logs_summary(tmp_path: Path, tiny_overrides: list[str], mlflow_store: Path) -> None:
    write_raw(tmp_path / "raw_dir")
    out = _invoke("eda", *_sets(tiny_overrides), "--allow-dirty")
    assert out["total_rows"] > 0  # type: ignore[operator]
    summary = read_json(tmp_path / "reports_dir" / "eda" / "eda_summary.json")
    assert "nan_rate_per_column" in summary and "subjects_missing_protocol_activities" in summary


@pytest.fixture
def clean_git(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend the working tree is clean at a fixed commit (CI has a real clean checkout)."""
    from src import tracking, workflows

    info = {"commit": "0" * 40, "branch": "test", "dirty": "false"}
    monkeypatch.setattr(tracking, "git_info", lambda: info)
    monkeypatch.setattr(workflows, "git_info", lambda: info)


@pytest.mark.parametrize("command", ["baseline", "train"])
def test_verify_repro_passes_for_identical_rerun(
    tmp_path: Path, tiny_overrides: list[str], mlflow_store: Path, clean_git: None, command: str
) -> None:
    write_raw(tmp_path / "raw_dir")
    sets = _sets(tiny_overrides)
    _invoke("clean", *sets, "--no-track")
    if command == "baseline":
        _invoke("preprocess", *sets)
        _invoke("features", *sets)
    original = _invoke(command, *sets)
    report = _invoke("verify-repro", str(original["run_id"]))
    assert report["passed"] is True and report["mismatches"] == []
    assert report["metrics_compared"] > 10  # type: ignore[operator]
    assert mlflow.get_run(str(original["run_id"])).data.tags["verify.passed"] == "true"


def test_verify_repro_detects_a_difference(
    tmp_path: Path, tiny_overrides: list[str], mlflow_store: Path, clean_git: None
) -> None:
    from src.workflows import compare_runs

    write_raw(tmp_path / "raw_dir")
    sets = _sets(tiny_overrides)
    _invoke("clean", *sets, "--no-track")
    a = _invoke("train", *sets, "--set", "train.epochs=1")
    b = _invoke("train", *sets, "--set", "train.epochs=1", "--set", "seed=7")
    report = compare_runs(str(a["run_id"]), str(b["run_id"]))
    assert report["passed"] is False
    assert any("weights_sha256" in m for m in report["mismatches"])


def test_loso_baseline(tmp_path: Path, tiny_overrides: list[str], mlflow_store: Path) -> None:
    write_raw(tmp_path / "raw_dir")
    sets = _sets(tiny_overrides)
    _invoke("clean", *sets, "--no-track")
    from src.config import load_config
    from src.workflows import run_loso

    summary = run_loso(
        load_config(overrides=tiny_overrides), "baseline", allow_dirty=True, subjects=[101, 102, 103, 104]
    )
    assert [f["subject_id"] for f in summary["folds"]] == [101, 102, 103, 104]
    assert 0.0 <= summary["loso_macro_f1_min"] <= summary["loso_macro_f1_mean"] <= 1.0


def test_bad_override_fails_cleanly() -> None:
    result = runner.invoke(app, ["preprocess", "--set", "train.nope=1"])
    assert result.exit_code != 0
