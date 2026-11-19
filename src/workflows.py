"""Multi-step workflows built from the pipeline pieces.

* :func:`run_evaluate`  - the single, final test evaluation of a model bundle.
* :func:`run_loso`      - leave-one-subject-out robustness evaluation.
* :func:`start_sweep`   - parent run of one HPO round (children are Slurm array tasks).
* :func:`verify_repro`  - re-run a tracked run and assert identical results.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import numpy as np

from src.config import Config, load_config
from src.evaluate import evaluate_predictions, log_evaluation
from src.models.bundle import load_bundle
from src.split import loso_folds
from src.tracking import download_run_artifact, fetch_run, start_run, with_retry
from src.utils import get_logger, git_info, kv, write_json

logger = get_logger(__name__)

TIME_METRIC_MARKERS: tuple[str, ...] = ("time", "seconds")
HISTORY_METRICS: tuple[str, ...] = ("train_loss", "val_loss", "val_macro_f1", "train_loss_step")


def run_evaluate(
    model_dir: Path, report_path: Path | None = None, allow_dirty: bool = False
) -> dict[str, float]:
    """Evaluate a bundle on the test split once and log into its training run.

    The bundle's own config (data version, split, windows, sensors) is used,
    and the recomputed normalisation statistics must equal the stored ones,
    which proves the test windows were built exactly like during training.

    Args:
        model_dir: Bundle folder (e.g. ``models/cnn``).
        report_path: Optional JSON file for DVC metrics.
        allow_dirty: Allow uncommitted code.

    Returns:
        Test metrics (prefixed ``test_``).

    Raises:
        RuntimeError: If the normalisation statistics differ.
    """
    from src.data_versions import resolve_clean_dir
    from src.windows import load_clean, prepare_data

    predictor = load_bundle(model_dir)
    cfg = Config.model_validate(predictor.meta.config)
    clean, meta = load_clean(resolve_clean_dir(cfg))
    data = prepare_data(cfg, clean, meta["channels"], splits_needed=("test",))  # type: ignore[arg-type]
    if not (np.array_equal(data.norm_mean, predictor.mean) and np.array_equal(data.norm_std, predictor.std)):
        raise RuntimeError("normalisation statistics differ from the bundle: data or split changed")

    test = data.splits["test"]
    result = evaluate_predictions(test.y, predictor.predict(test.x), test.subjects, data.class_names)
    metrics = {f"test_{k}": v for k, v in result.metrics.items()}
    with start_run(
        cfg,
        stage="evaluate",
        run_name="",
        allow_dirty=allow_dirty,
        run_id=predictor.meta.run_id,
        log_config=False,
        tag_prefix="evaluate.",
    ) as tracker:
        log_evaluation(tracker, result, "test", data.class_names)
        tracker.set_tags(
            {"test.evaluated": "true", "test.subjects": ",".join(map(str, data.split_subjects["test"]))}
        )
    if report_path is not None:
        write_json(Path(report_path), {"run_id": predictor.meta.run_id, **metrics})
    logger.info(
        "test evaluation done", extra=kv(run_id=predictor.meta.run_id, test_macro_f1=metrics["test_macro_f1"])
    )
    return metrics


def run_loso(
    cfg: Config, kind: str, allow_dirty: bool = False, subjects: list[int] | None = None
) -> dict[str, Any]:
    """Leave-one-subject-out evaluation of one fixed configuration.

    Every fold is a nested child run; the parent run gets mean/std/min of the
    held-out macro-F1 and a per-subject table.

    Args:
        cfg: Final configuration to evaluate.
        kind: ``baseline`` or ``deep`` (model from ``cfg.model.name``).
        allow_dirty: Allow uncommitted code.
        subjects: Subjects to rotate (default: all protocol subjects 101-109).

    Returns:
        Summary with per-fold scores.
    """
    from src.constants import SUBJECT_IDS
    from src.train import run_baseline, run_training

    folds = loso_folds(subjects or list(SUBJECT_IDS))
    label = "baseline" if kind == "baseline" else cfg.model.name
    scores: list[dict[str, float]] = []
    with start_run(
        cfg, stage="loso", run_name=f"loso-{label}", allow_dirty=allow_dirty, extra_tags={"loso.kind": label}
    ) as parent:
        for fold in folds:
            held_out = fold.test_subjects[0]
            fold_cfg = cfg.model_copy(
                update={"split": fold, "train": cfg.train.model_copy(update={"eval_test": True})}
            )
            tags = {"loso.held_out": held_out, "loso.val_subject": fold.val_subjects[0]}
            if kind == "baseline":
                outcome = run_baseline(fold_cfg, allow_dirty, parent.run_id, split=fold, extra_tags=tags)
            else:
                outcome = run_training(fold_cfg, allow_dirty, parent.run_id, extra_tags=tags)
            score = outcome.metrics["test_macro_f1"]
            scores.append(
                {
                    "subject_id": held_out,
                    "test_macro_f1": score,
                    "test_accuracy": outcome.metrics["test_accuracy"],
                }
            )
            parent.log_metrics({"fold_test_macro_f1": score}, step=held_out)
            logger.info("loso fold done", extra=kv(held_out=held_out, test_macro_f1=round(score, 4)))
        values = np.array([s["test_macro_f1"] for s in scores])
        summary = {
            "loso_macro_f1_mean": float(values.mean()),
            "loso_macro_f1_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "loso_macro_f1_min": float(values.min()),
            "loso_macro_f1_max": float(values.max()),
        }
        parent.log_metrics(summary)
        parent.log_dict(scores, "reports/loso_per_subject.json")
    return {"folds": scores, **summary}


def start_sweep(cfg: Config, name: str, hypothesis: str, allow_dirty: bool = False) -> str:
    """Create the parent run of an HPO round and return its id.

    Child runs (one per Slurm array task) attach to it with ``--parent-run-id``.

    Args:
        cfg: Base config of the round.
        name: Round name, e.g. ``round-a-lr``.
        hypothesis: What the round tests (logged as a tag, mirrored in docs/experiments.md).
        allow_dirty: Allow uncommitted code.

    Returns:
        The parent run id.
    """
    with start_run(
        cfg,
        stage="hpo",
        run_name=name,
        allow_dirty=allow_dirty,
        extra_tags={"hpo.round": name, "hpo.hypothesis": hypothesis},
    ) as tracker:
        run_id = tracker.run_id
    # start_run ends the run as FINISHED; children keep attaching to it.
    return run_id


def _comparable(metrics: dict[str, float]) -> dict[str, float]:
    """Drop wall-clock metrics, which legitimately differ between identical runs."""
    return {k: v for k, v in metrics.items() if not any(m in k for m in TIME_METRIC_MARKERS)}


def _history(run_id: str, key: str) -> list[tuple[int, float]]:
    """Return a metric's full ``(step, value)`` history."""
    client = mlflow.MlflowClient()
    return [(m.step, m.value) for m in with_retry(client.get_metric_history, run_id, key)]


def compare_runs(original_id: str, rerun_id: str) -> dict[str, Any]:
    """Compare final metrics, metric histories and weight hashes of two runs.

    Args:
        original_id: The run being verified.
        rerun_id: The re-run.

    Returns:
        Report with ``passed`` and every mismatch.
    """
    original, rerun = fetch_run(original_id), fetch_run(rerun_id)
    a, b = _comparable(original.data.metrics), _comparable(rerun.data.metrics)
    mismatches: list[dict[str, Any]] = []
    for key in sorted(a):
        if key not in b:
            mismatches.append({"metric": key, "original": a[key], "rerun": "missing"})
        elif not (a[key] == b[key] or (math.isnan(a[key]) and math.isnan(b[key]))):
            mismatches.append({"metric": key, "original": a[key], "rerun": b[key]})
    histories_checked = []
    for key in HISTORY_METRICS:
        if key in a:
            histories_checked.append(key)
            if _history(original_id, key) != _history(rerun_id, key):
                mismatches.append({"metric_history": key})
    hash_a = original.data.tags.get("model.weights_sha256")
    hash_b = rerun.data.tags.get("model.weights_sha256")
    if hash_a != hash_b:
        mismatches.append({"weights_sha256": {"original": hash_a, "rerun": hash_b}})
    return {
        "original_run_id": original_id,
        "rerun_run_id": rerun_id,
        "metrics_compared": len(a),
        "histories_compared": histories_checked,
        "weights_sha256": {"original": hash_a, "rerun": hash_b},
        "mismatches": mismatches,
        "passed": not mismatches,
    }


def verify_repro(run_id: str) -> dict[str, Any]:
    """Re-run a tracked run from its logged config and assert identical results.

    The current checkout must be the run's ``git.commit`` and the working tree
    must be clean; the data version comes from the logged config (fetched via
    DVC if needed). If the original was evaluated on test, the re-run is too.

    Args:
        run_id: Run to verify.

    Returns:
        The comparison report (also logged to the re-run and tagged on both runs).

    Raises:
        RuntimeError: On a commit mismatch.
    """
    from src.train import run_baseline, run_training

    original = fetch_run(run_id)
    tags = original.data.tags
    head = git_info()["commit"]
    if tags.get("git.commit") != head:
        raise RuntimeError(
            f"run was made at commit {tags.get('git.commit')}, HEAD is {head}; check it out first"
        )
    with tempfile.TemporaryDirectory() as tmp:
        cfg_file = download_run_artifact(run_id, "config/resolved_config.json", Path(tmp))
        cfg = load_config(cfg_file)

    extra = {"verify_of": run_id}
    stage = tags.get("stage")
    if stage == "baseline":
        outcome = run_baseline(cfg, extra_tags=extra)
    elif stage == "train":
        outcome = run_training(cfg, extra_tags=extra, fresh=True)
        if tags.get("test.evaluated") == "true" and not cfg.train.eval_test:
            run_evaluate(outcome.bundle_dir)
    else:
        raise RuntimeError(f"cannot verify a run with stage={stage!r}")

    report = compare_runs(run_id, outcome.run_id)
    client = mlflow.MlflowClient()
    with_retry(client.set_tag, outcome.run_id, "verify.passed", str(report["passed"]).lower())
    with_retry(client.set_tag, run_id, "verify.rerun_id", outcome.run_id)
    with_retry(client.set_tag, run_id, "verify.passed", str(report["passed"]).lower())
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "verify_report.json"
        write_json(path, report)
        with_retry(client.log_artifact, outcome.run_id, str(path), "reports")
    logger.info(
        "verify-repro finished", extra=kv(passed=report["passed"], mismatches=len(report["mismatches"]))
    )
    return report
