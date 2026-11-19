"""All MLflow interaction in one place.

* The tracking URI comes **only** from ``MLFLOW_TRACKING_URI`` (``.env`` or the job
  script); nothing is hard-coded, so switching to the university server is a
  config change.
* Every run automatically logs the fully resolved config (flattened params plus
  ``params.yaml`` / ``resolved_config.json`` artifacts) and lineage tags that link
  the run to the exact code, data and hardware.
* Writes are retried with back-off because the interim store is SQLite on a
  shared disk (``database is locked`` under concurrent jobs).
"""

from __future__ import annotations

import os
import platform
import shlex
import socket
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
from dotenv import load_dotenv
from matplotlib.figure import Figure
from mlflow.entities import Run

from src.config import DEFAULT_CONFIG_PATH, Config, config_hash, config_to_dict
from src.utils import (
    dvc_file_hash,
    dvc_lock_outs,
    flatten_dict,
    get_logger,
    git_info,
    git_show,
    kv,
    log_context,
    parse_dvc_lock,
    write_json,
)

logger = get_logger(__name__)

MAX_PARAMS_PER_BATCH = 100
_RETRYABLE = ("database is locked", "database table is locked", "disk i/o error")


class DirtyWorkingTreeError(RuntimeError):
    """Raised when a tracked run is started from uncommitted code without ``--allow-dirty``."""


def tracking_uri() -> str:
    """Return the tracking URI from the environment (``.env`` is loaded first).

    Returns:
        The URI.

    Raises:
        RuntimeError: If ``MLFLOW_TRACKING_URI`` is not set.
    """
    load_dotenv(override=False)
    uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if not uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI is not set. Copy .env.example to .env or export it "
            "(the Slurm scripts set it)."
        )
    return uri


def experiment_name(cfg: Config) -> str:
    """Return the experiment name (``MLFLOW_EXPERIMENT`` overrides the config).

    Args:
        cfg: Resolved config.

    Returns:
        Experiment name.
    """
    load_dotenv(override=False)
    return os.environ.get("MLFLOW_EXPERIMENT") or cfg.tracking.experiment


def with_retry[T](
    fn: Callable[..., T], *args: Any, attempts: int = 8, base_delay: float = 0.5, **kwargs: Any
) -> T:
    """Call ``fn`` and retry with exponential back-off on SQLite lock errors.

    Args:
        fn: Function to call.
        *args: Positional arguments for ``fn``.
        attempts: Maximum number of attempts.
        base_delay: First back-off delay in seconds (doubled each time, capped at 30 s).
        **kwargs: Keyword arguments for ``fn``.

    Returns:
        Whatever ``fn`` returns.
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            retryable = any(token in str(exc).lower() for token in _RETRYABLE)
            if not retryable or attempt == attempts:
                raise
            delay = min(base_delay * 2 ** (attempt - 1), 30.0)
            logger.warning("mlflow store busy, retrying", extra=kv(attempt=attempt, delay_s=delay))
            time.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


def get_or_create_experiment(name: str, artifact_location: str | None = None) -> str:
    """Return the id of an experiment, creating it (idempotently) if needed.

    Args:
        name: Experiment name.
        artifact_location: Artifact root for a new experiment (``MLFLOW_ARTIFACT_ROOT``).

    Returns:
        The experiment id.
    """
    mlflow.set_tracking_uri(tracking_uri())
    existing = with_retry(mlflow.get_experiment_by_name, name)
    if existing is not None:
        return existing.experiment_id
    try:
        return with_retry(mlflow.create_experiment, name, artifact_location=artifact_location)
    except mlflow.exceptions.MlflowException:
        # Another job created it in the meantime.
        found = mlflow.get_experiment_by_name(name)
        if found is None:
            raise
        return found.experiment_id


def gpu_tags() -> dict[str, str]:
    """Describe the accelerator and software stack (determinism depends on them).

    Returns:
        Tags such as ``gpu.name``, ``cuda.version``, ``torch.version``.
    """
    import torch

    tags = {
        "torch.version": torch.__version__,
        "cuda.version": str(torch.version.cuda),
        "cudnn.version": str(torch.backends.cudnn.version()),
        "python.version": platform.python_version(),
        "gpu.name": "cpu",
    }
    if torch.cuda.is_available():
        tags["gpu.name"] = torch.cuda.get_device_name(0)
        try:
            import subprocess

            tags["gpu.driver"] = (
                subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30,
                )
                .stdout.strip()
                .splitlines()[0]
            )
        except (OSError, subprocess.SubprocessError, IndexError):
            tags["gpu.driver"] = "unknown"
    return tags


def reproduce_command(commit: str, argv: list[str] | None = None) -> str:
    """Build the shell command that re-creates a run from scratch.

    Args:
        commit: Git commit of the code.
        argv: CLI arguments after ``python -m src.main`` (default: current ``sys.argv``).

    Returns:
        A copy-pasteable command.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    return f"git checkout {commit} && dvc pull && python -m src.main {shlex.join(args)}".rstrip()


def lineage_tags(cfg: Config, allow_dirty: bool, data_dir: Path | None = None) -> dict[str, str]:
    """Collect the tags that link a run to code, data, split and hardware.

    Args:
        cfg: Resolved config.
        allow_dirty: Whether uncommitted changes are tolerated.
        data_dir: Clean-data folder actually used (for its DVC hash).

    Returns:
        Tag mapping.

    Raises:
        DirtyWorkingTreeError: If the tree is dirty and ``allow_dirty`` is False.
    """
    git = git_info()
    if git["dirty"] == "true" and not allow_dirty:
        raise DirtyWorkingTreeError(
            "uncommitted changes in tracked files; commit first or pass --allow-dirty"
        )
    clean_key = Path(cfg.paths.clean_dir).as_posix()
    from_workspace = data_dir is None or Path(data_dir) == Path(cfg.paths.clean_dir)
    if from_workspace:
        data_rev, outs = "workspace", dvc_lock_outs()
    else:
        from src.data_versions import data_tag

        data_rev = data_tag(cfg.data.version)
        outs = parse_dvc_lock(git_show(data_rev, "dvc.lock") or "")
    tags = {
        "git.commit": git["commit"],
        "git.branch": git["branch"],
        "git.dirty": git["dirty"],
        "dvc.data_version": cfg.data.version,
        "dvc.data_rev": data_rev,
        "dvc.raw_hash": dvc_file_hash(Path(f"{Path(cfg.paths.raw_dir).as_posix()}.dvc")),
        "dvc.data_hash": outs.get(clean_key, "unknown"),
        "dvc.data_dir": str(data_dir or cfg.paths.clean_dir),
        "data.zip_sha256": cfg.dataset.sha256 or "unpinned",
        "split.val_subjects": ",".join(map(str, cfg.split.val_subjects)),
        "split.test_subjects": ",".join(map(str, cfg.split.test_subjects)),
        "config.hash": config_hash(cfg),
        "seed": str(cfg.seed),
        "slurm.job_id": os.environ.get("SLURM_JOB_ID", "none"),
        "slurm.array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", "none"),
        "slurm.node": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
        "reproduce.command": reproduce_command(git["commit"]),
        "env.cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "unset"),
        "env.pythonhashseed": os.environ.get("PYTHONHASHSEED", "unset"),
    }
    return tags


class RunTracker:
    """Thin, retrying wrapper around the active MLflow run."""

    def __init__(self, run: Run) -> None:
        """Wrap a started run.

        Args:
            run: The active MLflow run.
        """
        self.run = run
        self.run_id: str = run.info.run_id

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Log parameters in batches (MLflow accepts at most 100 per call).

        Args:
            params: Flat parameter mapping.
        """
        items = [(k, "None" if v is None else str(v)) for k, v in sorted(params.items())]
        for i in range(0, len(items), MAX_PARAMS_PER_BATCH):
            with_retry(mlflow.log_params, dict(items[i : i + MAX_PARAMS_PER_BATCH]))

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Log numeric metrics.

        Args:
            metrics: Name -> value.
            step: Optional step (epoch or global step).
        """
        clean = {k: float(v) for k, v in metrics.items()}
        with_retry(mlflow.log_metrics, clean, step=step)

    def set_tags(self, tags: Mapping[str, Any]) -> None:
        """Set run tags.

        Args:
            tags: Name -> value (converted to strings).
        """
        with_retry(mlflow.set_tags, {k: str(v) for k, v in tags.items()})

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        """Log a file or directory.

        Args:
            path: Local file or folder.
            artifact_path: Destination folder inside the run's artifacts.
        """
        path = Path(path)
        fn = mlflow.log_artifacts if path.is_dir() else mlflow.log_artifact
        with_retry(fn, str(path), artifact_path)

    def log_dict(self, obj: Any, name: str) -> None:
        """Log a JSON-serialisable object as an artifact file.

        Args:
            obj: Object to store.
            name: Artifact file path, e.g. ``reports/val_report.json``.
        """
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / Path(name).name
            write_json(target, obj)
            parent = str(Path(name).parent) if Path(name).parent != Path(".") else None
            self.log_artifact(target, parent)

    def log_text(self, text: str, name: str) -> None:
        """Log a text artifact.

        Args:
            text: Content.
            name: Artifact file path.
        """
        with_retry(mlflow.log_text, text, name)

    def log_figure(self, fig: Figure, name: str) -> None:
        """Log a matplotlib figure and close it.

        Args:
            fig: Figure to store.
            name: Artifact file path, e.g. ``figures/confusion_val.png``.
        """
        import matplotlib.pyplot as plt

        with_retry(mlflow.log_figure, fig, name)
        plt.close(fig)


def make_run_name(cfg: Config, kind: str) -> str:
    """Build a descriptive run name, e.g. ``cnn_lr0.001_bs64_w5.12_all_data-v2_s42``.

    Args:
        cfg: Resolved config.
        kind: ``baseline``, ``cnn``, ``cnn_gru``, ``eda``, ...

    Returns:
        The run name (``cfg.tracking.run_name`` wins if set).
    """
    if cfg.tracking.run_name:
        return cfg.tracking.run_name
    suffix = f"w{cfg.windows.window_seconds:g}_{cfg.data.sensor_subset}_data-{cfg.data.version}_s{cfg.seed}"
    if kind == "baseline":
        b = cfg.baseline
        name = "rf" if b.model == "random_forest" else "logreg"
        detail = f"n{b.n_estimators}_d{b.max_depth}" if b.model == "random_forest" else f"C{b.lr_c:g}"
        return f"baseline-{name}_{detail}_{suffix}"
    if kind in {"cnn", "cnn_gru"}:
        t = cfg.train
        return f"{kind}_lr{t.lr:g}_bs{t.batch_size}_do{cfg.model.dropout:g}_{suffix}"
    return f"{kind}_data-{cfg.data.version}"


@contextmanager
def start_run(
    cfg: Config,
    stage: str,
    run_name: str,
    allow_dirty: bool = False,
    parent_run_id: str | None = None,
    run_id: str | None = None,
    extra_tags: Mapping[str, Any] | None = None,
    data_dir: Path | None = None,
    log_config: bool = True,
    tag_prefix: str = "",
) -> Iterator[RunTracker]:
    """Start (or resume) a fully tagged MLflow run.

    On entry the resolved config is logged as flattened params and as
    artifacts, and lineage tags are set. Log lines inside the block carry
    ``run_id`` and ``stage``. On exit the Slurm log (if any) is attached.

    Args:
        cfg: Resolved config.
        stage: Pipeline stage tag (``eda``, ``clean``, ``baseline``, ``train``, ...).
        run_name: Display name.
        allow_dirty: Allow uncommitted code (the run is tagged ``git.dirty=true``).
        parent_run_id: Parent run for nested HPO children.
        run_id: Resume this run instead of creating a new one.
        extra_tags: Additional tags.
        data_dir: Clean-data folder actually used.
        log_config: Log params and config artifacts (skip when resuming).
        tag_prefix: Prefix for lineage/stage tags, e.g. ``evaluate.`` when a later
            step re-opens a run, so the training lineage is not overwritten.

    Yields:
        The run tracker.
    """
    tags = {
        f"{tag_prefix}{k}": v for k, v in {**lineage_tags(cfg, allow_dirty, data_dir), "stage": stage}.items()
    }
    exp_id = get_or_create_experiment(experiment_name(cfg), os.environ.get("MLFLOW_ARTIFACT_ROOT") or None)
    run = with_retry(
        mlflow.start_run,
        run_id=run_id,
        experiment_id=None if run_id else exp_id,
        run_name=None if run_id else run_name,
        parent_run_id=parent_run_id if not run_id else None,
        nested=parent_run_id is not None,
    )
    tracker = RunTracker(run)
    status = "FAILED"
    try:
        with log_context(run_id=tracker.run_id, stage=stage):
            tracker.set_tags({**tags, **(extra_tags or {})})
            if log_config:
                tracker.log_params(flatten_dict(config_to_dict(cfg)))
                tracker.log_dict(config_to_dict(cfg), "config/resolved_config.json")
                if DEFAULT_CONFIG_PATH.exists():
                    tracker.log_artifact(DEFAULT_CONFIG_PATH, "config")
            logger.info("mlflow run started", extra=kv(name=run_name, experiment=experiment_name(cfg)))
            yield tracker
            status = "FINISHED"
    finally:
        _attach_slurm_log(tracker)
        with_retry(mlflow.end_run, status=status)


def _attach_slurm_log(tracker: RunTracker) -> None:
    """Attach the Slurm log file named by ``SLURM_LOG_FILE`` (set by the job scripts)."""
    log_file = os.environ.get("SLURM_LOG_FILE")
    if log_file and Path(log_file).is_file():
        try:
            tracker.log_artifact(Path(log_file), "logs")
        except Exception as exc:  # the run result matters more than its log
            logger.warning("could not attach slurm log", extra=kv(error=str(exc)))


def fetch_run(run_id: str) -> Run:
    """Load a run by id from the configured store.

    Args:
        run_id: MLflow run id.

    Returns:
        The run.
    """
    mlflow.set_tracking_uri(tracking_uri())
    return with_retry(mlflow.get_run, run_id)


def download_run_artifact(run_id: str, artifact_path: str, dst: Path) -> Path:
    """Download one artifact of a run.

    Args:
        run_id: MLflow run id.
        artifact_path: Path inside the run's artifacts.
        dst: Local destination folder.

    Returns:
        Local path of the downloaded artifact.
    """
    mlflow.set_tracking_uri(tracking_uri())
    dst.mkdir(parents=True, exist_ok=True)
    return Path(
        with_retry(
            mlflow.artifacts.download_artifacts, run_id=run_id, artifact_path=artifact_path, dst_path=str(dst)
        )
    )
