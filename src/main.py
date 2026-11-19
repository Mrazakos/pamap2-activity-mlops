"""Command-line interface: ``python -m src.main <command> [--set key=value ...]``.

Every command reads ``configs/params.yaml`` (or ``--config``) and accepts dotted
overrides, so any tracked run can be reproduced from its logged command.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from src.config import DEFAULT_CONFIG_PATH, Config, load_config
from src.utils import get_logger, kv, log_context

app = typer.Typer(add_completion=False, no_args_is_help=True, help="PAMAP2 activity recognition pipeline.")
logger = get_logger("src.main")

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="YAML config file.")]
SetOpt = Annotated[
    list[str] | None, typer.Option("--set", "-s", help="Override, e.g. train.lr=3e-4 (repeatable).")
]
DirtyOpt = Annotated[
    bool, typer.Option("--allow-dirty", help="Allow uncommitted code (run is tagged git.dirty).")
]
ParentOpt = Annotated[str | None, typer.Option("--parent-run-id", help="Nest the run under this MLflow run.")]
OutOpt = Annotated[Path | None, typer.Option("--output-dir", help="Where to write the model bundle.")]


def _cfg(config: Path, overrides: list[str] | None) -> Config:
    """Load the config with CLI overrides."""
    return load_config(config, overrides or [])


def _echo_json(obj: object) -> None:
    """Print a JSON summary to stdout (the CLI's machine-readable result)."""
    typer.echo(json.dumps(obj, indent=2, sort_keys=True, default=str))


@app.command()
def download(config: ConfigOpt = DEFAULT_CONFIG_PATH, overrides: SetOpt = None) -> None:
    """Download the archive, verify its SHA-256 and convert recordings to Parquet."""
    from src.download import run_download

    cfg = _cfg(config, overrides)
    with log_context(stage="download"):
        manifest = run_download(
            cfg.dataset.url, cfg.dataset.sha256, cfg.paths.zip_path, cfg.paths.extract_dir, cfg.paths.raw_dir
        )
    _echo_json({"zip_sha256": manifest["zip_sha256"], "recordings": len(manifest["recordings"])})  # type: ignore[arg-type]


@app.command()
def eda(
    config: ConfigOpt = DEFAULT_CONFIG_PATH, overrides: SetOpt = None, allow_dirty: DirtyOpt = False
) -> None:
    """Compute EDA aggregates and figures and log them to MLflow (stage=eda)."""
    from src.eda import run_eda

    with log_context(stage="eda"):
        summary = run_eda(_cfg(config, overrides), allow_dirty=allow_dirty)
    _echo_json(
        {"total_rows": summary["total_rows"], "missing": summary["subjects_missing_protocol_activities"]}
    )


@app.command()
def clean(
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
    overrides: SetOpt = None,
    allow_dirty: DirtyOpt = False,
    track: Annotated[bool, typer.Option(help="Log data-quality counts to MLflow.")] = True,
) -> None:
    """Apply the data-quality rules of ``data.version`` (DVC stage ``clean``)."""
    from src.clean import run_clean

    cfg = _cfg(config, overrides)
    with log_context(stage="clean", version=cfg.data.version):
        meta = run_clean(cfg)
        totals = meta["quality"]["totals"]  # type: ignore[index]
        if track:
            from src.tracking import start_run

            with start_run(
                cfg, stage="clean", run_name=f"clean_data-{cfg.data.version}", allow_dirty=allow_dirty
            ) as tracker:
                tracker.log_metrics({f"dq_{k}": v for k, v in totals.items()})  # type: ignore[union-attr]
                tracker.log_dict(meta["quality"], "reports/data_quality.json")
    _echo_json(totals)


@app.command()
def preprocess(config: ConfigOpt = DEFAULT_CONFIG_PATH, overrides: SetOpt = None) -> None:
    """Subject-wise split, train-only normalisation and window index (DVC stage ``split_window``)."""
    from src.windows import run_preprocess

    cfg = _cfg(config, overrides)
    with log_context(stage="split_window", version=cfg.data.version):
        meta = run_preprocess(cfg)
    _echo_json({k: meta[k] for k in ("split_subjects", "windows_per_split", "missing_activities")})


@app.command()
def features(config: ConfigOpt = DEFAULT_CONFIG_PATH, overrides: SetOpt = None) -> None:
    """Compute hand-crafted window features for the baseline (DVC stage ``features``)."""
    from src.features import run_features

    cfg = _cfg(config, overrides)
    with log_context(stage="features", version=cfg.data.version):
        meta = run_features(cfg)
    _echo_json({"n_features": meta["n_features"], "channels": len(meta["channels"])})  # type: ignore[arg-type]


@app.command()
def baseline(
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
    overrides: SetOpt = None,
    allow_dirty: DirtyOpt = False,
    parent_run_id: ParentOpt = None,
    output_dir: OutOpt = None,
) -> None:
    """Train and validate the feature-based baseline (tracked in MLflow)."""
    from src.train import run_baseline

    outcome = run_baseline(_cfg(config, overrides), allow_dirty, parent_run_id, output_dir)
    _echo_json({"run_id": outcome.run_id, "best_val_macro_f1": outcome.metrics["best_val_macro_f1"]})


@app.command()
def train(
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
    overrides: SetOpt = None,
    allow_dirty: DirtyOpt = False,
    parent_run_id: ParentOpt = None,
    output_dir: OutOpt = None,
) -> None:
    """Train a deep model (CNN or CNN-GRU) with early stopping (tracked in MLflow)."""
    from src.train import run_training

    outcome = run_training(_cfg(config, overrides), allow_dirty, parent_run_id, output_dir)
    _echo_json(
        {
            "run_id": outcome.run_id,
            "best_val_macro_f1": outcome.metrics["best_val_macro_f1"],
            "weights_sha256": outcome.weights_sha256,
        }
    )


@app.command()
def evaluate(
    model_dir: Annotated[Path, typer.Option("--model-dir", help="Model bundle folder.")],
    report: Annotated[
        Path | None, typer.Option("--report", help="Write test metrics JSON here (DVC metrics).")
    ] = None,
    allow_dirty: DirtyOpt = False,
) -> None:
    """Evaluate a trained bundle on the held-out test subjects (once, for final models)."""
    from src.workflows import run_evaluate

    metrics = run_evaluate(model_dir, report, allow_dirty)
    _echo_json({k: v for k, v in metrics.items() if k in {"test_macro_f1", "test_accuracy"}})


@app.command("sweep-init")
def sweep_init(
    name: Annotated[str, typer.Option(help="Round name, e.g. round-a-lr.")],
    hypothesis: Annotated[str, typer.Option(help="What this round tests.")],
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
    overrides: SetOpt = None,
    allow_dirty: DirtyOpt = False,
) -> None:
    """Create the parent MLflow run of an HPO round and print its id."""
    from src.workflows import start_sweep

    typer.echo(start_sweep(_cfg(config, overrides), name, hypothesis, allow_dirty))


@app.command()
def loso(
    kind: Annotated[str, typer.Option(help="baseline or deep")] = "deep",
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
    overrides: SetOpt = None,
    allow_dirty: DirtyOpt = False,
) -> None:
    """Leave-one-subject-out robustness evaluation of a fixed configuration."""
    from src.workflows import run_loso

    if kind not in {"baseline", "deep"}:
        raise typer.BadParameter("kind must be 'baseline' or 'deep'")
    summary = run_loso(_cfg(config, overrides), kind, allow_dirty)
    _echo_json({k: v for k, v in summary.items() if k != "folds"})


@app.command("verify-repro")
def verify_repro(run_id: Annotated[str, typer.Argument(help="MLflow run id to re-run.")]) -> None:
    """Re-run a tracked run from its logged config and assert identical metrics and weights."""
    from src.workflows import verify_repro as _verify

    report = _verify(run_id)
    _echo_json({k: report[k] for k in ("passed", "metrics_compared", "mismatches", "rerun_run_id")})
    if not report["passed"]:
        raise typer.Exit(code=1)


@app.command("init-mlflow")
def init_mlflow(config: ConfigOpt = DEFAULT_CONFIG_PATH, overrides: SetOpt = None) -> None:
    """Create the tracking store schema and the experiment once (idempotent)."""
    import os

    from src.tracking import experiment_name, get_or_create_experiment, tracking_uri

    cfg = _cfg(config, overrides)
    exp_id = get_or_create_experiment(experiment_name(cfg), os.environ.get("MLFLOW_ARTIFACT_ROOT") or None)
    logger.info("mlflow ready", extra=kv(uri=tracking_uri(), experiment=experiment_name(cfg), id=exp_id))
    typer.echo(exp_id)


if __name__ == "__main__":
    app()
