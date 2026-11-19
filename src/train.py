"""Training entry points for the deep models and the classical baseline.

Deep models: deterministic PyTorch loop, per-step and per-epoch MLflow logging,
early stopping on validation macro-F1, an atomic checkpoint after every epoch
(so a Slurm TIMEOUT can be resumed with the same command), and a final bundle
with the best weights. The test split is *not* touched here unless
``train.eval_test`` is set; the final test evaluation is a separate step.
"""

from __future__ import annotations

import random
import shutil
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from src.config import Config, SplitConfig, config_hash, config_to_dict
from src.data_versions import resolve_clean_dir
from src.evaluate import (
    EvalResult,
    core_metrics,
    evaluate_predictions,
    learning_curve_figure,
    log_evaluation,
)
from src.models import build_model
from src.models.bundle import BundleMeta, Predictor, save_sklearn_bundle, save_torch_bundle
from src.tracking import RunTracker, gpu_tags, make_run_name, start_run
from src.utils import get_logger, kv, read_json, seed_worker, set_seed, write_json
from src.windows import PreparedData, load_clean, prepare_data

logger = get_logger(__name__)

IntArray = npt.NDArray[np.int64]


# --------------------------------------------------------------------------- helpers


@dataclass
class TrainOutcome:
    """Result of one tracked training run."""

    run_id: str
    metrics: dict[str, float]
    bundle_dir: Path
    weights_sha256: str


@dataclass
class TrainState:
    """Mutable loop state, stored in every checkpoint."""

    epoch: int = 0
    global_step: int = 0
    best_score: float = -1.0
    best_epoch: int = -1
    stale_epochs: int = 0
    stopped_early: bool = False
    history: list[dict[str, float]] = field(default_factory=list)


@dataclass
class EpochResult:
    """Loss and predictions of one pass over a loader."""

    loss: float
    y_true: IntArray
    y_pred: IntArray


def resolve_device(preference: str) -> torch.device:
    """Pick the compute device.

    Args:
        preference: ``auto``, ``cpu`` or ``cuda``.

    Returns:
        The device (``auto`` uses CUDA when available).
    """
    if preference == "cuda" or (preference == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def class_weights(y: IntArray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights (mean 1 over present classes, 0 for absent ones).

    Args:
        y: Training labels.
        n_classes: Number of classes.

    Returns:
        Float tensor of shape ``(n_classes,)``.
    """
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    present = counts > 0
    weights = np.zeros(n_classes)
    weights[present] = counts.sum() / (present.sum() * counts[present])
    return torch.tensor(weights, dtype=torch.float32)


def make_loader(
    x: npt.NDArray[np.float32], y: IntArray, batch_size: int, shuffle: bool, seed: int, num_workers: int
) -> DataLoader:
    """Build a DataLoader whose order depends only on ``seed``.

    Args:
        x: Windows ``(n, channels, length)``.
        y: Labels.
        batch_size: Batch size.
        shuffle: Shuffle every time the loader is iterated.
        seed: Seed of the loader's own generator.
        num_workers: Worker processes (fixed per config).

    Returns:
        The loader.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: Optimizer | None = None,
    on_step: Callable[[float], None] | None = None,
) -> EpochResult:
    """One pass over a loader; trains if an optimizer is given, otherwise evaluates.

    Args:
        model: The network.
        loader: Data.
        loss_fn: Loss function.
        device: Device.
        optimizer: Optimizer for training, ``None`` for evaluation.
        on_step: Callback receiving each training batch loss.

    Returns:
        Mean loss plus true and predicted labels.
    """
    training = optimizer is not None
    model.train(training)
    total_loss, total_n = 0.0, 0
    trues: list[torch.Tensor] = []
    preds: list[torch.Tensor] = []
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if on_step is not None:
                    on_step(float(loss.item()))
            total_loss += float(loss.item()) * len(yb)
            total_n += len(yb)
            trues.append(yb.cpu())
            preds.append(logits.argmax(dim=1).cpu())
    return EpochResult(
        loss=total_loss / max(total_n, 1),
        y_true=torch.cat(trues).numpy().astype(np.int64),
        y_pred=torch.cat(preds).numpy().astype(np.int64),
    )


def _rng_state() -> dict[str, Any]:
    """Capture all RNG states (needed for bit-identical resume)."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore RNG states captured by :func:`_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    state: TrainState,
    run_id: str,
    cfg_hash: str,
) -> None:
    """Atomically write a resumable checkpoint.

    Args:
        path: Destination (``last.pt``).
        model: Model.
        optimizer: Optimizer.
        scheduler: LR scheduler or ``None``.
        state: Loop state.
        run_id: MLflow run to continue on resume.
        cfg_hash: Hash of the config the checkpoint belongs to.
    """
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "state": asdict(state),
        "rng": _rng_state(),
        "run_id": run_id,
        "config_hash": cfg_hash,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


# --------------------------------------------------------------------------- deep models


def fit(
    cfg: Config,
    data: PreparedData,
    device: torch.device,
    tracker: RunTracker,
    run_dir: Path,
    resume: Mapping[str, Any] | None = None,
) -> tuple[nn.Module, TrainState]:
    """Train with early stopping on validation macro-F1.

    Args:
        cfg: Resolved config.
        data: Prepared windows (``train`` and ``val`` required).
        device: Device.
        tracker: Active MLflow run.
        run_dir: Folder for ``last.pt`` / ``best.pt``.
        resume: Checkpoint payload to continue from.

    Returns:
        The model loaded with the best weights, and the final loop state.
    """
    set_seed(cfg.seed)
    n_classes = len(data.class_ids)
    train, val = data.splits["train"], data.splits["val"]
    model = build_model(cfg.model, len(data.channels), n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
        if cfg.train.scheduler == "cosine"
        else None
    )
    weights = class_weights(train.y, n_classes).to(device) if cfg.train.class_weighting else None
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    state = TrainState()
    if resume is not None:
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        if scheduler is not None and resume["scheduler"] is not None:
            scheduler.load_state_dict(resume["scheduler"])
        state = TrainState(**resume["state"])
        _restore_rng_state(resume["rng"])
        logger.info("resumed from checkpoint", extra=kv(epoch=state.epoch))

    val_loader = make_loader(val.x, val.y, cfg.train.batch_size, False, cfg.seed, cfg.train.num_workers)
    best_path, last_path = run_dir / "best.pt", run_dir / "last.pt"
    cfg_digest = config_hash(cfg)

    def on_step(loss: float) -> None:
        state.global_step += 1
        if state.global_step % cfg.train.log_every_n_steps == 0:
            tracker.log_metrics({"train_loss_step": loss}, step=state.global_step)

    while state.epoch < cfg.train.epochs and not state.stopped_early:
        epoch = state.epoch
        started = time.perf_counter()
        # A fresh, epoch-seeded loader makes the batch order independent of resumes.
        train_loader = make_loader(
            train.x, train.y, cfg.train.batch_size, True, cfg.seed + epoch, cfg.train.num_workers
        )
        tr = run_epoch(model, train_loader, loss_fn, device, optimizer, on_step)
        va = run_epoch(model, val_loader, loss_fn, device)
        tr_m, va_m = (
            core_metrics(tr.y_true, tr.y_pred, n_classes),
            core_metrics(va.y_true, va.y_pred, n_classes),
        )
        metrics = {
            "train_loss": tr.loss,
            "train_accuracy": tr_m["accuracy"],
            "train_macro_f1": tr_m["macro_f1"],
            "val_loss": va.loss,
            "val_accuracy": va_m["accuracy"],
            "val_macro_f1": va_m["macro_f1"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_seconds": time.perf_counter() - started,
        }
        tracker.log_metrics(metrics, step=epoch)
        state.history.append({"epoch": float(epoch), **metrics})

        if va_m["macro_f1"] > state.best_score:
            state.best_score, state.best_epoch, state.stale_epochs = va_m["macro_f1"], epoch, 0
            torch.save(model.state_dict(), best_path)
        else:
            state.stale_epochs += 1
        if scheduler is not None:
            scheduler.step()
        state.epoch = epoch + 1
        state.stopped_early = state.stale_epochs >= cfg.train.patience
        save_checkpoint(last_path, model, optimizer, scheduler, state, tracker.run_id, cfg_digest)
        logger.info(
            "epoch done",
            extra=kv(
                epoch=epoch,
                train_loss=round(tr.loss, 4),
                val_loss=round(va.loss, 4),
                val_macro_f1=round(va_m["macro_f1"], 4),
                best_epoch=state.best_epoch,
            ),
        )

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    return model, state


def _bundle_meta(
    cfg: Config, data: PreparedData, run_id: str, feature_names: list[str] | None = None
) -> BundleMeta:
    """Collect the inference metadata of a trained model."""
    return BundleMeta(
        kind="",
        config=config_to_dict(cfg),
        channels=data.channels,
        class_ids=data.class_ids,
        class_names=data.class_names,
        norm_mean=data.norm_mean.tolist(),
        norm_std=data.norm_std.tolist(),
        window_length=data.window_length,
        sample_rate_hz=data.sample_rate_hz,
        run_id=run_id,
        feature_names=feature_names or [],
    )


def _evaluate_split(predictor: Predictor, data: PreparedData, split: str) -> EvalResult:
    """Predict one split with a bundle and compute all metrics."""
    arrays = data.splits[split]
    return evaluate_predictions(arrays.y, predictor.predict(arrays.x), arrays.subjects, data.class_names)


def _log_common_artifacts(tracker: RunTracker) -> None:
    """Attach the exact environment lock file to the run."""
    lock = Path("requirements-lock.txt")
    if lock.exists():
        tracker.log_artifact(lock, "environment")


def _load_data(cfg: Config, split: SplitConfig | None, splits: tuple[str, ...]) -> tuple[PreparedData, Path]:
    """Resolve the data version, load clean data and build windows."""
    clean_dir = resolve_clean_dir(cfg)
    clean, meta = load_clean(clean_dir)
    return prepare_data(cfg, clean, meta["channels"], split, splits), clean_dir  # type: ignore[arg-type]


def run_training(
    cfg: Config,
    allow_dirty: bool = False,
    parent_run_id: str | None = None,
    output_dir: Path | None = None,
    extra_tags: Mapping[str, Any] | None = None,
    fresh: bool = False,
) -> TrainOutcome:
    """Train a deep model as one tracked MLflow run.

    If an unfinished checkpoint for the identical config exists (e.g. after a
    Slurm TIMEOUT), training resumes from it and keeps logging to the same run.

    Args:
        cfg: Resolved config.
        allow_dirty: Allow uncommitted code.
        parent_run_id: Parent HPO/LOSO run.
        output_dir: Where to write the model bundle (default: inside the run folder).
        extra_tags: Additional MLflow tags.
        fresh: Ignore existing checkpoints (used by ``verify-repro``).

    Returns:
        The training outcome.
    """
    set_seed(cfg.seed)
    splits = ("train", "val", "test") if cfg.train.eval_test else ("train", "val")
    data, clean_dir = _load_data(cfg, None, splits)
    device = resolve_device(cfg.train.device)

    cfg_digest = config_hash(cfg)
    run_dir = Path(cfg.paths.runs_dir) / f"{cfg.model.name}-{cfg_digest}"
    resume: dict[str, Any] | None = None
    if run_dir.exists() and (fresh or (run_dir / "completed.json").exists()):
        shutil.rmtree(run_dir)
    if (run_dir / "last.pt").exists():
        resume = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        if resume["config_hash"] != cfg_digest:
            resume = None
    run_dir.mkdir(parents=True, exist_ok=True)

    tags = {"model.kind": cfg.model.name, "data.sensor_subset": cfg.data.sensor_subset, **(extra_tags or {})}
    with start_run(
        cfg,
        stage="train",
        run_name=make_run_name(cfg, cfg.model.name),
        allow_dirty=allow_dirty,
        parent_run_id=parent_run_id,
        run_id=resume["run_id"] if resume else None,
        extra_tags=tags,
        data_dir=clean_dir,
        log_config=resume is None,
    ) as tracker:
        tracker.set_tags({**gpu_tags(), "device": str(device)})
        if resume is None:
            tracker.log_params(
                {f"data.n_{name}_windows": len(arr.y) for name, arr in data.splits.items()}
                | {"data.n_channels": len(data.channels), "data.window_length": data.window_length}
            )
        started = time.perf_counter()
        model, state = fit(cfg, data, device, tracker, run_dir, resume)
        n_params = sum(p.numel() for p in model.parameters())

        bundle_dir = Path(output_dir) if output_dir else run_dir / "bundle"
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
        meta = save_torch_bundle(bundle_dir, model, _bundle_meta(cfg, data, tracker.run_id))
        predictor = Predictor(meta, model, str(device))

        best_val = _evaluate_split(predictor, data, "val")
        log_evaluation(tracker, best_val, "best_val", data.class_names)
        final = {f"best_val_{k}": v for k, v in best_val.metrics.items()}
        if cfg.train.eval_test:
            test = _evaluate_split(predictor, data, "test")
            log_evaluation(tracker, test, "test", data.class_names)
            final |= {f"test_{k}": v for k, v in test.metrics.items()}

        summary = {
            "best_epoch": float(state.best_epoch),
            "epochs_trained": float(state.epoch),
            "n_params": float(n_params),
            "train_time_seconds": time.perf_counter() - started,
        }
        tracker.log_metrics(summary)
        tracker.set_tags({"model.weights_sha256": meta.weights_sha256, "early_stopped": state.stopped_early})
        tracker.log_figure(learning_curve_figure(state.history), "figures/learning_curves.png")
        tracker.log_dict(state.history, "reports/history.json")
        tracker.log_artifact(bundle_dir, "bundle")
        _log_common_artifacts(tracker)
        write_json(run_dir / "completed.json", {"run_id": tracker.run_id})
        return TrainOutcome(tracker.run_id, final | summary, bundle_dir, meta.weights_sha256)


# --------------------------------------------------------------------------- baseline


def _feature_split(table: Any, split: str) -> tuple[npt.NDArray[np.float32], IntArray, IntArray]:
    """Return features, labels and subjects of one split of the feature table."""
    from src.schemas import FEATURE_PREFIX

    part = table.loc[table["split"] == split]
    cols = [c for c in table.columns if c.startswith(FEATURE_PREFIX)]
    return (
        part[cols].to_numpy(dtype=np.float32),
        part["label"].to_numpy(dtype=np.int64),
        part["subject_id"].to_numpy(dtype=np.int64),
    )


def run_baseline(
    cfg: Config,
    allow_dirty: bool = False,
    parent_run_id: str | None = None,
    output_dir: Path | None = None,
    split: SplitConfig | None = None,
    extra_tags: Mapping[str, Any] | None = None,
) -> TrainOutcome:
    """Fit and evaluate the feature-based baseline as one tracked run.

    Uses the ``features`` stage output for the default split, and computes
    features in memory for other splits (LOSO folds).

    Args:
        cfg: Resolved config.
        allow_dirty: Allow uncommitted code.
        parent_run_id: Parent HPO/LOSO run.
        output_dir: Where to write the model bundle.
        split: Optional split override.
        extra_tags: Additional MLflow tags.

    Returns:
        The training outcome.
    """
    from src.features import compute_feature_table, load_feature_table
    from src.models.baseline import build_baseline, feature_importances
    from src.schemas import FEATURE_PREFIX

    set_seed(cfg.seed)
    if split is not None:
        cfg = cfg.model_copy(update={"split": split})
    if split is None and (Path(cfg.paths.features_dir) / "meta.json").exists():
        table, fmeta = load_feature_table(cfg)
    else:
        table, fmeta = compute_feature_table(cfg, split)
    names = [c for c in table.columns if c.startswith(FEATURE_PREFIX)]
    x_tr, y_tr, _ = _feature_split(table, "train")
    x_va, y_va, s_va = _feature_split(table, "val")
    class_names = cfg.class_names
    n_classes = len(class_names)

    mlflow.sklearn.autolog(log_models=True, log_datasets=False, log_post_training_metrics=False, silent=True)
    try:
        with start_run(
            cfg,
            stage="baseline",
            run_name=make_run_name(cfg, "baseline"),
            allow_dirty=allow_dirty,
            parent_run_id=parent_run_id,
            extra_tags={"model.kind": f"baseline-{cfg.baseline.model}", **(extra_tags or {})},
            data_dir=resolve_clean_dir(cfg),
        ) as tracker:
            tracker.log_params(
                {
                    "data.n_train_windows": len(y_tr),
                    "data.n_val_windows": len(y_va),
                    "data.n_features": len(names),
                }
            )
            started = time.perf_counter()
            model = build_baseline(cfg.baseline, cfg.seed)
            model.fit(x_tr, y_tr)
            fit_seconds = time.perf_counter() - started

            train_m = core_metrics(y_tr, model.predict(x_tr).astype(np.int64), n_classes)
            val_pred = model.predict(x_va).astype(np.int64)
            val_res = evaluate_predictions(y_va, val_pred, s_va, class_names)
            tracker.log_metrics({f"train_{k}": v for k, v in train_m.items()})
            tracker.log_metrics(
                {"val_accuracy": val_res.metrics["accuracy"], "val_macro_f1": val_res.metrics["macro_f1"]}
            )
            log_evaluation(tracker, val_res, "best_val", class_names)
            final = {f"best_val_{k}": v for k, v in val_res.metrics.items()}
            if cfg.train.eval_test or split is not None:
                x_te, y_te, s_te = _feature_split(table, "test")
                test_res = evaluate_predictions(y_te, model.predict(x_te).astype(np.int64), s_te, class_names)
                log_evaluation(tracker, test_res, "test", class_names)
                final |= {f"test_{k}": v for k, v in test_res.metrics.items()}

            importances = feature_importances(model, [n.removeprefix(FEATURE_PREFIX) for n in names])
            tracker.log_dict(importances.to_dict(orient="records"), "reports/feature_importances.json")
            tracker.log_figure(_importance_figure(importances), "figures/feature_importances.png")

            meta = BundleMeta(
                kind="",
                config=config_to_dict(cfg),
                channels=list(fmeta["channels"]),  # type: ignore[call-overload]
                class_ids=cfg.class_ids,
                class_names=class_names,
                norm_mean=list(fmeta["norm_mean"]),  # type: ignore[call-overload]
                norm_std=list(fmeta["norm_std"]),  # type: ignore[call-overload]
                window_length=int(fmeta["window_length"]),  # type: ignore[call-overload]
                sample_rate_hz=float(fmeta["sample_rate_hz"]),  # type: ignore[arg-type]
                run_id=tracker.run_id,
                feature_names=names,
            )
            run_dir = Path(cfg.paths.runs_dir) / f"baseline-{config_hash(cfg)}"
            bundle_dir = Path(output_dir) if output_dir else run_dir / "bundle"
            if bundle_dir.exists():
                shutil.rmtree(bundle_dir)
            meta = save_sklearn_bundle(bundle_dir, model, meta)
            tracker.log_metrics({"train_time_seconds": fit_seconds})
            tracker.set_tags({"model.weights_sha256": meta.weights_sha256})
            tracker.log_artifact(bundle_dir, "bundle")
            _log_common_artifacts(tracker)
            return TrainOutcome(tracker.run_id, final, bundle_dir, meta.weights_sha256)
    finally:
        mlflow.sklearn.autolog(disable=True)


def _importance_figure(table: Any, top: int = 25) -> Any:
    """Horizontal bar chart of the most important features."""
    import matplotlib.pyplot as plt

    head = table.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(head) + 1.5))
    ax.barh(head["feature"], head["importance"], color="#3b6ea5")
    ax.set_xlabel("importance")
    ax.set_title(f"Top {len(head)} baseline features")
    fig.tight_layout()
    return fig


def read_bundle_run_id(bundle_dir: Path) -> str:
    """Return the MLflow run id stored in a bundle.

    Args:
        bundle_dir: Bundle folder.

    Returns:
        The run id.
    """
    return str(read_json(Path(bundle_dir) / "bundle.json")["run_id"])
