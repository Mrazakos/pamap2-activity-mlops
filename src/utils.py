"""Shared helpers: structured logging, seeding/determinism, hashing and git metadata."""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- logging

_LOG_CONTEXT: contextvars.ContextVar[Mapping[str, Any]] = contextvars.ContextVar(
    "log_context", default=MappingProxyType({})
)
_CONFIGURED = False
_RESERVED = set(vars(logging.makeLogRecord({})))


def _record_context(record: logging.LogRecord) -> dict[str, Any]:
    """Merge the global log context with the ``ctx`` passed via ``extra``."""
    merged = dict(_LOG_CONTEXT.get())
    merged.update(getattr(record, "ctx", {}) or {})
    return merged


class KeyValueFormatter(logging.Formatter):
    """Human-readable lines: ``time | LEVEL | logger | message | key=value ...``."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a record, appending its context as ``key=value`` pairs.

        Args:
            record: The log record.

        Returns:
            The formatted line.
        """
        base = super().format(record)
        ctx = _record_context(record)
        if not ctx:
            return base
        pairs = " ".join(f"{k}={v}" for k, v in ctx.items())
        return f"{base} | {pairs}"


class JsonFormatter(logging.Formatter):
    """One JSON object per line, for machine-readable Slurm logs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a record as JSON.

        Args:
            record: The log record.

        Returns:
            A single-line JSON string.
        """
        payload: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **_record_context(record),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str | None = None) -> None:
    """Configure the root logger once.

    Environment variables: ``LOG_LEVEL`` (default ``INFO``), ``LOG_FORMAT``
    (``text`` or ``json``), ``LOG_FILE`` (optional extra file handler, JSON lines).

    Args:
        level: Explicit level, overriding ``LOG_LEVEL``.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    root = logging.getLogger()
    root.setLevel(level or os.environ.get("LOG_LEVEL", "INFO"))
    text_fmt = KeyValueFormatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(JsonFormatter() if os.environ.get("LOG_FORMAT") == "json" else text_fmt)
    root.handlers = [stream]
    if log_file := os.environ.get("LOG_FILE"):
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        root.addHandler(file_handler)
    for noisy in ("matplotlib", "PIL", "urllib3", "alembic", "git", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger.

    Pass per-call context with ``logger.info("msg", extra=kv(subject=101))``.

    Args:
        name: Logger name, usually ``__name__``.

    Returns:
        The logger.
    """
    configure_logging()
    return logging.getLogger(name)


def kv(**context: Any) -> dict[str, dict[str, Any]]:
    """Build the ``extra`` argument that attaches key=value context to one log call.

    Args:
        **context: Arbitrary key/value pairs.

    Returns:
        ``{"ctx": context}``.
    """
    return {"ctx": context}


@contextmanager
def log_context(**context: Any) -> Iterator[None]:
    """Attach key=value context (e.g. ``stage``, ``run_id``) to every log line in a block.

    Args:
        **context: Key/value pairs added to the current context.

    Yields:
        Nothing; the context is removed on exit.
    """
    token = _LOG_CONTEXT.set({**_LOG_CONTEXT.get(), **context})
    try:
        yield
    finally:
        _LOG_CONTEXT.reset(token)


# --------------------------------------------------------------------------- determinism


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every random number generator and enable deterministic kernels.

    ``CUBLAS_WORKSPACE_CONFIG`` and ``PYTHONHASHSEED`` must be exported before
    Python starts (the Slurm scripts do this); they are set here as a fallback
    for code paths that have not initialised CUDA yet.

    Args:
        seed: The seed.
        deterministic: Also force deterministic algorithms in PyTorch/cuDNN.
    """
    import torch

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and ``random`` inside a DataLoader worker.

    PyTorch seeds each worker's torch RNG from the loader's generator; this
    propagates that seed to the other libraries.

    Args:
        worker_id: Worker index (unused, required by the DataLoader API).
    """
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --------------------------------------------------------------------------- hashing


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks.

    Args:
        path: File to hash.
        chunk_size: Read size in bytes.

    Returns:
        Hex digest.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(obj: Any) -> str:
    """Return the SHA-256 of an object's canonical JSON encoding.

    Args:
        obj: JSON-serialisable object.

    Returns:
        Hex digest.
    """
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def write_json(path: Path, obj: Any) -> None:
    """Write pretty, key-sorted JSON (stable diffs), creating parent folders.

    Args:
        path: Destination file.
        obj: JSON-serialisable object.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    """Read a JSON file.

    Args:
        path: File to read.

    Returns:
        The decoded object.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- git / dvc


def _git(*args: str, cwd: Path | None = None) -> str | None:
    """Run a git command and return its stripped stdout, or ``None`` on failure."""
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


DIRTY_CHECK_EXCLUDES: tuple[str, ...] = ("dvc.lock", "reports")
"""Tracked paths that pipeline stages rewrite themselves; they do not make the *code* dirty."""


def git_info(cwd: Path | None = None) -> dict[str, str]:
    """Collect commit, branch and dirty state of the working tree.

    Untracked files are ignored, and so are files that ``dvc repro`` rewrites
    while it runs (:data:`DIRTY_CHECK_EXCLUDES`); data lineage for those is
    recorded separately through the DVC hashes.

    Args:
        cwd: Repository directory (default: current directory).

    Returns:
        Mapping with ``commit``, ``branch`` and ``dirty`` (``"true"``/``"false"``/``"unknown"``).
    """
    commit = _git("rev-parse", "HEAD", cwd=cwd)
    if commit is None:
        return {"commit": "unknown", "branch": "unknown", "dirty": "unknown"}
    excludes = [f":(exclude){p}" for p in DIRTY_CHECK_EXCLUDES]
    status = _git("status", "--porcelain", "--untracked-files=no", "--", ".", *excludes, cwd=cwd) or ""
    return {
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd) or "unknown",
        "dirty": "true" if status else "false",
    }


def git_show(rev: str, path: str, cwd: Path | None = None) -> str | None:
    """Return a file's content at a git revision (e.g. ``dvc.lock`` at tag ``data-v1``).

    Args:
        rev: Commit, branch or tag.
        path: Repository-relative path.
        cwd: Repository directory.

    Returns:
        The file content, or ``None`` if it does not exist there.
    """
    return _git("show", f"{rev}:{path}", cwd=cwd)


def parse_dvc_lock(text: str) -> dict[str, str]:
    """Return ``{out_path: md5}`` for every stage output in a ``dvc.lock`` text.

    Args:
        text: YAML content of ``dvc.lock``.

    Returns:
        Mapping of output path to its DVC hash.
    """
    import yaml

    lock = yaml.safe_load(text) or {}
    result: dict[str, str] = {}
    for stage in (lock.get("stages") or {}).values():
        for out in stage.get("outs", []) or []:
            result[out["path"]] = str(out.get("md5") or out.get("hash") or "")
    return result


def dvc_lock_outs(lock_path: Path = Path("dvc.lock")) -> dict[str, str]:
    """Return ``{out_path: md5}`` for every stage output recorded in ``dvc.lock``.

    Args:
        lock_path: Path to ``dvc.lock``.

    Returns:
        Mapping of output path to its DVC hash (empty if the file is missing).
    """
    if not Path(lock_path).exists():
        return {}
    return parse_dvc_lock(Path(lock_path).read_text(encoding="utf-8"))


def dvc_file_hash(dvc_file: Path) -> str:
    """Return the hash recorded in a ``.dvc`` file (e.g. ``data/raw.dvc``).

    Args:
        dvc_file: Path of the ``.dvc`` file.

    Returns:
        The md5 of the tracked output, or ``"unknown"``.
    """
    import yaml

    if not Path(dvc_file).exists():
        return "unknown"
    meta = yaml.safe_load(Path(dvc_file).read_text(encoding="utf-8")) or {}
    outs = meta.get("outs") or [{}]
    return str(outs[0].get("md5", "unknown"))


def flatten_dict(nested: Mapping[str, Any], prefix: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten a nested mapping into dotted keys.

    Lists are kept as values (MLflow stores them as strings).

    Args:
        nested: Mapping to flatten.
        prefix: Key prefix used during recursion.
        sep: Separator between key levels.

    Returns:
        A flat dictionary, e.g. ``{"train.lr": 0.001}``.
    """
    flat: dict[str, Any] = {}
    for key, value in nested.items():
        full = f"{prefix}{sep}{key}" if prefix else str(key)
        if isinstance(value, Mapping) and value:
            flat.update(flatten_dict(value, full, sep))
        else:
            flat[full] = value
    return flat
