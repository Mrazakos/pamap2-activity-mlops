"""Resolve ``data.version`` to a folder with that version's clean data.

The workspace holds the version produced by the current ``dvc.lock``. Any
other version is fetched read-only from the DVC remote at its git tag
(``data-v1``, ``data-v2``) into ``data/versions/<version>/clean``. This lets the
same (current) code train on an older, immutable data version, which is how the
data-centric v1-vs-v2 experiment is run.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from src.config import Config
from src.utils import get_logger, kv, read_json

logger = get_logger(__name__)


def data_tag(version: str) -> str:
    """Return the git tag that marks a data version.

    Args:
        version: ``v1``, ``v2``, ...

    Returns:
        The tag name, e.g. ``data-v2``.
    """
    return f"data-{version}"


def _version_of(clean_dir: Path) -> str | None:
    """Return the data version recorded in a clean folder, if any."""
    meta = clean_dir / "meta.json"
    if not meta.is_file():
        return None
    return str(read_json(meta).get("data_version"))


def resolve_clean_dir(cfg: Config) -> Path:
    """Return a folder containing clean data of ``cfg.data.version``.

    Args:
        cfg: Resolved config.

    Returns:
        The workspace clean folder if it has the right version, otherwise a
        cached copy fetched from the DVC remote at the version's tag.

    Raises:
        RuntimeError: If the version cannot be fetched.
    """
    workspace = Path(cfg.paths.clean_dir)
    if _version_of(workspace) == cfg.data.version:
        return workspace

    cached = Path(cfg.paths.versions_dir) / cfg.data.version / "clean"
    if _version_of(cached) == cfg.data.version:
        logger.info("using cached data version", extra=kv(version=cfg.data.version, path=str(cached)))
        return cached

    tag = data_tag(cfg.data.version)
    if cached.exists():
        shutil.rmtree(cached)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["dvc", "get", ".", Path(cfg.paths.clean_dir).as_posix(), "--rev", tag, "-o", str(cached)]
    logger.info("fetching data version", extra=kv(version=cfg.data.version, tag=tag))
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise RuntimeError(f"could not fetch data version {tag}: {detail}") from exc
    if _version_of(cached) != cfg.data.version:
        raise RuntimeError(f"{cached} does not contain data version {cfg.data.version}")
    return cached
