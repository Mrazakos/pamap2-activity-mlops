"""Tests for resolving a data version to a clean-data folder (workspace, cache, or DVC tag)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src import data_versions
from src.config import Config
from src.data_versions import data_tag, resolve_clean_dir
from src.utils import write_json


def test_data_tag() -> None:
    assert data_tag("v1") == "data-v1"


def test_workspace_version_is_used(tiny_cfg: Config) -> None:
    write_json(Path(tiny_cfg.paths.clean_dir) / "meta.json", {"data_version": "v2"})
    assert resolve_clean_dir(tiny_cfg) == Path(tiny_cfg.paths.clean_dir)


def test_other_version_is_fetched_from_its_tag(tiny_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    write_json(Path(tiny_cfg.paths.clean_dir) / "meta.json", {"data_version": "v2"})
    cfg = tiny_cfg.model_copy(update={"data": tiny_cfg.data.model_copy(update={"version": "v1"})})
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        write_json(Path(cmd[cmd.index("-o") + 1]) / "meta.json", {"data_version": "v1"})
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(data_versions.subprocess, "run", fake_run)
    first = resolve_clean_dir(cfg)
    assert first == Path(cfg.paths.versions_dir) / "v1" / "clean"
    assert calls[0][:5] == ["dvc", "get", ".", Path(cfg.paths.clean_dir).as_posix(), "--rev"]
    assert calls[0][5] == "data-v1"
    assert resolve_clean_dir(cfg) == first and len(calls) == 1  # second call uses the cache


def test_fetch_failure_is_reported(tiny_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tiny_cfg.model_copy(update={"data": tiny_cfg.data.model_copy(update={"version": "v1"})})

    def failing(cmd: list[str], **_: object) -> None:
        raise subprocess.CalledProcessError(1, cmd, stderr="unknown revision data-v1")

    monkeypatch.setattr(data_versions.subprocess, "run", failing)
    with pytest.raises(RuntimeError, match="data-v1"):
        resolve_clean_dir(cfg)
