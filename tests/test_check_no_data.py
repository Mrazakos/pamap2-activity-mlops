"""Tests for the commit guard that keeps data, models and credentials out of git."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_no_data import check_files, content_problem, path_problem
from scripts.check_notebooks import notebook_problems


@pytest.mark.parametrize(
    "path",
    [
        "data/raw/protocol/subject101.parquet",
        "subject101.dat",
        "x/model.pt",
        "a.csv",
        ".env",
        "mlflow.db",
        "models/cnn/bundle.json",
        "runs/x/notes.txt",
    ],
)
def test_blocked_paths(path: str) -> None:
    assert path_problem(path) is not None


@pytest.mark.parametrize("path", ["src/main.py", "data/raw.dvc", ".env.example", "docs/img/heatmap.png"])
def test_allowed_paths(path: str) -> None:
    assert path_problem(path) is None


def test_sensor_like_rows_are_detected() -> None:
    row = " ".join(["5.38", "0", "NaN"] + ["1.234"] * 51)
    assert content_problem("\n".join([row] * 10)) is not None
    assert content_problem("normal text\n1 2 3\n") is None


def test_check_files_reads_content(tmp_path: Path) -> None:
    leak = tmp_path / "notes.txt"
    leak.write_text("\n".join([" ".join(["1.0"] * 30)] * 6))
    assert check_files([str(leak)])


def test_notebook_with_outputs_is_reported(tmp_path: Path) -> None:
    nb = tmp_path / "x.ipynb"
    nb.write_text('{"cells": [{"cell_type": "code", "execution_count": 1, "outputs": [{"text": "1"}]}]}')
    assert len(notebook_problems(nb)) == 2
    nb.write_text('{"cells": [{"cell_type": "code", "execution_count": null, "outputs": []}]}')
    assert notebook_problems(nb) == []
