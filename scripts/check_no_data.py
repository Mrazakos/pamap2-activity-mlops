"""Block data, models, credentials and tracking state from entering git.

Two modes:

* ``python scripts/check_no_data.py FILE ...``  - check staged files (pre-commit, CI).
* ``python scripts/check_no_data.py --history`` - check every blob of every commit
  on every branch/tag (run before each push; a single leaked data file anywhere
  in the history fails the project).

Besides file names, text content is scanned for rows that look like sensor
recordings (many numeric columns per line), so data hidden in e.g. a ``.txt``
or a notebook output is caught as well.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import PurePosixPath

BLOCKED_SUFFIXES: frozenset[str] = frozenset(
    {
        ".dat",
        ".parquet",
        ".csv",
        ".tsv",
        ".npy",
        ".npz",
        ".pt",
        ".pth",
        ".ckpt",
        ".safetensors",
        ".joblib",
        ".pkl",
        ".pickle",
        ".zip",
        ".gz",
        ".tar",
        ".h5",
        ".hdf5",
        ".feather",
        ".arrow",
        ".db",
        ".sqlite",
        ".sqlite3",
        ".onnx",
    }
)
BLOCKED_NAMES: frozenset[str] = frozenset({".env", "mlflow.db"})
BLOCKED_DIRS: tuple[str, ...] = ("data/", "models/", "mlruns/", "mlartifacts/", "runs/", "reference/")
ALLOWED_DATA_FILES: frozenset[str] = frozenset({"data/raw.dvc", "data/.gitignore"})
MAX_TEXT_BYTES = 5_000_000
MIN_NUMERIC_COLUMNS = 15
MIN_SUSPICIOUS_LINES = 5

_NUMBER = r"(?:-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|NaN|nan)"
SENSOR_ROW = re.compile(rf"^\s*{_NUMBER}(?:[\s,;]+{_NUMBER}){{{MIN_NUMERIC_COLUMNS - 1},}}\s*$")


def path_problem(path: str) -> str | None:
    """Return why a path must not be committed, or ``None`` if it is fine.

    Args:
        path: Repository-relative path (``/`` separators).

    Returns:
        A reason string or ``None``.
    """
    posix = PurePosixPath(path)
    if path in ALLOWED_DATA_FILES:
        return None
    if posix.name in BLOCKED_NAMES:
        return "credentials or tracking database"
    if posix.suffix.lower() in BLOCKED_SUFFIXES:
        return f"blocked file type {posix.suffix}"
    if any(path.startswith(d) for d in BLOCKED_DIRS):
        return "file inside a data/model/tracking folder"
    return None


def content_problem(text: str) -> str | None:
    """Detect text that looks like raw sensor rows.

    Args:
        text: File content.

    Returns:
        A reason string or ``None``.
    """
    hits = sum(1 for line in text.splitlines() if SENSOR_ROW.match(line))
    if hits >= MIN_SUSPICIOUS_LINES:
        return f"{hits} lines look like numeric sensor rows"
    return None


def check_files(paths: Iterable[str]) -> list[str]:
    """Check working-tree files by name and content.

    Args:
        paths: File paths.

    Returns:
        Problem descriptions (empty if all files are fine).
    """
    problems: list[str] = []
    for raw in paths:
        path = raw.replace("\\", "/")
        if reason := path_problem(path):
            problems.append(f"{path}: {reason}")
            continue
        try:
            with open(raw, "rb") as fh:
                data = fh.read(MAX_TEXT_BYTES + 1)
        except OSError:
            continue
        if b"\0" not in data[:8192] and (reason := content_problem(data.decode("utf-8", "ignore"))):
            problems.append(f"{path}: {reason}")
    return problems


def _git(*args: str) -> str:
    """Run git and return stdout."""
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def check_history() -> list[str]:
    """Check every blob reachable from any ref (branches, tags, remotes, stash).

    Returns:
        Problem descriptions including the first commit where each was found.
    """
    problems: list[str] = []
    seen: set[str] = set()
    commits = _git("rev-list", "--all").split()
    for commit in commits:
        for line in _git("ls-tree", "-r", "-l", commit).splitlines():
            meta, path = line.split("\t", 1)
            _, kind, sha, size = meta.split()
            if kind != "blob" or sha in seen:
                continue
            seen.add(sha)
            if reason := path_problem(path):
                problems.append(f"{commit[:10]} {path}: {reason}")
                continue
            if size != "-" and int(size) <= MAX_TEXT_BYTES:
                blob = subprocess.run(["git", "cat-file", "-p", sha], capture_output=True, check=True).stdout
                if b"\0" not in blob[:8192] and (reason := content_problem(blob.decode("utf-8", "ignore"))):
                    problems.append(f"{commit[:10]} {path}: {reason}")
    print(f"scanned {len(commits)} commits, {len(seen)} unique blobs")
    return problems


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Arguments (default: ``sys.argv[1:]``).

    Returns:
        Process exit code (1 if anything is blocked).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="*")
    parser.add_argument("--history", action="store_true", help="scan the full git history")
    args = parser.parse_args(argv)
    problems = check_history() if args.history else check_files(args.files)
    for problem in problems:
        print(f"BLOCKED  {problem}")
    if not problems:
        print("no data, models or credentials found")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
