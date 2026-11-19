"""Rewrite the absolute artifact paths of an exported MLflow store to a new location.

The interim store records artifact locations as absolute paths on the cluster
(``/shared/shmqak/mlflow/artifacts/...``). After unpacking the export archive
elsewhere, run this once so the MLflow UI finds the artifacts::

    python relocate_mlflow.py mlflow.db /new/path/to/mlflow/artifacts

It only touches the ``experiments.artifact_location``, ``runs.artifact_uri`` and
(if present) ``logged_models.artifact_location`` columns, and prints how many
rows were changed. A ``.bak`` copy of the database is written first.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
from pathlib import Path

OLD_ROOT_DEFAULT = "/shared/shmqak/mlflow/artifacts"
COLUMNS: tuple[tuple[str, str], ...] = (
    ("experiments", "artifact_location"),
    ("runs", "artifact_uri"),
    ("logged_models", "artifact_location"),
)


def relocate(db_path: Path, new_root: str, old_root: str = OLD_ROOT_DEFAULT) -> dict[str, int]:
    """Replace the artifact-root prefix in every artifact location column.

    Args:
        db_path: The exported ``mlflow.db``.
        new_root: New absolute artifact folder (or ``file:///...`` URI).
        old_root: Prefix to replace.

    Returns:
        Number of changed rows per table.
    """
    shutil.copy2(db_path, db_path.with_suffix(".db.bak"))
    changed: dict[str, int] = {}
    with sqlite3.connect(db_path) as con:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table, column in COLUMNS:
            if table not in tables:
                continue
            for prefix in (f"file://{old_root}", old_root):
                cur = con.execute(
                    f"UPDATE {table} SET {column} = ? || substr({column}, ?) WHERE {column} LIKE ?",
                    (new_root, len(prefix) + 1, f"{prefix}%"),
                )
                changed[table] = changed.get(table, 0) + cur.rowcount
    return changed


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("db", type=Path)
    parser.add_argument("new_root")
    parser.add_argument("--old-root", default=OLD_ROOT_DEFAULT)
    args = parser.parse_args()
    root = args.new_root
    if not root.startswith("file:") and Path(root).drive:  # Windows path -> file URI
        root = Path(root).resolve().as_uri()
    print(relocate(args.db, root.rstrip("/"), args.old_root.rstrip("/")))


if __name__ == "__main__":
    main()
