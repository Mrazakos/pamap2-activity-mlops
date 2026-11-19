"""Fail if any Jupyter notebook contains outputs or execution counts.

Notebook outputs could print real sensor rows, so notebooks are only ever
committed stripped. Usage: ``python scripts/check_notebooks.py [NOTEBOOK ...]``
(without arguments, every ``*.ipynb`` under the current folder is checked).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def notebook_problems(path: Path) -> list[str]:
    """List cells of a notebook that still carry outputs.

    Args:
        path: Notebook file.

    Returns:
        Problem descriptions.
    """
    notebook = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    for i, cell in enumerate(notebook.get("cells", [])):
        if cell.get("outputs"):
            problems.append(f"{path}: cell {i} has outputs")
        if cell.get("execution_count") is not None:
            problems.append(f"{path}: cell {i} has an execution count")
    return problems


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Notebook paths (default: all notebooks below the current folder).

    Returns:
        Exit code (1 if any notebook has outputs).
    """
    args = sys.argv[1:] if argv is None else argv
    paths = [Path(a) for a in args] or sorted(
        p for p in Path(".").rglob("*.ipynb") if ".venv" not in p.parts and "reference" not in p.parts
    )
    problems = [p for path in paths for p in notebook_problems(path)]
    for problem in problems:
        print(f"NOT STRIPPED  {problem}")
    print(f"checked {len(paths)} notebook(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
