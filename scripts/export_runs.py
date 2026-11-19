"""Write the comparison table of all tracked runs into ``docs/results.md``.

Queries MLflow (``MLFLOW_TRACKING_URI``) and replaces the block between
``<!-- runs-table:start -->`` and ``<!-- runs-table:end -->`` with a Markdown
table of parameters and metrics only (never data). Every number in the results
page therefore comes straight from the tracking store::

    python scripts/export_runs.py [--out docs/results.md]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import mlflow
import pandas as pd

from src.config import load_config
from src.tracking import experiment_name, tracking_uri

START, END = "<!-- runs-table:start -->", "<!-- runs-table:end -->"
PARAMS: dict[str, str] = {
    "data.version": "data",
    "data.sensor_subset": "sensors",
    "windows.window_seconds": "window s",
    "train.lr": "lr",
    "train.batch_size": "bs",
    "model.dropout": "dropout",
    "model.conv_channels": "channels",
    "train.class_weighting": "cls-w",
    "baseline.n_estimators": "trees",
    "baseline.max_depth": "depth",
}
METRICS: dict[str, str] = {
    "best_val_macro_f1": "val macro-F1",
    "best_val_accuracy": "val acc",
    "test_macro_f1": "test macro-F1",
    "test_accuracy": "test acc",
    "best_epoch": "best ep",
    "loso_macro_f1_mean": "LOSO mean",
    "loso_macro_f1_std": "LOSO std",
}


def runs_frame() -> pd.DataFrame:
    """Fetch every finished run of the experiment with selected params, metrics and tags.

    Returns:
        One row per run.
    """
    mlflow.set_tracking_uri(tracking_uri())
    runs = mlflow.search_runs(
        experiment_names=[experiment_name(load_config())],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time ASC"],
    )
    if runs.empty:
        return runs
    names = runs.set_index("run_id")["tags.mlflow.runName"].to_dict()
    table = pd.DataFrame(
        {
            "round": runs.get("tags.mlflow.parentRunId", pd.Series(dtype=str)).map(names).fillna("-"),
            "run": runs["tags.mlflow.runName"],
            "stage": runs.get("tags.stage", "-"),
            "model": runs.get("tags.model.kind", pd.Series(dtype=str)).fillna("-"),
        }
    )
    for key, label in PARAMS.items():
        table[label] = runs.get(f"params.{key}", pd.Series(dtype=str)).fillna("")
    for key, label in METRICS.items():
        table[label] = runs.get(f"metrics.{key}", pd.Series(dtype=float)).round(4)
    table["run_id"] = runs["run_id"].str[:8]
    keep = table["stage"].isin(["baseline", "train", "loso", "hpo"])
    return table.loc[keep].reset_index(drop=True)


def render(table: pd.DataFrame) -> str:
    """Render the table as Markdown (empty cells for missing values).

    Args:
        table: Output of :func:`runs_frame`.

    Returns:
        Markdown text.
    """
    if table.empty:
        return "_No finished runs yet._"
    shown = table.fillna("").astype(str).replace({"nan": "", "None": ""})
    header = "| " + " | ".join(shown.columns) + " |"
    rule = "|" + "|".join("---" for _ in shown.columns) + "|"
    rows = ["| " + " | ".join(r) + " |" for r in shown.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("docs/results.md"))
    args = parser.parse_args()
    text = args.out.read_text(encoding="utf-8")
    block = f"{START}\n{render(runs_frame())}\n{END}"
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    if not pattern.search(text):
        raise SystemExit(f"{args.out} has no {START} ... {END} block")
    args.out.write_text(pattern.sub(lambda _: block, text), encoding="utf-8")
    print(f"updated {args.out}")


if __name__ == "__main__":
    main()
