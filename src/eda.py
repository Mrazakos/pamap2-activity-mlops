"""Exploratory data analysis on the raw recordings: aggregates and figures only.

Nothing here prints or stores individual sensor rows. Results go to an MLflow
run tagged ``stage=eda`` (figures + ``eda_summary.json``) and to
``reports/eda/`` (git-ignored).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from src.config import Config
from src.constants import (
    ACC6_LIMIT,
    ACC16_LIMIT,
    ACTIVITY_NAMES,
    IMU_LOCATIONS,
    IMU_RATE_HZ,
    PROTOCOL_ACTIVITIES,
    RAW_SENSOR_COLUMNS,
)
from src.utils import get_logger, kv, write_json

logger = get_logger(__name__)

SATURATION_SHARE = 0.98
SPECTRUM_WINDOW = 512


def recording_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Aggregate statistics of one raw recording.

    Args:
        df: One raw recording (single subject/session).

    Returns:
        Row counts, duration, NaN counts, timestamp-step statistics, heart-rate
        sampling and accelerometer saturation counts.
    """
    ts = df["timestamp"].to_numpy()
    steps = np.diff(ts)
    hr_rows = np.flatnonzero(df["heart_rate"].notna().to_numpy())
    stats: dict[str, Any] = {
        "rows": len(df),
        "duration_min": float((ts[-1] - ts[0]) / 60) if len(ts) else 0.0,
        "nan_counts": df[RAW_SENSOR_COLUMNS].isna().sum().astype(int).to_dict(),
        "duplicate_timestamps": int((steps == 0).sum()),
        "backward_timestamps": int((steps < 0).sum()),
        "gaps_over_0.05s": int((steps > 0.05).sum()),
        "gaps_over_1s": int((steps > 1.0).sum()),
        "max_gap_s": float(steps.max()) if len(steps) else 0.0,
        "hr_rows": len(hr_rows),
        "hr_median_interval_rows": float(np.median(np.diff(hr_rows))) if len(hr_rows) > 1 else float("nan"),
        "rows_per_activity": df["activity_id"].value_counts().sort_index().astype(int).to_dict(),
    }
    for loc in IMU_LOCATIONS:
        for kind, limit in (("acc6", ACC6_LIMIT), ("acc16", ACC16_LIMIT)):
            cols = [f"{loc}_{kind}_{a}" for a in "xyz"]
            stats[f"{loc}_{kind}_saturated"] = int((df[cols].abs() >= SATURATION_SHARE * limit).sum().sum())
    return stats


def mean_spectrum_by_activity(df: pd.DataFrame, column_prefix: str = "ankle_acc16") -> dict[int, np.ndarray]:
    """Sum of power spectra of the acceleration magnitude per activity (aggregate only).

    Args:
        df: One raw recording.
        column_prefix: Triad to use.

    Returns:
        ``{activity_id: summed power spectrum}`` plus the window count under key ``-1``.
    """
    mag = np.sqrt(sum(df[f"{column_prefix}_{a}"].to_numpy() ** 2 for a in "xyz"))
    labels = df["activity_id"].to_numpy()
    sums: dict[int, np.ndarray] = {}
    counts: dict[int, int] = defaultdict(int)
    for start in range(0, len(df) - SPECTRUM_WINDOW + 1, SPECTRUM_WINDOW):
        seg, lab = mag[start : start + SPECTRUM_WINDOW], labels[start : start + SPECTRUM_WINDOW]
        if lab[0] == 0 or (lab != lab[0]).any() or np.isnan(seg).any():
            continue
        power = np.abs(np.fft.rfft(seg - seg.mean())) ** 2
        sums[int(lab[0])] = sums.get(int(lab[0]), 0) + power
        counts[int(lab[0])] += 1
    return {a: sums[a] / counts[a] for a in sums} | {-1: np.array([sum(counts.values())])}


def _heatmap(minutes: pd.DataFrame) -> Figure:
    """Minutes per subject x activity."""
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(minutes, annot=True, fmt=".1f", cmap="viridis", ax=ax, cbar_kws={"label": "minutes"})
    ax.set_title("Protocol data: minutes per subject and activity")
    ax.set_xlabel("activity")
    ax.set_ylabel("subject")
    fig.tight_layout()
    return fig


def _bar(series: pd.Series, title: str, xlabel: str, log: bool = False) -> Figure:
    """Simple horizontal bar chart."""
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(series) + 1.5))
    series.plot.barh(ax=ax, color="#3b6ea5", logx=log)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    return fig


def run_eda(cfg: Config, allow_dirty: bool = False) -> dict[str, Any]:
    """Compute EDA aggregates and figures for all raw recordings and log them.

    Args:
        cfg: Resolved config (paths, tracking).
        allow_dirty: Allow uncommitted code.

    Returns:
        The EDA summary.
    """
    from src.tracking import start_run

    raw_dir = Path(cfg.paths.raw_dir)
    files = sorted(raw_dir.glob("*/subject*.parquet"))
    per_recording: dict[str, dict[str, Any]] = {}
    minutes: dict[int, dict[str, float]] = defaultdict(dict)
    hr_by_activity: dict[str, list[np.ndarray]] = defaultdict(list)
    spectra: dict[int, list[np.ndarray]] = defaultdict(list)
    channel_min = pd.Series(np.inf, index=RAW_SENSOR_COLUMNS)
    channel_max = pd.Series(-np.inf, index=RAW_SENSOR_COLUMNS)
    sat_by_activity: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_rows_by_activity: dict[str, int] = defaultdict(int)

    for path in files:
        df = pd.read_parquet(path)
        session, subject = path.parent.name, int(df["subject_id"].iloc[0])
        key = f"{session}/{subject}"
        per_recording[key] = recording_stats(df)
        channel_min = np.minimum(channel_min, df[RAW_SENSOR_COLUMNS].min())
        channel_max = np.maximum(channel_max, df[RAW_SENSOR_COLUMNS].max())
        if session != "protocol":
            continue
        for activity, rows in per_recording[key]["rows_per_activity"].items():
            minutes[subject][ACTIVITY_NAMES[int(activity)]] = rows / IMU_RATE_HZ / 60
        for activity, group in df.loc[df["heart_rate"].notna()].groupby("activity_id"):
            hr_by_activity[ACTIVITY_NAMES[int(activity)]].append(group["heart_rate"].to_numpy())
        for activity, group in df.groupby("activity_id"):
            name = ACTIVITY_NAMES[int(activity)]
            total_rows_by_activity[name] += len(group)
            for kind, limit in (("acc6", ACC6_LIMIT), ("acc16", ACC16_LIMIT)):
                cols = [f"{loc}_{kind}_{a}" for loc in IMU_LOCATIONS for a in "xyz"]
                hit = (group[cols].abs() >= SATURATION_SHARE * limit).any(axis=1)
                sat_by_activity[name][kind] += int(hit.sum())
        for activity, spectrum in mean_spectrum_by_activity(df).items():
            if activity >= 0:
                spectra[activity].append(spectrum)
        logger.info("eda recording", extra=kv(subject=subject, session=session, rows=len(df)))

    order = [ACTIVITY_NAMES[a] for a in (0, *PROTOCOL_ACTIVITIES)]
    minutes_df = pd.DataFrame(minutes).T.reindex(columns=order).fillna(0.0).sort_index()
    total_rows = sum(r["rows"] for r in per_recording.values())
    nan_totals = pd.DataFrame({k: v["nan_counts"] for k, v in per_recording.items()}).sum(axis=1)
    nan_rate = (nan_totals / total_rows).sort_values(ascending=False)
    missing = {
        int(s): sorted(a for a in PROTOCOL_ACTIVITIES if minutes_df.loc[s, ACTIVITY_NAMES[a]] == 0)
        for s in minutes_df.index
    }
    sat_rate = pd.DataFrame(
        {
            k: {kind: n / max(total_rows_by_activity[k], 1) for kind, n in v.items()}
            for k, v in sat_by_activity.items()
        }
    ).T.reindex([a for a in order if a in sat_by_activity])

    summary: dict[str, Any] = {
        "recordings": per_recording,
        "total_rows": int(total_rows),
        "protocol_minutes_per_subject_activity": minutes_df.round(2).to_dict(orient="index"),
        "protocol_minutes_per_activity": minutes_df.sum().round(2).to_dict(),
        "subjects_missing_protocol_activities": missing,
        "nan_rate_per_column": nan_rate.round(5).to_dict(),
        "channel_min": channel_min.round(3).to_dict(),
        "channel_max": channel_max.round(3).to_dict(),
        "saturation_rate_by_activity": sat_rate.round(5).to_dict(orient="index"),
        "heart_rate_by_activity": {
            k: {
                "median": float(np.median(np.concatenate(v))),
                "p10": float(np.percentile(np.concatenate(v), 10)),
                "p90": float(np.percentile(np.concatenate(v), 90)),
                "n": int(sum(len(x) for x in v)),
            }
            for k, v in hr_by_activity.items()
        },
    }

    figures: dict[str, Figure] = {"figures/minutes_subject_activity.png": _heatmap(minutes_df)}
    figures["figures/class_distribution.png"] = _bar(
        minutes_df.drop(columns="transient").sum().sort_values(), "Protocol minutes per activity", "minutes"
    )
    figures["figures/nan_rate_per_column.png"] = _bar(
        nan_rate.head(30).iloc[::-1], "NaN rate (top 30 columns)", "share of rows"
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    hr_frame = pd.DataFrame(
        [(k, v) for k, arrs in hr_by_activity.items() for v in np.concatenate(arrs)[::5]],
        columns=["activity", "hr"],
    )
    sns.boxplot(
        data=hr_frame,
        x="activity",
        y="hr",
        order=[a for a in order if a in hr_by_activity],
        ax=ax,
        fliersize=1,
    )
    ax.set_title("Heart rate per activity (protocol)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    figures["figures/heart_rate_by_activity.png"] = fig
    fig, ax = plt.subplots(figsize=(10, 4))
    sat_rate.plot.bar(ax=ax)
    ax.set_title(f"Share of rows with any accelerometer axis >= {SATURATION_SHARE:.0%} of its range")
    ax.set_ylabel("share of rows")
    fig.tight_layout()
    figures["figures/accelerometer_saturation.png"] = fig
    freqs = np.fft.rfftfreq(SPECTRUM_WINDOW, d=1 / IMU_RATE_HZ)
    fig, ax = plt.subplots(figsize=(10, 5))
    for activity in PROTOCOL_ACTIVITIES:
        if spectra.get(activity):
            ax.semilogy(freqs, np.mean(spectra[activity], axis=0), label=ACTIVITY_NAMES[activity], lw=1)
    ax.set_xlim(0, 15)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("mean power")
    ax.set_title("Average spectrum of ankle acceleration magnitude per activity")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    figures["figures/ankle_spectrum_by_activity.png"] = fig

    out = Path(cfg.paths.reports_dir) / "eda"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "eda_summary.json", summary)
    with start_run(cfg, stage="eda", run_name="eda-raw", allow_dirty=allow_dirty) as tracker:
        tracker.log_metrics({"total_rows": float(total_rows), "recordings": float(len(per_recording))})
        tracker.log_dict(summary, "eda_summary.json")
        for name, figure in figures.items():
            figure.savefig(out / Path(name).name, dpi=120)
            tracker.log_figure(figure, name)
    return summary
