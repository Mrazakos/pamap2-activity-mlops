# Data

## Dataset

PAMAP2 Physical Activity Monitoring (Reiss & Stricker, 2012), UCI ML Repository id 231,
licensed CC BY 4.0. Nine subjects (101-109) wore three Colibri IMUs (wrist of the dominant arm,
chest, dominant-side ankle; 100 Hz) and a heart-rate chest strap (~9 Hz).

* **Archive:** 656 MB, SHA-256 `76b3580bd5a804121f507717cb498c66a312ef5c774f4a78ac470e6558add352`,
  pinned in `configs/params.yaml` and verified on every download.
* **Files:** 14 recordings, one per subject and session: `Protocol` for 101-109, `Optional`
  for 101, 105, 106, 108, 109. 3,850,505 rows in total. Converted to Parquet with the documented
  column names plus `subject_id` and `session` (`src/download.py`). Every file passes the raw
  Pandera schema.
* **Columns (verified against `readme.pdf`):** timestamp (s), `activity_id`, heart rate (bpm),
  then 17 columns per IMU (hand, chest, ankle): temperature, ±16 g accelerometer, ±6 g
  accelerometer, gyroscope, magnetometer, and 4 orientation columns that the documentation
  marks as *invalid in this data collection*.
* **Labels:** 18 activities plus `0` = transient, which the authors say "should be discarded in
  any kind of analysis". This project uses the **12 protocol activities** of the `Protocol`
  sessions.

## Exploratory analysis

All figures and `eda_summary.json` are logged to the MLflow run tagged `stage=eda`
(`python -m src.main eda`, a CPU-only job). Only aggregates appear here.

| Finding | Value |
|---|---|
| Protocol data per subject | 42-75 min for 101-108; **subject 109: 1.4 min, all rope jumping** |
| Transient (label 0) share of protocol time | 154.9 of 478.8 min (32.4 %); 13-31 min per subject |
| Activities missing | 103: running, cycling, Nordic walking, rope jumping; 104 and 107: rope jumping; 106: only 2.5 s of rope jumping (less than one window) |
| Rarest classes | rope jumping (8.2 min in total), running (16.4 min); walking and ironing about 40 min each |
| Heart rate | present on exactly every 11th row (9.09 Hz); medians from 74 bpm (lying) to 166 bpm (running, rope jumping) |
| NaN bursts (dropped wireless packets) | 0.11 % (chest) to 0.37 % (hand) of IMU rows |
| ±6 g accelerometer saturation | 3.9 % of running rows and 1.7 % of rope-jumping rows reach ≥ 98 % of its range (±16 g: 0.1 %) |
| Values beyond the ±16 g range | up to 160.5 m/s² (16 g = 156.9 m/s²) |
| Duplicated or backward timestamps | none in any recording |
| Timestamp gaps inside a raw file | none: the clock steps by exactly 0.01 s. Gaps appear only *after* transient and NaN rows are removed, which is where segments must be split |

## Data-quality issues: found, fixed and counted

Each rule is a small, unit-tested function in `src/clean.py`. Both profiles run through the same
code; the counts below come from `data/clean/meta.json` and the MLflow runs tagged `stage=clean`
(Protocol sessions, 2,872,533 raw rows).

| Issue (how it was found) | Rule | data-v1 (naive) | data-v2 (full) |
|---|---|---|---|
| Transient label 0 between activities (EDA, readme) | drop rows | 929,661 rows | 929,661 rows |
| Heart rate only on every 11th row (EDA) | v1: interpolate; v2: carry the last reading forward ≤ 2 s (causal) | 2,610,265 values filled | 2,610,147 values filled |
| NaN bursts from dropped packets (EDA) | v1: interpolate everything; v2: interpolate runs of ≤ 10 samples (0.1 s), drop rows in longer gaps | 483,701 values interpolated, 0 rows dropped | 236,016 values interpolated, **904 rows dropped** |
| ±6 g accelerometer saturates (EDA: 3.9 % of running rows) | v2 keeps only the ±16 g accelerometer | 6,496 saturated values kept | channel removed |
| Orientation columns invalid (readme) | v2 removes them | kept | removed |
| Temperature is a slow environmental signal, not movement (a design choice: it could act as a shortcut for the recording session) | v2 removes it | kept | removed |
| Spikes beyond the sensor range (EDA: up to 160.5 m/s²) | v2 clips to ±16 g, ±2000 °/s, 30-230 bpm | not clipped | **13 values clipped** |
| Discontinuities after rows are removed; activity boundaries | v2 starts a new segment at a step > 0.05 s or a label change; windows never cross segments | 9 segments (one per file) | **156 segments** |
| Windows that mix two activities | v1 keeps them with the majority label; v2 drops them | 7,575 windows | 7,373 windows |
| Duplicated timestamps (candidate) | v2 keeps the first | not checked | 0 found |
| Channels used by the models | | 52 | 28 (3 × [acc ±16 g, gyro, mag] + HR) |

The clean data is validated by the Pandera `clean` schema (no NaN, known labels, increasing
timestamps, sensor ranges for v2, one activity per segment) before it is written.

## Data versions

| Tag | Profile | `data/clean` hash | Windows (train / val / test) |
|---|---|---|---|
| `data-v1` | naive | `c7238dce9fe90911e1d3a6140a5df78d.dir` | 5,573 / 1,027 / 975 |
| `data-v2` | full quality pipeline | `fd82a96c985b17672da647b989f1ee5e.dir` | 5,425 / 999 / 949 |

Both versions were produced by `dvc repro` in Slurm CPU jobs and pushed to the DVC remote.
Restoring them is described in [Reproducibility](reproducibility.md).

## Split and leakage controls

**Rule, fixed before any training.** A validation or test subject must have performed every
one of the 12 activities for at least two windows (10.24 s), otherwise per-class metrics on
that subject are undefined. Eligible subjects are ranked by their *smallest* per-activity
duration, so the chosen subjects give every class as many evaluation windows as possible
(`src/split.py::rank_evaluation_candidates`; the ranking is part of `eda_summary.json`).

| Rank | Subject | Smallest per-activity duration |
|---|---|---|
| 1 | 101 → **test** | 129.1 s (rope jumping) |
| 2 | 102 → **validation** | 92.4 s (running) |
| 3 | 108 | 88.1 s (rope jumping) |
| 4 | 105 | 77.3 s (rope jumping) |

All other subjects (103-109 except 102) are training subjects. Single-subject validation and
test sets are noisy, so leave-one-subject-out results are reported as well.

* **Subject-wise split.** All windows of one person belong to exactly one split. This is enforced by
  `src/split.py`, the Pandera window schema (`no subject in two splits`) and `tests/test_split.py`.
* **Normalisation statistics are fitted on training subjects only** (`fit_normalization`,
  `tests/test_windows.py::test_normalization_is_fit_on_train_only`).
* **Overlapping windows stay inside one split**, because windows never cross a segment and
  segments never span subjects.
* **The test subject is evaluated once**, for the final models (DVC stage `evaluate`). All
  model selection uses the validation subject.
