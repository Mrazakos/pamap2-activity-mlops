# Data

## Dataset

PAMAP2 Physical Activity Monitoring (Reiss & Stricker, 2012), UCI ML Repository id 231,
licensed CC BY 4.0. Nine subjects (101-109) wore three Colibri IMUs (wrist of the dominant arm,
chest, dominant-side ankle) and a heart-rate chest strap. Recordings are in `Protocol/` (the
12-activity protocol) and `Optional/` (extra activities for some subjects).

The archive is downloaded by `python -m src.main download`; its SHA-256 is pinned in
`configs/params.yaml` and checked on every download.

## Leakage controls

* **Subject-wise split.** All windows of one person belong to exactly one split; enforced by
  `src/split.py`, the Pandera window schema and `tests/test_split.py`.
* **Normalisation statistics are fitted on training subjects only** (`fit_normalization`,
  tested in `tests/test_windows.py::test_normalization_is_fit_on_train_only`).
* **Overlapping windows exist only within one split** because windows never cross segments,
  and segments never span subjects.
* **Test subjects are evaluated once**, for the final models (`evaluate` stage); all model
  selection uses the validation subject.
