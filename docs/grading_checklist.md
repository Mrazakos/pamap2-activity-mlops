# Grading checklist

One row per rubric line, pointing at the evidence. Rows marked *pending* are completed with
MLflow run ids once the runs exist.

| # | Requirement | Evidence |
|---|---|---|
| 1a | Task, data, target, metrics | [Overview](index.md), README |
| 1b | Repository structure, `.gitignore` | README "Repository layout", `.gitignore` |
| 1c | Reproducible environment | `requirements.txt` (exact pins), `requirements-lock.txt`, [Environment](environment.md), README setup |
| 2a | Data loading + structure | `src/download.py`, `src/schemas.py::RAW_SCHEMA`, [Data](data.md) |
| 2b | EDA | `src/eda.py`, MLflow run `1a9feb20c1254b23a5c3b421497273de` (`stage=eda`, job 10149), findings in [Data](data.md) |
| 2c | Quality issues found **and** fixed | `src/clean.py`, counts per rule in [Data](data.md) (MLflow runs `7dbe54f1…` v1, `92db93c0…` v2, `stage=clean`), `tests/test_clean.py` |
| 2d | Re-runnable preprocessing | `dvc.yaml` stages `clean`, `split_window`, `features` |
| 2e | Split, leakage prevention | `src/split.py`, `windows_schema` leakage check, `tests/test_split.py`, `tests/test_windows.py` |
| 3a | Baseline | `src/models/baseline.py`, `src/train.py::run_baseline`, DVC stage `baseline` (job 10154), MLflow run `4dd27f1a3324468f953ad071965d110c`: val macro-F1 0.9208 |
| 3b | Additional model, configurable training | `src/models/cnn.py`, `src/models/cnn_gru.py`, `configs/params.yaml`, `--set` overrides |
| 4a | MLflow integration | `src/tracking.py` |
| 4b | Hyperparameters logged automatically | `start_run` logs the flattened resolved config + `resolved_config.json` |
| 4c | Train + validation metrics | `train_loss_step`, `train_*`/`val_*` per epoch (`src/train.py::fit`) |
| 4d | Artifacts | bundle, confusion matrices, learning curves, reports, feature importances, lock file, Slurm log |
| 4e | ≥ 4 runs, logical HPO | [Experiments](experiments.md), [Results](results.md) (pending) |
| 5a | Modular code | `src/` modules, single `plan_windows` code path |
| 5b | Docstrings | ruff `D` rules (Google convention), [API reference](api.md) |
| 5c | Type hints | ruff `ANN` rules |
| 5d | Readability | ruff `E,F,I,UP,B,SIM,RUF,N` |
| 5e | Ruff + pre-commit | `pyproject.toml`, `.pre-commit-config.yaml` |
| 5f | Structured logging | `src/utils.py` (`key=value` text or JSON lines, run/stage context) |
| 6a | ≥ 3 unit tests | `tests/test_clean.py`, `test_windows.py`, `test_features.py`, `test_split.py`, `test_utils.py`, `test_config.py` |
| 6b | Pandera, valid + invalid | `tests/test_schemas.py` |
| 6c | ≥ 3 model-behaviour tests | `tests/test_model_behaviour.py` (INV noise, INV shift, MFT, DIR) + `tests/test_models.py` |
| 6d | Integration test (CLI) | `tests/test_cli.py` |
| 6e | Tests in pre-commit | `.pre-commit-config.yaml` hook `pytest` |
| 7a | DVC | `dvc.yaml`, `dvc.lock`, `data/raw.dvc` |
| 7b | ≥ 2 data versions | git tags `data-v1` (naive), `data-v2` (full quality pipeline), table in [Data](data.md#data-versions) |
| 7c | External storage + restore | DVC remote on `/shared`, restore from an empty cache in job 10152, output in [Reproducibility](reproducibility.md) |
| 7d | Versioned pipeline, traceability | lineage tags `git.commit`, `dvc.data_hash`, `config.hash`, `reproduce.command` |
| 8a | Automatic CI | `.github/workflows/ci.yml` (push + PR) |
| 8b | Lint + tests in CI | jobs `lint`, `test` |
| 8c | Reporting | JUnit check run, step summary, uploaded reports, README badges |
| 9a | Comparisons | [Results](results.md) (pending) |
| 9b | Weaknesses | [Weaknesses](weaknesses.md) (pending) |
