# Video outline (≤ 10 minutes)

| Time | Segment |
|---|---|
| 0:00-0:45 | Problem, task, metrics (README / overview page) |
| 0:45-1:45 | Repository tour: `src/`, `configs/params.yaml`, `dvc.yaml`, `slurm/`, `tests/` |
| 1:45-2:45 | `dvc pull` restore on a fresh clone; `git checkout data-v1 && dvc checkout` and back |
| 2:45-3:45 | Slurm: `sbatch slurm/repro.sh`, `squeue --me`, log with structured lines |
| 3:45-5:30 | MLflow UI: params, learning curves, artifacts, lineage tags, `reproduce.command` |
| 5:30-6:45 | HPO reasoning round by round; the smartwatch result |
| 6:45-7:45 | Tests and pre-commit (a commit blocked by a failing hook) |
| 7:45-8:30 | CI: green pipeline and the deliberately failing PR |
| 8:30-10:00 | Results: baseline vs final, LOSO, data-v1 vs v2, weaknesses |
