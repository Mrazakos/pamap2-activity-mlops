# Reproducibility

## Determinism measures

| Source of randomness / variation | Control |
|---|---|
| Python, NumPy, PyTorch, CUDA RNGs | `set_seed(cfg.seed)` (`src/utils.py`) |
| cuDNN / cuBLAS kernels | `torch.use_deterministic_algorithms(True)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` exported **before** Python starts (`slurm/common.sh`) |
| Hash randomisation | `PYTHONHASHSEED=0` exported in the job script |
| Batch order | each epoch uses a fresh `DataLoader` with a `torch.Generator` seeded `seed + epoch`; `worker_init_fn` seeds workers; fixed `num_workers` |
| Resume after TIMEOUT | checkpoint after every epoch incl. all RNG states; identical results with or without interruption |
| scikit-learn | fixed `random_state`; RandomForest results do not depend on `n_jobs` |
| Data | archive SHA-256 pinned; DVC hashes for every stage output; sorted file lists |
| Hardware | all tracked GPU runs on `rtx2080ti`; `gpu.name`, driver, CUDA, cuDNN and torch versions are run tags |

## External storage and restoration (DVC)

* Remote `shared` = `/shared/shmqak/dvc-remote/pamap2-activity-mlops`: on the cluster disk,
  outside the repository, configured in `.dvc/config`.
* `scripts/verify_restore.sh` (job: `sbatch --job-name=verify-restore slurm/run_shell.sh
  scripts/verify_restore.sh <branch>`) makes a fresh clone from GitHub with an **empty** DVC cache,
  pulls from the remote, then switches between the data tags.

Output of job 10152 (2026-11-19, node deep23, commit `f35bdfe`):

```text
== fresh clone of feat/data-versions into /shared/shmqak/tmp/restore-check-313265
  commit f35bdfe, local cache: 0 entries
== dvc pull (empty cache, everything comes from the remote)
Data and pipelines are up to date.
  data_version=v2 clean_rows=1941968 channels=28 segments=156 windows={'test': 949, 'train': 5425, 'val': 999}
== git checkout data-v1 && dvc pull
Data and pipelines are up to date.
  data/clean hash in dvc.lock: c7238dce9fe90911e1d3a6140a5df78d.dir
  clean.parquet md5 on disk:   4235cfddff22de00e1025029b8a7f573
  data_version=v1 clean_rows=1942872 channels=52 segments=9 windows={'test': 975, 'train': 5573, 'val': 1027}
== git checkout data-v2 && dvc pull
Data and pipelines are up to date.
  data/clean hash in dvc.lock: fd82a96c985b17672da647b989f1ee5e.dir
  clean.parquet md5 on disk:   edad68e95a3362af08f1a86e4076cb85
  data_version=v2 clean_rows=1941968 channels=28 segments=156 windows={'test': 949, 'train': 5425, 'val': 999}
== restore check passed
```

`Data and pipelines are up to date` means every restored file matches the hash in that
revision's `dvc.lock`.

Manual equivalent: `git checkout data-v1 && dvc pull`, then `git checkout data-v2 && dvc pull`.

## Reproducing a single run

Every MLflow run has a `reproduce.command` tag, e.g.

```bash
git checkout <commit> && dvc pull && python -m src.main train --set train.lr=0.0003
```

On the cluster, submit the same arguments through the job script:
`sbatch slurm/train.sh --set train.lr=0.0003`.

`python -m src.main verify-repro <run_id>` (as a job: `sbatch slurm/verify_repro.sh <run_id>`)
re-runs a run from its logged config and asserts identical final metrics, identical metric
histories and an identical model-weights hash.
