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
