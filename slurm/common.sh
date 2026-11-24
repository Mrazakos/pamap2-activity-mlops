#!/usr/bin/env bash
# Sourced by every job script (not submitted directly).
# Sets up conda, determinism variables (before Python starts) and the MLflow store.
set -eo pipefail

echo "Job ${SLURM_JOB_ID:-local} (${SLURM_JOB_NAME:-manual}) on $(hostname) at $(date -Is)"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# conda's activation scripts reference unset variables (so no `set -u` here) and read the
# positional arguments of the sourcing script, so the job's arguments are stashed meanwhile.
_JOB_ARGS=("$@")
set --
source /shared/anaconda/anaconda/bin/activate
conda activate "${CONDA_ENV:-mlops}"
set -- "${_JOB_ARGS[@]}"
unset _JOB_ARGS

# --- reproducibility: must be set before the Python interpreter starts
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# --- experiment tracking: university server from .env, else the interim SQLite store
source slurm/mlflow_env.sh

# --- the job's own log file, attached to the MLflow run at the end
if [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
    export SLURM_LOG_FILE="$PWD/slurm/logs/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
else
    export SLURM_LOG_FILE="$PWD/slurm/logs/${SLURM_JOB_NAME:-manual}-${SLURM_JOB_ID:-0}.out"
fi

echo "python: $(command -v python)  git: $(git rev-parse --short HEAD 2>/dev/null || echo none)"
if command -v nvidia-smi > /dev/null 2>&1 && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

finish() {
    local status=$?
    echo "Job finished at $(date -Is) with exit code ${status}"
}
trap finish EXIT
