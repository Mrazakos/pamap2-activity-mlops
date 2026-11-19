#!/usr/bin/env bash
# Leave-one-subject-out evaluation of the final config (9 folds as nested runs).
#   sbatch slurm/loso.sh --kind deep
#   sbatch --job-name=loso-baseline slurm/run_cpu.sh loso --kind baseline   # baseline needs no GPU
#SBATCH --job-name=loso
#SBATCH --account=mlops_2026_fall
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x-%j.out

source slurm/common.sh
python -m src.main loso "$@"
