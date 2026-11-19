#!/usr/bin/env bash
# One deep-model training run on the pinned GPU type (determinism needs one GPU model).
#   sbatch slurm/train.sh --set train.lr=3e-4
# Re-submitting the identical command after a TIMEOUT resumes from the last checkpoint.
#SBATCH --job-name=train
#SBATCH --account=mlops_2026_fall
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out

source slurm/common.sh
python -m src.main train "$@"
