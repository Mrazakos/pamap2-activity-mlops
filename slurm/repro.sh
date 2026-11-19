#!/usr/bin/env bash
# Runs `dvc repro [targets]` inside a job. CPU stages by default (no GPU budget):
#   sbatch slurm/repro.sh baseline                                  # clean -> split_window -> features -> baseline
#   sbatch --gres=gpu:rtx2080ti:1 --job-name=repro-gpu slurm/repro.sh  # everything incl. train + evaluate
#SBATCH --job-name=repro
#SBATCH --account=mlops_2026_fall
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out

source slurm/common.sh
dvc repro "$@"
dvc status
git status --short dvc.lock reports || true
