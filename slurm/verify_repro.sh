#!/usr/bin/env bash
# Re-run a tracked run from its logged config and assert identical metrics + weights hash.
# Must run on the same GPU type as the original (rtx2080ti) and at the run's git commit.
#   sbatch slurm/verify_repro.sh <run_id>
#SBATCH --job-name=verify-repro
#SBATCH --account=mlops_2026_fall
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/%x-%j.out

source slurm/common.sh
python -m src.main verify-repro "$1"
