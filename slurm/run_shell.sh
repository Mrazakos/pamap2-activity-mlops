#!/usr/bin/env bash
# Generic CPU-only job that runs a shell script inside the project environment.
#   sbatch --job-name=verify-restore slurm/run_shell.sh scripts/verify_restore.sh main
#SBATCH --job-name=shell
#SBATCH --account=mlops_2026_fall
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out

source slurm/common.sh
bash "$@"
