#!/usr/bin/env bash
# Generic CPU-only job (no GPU budget): runs `python -m src.main <args>`.
#   sbatch --job-name=download slurm/run_cpu.sh download
#   sbatch --job-name=eda      slurm/run_cpu.sh eda
#   sbatch --job-name=baseline slurm/run_cpu.sh baseline --set baseline.n_estimators=500
#SBATCH --job-name=cpu
#SBATCH --account=mlops_2026_fall
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out

source slurm/common.sh
python -m src.main "$@"
