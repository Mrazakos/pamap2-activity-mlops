#!/usr/bin/env bash
# Full test suite on a compute node, including the model-behaviour tests against models/deep.
#   sbatch slurm/tests.sh
#SBATCH --job-name=tests
#SBATCH --account=mlops_2026_fall
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x-%j.out

source slurm/common.sh
python -m pytest -ra --cov=src --cov=scripts --cov-report=term "$@"
