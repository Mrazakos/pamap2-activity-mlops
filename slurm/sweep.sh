#!/usr/bin/env bash
# One HPO round as a job array: task i runs line i of a sweep file as a nested MLflow child run.
# Submit through scripts/submit_sweep.sh, which creates the parent run and sets --array.
#   SWEEP_FILE=configs/sweeps/round_a_lr.txt PARENT_RUN_ID=<id> COMMAND=train sbatch --array=0-3%2 slurm/sweep.sh
#SBATCH --job-name=sweep
#SBATCH --account=mlops_2026_fall
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%A_%a.out

source slurm/common.sh
: "${SWEEP_FILE:?set SWEEP_FILE}" "${PARENT_RUN_ID:?set PARENT_RUN_ID}"
COMMAND="${COMMAND:-train}"

mapfile -t CONFIGS < <(grep -vE '^\s*(#|$)' "$SWEEP_FILE")
LINE="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "sweep task ${SLURM_ARRAY_TASK_ID}: ${LINE}"

ARGS=()
for override in $LINE; do
    ARGS+=(--set "$override")
done
python -m src.main "$COMMAND" --parent-run-id "$PARENT_RUN_ID" "${ARGS[@]}"
