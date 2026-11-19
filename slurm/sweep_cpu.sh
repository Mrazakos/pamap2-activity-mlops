#!/usr/bin/env bash
# CPU variant of slurm/sweep.sh for baseline sweeps (no GPU budget).
#SBATCH --job-name=sweep-cpu
#SBATCH --account=mlops_2026_fall
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%A_%a.out

source slurm/common.sh
: "${SWEEP_FILE:?set SWEEP_FILE}" "${PARENT_RUN_ID:?set PARENT_RUN_ID}"
COMMAND="${COMMAND:-baseline}"

mapfile -t CONFIGS < <(grep -vE '^\s*(#|$)' "$SWEEP_FILE")
LINE="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "sweep task ${SLURM_ARRAY_TASK_ID}: ${LINE}"

ARGS=()
for override in $LINE; do
    ARGS+=(--set "$override")
done
python -m src.main "$COMMAND" --parent-run-id "$PARENT_RUN_ID" "${ARGS[@]}"
