#!/usr/bin/env bash
# Submit one HPO round: create the MLflow parent run (light, on deep21), then a Slurm job array.
#   scripts/submit_sweep.sh configs/sweeps/round_a_lr.txt "Round A: which LR trains the CNN best?" [--time=01:00:00]
# At most 2 array tasks run at once (cluster limit), enforced with %2.
set -euo pipefail

SWEEP_FILE="${1:?usage: submit_sweep.sh SWEEP_FILE HYPOTHESIS [sbatch options...]}"
HYPOTHESIS="${2:?give the hypothesis of this round}"
shift 2
NAME="$(basename "$SWEEP_FILE" .txt)"
COMMAND="${COMMAND:-train}"
N=$(grep -cvE '^\s*(#|$)' "$SWEEP_FILE")

set +u
source /shared/anaconda/anaconda/bin/activate
conda activate "${CONDA_ENV:-mlops}"
set -u
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:////shared/shmqak/mlflow/mlflow.db}"
export MLFLOW_ARTIFACT_ROOT="${MLFLOW_ARTIFACT_ROOT:-/shared/shmqak/mlflow/artifacts}"
export MLFLOW_EXPERIMENT="${MLFLOW_EXPERIMENT:-shmqak-pamap2-har}"

PARENT_RUN_ID=$(python -m src.main sweep-init --name "$NAME" --hypothesis "$HYPOTHESIS" 2>/dev/null | tail -1)
echo "parent run: $PARENT_RUN_ID  ($N configs from $SWEEP_FILE)"

mkdir -p slurm/logs
SCRIPT=slurm/sweep.sh
[ "$COMMAND" = "baseline" ] && SCRIPT=slurm/sweep_cpu.sh
SWEEP_FILE="$SWEEP_FILE" PARENT_RUN_ID="$PARENT_RUN_ID" COMMAND="$COMMAND" \
    sbatch --job-name="$NAME" --array="0-$((N - 1))%2" "$@" "$SCRIPT"
