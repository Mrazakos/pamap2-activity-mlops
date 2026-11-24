#!/usr/bin/env bash
# Submit one HPO round: create the MLflow parent run (light, on deep21), then a Slurm job array.
#   scripts/submit_sweep.sh configs/sweeps/round_a_lr.txt "Round A: which LR trains the CNN best?" [--time=01:00:00]
# Course QOS: at most 2 running and 10 submitted jobs per user, and every array task counts.
# %2 keeps the array within the running limit; a round that would exceed the submit limit is
# refused before its parent run is created (SUBMIT_LIMIT overrides the 10).
set -euo pipefail

SWEEP_FILE="${1:?usage: submit_sweep.sh SWEEP_FILE HYPOTHESIS [sbatch options...]}"
HYPOTHESIS="${2:?give the hypothesis of this round}"
shift 2
NAME="$(basename "$SWEEP_FILE" .txt)"
COMMAND="${COMMAND:-train}"
N=$(grep -cvE '^\s*(#|$)' "$SWEEP_FILE")

QUEUED=$(squeue --me -h -r | wc -l)
if [ $((QUEUED + N)) -gt "${SUBMIT_LIMIT:-10}" ]; then
    echo "refused: $N tasks + $QUEUED already queued > ${SUBMIT_LIMIT:-10} (QOS submit limit); retry later" >&2
    exit 1
fi

# conda's activation scripts read the positional arguments, so the sbatch options are stashed.
SBATCH_ARGS=("$@")
set --
set +u
source /shared/anaconda/anaconda/bin/activate
conda activate "${CONDA_ENV:-mlops}"
source slurm/mlflow_env.sh
set -u
set -- "${SBATCH_ARGS[@]}"

PARENT_RUN_ID=$(python -m src.main sweep-init --name "$NAME" --hypothesis "$HYPOTHESIS" 2>/dev/null | tail -1)
echo "parent run: $PARENT_RUN_ID  ($N configs from $SWEEP_FILE)"

mkdir -p slurm/logs
SCRIPT=slurm/sweep.sh
[ "$COMMAND" = "baseline" ] && SCRIPT=slurm/sweep_cpu.sh
SWEEP_FILE="$SWEEP_FILE" PARENT_RUN_ID="$PARENT_RUN_ID" COMMAND="$COMMAND" \
    sbatch --job-name="$NAME" --array="0-$((N - 1))%2" "$@" "$SCRIPT"
