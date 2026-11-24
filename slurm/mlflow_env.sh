#!/usr/bin/env bash
# MLflow store selection, sourced by slurm/common.sh, scripts/submit_sweep.sh and the Makefile.
#
# * `.env` (git-ignored) wins: on the cluster it points at the university server
#   (MLFLOW_TRACKING_URI=http://..., MLFLOW_TRACKING_USERNAME / _PASSWORD).
#   Values must be single-quoted if they contain spaces or `$` (read by bash and python-dotenv).
# * Without `.env`, the interim SQLite store on /shared is used.
# * MLFLOW_ARTIFACT_ROOT only applies to the SQLite store; a tracking server decides the
#   artifact location itself (artifacts are uploaded through the server).

MLFLOW_INTERIM_URI="sqlite:////shared/shmqak/mlflow/mlflow.db"
MLFLOW_INTERIM_ARTIFACTS="/shared/shmqak/mlflow/artifacts"

if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-$MLFLOW_INTERIM_URI}"
if [[ "$MLFLOW_TRACKING_URI" == sqlite:* ]]; then
    export MLFLOW_ARTIFACT_ROOT="${MLFLOW_ARTIFACT_ROOT:-$MLFLOW_INTERIM_ARTIFACTS}"
else
    unset MLFLOW_ARTIFACT_ROOT
fi
export MLFLOW_EXPERIMENT="${MLFLOW_EXPERIMENT:-shmqak-pamap2-har}"
echo "mlflow: ${MLFLOW_TRACKING_URI} experiment=${MLFLOW_EXPERIMENT}"
