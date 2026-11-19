#!/usr/bin/env bash
# MLflow UI as a short CPU-only job (alternative to `make mlflow-ui` on deep21).
#   sbatch slurm/mlflow_ui.sh
#   grep 'tunnel' slurm/logs/mlflow-ui-<jobid>.out    # prints the ssh command for the laptop
#SBATCH --job-name=mlflow-ui
#SBATCH --account=mlops_2026_fall
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out

source slurm/common.sh
PORT=$((5000 + RANDOM % 1000))
echo "tunnel from the laptop:  ssh -L 5000:$(hostname):${PORT} deep21   then open http://localhost:5000"
mlflow ui --backend-store-uri "$MLFLOW_TRACKING_URI" --default-artifact-root "$MLFLOW_ARTIFACT_ROOT" \
    --host 0.0.0.0 --port "$PORT"
