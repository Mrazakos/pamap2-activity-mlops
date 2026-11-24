#!/usr/bin/env bash
# CPU-only smoke test: compute node -> internet, and compute node -> MLflow store (.env or /shared).
#   sbatch slurm/smoke_mlflow.sh
#SBATCH --job-name=smoke-mlflow
#SBATCH --account=mlops_2026_fall
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=slurm/logs/%x-%j.out

source slurm/common.sh

echo "--- internet check from compute node"
for url in https://archive.ics.uci.edu https://pypi.org/simple/ https://github.com; do
    echo "  $url -> HTTP $(curl -s -o /dev/null -m 15 -w '%{http_code}' "$url" || echo 000)"
done

echo "--- MLflow round-trip"
python - <<'EOF'
import os
import tempfile
from pathlib import Path

import mlflow

from src.tracking import get_or_create_experiment, with_retry

exp_id = get_or_create_experiment(os.environ["MLFLOW_EXPERIMENT"], os.environ.get("MLFLOW_ARTIFACT_ROOT") or None)
with mlflow.start_run(experiment_id=exp_id, run_name="smoke-compute-node") as run:
    with_retry(mlflow.set_tags, {"stage": "smoke", "slurm.node": os.uname().nodename})
    with_retry(mlflow.log_param, "smoke_param", 1)
    with_retry(mlflow.log_metric, "smoke_metric", 0.5)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "hello.txt"
        p.write_text("hello from a compute node\n")
        with_retry(mlflow.log_artifact, str(p))
    print("run_id:", run.info.run_id, "artifact_uri:", run.info.artifact_uri)
EOF
