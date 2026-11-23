#!/usr/bin/env bash
# Package the interim MLflow store (database snapshot + artifacts) into ONE archive for the grader.
# Run on deep21:   scripts/export_mlflow.sh [OUT_DIR]      default OUT_DIR: /shared/shmqak/exports
# The archive is written OUTSIDE the repository and must never be committed (*.mlflow-export.tar.gz is git-ignored).
# How to open it on another machine: docs/reproducibility.md, section "Opening the MLflow export".
set -euo pipefail

SRC="${SRC:-/shared/shmqak/mlflow}"
OUT_DIR="${1:-/shared/shmqak/exports}"
STAMP="$(date +%Y%m%d-%H%M%S)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$OUT_DIR" "$WORK/mlflow"
python3 - "$SRC/mlflow.db" "$WORK/mlflow/mlflow.db" <<'EOF'
import sqlite3
import sys

src, dst = sqlite3.connect(sys.argv[1]), sqlite3.connect(sys.argv[2])
src.backup(dst)
dst.close()
src.close()
EOF
cp -a "$SRC/artifacts" "$WORK/mlflow/artifacts"
cat > "$WORK/mlflow/ORIGINAL_ROOT" <<EOF
$SRC
EOF
cp "$(dirname "$0")/relocate_mlflow.py" "$WORK/mlflow/"

ARCHIVE="$OUT_DIR/pamap2-mlflow-$STAMP.mlflow-export.tar.gz"
tar -C "$WORK" -czf "$ARCHIVE" mlflow
sha256sum "$ARCHIVE" | tee "$ARCHIVE.sha256"
echo "archive: $ARCHIVE  ($(du -h "$ARCHIVE" | cut -f1))"
