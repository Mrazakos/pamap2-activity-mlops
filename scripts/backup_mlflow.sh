#!/usr/bin/env bash
# Back up the interim MLflow store (/shared has no backup) to the laptop.
# Run ON THE LAPTOP (Git Bash / WSL / Linux / macOS):
#   scripts/backup_mlflow.sh [DEST]        default DEST: ~/mlflow-backup/pamap2
# Uses rsync if available, otherwise scp. The SQLite file is copied from a consistent
# snapshot made with `sqlite3 .backup`, so a backup taken while jobs log is not torn.
set -euo pipefail

REMOTE="${REMOTE:-deep21}"
SRC="${SRC:-/shared/shmqak/mlflow}"
DEST="${1:-$HOME/mlflow-backup/pamap2}"
STAMP="$(date +%Y%m%d-%H%M%S)"
mkdir -p "$DEST"

ssh "$REMOTE" "cd '$SRC' && python3 -c \"import sqlite3; s=sqlite3.connect('mlflow.db'); d=sqlite3.connect('mlflow.snapshot.db'); s.backup(d); d.close(); s.close()\""
if command -v rsync > /dev/null 2>&1; then
    rsync -av --partial "$REMOTE:$SRC/artifacts/" "$DEST/artifacts/"
    rsync -av "$REMOTE:$SRC/mlflow.snapshot.db" "$DEST/mlflow-$STAMP.db"
else
    scp -r "$REMOTE:$SRC/artifacts" "$DEST/"
    scp "$REMOTE:$SRC/mlflow.snapshot.db" "$DEST/mlflow-$STAMP.db"
fi
cp "$DEST/mlflow-$STAMP.db" "$DEST/mlflow.db"
echo "backup written to $DEST (database snapshot mlflow-$STAMP.db)"
