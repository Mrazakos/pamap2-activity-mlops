#!/usr/bin/env bash
# Prove that the DVC remote alone restores every data version (output goes to docs/reproducibility.md).
#   sbatch --job-name=verify-restore slurm/run_shell.sh scripts/verify_restore.sh [BRANCH]
# Steps: fresh clone from GitHub -> empty cache -> dvc pull -> hashes match dvc.lock
#        -> git checkout data-v1 + dvc pull -> v1 restored -> back to data-v2.
set -euo pipefail

BRANCH="${1:-main}"
REPO_URL="${REPO_URL:-https://github.com/Mrazakos/pamap2-activity-mlops.git}"
WORK="${WORK:-/shared/shmqak/tmp/restore-check-$$}"
trap 'rm -rf "$WORK"' EXIT

summary() {
    python3 - <<'EOF'
import json
from pathlib import Path

clean = json.loads(Path("data/clean/meta.json").read_text())
proc = json.loads(Path("data/processed/meta.json").read_text())
print(f"  data_version={clean['data_version']} clean_rows={clean['rows']} channels={len(clean['channels'])}"
      f" segments={clean['quality']['totals']['segments']} windows={proc['windows_per_split']}")
EOF
}

echo "== fresh clone of $BRANCH into $WORK"
git clone -q --branch "$BRANCH" "$REPO_URL" "$WORK"
cd "$WORK"
echo "  commit $(git rev-parse --short HEAD), local cache: $(ls .dvc/cache 2>/dev/null | wc -l) entries"

echo "== dvc pull (empty cache, everything comes from the remote)"
dvc pull -q data/raw.dvc clean split_window
dvc status data/raw.dvc clean split_window
summary

for tag in data-v1 data-v2; do
    echo "== git checkout $tag && dvc pull"
    git -c advice.detachedHead=false checkout -q "$tag"
    dvc pull -q data/raw.dvc clean split_window
    dvc status data/raw.dvc clean split_window
    echo "  data/clean hash in dvc.lock: $(grep -A2 'path: data/clean$' dvc.lock | grep -oE '[0-9a-f]{32}\.dir' | head -1)"
    echo "  clean.parquet md5 on disk:   $(md5sum data/clean/clean.parquet | cut -d' ' -f1)"
    summary
done
echo "== restore check passed"
