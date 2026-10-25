#!/usr/bin/env bash
# Rebuild the DVC remote on a machine whose MinIO is empty (fresh clone, new laptop).
# Downloads the two upstream files at their pinned commits, proves they are the
# exact bytes data/raw.dvc describes, and pushes them to the remote.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE=https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset
OFFICIAL_COMMIT=58ee75bc20113181b9ddf8029d4cbf3a62d8cd10   # 2026-09-08, last official update
VOLUNTEER_COMMIT=346bcff708d02a0c908bcb26e0b8600793e8fc05  # 2026-10-24

mkdir -p data/raw
curl -fsSL -o data/raw/official_data_en.csv "$BASE/$OFFICIAL_COMMIT/datasets/official_data_en.csv"
curl -fsSL -o data/raw/volunteer_data_en.csv "$BASE/$VOLUNTEER_COMMIT/datasets/volunteer_data_en.csv"

# Re-adding identical bytes reproduces data/raw.dvc exactly; any difference is a failure.
uv run dvc add data/raw
if ! git diff --quiet -- data/raw.dvc; then
    git diff -- data/raw.dvc
    git checkout -- data/raw.dvc
    echo "downloaded files differ from data/raw.dvc; nothing pushed" >&2
    exit 1
fi
uv run dvc push data/raw.dvc
uv run python scripts/checksum.py data/raw
