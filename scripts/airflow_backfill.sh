#!/usr/bin/env bash
# Run alerts_etl for every month from FROM to TO (inclusive), oldest first, and wait.
# Airflow creates one run per data interval; each run's batch = its data_interval_start.
#   scripts/airflow_backfill.sh 2022-03 2026-09
set -euo pipefail
cd "$(dirname "$0")/.."
from=$1
to=$2
airflow() { docker compose exec -T airflow-scheduler airflow "$@"; }

airflow dags unpause alerts_etl >/dev/null
airflow backfill create --dag-id alerts_etl \
    --from-date "${from}-01T00:00:00+00:00" --to-date "${to}-01T00:00:00+00:00" \
    --reprocess-behavior completed --max-active-runs 1
echo "waiting for the backfill (one month at a time) ..."
while :; do
    sleep 15
    summary=$(airflow dags list-runs alerts_etl -o json 2>/dev/null | python3 -c '
import collections, json, sys
runs = [r for r in json.load(sys.stdin) if r["run_id"].startswith("backfill__")]
print(" ".join(f"{k}={v}" for k, v in sorted(collections.Counter(r["state"] for r in runs).items())))')
    echo "  $summary"
    [[ $summary != *queued* && $summary != *running* ]] && break
done
