#!/usr/bin/env bash
# Trigger any DAG once and wait for the result:  scripts/airflow_trigger.sh ingest_telegram
set -euo pipefail
cd "$(dirname "$0")/.."
dag=$1
airflow() { docker compose exec -T airflow-scheduler airflow "$@"; }

run_id="manual__$(date -u +%Y%m%dT%H%M%S)"
airflow dags unpause "$dag" >/dev/null
airflow dags trigger "$dag" --run-id "$run_id" >/dev/null
echo "triggered $dag $run_id"

state=queued
while [[ $state != success && $state != failed ]]; do
    sleep 5
    state=$(airflow dags list-runs "$dag" -o json 2>/dev/null | python3 -c '
import json, sys
runs = json.load(sys.stdin)
print(next((r["state"] for r in runs if r["run_id"] == sys.argv[1]), "queued"))' "$run_id")
    printf '.'
done
echo
echo "$dag $run_id: $state   (UI: http://localhost:8080/dags/$dag/runs/$run_id)"
[[ $state == success ]]
