#!/usr/bin/env bash
# Trigger alerts_etl for one month and wait for the result.
#   scripts/airflow_run.sh 2025-06                       # normal run
#   scripts/airflow_run.sh 2025-06 /opt/airflow/data/... # with a raw override file
# The batch comes from the batch_month param; the clock only names the run id.
set -euo pipefail
cd "$(dirname "$0")/.."
month=$1
override=${2:-}
airflow() { docker compose exec -T airflow-scheduler airflow "$@"; }

run_id="manual__${month}__$(date -u +%Y%m%dT%H%M%S)"
conf=$(printf '{"batch_month": "%s", "raw_override": "%s"}' "$month" "$override")
airflow dags unpause alerts_etl >/dev/null
airflow dags trigger alerts_etl --run-id "$run_id" --conf "$conf" >/dev/null
echo "triggered $run_id  conf=$conf"

state=queued
while [[ $state != success && $state != failed ]]; do
    sleep 5
    state=$(airflow dags list-runs alerts_etl -o json 2>/dev/null | python3 -c '
import json, sys
runs = json.load(sys.stdin)
print(next((r["state"] for r in runs if r["run_id"] == sys.argv[1]), "queued"))' "$run_id")
    printf '.'
done
echo
echo "$run_id: $state   (UI: http://localhost:8080/dags/alerts_etl/runs/$run_id)"
[[ $state == success ]]
