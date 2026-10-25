#!/bin/bash
# Runs once, on the first start of an empty volume: one user and database per component.
set -euo pipefail
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    CREATE USER airflow WITH PASSWORD '${AIRFLOW_DB_PASSWORD}';
    CREATE DATABASE airflow OWNER airflow;
    CREATE USER mlflow WITH PASSWORD '${MLFLOW_DB_PASSWORD}';
    CREATE DATABASE mlflow OWNER mlflow;
    CREATE USER warehouse WITH PASSWORD '${WAREHOUSE_DB_PASSWORD}';
    CREATE DATABASE warehouse OWNER warehouse;
EOSQL
