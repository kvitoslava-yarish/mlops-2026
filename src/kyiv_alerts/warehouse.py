"""load: PostgreSQL warehouse access. Every write replaces exactly one batch."""

import json
from datetime import datetime
from importlib import resources

import pandas as pd
import psycopg2
from psycopg2.extras import Json, execute_values

AUDIT_FIELDS = (
    "source", "raw_key", "raw_sha256", "rows_raw", "rows_duplicate", "rows_unfinished",
    "rows_bad_time", "rows_valid", "rows_merged", "rows_clean", "invalid_share",
    "m1_rows", "m2_rows", "gate_passed", "gate_report", "dag_run_id",
)  # fmt: skip


def connect(dsn: str):
    return psycopg2.connect(dsn)


def ensure_schema(conn) -> None:
    sql = resources.files("kyiv_alerts").joinpath("sql/warehouse.sql").read_text()
    with conn, conn.cursor() as cur:
        cur.execute(sql)


def replace_batch(cur, table: str, batch_month: str, df: pd.DataFrame) -> int:
    """Delete-then-insert: the statement pair that makes a rerun of a batch harmless."""
    cur.execute(f"DELETE FROM {table} WHERE batch_month = %s", (batch_month,))
    if df.empty:
        return 0
    columns = ", ".join(df.columns)
    rows = [tuple(r.values()) for r in df.to_dict("records")]
    execute_values(cur, f"INSERT INTO {table} ({columns}) VALUES %s", rows, page_size=1000)
    return len(rows)


def upsert_audit(cur, batch_month: str, fields: dict) -> None:
    unknown = set(fields) - set(AUDIT_FIELDS)
    if unknown:
        raise ValueError(f"unknown audit fields: {sorted(unknown)}")
    values = {k: Json(v) if k == "gate_report" else v for k, v in fields.items()}
    columns = ["batch_month", *values]
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in values) + ", updated_at = now()"
    cur.execute(
        f"INSERT INTO dwh.batch_audit ({', '.join(columns)}) "
        f"VALUES ({', '.join(['%s'] * len(columns))}) "
        f"ON CONFLICT (batch_month) DO UPDATE SET {updates}",
        (batch_month, *values.values()),
    )


def read_episodes(conn, start: datetime, end: datetime) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT oblast, started_at, finished_at, duration_min, end_is_guess "
            "FROM dwh.alerts WHERE started_at >= %s AND started_at < %s "
            "ORDER BY oblast, started_at",
            (start, end),
        )
        rows = cur.fetchall()
    df = pd.DataFrame(
        rows, columns=["oblast", "started_at", "finished_at", "duration_min", "end_is_guess"]
    )
    for col in ("started_at", "finished_at"):
        df[col] = pd.to_datetime(df[col], utc=True)
    return df


def scalar(conn, sql: str, params: tuple = ()):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()[0]


def row_counts(conn, batch_month: str) -> dict:
    """Rows per table for one batch, plus the min/max start that prove the interval."""
    counts = {}
    for table in ("dwh.alerts", "features.m1_hourly", "features.m2_alert"):
        sql = f"SELECT count(*) FROM {table} WHERE batch_month = %s"
        counts[table] = scalar(conn, sql, (batch_month,))
    for agg in ("min", "max"):
        sql = f"SELECT {agg}(started_at) FROM dwh.alerts WHERE batch_month = %s"
        counts[f"alerts_{agg}_started_at"] = str(scalar(conn, sql, (batch_month,)))
    return counts


if __name__ == "__main__":
    import sys

    from kyiv_alerts.config import get_settings

    with connect(get_settings().warehouse_dsn) as c:
        print(json.dumps({"batch_month": sys.argv[1], **row_counts(c, sys.argv[1])}, indent=2))
