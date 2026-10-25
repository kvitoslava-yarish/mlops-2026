"""The six ETL steps for one monthly batch. The Airflow DAG is a thin wrapper.

Data moves through storage, not XCom: raw and clean files in the MinIO lake,
episodes and features in PostgreSQL. Each step returns a small dict for the next.
"""

import io
import json

import pandas as pd

from kyiv_alerts.batch import Batch
from kyiv_alerts.config import Settings
from kyiv_alerts.etl import quality, warehouse
from kyiv_alerts.etl.clean import clean_batch
from kyiv_alerts.etl.validate import validate_raw
from kyiv_alerts.features import build_m1, build_m2, lookback_start
from kyiv_alerts.sources import alerts_in_ua, dataset
from kyiv_alerts.storage import Lake, sha256

SOURCES = ("official", "alerts_in_ua")


def raw_key(batch: Batch, source: str) -> str:
    return f"raw/alerts/{batch.month}/{source}.csv"


def clean_key(batch: Batch) -> str:
    return f"clean/alerts/{batch.month}.parquet"


def _csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def _read_raw(lake: Lake, key: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(lake.get(key)), dtype=str, keep_default_na=False)


def extract(
    settings: Settings, batch: Batch, raw_override: str = "", refresh: bool = False
) -> dict:
    """Write the batch's raw rows to the lake once; reruns reuse that immutable copy.

    Source order: the pinned official file (must cover the whole month), then the
    alerts.in.ua API if a token is set. The volunteer file is kept for cross-checks
    only: it is oblast-level and splits alerts differently (README, "Sources").
    """
    lake = Lake(settings)
    if raw_override:
        df = pd.read_csv(raw_override, dtype=str, keep_default_na=False)
        key = raw_key(batch, "override")
        data = _csv(df)
        return {"source": "override", "raw_key": key, "raw_sha256": lake.put(key, data),
                "rows": len(df), "reused": False}  # fmt: skip

    if not refresh:
        for source in SOURCES:
            key = raw_key(batch, source)
            if lake.exists(key):
                data = lake.get(key)
                rows = len(pd.read_csv(io.BytesIO(data), dtype=str))
                return {"source": source, "raw_key": key, "raw_sha256": sha256(data),
                        "rows": rows, "reused": True}  # fmt: skip

    df, source = dataset.official_slice(settings.raw_data_dir, batch), "official"
    if df is None and settings.alerts_in_ua_token:
        alerts = alerts_in_ua.fetch_month_ago(settings.alerts_in_ua_token)
        lake.put(f"raw/alerts/{batch.month}/alerts_in_ua.json", json.dumps(alerts).encode())
        df, source = alerts_in_ua.to_raw_rows(alerts, batch), "alerts_in_ua"
    if df is None:
        raise ValueError(
            f"no source covers {batch.month}: the pinned official file ends 2026-09-07 and "
            "ALERTS_IN_UA_TOKEN is not set"
        )
    key = raw_key(batch, source)
    data = _csv(df)
    return {"source": source, "raw_key": key, "raw_sha256": lake.put(key, data),
            "rows": len(df), "reused": False}  # fmt: skip


def validate(settings: Settings, batch: Batch, info: dict) -> dict:
    return validate_raw(_read_raw(Lake(settings), info["raw_key"]), batch)


def transform(settings: Settings, batch: Batch, info: dict) -> dict:
    lake = Lake(settings)
    episodes, counts = clean_batch(_read_raw(lake, info["raw_key"]), batch)
    buf = io.BytesIO()
    episodes.to_parquet(buf, index=False)
    lake.put(clean_key(batch), buf.getvalue())
    return counts


def load(settings: Settings, batch: Batch, info: dict, counts: dict, run_id: str = "") -> int:
    episodes = pd.read_parquet(io.BytesIO(Lake(settings).get(clean_key(batch))))
    conn = warehouse.connect(settings.warehouse_dsn)
    try:
        warehouse.ensure_schema(conn)
        with conn, conn.cursor() as cur:  # one transaction: delete + insert + audit
            n = warehouse.replace_batch(cur, "dwh.alerts", batch.month, episodes)
            warehouse.upsert_audit(
                cur,
                batch.month,
                {
                    "source": info["source"],
                    "raw_key": info["raw_key"],
                    "raw_sha256": info["raw_sha256"],
                    **{k: v for k, v in counts.items() if k in warehouse.AUDIT_FIELDS},
                    "gate_passed": None,
                    "gate_report": None,
                    "dag_run_id": run_id,
                },
            )
    finally:
        conn.close()
    return n


def build_features(settings: Settings, batch: Batch) -> dict:
    conn = warehouse.connect(settings.warehouse_dsn)
    try:
        episodes = warehouse.read_episodes(conn, lookback_start(batch), batch.end)
        m1, m2 = build_m1(episodes, batch), build_m2(episodes, batch)
        with conn, conn.cursor() as cur:
            n1 = warehouse.replace_batch(cur, "features.m1_hourly", batch.month, m1)
            n2 = warehouse.replace_batch(cur, "features.m2_alert", batch.month, m2)
            warehouse.upsert_audit(cur, batch.month, {"m1_rows": n1, "m2_rows": n2})
    finally:
        conn.close()
    return {"m1_rows": n1, "m2_rows": n2}


def quality_check(settings: Settings, batch: Batch, counts: dict) -> list[dict]:
    conn = warehouse.connect(settings.warehouse_dsn)
    try:
        checks = quality.run_gate(conn, batch, counts)
        for c in checks:
            print(f"{'PASS' if c.passed else 'FAIL'} {c.name}: {c.value} (limit {c.limit})")
        passed = all(c.passed for c in checks)
        with conn, conn.cursor() as cur:
            warehouse.upsert_audit(
                cur, batch.month, {"gate_passed": passed, "gate_report": quality.report(checks)}
            )
    finally:
        conn.close()
    if not passed:
        raise quality.QualityGateError(f"batch {batch.month} failed: {quality.failures(checks)}")
    return quality.report(checks)
