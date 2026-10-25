"""Export the feature tables to data/features/ for training (then ``dvc add`` it).

Only batches whose quality gate passed are exported; rows are sorted so the
same warehouse state always gives byte-identical Parquet files.
"""

from pathlib import Path

import pandas as pd

from kyiv_alerts.config import REPO_ROOT, get_settings
from kyiv_alerts.etl import warehouse

OUT_DIR = REPO_ROOT / "data" / "features"
TABLES = {"m1_hourly": "hour_start", "m2_alert": "alert_start"}


def export(out_dir: Path = OUT_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = warehouse.connect(get_settings().warehouse_dsn)
    summary = {}
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT batch_month FROM dwh.batch_audit WHERE NOT gate_passed "
                "OR gate_passed IS NULL ORDER BY 1"
            )
            summary["excluded_batches"] = [r[0] for r in cur.fetchall()]
            for table, key in TABLES.items():
                cur.execute(
                    f"SELECT f.* FROM features.{table} f "
                    "JOIN dwh.batch_audit a USING (batch_month) "
                    f"WHERE a.gate_passed ORDER BY f.{key}, f.feature_version"
                )
                cols = [d.name for d in cur.description]
                df = pd.DataFrame(cur.fetchall(), columns=cols)
                for col in df.columns:
                    if col in ("hour_start", "issue_time", "alert_start", "alert_end"):
                        df[col] = pd.to_datetime(df[col], utc=True)
                df.to_parquet(out_dir / f"{table}.parquet", index=False)
                summary[table] = {
                    "rows": len(df),
                    "batches": f"{df.batch_month.min()} .. {df.batch_month.max()}",
                }
    finally:
        conn.close()
    return summary


if __name__ == "__main__":
    import json

    print(json.dumps(export(), indent=2))
