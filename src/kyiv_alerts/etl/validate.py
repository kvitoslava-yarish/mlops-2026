"""validate: is the raw batch readable? Schema, types and nulls, before any cleaning."""

import pandas as pd

from kyiv_alerts.batch import Batch
from kyiv_alerts.sources.dataset import OBLASTS, RAW_COLUMNS


class ValidationError(Exception):
    pass


def validate_raw(df: pd.DataFrame, batch: Batch, max_unfinished_share: float = 0.01) -> dict:
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValidationError(f"raw batch {batch.month}: missing columns {missing}")
    if df.empty:
        raise ValidationError(f"raw batch {batch.month}: no rows")

    started = pd.to_datetime(df["started_at"], utc=True, errors="coerce")
    finished = pd.to_datetime(df["finished_at"].replace("", None), utc=True, errors="coerce")
    report = {
        "rows": len(df),
        "started_unparseable": int(started.isna().sum()),
        "finished_unparseable": int((finished.isna() & (df["finished_at"] != "")).sum()),
        "unfinished": int((df["finished_at"] == "").sum()),
        "unknown_oblast": int((~df["oblast"].isin(OBLASTS)).sum()),
        "outside_batch": int(
            ((started < pd.Timestamp(batch.start)) | (started >= pd.Timestamp(batch.end))).sum()
        ),
    }
    problems = [
        f"{k}={v}"
        for k, v in report.items()
        if k in ("started_unparseable", "finished_unparseable", "unknown_oblast", "outside_batch")
        and v
    ]
    if report["unfinished"] > max_unfinished_share * len(df):
        problems.append(f"unfinished={report['unfinished']} (> {max_unfinished_share:.0%})")
    if problems:
        raise ValidationError(f"raw batch {batch.month}: " + ", ".join(problems))
    return report
