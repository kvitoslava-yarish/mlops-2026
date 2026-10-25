"""quality_check: the gate that stops a bad batch before training can read it.

Thresholds come from the 2022-03 ... 2026-07 official batches (see README):
invalid share was at most 0.02%, batch size stayed within 0.61-1.40x of the
trailing 3-month median, and the newest alert was at most 4.3 h before month end.
"""

from dataclasses import asdict, dataclass

from kyiv_alerts import warehouse
from kyiv_alerts.batch import Batch
from kyiv_alerts.features import FEATURE_VERSION, M1_FEATURES, M2_FEATURES
from kyiv_alerts.sources.dataset import KYIV

FIRST_BATCH = "2022-03"
MAX_INVALID_SHARE = 0.01
MAX_SIZE_DEVIATION = 0.5
MAX_STALENESS_H = 48


class QualityGateError(Exception):
    pass


@dataclass(frozen=True)
class Check:
    name: str
    value: float | int | str | None
    limit: str
    passed: bool


def run_gate(conn, batch: Batch, counts: dict) -> list[Check]:
    q = lambda sql, *p: warehouse.scalar(conn, sql, p)  # noqa: E731
    m = batch.month
    checks = []

    invalid = counts["invalid_share"]
    ok = invalid <= MAX_INVALID_SHARE
    checks.append(Check("invalid_share", invalid, f"<= {MAX_INVALID_SHARE}", ok))

    history = (
        q(
            "SELECT array_agg(rows_clean ORDER BY batch_month DESC) FROM ("
            " SELECT batch_month, rows_clean FROM dwh.batch_audit"
            " WHERE batch_month < %s AND gate_passed"
            " ORDER BY batch_month DESC LIMIT 3) t",
            m,
        )
        or []
    )
    rows_clean = counts["rows_clean"]
    if len(history) == 3:
        median = sorted(history)[1]
        ratio = round(rows_clean / median, 3)
        ok = abs(ratio - 1) <= MAX_SIZE_DEVIATION
        checks.append(Check("rows_vs_3m_median", ratio, f"1 ± {MAX_SIZE_DEVIATION}", ok))
    else:
        checks.append(Check("rows_vs_3m_median", None, "skipped: < 3 earlier batches", True))

    dupes = q(
        "SELECT count(*) - count(DISTINCT (oblast, started_at)) FROM dwh.alerts"
        " WHERE batch_month=%s",
        m,
    )
    checks.append(Check("duplicate_episodes", dupes, "== 0", dupes == 0))

    newest = q("SELECT max(started_at) FROM dwh.alerts WHERE batch_month=%s", m)
    stale_h = None if newest is None else round((batch.end - newest).total_seconds() / 3600, 1)
    checks.append(
        Check("hours_since_newest_alert", stale_h, f"<= {MAX_STALENESS_H}",
              stale_h is not None and stale_h <= MAX_STALENESS_H)
    )  # fmt: skip

    kyiv = q("SELECT count(*) FROM dwh.alerts WHERE batch_month=%s AND oblast=%s", m, KYIV)
    checks.append(Check("kyiv_alerts", kyiv, ">= 1", kyiv >= 1))

    if m != FIRST_BATCH:
        prev = q("SELECT count(*) FROM dwh.alerts WHERE batch_month=%s", batch.previous.month)
        checks.append(Check("previous_batch_rows", prev, "> 0 (feature lookback)", prev > 0))

    m1 = q("SELECT count(*) FROM features.m1_hourly WHERE batch_month=%s", m)
    checks.append(Check("m1_rows", m1, f"== {batch.hours} (hours in month)", m1 == batch.hours))
    m2 = q("SELECT count(*) FROM features.m2_alert WHERE batch_month=%s", m)
    checks.append(Check("m2_rows", m2, f"== {kyiv} (Kyiv alerts)", m2 == kyiv))

    null_sql = " + ".join(f"count(*) - count({c})" for c in M1_FEATURES)
    m1_nulls = q(f"SELECT {null_sql} FROM features.m1_hourly WHERE batch_month=%s", m)
    null_sql = " + ".join(f"count(*) - count({c})" for c in M2_FEATURES)
    m2_nulls = q(f"SELECT coalesce({null_sql}, 0) FROM features.m2_alert WHERE batch_month=%s", m)
    checks.append(Check("feature_nulls", m1_nulls + m2_nulls, "== 0", m1_nulls + m2_nulls == 0))

    versions = q(
        "SELECT count(*) FROM features.m1_hourly WHERE batch_month=%s AND feature_version<>%s",
        m, FEATURE_VERSION,
    )  # fmt: skip
    limit = f"all rows == {FEATURE_VERSION}"
    checks.append(Check("feature_version", versions, limit, versions == 0))
    return checks


def report(checks: list[Check]) -> list[dict]:
    return [asdict(c) for c in checks]


def failures(checks: list[Check]) -> str:
    return "; ".join(f"{c.name}={c.value} (limit {c.limit})" for c in checks if not c.passed)
