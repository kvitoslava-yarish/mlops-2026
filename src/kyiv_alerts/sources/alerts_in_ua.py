"""alerts.in.ua API: finished alerts of the last 30 days, per oblast.

Used only for months that no pinned file covers. The JSON is stored in the lake
as fetched, and reruns of the same batch read that copy, never the API again.
Limits: history endpoints allow 2 requests/min, hence the pause between oblasts.
"""

import time
from datetime import datetime

import pandas as pd
import requests

from kyiv_alerts.batch import Batch
from kyiv_alerts.sources.dataset import KYIV, RAW_COLUMNS

API_URL = "https://api.alerts.in.ua/v1"

# alerts.in.ua oblast UID -> oblast name as written in the dataset
OBLAST_UIDS = {
    3: "Khmelnytska oblast", 4: "Vinnytska oblast", 5: "Rivnenska oblast",
    8: "Volynska oblast", 9: "Dnipropetrovska oblast", 10: "Zhytomyrska oblast",
    11: "Zakarpatska oblast", 12: "Zaporizka oblast", 13: "Ivano-Frankivska oblast",
    14: "Kyivska oblast", 15: "Kirovohradska oblast", 16: "Luhanska oblast",
    17: "Mykolaivska oblast", 18: "Odeska oblast", 19: "Poltavska oblast",
    20: "Sumska oblast", 21: "Ternopilska oblast", 22: "Kharkivska oblast",
    23: "Khersonska oblast", 24: "Cherkaska oblast", 25: "Chernihivska oblast",
    26: "Chernivetska oblast", 27: "Lvivska oblast", 28: "Donetska oblast",
    31: KYIV,
}  # fmt: skip


def fetch_month_ago(token: str, pause_s: float = 31.0, timeout_s: float = 30.0) -> list[dict]:
    """One request per oblast; each alert gets the oblast UID it was requested for."""
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    alerts: list[dict] = []
    for i, uid in enumerate(sorted(OBLAST_UIDS)):
        if i:
            time.sleep(pause_s)
        url = f"{API_URL}/regions/{uid}/alerts/month_ago.json"
        resp = requests.get(url, headers=headers, timeout=timeout_s)
        if resp.status_code == 429:
            time.sleep(60)
            resp = requests.get(url, headers=headers, timeout=timeout_s)
        resp.raise_for_status()
        for alert in resp.json().get("alerts", []):
            alerts.append({**alert, "requested_oblast_uid": uid})
    return alerts


def _fmt(value: str | None) -> str:
    if not value:
        return ""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S+00:00")


def _level(alert: dict) -> str:
    kind = alert.get("location_type") or "oblast"
    if kind == "city":
        return "oblast" if alert["requested_oblast_uid"] == 31 else "hromada"
    return kind


def to_raw_rows(alerts: list[dict], batch: Batch) -> pd.DataFrame:
    rows = [
        {
            "oblast": OBLAST_UIDS[a["requested_oblast_uid"]],
            "raion": a.get("location_raion") or "",
            "hromada": a.get("location_title") if _level(a) == "hromada" else "",
            "level": _level(a),
            "started_at": _fmt(a.get("started_at")),
            "finished_at": _fmt(a.get("finished_at")),
            "source": "alerts_in_ua",
            "naive": str(bool(a.get("calculated"))),
        }
        for a in alerts
        if a.get("alert_type", "air_raid") == "air_raid"
    ]
    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    start, end = (b.strftime("%Y-%m-%d %H:%M:%S+00:00") for b in (batch.start, batch.end))
    return df[(df["started_at"] >= start) & (df["started_at"] < end)].reset_index(drop=True)


if __name__ == "__main__":
    # make api-smoke: one request (Kyiv City), no lake writes.
    import os
    from collections import Counter

    token = os.getenv("ALERTS_IN_UA_TOKEN", "")
    if not token:
        raise SystemExit("ALERTS_IN_UA_TOKEN is empty: put the token in .env")
    resp = requests.get(
        f"{API_URL}/regions/31/alerts/month_ago.json",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    alerts = resp.json().get("alerts", [])
    starts = sorted(a["started_at"] for a in alerts if a.get("started_at"))
    print(f"HTTP {resp.status_code}: {len(alerts)} alerts for Kyiv City")
    print("by type:", dict(Counter(a.get("alert_type") for a in alerts)))
    if starts:
        print(f"first start {starts[0]}, last start {starts[-1]}")
