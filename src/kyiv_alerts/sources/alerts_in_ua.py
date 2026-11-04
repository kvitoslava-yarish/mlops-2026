"""alerts.in.ua API: alerts of the last 30 days, per oblast, with their level.

``month_ago`` only reaches 30 days back, so the DAG ``fetch_alerts_in_ua`` (1st and 15th)
archives each response in the lake (raw/alerts_in_ua/month_ago_<fetch time>/<uid>.json)
and the pipeline builds months from the union of all archived fetches.
Levels: yellow = drone threat (30 s siren), red = missile / ballistic / massive attack
(1 min siren); both are air-raid alerts. Limits: 2 history requests/min.
"""

import json
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta

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


ARCHIVE_PREFIX = "raw/alerts_in_ua"
WINDOW = timedelta(days=30)  # month_ago reaches about 30 days back from the fetch


def archive_month_ago(
    token: str, put: Callable[[str, bytes], object], pause_s: float = 31.0, timeout_s: float = 30.0
) -> str:
    """Save every oblast's ``month_ago`` response, untouched, under one folder per fetch.
    The fetch time names the folder: it is when the 30-day window ends, not a batch key."""
    folder = f"{ARCHIVE_PREFIX}/month_ago_{datetime.now(UTC):%Y-%m-%dT%H%M%SZ}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    for i, uid in enumerate(sorted(OBLAST_UIDS)):
        if i:
            time.sleep(pause_s)
        url = f"{API_URL}/regions/{uid}/alerts/month_ago.json"
        resp = requests.get(url, headers=headers, timeout=timeout_s)
        if resp.status_code == 429:
            time.sleep(60)
            resp = requests.get(url, headers=headers, timeout=timeout_s)
        resp.raise_for_status()
        put(f"{folder}/{uid:02d}.json", resp.content)
        n = len(resp.json().get("alerts", []))
        print(f"[{i + 1:2d}/{len(OBLAST_UIDS)}] {uid:2d} {OBLAST_UIDS[uid]:24s} {n:5d} alerts")
    return folder


def read_archive(
    files: Iterable[tuple[str, bytes]],
) -> tuple[list[dict], list[tuple[datetime, datetime]]]:
    """All archived alerts, de-duplicated by alert id (latest ``updated_at`` wins, so an alert
    still running in one fetch gets its end from a later one), and the windows covered by
    complete fetches (all oblasts present; a fetch that failed halfway adds alerts only)."""
    by_id: dict[int, dict] = {}
    regions: dict[str, set[int]] = {}
    for key, content in files:
        folder, name = key.rsplit("/", 1)
        uid = int(name.removesuffix(".json"))
        regions.setdefault(folder, set()).add(uid)
        for a in json.loads(content)["alerts"]:
            a = {**a, "requested_oblast_uid": uid}
            old = by_id.get(a["id"])
            if old is None or (a.get("updated_at") or "") >= (old.get("updated_at") or ""):
                by_id[a["id"]] = a
    windows = []
    for folder, uids in sorted(regions.items()):
        if uids >= set(OBLAST_UIDS):
            fetched = datetime.strptime(folder.rsplit("/", 1)[-1], "month_ago_%Y-%m-%dT%H%M%SZ")
            fetched = fetched.replace(tzinfo=UTC)
            windows.append((fetched - WINDOW, fetched))
    return list(by_id.values()), windows


def coverage(windows: list[tuple[datetime, datetime]], batch: Batch) -> list[tuple]:
    """The stretches of the batch that archived fetches cover (union of their windows,
    clipped to the month). A complete month is exactly [(batch.start, batch.end)]."""
    merged: list[list[datetime]] = []
    for lo, hi in sorted(windows):
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [
        (max(lo, batch.start), min(hi, batch.end))
        for lo, hi in merged
        if lo < batch.end and hi > batch.start
    ]


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
            "alert_level": a.get("alert_level") or "",
        }
        for a in alerts
        if a.get("alert_type", "air_raid") == "air_raid"
    ]
    df = pd.DataFrame(rows, columns=[*RAW_COLUMNS, "alert_level"])
    start, end = (b.strftime("%Y-%m-%d %H:%M:%S+00:00") for b in (batch.start, batch.end))
    return df[(df["started_at"] >= start) & (df["started_at"] < end)].reset_index(drop=True)


if __name__ == "__main__":
    # make api-smoke: one request (Kyiv City), nothing written.
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
