import json
from datetime import UTC, datetime

from kyiv_alerts.batch import batch_for
from kyiv_alerts.etl.clean import clean_batch
from kyiv_alerts.etl.quality import regime
from kyiv_alerts.sources.alerts_in_ua import (
    ARCHIVE_PREFIX,
    OBLAST_UIDS,
    coverage,
    read_archive,
    to_raw_rows,
)


def alert(id_, start, end, level="yellow", updated="2026-11-01T00:00:00Z", kind="raion"):
    return {"id": id_, "location_title": "Бучанський район", "location_type": kind,
            "location_raion": "Бучанський район", "started_at": start, "finished_at": end,
            "updated_at": updated, "alert_type": "air_raid", "alert_level": level}  # fmt: skip


def fetch(stamp, alerts_by_uid, uids=OBLAST_UIDS):
    """Archive files of one fetch: every requested oblast gets a file, most of them empty."""
    folder = f"{ARCHIVE_PREFIX}/month_ago_{stamp}"
    return [
        (f"{folder}/{uid:02d}.json", json.dumps({"alerts": alerts_by_uid.get(uid, [])}).encode())
        for uid in uids
    ]


def test_archive_union_keeps_latest_version_of_each_alert():
    running = alert(1, "2026-10-20T10:00:00.000Z", None, updated="2026-10-20T10:00:00Z")
    finished = alert(1, "2026-10-20T10:00:00.000Z", "2026-10-20T10:40:00.000Z",
                     updated="2026-10-20T10:40:00Z")  # fmt: skip
    red = alert(2, "2026-11-01T01:00:00Z", "2026-11-01T01:30:00Z", level="red")
    files = fetch("2026-10-20T100500Z", {14: [running]}) + fetch(
        "2026-11-04T100931Z", {14: [finished, red]}
    )
    alerts, windows = read_archive(files)
    assert len(alerts) == 2 and len(windows) == 2
    assert next(a for a in alerts if a["id"] == 1)["finished_at"] is not None
    assert {a["requested_oblast_uid"] for a in alerts} == {14}


def test_half_written_fetch_adds_alerts_but_no_coverage():
    partial = fetch("2026-11-15T003000Z", {14: [alert(3, "2026-11-10T01:00:00Z",
                    "2026-11-10T01:20:00Z")]}, uids=[3, 4, 14])  # fmt: skip
    alerts, windows = read_archive(partial)
    assert len(alerts) == 1 and windows == []


def test_coverage_detects_partial_and_complete_months():
    w1 = (datetime(2026, 10, 5, tzinfo=UTC), datetime(2026, 11, 4, tzinfo=UTC))
    october = batch_for("2026-10")
    assert coverage([w1], october) == [(w1[0], october.end)]  # starts late: partial
    w0 = (datetime(2026, 9, 20, tzinfo=UTC), datetime(2026, 10, 20, tzinfo=UTC))
    assert coverage([w0, w1], october) == [(october.start, october.end)]  # overlapping: full
    w2 = (datetime(2026, 11, 20, tzinfo=UTC), datetime(2026, 12, 20, tzinfo=UTC))
    november = batch_for("2026-11")
    assert len(coverage([w1, w2], november)) == 2  # hole between the fetches


def test_red_and_yellow_records_merge_into_one_episode_with_the_higher_level():
    alerts = [
        {**alert(1, "2026-10-10T20:00:00Z", "2026-10-10T20:40:00Z", "yellow"),
         "requested_oblast_uid": 14},
        {**alert(2, "2026-10-10T20:30:00Z", "2026-10-10T21:00:00Z", "red"),
         "requested_oblast_uid": 14},
        {**alert(3, "2026-10-11T02:00:00Z", "2026-10-11T02:20:00Z", "yellow"),
         "requested_oblast_uid": 14},
    ]  # fmt: skip
    raw = to_raw_rows(alerts, batch_for("2026-10"))
    episodes, _ = clean_batch(raw, batch_for("2026-10"))
    assert list(episodes["alert_level"]) == ["red", "yellow"]
    assert list(episodes["oblast"]) == ["Kyivska oblast", "Kyivska oblast"]


def test_history_without_levels_still_cleans(make_raw):
    raw = make_raw(("Kyiv City", "oblast", "2025-06-01 00:00:00", "2025-06-01 00:30:00"))
    episodes, _ = clean_batch(raw, batch_for("2025-06"))
    assert episodes["alert_level"].tolist() == [""]


def test_regime_boundary():
    assert regime("2026-09") == "single_level" and regime("2026-10") == "two_level"


def test_raion_alerts_keep_their_raion_name():
    def raion_alert(id_, title):
        base = alert(id_, "2026-10-10T20:00:00Z", "2026-10-10T20:40:00Z")
        return {**base, "location_title": title, "location_raion": None, "requested_oblast_uid": 14}

    raw = to_raw_rows([raion_alert(1, "Бучанський район"), raion_alert(2, "Броварський район")],
                      batch_for("2026-10"))  # fmt: skip
    assert sorted(raw["raion"]) == ["Броварський район", "Бучанський район"]
    _, counts = clean_batch(raw, batch_for("2026-10"))
    assert counts["rows_duplicate"] == 0  # same times, different raions: not duplicates
