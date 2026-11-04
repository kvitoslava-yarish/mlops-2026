import json
from datetime import UTC, datetime

from kyiv_alerts.batch import batch_for
from kyiv_alerts.etl.clean import clean_batch
from kyiv_alerts.etl.quality import regime
from kyiv_alerts.sources.alerts_in_ua import coverage, read_archive, to_raw_rows


def alert(id_, start, end, level="yellow", updated="2026-11-01T00:00:00Z", kind="raion"):
    return {"id": id_, "location_title": "Бучанський район", "location_type": kind,
            "location_raion": "Бучанський район", "started_at": start, "finished_at": end,
            "updated_at": updated, "alert_type": "air_raid", "alert_level": level}  # fmt: skip


def write_fetch(root, stamp, uid, alerts):
    folder = root / f"month_ago_{stamp}"
    folder.mkdir(exist_ok=True)
    (folder / f"{uid:02d}.json").write_text(json.dumps({"alerts": alerts}))


def test_archive_union_keeps_latest_version_of_each_alert(tmp_path):
    running = alert(1, "2026-10-20T10:00:00.000Z", None, updated="2026-10-20T10:00:00Z")
    finished = alert(1, "2026-10-20T10:00:00.000Z", "2026-10-20T10:40:00.000Z",
                     updated="2026-10-20T10:40:00Z")  # fmt: skip
    write_fetch(tmp_path, "2026-10-20T100500Z", 14, [running])
    write_fetch(tmp_path, "2026-11-04T100931Z", 14, [finished, alert(2, "2026-11-01T01:00:00Z",
                "2026-11-01T01:30:00Z", level="red")])  # fmt: skip
    alerts, windows = read_archive(tmp_path)
    assert len(alerts) == 2 and len(windows) == 2
    assert next(a for a in alerts if a["id"] == 1)["finished_at"] is not None
    assert {a["requested_oblast_uid"] for a in alerts} == {14}


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
