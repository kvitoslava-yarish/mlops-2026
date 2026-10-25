from datetime import datetime

import pandas as pd

from kyiv_alerts.batch import batch_for
from kyiv_alerts.features import (
    KYIV_TZ,
    AlertHistory,
    build_m1,
    build_m2,
    issue_time,
    m1_row,
)


def ts(local: str) -> int:
    """Kyiv local time -> epoch seconds."""
    return int(datetime.fromisoformat(local).replace(tzinfo=KYIV_TZ).timestamp())


def episodes(*rows):
    df = pd.DataFrame(rows, columns=["oblast", "started_at", "finished_at"])
    for c in ("started_at", "finished_at"):
        df[c] = pd.to_datetime(df[c]).dt.tz_localize(KYIV_TZ).dt.tz_convert("UTC")
    df["duration_min"] = (df.finished_at - df.started_at).dt.total_seconds() / 60
    df["end_is_guess"] = False
    return df


def test_issue_time_four_windows_and_dst():
    assert issue_time(ts("2025-06-10 04:00")) == ts("2025-06-10 04:00")
    assert issue_time(ts("2025-06-10 09:00")) == ts("2025-06-10 04:00")
    assert issue_time(ts("2025-06-10 10:00")) == ts("2025-06-10 10:00")
    assert issue_time(ts("2025-06-10 15:00")) == ts("2025-06-10 10:00")
    assert issue_time(ts("2025-06-10 16:00")) == ts("2025-06-10 16:00")
    assert issue_time(ts("2025-06-10 23:00")) == ts("2025-06-10 22:00")
    assert issue_time(ts("2025-06-11 03:00")) == ts("2025-06-10 22:00")
    # DST starts 2025-03-30 03:00: the 22:00 window is one hour shorter, 04:00 still issues.
    assert issue_time(ts("2025-03-30 02:00")) == ts("2025-03-29 22:00")
    assert issue_time(ts("2025-03-30 04:00")) == ts("2025-03-30 04:00")


def test_leads_run_0_to_5():
    ep = episodes(("Kyiv City", "2025-06-10 20:00", "2025-06-10 20:30"))
    m1 = build_m1(ep, batch_for("2025-06"))
    assert sorted(m1["lead"].unique()) == [0, 1, 2, 3, 4, 5]
    assert sorted(m1["issue_slot"].unique()) == [4, 10, 16, 22]


def test_features_ignore_alerts_after_issue_time_but_label_sees_them():
    ep = episodes(
        ("Kyiv City", "2025-06-10 20:00", "2025-06-10 22:30"),  # running at 22:00 issue
        ("Kyiv City", "2025-06-10 23:10", "2025-06-10 23:40"),  # after issue
        ("Chernihivska oblast", "2025-06-10 21:30", "2025-06-10 23:00"),
        ("Kharkivska oblast", "2025-06-10 21:00", "2025-06-10 23:00"),  # not a neighbour
    )
    h = AlertHistory(ep)
    row = m1_row(h, ts("2025-06-10 23:00"))
    assert row["lead"] == 1 and row["issue_slot"] == 22
    assert row["kyiv_under_alert_at_issue"] == 1
    assert row["kyiv_alert_min_prev_12h"] == 120  # 20:00-22:00, clipped at the issue time
    assert row["neighbours_under_alert_at_issue"] == 1  # Chernihiv counts, Kharkiv does not
    assert row["neighbour_starts_prev_12h"] == 1
    assert row["days_since_long_alert"] == 60  # the 150-min alert had not ended by 22:00

    m1 = build_m1(ep, batch_for("2025-06"))
    hour = m1[m1.hour_start == pd.Timestamp(ts("2025-06-10 23:00"), unit="s", tz="UTC")]
    assert hour.label_start.item() == 1
    assert len(m1) == 30 * 24


def test_m2_features_use_previous_alert_only():
    ep = episodes(
        ("Kyiv City", "2025-06-10 01:00", "2025-06-10 01:40"),
        ("Kyivska oblast", "2025-06-10 02:50", "2025-06-10 03:30"),
        ("Kyiv City", "2025-06-10 03:00", "2025-06-10 04:00"),
    )
    m2 = build_m2(ep, batch_for("2025-06"))
    second = m2.iloc[1]
    assert second.prev_kyiv_duration_min == 40
    assert second.min_since_prev_kyiv_end == 80
    assert second.min_since_kyivska_start == 10
    assert second.duration_min == 60 and bool(second.label_valid)


def test_oblasts_started_prev_5m_counts_distinct_oblasts_including_same_second():
    ep = episodes(
        ("Kharkivska oblast", "2025-06-10 02:54", "2025-06-10 03:20"),  # 6 min before: no
        ("Sumska oblast", "2025-06-10 02:56", "2025-06-10 03:20"),
        ("Poltavska oblast", "2025-06-10 02:58", "2025-06-10 03:20"),
        ("Poltavska oblast", "2025-06-10 02:59", "2025-06-10 03:20"),  # same oblast again
        ("Kyivska oblast", "2025-06-10 03:00", "2025-06-10 03:20"),  # same second as Kyiv
        ("Kyiv City", "2025-06-10 03:00", "2025-06-10 03:15"),
        ("Odeska oblast", "2025-06-10 03:01", "2025-06-10 03:20"),  # after Kyiv: no
    )
    row = build_m2(ep, batch_for("2025-06")).iloc[0]
    assert row.oblasts_started_prev_5m == 3  # Sumska, Poltavska, Kyivska
