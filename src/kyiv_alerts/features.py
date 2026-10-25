from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from kyiv_alerts.batch import Batch
from kyiv_alerts.sources.dataset import KYIV, KYIV_OBLAST

FEATURE_VERSION = "v3"  # v3: 6-h M1 windows from 4 issues/day; M2 oblasts_started_prev_5m
# (v2: "other oblasts" features count only NEIGHBOURS)
KYIV_TZ = ZoneInfo("Europe/Kyiv")
ISSUE_HOURS = (4, 10, 16, 22)
HORIZON_H = 6
LOOKBACK_DAYS = 60
LONG_ALERT_MIN = 180
NEIGHBOURS = frozenset(
    {
        KYIV_OBLAST,
        "Zhytomyrska oblast",
        "Chernihivska oblast",
        "Cherkaska oblast",
        "Poltavska oblast",
        "Vinnytska oblast",
    }
)
CAP_MIN = 7 * 24 * 60  # "minutes since" features are capped at one week
HOUR, DAY = 3600, 86400

M1_FEATURES = [
    "hour_local",
    "lead",
    "issue_slot",
    "weekday",
    "kyiv_rate_same_hour_30d",
    "kyiv_alert_min_prev_12h",
    "days_since_long_alert",
    "kyiv_under_alert_at_issue",
    "neighbours_under_alert_at_issue",
    "neighbour_starts_prev_12h",
]
M1_LABEL = "label_start"

M2_FEATURES = [
    "start_hour_local",
    "is_night",
    "weekday",
    "neighbours_under_alert_at_start",
    "neighbour_starts_prev_60m",
    "oblasts_started_prev_5m",
    "min_since_kyivska_start",
    "min_since_prev_kyiv_end",
    "prev_kyiv_duration_min",
    "kyiv_alerts_prev_7d",
]
M2_LABEL = "duration_min"


def epoch_s(s: pd.Series) -> np.ndarray:
    return ((s - pd.Timestamp(0, tz="UTC")) // timedelta(seconds=1)).to_numpy(dtype=np.int64)


def to_ts(seconds) -> pd.Series:
    return pd.Series(pd.to_datetime(np.asarray(seconds, dtype=np.int64), unit="s", utc=True))


class AlertHistory:
    """Alert episodes known to the system, indexed for as-of lookups."""

    def __init__(self, episodes: pd.DataFrame):
        ep = episodes.sort_values("started_at")
        start, end = epoch_s(ep["started_at"]), epoch_s(ep["finished_at"])
        kyiv = (ep["oblast"] == KYIV).to_numpy()
        kyivska = (ep["oblast"] == KYIV_OBLAST).to_numpy()
        near = ep["oblast"].isin(NEIGHBOURS).to_numpy()
        self.kyiv_start, self.kyiv_end = start[kyiv], end[kyiv]
        self.kyiv_guess = ep["end_is_guess"].to_numpy(dtype=bool)[kyiv]
        self.kyivska_start = start[kyivska]
        self.near_start, self.near_end = start[near], end[near]
        codes = pd.factorize(ep["oblast"])[0]
        self.near_oblast = codes[near]
        self.other_start, self.other_oblast = start[~kyiv], codes[~kyiv]
        self.kyiv_start_hours = set((self.kyiv_start // HOUR).tolist())

    # --- Kyiv City -------------------------------------------------------
    def kyiv_starts_between(self, a: int, b: int) -> int:
        return int(np.searchsorted(self.kyiv_start, b) - np.searchsorted(self.kyiv_start, a))

    def kyiv_active(self, t: int) -> bool:
        known = self.kyiv_start < t
        return bool(np.any(self.kyiv_end[known] > t))

    def kyiv_alert_minutes(self, a: int, t: int) -> float:
        known = self.kyiv_start < t
        overlap = np.minimum(self.kyiv_end[known], t) - np.maximum(self.kyiv_start[known], a)
        return float(np.clip(overlap, 0, None).sum() / 60)

    def kyiv_rate_same_hour(self, hour_start: int, days: int = 30) -> float:
        """Share of the previous ``days`` days with a Kyiv start in the same hour (24 h steps)."""
        h = hour_start // HOUR
        return sum((h - 24 * d) in self.kyiv_start_hours for d in range(1, days + 1)) / days

    def days_since_long_alert(self, t: int) -> float:
        ended = (self.kyiv_end <= t) & (self.kyiv_end - self.kyiv_start >= LONG_ALERT_MIN * 60)
        if not ended.any():
            return float(LOOKBACK_DAYS)
        return min((t - self.kyiv_end[ended].max()) / DAY, LOOKBACK_DAYS)

    def previous_kyiv(self, t: int) -> tuple[int, int] | None:
        """Start and end of the last Kyiv alert that had ended by ``t``."""
        ended = np.flatnonzero(self.kyiv_end <= t)
        if not ended.size:
            return None
        i = ended[np.argmax(self.kyiv_end[ended])]
        return int(self.kyiv_start[i]), int(self.kyiv_end[i])

    def minutes_since_kyivska_start(self, t: int) -> float:
        i = np.searchsorted(self.kyivska_start, t, side="right")
        if i == 0:
            return float(CAP_MIN)
        return min((t - self.kyivska_start[i - 1]) / 60, CAP_MIN)

    # --- neighbour oblasts -------------------------------------------------
    def neighbours_active(self, t: int) -> int:
        active = (self.near_start < t) & (self.near_end > t)
        return int(np.unique(self.near_oblast[active]).size)

    def neighbour_starts_between(self, a: int, b: int) -> int:
        return int(np.searchsorted(self.near_start, b) - np.searchsorted(self.near_start, a))

    # --- all oblasts except Kyiv City ----------------------------------------
    def oblasts_started_between(self, a: int, b: int) -> int:
        """Distinct oblasts anywhere in Ukraine whose alert started in [a, b] (b included:
        oblasts that start in the same second as Kyiv are part of the same launch)."""
        i = np.searchsorted(self.other_start, a, side="left")
        j = np.searchsorted(self.other_start, b, side="right")
        return int(np.unique(self.other_oblast[i:j]).size)


def issue_time(hour_start: int) -> int:
    local = datetime.fromtimestamp(hour_start, KYIV_TZ)
    earlier = [h for h in ISSUE_HOURS if h <= local.hour]
    if earlier:
        day, hour = local.date(), earlier[-1]
    else:
        day, hour = local.date() - timedelta(days=1), ISSUE_HOURS[-1]
    return int(datetime(day.year, day.month, day.day, hour, tzinfo=KYIV_TZ).timestamp())


def m1_row(history: AlertHistory, hour_start: int) -> dict:
    """Features for one forecast hour, as of its issue time."""
    t = issue_time(hour_start)
    local = datetime.fromtimestamp(hour_start, KYIV_TZ)
    return {
        "hour_local": local.hour,
        "lead": (hour_start - t) // HOUR,
        "issue_slot": datetime.fromtimestamp(t, KYIV_TZ).hour,
        "weekday": local.weekday(),
        "kyiv_rate_same_hour_30d": history.kyiv_rate_same_hour(hour_start),
        "kyiv_alert_min_prev_12h": history.kyiv_alert_minutes(t - 12 * HOUR, t),
        "days_since_long_alert": history.days_since_long_alert(t),
        "kyiv_under_alert_at_issue": int(history.kyiv_active(t)),
        "neighbours_under_alert_at_issue": history.neighbours_active(t),
        "neighbour_starts_prev_12h": history.neighbour_starts_between(t - 12 * HOUR, t),
    }


def m2_row(history: AlertHistory, start: int) -> dict:
    """Features for one Kyiv alert, as of its start."""
    local = datetime.fromtimestamp(start, KYIV_TZ)
    prev = history.previous_kyiv(start)
    return {
        "start_hour_local": local.hour,
        "is_night": int(local.hour >= 22 or local.hour < 6),
        "weekday": local.weekday(),
        "neighbours_under_alert_at_start": history.neighbours_active(start),
        "neighbour_starts_prev_60m": history.neighbour_starts_between(start - HOUR, start),
        "oblasts_started_prev_5m": history.oblasts_started_between(start - 300, start),
        "min_since_kyivska_start": history.minutes_since_kyivska_start(start),
        "min_since_prev_kyiv_end": min((start - prev[1]) / 60, CAP_MIN) if prev else CAP_MIN,
        "prev_kyiv_duration_min": (prev[1] - prev[0]) / 60 if prev else -1.0,
        "kyiv_alerts_prev_7d": history.kyiv_starts_between(start - 7 * DAY, start),
    }


def build_m1(episodes: pd.DataFrame, batch: Batch) -> pd.DataFrame:
    """One row per UTC hour of the batch: features as of the issue time, label = Kyiv start."""
    history = AlertHistory(episodes)
    hours = np.arange(int(batch.start.timestamp()), int(batch.end.timestamp()), HOUR)
    rows = []
    for h in hours.tolist():
        row = {"hour_start": h, "issue_time": issue_time(h), **m1_row(history, h)}
        row[M1_LABEL] = int(history.kyiv_starts_between(h, h + HOUR) > 0)
        rows.append(row)
    df = pd.DataFrame(rows)
    df["hour_start"], df["issue_time"] = to_ts(df["hour_start"]), to_ts(df["issue_time"])
    df.insert(0, "batch_month", batch.month)
    df["feature_version"] = FEATURE_VERSION
    return df


def build_m2(episodes: pd.DataFrame, batch: Batch) -> pd.DataFrame:
    """One row per Kyiv alert that started in the batch; label = duration in minutes."""
    history = AlertHistory(episodes)
    lo, hi = int(batch.start.timestamp()), int(batch.end.timestamp())
    in_batch = (history.kyiv_start >= lo) & (history.kyiv_start < hi)
    rows = []
    for s, e, guess in zip(
        history.kyiv_start[in_batch],
        history.kyiv_end[in_batch],
        history.kyiv_guess[in_batch],
        strict=True,
    ):
        row = {"alert_start": int(s), "alert_end": int(e), **m2_row(history, int(s))}
        row[M2_LABEL] = round((e - s) / 60, 2)
        row["label_valid"] = not guess
        rows.append(row)
    columns = ["alert_start", "alert_end", *M2_FEATURES, M2_LABEL, "label_valid"]
    df = pd.DataFrame(rows, columns=columns)
    df["alert_start"], df["alert_end"] = to_ts(df["alert_start"]), to_ts(df["alert_end"])
    df.insert(0, "batch_month", batch.month)
    df["feature_version"] = FEATURE_VERSION
    return df


def lookback_start(batch: Batch) -> datetime:
    return (batch.start - timedelta(days=LOOKBACK_DAYS)).astimezone(UTC)
