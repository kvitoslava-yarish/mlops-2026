"""transform: raw records -> one row per oblast alert episode.

Rules, in order, each counted in the batch audit:
1. exact duplicate rows are dropped (38% of the official file);
2. unfinished records (no end yet) are dropped;
3. records whose end is not after their start are dropped;
4. raion / hromada records are merged into oblast episodes: overlapping or
   touching intervals of the same oblast become one episode (an oblast is
   "under alert" while any part of it is).
Guessed ends (volunteer ``naive=True``, alerts.in.ua ``calculated``) are kept for
the start time but flagged, so duration labels can exclude them.
"""

import numpy as np
import pandas as pd

from kyiv_alerts.batch import Batch

CLEAN_COLUMNS = [
    "batch_month",
    "oblast",
    "started_at",
    "finished_at",
    "duration_min",
    "n_records",
    "end_is_guess",
    "source",
]


def _merge_intervals(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values(["started_at", "finished_at"])
    starts = group["started_at"].to_numpy()
    ends = group["finished_at"].to_numpy()
    guesses = group["naive"].to_numpy()
    # A new episode starts where the start is after the running maximum of earlier ends.
    running_end = np.maximum.accumulate(ends)
    new_episode = np.ones(len(group), dtype=bool)
    new_episode[1:] = starts[1:] > running_end[:-1]
    episode = np.cumsum(new_episode)
    return (
        pd.DataFrame({"episode": episode, "started_at": starts, "finished_at": ends, "g": guesses})
        .groupby("episode")
        .agg(
            started_at=("started_at", "min"),
            finished_at=("finished_at", "max"),
            n_records=("started_at", "size"),
            end_is_guess=("g", "any"),
        )
    )


def clean_batch(raw: pd.DataFrame, batch: Batch) -> tuple[pd.DataFrame, dict]:
    counts = {"rows_raw": len(raw)}

    df = raw.drop_duplicates()
    counts["rows_duplicate"] = counts["rows_raw"] - len(df)

    unfinished = df["finished_at"] == ""
    counts["rows_unfinished"] = int(unfinished.sum())
    df = df.loc[~unfinished].copy()

    df["started_at"] = pd.to_datetime(df["started_at"], utc=True)
    df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True)
    df["naive"] = df["naive"].str.lower() == "true"
    bad_time = df["finished_at"] <= df["started_at"]
    counts["rows_bad_time"] = int(bad_time.sum())
    df = df.loc[~bad_time]
    counts["rows_valid"] = len(df)

    if df.empty:
        episodes = pd.DataFrame(columns=CLEAN_COLUMNS)
    else:
        episodes = (
            df.groupby("oblast", sort=True)[["started_at", "finished_at", "naive"]]
            .apply(_merge_intervals)
            .reset_index()
            .drop(columns="episode")
        )
        episodes["batch_month"] = batch.month
        episodes["duration_min"] = (
            (episodes["finished_at"] - episodes["started_at"]).dt.total_seconds() / 60
        ).round(2)
        episodes["source"] = ",".join(sorted(df["source"].unique()))
        episodes = episodes[CLEAN_COLUMNS].sort_values(["oblast", "started_at"])
        episodes = episodes.reset_index(drop=True)

    counts["rows_clean"] = len(episodes)
    counts["rows_merged"] = counts["rows_valid"] - counts["rows_clean"]
    # Share of de-duplicated records that cleaning had to throw away as invalid.
    deduped = counts["rows_raw"] - counts["rows_duplicate"]
    counts["invalid_share"] = round(
        (counts["rows_unfinished"] + counts["rows_bad_time"]) / max(deduped, 1), 4
    )
    return episodes, counts
