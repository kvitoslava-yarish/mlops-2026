"""Vadimkin air-raid sirens dataset (MIT), pinned by DVC in data/raw/.

Raw rows are kept as published (strings, UTC timestamps, every level from
oblast down to hromada); only the column set is unified across sources.
"""

from pathlib import Path

import pandas as pd

from kyiv_alerts.batch import Batch

OFFICIAL_FILE = "official_data_en.csv"
VOLUNTEER_FILE = "volunteer_data_en.csv"  # cross-checks only, not a pipeline source

RAW_COLUMNS = [
    "oblast", "raion", "hromada", "level", "started_at", "finished_at", "source", "naive"
]  # fmt: skip
KYIV = "Kyiv City"
KYIV_OBLAST = "Kyivska oblast"
OBLASTS = frozenset(
    {
        "Cherkaska oblast", "Chernihivska oblast", "Chernivetska oblast",
        "Dnipropetrovska oblast", "Donetska oblast", "Ivano-Frankivska oblast",
        "Kharkivska oblast", "Khersonska oblast", "Khmelnytska oblast",
        "Kirovohradska oblast", KYIV, KYIV_OBLAST, "Luhanska oblast", "Lvivska oblast",
        "Mykolaivska oblast", "Odeska oblast", "Poltavska oblast", "Rivnenska oblast",
        "Sumska oblast", "Ternopilska oblast", "Vinnytska oblast", "Volynska oblast",
        "Zakarpatska oblast", "Zaporizka oblast", "Zhytomyrska oblast",
    }
)  # fmt: skip


def _ts(dt) -> str:
    """Batch bound in the dataset's own text format, so bounds compare as strings."""
    return dt.strftime("%Y-%m-%d %H:%M:%S+00:00")


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _slice(df: pd.DataFrame, batch: Batch) -> pd.DataFrame | None:
    """Rows that started inside the batch, or None if the file does not cover the whole month."""
    if df["started_at"].max() < _ts(batch.end):
        return None
    mask = (df["started_at"] >= _ts(batch.start)) & (df["started_at"] < _ts(batch.end))
    return df.loc[mask].reset_index(drop=True)


def official_slice(raw_dir: Path, batch: Batch) -> pd.DataFrame | None:
    df = _read(raw_dir / OFFICIAL_FILE)
    df["naive"] = "False"
    return _slice(df[RAW_COLUMNS], batch)

