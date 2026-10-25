import pandas as pd
import pytest

from kyiv_alerts.sources.dataset import RAW_COLUMNS


def raw_rows(*rows: tuple) -> pd.DataFrame:
    """Rows as (oblast, level, started_at, finished_at[, naive]) in the raw string format."""
    out = []
    for r in rows:
        oblast, level, start, end, *naive = r
        out.append(
            {
                "oblast": oblast,
                "raion": "" if level == "oblast" else f"{oblast} raion",
                "hromada": "",
                "level": level,
                "started_at": f"{start}+00:00",
                "finished_at": f"{end}+00:00" if end else "",
                "source": "official",
                "naive": str(bool(naive and naive[0])),
            }
        )
    return pd.DataFrame(out, columns=RAW_COLUMNS)


@pytest.fixture
def make_raw():
    return raw_rows
