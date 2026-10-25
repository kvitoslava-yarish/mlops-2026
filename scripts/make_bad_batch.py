"""Write a deliberately broken copy of one month for the quality-gate demo.

The schema stays valid (validate passes), but 30% of rows get an end five minutes
before their start. transform drops them, and quality_check fails on
invalid_share (limit 1%). Fixed seed, so the same rows break every time.

    uv run python scripts/make_bad_batch.py 2025-06
"""

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from kyiv_alerts.batch import batch_for
from kyiv_alerts.sources.dataset import official_slice

SHARE, SEED = 0.30, 7

month = sys.argv[1] if len(sys.argv) > 1 else "2025-06"
root = Path(__file__).resolve().parents[1]
df = official_slice(root / "data" / "raw", batch_for(month))
rng = np.random.default_rng(SEED)
broken = rng.random(len(df)) < SHARE
start = pd.to_datetime(df.loc[broken, "started_at"], utc=True)
df.loc[broken, "finished_at"] = (start - timedelta(minutes=5)).dt.strftime(
    "%Y-%m-%d %H:%M:%S+00:00"
)
out = root / "data" / "fixtures" / f"bad_batch_{month}.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"{out}: {len(df)} rows, {int(broken.sum())} with end before start")
