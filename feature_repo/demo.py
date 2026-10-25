"""Point-in-time correctness with Feast (optional Lab 2 point).

The feature table has one Kyiv state row per issue time (10:00 and 22:00 Kyiv).
We ask for the state at 15:30, which falls between two rows: Feast must return
the 10:00 row (the last one known at 15:30), never the 22:00 row from the future.

    make feast-demo     # after make snapshot
"""

import sys
from pathlib import Path

import pandas as pd
from feast import FeatureStore

REPO = Path(__file__).parent
sys.path.insert(0, str(REPO))
from definitions import SOURCE_PATH, city, kyiv_state  # noqa: E402

FEATURES = [f.name for f in kyiv_state.schema]

# 1. Offline source: the issue-time columns of the DVC-versioned M1 snapshot.
m1 = pd.read_parquet(REPO.parent / "data" / "features" / "m1_hourly.parquet")
state = (
    m1.drop_duplicates("issue_time")
    .rename(columns={"issue_time": "event_timestamp"})
    .assign(city="kyiv")[["city", "event_timestamp", *FEATURES]]
    .sort_values("event_timestamp")
)
SOURCE_PATH.parent.mkdir(exist_ok=True)
state.to_parquet(SOURCE_PATH, index=False)

store = FeatureStore(repo_path=str(REPO))
store.apply([city, kyiv_state])

# 2. Two consecutive feature rows, and a request timestamp between them.
earlier, later = state.iloc[-20], state.iloc[-19]
asked = earlier.event_timestamp + (later.event_timestamp - earlier.event_timestamp) / 2
entity_df = pd.DataFrame({"city": ["kyiv"], "event_timestamp": [asked]})
got = store.get_historical_features(
    entity_df=entity_df, features=[f"kyiv_state:{f}" for f in FEATURES]
).to_df()

kyiv = "Europe/Kyiv"
print(f"feature row A at {earlier.event_timestamp.tz_convert(kyiv)}: {earlier[FEATURES].to_dict()}")
print(f"feature row B at {later.event_timestamp.tz_convert(kyiv)}: {later[FEATURES].to_dict()}")
print(f"requested at     {asked.tz_convert(kyiv)}")
print(f"Feast returned   {got[FEATURES].iloc[0].to_dict()}")
same_as_a = all(float(got[f].iloc[0]) == float(earlier[f]) for f in FEATURES)
print("point-in-time correct (returned the earlier row A):", same_as_a)
sys.exit(0 if same_as_a else 1)
