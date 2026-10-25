"""Feast objects for the issue-time state of Kyiv (the M1 inputs known at 10:00 / 22:00)."""

from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64
from feast.value_type import ValueType

SOURCE_PATH = Path(__file__).parent / "data" / "kyiv_state.parquet"

city = Entity(name="city", join_keys=["city"], value_type=ValueType.STRING)

kyiv_state_source = FileSource(
    name="kyiv_state_source",
    path=str(SOURCE_PATH),
    timestamp_field="event_timestamp",
)

# One row per forecast issue time; a row is valid for the 12 hours it covers.
kyiv_state = FeatureView(
    name="kyiv_state",
    entities=[city],
    ttl=timedelta(hours=12),
    schema=[
        Field(name="kyiv_under_alert_at_issue", dtype=Int64),
        Field(name="oblasts_under_alert_at_issue", dtype=Int64),
        Field(name="kyiv_alert_min_prev_12h", dtype=Float64),
        Field(name="days_since_long_alert", dtype=Float64),
        Field(name="oblast_starts_prev_12h", dtype=Int64),
    ],
    source=kyiv_state_source,
    online=False,
)
