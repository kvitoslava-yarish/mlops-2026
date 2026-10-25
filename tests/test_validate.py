import pytest

from kyiv_alerts.batch import batch_for
from kyiv_alerts.etl.validate import ValidationError, validate_raw

B = batch_for("2025-06")


def test_missing_column(make_raw):
    raw = make_raw(("Kyiv City", "oblast", "2025-06-01 00:00:00", "2025-06-01 00:30:00"))
    with pytest.raises(ValidationError, match="missing columns"):
        validate_raw(raw.drop(columns=["finished_at"]), B)


def test_rows_outside_batch_and_unknown_oblast(make_raw):
    raw = make_raw(
        ("Kyiv City", "oblast", "2025-07-01 00:00:00", "2025-07-01 00:30:00"),
        ("Atlantis", "oblast", "2025-06-01 00:00:00", "2025-06-01 00:30:00"),
    )
    with pytest.raises(ValidationError, match="unknown_oblast=1, outside_batch=1"):
        validate_raw(raw, B)
