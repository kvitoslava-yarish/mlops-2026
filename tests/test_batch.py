from datetime import UTC, datetime

import pytest

from kyiv_alerts.batch import batch_for, month_range, resolve_batch


def test_month_bounds_and_hours():
    b = batch_for("2024-02")
    assert b.start == datetime(2024, 2, 1, tzinfo=UTC)
    assert b.end == datetime(2024, 3, 1, tzinfo=UTC)
    assert b.hours == 29 * 24
    assert batch_for("2025-12").end == datetime(2026, 1, 1, tzinfo=UTC)
    assert batch_for("2026-01").previous.month == "2025-12"


def test_param_wins_over_interval_and_interval_is_used_otherwise():
    interval = datetime(2025, 6, 1, tzinfo=UTC)
    assert resolve_batch("2024-03", interval).month == "2024-03"
    assert resolve_batch("", interval).month == "2025-06"
    with pytest.raises(ValueError):
        resolve_batch("", None)
    with pytest.raises(ValueError):
        batch_for("2025-6")


def test_month_range():
    assert month_range("2025-11", "2026-02") == ["2025-11", "2025-12", "2026-01", "2026-02"]
