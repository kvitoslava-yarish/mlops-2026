import numpy as np
import pandas as pd
import pytest

from kyiv_alerts.metrics import f1_tolerant, pinball


def test_flag_at_midnight_catches_start_at_2358():
    hours = pd.Series(pd.date_range("2025-06-10 22:00", periods=4, freq="h", tz="UTC"))
    y = np.array([0, 1, 0, 0])  # start at 23:58 -> hour 23
    flagged = np.array([False, False, True, False])  # flagged 00:00
    assert f1_tolerant(hours, y, flagged)["f1_pm1h"] == 1.0


def test_pinball_examples_from_readme():
    assert pinball(np.array([90.0]), np.array([240.0]), 0.9) == pytest.approx(15.0)
    assert pinball(np.array([300.0]), np.array([240.0]), 0.9) == pytest.approx(54.0)
