import numpy as np
import pandas as pd

from kyiv_alerts.training.models import QuantilePair
from kyiv_alerts.training.train import monthly_folds


def test_monthly_folds_cover_holdout_month_by_month():
    folds = monthly_folds({"holdout_start": "2026-03-01", "holdout_end": "2026-08-28"})
    assert len(folds) == 6
    assert str(folds[0][0].date()) == "2026-03-01" and str(folds[0][1].date()) == "2026-04-01"
    assert str(folds[-1][0].date()) == "2026-08-01" and str(folds[-1][1].date()) == "2026-08-29"


def test_quantile_pair_respects_monotonic_constraints():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"near": rng.integers(0, 7, 400), "starts": rng.integers(0, 15, 400)})
    # noisy target that, by itself, would not be monotonic
    y = 30 + 8 * X["near"] - 2 * X["starts"] + rng.normal(0, 25, 400) + 40 * (X["near"] == 3)
    params = {"max_iter": 50, "monotonic_cst": {"near": 1, "starts": -1}}
    model = QuantilePair(params=params).fit(X, y)
    grid = pd.DataFrame({"near": range(7), "starts": [5] * 7})
    p90 = model.predict(grid)[:, 1]
    assert np.all(np.diff(p90) >= -1e-9)  # never shorter with more neighbours
    grid = pd.DataFrame({"near": [3] * 15, "starts": range(15)})
    p50 = model.predict(grid)[:, 0]
    assert np.all(np.diff(p50) <= 1e-9)  # never longer with more simultaneous starts
