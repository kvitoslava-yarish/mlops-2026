"""Model classes that are logged to MLflow."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor


class QuantilePair(BaseEstimator, RegressorMixin):
    """M2: one quantile GBM per quantile; ``predict`` returns columns [P50, P90], P90 >= P50."""

    def __init__(self, quantiles=(0.5, 0.9), params=None, random_state=42):
        self.quantiles = quantiles
        self.params = params
        self.random_state = random_state

    def fit(self, X, y):
        self.models_ = [
            HistGradientBoostingRegressor(
                loss="quantile",
                quantile=q,
                early_stopping=False,
                random_state=self.random_state,
                **(self.params or {}),
            ).fit(X, y)
            for q in self.quantiles
        ]
        return self

    def predict(self, X):
        out = np.column_stack([m.predict(X) for m in self.models_])
        return np.maximum.accumulate(np.clip(out, 0, None), axis=1)
