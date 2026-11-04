from itertools import product

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor


class QuantilePair(BaseEstimator, RegressorMixin):

    def __init__(self, quantiles=(0.5, 0.9), params=None, random_state=42):
        self.quantiles = quantiles
        self.params = params
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y):
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
        self.monotonic_ = dict((self.params or {}).get("monotonic_cst") or {})
        self.grid_ = {
            f: np.arange(int(X[f].min()), int(X[f].max()) + 1) for f in self.monotonic_
        }
        return self

    def _raw(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([m.predict(X) for m in self.models_])

    def _monotone(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy()
        feats = list(self.monotonic_)
        for f in feats:  # trees treat values beyond the training range like its edge
            X[f] = X[f].clip(self.grid_[f][0], self.grid_[f][-1]).round()
        x = X[feats].to_numpy()
        sign = np.array([self.monotonic_[f] for f in feats])
        upper = np.full((len(X), len(self.quantiles)), -np.inf)
        lower = np.full((len(X), len(self.quantiles)), np.inf)
        for point in product(*(self.grid_[f] for f in feats)):
            g = np.array(point)
            Xg = X.copy()
            Xg[feats] = np.broadcast_to(g, (len(X), len(feats)))
            pred = self._raw(Xg)
            below = np.all(sign * (g - x) <= 0, axis=1)  # must predict <= the row
            above = np.all(sign * (g - x) >= 0, axis=1)  # must predict >= the row
            upper = np.where(below[:, None], np.maximum(upper, pred), upper)
            lower = np.where(above[:, None], np.minimum(lower, pred), lower)
        return (upper + lower) / 2

    def predict(self, X):
        X = pd.DataFrame(X)
        out = self._monotone(X) if self.monotonic_ else self._raw(X)
        return np.maximum.accumulate(np.clip(out, 0, None), axis=1)
