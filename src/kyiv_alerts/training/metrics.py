from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def m1_scores(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def f1_tolerant(hours: pd.Series, y: np.ndarray, flagged: np.ndarray) -> dict:
    """F1 with ±1 h tolerance: a start is caught if its hour or a neighbouring hour was
    flagged, and a flag is correct if a start fell in that hour or a neighbouring one,
    so "flagged 00:00, started 23:58" is a hit."""
    h = pd.to_datetime(hours, utc=True).reset_index(drop=True)
    one = timedelta(hours=1)
    starts, flags = set(h[y == 1]), set(h[flagged])

    def near(t, pool):
        return t in pool or (t - one) in pool or (t + one) in pool

    caught = sum(near(s, flags) for s in starts)
    correct = sum(near(f, starts) for f in flags)
    precision = correct / len(flags) if flags else 0.0
    recall = caught / len(starts) if starts else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1_pm1h": f1, "precision_pm1h": precision, "recall_pm1h": recall}


def best_threshold(hours: pd.Series, y: np.ndarray, p: np.ndarray) -> float:
    candidates = np.unique(np.quantile(p, np.linspace(0.5, 0.99, 50)))
    scores = [f1_tolerant(hours, y, p >= t)["f1_pm1h"] for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def pinball(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    diff = y - pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def m2_scores(y: np.ndarray, p50: np.ndarray, p90: np.ndarray) -> dict:
    return {
        "pinball_50": pinball(y, p50, 0.5),
        "pinball_90": pinball(y, p90, 0.9),
        "p50_within_15min": float(np.mean(np.abs(y - p50) <= 15)),
        "p90_kept": float(np.mean(y <= p90)),
        "p90_width_median": float(np.median(p90)),
    }
