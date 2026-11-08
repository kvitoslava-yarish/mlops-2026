import numpy as np


def hit_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return float(any(p in relevant for p in ranked[:k]))


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for i, p in enumerate(ranked, start=1):
        if p in relevant:
            return 1.0 / i
    return 0.0


def prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def latency_stats(latencies: list[float]) -> dict:
    a = np.asarray(latencies, dtype=float)
    return {
        "latency_p50_s": float(np.percentile(a, 50)),
        "latency_p95_s": float(np.percentile(a, 95)),
        "latency_mean_s": float(a.mean()),
        "n_calls": int(a.size),
    }


def cost_usd(tokens_in: int, tokens_out: int, price: dict) -> float:
    """Output price applies to visible and thinking tokens alike (both are billed)."""
    return tokens_in / 1e6 * price["input"] + tokens_out / 1e6 * price["output"]
