"""Run the evaluation set and log one MLflow run (experiment kyiv-llm-rag).

    uv run python -m kyiv_alerts.llm.evaluate rag [--model e5_base]   # retrieval + Gemini
    uv run python -m kyiv_alerts.llm.evaluate baseline                # Gemini, all window posts
    uv run python -m kyiv_alerts.llm.evaluate rule                    # keyword rule, no LLM

Retrieval metrics (hit@k, MRR) measure the encoder + index; the answer metrics (precision,
recall, F1 of the post ids the LLM returns, threat-type accuracy) measure the whole path. Top-k caps
recall when a window has more relevant posts than k, so ``recall_ceiling`` (share of relevant
posts the LLM was shown) and ``answer_*_given_shown`` (scored against the relevant posts it
saw) separate the retriever's limit from the LLM's choices.
Every run logs the prompt-file hash, the eval-set hash, tokens (incl. thinking), cost and
p50 / p95 latency, plus the corpus DVC hash and the git commit as lineage.
"""

import argparse
import hashlib
import os
from collections import Counter

import mlflow
import pandas as pd

from kyiv_alerts.config import REPO_ROOT, get_settings
from kyiv_alerts.etl import warehouse
from kyiv_alerts.llm import gemini, metrics
from kyiv_alerts.llm.index import load_config, load_corpus
from kyiv_alerts.llm.rag import ask, prompt_hash, retrieve, window_posts
from kyiv_alerts.llm.windows import rule_relevant, rule_threat
from kyiv_alerts.training import lineage
from kyiv_alerts.training.tracking import TrackedRun

EXPERIMENT = "kyiv-llm-rag"
CORPUS_DVC = REPO_ROOT / "data" / "corpus.dvc"
K_VALUES = (1, 3, 5, 8)


def eval_set_hash(cfg: dict) -> str:
    h = hashlib.sha256()
    for key in ("windows", "labels"):
        h.update((REPO_ROOT / cfg["eval"][key]).read_bytes())
    return h.hexdigest()


def window_truth(labels: pd.DataFrame) -> dict:
    truth = {}
    for wid, g in labels.groupby("window_id"):
        rel = g[g["relevant"] == 1]
        types = Counter(t for t in rel["threat_type"] if t != "none")
        truth[wid] = {"relevant": set(rel["post_id"]),
                      "threat": types.most_common(1)[0][0] if types else "none"}  # fmt: skip
    return truth


def evaluate(cfg: dict, mode: str, model_key: str) -> tuple[dict, list[dict], list]:
    windows = pd.read_csv(REPO_ROOT / cfg["eval"]["windows"])
    truth = window_truth(pd.read_csv(REPO_ROOT / cfg["eval"]["labels"]))
    posts = load_corpus(cfg)
    hours, k = cfg["retrieval"]["window_hours"], cfg["retrieval"]["top_k"]
    cli = gemini.client() if mode in ("rag", "baseline") else None
    conn = warehouse.connect(get_settings().warehouse_dsn) if mode == "rag" else None
    tp = fp = fn = hallucinated = type_ok = 0
    tp_s = fn_s = n_rel_shown = 0
    hits = {kk: [] for kk in K_VALUES}
    rrs, calls, preds = [], [], []
    for w in windows.itertuples(index=False):
        t = pd.Timestamp(w.t_utc)
        rel, true_type = truth.get(w.window_id, {"relevant": set(), "threat": "none"}).values()
        record = {"window_id": w.window_id, "t_utc": w.t_utc, "question": w.question}
        if mode == "rule":
            wp = window_posts(posts, t, hours)
            pairs = zip(wp["text"], wp["channel"], strict=True)
            mask = [rule_relevant(x, c) == 1 for x, c in pairs]
            picked = wp[mask]
            predicted = set(picked["post_id"])
            types = Counter(rule_threat(x) for x in picked["text"]) - Counter({"none": 10**9})
            pred_type = types.most_common(1)[0][0] if types else "none"
        else:
            if mode == "rag":
                rows = retrieve(conn, cfg, model_key, t.to_pydatetime(), w.question, k)
                ranked = [r["post_id"] for r in rows]
                if rel:
                    for kk in K_VALUES:
                        hits[kk].append(metrics.hit_at_k(ranked, rel, kk))
                    rrs.append(metrics.reciprocal_rank(ranked, rel))
                record["retrieved"] = ranked
                prompt = cfg["prompts"]["answer"]
            else:
                rows = window_posts(posts, t, hours).to_dict("records")
                prompt = cfg["prompts"]["baseline"]
            call = ask(cli, cfg, prompt, w.question, t.to_pydatetime(), rows)
            calls.append(call)
            shown = {r["post_id"] for r in rows}
            answer = call.parsed
            predicted = set(answer.relevant_post_ids) if answer else set()
            hallucinated += len(predicted - shown)
            predicted &= shown
            rel_shown = rel & shown
            n_rel_shown += len(rel_shown)
            tp_s, fn_s = tp_s + len(predicted & rel_shown), fn_s + len(rel_shown - predicted)
            pred_type = answer.threat_type if answer else "none"
            record.update({
                "summary": answer.summary if answer else "", "error": call.error,
                "latency_s": call.latency_s, "tokens_in": call.tokens_in,
                "tokens_out": call.tokens_out, "tokens_thoughts": call.tokens_thoughts,
            })  # fmt: skip
        tp, fp, fn = tp + len(predicted & rel), fp + len(predicted - rel), fn + len(rel - predicted)
        type_ok += int(pred_type == true_type)
        record.update({"predicted": sorted(predicted), "relevant": sorted(rel),
                       "threat_pred": pred_type, "threat_true": true_type})  # fmt: skip
        preds.append(record)
    if conn:
        conn.close()
    result = {f"answer_{m}": v for m, v in metrics.prf(tp, fp, fn).items()}
    result |= {"threat_type_accuracy": type_ok / len(windows), "n_windows": len(windows),
               "n_relevant_posts": sum(len(v["relevant"]) for v in truth.values())}  # fmt: skip
    if mode == "rag":
        result |= {f"hit_at_{kk}": sum(v) / len(v) for kk, v in hits.items() if v}
        result |= {"mrr": sum(rrs) / len(rrs), "n_windows_with_relevant": len(rrs)}
    if calls:
        shown_prf = metrics.prf(tp_s, fp, fn_s)
        result |= {f"answer_{m}_given_shown": v for m, v in shown_prf.items()}
        result["recall_ceiling"] = n_rel_shown / max(result["n_relevant_posts"], 1)
        ok = [c for c in calls if not c.error]
        t_in = sum(c.tokens_in for c in ok)
        t_out = sum(c.tokens_out + c.tokens_thoughts for c in ok)
        result |= metrics.latency_stats([c.latency_s for c in ok])
        result |= {
            "tokens_in": t_in, "tokens_out_visible": sum(c.tokens_out for c in ok),
            "tokens_thoughts": sum(c.tokens_thoughts for c in ok), "tokens_out_billed": t_out,
            "n_errors": len(calls) - len(ok), "hallucinated_ids": hallucinated,
        }  # fmt: skip
        price = cfg["llm"]["price_usd_per_1m"]
        if price.get("input") is not None:
            cost = metrics.cost_usd(t_in, t_out, price)
            result |= {"cost_usd": cost, "cost_usd_per_call": cost / max(len(ok), 1)}
    return result, preds, calls


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the LLM component on the eval set")
    ap.add_argument("mode", choices=["rag", "baseline", "rule"])
    ap.add_argument("--model", help="embedding model key (rag mode)")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    model_key = a.model or cfg["retrieval"]["embedding_model"]
    dirty = lineage.git_dirty()
    if dirty and not a.allow_dirty:
        raise SystemExit("uncommitted changes: commit first, so git_commit identifies the code")
    tags = {"git_commit": lineage.git_commit(), "git_dirty": str(dirty).lower(),
            "data_version": lineage.dvc_md5(CORPUS_DVC), "eval_set_sha256": eval_set_hash(cfg),
            "mode": mode_tag(a.mode)}  # fmt: skip
    prompt_key = {"rag": "answer", "baseline": "baseline"}.get(a.mode)
    if prompt_key:
        tags[f"prompt_{prompt_key}_sha256"] = prompt_hash(cfg["prompts"][prompt_key])
    result, preds, calls = evaluate(cfg, a.mode, model_key)
    if calls:
        tags["llm_model_version"] = next((c.model_version for c in calls if c.model_version), "")
    params = {"mode": a.mode, "window_hours": cfg["retrieval"]["window_hours"]}
    if a.mode != "rule":
        params |= {"llm_model": cfg["llm"]["model"], "temperature": cfg["llm"]["temperature"],
                   "max_output_tokens": cfg["llm"]["max_output_tokens"],
                   "thinking_budget": cfg["llm"]["thinking_budget"]}  # fmt: skip
    if a.mode == "rag":
        m = cfg["retrieval"]["embedding_models"][model_key]
        params |= {"top_k": cfg["retrieval"]["top_k"], "embedding_model": model_key,
                   "embedding_hf_id": m["hf_id"], "chunk_chars": cfg["chunking"]["chunk_chars"],
                   "overlap_chars": cfg["chunking"]["overlap_chars"]}  # fmt: skip
    name = f"{a.mode}-{model_key}" if a.mode == "rag" else a.mode
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    with TrackedRun(EXPERIMENT, f"{name}-{tags['git_commit'][:7]}", tags) as run:
        run.log_params(params)
        run.log_metrics(result)
        run.log_dict(cfg, "config/rag.yaml")
        run.log_dict(preds, "predictions.json")
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()})


def mode_tag(mode: str) -> str:
    return {"rag": "retrieval + LLM", "baseline": "LLM, no retrieval", "rule": "keyword rule"}[mode]


if __name__ == "__main__":
    main()
