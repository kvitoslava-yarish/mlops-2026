from datetime import UTC, datetime

import pandas as pd
import pytest

from kyiv_alerts.llm.index import chunk_posts
from kyiv_alerts.llm.metrics import cost_usd, hit_at_k, latency_stats, prf, reciprocal_rank
from kyiv_alerts.llm.rag import Answer, format_posts, prompt_hash, render, window_posts
from kyiv_alerts.llm.windows import QUESTIONS, quotes_post, rule_relevant, rule_threat


def posts(*rows):
    df = pd.DataFrame(rows, columns=["post_id", "channel", "posted_at", "text"])
    df["posted_at"] = pd.to_datetime(df["posted_at"], utc=True)
    return df


def test_window_keeps_only_posts_before_t():
    p = posts(("kpszsu/1", "kpszsu", "2026-09-01 08:59", "a"),
              ("kpszsu/2", "kpszsu", "2026-09-01 10:30", "b"),
              ("kpszsu/3", "kpszsu", "2026-09-01 12:00", "c"))  # fmt: skip
    w = window_posts(p, datetime(2026, 9, 1, 12, tzinfo=UTC), hours=3)
    assert list(w["post_id"]) == ["kpszsu/2"]  # 08:59 too old, 12:00 not yet known


def test_long_posts_are_split_short_posts_stay_whole():
    p = posts(
        ("c/1", "c", "2026-09-01 10:00", "x" * 50), ("c/2", "c", "2026-09-01 10:01", "y" * 1200)
    )
    ch = chunk_posts(p, chunk_chars=500, overlap_chars=100)
    assert (ch["post_id"] == "c/1").sum() == 1
    assert (ch["post_id"] == "c/2").sum() == 3 and ch["text"].str.len().max() <= 500


def test_retrieval_metrics():
    ranked = ["a", "b", "c", "d"]
    assert hit_at_k(ranked, {"c"}, 2) == 0.0 and hit_at_k(ranked, {"c"}, 3) == 1.0
    assert reciprocal_rank(ranked, {"c", "d"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, {"z"}) == 0.0


def test_answer_metrics_cost_latency():
    assert prf(tp=3, fp=1, fn=2) == pytest.approx({"precision": 0.75, "recall": 0.6, "f1": 2 / 3})
    assert prf(0, 0, 0)["f1"] == 0.0
    assert cost_usd(1_000_000, 500_000, {"input": 0.3, "output": 2.5}) == pytest.approx(1.55)
    stats = latency_stats([1.0] * 9 + [10.0])
    assert stats["latency_p50_s"] == 1.0 and stats["latency_p95_s"] > 5 and stats["n_calls"] == 10


def test_keyword_rule():
    assert rule_relevant("Група ударних БпЛА курсом на Бровари") == 1
    assert rule_relevant("Зліт МіГ-31К, загроза балістики по всій території") == 1
    assert rule_relevant("Шахед на Умань") == 0
    assert rule_threat("Загроза застосування балістики") == "ballistic"
    assert rule_threat("Шахед курсом на Житомир") == "drones"


def test_questions_do_not_quote_posts():
    texts = ["🛵 Київщина: група ударних БпЛА курсом на Бровари", "Загроза балістики для столиці"]
    assert not any(quotes_post(q, texts) for q in QUESTIONS)
    assert quotes_post("група ударних БпЛА курсом на Бровари", texts)


def test_prompts_render_and_hash():
    p = posts(("kpszsu/9", "kpszsu", "2026-09-01 10:00", "Шахед\\nна Київ"))
    text = render("prompts/answer_v1.md", question="Q?", now="2026-09-01 13:00", hours=3,
                  posts=format_posts(p.to_dict("records")))  # fmt: skip
    assert "[kpszsu/9] 2026-09-01 13:00 | kpszsu | Шахед\\nна Київ" in text and "Q?" in text
    assert len(prompt_hash("prompts/answer_v1.md")) == 64


def test_answer_schema_parses_llm_json():
    a = Answer.model_validate_json(
        '{"relevant_post_ids": ["kpszsu/9"], "threat_type": "drones", "summary": "Drones."}'
    )
    assert a.relevant_post_ids == ["kpszsu/9"]
