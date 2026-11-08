import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from pgvector.psycopg2 import register_vector
from pydantic import BaseModel

from kyiv_alerts.config import REPO_ROOT
from kyiv_alerts.llm import gemini
from kyiv_alerts.llm.index import embed, table

KYIV_TZ = ZoneInfo("Europe/Kyiv")
THREAT_TYPES = ("drones", "cruise_missiles", "ballistic", "aviation", "none")


class Answer(BaseModel):
    relevant_post_ids: list[str]
    threat_type: str
    summary: str


def prompt_hash(path: str | Path) -> str:
    return hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()


def window_posts(posts: pd.DataFrame, t: datetime, hours: float) -> pd.DataFrame:
    lo = t - timedelta(hours=hours)
    return posts[(posts["posted_at"] >= lo) & (posts["posted_at"] < t)]


def retrieve(conn, cfg: dict, model_key: str, t: datetime, question: str, k: int) -> list[dict]:
    model_cfg = cfg["retrieval"]["embedding_models"][model_key]
    q = embed([question], model_cfg, "query")[0]
    lo = t - timedelta(hours=cfg["retrieval"]["window_hours"])
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT post_id, channel, posted_at, text, embedding <=> %s AS distance "
            f"FROM {table(model_key)} WHERE posted_at >= %s AND posted_at < %s "
            f"ORDER BY distance, post_id",
            (q, lo, t),
        )
        rows = cur.fetchall()
    seen, out = set(), []
    for post_id, channel, posted_at, text, distance in rows:
        if post_id in seen:
            continue
        seen.add(post_id)
        out.append({"post_id": post_id, "channel": channel, "posted_at": posted_at,
                    "text": text, "distance": float(distance)})  # fmt: skip
        if len(out) == k:
            break
    return out


def format_posts(rows) -> str:
    lines = []
    for r in rows:
        when = pd.Timestamp(r["posted_at"]).tz_convert(KYIV_TZ).strftime("%Y-%m-%d %H:%M")
        text = " ".join(str(r["text"]).split())
        lines.append(f"[{r['post_id']}] {when} | {r['channel']} | {text}")
    return "\n".join(lines) if lines else "(no posts)"


def render(prompt_path: str, **fields) -> str:
    return (REPO_ROOT / prompt_path).read_text().format(**fields)


def ask(cli, cfg: dict, prompt_path: str, question: str, t: datetime, rows) -> gemini.Call:
    prompt = render(
        prompt_path,
        question=question,
        now=pd.Timestamp(t).tz_convert(KYIV_TZ).strftime("%Y-%m-%d %H:%M"),
        hours=cfg["retrieval"]["window_hours"],
        posts=format_posts(rows),
    )
    return gemini.generate(cli, cfg["llm"], prompt, Answer)
