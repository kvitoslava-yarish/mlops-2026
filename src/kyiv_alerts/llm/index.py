"""Corpus -> chunks -> embeddings -> pgvector table, rebuilt from scratch by ``make index``.

One table per embedding model, ``rag.chunks_<key>``. Questions only search the posts of the
last few hours, so the table has a btree on ``posted_at`` and no HNSW index: the time filter
leaves a few hundred rows, which are ranked exactly (HNSW would filter after the vector search
and could return fewer than k posts).
"""

import argparse
import csv
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values

from kyiv_alerts.config import REPO_ROOT, get_settings
from kyiv_alerts.etl import warehouse

CONFIG = REPO_ROOT / "configs" / "rag.yaml"


def load_config(path: Path = CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text())


def load_corpus(cfg: dict) -> pd.DataFrame:
    frames = [pd.read_csv(REPO_ROOT / f, dtype=str, keep_default_na=False)
              for f in cfg["corpus"]["files"]]  # fmt: skip
    df = pd.concat(frames, ignore_index=True)
    day = df["datetime_utc"].str[:10]
    df = df[(day >= cfg["corpus"]["since"]) & (day < cfg["corpus"]["until"])]
    df = df[df["text"].str.strip() != ""].drop_duplicates("post_id")
    df["posted_at"] = pd.to_datetime(df["datetime_utc"], utc=True)
    return df.sort_values(["posted_at", "post_id"]).reset_index(drop=True)


def chunk_posts(posts: pd.DataFrame, chunk_chars: int, overlap_chars: int) -> pd.DataFrame:
    """Short posts stay whole; longer ones are split into overlapping character windows."""
    rows, step = [], chunk_chars - overlap_chars
    for p in posts.itertuples(index=False):
        starts = (
            range(0, max(len(p.text) - overlap_chars, 1), step)
            if len(p.text) > chunk_chars
            else [0]
        )
        for n, s in enumerate(starts):
            rows.append({"chunk_id": f"{p.post_id}#{n}", "post_id": p.post_id, "channel": p.channel,
                         "posted_at": p.posted_at, "text": p.text[s:s + chunk_chars]})  # fmt: skip
    return pd.DataFrame(rows)


@lru_cache(maxsize=4)
def embedder(hf_id: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(hf_id, device="cpu")


def embed(texts: list[str], model_cfg: dict, kind: str) -> np.ndarray:
    prefix = model_cfg[f"{kind}_prefix"]
    vectors = embedder(model_cfg["hf_id"]).encode(
        [prefix + t for t in texts], batch_size=128, normalize_embeddings=True,
        show_progress_bar=kind == "passage",
    )  # fmt: skip
    return np.asarray(vectors, dtype=np.float32)


def table(model_key: str) -> str:
    return f"rag.chunks_{model_key}"


def build_index(cfg: dict, model_key: str) -> dict:
    model_cfg = cfg["retrieval"]["embedding_models"][model_key]
    posts = load_corpus(cfg)
    chunks = chunk_posts(posts, cfg["chunking"]["chunk_chars"], cfg["chunking"]["overlap_chars"])
    vectors = embed(chunks["text"].tolist(), model_cfg, "passage")
    conn = warehouse.connect(get_settings().warehouse_dsn)
    try:
        register_vector(conn)
        with conn, conn.cursor() as cur:
            cur.execute("CREATE SCHEMA IF NOT EXISTS rag")
            cur.execute(f"DROP TABLE IF EXISTS {table(model_key)}")
            cur.execute(
                f"CREATE TABLE {table(model_key)} (chunk_id text PRIMARY KEY,"
                " post_id text NOT NULL, channel text NOT NULL, posted_at timestamptz NOT NULL,"
                f" text text NOT NULL, embedding vector({model_cfg['dim']}) NOT NULL)"
            )
            rows = [
                (c.chunk_id, c.post_id, c.channel, c.posted_at.to_pydatetime(), c.text, v)
                for c, v in zip(chunks.itertuples(index=False), vectors, strict=True)
            ]
            execute_values(cur, f"INSERT INTO {table(model_key)} VALUES %s", rows, page_size=2000)
            cur.execute(f"CREATE INDEX ON {table(model_key)} (posted_at)")
            cur.execute(f"SELECT count(*), count(DISTINCT post_id) FROM {table(model_key)}")
            n_chunks, n_posts = cur.fetchone()
    finally:
        conn.close()
    by_channel = posts["channel"].value_counts().to_dict()
    return {
        "table": table(model_key),
        "posts": n_posts,
        "chunks": n_chunks,
        "by_channel": by_channel,
    }


def corpus_counts(cfg: dict) -> dict:
    return {Path(f).stem: sum(1 for _ in csv.DictReader(open(REPO_ROOT / f, encoding="utf-8")))
            for f in cfg["corpus"]["files"]}  # fmt: skip


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Rebuild the pgvector index from the corpus")
    ap.add_argument("--model", help="embedding model key (default: retrieval.embedding_model)")
    ap.add_argument("--config", default=str(CONFIG))
    a = ap.parse_args()
    cfg = load_config(Path(a.config))
    key = a.model or cfg["retrieval"]["embedding_model"]
    r = build_index(cfg, key)
    print(f"{r['table']}: {r['posts']} posts -> {r['chunks']} chunks "
          f"({', '.join(f'{k} {v}' for k, v in sorted(r['by_channel'].items()))})")  # fmt: skip
