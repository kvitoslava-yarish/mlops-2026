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


def create_table(cur, model_key: str, dim: int) -> None:
    cur.execute("CREATE SCHEMA IF NOT EXISTS rag")
    cur.execute(
        f"CREATE TABLE IF NOT EXISTS {table(model_key)} (chunk_id text PRIMARY KEY,"
        " post_id text NOT NULL, channel text NOT NULL, posted_at timestamptz NOT NULL,"
        f" text text NOT NULL, embedding vector({dim}) NOT NULL)"
    )
    cur.execute(f"CREATE INDEX IF NOT EXISTS chunks_{model_key}_posted_at_idx"
                f" ON {table(model_key)} (posted_at)")  # fmt: skip


def insert_chunks(cur, model_key: str, chunks: pd.DataFrame, vectors: np.ndarray) -> None:
    rows = [
        (c.chunk_id, c.post_id, c.channel, c.posted_at.to_pydatetime(), c.text, v)
        for c, v in zip(chunks.itertuples(index=False), vectors, strict=True)
    ]
    execute_values(cur, f"INSERT INTO {table(model_key)} VALUES %s ON CONFLICT DO NOTHING",
                   rows, page_size=2000)  # fmt: skip


def update_index(conn, cfg: dict, model_key: str) -> int:
    """Embed the posts in telegram.posts that the index does not have yet (DAG ingest_telegram)."""
    model_cfg = cfg["retrieval"]["embedding_models"][model_key]
    register_vector(conn)
    with conn, conn.cursor() as cur:
        create_table(cur, model_key, model_cfg["dim"])
        cur.execute("SELECT to_regclass('telegram.posts')")
        if cur.fetchone()[0] is None:
            return 0
        cur.execute(
            "SELECT post_id, channel, posted_at, text FROM telegram.posts p WHERE text <> ''"
            f" AND NOT EXISTS (SELECT 1 FROM {table(model_key)} c WHERE c.post_id = p.post_id)"
        )
        posts = pd.DataFrame(cur.fetchall(), columns=["post_id", "channel", "posted_at", "text"])
        if posts.empty:
            return 0
        posts["posted_at"] = pd.to_datetime(posts["posted_at"], utc=True)
        chunks = chunk_posts(posts, cfg["chunking"]["chunk_chars"],
                             cfg["chunking"]["overlap_chars"])  # fmt: skip
        insert_chunks(cur, model_key, chunks, embed(chunks["text"].tolist(), model_cfg, "passage"))
    return len(posts)


def build_index(cfg: dict, model_key: str) -> dict:
    """Rebuild from the DVC corpus, then add the posts ingested since (telegram.posts)."""
    model_cfg = cfg["retrieval"]["embedding_models"][model_key]
    posts = load_corpus(cfg)
    chunks = chunk_posts(posts, cfg["chunking"]["chunk_chars"], cfg["chunking"]["overlap_chars"])
    vectors = embed(chunks["text"].tolist(), model_cfg, "passage")
    conn = warehouse.connect(get_settings().warehouse_dsn)
    try:
        register_vector(conn)
        with conn, conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table(model_key)}")
            create_table(cur, model_key, model_cfg["dim"])
            insert_chunks(cur, model_key, chunks, vectors)
            cur.execute(f"SELECT count(*), count(DISTINCT post_id) FROM {table(model_key)}")
            n_chunks, n_posts = cur.fetchone()
        n_ingested = update_index(conn, cfg, model_key)
    finally:
        conn.close()
    by_channel = posts["channel"].value_counts().to_dict()
    return {
        "table": table(model_key),
        "posts": n_posts,
        "chunks": n_chunks,
        "by_channel": by_channel,
        "ingested_posts": n_ingested,
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
          f"({', '.join(f'{k} {v}' for k, v in sorted(r['by_channel'].items()))})"
          f" + {r['ingested_posts']} ingested posts")  # fmt: skip
