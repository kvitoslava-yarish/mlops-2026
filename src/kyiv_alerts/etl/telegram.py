import json
from collections import Counter
from datetime import UTC, datetime

from psycopg2.extras import execute_values

from kyiv_alerts.sources import telegram

RAW_PREFIX = "raw/telegram"


def last_ids(conn) -> dict[str, int]:
    with conn.cursor() as cur:
        cur.execute("SELECT channel, max(id) FROM telegram.posts GROUP BY channel")
        return dict(cur.fetchall())


def raw_key(channel: str, fetched_at: datetime) -> str:
    return f"{RAW_PREFIX}/{channel}/{fetched_at:%Y-%m-%d}/{fetched_at:%Y%m%dT%H%M%SZ}.jsonl"


def fetch_to_lake(conn, put, channels: list[str], first_since: str, sleep_s: float = 0.8,
                  fetch=telegram.fetch_new) -> list[str]:  # fmt: skip
    """New posts per channel -> one immutable JSONL file each in the lake; returns the keys."""
    known, keys = last_ids(conn), []
    for channel in channels:
        posts = fetch(channel, known.get(channel), first_since, sleep_s)
        print(f"{channel}: {len(posts)} new posts after id {known.get(channel)}")
        if posts:
            key = raw_key(channel, datetime.now(UTC))
            put(key, "\n".join(json.dumps(p, ensure_ascii=False) for p in posts).encode())
            keys.append(key)
    return keys


def parse_raw(key: str, data: bytes) -> list[tuple]:
    rows = []
    for line in data.decode("utf-8").splitlines():
        p = json.loads(line)
        if p["datetime_utc"]:
            rows.append((p["post_id"], p["channel"], p["id"], p["datetime_utc"], p["text"],
                         p["views"], key))  # fmt: skip
    return rows


def load_posts(conn, get, keys: list[str]) -> dict[str, int]:
    rows = [r for key in keys for r in parse_raw(key, get(key))]
    if not rows:
        return {}
    with conn, conn.cursor() as cur:
        inserted = execute_values(
            cur,
            "INSERT INTO telegram.posts (post_id, channel, id, posted_at, text, views, raw_key)"
            " VALUES %s ON CONFLICT (post_id) DO NOTHING RETURNING channel",
            rows, fetch=True,
        )  # fmt: skip
    return dict(Counter(r[0] for r in inserted))


def hours_since_newest(conn) -> dict[str, float]:
    with conn.cursor() as cur:
        cur.execute("SELECT channel, extract(epoch FROM now() - max(posted_at)) / 3600"
                    " FROM telegram.posts GROUP BY channel")  # fmt: skip
        return {ch: float(h) for ch, h in cur.fetchall()}
