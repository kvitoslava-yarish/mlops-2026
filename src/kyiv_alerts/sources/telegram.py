"""Public Telegram channels through the web preview (t.me/s/<channel>), no login.

Ported from lab_1/air_alerts_telegram/fetch_telegram_preview.py: same parsing, plus the
channel name and a stable post id "<channel>/<id>". Walks back 20 posts per page and is
resumable, so the future ingest DAG can reuse it.

    uv run python -m kyiv_alerts.sources.telegram --channel smolii_ukraine \\
        --since 2026-07-23 --until 2026-10-24 --out data/corpus/smolii_ukraine.csv
"""

import argparse
import csv
import html
import re
import time
from pathlib import Path

import requests

COLUMNS = ["post_id", "channel", "id", "datetime_utc", "text", "views"]
UA = "Mozilla/5.0 (compatible; kyiv-alerts-course-project)"
TEXT_RE = re.compile(r'<div class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>', re.S)
TIME_RE = re.compile(r'<time[^>]*datetime="([^"]+)"')
VIEWS_RE = re.compile(r'tgme_widget_message_views">([^<]+)<')


def _clean(fragment: str) -> str:
    fragment = re.sub(r"<br\s*/?>", "\n", fragment)
    fragment = re.sub(r"<[^>]+>", "", fragment)
    return html.unescape(fragment).strip()


def parse_page(page: str, channel: str) -> list[dict]:
    post_re = re.compile(rf'data-post="{re.escape(channel)}/(\d+)"')
    chunks = post_re.split(page)  # [pre, id1, body1, id2, body2, ...]
    posts = []
    for i in range(1, len(chunks) - 1, 2):
        pid, body = int(chunks[i]), chunks[i + 1]
        t, txt, v = TIME_RE.search(body), TEXT_RE.search(body), VIEWS_RE.search(body)
        posts.append({
            "post_id": f"{channel}/{pid}", "channel": channel, "id": pid,
            "datetime_utc": t.group(1) if t else "", "text": _clean(txt.group(1)) if txt else "",
            "views": v.group(1) if v else "",
        })  # fmt: skip
    return posts


def fetch_page(channel: str, before: int | None, retries: int = 6) -> str:
    url = f"https://t.me/s/{channel}" + (f"?before={before}" if before else "")
    for attempt in range(retries):
        resp = requests.get(url, headers={"User-Agent": UA}, timeout=60)
        if resp.status_code == 200:
            return resp.text
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"{url}: HTTP {resp.status_code} after {retries} tries")


def scrape(channel: str, since: str, until: str, out: Path, sleep_s: float = 0.8) -> int:
    """Posts with since <= date < until (UTC dates), newest first, appended to ``out``.
    Resumes from the oldest id already in ``out``."""
    seen, before = set(), None
    if out.exists():
        with open(out, newline="", encoding="utf-8") as f:
            ids = [int(r["id"]) for r in csv.DictReader(f)]
        seen, before = set(ids), (min(ids) if ids else None)
    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            w.writeheader()
        while True:
            posts = [p for p in parse_page(fetch_page(channel, before), channel)
                     if p["id"] not in seen]  # fmt: skip
            if not posts:
                break
            for p in sorted(posts, key=lambda p: -p["id"]):
                seen.add(p["id"])
                if since <= p["datetime_utc"][:10] < until:
                    w.writerow(p)
            f.flush()
            before = min(p["id"] for p in posts)
            dates = [p["datetime_utc"] for p in posts if p["datetime_utc"]]
            if dates and min(dates)[:10] < since:
                break
            time.sleep(sleep_s)
    with open(out, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Scrape a public Telegram channel preview")
    ap.add_argument("--channel", required=True)
    ap.add_argument("--since", required=True, help="first UTC date kept, YYYY-MM-DD")
    ap.add_argument("--until", required=True, help="first UTC date not kept, YYYY-MM-DD")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    print(f"{a.out}: {scrape(a.channel, a.since, a.until, a.out)} posts")
