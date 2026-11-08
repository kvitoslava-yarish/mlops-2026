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


def fetch_new(channel: str, after_id: int | None, since: str, sleep_s: float = 0.8,
              fetch=fetch_page) -> list[dict]:  # fmt: skip
    """Posts newer than ``after_id`` (first run: from UTC date ``since``), oldest first."""
    found, before, first = {}, None, True
    while True:
        posts = parse_page(fetch(channel, before), channel)
        if first and not posts:
            raise RuntimeError(f"t.me/s/{channel}: no posts parsed (markup changed?)")
        first = False
        new = [p for p in posts if (after_id is None or p["id"] > after_id)
               and p["datetime_utc"][:10] >= since and p["id"] not in found]  # fmt: skip
        found.update((p["id"], p) for p in new)
        oldest = min((p["id"] for p in posts), default=None)
        dates = [p["datetime_utc"] for p in posts if p["datetime_utc"]]
        reached_known = after_id is not None and oldest is not None and oldest <= after_id
        reached_since = bool(dates) and min(dates)[:10] < since
        if not posts or reached_known or reached_since or oldest == before:
            break
        before = oldest
        time.sleep(sleep_s)
    return [found[i] for i in sorted(found)]


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
