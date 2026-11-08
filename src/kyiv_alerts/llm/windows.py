import argparse
import re
from datetime import timedelta

import numpy as np
import pandas as pd
from pydantic import BaseModel

from kyiv_alerts.config import REPO_ROOT, get_settings
from kyiv_alerts.etl import warehouse
from kyiv_alerts.llm import gemini
from kyiv_alerts.llm.index import load_config, load_corpus
from kyiv_alerts.llm.rag import format_posts, render, window_posts

SEED = 7
QUESTIONS = [
    "Що зараз загрожує столиці?",
    "Чи є небезпека для Києва та області прямо зараз?",
    "Які загрози зараз рухаються в бік столичного регіону?",
    "Чи варто мешканцям Києва чекати на тривогу найближчим часом?",
    "Що відомо про загрози для Київщини на цей момент?",
    "Чи є зараз щось, що може долетіти до столиці?",
    "Яка ситуація з повітряними загрозами для Києва зараз?",
    "Чи загрожує щось столиці з повітря в цю хвилину?",
    "Які цілі можуть бути спрямовані на Київ просто зараз?",
    "Чи спокійно зараз у небі над столичним регіоном?",
    "Що з повітряної обстановки зараз важливо для киян?",
    "Чи є загальнонаціональні загрози, які зачіпають і столицю?",
]
KYIV_RE = re.compile(
    r"київ|києв|столиц|бровар|борисп|біл\w*\s*церкв|вишгород|обух|фастів|\bбуч|ірпін|васильк"
    r"|гостомел|вишнев|святопетрівськ|коцюбинськ|чайк|троєщин|оболон|печерськ|осокорк|позняк"
    r"|русанівк|березняк|дарниц|деснянськ|святошин|голосі|солом.?ян|поділ|подільськ|шулявк|нивк"
    r"|борщагівк|теремк|видубич|виноградар|лівий берег|правий берег|лівобереж|правобереж",
    re.I,
)
NATIONWIDE_RE = re.compile(
    r"міг-?31|mig-?31|ту-?95|ту-?160|ту-?22|стратегічн\w* авіац|масован|по всій території", re.I
)
BALLISTIC_RE = re.compile(r"балістик|циркон", re.I)
OTHER_REGION_RE = re.compile(r"\bна\s+(?!київ|києв|столиц)[А-ЯІЇЄҐ]", re.U)
THREAT_RE = re.compile(
    r"шахед|бпла|дрон|ракет|балістик|циркон|калібр|кр\b|падає|падают|удар|вибух|збит|знижен|"
    r"мопед|🛵|🏍|🚀|летить|летять|рух|курс",
    re.I,
)
# Channels that only cover Kyiv: their threat updates matter for Kyiv even without a place name.
KYIV_CHANNELS = frozenset({"smolii_ukraine"})
ALL_CLEAR_RE = re.compile(r"відбій|без бпла|чисто", re.I)


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def quotes_post(question: str, texts, n: int = 4) -> bool:
    """True if the question shares any n consecutive words with one of the posts."""
    q = _words(question)
    grams = {tuple(q[i : i + n]) for i in range(len(q) - n + 1)}
    for text in texts:
        w = _words(text)
        if grams & {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}:
            return True
    return False


def rule_relevant(text: str, channel: str = "") -> int:
    if ALL_CLEAR_RE.search(text):
        return 0
    if KYIV_RE.search(text) or NATIONWIDE_RE.search(text):
        return 1
    if BALLISTIC_RE.search(text) and not OTHER_REGION_RE.search(text):
        return 1
    return int(channel in KYIV_CHANNELS and bool(THREAT_RE.search(text)))


def rule_threat(text: str) -> str:
    t = text.lower()
    if "балістик" in t:
        return "ballistic"
    if re.search(r"міг-?31|ту-?95|ту-?160|ту-?22|авіац", t):
        return "aviation"
    if re.search(r"крилат|калібр|х-101|х-59|ракет", t):
        return "cruise_missiles"
    if re.search(r"шахед|бпла|дрон|мопед|🛵|🏍|гербер", t):
        return "drones"
    return "none"


def _kyiv_alert_starts(since: str, until: str) -> pd.Series:
    conn = warehouse.connect(get_settings().warehouse_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT started_at FROM dwh.alerts WHERE oblast = 'Kyiv City'"
                " AND started_at >= %s AND started_at < %s ORDER BY started_at",
                (since, until),
            )
            return pd.to_datetime(pd.Series([r[0] for r in cur.fetchall()]), utc=True)
    finally:
        conn.close()


def select(cfg: dict, n_before: int = 15, n_nation: int = 10, n_quiet: int = 15) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    posts = load_corpus(cfg)
    hours = cfg["retrieval"]["window_hours"]
    lo = pd.Timestamp(cfg["corpus"]["since"], tz="UTC") + timedelta(hours=hours + 1)
    hi = pd.Timestamp(cfg["corpus"]["until"], tz="UTC")
    starts = _kyiv_alert_starts(str(lo), str(hi))
    candidates = {
        "before_kyiv_alert": starts - timedelta(minutes=5),
        "after_nationwide_post": posts.loc[
            posts["text"].str.contains(NATIONWIDE_RE) & (posts["channel"] == "kpszsu"), "posted_at"
        ]
        + timedelta(minutes=10),
    }
    quiet = pd.date_range(lo, hi - timedelta(hours=1), freq="37min", tz="UTC")
    near = np.zeros(len(quiet), dtype=bool)
    for s in starts:
        near |= (quiet > s - timedelta(hours=hours)) & (quiet < s + timedelta(hours=2))
    candidates["quiet"] = pd.Series(quiet[~near])
    chosen, rows = [], []
    for stratum, n in (("before_kyiv_alert", n_before), ("after_nationwide_post", n_nation),
                       ("quiet", n_quiet)):  # fmt: skip
        pool = candidates[stratum].sample(frac=1.0, random_state=int(rng.integers(1e9)))
        for t in pool:
            t = t.floor("min")
            if not (lo <= t < hi) or any(abs(t - c) < timedelta(hours=2 * hours) for c in chosen):
                continue
            if window_posts(posts, t, hours).empty:
                continue
            chosen.append(t)
            rows.append({"t_utc": t.isoformat(), "stratum": stratum})
            if sum(r["stratum"] == stratum for r in rows) == n:
                break
    df = pd.DataFrame(rows).sort_values("t_utc").reset_index(drop=True)
    df.insert(0, "window_id", [f"w{i:02d}" for i in range(len(df))])
    df["question"] = [QUESTIONS[i % len(QUESTIONS)] for i in rng.permutation(len(df))]
    return df


class _Label(BaseModel):
    post_id: str
    relevant: int
    threat_type: str


class _Labels(BaseModel):
    labels: list[_Label]


def prelabel(cfg: dict, windows: pd.DataFrame, label_model: str) -> pd.DataFrame:
    posts = load_corpus(cfg)
    hours = cfg["retrieval"]["window_hours"]
    cli = gemini.client()
    llm_cfg = {**cfg["llm"], "max_output_tokens": 32768, "thinking_budget": None}
    out = []
    for w in windows.itertuples(index=False):
        wp = window_posts(posts, pd.Timestamp(w.t_utc), hours)
        prompt = render(cfg["prompts"]["label"], posts=format_posts(wp.to_dict("records")))
        call = gemini.generate(cli, llm_cfg, prompt, _Labels, model=label_model)
        llm = {lab.post_id: lab for lab in (call.parsed.labels if call.parsed else [])}
        for p in wp.itertuples(index=False):
            lab = llm.get(p.post_id)
            rr, rt = rule_relevant(p.text, p.channel), rule_threat(p.text)
            lr = lab.relevant if lab else None
            out.append({
                "window_id": w.window_id, "post_id": p.post_id, "channel": p.channel,
                "posted_at": p.posted_at.isoformat(), "text": " ".join(p.text.split()),
                "rule_relevant": rr, "rule_threat": rt,
                "llm_relevant": lr, "llm_threat": lab.threat_type if lab else "",
                "relevant": lr if lr is not None else rr,
                "threat_type": (lab.threat_type if lab else rt),
                "needs_review": int(lr is None or lr != rr), "reviewed": 0,
            })  # fmt: skip
        print(f"{w.window_id} {w.t_utc} {len(wp):4d} posts  "
              f"{'error: ' + call.error if call.error else ''}")  # fmt: skip
    return pd.DataFrame(out)


def rerule(labels: pd.DataFrame) -> pd.DataFrame:
    """Recompute the keyword-rule columns and the review flag; LLM labels and reviews stay."""
    labels = labels.copy()
    pairs = zip(labels["text"], labels["channel"], strict=True)
    labels["rule_relevant"] = [rule_relevant(t, c) for t, c in pairs]
    labels["rule_threat"] = labels["text"].map(rule_threat)
    unreviewed = labels["reviewed"] == 0
    labels.loc[unreviewed, "needs_review"] = (
        labels["llm_relevant"].isna() | (labels["llm_relevant"] != labels["rule_relevant"])
    )[unreviewed].astype(int)
    return labels


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the Lab 3 evaluation set")
    ap.add_argument("step", choices=["select", "prelabel", "rerule"])
    ap.add_argument("--label-model", help="default: llm.label_model in configs/rag.yaml")
    a = ap.parse_args()
    cfg = load_config()
    wpath, lpath = REPO_ROOT / cfg["eval"]["windows"], REPO_ROOT / cfg["eval"]["labels"]
    if a.step == "select":
        w = select(cfg)
        w.to_csv(wpath, index=False)
        print(w.groupby("stratum").size().to_dict(), "->", wpath)
    elif a.step == "rerule":
        labels = rerule(pd.read_csv(lpath))
        labels.to_csv(lpath, index=False)
        print(f"{int(labels['needs_review'].sum())} posts to review -> {lpath}")
    else:
        labels = prelabel(cfg, pd.read_csv(wpath), a.label_model or cfg["llm"]["label_model"])
        labels.to_csv(lpath, index=False)
        print(f"{len(labels)} posts, {int(labels['relevant'].sum())} relevant, "
              f"{int(labels['needs_review'].sum())} to review -> {lpath}")  # fmt: skip
