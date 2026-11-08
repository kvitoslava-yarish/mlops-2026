# Kyiv air-alert forecasting — design document

KSE MLOps 2026 · Lab 3 (revised 5.1–5.5, 6.1, 6.3–6.5, 6.7–6.9, 7.1, 7.2, 7.4) · Template: [ml-design-docs](https://github.com/eugeneyan/ml-design-docs). Section numbers are fixed; later labs revise them. **A** = assumption.

## 1 Overview

Kyiv alerts are known only when the siren starts, with no estimate of when the next one comes or how long it lasts.
The system forecasts four times a day (04:00, 10:00, 16:00, 22:00) the probability that a Kyiv alert starts in each of the next 6 hours, with the likely threat type, and predicts the duration (P50, P90) of every Kyiv alert when it starts.
It learns from the public alert history of all Ukraine since 2022 and the Ukrainian Air Force Telegram channel, retrains monthly and monitors drift daily.
Predictions are served by a REST API for planning only; the official alert always takes precedence.

## 2 Motivation

**What the system predicts**

| | Question it answers | When it runs | Output | Ground truth (label) and when it is known |
|---|---|---|---|---|
| **M1** start risk | "In which of the next 6 hours is a Kyiv alert likely to **start**?" | 04:00, 10:00, 16:00, 22:00 — each covers the next 6 h | 6 probabilities, one per hour, + risk band 🟢 / 🟡 / 🔴 | 1 if a Kyiv alert started in that hour, else 0 — known at the end of the hour |
| **M3** threat type | "If an alert comes in that hour, what is it likely to be?" | together with M1 | per hour: P(drones), P(cruise missiles), P(ballistic) | type of the alert, read from Air Force Telegram posts |
| **M2** duration | "This alert just started — when will it end?" | at the start of every Kyiv alert | **P50** = typically ends by (half of such alerts end earlier) · **P90** = almost surely ends by (9 of 10 end earlier) | actual duration — known when the alert ends |

Example: at 22:00 the forecast says 00–01 h has an 11.9% chance that an alert starts (🔴, above the 6.4% average hour), and if one starts it is most likely drones (0.84). An alert then starts at 23:41 after an Air Force drone post; M2 returns P50 = 95 min and P90 = 240 min, i.e. "typically over by 01:16, almost surely by 03:41".

**Why a system, not a notebook**

- **Non-stationary target.** Nights with ≥ 60 min of alert in Kyiv: 21% (2023) → 41% (2025) → 29% (2026). Median alert duration: 55 → 30 min. Nov 2025: alerts switched from oblast to raion level. A one-off model is stale within months.
- **Consumers:** API clients (later a planning bot; the Lab 5 agent). M1: 4 forecasts/day. M2: ~1.4 predictions/day (511 Kyiv alerts in 2025).
- **Cost of errors:** a missed start → disrupted plan; frequent false alarms → forecast ignored; P90 too short → user resumes activity too early (worst case, so M2 is judged on whether P90 holds, not only on average error).

## 3 Success metrics

Lab 1 analysis below; the Lab 2 pipeline results (hold-out 2026-03-01 … 2026-08-28) are in 7.2. Hold-out 2026-03-01 … 2026-09-05. Baseline = the simplest rule a model must beat. **Main** = metric we optimise; **guardrail** = must only stay within its limit. Formulas in [README](../README.md#how-predictions-are-scored).

**M1 — is the hourly start probability good?** Scored on 4,536 hold-out hours; an alert started in 7.5% of them. Baseline = average start rate for that hour of day.

| Level | Metric | What it checks | Target | Baseline | First model |
|---|---|---|---|---|---|
| Model (main) | Brier = mean (p − y)² over hours | probabilities close to what happened (y = 1 start, 0 none); lower is better | < baseline | 0.0695 | 0.0687 |
| Model (guardrail) | AUC | start hours get higher p than quiet hours; 0.5 = random | ≥ 0.60 | 0.57 | 0.58 |
| Business | F1 with ±1 h tolerance¹ | if we flag "high risk" hours, how many real starts we catch and how many flags are right | ≥ 0.30 **A** | 0.25 | 0.27 |
| System | publish delay | forecast available after 04:00 / 10:00 / 16:00 / 22:00 | ≤ 5 min | — | — |

¹ Hours with p ≥ threshold are flagged (threshold tuned on 2025-09 … 2026-02, then frozen). A start is *caught* if its hour or a neighbouring hour was flagged; a flag is *correct* if a start fell in that hour or a neighbouring one, so "flagged 00:00, started 23:58" is a hit. F1 = 2·P·R / (P + R).

**Horizon change after Lab 2 (12 h → 6 h).** The numbers above score a 12-hour forecast. The signal is concentrated in the first hours, so M1 now covers 6 hours and is issued every 6 hours, and every M1 metric is also reported **per lead hour** (0–5); the AUC guardrail applies to the 6-hour average. Evidence in 5.1.

**M2 — are the duration estimates accurate and reliable?** Scored on 349 hold-out alerts. Baseline = always the last-12-months median and P90 (33 / 224 min).

| Level | Metric | What it checks | Target | Baseline |
|---|---|---|---|---|
| Model (main) | pinball loss, q = 0.5 / 0.9 | error of P50 and P90 in minutes; running past P90 costs 9× more than ending early; lower is better | < baseline | 17.3 / 20.9 |
| Business | P50 within ±15 min | share of alerts where "typically ends by" was ≤ 15 min off | ≥ 55% **A** | 47% |
| Business (guardrail) | P90 kept | share of alerts that ended before "almost surely ends by" | 85–95% | 97% |
| Business (guardrail) | P90 width | median P90 in minutes; shorter = more useful | < 224 min | 224 min |
| System | alert start → prediction (p95) | M2 arrives in time to be useful | ≤ 60 s | — |
| System | API latency (p95) | | ≤ 200 ms | — |

Why pinball is the main metric: "P90 by hour of day" is shorter (133 min) but kept less often (88%) and scores the same 20.9 as the baseline, so neither "always safe" nor "always short" wins. Results are also split by threat type and night/day (hold-out median 29 min, 29% < 20 min, 5% > 3 h).

**M3 — is the threat type right?** Scored on alerts with a known type. Macro-F1 = F1 per type, averaged, so rare ballistic alerts count as much as drones (target: "always the most common type" + 0.10 **A**); log-loss of the probabilities (target: below "last month's type mix"). Measured in Lab 3 (needs Telegram labels).

## 4 Requirements and constraints

### 4.1 Functional

| # | Requirement |
|---|---|
| F1 | Inputs: finished alert intervals (all oblasts), active alerts (poll 20 s), Air Force Telegram posts (scrape 2 min) |
| F2 | M1 + M3 at 04:00, 10:00, 16:00 and 22:00: 6 × {P(start), risk band, P(type \| alert)} + window summary |
| F3 | M2 at each Kyiv alert start: P50, P90, threat type used |
| F4 | Read API: `GET /v1/forecast/latest`, `/v1/alert/current`, `/v1/health`. Internal: `POST /v1/predict/duration` (poller only) |
| F5 | Every response: `model_version`, `issued_at`, `degraded`. Fallback to baseline with `degraded: true`, never an error |
| F6 | Predictions joined with labels when known; stored for monitoring and retraining |

### 4.2 Non-functional

| Category | Requirement |
|---|---|
| Freshness | M1 ≤ 5 min after issue; alert state ≤ 30 s; Telegram ≤ 5 min; labels daily (monitoring), monthly (training) |
| Latency, throughput | API p95 ≤ 200 ms; M2 ≤ 60 s p95; ≤ 10 req/s **A**; live feed (Ubilling proxy) 1 request / 20 s (limit 2 req/s); alerts.in.ua history ≤ 8 req/min (hard limit 12) |
| Availability, fallback | 99%/month on Minikube **A**; baseline fallback; live feed down or stale > 2 min → backup source `ual`, else M2 `degraded` |
| Reproducibility | model ↔ DVC data version + git commit + params in MLflow; same inputs → same metrics (fixed seeds); `uv.lock` |
| Adaptability | monthly retrain on expanding window + on alarm; new model enters as `challenger` (5.4) |
| Observability without labels | labels lag ≤ 6 h (M1), ≤ ~11 h (M2, max alert 673 min); until then: input drift, prediction drift, freshness (6.6) |
| Cost, resources | laptop, Minikube 4 GB, no GPU; cloud variant ≤ $200/month **A** (6.7) |

### 4.3 Scope

- **In:** Kyiv City; M1, M2, M3 (M3 conditional: ≥ 80% of Kyiv alerts must match a typed Air Force post, **A**, checked in Lab 3); REST API.
- **Out:** safety advice, push notifications, red/yellow alert level as target (no history; the live feed carries `alert_level`, collected by the poller from Lab 4), strike locations or air-defence data, weather (7.1).

## 5 Methodology

### 5.1 Problem statement

| | Task | Unit | Label |
|---|---|---|---|
| M1 | binary classification, probabilistic | (issue time ∈ {04, 10, 16, 22}, lead 0–5 h) → one model for all four | a Kyiv alert **started** in the hour |
| M2 | quantile regression, q = 0.5, 0.9 | one Kyiv alert, at start | duration, minutes |
| M3 | multiclass, conditional on alert | same rows as M1 | drones / cruise missiles / ballistic, from Air Force posts (guided bombs don't reach Kyiv) |

**Hour label.** 62% of Kyiv alerts (2025–26, n = 959) cross an hour boundary.

| Alert | ≥ 10 min in hour | any minute in hour | **starts in hour** |
|---|---|---|---|
| 23:58–00:40 | 23h = 0, 00h = 1 | 23h = 1, 00h = 1 | 23h = 1 |
| 12:55–13:08 | both 0 (alert lost) | 12h = 1, 13h = 1 | 12h = 1 |
| hours = 1 | 11.8% | 14.0% | 6.4% |
| issue | 4% of alerts lost | tails and long alerts over-counted | each alert exactly once |

The 23:58 vs 00:00 boundary case is handled in evaluation by the ±1 h tolerance. Duration is M2's job.

**Horizon: 6 hours, issued every 6 hours.** The features describe the situation at issue time (who is under alert now, what started in the last hours), so they say most about the next 1–2 hours. Test: same features (neighbour ring, Kyiv state), same GBM settings as `train.yaml`, train 2023-01 … 2026-02, hold-out 2026-03 … 2026-08-28.

| Hours ahead | 0 | 1 | 2 | 4 | 6 | 9 | 11 |
|---|---|---|---|---|---|---|---|
| Brier improvement over hour-of-day | 4.4% | 1.2% | 1.0% | 0.7% | 0.4% | 0.5% | 0.2% |
| AUC | 0.69 | 0.62 | 0.61 | 0.59 | 0.57 | 0.58 | 0.57 |

| Horizon, issued every … | 12 h | **6 h** | 3 h | 1 h |
|---|---|---|---|---|
| Brier improvement over hour-of-day | 0.9% | **1.3%** | 2.3% | 4.4% |
| AUC (hour-of-day baseline 0.571) | 0.591 | **0.610** | 0.641 | 0.688 |

6 h keeps a window long enough to plan around and clears the 0.60 AUC guardrail; shorter horizons gain more but give the user less time ahead. Issue times 04:00 / 10:00 / 16:00 / 22:00 keep the familiar 10:00 and 22:00, and 22:00 still covers the main night hours (22–04).

**Rejected:** one binary "long alert tonight" label (arbitrary 60-min threshold, no timing, separate day model); duration buckets for M2 (lose the tail).

**Output, 22:00 issue (covers 22–04).** P(start) = current baseline (2025–26 start rate by hour); type values illustrative until M3 is trained. Risk: 🟢 < 5%, 🟡 5–9%, 🔴 ≥ 9% (mean 6.4%).

| Hour | 22 | 23 | 00 | 01 | 02 | 03 |
|---|---|---|---|---|---|---|
| P(start), % | 9.3 | 10.1 | 11.9 | 10.8 | 6.4 | 8.2 |
| Risk | 🔴 | 🔴 | 🔴 | 🔴 | 🟡 | 🟡 |
| P(drones \| alert) | .80 | .82 | .84 | .82 | .75 | .60 |

Baseline window summary (2025–26): P(at least one start) = 0.46 for 22–04, 0.24 for 04–10, 0.26 for 10–16, 0.27 for 16–22.

```json
{
  "issued_at": "2026-10-24T22:00:00+03:00",
  "hours": [{"hour_start": "2026-10-24T22:00:00+03:00", "p_start": 0.093, "risk": "high",
             "type_if_alert": {"drones": 0.80, "cruise_missiles": 0.12, "ballistic": 0.08}}],
  "summary": {"p_any_start": 0.46, "expected_alerts": 0.57},
  "model_version": {"m1": "m1@champion:7", "m3": "m3@champion:2"},
  "degraded": false
}
```

```json
{
  "alert_started_at": "2026-10-24T23:41:10+03:00",
  "threat_type": "drones",
  "threat_source": {"post_id": 90211, "posted_at": "2026-10-24T23:36:02+03:00"},
  "duration_min": {"p50": 95, "p90": 240},
  "model_version": "m2@champion:3",
  "degraded": false
}
```

**Auxiliary LLM job (Lab 3): "what currently threatens Kyiv?"** At any time *t* a client (the Lab 5 agent, later the API) asks a free-text question. The job takes the posts of the last 3 hours from two public Telegram channels, retrieves the 8 closest to the question and asks an LLM which of them matter for Kyiv now. Output (JSON schema, enforced by the API):

```json
{"relevant_post_ids": ["kpszsu/71317", "kpszsu/71318"],
 "threat_type": "drones",
 "summary": "A drone is heading toward the Kyiv reservoir from Chernihiv region, alongside a threat of ballistic weapon use from the north."}
```

`threat_type` ∈ drones / cruise_missiles / ballistic / aviation / none. A post is **relevant** if it mentions Kyiv or Kyiv oblast, a threat moving toward them, or a nationwide threat (MiG-31K or strategic-aviation takeoff, ballistic launch, massive combined attack). Why posts and not the alert map: the map says *that* Kyiv is under alert, the posts say *what* is flying and from where, often minutes before the siren. Scored per window on the post ids (precision / recall / F1 against hand-reviewed labels) and the threat type; retrieval separately with hit@k and MRR (5.4). The same relevant-post counts are planned as M1/M2 features and as M3 labels once ingestion is continuous.

### 5.2 Data

| Source | Content | Version in use | Role |
|---|---|---|---|
| [Vadimkin dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset) `official_data_en.csv` (MIT) | official alerts, UTC, oblast / raion / hromada level; 299,275 rows, 2022-03-15 … 2026-09-07 | upstream commit `58ee75b` (2026-09-08, the last update of this file) | batches 2022-03 … 2026-08 (the only pipeline source): history, labels |
| same repo `volunteer_data_en.csv` | oblast-level alerts, `naive` = guessed end; 9.1 MB, 118,544 rows, 2022-02-25 … 2026-10-24 | upstream commit `346bcff` (2026-10-24) | cross-checks only (coarser: 1.3–1.8× more episodes), not a pipeline source |
| [alerts.in.ua API](https://devs.alerts.in.ua/) `regions/{uid}/alerts/month_ago.json` | alerts of the last 30 days per oblast, raion level, with `alert_level`: **yellow** = drone threat (30 s siren), **red** = missile / ballistic / massive attack (1 min siren); 25 requests at 2/min | DAG `fetch_alerts_in_ua` on the 1st and 15th, 00:30 UTC: each fetch kept as an immutable folder in the lake (`raw/alerts_in_ua/month_ago_<fetch time>/`); first fetch 2026-11-04 (covers 2026-10-04 …, also under DVC as `data/api.dvc`) | 2026-10 onwards: the pipeline builds a month from the union of archived fetches |
| [Ubilling Aerial Alerts API](https://wiki.ubilling.net.ua/doku.php?id=aerialalertsapi), `?source=aiu&raw` | **active** alerts as published by alerts.in.ua, no token: `started_at` (exact, UTC), `location_type` (oblast / raion / hromada), `alert_type`, `alert_level` (red / yellow); no history, `finished_at` always null | live, checked 2026-10-25 | Lab 4 poller: M2 trigger, live features, own alert log (new batches and labels at raion level, like the official file) |
| same API, `?source=ual&raw` | the same from ukrainealarm.com (`createdAt`, `alertLevel`, `type: AIR`) | live | backup when `aiu` fails or is stale |
| Telegram [@kpszsu](https://t.me/s/kpszsu) | Ukrainian Air Force, official: threat posts for the whole country ("БпЛА на Київщину", MiG-31K takeoffs, ballistic threats); 20,038 posts 2026-07-24 … 2026-10-23, median 53 characters | `data/corpus/kpszsu.csv` in `data/corpus.dvc` (`5402f138…dir`) | Lab 3 retrieval corpus; later M3 labels and M1/M2 features |
| Telegram [@smolii_ukraine](https://t.me/s/smolii_ukraine) | independent Kyiv-only monitoring: live district-level updates ("Троєщина шахед", "ПАДАЄ"), all-clears, news; 20,820 posts, same period, very short | same DVC version | Lab 3 retrieval corpus; its context-free threat updates are Kyiv-relevant by construction |

**Live feed check (2026-10-25).** The proxy's `default` source (Vadym Klymenko API) returns `changed: 1970-01-01` for most regions → unusable timestamps, so the poller pins `source=aiu`. Its JSON matches the alerts.in.ua API, so switching to the direct API once the token arrives changes only the URL and auth. At the same moment the `ual` cache was ~4 min older than `aiu` → every response's `cachedat` is checked (6.6). The feed lists only active alerts: an alert's end is the first poll where it is gone (±20 s). Filters: `alert_type = air_raid`; drop permanent alerts (Luhansk since 2022-04-04, Crimea since 2022-12-10), as in the dataset. The proxy has no history, so it cannot back-fill past months: batches 2026-09 and 2026-10 wait for alerts.in.ua `month_ago`; from Lab 4 the poller's own log produces new monthly batches.

**Batch grain:** one calendar month in UTC, equal to the run's data interval (`CronDataIntervalTimetable("0 3 1 * *", timezone="UTC")`: 03:00 on the 1st, after the alerts.in.ua fetch at 00:30). A batch holds 1.7k–16k raw records → 0.8k–2.2k oblast episodes (official file) → 672–744 M1 rows (one per hour) and 6–135 M2 rows (one per Kyiv alert).

**Label delay:** M1 → end of the hour (a 6-h window is fully labelled 6 h after issue). M2 → end of the alert (median 38, P90 161, max 673 min). In the monthly pipeline every label of a batch is known when the batch runs, after the month ends.

**Data version identifier:** the DVC hash. Raw: `data/raw.dvc` = `352f88a87031f9e22af2cce021c2a4a7.dir` (both CSVs). Training input: `data/features.dvc` (the snapshot of the feature tables). Every MLflow run carries both as tags `raw_data_version` and `data_version`.

**Layers**

| Layer | Where | Kept | Written by |
|---|---|---|---|
| raw | DVC remote (MinIO bucket `dvc`) + lake `raw/alerts/YYYY-MM/<source>.csv` | rows as published, unified columns; immutable per batch, sha256 in the audit table | `dvc add`, task `extract` |
| clean | lake `clean/alerts/YYYY-MM.parquet` + PostgreSQL `dwh.alerts` | one row per oblast alert episode (typed, UTC) | `transform`, `load` |
| features | PostgreSQL `features.m1_hourly`, `features.m2_alert` (`feature_version` column) | one row per hour / per Kyiv alert, label alongside | `build_features` |
| training snapshot | `data/features/*.parquet`, `data/features.dvc` | batches whose gate passed | `make snapshot` |

**Cleaning rules and rows affected (2022-03 … 2026-08, official file):** 297,338 raw records → exact duplicates 113,845 (every row appears twice until 2026-01, never after) → end ≤ start 2 → unfinished 0 → raion / hromada records merged into oblast episodes 100,145 → 83,346 episodes. Guessed ends are kept for the start and flagged `end_is_guess`; M2 labels exclude them.

**Quality findings (measured):**

- The official file has a gap: nothing after 2026-08-29 05:00 UTC, while volunteer data shows ~170 alerts a day. The gate's freshness check stopped batch 2026-08 (newest alert 67 h before month end, limit 48 h). Decision: accept the 2.8 missing days for this month (`ACCEPTED_GAPS`, limit 72 h, reason stored in the gate report) and end the hold-out on 2026-08-28, so the missing hours are not scored as quiet.
- Volunteer data is coarser: for the same months it gives 1.3–1.8× more oblast episodes than the official file (2026-08: 3,916 vs 2,231). Batch 2026-09 built from it failed the size check (2.20× the median of the last 3 passed batches): a source switch the gate is meant to catch. Decision: the volunteer file is no longer a pipeline source; 2026-09 and 2026-10 come from alerts.in.ua.
- Raion-level records since Nov 2025 triple the raw rows (16k vs 4–5k a month) but not the oblast episodes (1.5k), because the transform merges them.
- Kyiv City: 2,245 alerts in 2022-03 … 2026-08; an alert started in 5.6% of hours.
- **Two-level alerts (regime change).** Since autumn 2026 every alert carries a level: yellow (drones) or red (missiles, ballistic, massive attack). Both are sirens, so a Kyiv alert = red **or** yellow, the same meaning as the history. But the system now declares far more alerts: Oct 5–31 2026 gives ≈ 4.5k oblast episodes per 30 days (history 1.8k–2.2k) and ≈ 206 Kyiv alerts per 30 days (history 39–90; 2026-10: 33 red, median 25 min, and 155 yellow, median 40 min). Red and yellow are separate events (2 of 213 Kyiv alerts chain into the next). Decisions: from `TWO_LEVEL_FROM = 2026-10` batches belong to the `two_level` regime; the gate compares batch size only within a regime; `alert_level` is kept on raw records and on `dwh.alerts` episodes (an episode takes its highest level), giving M3's label (drones vs missiles) and an M2 signal for free; models are retrained once ~3 two-level months exist. The official GitHub file stopping on 2026-09-07 may be the same switch (its parser may not read the new messages), not verified.
- **2026-09 has no consistent source** (official file stopped, volunteer file coarser, API history starts 2026-10-04), and 2026-10 is partial (from 2026-10-04 13:32 UTC). Both are listed in `ACCEPTED_GAPS` with their reason; 2026-10 runs without the September lookback.

**Formats:** raw CSV as published (replayable); clean and features as typed Parquet in the lake and PostgreSQL tables (point lookups, transactional delete-then-insert); training snapshot as Parquet under DVC.

**Lab checklist**

| Item | Answer |
|---|---|
| Time axis | UTC start/end since 2022-03-15; batch = calendar month (data interval) |
| Hold-out | train 2023-01 … 2026-02, hold-out 2026-03-01 … 2026-08-28 (5.4) |
| Label delay | 1–6 h (M1), ≤ 11.2 h (M2); all known when the monthly batch runs |
| Data version | DVC md5 of `data/raw.dvc` and `data/features.dvc` |
| Text surface | 40,858 Telegram posts (two channels), web preview scrape, DVC-versioned; evaluation set of 40 windows / 2,404 labelled posts in `data/eval/` (5.4) |

### 5.3 Techniques

| | Trivial baseline | Rule baseline | Candidate (Lab 2) |
|---|---|---|---|
| M1 | start rate by hour of day (from the training period) | same-hour start rate over the last 30 days (feature `kyiv_rate_same_hour_30d`) | `HistGradientBoostingClassifier` on the 10 features below |
| M2 | — | last-12-months P50 / P90 of duration (before the hold-out) | two `HistGradientBoostingRegressor(loss="quantile")`, q = 0.5 / 0.9, P90 ≥ P50 enforced, with monotonic constraints (v3) — longer with more neighbours under alert (`neighbours_under_alert_at_start` +1), shorter with more simultaneous starts (`oblasts_started_prev_5m` −1) — to keep 1.4k training alerts from fitting noise. scikit-learn re-estimates quantile leaf values after each tree, which can break `monotonic_cst`, so `QuantilePair.predict` also projects the output onto monotone functions (mean of the upper and lower monotone envelopes over the training grid) |
| M3 | most common type / last month's mix | type by hour of day | alert level as label from 2026-10 (yellow = drones, red = missiles / ballistic), Telegram detail in Lab 3 |

scikit-learn only: no LightGBM / XGBoost, which need `libomp` from Homebrew on macOS. Hold-out results: 7.2.

**Features and where each value comes from at serving time** (`src/kyiv_alerts/features.py`, one code path; `feature_version = v3`). The features about other regions count only Kyiv's **neighbour oblasts** — Kyivska, Zhytomyrska, Chernihivska, Cherkaska, Poltavska, Vinnytska — so front-line oblasts with near-permanent local alerts do not dominate them (v1 counted all 24). Every feature is computed as of time *t* — the issue time (M1) or the alert start (M2) — from alerts that started before *t*; an alert still running at *t* counts as active, with its minutes clipped at *t*.

| Feature | Model | Training source | Serving source (Lab 4) | Kind |
|---|---|---|---|---|
| `hour_local`, `lead`, `issue_slot`, `weekday` | M1 | computed from the hour and its issue time | computed from the request (forecast hour) | request |
| `kyiv_rate_same_hour_30d` | M1 | `dwh.alerts`, Kyiv starts in the same hour on each of the previous 30 days | same query on `dwh.alerts` + the poller's live alerts table | windowed aggregate (warehouse) |
| `kyiv_alert_min_prev_12h` | M1 | `dwh.alerts`, Kyiv alert minutes in [t − 12 h, t) | same query + the running alert from poller state | windowed aggregate (warehouse + poller) |
| `days_since_long_alert` | M1 | `dwh.alerts`, last Kyiv alert ≥ 180 min that had ended (cap 60) | same query | warehouse lookup |
| `kyiv_under_alert_at_issue` | M1 | `dwh.alerts` episodes active at t | poller state (live feed `source=aiu`, 20 s) | stream (poll) |
| `neighbours_under_alert_at_issue` | M1 | `dwh.alerts` neighbour-oblast episodes active at t (0–6) | poller state ← **skew risk**: different producer (6.6) | stream (poll) |
| `neighbour_starts_prev_12h` | M1 | `dwh.alerts`, neighbour-oblast episode starts in [t − 12 h, t) | poller history table ← **skew risk** | windowed aggregate (poller) |
| `start_hour_local`, `is_night`, `weekday` | M2 | computed from the alert start | computed from the poller's start event | request |
| `neighbours_under_alert_at_start` | M2 | `dwh.alerts` neighbour-oblast episodes active at the start | poller state ← **skew risk** | stream (poll) |
| `neighbour_starts_prev_60m` | M2 | `dwh.alerts`, neighbour-oblast starts in the last 60 min | poller history table | windowed aggregate (poller) |
| `min_since_kyivska_start` | M2 | `dwh.alerts`, last Kyiv oblast start (cap 1 week) | poller history table | windowed aggregate (poller) |
| `min_since_prev_kyiv_end`, `prev_kyiv_duration_min` | M2 | `dwh.alerts`, previous ended Kyiv alert | `dwh.alerts` + poller history | warehouse lookup |
| `kyiv_alerts_prev_7d` | M2 | `dwh.alerts`, Kyiv starts in the last 7 days | `dwh.alerts` + poller history | windowed aggregate (warehouse) |
| `oblasts_started_prev_5m` (**v3**) | M2 | `dwh.alerts`, oblasts in all Ukraine whose episode **started** in [t − 5 min, t] | poller history table | windowed aggregate (poller) |

Labels: `label_start` (M1, a Kyiv alert started in the hour) and `duration_min` (M2, `label_valid = false` when the end is a guess). Planned, not in v3: holiday and moon illumination (M1), Telegram threat features (M1, M2, M3; Lab 3).

**v3 changes.** M1: issue times 04 / 10 / 16 / 22 and leads 0–5 (5.1). M2: `oblasts_started_prev_5m` — many oblasts starting within the same 5 minutes is the signature of a ballistic launch, i.e. a short alert. It counts *starts*, so near-permanent front-line alerts do not leak in, and it points the opposite way from the neighbour count (Spearman with duration −0.21 vs +0.41). Kyiv alerts 2025-01 … 2026-08:

| Oblasts started within 5 min before Kyiv | 0 | 1–3 | 4–10 | > 10 |
|---|---|---|---|---|
| Kyiv alerts | 470 | 379 | 75 | 10 |
| Median / P90 duration, min | 39 / 233 | 30 / 116 | 24 / 72 | 20 / 28 |

**Point-in-time correctness.** Training rows are built by the pipeline as of *t* (issue time or alert start) from alerts that started before *t*, so no join can pull in a later row; `tests/test_features.py` checks it.

**LLM job (Lab 3).**

| | Rule baseline | LLM baseline (no retrieval) | Candidate: retrieval + LLM |
|---|---|---|---|
| Input | every post in [t − 3 h, t) | every post in [t − 3 h, t), time order | top 8 posts in [t − 3 h, t) by cosine similarity to the question |
| Method | keyword rule: Kyiv place names (city, districts, nearby towns), nationwide carriers, ballistic not aimed at another region, any threat word in the Kyiv-only channel; all-clear = not relevant | `gemini-3.8-flash`, [prompts/baseline_v1.md](../prompts/baseline_v1.md) | same model, [prompts/answer_v1.md](../prompts/answer_v1.md) |
| Output | relevant ids; type from keywords | JSON schema | JSON schema |

- **Chunking:** 500 characters with 100 overlap. Posts have a median of 39 characters, so 40,858 posts give 41,306 chunks; only long morning summaries split. Retrieval keeps the best chunk per post.
- **Encoders (the varied parameter):** `intfloat/multilingual-e5-small` (384-d, `query:` / `passage:` prefixes), `intfloat/multilingual-e5-base` (768-d), `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-d). All run on CPU in the stack; Ukrainian is in their training data.
- **Vector store:** pgvector 0.8.7 in the existing PostgreSQL 16, one table per encoder (`rag.chunks_<key>`). The time filter leaves a few hundred rows, so the query is an exact `ORDER BY embedding <=> q` after a btree range scan on `posted_at`; an HNSW index would rank first and filter after, returning fewer than k posts for a narrow window (7.1).
- **LLM:** `gemini-3.8-flash` through the Google GenAI API, temperature 0, `max_output_tokens` 1024, `thinking_budget` 0, response schema = pydantic model. The id is pinned: the provider re-routes deprecated names (2.5 Flash / Pro are re-routed on 2026-12-01). Pre-labels use the stronger `gemini-4-argon`, so the answerer is never graded against its own output.
- **Prompts** are files with a version suffix; editing one changes its sha256, which is a tag on every run. The prompts tell the model that @smolii_ukraine covers only Kyiv, so its bare updates count without a place name.

### 5.4 Experimentation and validation

- **Time split only.** Attacks come in waves of several days, and neighbouring hours share the same rolling features (30-day rate, last 12 h). A random split would put hours of the same attack in train and test and score the model on a wave it has already seen.
- **Lab 2 split** (`configs/train.yaml`): hold-out 2026-03-01 … 2026-08-28 (4,344 hours, 325 Kyiv alerts with a valid end); training data starts 2023-01-01 (2022 is left out: early-war alerting behaved differently, raw rows halve after 2022-10). The hold-out stops before the upstream gap of 2026-08-29 … 31 (5.2).
  - **M1, monthly retraining back-tested:** each hold-out month is predicted by a model fit on all data from 2023-01 up to the start of that month (6 refits); scores are pooled over the 6 months and reported per lead hour. The registered model is the latest refit, on everything up to 2026-08-28.
  - **M2, single split:** train 2023-01-01 … 2026-02-28 (1.4k alerts are too few for monthly folds to differ much); monthly refits come with the Lab 4 schedule.
- **F1 threshold:** for each monthly fold, tuned on the 6 months before that month (inside its training data), then frozen for that month.
- **As-of:** features use only alerts that started before the issue time (M1) or the alert start (M2); see 5.3.
- **Metrics:** M1 Brier (main), AUC, log-loss, F1 ±1 h — overall and per lead hour 0–5. M2 pinball q = 0.9 (main) and q = 0.5, P50 within ±15 min, P90 kept, P90 width.
- **Determinism:** fixed seed, early stopping off, `uv.lock`; the same commit and data version give the same metrics (reproduction tolerance in the README).
- **Training window: expanding, retrained monthly.** A sliding window of the last 3 months was tested (retrain every month, predict the next; 6-h horizon, test months 2026-03 … 08): it lost to the hour-of-day baseline in all 6 months.

  | Training data | Brier improvement | AUC |
  |---|---|---|
  | all history, expanding | **+1.5%** | **0.619** |
  | last 12 months | +0.7% | 0.607 |
  | all history, recency-weighted (half-life 3 months) | +0.1% | 0.595 |
  | last 3 months | −2.9% | 0.542 |

  3 months hold ~150 Kyiv starts — too few — and never contain the same season twice. The features already describe the current situation, so the model follows a change in attack intensity without forgetting older data; a real change is caught by drift monitoring (6.6), which triggers an early retrain.
- **Later labs:** the promotion rule — challenger vs champion on the last 3 monthly folds, day-level bootstrap 95% CI excluding 0, all guardrails passed, beats the rule baseline, then 1 week in shadow mode and a manual alias move.

**LLM job (Lab 3).**

- **Evaluation set** (`data/eval/`, committed; `eval_set_sha256` = sha256 of windows + labels, logged on every run): 40 times *t* drawn with seed 7 from three strata: 15 five minutes before a Kyiv alert started, 10 ten minutes after an Air Force nationwide-threat post, 15 quiet times (no Kyiv alert from 3 h before to 2 h after). Windows are ≥ 6 h apart. Each has a paraphrased Ukrainian question (12 templates); a unit test fails if a question shares 4 consecutive words with a post in its window, so retrieval cannot win by copying.
- **Labels:** all 2,404 posts in the windows. Pre-labels from `gemini-4-argon` ([prompts/label_v1.md](../prompts/label_v1.md)) and the keyword rule; every disagreement (567) plus 87 spot checks reviewed by hand. The reviewer sided with argon on 458, with the rule on 109, and overturned 5 agreed labels. 1,134 relevant; 34 windows have ≥ 1.
- **Rubric check:** a Kyiv alert is active or starts within 1 h for 71.7% of relevant posts vs 47.8% of the others, in both channels. The keyword rule was revised once after reading argon's disagreements, so its score is optimistic (7.2).
- **Metrics:** retrieval hit@1/3/5/8 and MRR over the 34 windows with a relevant post; answer precision / recall / F1 of the returned ids (micro over windows); the same restricted to the relevant posts the LLM was shown (`*_given_shown`) and the share it was shown (`recall_ceiling`), so a k-limit is not blamed on the LLM; threat-type accuracy per window; ids cited but not shown (hallucinations); tokens in / out / thinking, cost, latency p50 / p95.
- **No time split** is needed: nothing is trained. The windows only have to be outside any prompt example and fixed before the comparison; they were selected before any run and are versioned by hash.

### 5.5 Human in the loop

- Drift alarm → on-call (author) reviews the Evidently report within 24 h: ignore / retrain / fall back to baseline.
- Promotion: manual move of the MLflow alias `champion`.
- Override: `FORCE_BASELINE=true` (ConfigMap) or pin `champion` to an older version.
- LLM evaluation labels: pre-labelled by two independent sources, every disagreement reviewed by a person (5.4); a new prompt or model is compared on the same hashed set before it replaces the current one.

## 6 Implementation

### 6.1 High-level design

#### System context

```mermaid
flowchart LR
    vad["Vadimkin air-raid sirens dataset<br/>(GitHub, history 2022-03 to 2026-08)"]
    aiu["alerts.in.ua data<br/>month_ago (token) · live feed via<br/>Ubilling proxy (no token)"]
    tg["Telegram @kpszsu<br/>(Ukrainian Air Force posts)"]
    sys["Kyiv Alert Forecaster<br/>ingests, trains, serves, monitors"]
    cli["API client<br/>(planning app later, Lab 5 agent)"]
    onc["On-call engineer (author)<br/>Grafana alert to Telegram / email"]

    vad -- "raw alert history (bootstrap CSV)" --> sys
    aiu -- "raw data: active alerts every 20 s" --> sys
    aiu -- "labels: finished alerts (monthly)" --> sys
    tg -- "raw data: posts every 2 min" --> sys
    sys -- "predictions: 6-h start risk + type, duration P50/P90" --> cli
    sys -- "alerts: drift, staleness, failures" --> onc
    onc -. "promotion approval, override" .-> sys

    style sys fill:#0b3d91,color:#ffffff
```

#### Data flow (Lab 2, real component names)

The batch side as it runs now. Marks: **version points** (DVC: raw files, training snapshot), **quality gate** (`quality_check`, before training can read a batch), **training path** (snapshot → `kyiv_alerts.training.train`) vs **serving path** (feature code on live state, Lab 4).

```mermaid
flowchart TB
    subgraph SRC["Sources"]
        off["official_data_en.csv<br/>commit 58ee75b"]
        vol["volunteer_data_en.csv<br/>commit 346bcff (cross-checks only)"]
        api["alerts.in.ua month_ago<br/>(token; 2026-10 onwards)"]
    end
    dvcraw[("version point: data/raw.dvc<br/>DVC remote = MinIO bucket dvc")]
    subgraph DAG["Airflow 3.3.1 DAG alerts_etl · batch = one month = data interval"]
        ex["extract"] --> va["validate"] --> tr["transform"] --> lo["load"] --> bf["build_features"] --> qc["quality_check"]
    end
    lakeraw[("MinIO lake<br/>raw/alerts/YYYY-MM/source.csv<br/>immutable")]
    lakeclean[("MinIO lake<br/>clean/alerts/YYYY-MM.parquet")]
    subgraph PG["PostgreSQL 16 · database warehouse"]
        dwh["dwh.alerts (oblast episodes)<br/>dwh.batch_audit (counts, gate result)"]
        feat["features.m1_hourly · features.m2_alert<br/>feature_version = v3"]
    end
    snap[("version point: data/features.dvc<br/>passed batches only")]
    train["make train → kyiv_alerts.training.train<br/>rule baselines + HistGradientBoosting"]
    mlf["MLflow 3.16.1 tracking server<br/>backend: PostgreSQL db mlflow · artifacts: MinIO bucket mlflow"]
    reg["Model registry<br/>kyiv-m1-start-risk · kyiv-m2-duration<br/>tags data_version + git_commit · alias champion"]
    serve["Lab 4: forecast_6h DAG + API<br/>same features.py on live state"]

    off --> dvcraw
    vol --> dvcraw
    dvcraw -- "dvc pull; mounted read-only" --> ex
    fetch["DAG fetch_alerts_in_ua<br/>1st + 15th, 00:30 UTC"]
    lakeapi[("MinIO lake<br/>raw/alerts_in_ua/month_ago_fetch-time/")]
    api --> fetch --> lakeapi
    lakeapi -- "union of fetches" --> ex
    ex --> lakeraw
    tr --> lakeclean
    lo -- "DELETE + INSERT the month" --> dwh
    bf -- "DELETE + INSERT the month" --> feat
    qc -- "gate_passed" --> dwh
    feat -- "make snapshot" --> snap
    snap -- "training path" --> train
    train -- "params, metrics, lineage tags" --> mlf
    mlf -- "register" --> reg
    reg -- "load by alias champion" --> serve
    feat -. "serving path (Lab 4)" .-> serve

    style qc fill:#0b3d91,color:#ffffff
```

- Rerunning a month is harmless: `extract` reuses the immutable raw file, and `load` / `build_features` delete and re-insert that month inside one transaction.
- Quality gate (`src/kyiv_alerts/etl/quality.py`): invalid records ≤ 1% after de-duplication; batch size within ±50% of the median of the last 3 passed batches; no duplicate (oblast, start); newest alert ≤ 48 h before month end; Kyiv present; previous month loaded (feature lookback); M1 rows = hours in the month; M2 rows = Kyiv alerts; no feature nulls; one `feature_version`. A failure marks the batch in `dwh.batch_audit`, fails the run, and keeps the batch out of the training snapshot.
- Feature store = PostgreSQL feature tables (offline + the online lookups Lab 4 needs) with a DVC-versioned Parquet snapshot for training. No Feast (7.1).
- Lab 3 adds the Telegram corpus and the LLM job below; Lab 4 adds the poller, the forecast DAG and the API that read the same feature code.

#### Retrieval path (Lab 3, real component names)

```mermaid
flowchart LR
    subgraph SRC["Public Telegram web preview"]
        kp["t.me/s/kpszsu"]
        sm["t.me/s/smolii_ukraine"]
    end
    scr["make corpus<br/>kyiv_alerts.sources.telegram"]
    dvc[("version point: data/corpus.dvc<br/>40,858 posts · MinIO bucket dvc")]
    idx["make index MODEL=key<br/>kyiv_alerts.llm.index<br/>500-char chunks · sentence-transformers (CPU)"]
    subgraph PG["PostgreSQL 16 + pgvector 0.8.7 · database warehouse"]
        tab["rag.chunks_e5_small · rag.chunks_e5_base · rag.chunks_minilm<br/>41,306 chunks each · btree posted_at"]
    end
    cfg["configs/rag.yaml<br/>window 3 h · top_k 8 · encoder · LLM · prices"]
    q["question + time t"]
    ret["kyiv_alerts.llm.rag.retrieve<br/>posted_at ∈ [t − 3 h, t) · exact cosine · best chunk per post"]
    pr["prompts/answer_v1.md<br/>sha256 logged"]
    gem["Gemini API gemini-3.8-flash<br/>JSON schema · temperature 0"]
    out["relevant_post_ids · threat_type · summary"]

    kp --> scr
    sm --> scr
    scr --> dvc -- "dvc pull" --> idx --> tab
    q --> ret
    tab --> ret
    cfg -.-> idx
    cfg -.-> ret
    ret -- "top 8 posts" --> gem
    pr --> gem
    gem --> out

    style ret fill:#0b3d91,color:#ffffff
```

#### Evaluation loop (Lab 3)

```mermaid
flowchart TB
    alerts[("dwh.alerts<br/>Kyiv alert starts")] --> sel["make eval-windows<br/>40 windows, 3 strata, seed 7<br/>paraphrased questions + non-quoting test"]
    corpus[("data/corpus.dvc")] --> sel
    sel --> win[("data/eval/windows.csv")]
    win --> pre["make eval-prelabel<br/>gemini-4-argon (label_v1) + keyword rule"]
    pre --> rev["hand review of 567 disagreements + 87 spot checks"]
    rev --> lab[("data/eval/labels.csv<br/>eval_set_sha256")]
    win --> ev
    lab --> ev["kyiv_alerts.llm.evaluate<br/>rule · baseline · rag × 3 encoders"]
    ev -- "params · prompt / eval-set / corpus hashes · git_commit<br/>hit@k · MRR · F1 · tokens · cost · p50 / p95<br/>predictions.json" --> mlf["MLflow experiment kyiv-llm-rag"]
    mlf --> cmp["comparison table (7.2) → retrieval.embedding_model"]
```

#### Deployment, first draft

Lab 2 runs the batch side with `docker compose` on the laptop (6.2); Lab 4 moves it to Minikube (Kubernetes 1.30.0, 4 GB) with the real cluster topology. Each box shows the local tool and, in italics, the AWS service that would replace it. Arrows read "A → B: A writes to / sends to / loads from B"; numbers give the order of a typical run. The system has two sides that share MinIO, PostgreSQL and MLflow, drawn separately for readability.

**Scheduled (batch) side** — Airflow jobs ingest data, train models and publish the 04:00 / 10:00 / 16:00 / 22:00 forecast.

```mermaid
flowchart LR
    subgraph src["External sources"]
        gh["GitHub alert dataset<br/>history 2022–2026"]
        aiu["alerts.in.ua API<br/>last month of alerts"]
        tg["Telegram @kpszsu<br/>Air Force posts"]
    end
    subgraph laptop["Laptop · Minikube — scheduled (batch) side"]
        af["<b>Apache Airflow 3.3.1</b><br/>DAGs: alerts_etl, fetch_alerts_in_ua (Lab 2),<br/>ingest_telegram (Lab 3), forecast_6h (Lab 4)<br/><i>AWS: MWAA</i>"]
        minio[("<b>MinIO</b> pgsty/silo<br/>buckets dvc · lake · mlflow<br/><i>AWS: S3</i>")]
        dvc["<b>DVC 3.67.1</b><br/>data versions<br/><i>AWS: S3 remote</i>"]
        mlf["<b>MLflow 3.16.1</b><br/>runs + model registry<br/><i>AWS: SageMaker with MLflow</i>"]
        pg[("<b>PostgreSQL 16</b><br/>warehouse · MLflow · Airflow<br/><i>AWS: RDS</i>")]
    end
    gh -- "history CSV, once" --> af
    aiu -- "finished alerts, monthly" --> af
    tg -- "new posts, every 2 min" --> af
    af -- "1 writes raw files and features" --> minio
    dvc -- "2 snapshots raw/ and features/<br/>as a data version" --> minio
    af -- "3 trains models, logs params,<br/>metrics, data version" --> mlf
    mlf -- "4 stores model files" --> minio
    mlf -- "4 stores run metadata" --> pg
    af -- "5 writes the 6-h forecast<br/>at 04 / 10 / 16 / 22 (M1 + M3)" --> pg
    style src fill:#f5f5f5,stroke:#999
```

**Always-on (online) side** — the poller and the API react to live alerts and serve requests; monitoring watches both.

```mermaid
flowchart LR
    aiu["Live alerts feed<br/>Ubilling, source=aiu (no token)"]
    subgraph laptop["Laptop · Minikube — always-on (online) side"]
        reg["<b>Image registry</b><br/><i>AWS: ECR</i>"]
        poll["<b>alerts-poller</b> container<br/><i>AWS: EKS</i>"]
        api["<b>kyiv-alerts-api</b> container<br/>FastAPI + models<br/><i>AWS: EKS</i>"]
        mlf["<b>MLflow</b> model registry<br/><i>AWS: SageMaker with MLflow</i>"]
        pg[("<b>PostgreSQL 16</b><br/>live alerts · forecasts ·<br/>predictions + labels<br/><i>AWS: RDS</i>")]
        mon["<b>Evidently + Prometheus + Grafana</b><br/><i>AWS: Model Monitor + CloudWatch</i>"]
    end
    cli["API client<br/>app / Lab 5 agent"]
    onc["On-call engineer<br/>Telegram bot / email"]

    aiu -- "1 polled every 20 s" --> poll
    poll -- "2 stores live alert state" --> pg
    poll -- "3 Kyiv alert started:<br/>POST predict/duration" --> api
    mlf -- "loads champion model" --> api
    reg -- "container images" --> poll
    reg -- "container images" --> api
    api -- "4 reads forecasts + features,<br/>writes M2 predictions" --> pg
    cli -- "5 GET /v1/forecast/latest,<br/>/v1/alert/current" --> api
    mon -- "6 reads predictions + labels,<br/>scrapes API latency / errors" --> pg
    mon -- "7 alarms" --> onc
```

### 6.2 Infra

Lab 2 runs as one `docker compose` project on the laptop (`docker-compose.yml`, all images arm64-native for Apple Silicon); Lab 4 moves the services to Minikube. AWS is the reference cloud because the cost estimate (6.7) is for AWS.

| Component | Local (pinned) | Compose service, host port | AWS managed equivalent |
|---|---|---|---|
| Object storage | MinIO-compatible `pgsty/silo:RELEASE.2026-09-16T00-00-00Z`; buckets `dvc`, `lake`, `mlflow` | `minio`, 9000 (S3) / 9001 (console) | Amazon S3 |
| Data versioning | DVC 3.67.1 + dvc-s3 3.3.0, remote `s3://dvc` on MinIO | host CLI (`uv run dvc`) | DVC with S3 remote |
| Warehouse, serving DB | PostgreSQL 16; databases `airflow`, `mlflow`, `warehouse` | `postgres`, 5433 | Amazon RDS for PostgreSQL |
| Orchestrator | Airflow 3.3.1, LocalExecutor (api-server, scheduler, dag-processor) | `airflow-*`, 8080 | Amazon MWAA |
| Tracking, registry | MLflow 3.16.1, PostgreSQL backend, MinIO artifacts (proxied) | `mlflow`, 5001 (macOS AirPlay holds 5000) | SageMaker with MLflow |
| Feature store | PostgreSQL tables + DVC Parquet snapshot | — | SageMaker Feature Store |
| Container registry | Minikube registry (Lab 4) | — | Amazon ECR |
| Cluster (API, poller) | Minikube 1.33.1, Kubernetes 1.30.0 (Lab 4) | — | Amazon EKS |
| Model monitoring | Evidently (Lab 4) | — | SageMaker Model Monitor |
| Metrics, dashboards | Prometheus, Grafana (Lab 4) | — | Amazon CloudWatch |

Resources: Docker Desktop with ≥ 6 GB memory (the stack uses ~3 GB idle). A monthly batch runs in ~10 s; training both models takes < 30 s on CPU.

**Managed vs self-run, example (Airflow → MWAA):** gain no scheduler or database upkeep, upgrades and scaling handled by AWS; lose control over versions and plugins, and pay the largest line of the bill (6.7) for ~5 small DAGs. Hence the cost estimate assumes Airflow self-hosted on EKS.

Lab 3 adds pgvector 0.8.7 and an LLM endpoint (hosted API **A**, no local GPU).

### 6.3 Performance

- **M1 batch:** the answer changes four times a day → precompute, and the API reads a row. DAG ≤ 5 min (features ≤ 1 min, scoring < 1 s).
- **M2 online, event-triggered:** needs the alert map at the start moment. Budget ≤ 60 s p95 = 20 s poll + ≤ 5 s features + ≤ 1 s inference + feed lag (≤ 30 s **A**, measured in Lab 4).
- **LLM job (measured, 40 calls each):** retrieval + Gemini p50 2.35 s, p95 3.02 s, first (cold) call 2.07 s; Flash on all window posts p50 3.79 s, p95 12.85 s, so the top-8 prompt (~950 input tokens vs ~4,000) cuts the tail by 4×. The window query is a btree range scan + exact ranking of a few hundred rows. Budget for an interactive answer: ≤ 5 s p95, met. Rebuilding an index for the whole corpus on an Apple-silicon CPU: e5-small 172 s, MiniLM 227 s, e5-base 572 s.
- **Size:** ~8.8k hourly rows and ~500 alerts per year → training takes minutes on CPU. One API replica handles ≤ 10 req/s.

### 6.4 Security

- **Secrets live only in `.env`** (git-ignored; `.env.example` holds local placeholders): alerts.in.ua token (only for `month_ago` back-fill; the live feed needs none), MinIO keys, PostgreSQL passwords, Airflow JWT secret, `GEMINI_API_KEY`. Kubernetes Secrets in Lab 4, a cloud secret manager in the cloud variant.
- **DVC:** the remote is committed in `.dvc/config` (`s3://dvc`, endpoint only); credentials come from the environment (`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`, exported by the Makefile) and are never written to git.
- **MLflow** serves artifacts through the tracking server (`--serve-artifacts`), so training clients need no object-store credentials; `--allowed-hosts` limits the Host header to localhost and the compose network.
- **Ports bind to 127.0.0.1 only.** Airflow runs with `SIMPLE_AUTH_MANAGER_ALL_ADMINS` (no login) — acceptable on localhost, never on a shared network.
- `POST /v1/predict/duration` is reachable only from the poller (NetworkPolicy, Lab 4); cloud read endpoints use an API key + rate limit.
- alerts.in.ua terms: proxy the data and never expose the token; our API is that proxy. Ubilling asks for ≤ 2 req/s (ban risk); the poller sends 1 per 20 s.

### 6.5 Data privacy

- No personal data: alerts are public region-level events; the Air Force channel is official (no comments, no users). @smolii_ukraine is a public channel; only post text, time, id and views are kept, no comments, reactions or user names.
- Post text is sent to the Gemini API (paid tier: under Google's paid-tier terms prompts are not used to improve their models). Only public posts go there, never alert-map state or user questions with personal data.
- Prompt injection: posts are untrusted text inside the prompt. The answer is confined to a JSON schema, ids not shown to the model are dropped and counted, and the summary is advisory only.
- License: the Vadimkin dataset is MIT; we keep its files unchanged in the DVC remote and cite the source and commit.
- alerts.in.ua data is stored only in the private lake (raw JSON as fetched) and is not redistributed; the DVC remote is local MinIO.
- Retention: data and models until 2027-12 **A**; API logs 30 days (Lab 4). Nothing about strike locations or air defence is produced: the corpus holds only posts that were already public, and the job summarises them after the fact.

### 6.6 Monitoring and alarms

Labels arrive late, so there are four layers; only the last one needs labels.

- **Drift measures:** PSI = Σ (cur% − ref%) · ln(cur% / ref%) for numeric features. Jensen–Shannon distance for distributions (0 = identical, 1 = disjoint).
- **Reference = champion's training data**, fixed until retrain. With a rolling 90-day reference the raion-switch alarm faded after ~5 weeks.
- **Thresholds = 95th percentile of normal variation** (2025–26), alarm after 2 consecutive daily checks. The textbook PSI 0.2 would fire every few weeks here.

| Layer | Signal | Window | Threshold | Severity |
|---|---|---|---|---|
| Data health | live-feed poll failing, or response `cachedat` older than 2 min | — | > 2 min | 🔴 |
| | newest finished alert (catches re-reading one partition) | — | > 2 days | 🔴 |
| | Telegram silent / unparsed posts | — / 24 h | > 6 h / > 20% | 🟡 / 🔴 |
| | rows per batch vs 3-month median | batch | ±50% | gate fails |
| | DAG failure; M1 not published | — | any; issue + 5 min | 🔴 |
| | API p95 latency / 5xx | 15 min | > 200 ms / > 1% | 🔴 |
| Input drift | raw active records per hour (upstream signal) | 7 d | PSI > 0.5 (Nov 2025: 0.11 → 0.70 → 1.00) | 🟡 |
| | oblasts under alert | 7 d | PSI > 0.5 (normal median 0.11, p95 0.47) | 🟡 |
| | Kyiv start profile by hour of day | 30 d vs 180 d (7 d too noisy: one week = 0.34) | JS > 0.13 (normal median 0.075, p95 0.131) | 🟡 |
| | Telegram threat-type mix | 7 d | JS, calibrated in Lab 2 | 🟡 |
| | skew: live vs rebuilt oblasts under alert | 7 d | mean diff > 1 | 🟡 |
| Prediction drift | mean P(start) per issue time; share of 🔴 hours; mean P90; type mix | 7 d vs 30 d | > 50% change (type: JS > 0.1) | 🟡 |
| Performance | M1: 30-day Brier minus baseline Brier, same hours | rolling | > 0 for 14 days in a row | 🟠 |
| | M2: P90 kept, last 30 alerts | rolling | < 80%² | 🟠 |
| | M3: log-loss vs last-month mix, last 30 typed alerts | rolling | worse than baseline | 🟠 |

² If true coverage is 90%, observing < 80% in 30 alerts has probability 2.6% (false alarm); at true 75% it is caught with probability 65%.

**Routing:** Prometheus + Evidently → Grafana alert rules → Telegram bot + email. 🔴 immediately, 🟡 daily digest, 🟠 immediately with the comparison report.

| Input drift | Performance | Action |
|---|---|---|
| yes | OK | log, watch (e.g. raion switch, absorbed by aggregation) |
| yes | worse | retrain, compare (5.4), promote or `FORCE_BASELINE` |
| no | worse | concept drift: retrain, look for a new (Telegram) feature |
| data health 🔴 | any | serve baseline with `degraded: true`, fix or switch source |

### 6.7 Cost

AWS eu-central-1, monthly. **A:** rough list prices, to verify in the [AWS Pricing Calculator](https://calculator.aws/).

| Item | Sizing | USD |
|---|---|---|
| S3 + ECR | ≤ 10 GB + ≤ 1 GB | < 2 |
| RDS PostgreSQL | db.t4g.micro, 20 GB | 15–20 |
| EKS | control plane + 2 × t3.medium (API, poller, Airflow, MLflow self-hosted) | ~135 |
| CloudWatch | basic metrics, ≤ 5 GB logs | 5–10 |
| **Self-hosted total** | | **~150–170** |
| + MWAA, SageMaker MLflow, Model Monitor instead | smallest classes | + 300–570 |

The orchestrator and tracking server dominate the bill, not the model → self-hosted variant.

**LLM job (Lab 3), measured.** Gemini 3.8 Flash list price $0.75 / $3.75 per 1M input / output tokens (thinking tokens billed as output), pinned in `configs/rag.yaml` and multiplied by the tokens each run reports.

| Path | Tokens per request (in / out incl. thinking) | USD per request | 40-window evaluation run | One question every 10 min (4,320 / month) |
|---|---|---|---|---|
| retrieval + Flash (e5-small) | ~955 / ~102 | **0.0011** | 0.044 | **~4.75** |
| Flash on all window posts | ~3,970 / ~459 | 0.0047 | 0.188 | ~20.30 |
| pre-labelling with gemini-4-argon (one-off, $2 / $10 per 1M) | all posts of a window, thinking on | — | not logged (one-off, 40 calls) | — |

Retrieval makes each answer ~4× cheaper and shorter-tailed. A local model would remove the API bill but needs a GPU or a slow CPU server (7.1).

### 6.8 Integration points

| Party | Contract |
|---|---|
| Vadimkin dataset (upstream) | no SLA; files pinned by commit (`58ee75b` official, `346bcff` volunteer) and by DVC hash; columns `oblast, raion, hromada, level, started_at, finished_at, source` (official) / `region, started_at, finished_at, naive` (volunteer), UTC. A schema change fails `validate`; a stale or coarser file fails `quality_check` |
| alerts.in.ua (upstream) | token; ≤ 8 req/min (hard limit 12), history 2/min → 25 oblasts take ~13 min; `month_ago` covers only the last 30 days, so the DAG `fetch_alerts_in_ua` runs on the 1st and 15th (gaps ≤ 17 days) and keeps every response as fetched; a month is built from the union of fetches (latest version of each alert id wins). Raion-level alerts name the raion in `location_title` (`location_raion` is empty) |
| Ubilling live feed (upstream, Lab 4) | unofficial, no SLA or terms; ≤ 2 req/s; `source` pinned to `aiu` (the unnamed `default` can switch provider); `cachedat` checked; schema change → quality gate fails |
| Telegram web preview (upstream) | no contract or SLA; `t.me/s/<channel>?before=<id>` returns 20 posts of HTML, scraped at ≤ 1.25 req/s with a descriptive User-Agent; post id `<channel>/<id>` is stable; markup change → the parser returns 0 posts (smoke test) |
| Gemini API (upstream, Lab 3) | Google GenAI SDK, paid tier; `GEMINI_API_KEY`; model id pinned (`gemini-3.8-flash`), price pinned with an expiry date in `configs/rag.yaml`; structured output by response schema; errors recorded per call, never raised mid-evaluation |
| Feature tables (internal) | `features.m1_hourly`, `features.m2_alert`, keyed by time + `feature_version`; a new version is added alongside, never overwrites; training reads the DVC snapshot, serving (Lab 4) the same tables |
| Model registry (internal → Lab 4) | names `kyiv-m1-start-risk`, `kyiv-m2-duration`; Lab 4 loads `models:/<name>@champion`; every version is tagged `data_version` and `git_commit`; clients set `MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false` |
| API clients (downstream) | versioned JSON `/v1`; `model_version`, timestamps, `degraded`; breaking changes only in `/v2` |

### 6.9 Risks and uncertainties

| Risk | Mitigation |
|---|---|
| upstream stale (official file stopped 2026-09-07) | freshness alarm; alerts.in.ua history (token); from Lab 4 the poller's own log |
| live proxy is unofficial: can stop, rate-limit or change provider | pin `source=aiu`; backup `source=ual`; `cachedat` age check; switch to direct alerts.in.ua when the token arrives (same schema) |
| permanent and non-air alerts in the live feed (Luhansk, Crimea, artillery) | filter `alert_type = air_raid`, drop permanent regions |
| upstream semantics change (raion switch) | aggregation in preprocessing; raw-record PSI |
| Telegram blocks scraping / changes markup | parser smoke test in gate; Telethon fallback |
| leakage from edited or late posts | `datetime_utc < t`; compare history vs live features in Lab 4 |
| tactics change faster than monthly retrain | daily drift + performance monitoring; on-demand retrain; baseline fallback |
| small M2 data (~500 alerts/year) | simple regularised quantile models; report intervals |
| M1 start timing barely predictable (F1 0.27 vs 0.25) | honest targets; Telegram features tested in Lab 2; baseline stays the fallback |
| Gemini model deprecated or re-routed, price change | pinned id + price with its source and end date in `configs/rag.yaml`; `llm_model_version` from the response is a tag on every run; re-run the evaluation set before switching |
| LLM labels biased toward the answering model | labels from a different, stronger model plus the keyword rule; every disagreement reviewed by hand |
| token blocked for rate limits | client-side limiter 8/min, `If-Modified-Since` |

## 7 Appendix

### 7.1 Alternatives

| Alternative | Evidence / reason | Decision |
|---|---|---|
| Weather features (Open-Meteo Previous Runs, 24 h-ahead forecast) | Kyiv and Tsymbulova launch area: AUC 0.46–0.55 in every year; hold-out Brier worse than climatology | rejected |
| Hour label "≥ 10 min" / "any minute" | loses 4% of alerts / over-counts long alerts (5.1) | rejected → "starts in hour" |
| One binary label per night | arbitrary threshold, no timing | rejected → hourly M1 |
| 12-h horizon issued at 10:00 / 22:00 (Lab 1–2 design) | skill sits in the first 1–2 hours; averaged over 12 h AUC 0.591, below the 0.60 guardrail (5.1) | replaced → 6 h every 6 h (AUC 0.610) |
| Sliding 3-month training window | lost to the hour-of-day baseline in every test month (−2.9% Brier, AUC 0.542); too few starts, no seasonality (5.4) | rejected → expanding window, monthly retrain |
| Feast as the feature store | 20 features from two tables, already built as of *t* by the pipeline (no join that could leak); another service on 4 GB | rejected; training reads the DVC snapshot, serving reads PostgreSQL |
| Kafka + Spark | ~1.4 Kyiv alerts/day; 20 s polling is enough | rejected |
| alerts.in.ua API directly for live alerts | official feed; needs a token (requested, not received) | Ubilling proxy (`source=aiu`, same schema) for now; switch when the token arrives |
| Local LLM with vLLM / llama.cpp instead of a hosted API | no GPU in the stack; a 7–8B model on the laptop CPU takes tens of seconds per answer and needs ~6 GB RAM next to Airflow and MLflow; the hosted Flash costs ~$0.001 per answer (6.7) | rejected → Gemini API, pinned id |
| HNSW index on the embeddings | the query is always limited to a 3-h window; HNSW ranks first and filters after, so it can return fewer than k posts; exact ranking of a few hundred rows is milliseconds | rejected → btree on `posted_at` + exact cosine |
| A stronger LLM (gemini-4-argon) as the answerer | ~2.7× Flash's price per token, and it produced the labels, so it cannot be graded fairly on them | used for labels only |
| Varying top_k or chunk size instead of the encoder | posts are one chunk each (chunk size changes nothing); `top_k` mostly trades the recall ceiling for cost, visible already from `recall_ceiling` | encoder varied; `top_k` is the next knob (7.2) |
| Telethon instead of web preview | needs a personal account and a session secret | fallback only |

### 7.2 Data pipelines, versioning and baseline

**Pipeline.** `alerts_etl` backfills 54 monthly batches (2022-03 … 2026-08) in ~7 min; all pass the gate. The first run also failed 2026-08 (official file stops 2026-08-29 05:00 UTC) and 2026-09 (volunteer source, 2.20× the 3-month median); 2026-08 was then accepted as a documented gap and 2026-09 removed (5.2). A broken copy of 2025-06 (30% of ends before starts) passed `validate` and failed `quality_check` on `invalid_share` = 0.37. Rerunning 2025-06 twice gave identical counts (1,440 episodes, 720 M1 rows, 34 M2 rows). Evidence: README §2–3.

**Experiment results** (MLflow at http://localhost:5001, experiments `kyiv-m1-start-risk` and `kyiv-m2-duration`; registered M1 v6 and M2 v7, alias `champion`, commit `52cc28c`; M1 per-lead scores are step metrics (`*_by_lead`, step = lead hour); every run carries `logging_manifest.json`, its train / hold-out datasets and, for M1, `metrics_by_lead.json`). Data version `3e7de26af5034174fa82e86bb10110d9.dir` (features v3), `352f88a87031f9e22af2cce021c2a4a7.dir` (raw); hold-out 2026-03-01 … 2026-08-28; M1 refit monthly on all data from 2023-01, M2 trained 2023-01 … 2026-02.

| M1, 6-h forecast, 4,344 hours (start rate 7.3%) | Brier (main) | AUC | log-loss | F1 ±1 h |
|---|---|---|---|---|
| trivial: hour-of-day rate | 0.06744 | 0.570 | 0.2625 | **0.351** |
| rule: same hour, last 30 days | 0.06817 | 0.549 | 0.3423 | 0.323 |
| HistGradientBoostingClassifier, monthly refit | **0.06655** | **0.614** | **0.2554** | 0.339 |

| M1 by lead hour | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Brier model / trivial | 0.0541 / 0.0559 | 0.0630 / 0.0634 | 0.0808 / 0.0820 | 0.0589 / 0.0595 | 0.0616 / 0.0623 | 0.0810 / 0.0816 |
| AUC model / trivial | 0.707 / 0.522 | 0.599 / 0.549 | 0.616 / 0.627 | 0.609 / 0.551 | 0.576 / 0.527 | 0.543 / 0.540 |

| M2, 325 alerts | pinball P50 | pinball P90 (main) | P50 ±15 min | P90 kept | P90 width |
|---|---|---|---|---|---|
| rule: last-12-months 33 / 224 min | 16.13 | 21.39 | **48.9%** | 97.5% | 224 min |
| quantile GBM, monotonic, + `oblasts_started_prev_5m` | **15.83** | **14.72** | 48.3% | 92.3% | **133 min** |

- M1: with the 6-h horizon and monthly refits the model beats the hour-of-day profile by 1.3% Brier and at every lead, and clears the 0.60 AUC guardrail (0.614), in line with the horizon analysis in 5.1 (1.3%, 0.610). Skill sits in lead 0 (AUC 0.71) and fades to 0.54 by lead 5. The trivial profile still wins on F1 and on AUC at lead 2. Whether the gain survives the 5.4 promotion rule (bootstrap CI) is a Lab 4 check.
- M2: the model wins the main metric (−31% P90 pinball, P90 91 min shorter, coverage 92.3% inside the band); the rule is marginally better on P50 ±15 min. Versus v2 features (no simultaneous-starts feature, no constraints): P90 pinball 15.00 → 14.72, P50 pinball 15.68 → 15.83. The monotonic constraints hold on real hold-out alerts.
- Reproduction: `make reproduce` retrained both registered versions from `git_commit` and `data_version` with |Δ| = 0 on the same machine (tolerance 1e-4 Brier, 0.1 min pinball).

**Lab 3: LLM job** (MLflow experiment `kyiv-llm-rag`, commit `3039de4`; corpus `5402f138e78962a8febde1b153cd99b9.dir`; eval set `3b97ec694dd1…`; prompts `answer_v1` `1af0377f2374…`, `baseline_v1` `51fa64296329…`). 40 windows, 2,404 labelled posts (1,134 relevant). Every LLM run: 40 calls, 0 errors, 0 hallucinated ids.

| Run | hit@1 | hit@3 | hit@5 | hit@8 | MRR | answer F1 | F1 given shown | recall ceiling | threat-type acc. | USD / request | p50 / p95 s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| keyword rule | — | — | — | — | — | **0.794** | — | — | 0.700 | 0 | — |
| Flash, all window posts | — | — | — | — | — | 0.508 | 0.508 | 1.000 | 0.625 | 0.0047 | 3.79 / 12.85 |
| **RAG, multilingual-e5-small** | **0.676** | **0.882** | 0.912 | **0.971** | **0.785** | 0.202 | **0.896** | 0.126 | **0.875** | **0.0011** | 2.35 / 3.02 |
| RAG, multilingual-e5-base | 0.618 | 0.853 | **0.941** | **0.971** | 0.755 | 0.199 | 0.885 | 0.126 | 0.825 | 0.0011 | 2.52 / 3.78 |
| RAG, paraphrase-multilingual-MiniLM-L12 | 0.647 | 0.794 | 0.824 | 0.882 | 0.726 | 0.172 | 0.871 | 0.111 | 0.675 | 0.0011 | 2.43 / 3.53 |

- **Encoder:** e5-small is best or tied on every rank metric except hit@5, and 3.3× faster to index than e5-base → `retrieval.embedding_model: e5_small`. MiniLM (a paraphrase model, used without query / passage prefixes) drops a relevant post from the top 8 in 4 of 34 windows.
- **Answer F1 is bounded by k:** windows hold ~33 relevant posts, the LLM sees 8, so at most 12.6% of the relevant posts reach it. Given what it sees, Flash returns the right ids with precision 0.89 and recall 0.90. With every post in the prompt it is precise (0.88) but finds only 36% of them.
- **Threat type** — the part of the answer a user acts on — is best with retrieval (35 / 40 windows vs 25 without retrieval and 28 for the rule): a short, ranked context beats a long, noisy one. Misses (5): drone windows answered "ballistic" (2), "aviation" (1) and "none" (1); one aviation window answered "none".
- **The rule baseline wins post-level F1 (0.79)** but is optimistic: it was revised after reading the pre-label disagreements, and its Kyiv-channel clause mirrors the labelling rubric. It gives no summary and cannot weigh conflicting posts.
- `thinking_budget: 0` still produced 0–915 thinking tokens per run (up to 20% of output tokens, ≤ 8% of a run's cost); they are billed and counted.
- **Next:** if the job must return every relevant post (e.g. counts as M1/M2 features), raise `top_k` or retrieve per channel and re-run the same hashed set; for the summary and threat type the current setting is enough.

### 7.3 Serving, monitoring and drift

*Placeholder — Lab 4.*

### 7.4 Milestones

- **Lab 2 (done):** DVC-versioned raw data on MinIO; `alerts_etl` monthly DAG with idempotent loads and a quality gate; feature tables with `feature_version`; M1 and M2 rule baselines vs GBMs on a time split; MLflow with lineage tags, registry and `champion` alias; one-command reproduction; Feast point-in-time demo; M1 6-h forecast with monthly refits and per-lead metrics; M2 monotonic constraints; alerts.in.ua history archived by a DAG; two-level (red/yellow) alert regime from 2026-10.
- **Lab 3:** Telegram ingestion (back-fill from 2024-01), threat-type parser, M3 and the Telegram features for M1/M2; LLM answers "what is the current threat to Kyiv and why this forecast", retrieving Air Force posts from pgvector; evaluated on a fixed question set.
- **Lab 4:** API + poller on Minikube with probes; forecast DAG; Evidently / Prometheus / Grafana; replayed drifts (raion switch, 2025 long nights, source switch) reach on-call.
- **Lab 5:** agent with tools (active alerts, forecast, duration, post search); traced, ten-task evaluation.
