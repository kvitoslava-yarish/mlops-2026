# Kyiv air-alert forecasting — design document

Author: Kvitoslava Yarish · Course: KSE Introduction to MLOps and ML System Design, 2026 · Status: Lab 1 (design)

Template: [eugeneyan/ml-design-docs](https://github.com/eugeneyan/ml-design-docs). Section numbers are fixed; later labs revise sections by these numbers. Items marked **Assumption** are design decisions taken without full information and are revisited in the lab named next to them.

---

## 1 Overview

Kyiv residents and businesses plan their days and nights around air-raid alerts, but they learn about an alert only when the siren starts and never know how long it will last.
The system issues a 12-hour forecast twice a day (10:00 and 22:00 Kyiv time) with the probability of an alert in each hour of the window, and predicts the duration of every Kyiv alert (median and 90th percentile) the moment it starts.
It learns from the public history of air-raid alerts across Ukraine since 2022 and from the Ukrainian Air Force's public Telegram channel, and it retrains as new alerts and posts arrive.
The desired outcome is a planning aid served through a REST API that is measurably better than simple hour-of-day and recent-history baselines and that notices when attack patterns change; the official alert always takes precedence over the model.

## 2 Motivation

**Why a system and not a notebook.** Alert patterns are not stationary. In our data, the share of nights with at least 60 minutes of alert in Kyiv moved from 21% (2023) to 41% (2025) and back to 29% (2026), the median alert got shorter (55 → 30 min), and in November 2025 alerts across Ukraine switched from oblast-level to raion-level announcements. A model fitted once in a notebook is wrong within months. The value comes from re-ingesting new alerts every day, retraining on a schedule, and detecting when the inputs change.

**Who consumes the predictions, and how often.**

| Prediction | Consumer | Frequency |
|---|---|---|
| M1: 12-hour hourly alert probabilities | API clients: a planning app or bot (later), the Lab 5 agent | twice a day (10:00, 22:00), read many times |
| M2: duration of the current alert (P50, P90) | same API clients | at every Kyiv alert start, about 1.4 times a day (511 alerts in 2025) |

**What happens when they are wrong.**

- M1 says low risk and a long night alert happens: a user who planned around the forecast is disrupted (meeting, delivery, sleep). Cost is inconvenience, not safety, because the siren still sounds.
- M1 says high risk and nothing happens: users over-prepare; repeated false alarms erode trust and people stop using the forecast.
- M2 P90 is too short: a user leaves shelter planning too early or schedules something that gets interrupted. This is the more harmful error, so M2 is evaluated on whether P90 is kept, not only on average error.
- Safety framing: every response states that it is a planning estimate and that the official alert overrides it. The system never advises whether to go to a shelter.

## 3 Success metrics

Three layers, kept apart. Each has a target or a stated assumption.

| Layer | M1: 12-hour forecast | M2: alert duration |
|---|---|---|
| **Business** (decision) | **Warned share:** of 12-hour windows with ≥ 60 min of Kyiv alert, at least **60%** are flagged high-risk in advance while at most **30%** of quiet windows are flagged. **Assumption:** 60/30 is the trade-off a planner accepts; revisited with consumers in Lab 4. Current evidence: history-only GBM reaches 59% at 30%; the hour-of-day baseline cannot separate windows at all (0%). | **P90 kept:** at least **90%** of Kyiv alerts end before the predicted P90, **and** the median predicted P90 is narrower than the baseline's 224 min (last-12-months P90). |
| **Model** (offline) | **Brier score** on hourly probabilities (optimising). **AUC ≥ 0.60** on hourly labels (satisficing). Baselines on the 2026-03…2026-09 hold-out: hour-of-day climatology Brier 0.0981 / AUC 0.578. | **Pinball loss** at q = 0.5 and q = 0.9 (optimising). **P90 coverage 85–95%** (satisficing). Baseline (last-12-months median): MAE 34.6 min. |
| **System** | Forecast published **≤ 5 min** after 10:00 / 22:00. Feature freshness at issue time: alert map ≤ 30 s old, Telegram ≤ 5 min old. | Prediction available **≤ 60 s (p95)** after the alert starts in the alerts.in.ua feed. API p95 latency ≤ 200 ms. |

Optimising vs satisficing: Brier and pinball loss are minimised; AUC, P90 coverage, freshness and latency only have to clear their thresholds.

## 4 Requirements and constraints

### 4.1 Functional requirements

| # | Requirement |
|---|---|
| F1 | **Inputs:** finished Kyiv and Ukraine-wide alert intervals (history and monthly), currently active alerts (polled every 20 s), Air Force Telegram posts (scraped every 2 min). |
| F2 | **M1 output:** at 10:00 and 22:00 Kyiv time, for the next 12 hours: 12 hourly probabilities that Kyiv is under alert for ≥ 10 min of that hour, P(≥ 60 min of alert in the window), expected alert minutes, `model_version`, `issued_at`, `degraded` flag. |
| F3 | **M2 output:** when a Kyiv alert starts: predicted duration P50 and P90 in minutes, threat type used, `model_version`, `alert_started_at`, `degraded` flag. |
| F4 | **Who calls:** any API client, read-only: `GET /v1/forecast/latest`, `GET /v1/alert/current`, `GET /v1/health`. The internal `POST /v1/predict/duration` is called only by the alerts poller inside the cluster. |
| F5 | **Fallback:** if a model, a feature or a data source is unavailable, the API returns the baseline prediction with `degraded: true` instead of an error. |
| F6 | **Labels:** every prediction is joined with its outcome once known (window end for M1, alert end for M2) and stored for monitoring and retraining. |

### 4.2 Non-functional requirements

One row per category; every row carries a number or an explicit assumption.

| Category | Requirement |
|---|---|
| **Freshness** | M1 published ≤ 5 min after issue time. Active-alert state ≤ 30 s old (20 s poll). Telegram posts ingested ≤ 5 min after publication (2-min scrape). Finished-alert labels ingested monthly for training (alerts.in.ua `month_ago`), daily for monitoring. |
| **Latency and throughput** | API p95 ≤ 200 ms (reads precomputed rows). M2 end-to-end ≤ 60 s p95 from alert start. Load: **Assumption** ≤ 10 requests/s peak (single consumer and demo); about 2 M1 runs and ~1.4 M2 predictions per day. External limits: alerts.in.ua ≤ 8 requests/min (hard limit 12/min; history endpoint 2/min). |
| **Availability and fallback** | **Assumption:** 99% monthly availability on the local Minikube deployment (laptop may sleep). Fallback per F5: hour-of-day climatology for M1, last-12-months median/P90 for M2. If alerts.in.ua is unreachable for > 2 min, M2 is marked `degraded` and `/v1/health` reports the stale source. |
| **Reproducibility** | Every registered model is traceable to a DVC data version, a git commit and its parameters in MLflow; re-running training with the same data version and commit gives identical metrics (fixed seeds). The environment is pinned with `uv.lock`; `make setup` rebuilds it from a clean clone. |
| **Adaptability** | New month of alerts and posts is ingested by one DAG run. Retraining is monthly on an expanding window, plus on demand when monitoring fires. A new model or feature enters as `challenger` and is promoted only by the rule in 5.4. |
| **Observability without ground truth** | Labels arrive late: up to 12 h for M1 (window end) and up to ~11 h for M2 (longest alert 673 min; P99 440 min). Before labels arrive we monitor input drift (oblasts-under-alert distribution, Telegram threat-type mix and post volume), prediction drift (mean predicted probability, mean predicted P90) and data freshness. |
| **Cost and resource ceilings** | Local: laptop with Minikube at 4 GB RAM (course reference), no GPU. Cloud: estimated in 6.7, **Assumption** ceiling $200/month for the self-hosted-on-Kubernetes variant. External: free tiers only (alerts.in.ua token, public Telegram preview). |

### 4.3 In scope

- Kyiv City only (other oblasts are inputs, not targets).
- M1 (12-hour hourly forecast at 10:00 and 22:00) and M2 (duration at alert start).
- Data from two sources: air-alert data (dataset + alerts.in.ua API) and the Air Force Telegram channel.
- REST API as the only consumer interface.

### 4.4 Out of scope

- Safety advice (whether to go to a shelter), push notifications to the public, a user-facing app.
- Predicting targets, strike locations, damage, or anything about air-defence positions.
- Red/yellow alert levels as a target: alerts.in.ua exposes them only for current alerts, so we start collecting them in Lab 2 but do not model them yet.
- Threat-type forecast for a window (drones vs missiles) — **stretch goal (M3)** after the Telegram label quality is verified.
- Weather features (rejected with evidence, see 7.1).

## 5 Methodology

### 5.1 Problem statement

- **M1: probabilistic classification, one row per (issue time, hour ahead).** For issue time *t* ∈ {10:00, 22:00} and lead *h* ∈ {0…11}, predict P(Kyiv under alert ≥ 10 min during hour *t + h*). Window summaries (P(≥ 60 min), expected minutes) are derived from the 12 hourly probabilities. One model serves both issue times; issue time and lead are features.
- **M2: quantile regression.** At the start of a Kyiv alert, predict the 0.5 and 0.9 quantiles of its duration in minutes.

**Rejected framing: one binary label per night** ("will there be ≥ 60 min of alert in Kyiv tonight?"). It is simpler, but the 60-minute threshold is arbitrary, it says nothing about *when* during the night, and it needs a second model for the day. The hourly framing gives timing, covers day and night with one model, and the window-level answer falls out of it.

**Also rejected: M2 as classification into duration buckets** (< 30 min, 30–120, > 120). Buckets lose the tail that matters for planning; quantiles keep it.

### 5.2 Data

**Sources and arrival pattern**

| Source | Content | Arrival | Role |
|---|---|---|---|
| [Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset) (MIT), `official_data_en.csv` | All official alerts since 2022-03-15, UTC start/end, oblast/raion/hromada level | Bootstrap once; repo says "updated daily" but our copy ends **2026-09-07** | Training history and labels |
| Same repo, `volunteer_data_en.csv` | Alerts since 2022-02-25, oblast level, collected from eTryvoga | Current to 2026-10-23 | Cross-check; fills months after 2026-09-07 |
| [alerts.in.ua API](https://devs.alerts.in.ua/) | Active alerts (`/v1/alerts/active.json`, IoT v2 with red/yellow), last month of finished alerts (`/v1/regions/{uid}/alerts/month_ago.json`) | Active: poll every 20 s. History: monthly batch | Live features (serving), monthly batches and labels going forward |
| Telegram [@kpszsu](https://t.me/s/kpszsu) (Ukrainian Air Force), public web preview | Short typed posts: emoji + threat type + location + heading (e.g. "🛵 Ударні БпЛА … курсом на …"); morning summaries with counts of drones and missiles | ~10+ posts/hour during attacks; scraped every 2 min; history back-filled from 2024-01-01 | Threat-type features, attack-size features, M2 threat type, Lab 3 corpus |

**Labels and their delay**

- M1 hourly label: Kyiv under alert ≥ 10 min in that hour → known at the end of the hour; the whole window is labelled 12 h after issue.
- M2 label: alert duration → known when the alert ends (median 38 min, P90 161 min, P99 440 min, max 673 min).

**Available at serving time vs training time**

| Information | Training (offline) | Serving (online) |
|---|---|---|
| Alert intervals | complete start + end from the dataset | only *started* alerts are known; ends arrive later |
| Oblasts under alert at time *t* | reconstructed from dataset intervals | current snapshot from alerts.in.ua active endpoint (different producer) |
| Telegram posts | scraped history: **current** (possibly edited) text | text as first published |
| Alert level red/yellow | not available | available (collected from Lab 2, not used yet) |

**Data quality findings (measured, Lab 1)**

- 38% of rows in `official_data_en.csv` are exact duplicates (113,845 rows); Kyiv: 4,141 rows → 2,267 alerts after de-duplication.
- The official file stopped updating on 2026-09-07 despite "updated daily" → freshness check needed (6.6).
- Since November 2025 almost all alerts are raion-level (oblast-level rows fell from ~700/month to < 60/month) → features that count "oblasts under alert" changed meaning; this is real drift.
- Volunteer data: 8.6% of Kyiv rows have no recorded end and a guessed `+30 min` end (`naive=True`) → excluded from M2 labels.
- Yearly Kyiv alert counts agree between official and volunteer data within ~5% (2023–2025).

**Storage formats**

- Raw layer: exactly as fetched (CSV from GitHub, JSON from alerts.in.ua, CSV parsed from the Telegram preview) in MinIO `raw/`, partitioned by source and date. Raw is never edited, so any bug in preprocessing can be replayed.
- Clean and feature layers: Parquet in MinIO (columnar, typed, compressed, fast for time-range scans in training).
- Serving layer: PostgreSQL tables (current alert state, online features, forecasts, predictions joined with labels) because the API needs indexed point lookups and the poller needs transactional writes.

**Checklist justification (step 1 of the lab)**

| Item | Justification |
|---|---|
| **A time axis** | Every alert has UTC start and end since 2022; every post has a UTC timestamp. Lab 2 ingests "the next month" from alerts.in.ua `month_ago` plus newly scraped posts. Lab 4 can replay real drift: the November 2025 raion switch and the 2025 rise in long night alerts. |
| **Enough history to hold out a period** | 4.5 years of alerts (2,267 Kyiv alerts; ~1,300 nights). Final hold-out: 2026-03-01 → 2026-08-31 (6 months, 4,536 hourly rows for M1, 349 alerts for M2). Telegram is back-filled from 2024-01-01, so models that use Telegram train on 2024-01 → 2026-02 (26 months). |
| **Labels with a known arrival delay** | M1: 1–12 h (end of the hour / end of the window). M2: end of the alert, ≤ 11.2 h observed. Both are much shorter than the monthly retraining cycle, so Lab 4 monitors performance on the same day, not a quarter later. |
| **A text surface** | The Air Force Telegram channel (tens of thousands of posts since 2024) is both a feature source and the Lab 3 retrieval corpus: "what is the current threat to Kyiv and why does the forecast say so?" grounded in recent posts and morning summaries. |

### 5.3 Techniques

**Baselines first (rule or trivial model), then the candidate.** A candidate is useful only if it beats the rule baseline, not just the trivial one.

| | Trivial baseline | Rule baseline | Candidate |
|---|---|---|---|
| M1 | P(alert hour) by hour of day from training data. Hold-out: Brier 0.0981, AUC 0.578 | Same hour's alert rate over the last 30 days. Hold-out: Brier 0.0984, AUC 0.584 | Gradient-boosted trees (LightGBM / sklearn HistGradientBoosting), later logistic regression for comparison. First history-only version: Brier 0.0946, AUC 0.636 (AUC 0.79 at lead 0 h) |
| M2 | Last-12-months median and P90 (33 / 224 min). Hold-out MAE 34.6 min, P90 coverage 0.97 | By threat type at start, from Telegram: ballistic → short, drones → long (to be measured in Lab 2) | Quantile gradient boosting (q = 0.5, 0.9). A naive first version scored MAE 45.1 (worse than baseline: drift + raion switch), so features are rebuilt as raion-aware |

**Features and where each value comes from at serving time**

| Feature | Model | Training source | Serving source |
|---|---|---|---|
| hour of day, lead, issue time, day of week, holiday flag | M1 | computed from timestamp | computed from timestamp |
| moon illumination | M1 | astronomical formula | same formula |
| Kyiv alert rate in the same hour, last 30 days | M1 | clean alert table (Parquet) | same feature code on PostgreSQL alert table |
| Kyiv alert minutes in the previous window | M1 | clean alert table | PostgreSQL alert table |
| days since last massive night (≥ 180 min of alert) | M1 | clean alert table | PostgreSQL alert table |
| Kyiv currently under alert | M1 | reconstructed from intervals at *t* | poller state (alerts.in.ua active) |
| **number of oblasts under alert now** (raion alerts aggregated to oblast) | M1, M2 | reconstructed from dataset intervals | alerts.in.ua active snapshot ← **different producer from training** |
| oblasts that started an alert in the last 5 / 60 min | M2 | dataset intervals | poller history in PostgreSQL |
| minutes since Kyiv oblast (Kyivska) alert started | M2 | dataset intervals | poller state |
| minutes since previous Kyiv alert ended | M2 | clean alert table | PostgreSQL alert table |
| count of Air Force posts by threat type (drones / jet drones / missiles / ballistic / guided bombs) in last 6 h and 12 h, and mentions of Kyiv | M1 | scraped posts, `datetime_utc < t` | live Telegram ingest |
| drones and missiles in last night's morning summary | M1 | parsed summaries | parsed live summary |
| threat type in the 30 min before alert start | M2 | scraped posts | live Telegram ingest |

Two features have a serving source different from their training source: "oblasts under alert now" (Vadimkin dataset vs alerts.in.ua live) and every Telegram feature (edited history vs first-published text). Both are monitored for training–serving skew (6.6). All features are computed by one Python module used by both the offline and the online path.

### 5.4 Experimentation and validation

- **Split by time, never at random.** Alerts come in waves: consecutive hours and nights are strongly correlated, so a random split puts the same attack into train and test and inflates scores (leakage).
- **Folds:** expanding window with monthly test folds (train on everything before month *m*, test on month *m*). **Final hold-out:** 2026-03-01 → 2026-08-31, touched once per model family.
- **As-of discipline:** every feature is computed only from data with timestamp < issue time (M1) or < alert start (M2); Telegram posts filtered by `datetime_utc`.
- **Offline metrics:** M1 Brier (main), log-loss, AUC, reliability curve, per-lead Brier. M2 pinball loss at 0.5 and 0.9 (main), MAE, P90 coverage.
- **Comparing a new model with the current one (champion/challenger):** both scored on the same last 3 monthly folds; the challenger must be better on the main metric with a day-level bootstrap 95% interval that excludes zero, must not break any satisficing threshold, and must beat the rule baseline. Then it runs one week in shadow mode (predictions logged, not served) before manual promotion.

### 5.5 Human in the loop

- **Drift alert:** goes to the on-call engineer (the author). Within 24 h they open the Evidently report and choose: ignore (documented), retrain, or switch to baseline.
- **Model promotion:** never automatic. The on-call engineer reviews the comparison report (5.4) and moves the MLflow alias `champion` to the new version.
- **Override:** setting `FORCE_BASELINE=true` (environment variable / Kubernetes ConfigMap) makes the API serve baseline predictions with `degraded: true`; pinning the `champion` alias to an older version rolls back the model.

## 6 Implementation

### 6.1 High-level design

#### System context

```mermaid
flowchart LR
    vad["Vadimkin air-raid sirens dataset<br/>(GitHub, history 2022 to 2026-09)"]
    aiu["alerts.in.ua API<br/>(active alerts, last-month history)"]
    tg["Telegram @kpszsu<br/>(Ukrainian Air Force posts)"]
    sys["Kyiv Alert Forecaster<br/>ingests, trains, serves, monitors"]
    cli["API client<br/>(planning app later, Lab 5 agent)"]
    onc["On-call engineer (author)<br/>Grafana alert to Telegram / email"]

    vad -- "raw alert history (bootstrap CSV)" --> sys
    aiu -- "raw data: active alerts every 20 s" --> sys
    aiu -- "labels: finished alerts (monthly)" --> sys
    tg -- "raw data: posts every 2 min" --> sys
    sys -- "predictions: 12-h hourly forecast, duration P50/P90" --> cli
    sys -- "alerts: drift, staleness, failures" --> onc
    onc -. "promotion approval, override" .-> sys

    style sys fill:#0b3d91,color:#ffffff
```

#### Data flow on the seven layers

The three marks: **version point** (DVC snapshot of `raw/` after ingestion), **quality gate** (before the feature store), and the split between the **training path** (offline store) and the **serving path** (online store).

```mermaid
flowchart TB
    subgraph L1["1 Data ingestion"]
        i1["DAG ingest_alerts (monthly)<br/>dataset bootstrap + alerts.in.ua month_ago"]
        i2["DAG ingest_telegram (every 2 min)<br/>t.me/s/kpszsu"]
        i3["alerts-poller (every 20 s)<br/>alerts.in.ua active"]
    end
    raw[("MinIO raw/<br/>as fetched")]
    vp(("version<br/>point<br/>DVC"))
    subgraph L2["2 Preprocessing"]
        p1["alerts: dedup, UTC, raion to oblast,<br/>end after start, drop naive ends"]
        p2["Telegram: threat type from emoji/keywords,<br/>morning-summary counts"]
    end
    qg(("quality<br/>gate"))
    subgraph L3["3 Feature store: features.py, one code path"]
        off["offline: Parquet in MinIO features/<br/>as of issue time / alert start"]
        on["online: PostgreSQL features_online<br/>alert map, last-30-min threat, rolling rates"]
    end
    subgraph L4["4 Training pipeline"]
        t1["DAG train_monthly<br/>expanding-window folds,<br/>baselines + GBM + quantile GBM"]
    end
    subgraph L5["5 Registry and deploy"]
        r1["MLflow registry<br/>aliases champion / challenger<br/>promotion rule 5.4 + manual approval"]
    end
    subgraph L6["6 Inference"]
        b1["batch: DAG forecast_12h<br/>10:00 and 22:00 to PostgreSQL"]
        rt["real-time: FastAPI predict/duration<br/>called by poller at Kyiv alert start"]
    end
    cons["Consumer: API client<br/>GET /v1/forecast/latest, /v1/alert/current"]
    subgraph L7["7 Monitoring and retraining loop"]
        m1["Evidently daily: input drift (oblasts under alert,<br/>threat mix), prediction drift; performance when labels<br/>arrive; Prometheus + Grafana; alert to on-call"]
    end

    i1 --> raw
    i2 --> raw
    i3 --> raw
    raw --> vp --> p1
    vp --> p2
    p1 --> qg
    p2 --> qg
    qg --> off
    qg --> on
    off -- "training path (offline store)" --> t1
    t1 -- "evaluate, register" --> r1
    on -- "serving path (online store)" --> b1
    on -- "serving path (online store)" --> rt
    r1 -- "load by alias champion" --> b1
    r1 -- "load by alias champion" --> rt
    b1 --> cons
    rt --> cons
    b1 -- "inputs, predictions, labels" --> m1
    rt -- "inputs, predictions, labels" --> m1
    m1 -- "trigger retraining" --> t1
```

Layer notes:

- **Real-time path:** the poller writes raw events and calls the same preprocessing and feature code before updating the online store, so one module defines every feature (no separate streaming stack; see 7.1).
- **Quality gate checks:** no duplicate alerts after dedup, every end after its start, newest event < 2 days old, row count within ±50% of the trailing 3-month median, both oblast- and raion-level rows mapped to an oblast, Telegram not silent for > 6 h. A failed check stops the run and alerts the on-call engineer.
- **Feature store layer** is kept, but implemented without Feast (see 7.1): offline side = Parquet in MinIO, online side = PostgreSQL.

#### Deployment, first draft

Local components, each with its managed cloud equivalent (AWS / GCP / Azure). Lab 4 replaces this with the real cluster topology.

```mermaid
flowchart TB
    subgraph laptop["Laptop / Minikube 1.33.1 (Kubernetes 1.30.0, docker driver, 4 GB) — first draft, replaced in Lab 4"]
        minio["MinIO-compatible object store<br/>pgsty/silo<br/><i>S3 / Cloud Storage / Blob Storage</i>"]
        dvc["DVC 3.67.1 remote on MinIO<br/><i>S3 / GCS / Azure Blob remote</i>"]
        af["Apache Airflow 3.3.1<br/><i>MWAA / Cloud Composer / Data Factory</i>"]
        pg["PostgreSQL 16<br/><i>RDS / Cloud SQL / Azure DB for PostgreSQL</i>"]
        mlf["MLflow 3.16.1<br/><i>SageMaker MLflow / Vertex AI Experiments + Model Registry / Azure ML with MLflow</i>"]
        reg["Minikube registry<br/><i>ECR / Artifact Registry / ACR</i>"]
        api["kyiv-alerts-api<br/>FastAPI + models<br/><i>EKS / GKE / AKS</i>"]
        poll["alerts-poller<br/><i>EKS / GKE / AKS</i>"]
        mon["Evidently + Prometheus + Grafana<br/><i>SageMaker Model Monitor + CloudWatch /<br/>Vertex AI Model Monitoring + Cloud Monitoring /<br/>Azure ML monitoring + Azure Monitor</i>"]
    end
    af --> minio
    af --> pg
    af --> mlf
    dvc --> minio
    mlf --> pg
    mlf --> minio
    reg --> api
    reg --> poll
    poll --> api
    api --> pg
    mon --> pg
    mon --> api
```

### 6.2 Infra

Local stack (pinned as in the course reference table) and the managed cloud service per component. Naming the cloud services is required; running on them is not.

| Component | Local (pinned) | AWS | GCP | Azure |
|---|---|---|---|---|
| Object storage, data lake | MinIO-compatible `pgsty/silo:RELEASE.2026-09-16T00-00-00Z` | Amazon S3 | Cloud Storage | Blob Storage |
| Data versioning remote | DVC 3.67.1 + dvc-s3 3.3.0, remote on MinIO | S3 remote | GCS remote | Azure Blob remote |
| Warehouse, serving database | PostgreSQL 16 | Amazon RDS | Cloud SQL | Azure Database for PostgreSQL |
| Orchestrator | Apache Airflow 3.3.1, LocalExecutor | Amazon MWAA | Cloud Composer | Azure Data Factory |
| Experiment tracking, model registry | MLflow 3.16.1, PostgreSQL backend, MinIO artifacts | SageMaker with MLflow | Vertex AI Experiments + Model Registry | Azure ML with MLflow |
| Feature store | Parquet (offline) + PostgreSQL (online), shared feature code; Feast 0.66.0 not used (7.1) | SageMaker Feature Store | Vertex AI Feature Store | Azure ML managed feature store |
| Container registry | Minikube built-in registry | Amazon ECR | Artifact Registry | Azure Container Registry |
| Kubernetes cluster (API, poller) | Minikube 1.33.1 (Kubernetes 1.30.0) | Amazon EKS | GKE | AKS |
| Model and data monitoring | Evidently | SageMaker Model Monitor | Vertex AI Model Monitoring | Azure ML model monitoring |
| Metrics store, dashboards | Prometheus, Grafana | Amazon CloudWatch | Cloud Monitoring | Azure Monitor |

Every component in this table appears in the diagrams of 6.1, and every component in the diagrams appears here. The Lab 3 components (vector store: pgvector 0.8.7; LLM inference: **Assumption** a hosted API, since there is no local GPU) are added to both the table and the diagrams in Lab 3.

### 6.3 Performance

- **Batch vs online.** M1 is **batch**: the answer only changes twice a day, so forecasts are precomputed at 10:00 and 22:00 and the API reads a row (cheap, cacheable, survives a model-service outage). M2 is **online, event-triggered**: it depends on the alert map at the exact start time, so it cannot be precomputed; the poller detects the start and calls the model once.
- **Budgets:** M1 DAG run ≤ 5 min end to end (feature query ≤ 1 min, scoring 12 rows ≤ 1 s, write ≤ 1 s). M2 ≤ 60 s p95 from alert start = ≤ 20 s poll interval + ≤ 5 s feature update + ≤ 1 s inference + margin for the alerts.in.ua feed delay (**Assumption:** feed lag ≤ 30 s; measured in Lab 4). API p95 ≤ 200 ms.
- **Throughput:** **Assumption** ≤ 10 requests/s; one FastAPI replica handles this. Training: ~30k hourly rows per year and ~500 alerts per year → minutes on a laptop CPU.

### 6.4 Security

- **Secrets:** the alerts.in.ua token is the only secret. Locally: `.env` (git-ignored, `.env.example` committed without values). Minikube: Kubernetes Secret. Cloud: AWS Secrets Manager / Secret Manager / Key Vault. Telegram scraping uses the public preview and needs no credentials; if we switch to Telethon, its session file becomes a secret too.
- **Who may call the service:** read-only `GET` endpoints are open on localhost / inside the cluster. In the cloud variant they require an API key and rate limiting (**Assumption:** 60 requests/min per key). `POST /v1/predict/duration` is reachable only from the poller (Kubernetes NetworkPolicy).
- **Upstream terms:** alerts.in.ua requires public services to proxy its data and never expose the token to clients; our API is that proxy. Request rate stays under 8/min.

### 6.5 Data privacy

- The data contains **no personal data**: alerts are public, region-level events; Telegram data is posts of an official public channel (no comments, no user accounts). View counts are aggregate.
- **Retention:** raw and processed data and models are kept for the duration of the course plus one year (**Assumption:** until 2027-12). API access logs: 30 days, no request bodies.
- **Sensitive use:** the system publishes only probabilities and durations for Kyiv; it stores and outputs nothing about air-defence positions or strike locations.

### 6.6 Monitoring and alarms

First draft; Lab 4 makes it real. Alerts go to the on-call engineer (author) through Grafana alerting to a Telegram bot and email.

| What | Metric | Threshold → action |
|---|---|---|
| Source freshness | age of the newest event per source | alerts.in.ua poll failing > 2 min → page; newest finished alert in the clean table > 2 days old → page (catches a job that re-reads the same partition); Telegram no new posts > 6 h → warn |
| Batch volume | rows per monthly batch vs trailing 3-month median | outside ±50% → quality gate fails the run |
| Pipeline health | DAG run status, duration | any failed run → page; M1 not published by issue time + 5 min → page |
| Input drift | PSI of "oblasts under alert" and of Telegram threat-type mix, 7 days vs training window | PSI > 0.2 → warn and attach Evidently report |
| Prediction drift | mean predicted hourly probability, mean predicted P90 (7-day vs 30-day) | change > 50% → warn |
| M1 performance (labels ≤ 12 h) | 30-day rolling Brier vs hour-of-day baseline | worse than baseline 14 days in a row → retrain alert |
| M2 performance (labels at alert end) | P90 coverage over the last 30 alerts | < 80% → retrain alert |
| Training–serving skew | distribution of "oblasts under alert" from live snapshots vs reconstructed from history for the same hours | mean difference > 1 oblast → warn |
| Service | API p95 latency, 5xx rate | p95 > 200 ms or 5xx > 1% over 15 min → page |

### 6.7 Cost

Estimated monthly cost of the cloud column on AWS (eu-central-1), even though the project runs locally. **Assumption:** rough list-price estimates, to be confirmed with the [AWS Pricing Calculator](https://calculator.aws/) before the defense.

| Component | Sizing assumption | Estimate, USD/month |
|---|---|---|
| Amazon S3 | ≤ 10 GB (raw, Parquet, DVC cache, MLflow artifacts), low request volume | < 1 |
| Amazon RDS for PostgreSQL | db.t4g.micro, single-AZ, 20 GB gp3 | ~15–20 |
| Amazon EKS | control plane + 2 small nodes (t3.medium) for API, poller, Airflow and MLflow self-hosted | ~73 + ~60 |
| Amazon ECR | < 1 GB of images | < 1 |
| Amazon CloudWatch | basic metrics, logs ≤ 5 GB, a few alarms | ~5–10 |
| **Self-hosted on EKS total** | | **~150–170** |
| Amazon MWAA (instead of Airflow on EKS) | smallest environment class | + ~200–400 |
| SageMaker with MLflow (instead of MLflow on EKS) | smallest tracking server | + ~100–150 |
| SageMaker Model Monitor (instead of Evidently) | daily small jobs | + ~10–20 |
| **Fully managed total** | | **~450–750** |

The orchestrator and the tracking server dominate the bill, not compute for the model. The design therefore assumes the self-hosted variant (under the $200 ceiling in 4.2).

### 6.8 Integration points

| Party | Direction | Contract |
|---|---|---|
| Vadimkin dataset (volunteer maintainer, MIT) | upstream | Best effort, no SLA; CSV columns `oblast, raion, hromada, level, started_at, finished_at, source`, UTC. We de-duplicate, check freshness, and use it only for bootstrap. |
| alerts.in.ua | upstream | Personal token; ≤ 8–10 requests/min (hard 12/min), history 2/min; public services must proxy; no use for critical infrastructure; availability not guaranteed. On schema change the quality gate fails. |
| Ukrainian Air Force Telegram channel | upstream | Public channel, no API contract; post format (emoji + type + location) can change at any time → keyword/emoji parser is versioned and the share of unparsed posts is monitored. |
| API clients (planning app, Lab 5 agent) | downstream | Versioned JSON API `/v1`. Every response carries `model_version`, `issued_at` / `alert_started_at`, `degraded`, and a disclaimer field. Breaking changes only in `/v2`. Probabilities are calibrated per hour; durations are minutes. |

### 6.9 Risks and uncertainties

| Risk | Mitigation |
|---|---|
| Upstream goes stale or disappears (official dataset already stopped on 2026-09-07) | Freshness alarm; volunteer dataset and alerts.in.ua history as secondary sources |
| Upstream schema or semantics change (Nov 2025 raion switch) | Quality gate on levels and columns; raion-aware aggregation; drift monitoring on "oblasts under alert" |
| Telegram blocks scraping or changes page markup | Parser smoke test in the quality gate; Telethon with a personal account as fallback |
| Leakage from Telegram (edited posts, posts written after the event) | Strict `datetime_utc < t` filter; compare history-trained vs live-collected features in Lab 4 |
| Concept drift: attack tactics change faster than monthly retraining | Daily drift and performance monitoring; on-demand retraining; baseline fallback |
| Too little data for M2 (≈ 500 alerts per year) | Simple, regularised quantile models; pool years with a drift-aware weight; report intervals, not only point scores |
| Over-reliance or misuse of forecasts as safety advice | Planning-only framing, disclaimer in every response, no shelter advice, nothing about air-defence or strike locations |
| alerts.in.ua token blocked for exceeding limits | Client-side rate limiter at 8 requests/min; `If-Modified-Since` caching |

## 7 Appendix

### 7.1 Alternatives

| Alternative | Pros | Cons | Decision and reason |
|---|---|---|---|
| **Weather features** (Open-Meteo Previous Runs API, forecast made 24 h earlier, from 2024-01-19) | Free, no key; the 24-h-earlier forecast matches serving exactly | Tested at Kyiv (night window) and at the Tsymbulova launch area (Oryol oblast, 15:00–22:00): univariate AUC 0.46–0.55 in every year; hold-out Brier worse than climatology (0.200–0.208 vs 0.198); no gain when added to history | **Rejected with evidence**: launches are scheduled by the attacker, ordinary weather variation does not predict them |
| **One binary label per night** | Simplest model and metric | Arbitrary 60-min threshold, no timing, needs a separate day model | **Rejected**: hourly framing (5.1) |
| **Feast** as the feature store | Standard offline/online store, point-in-time joins | ~15 features from two tables; adds a registry, a materialisation job and another service to run on 4 GB | **Rejected for now**: shared feature code over Parquet + PostgreSQL gives the same offline/online split; revisit if features multiply |
| **Kafka + Spark streaming** for live alerts | True streaming, scalable | ~1.4 Kyiv alerts/day, a 20-s poll is enough; two heavy services for no gain | **Rejected**: polling service |
| **Telethon (Telegram API) instead of the web preview** | Exact edit history, faster back-fill | Needs a personal account, phone login and a secret session file | **Web preview chosen**; Telethon kept as fallback |
| **Historical Weather API (reanalysis)** for weather | Longest history (since 1940) | Observed-after-the-fact weather → training–serving skew | Not used (and weather rejected overall) |

### 7.2 Data pipelines, versioning and baseline

*Placeholder — filled in Lab 2.*

### 7.3 Serving, monitoring and drift

*Placeholder — filled in Lab 4.*

### 7.4 Milestones

- **Lab 2:** ingestion DAGs for alerts and Telegram into MinIO with DVC versioning, the quality gate, and the baselines and first models for M1 and M2 tracked in MLflow on time-based folds.
- **Lab 3:** an LLM component that answers "what is the current threat to Kyiv and why does the forecast say so" by retrieving Air Force posts and morning summaries from pgvector, evaluated on a fixed question set with retrieval and answer metrics.
- **Lab 4:** the API and the poller containerised and deployed on Minikube with health probes, Evidently/Prometheus/Grafana monitoring, and a replayed drift (the November 2025 raion switch) that reaches the on-call engineer.
- **Lab 5:** an agent with tools for active alerts, the 12-hour forecast, the duration prediction and post search that answers planning questions, with persisted traces and a ten-task evaluation.
