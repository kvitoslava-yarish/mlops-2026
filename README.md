# Kyiv air-alert forecasting

Course project for KSE "Introduction to MLOps and ML System Design" (2026). The full design is in [docs/design.md](docs/design.md).

**Status:** Lab 2. Raw data versioned with DVC on MinIO, a monthly Airflow ETL with a quality gate, feature tables, rule baselines vs trained models on a time split, MLflow tracking and registry with lineage, one-command reproduction, and an optional Feast demo. Runs locally with Docker Compose; built and tested on Apple Silicon.

## Problem

Kyiv alerts are known only when the siren starts, with no estimate of when the next one comes or how long it lasts.
Twice a day (10:00 and 22:00) the system forecasts, for each of the next 12 hours, the probability that a Kyiv alert starts and the likely threat type.
When a Kyiv alert starts, it predicts how long the alert will last.
It learns from the public alert history of all Ukraine since 2022 and from the Ukrainian Air Force Telegram channel, retrains monthly and checks daily that the data has not changed.
Predictions are served by a REST API for planning only; the official alert always takes precedence.

## What the system predicts

| Model | Question | When it runs | Output | When we learn the truth |
|---|---|---|---|---|
| **M1** start risk | In which of the next 12 hours is a Kyiv alert likely to **start**? | 10:00 (for 10–22) and 22:00 (for 22–10) | one probability per hour | at the end of each hour |
| **M2** duration | This alert just started: when will it end? | the moment a Kyiv alert starts | **P50** "typically ends by" and **P90** "almost surely ends by" | when the alert ends |
| **M3** threat type | If an alert starts in that hour, what is it likely to be? | together with M1 | per hour: probability of drones, cruise missiles, ballistic | when an Air Force post names the type (Lab 3) |

Lab 2 builds M1 and M2 from alert data. M3 and the Telegram features need the Air Force posts, which arrive in Lab 3 together with the retrieval corpus.

**Why "starts in the hour".** 62% of Kyiv alerts cross an hour boundary. With "starts in the hour" every alert belongs to exactly one hour, so short alerts are never lost and long ones are never counted twice.

## How predictions are scored

Targets come from the Lab 1 analysis ([design.md §3](docs/design.md#3-success-metrics)); Lab 2 hold-out results are in [§5](#5-baseline-vs-model-time-split).

- **M1:** Brier score (main; how close the hourly probabilities are to what happened), AUC (do start hours get higher probabilities; 0.5 = random), F1 with ±1 h tolerance (flagging "high-risk" hours: an alert at 23:58 counts as caught by a 00:00 flag).
- **M2:** pinball loss for P50 and P90 (main: P90), share of alerts with P50 within ±15 min, **P90 kept** (share of alerts that ended before P90; target 85–95%), P90 width (median P90, shorter is more useful).
- **M3** (Lab 3): macro-F1 and log-loss against "always the most common type".

### Formulas

Brier score, over $N$ hours, where $p_i$ is the predicted probability that an alert starts in hour $i$ and $y_i$ is 1 if one did, 0 if not:

```math
\text{Brier} = \frac{1}{N}\sum_{i=1}^{N} (p_i - y_i)^2
```

Example: we said $p = 0.12$ and an alert started, so that hour scores $(0.12 - 1)^2 = 0.77$; a quiet hour with $p = 0.05$ scores $0.05^2 = 0.0025$.

Pinball loss for quantile $q$ ($q = 0.5$ for P50, $0.9$ for P90), real duration $y$ and prediction $\hat{y}$, averaged over alerts:

```math
L_q(y, \hat{y}) =
\begin{cases}
q\,(y - \hat{y}) & y \ge \hat{y} \quad \text{(alert lasted longer than predicted)} \\
(1 - q)\,(\hat{y} - y) & y < \hat{y} \quad \text{(alert ended earlier)}
\end{cases}
```

For P50 this is half the absolute error in minutes. For P90 = 240 min: an alert that ends after 90 min costs $0.1 \cdot 150 = 15$; one that lasts 300 min costs $0.9 \cdot 60 = 54$. Running past P90 costs 9× more per minute, so the score is best when about 1 alert in 10 outlasts P90.

## Lab 2 at a glance

```mermaid
flowchart LR
    gh["Vadimkin dataset<br/>(pinned commits)"] -- "make data / data-seed" --> dvc[("DVC remote<br/>MinIO bucket dvc")]
    dvc --> dag["Airflow DAG alerts_etl<br/>extract → validate → transform →<br/>load → build_features → quality_check"]
    dag -- "one month per run,<br/>delete + insert" --> pg[("PostgreSQL warehouse<br/>dwh.alerts · features.*")]
    pg -- "make snapshot" --> snap[("data/features.dvc")]
    snap -- "make train" --> mlf["MLflow server<br/>runs + registry (champion)"]
    mlf -- "make reproduce" --> mlf
```

Design details: [design.md §5.2–5.4, §6.1](docs/design.md#52-data).

## Quick start (clean clone)

Requirements: [uv](https://docs.astral.sh/uv/), Docker Desktop with **≥ 6 GB memory** (Settings → Resources), free ports 8080, 5001, 5433, 9000, 9001.

```bash
make setup     # Python 3.12 env from uv.lock (includes DVC 3.67.1 with dvc-s3 3.3.0)
make up        # copies .env.example to .env, builds and starts the stack, waits until healthy
make data      # dvc pull of the raw data + checksums  (on a new machine: make data-seed first)
make backfill  # alerts_etl for every month 2022-03 … 2026-08, oldest first (~7 min)
make snapshot  # export passed batches to data/features, dvc add + push
git add data/features.dvc && git commit -m "feature snapshot"
make train     # M1 + M2: baselines vs models, logged and registered in MLflow
make reproduce MODEL=kyiv-m1-start-risk ALIAS=champion
```

| UI | URL | Login |
|---|---|---|
| Airflow | http://localhost:8080 | none (local admin mode) |
| MLflow | http://localhost:5001 | none |
| MinIO console | http://localhost:9001 | `minioadmin` / `minioadmin123` (from `.env`) |

Every target in the [Makefile](Makefile) has a one-line comment. `make down` stops the stack and keeps its data; `make nuke` deletes all volumes.

### Notes for Apple Silicon

- Every image in [docker-compose.yml](docker-compose.yml) has an arm64 build; nothing runs under emulation.
- MLflow is on port **5001** because macOS AirPlay Receiver holds 5000. PostgreSQL is on **5433** so a Homebrew Postgres on 5432 does not clash.
- Models are scikit-learn `HistGradientBoosting*`; LightGBM / XGBoost are avoided because they need `libomp` from Homebrew.
- **Exit code 132** (illegal instruction in `cryptography`, older Docker Desktop): update Docker Desktop first. If that is not possible, uncomment the `cryptography==46.0.5` line in [docker/airflow/Dockerfile](docker/airflow/Dockerfile) and [docker/mlflow/Dockerfile](docker/mlflow/Dockerfile), then `make up`. This build did not need it (Docker Desktop with engine 29.2.1).
- `MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false` is set in `.env.example`, so `mlflow.*.load_model` on the host downloads through the tracking server instead of the unreachable `minio:9000`.
- All ports are bound to 127.0.0.1.

## 1. Data versioning

| | |
|---|---|
| Tool | DVC 3.67.1 + dvc-s3 3.3.0 |
| Remote | S3-compatible MinIO (`pgsty/silo`) in Docker Compose, bucket `dvc`: `s3://dvc`, endpoint `http://localhost:9000` ([.dvc/config](.dvc/config), committed). Credentials come from `.env` via the Makefile, never from git |
| Tracked | `data/raw/` → [data/raw.dvc](data/raw.dvc) (hash `352f88a87031f9e22af2cce021c2a4a7.dir`); training snapshot `data/features/` → `data/features.dvc` |
| Size | 40 MB in total, below the 200 MB limit, so **no sampling**: the files are the complete upstream files |

**Retrieve the data on a clean clone:**

```bash
make up      # MinIO must be running
make data    # = uv run dvc pull data/raw.dvc  &&  checksums
```

What arrives (md5, bytes):

```
e705cfc6d7054b8b09999be31c7c423c    31025633  data/raw/official_data_en.csv
938e6c53bd8f953198e1eafba1654822     9103166  data/raw/volunteer_data_en.csv
```

**New machine, empty MinIO:** `make data-seed` downloads both files from GitHub at their pinned commits (`58ee75b` official, `346bcff` volunteer), re-runs `dvc add` and fails unless the result is byte-identical to the committed `data/raw.dvc`, then `dvc push`es. After that `make data` works as above.

## 2. ETL pipeline: `alerts_etl`

[dags/alerts_etl.py](dags/alerts_etl.py) (thin) calls [src/kyiv_alerts/pipeline.py](src/kyiv_alerts/pipeline.py):

| Task | Does | Writes |
|---|---|---|
| `extract` | picks the source that covers the whole month (official file → alerts.in.ua API if a token is set) and stores the month's raw rows once; a rerun reuses that immutable copy | MinIO `lake/raw/alerts/YYYY-MM/<source>.csv` |
| `validate` | columns, timestamp parsing, known oblasts, every row inside the month, unfinished ≤ 1% | — (fails on a broken schema) |
| `transform` | exact duplicates, unfinished, end ≤ start; raion / hromada records merged into oblast episodes; counts per rule | MinIO `lake/clean/alerts/YYYY-MM.parquet` |
| `load` | **delete-then-insert of the month in one transaction**, plus the audit row | `dwh.alerts`, `dwh.batch_audit` |
| `build_features` | M1 and M2 feature rows for the month, as of issue time / alert start, using a 60-day lookback | `features.m1_hourly`, `features.m2_alert` |
| `quality_check` | the gate (§3 below) | `dwh.batch_audit.gate_passed` |

**The batch is the data interval, never the clock.** I chose the explicit timetable: `schedule=CronDataIntervalTimetable("@monthly", timezone="UTC")`, so each run has a real interval `[month start, next month start)` and the batch is the month of `data_interval_start`. A manual run can name the month with the `batch_month` param instead (`make run MONTH=2025-06` passes `{"batch_month": "2025-06"}`). No task reads `datetime.now()`.

```bash
make run MONTH=2025-06            # one month, waits for the result
make row-counts MONTH=2025-06     # rows for that month in each table + min/max start
make backfill FROM=2022-03 TO=2026-08
```

Triggering by logical date also works, with one Airflow subtlety: a *manual* run gets the interval that **ends** at its logical date, so `airflow dags trigger alerts_etl --logical-date 2025-07-01T00:00:00+00:00` processes June 2025 (scheduled and backfill runs are named by the interval start, e.g. `backfill__2025-07-01…` = logical date 2025-06-01 = June). The `batch_month` param avoids the ambiguity; in the UI use *Trigger* → *batch_month*.

**Only that interval:** after `make run MONTH=2025-06`, every row of the month has `started_at` between `2025-06-01` and `2025-06-30`:

```
$ make row-counts MONTH=2025-06
{
  "batch_month": "2025-06",
  "dwh.alerts": 1440,
  "features.m1_hourly": 720,
  "features.m2_alert": 34,
  "alerts_min_started_at": "2025-06-01 03:16:05+00:00",
  "alerts_max_started_at": "2025-06-30 22:03:45+00:00"
}
```

### Idempotency: the same interval run twice

The statement that makes a rerun harmless ([warehouse.py](src/kyiv_alerts/warehouse.py), `replace_batch`), executed for `dwh.alerts`, `features.m1_hourly` and `features.m2_alert` inside one transaction:

```sql
DELETE FROM dwh.alerts WHERE batch_month = %s;
INSERT INTO dwh.alerts (...) VALUES ...;
```

`make run MONTH=2025-06` twice, then `make row-counts MONTH=2025-06` after each run:

```
run 1: manual__2025-06__20261024T205222  success
  dwh.alerts 1440 · features.m1_hourly 720 · features.m2_alert 34
  started_at 2025-06-01 03:16:05+00:00 … 2025-06-30 22:03:45+00:00

run 2: manual__2025-06__20261024T205240  success
  dwh.alerts 1440 · features.m1_hourly 720 · features.m2_alert 34
  started_at 2025-06-01 03:16:05+00:00 … 2025-06-30 22:03:45+00:00
```

### Sources

- **Official file** (commit `58ee75b`): months 2022-03 … 2026-08, the only pinned source the pipeline uses. Its last two days are missing (no alerts after 2026-08-29 05:00 UTC); accepted, see §3.
- **Volunteer file** (commit `346bcff`): DVC-tracked for cross-checks only, not a pipeline source. It is oblast-level and splits alerts into ~1.3–1.8× more episodes than the official file, so mixing it in would change what the features mean.
- **alerts.in.ua** (`ALERTS_IN_UA_TOKEN` in `.env`): 2026-09 onwards. `month_ago` returns only the last 30 days, so a month has to be fetched within 30 days of its end; the JSON is kept in the lake as fetched and reused on rerun. `make api-smoke` checks the token with one request.
- **Live alerts without a token** ([Ubilling Aerial Alerts API](https://wiki.ubilling.net.ua/doku.php?id=aerialalertsapi), `?source=aiu&raw`): the alerts.in.ua active-alerts feed relayed by Ubilling, with exact `started_at`, raion level and red/yellow `alert_level`; limit 2 req/s. Only current alerts, no history, so it does not replace `month_ago` for back-fill; it is the Lab 4 poller's source (backup `source=ual`). The `default` source is not used: its timestamps are `1970-01-01`.

### Backfill result

`make backfill FROM=2022-03 TO=2026-08`: 54 monthly runs, ~7 minutes, **all 54 pass the gate** (2026-08 with its accepted gap, §3). `dwh.batch_audit` keeps the source, raw sha256, row counts per cleaning rule and the gate report of every month:

```sql
-- docker compose exec postgres psql -U warehouse -d warehouse
SELECT batch_month, source, rows_raw, rows_duplicate, rows_bad_time, rows_clean, m1_rows, m2_rows, gate_passed
FROM dwh.batch_audit ORDER BY batch_month;
```

Unpausing the DAG also makes Airflow schedule the latest complete interval (2026-09) right away. Until `ALERTS_IN_UA_TOKEN` is set it fails in `extract` ("no source covers 2026-09"), which is expected.

## 3. Quality gate

[src/kyiv_alerts/quality.py](src/kyiv_alerts/quality.py). Thresholds were set from the 2022-03 … 2026-07 official batches, where invalid records were ≤ 0.02%, batch size stayed within 0.61–1.40× of the trailing median and the newest alert was ≤ 4.3 h before month end.

| Check | Limit |
|---|---|
| invalid records (unfinished or end ≤ start) after de-duplication | ≤ 1% |
| clean rows vs median of the last 3 passed batches | within ±50% |
| duplicate (oblast, start) in the warehouse | 0 |
| hours between the newest alert and month end | ≤ 48 (2026-08: ≤ 72, accepted gap) |
| Kyiv alerts in the month | ≥ 1 |
| previous month loaded (feature lookback) | > 0 rows |
| M1 rows = hours in the month; M2 rows = Kyiv alerts | equal |
| feature nulls; single `feature_version` | 0; `v2` |

A failed batch fails the run, is marked in `dwh.batch_audit`, and is left out of the training snapshot.

**Tested on purpose.** `make bad-batch` writes [scripts/make_bad_batch.py](scripts/make_bad_batch.py)'s broken copy of June 2025: the schema is intact (so `validate` passes), but 30% of rows (seed 7) get an end 5 minutes before their start. It is fed to the DAG through the `raw_override` param, `transform` drops those rows, and `quality_check` fails:

```
$ make bad-batch
data/fixtures/bad_batch_2025-06.csv: 4874 rows, 1500 with end before start
triggered manual__2025-06__20261024T205309  conf={"batch_month": "2025-06", "raw_override": "/opt/airflow/data/fixtures/bad_batch_2025-06.csv"}
manual__2025-06__20261024T205309: failed

extract success · validate success · transform success · load success · build_features success · quality_check FAILED

transform: {'rows_raw': 4874, 'rows_duplicate': 1389, 'rows_unfinished': 0, 'rows_bad_time': 1274, 'rows_valid': 2211, 'rows_clean': 1346, 'invalid_share': 0.3656}
quality_check log:
  FAIL invalid_share: 0.3656 (limit <= 0.01)
  PASS rows_vs_3m_median: 0.996 (limit 1 ± 0.5)
  PASS duplicate_episodes: 0 (limit == 0)
  PASS hours_since_newest_alert: 1.9 (limit <= 48)
  ... (6 more PASS)
  AirflowFailException: batch 2025-06 failed: invalid_share=0.3656 (limit <= 0.01)
```

`make run MONTH=2025-06` restores the month afterwards.

| Failed gate | Successful run |
|---|---|
| ![quality_check failed](docs/screenshots/gate_failed.png) | ![successful run graph](docs/screenshots/run_success.png) |

**Caught for real.** The first backfill also stopped two real batches, which led to two decisions:

| Batch | Failed check | Why | Decision |
|---|---|---|---|
| 2026-08 | `hours_since_newest_alert` = 67.1 (limit 48) | the official file has no alerts after 2026-08-29 05:00 UTC, while the volunteer file has ~170 a day: an upstream gap, the "stale extract" case | accepted: `ACCEPTED_GAPS` in [quality.py](src/kyiv_alerts/quality.py) sets 72 h for this month only and the gate report records the reason; the hold-out stops on 2026-08-28 so the missing hours are not scored as quiet |
| 2026-09 | `rows_vs_3m_median` = 2.20 (limit 1 ± 0.5) | the only source was the volunteer file, which splits alerts into ~2× more oblast episodes: a source switch | rejected: the volunteer file was removed as a pipeline source and the batch deleted; 2026-09 and 2026-10 come from alerts.in.ua once the token arrives |

Note the bad batch passed the size check (0.996), because merged episodes hide dropped records. That is why the gate checks the invalid share separately.

## 4. Feature tables

`features.m1_hourly` (one row per hour) and `features.m2_alert` (one row per Kyiv alert), each with `feature_version = v2` and the label alongside. Features about other regions count only Kyiv's six neighbour oblasts (Kyivska, Zhytomyrska, Chernihivska, Cherkaska, Poltavska, Vinnytska). The serving-time source of every feature is in [design.md §5.3](docs/design.md#53-techniques). All features come from one module, [src/kyiv_alerts/features.py](src/kyiv_alerts/features.py), that Lab 4's service will reuse on live state.

**Optional: Feast point-in-time demo.** [feature_repo/](feature_repo/) defines entity `city` and feature view `kyiv_state` (the issue-time M1 inputs, one row per 10:00 / 22:00 issue) over the exported snapshot. `make feast-demo` asks for the state at a time between two rows and checks that Feast returns the earlier one:

```
$ make feast-demo
feature row A at 2026-07-22 10:00:00+03:00: {'days_since_long_alert': 16.20, 'kyiv_alert_min_prev_12h': 5.4, 'kyiv_under_alert_at_issue': 0, 'neighbours_under_alert_at_issue': 1, 'neighbour_starts_prev_12h': 12}
feature row B at 2026-07-22 22:00:00+03:00: {'days_since_long_alert': 16.70, 'kyiv_alert_min_prev_12h': 22.2, 'kyiv_under_alert_at_issue': 0, 'neighbours_under_alert_at_issue': 2, 'neighbour_starts_prev_12h': 12}
requested at     2026-07-22 16:00:00+03:00
Feast returned   {'days_since_long_alert': 16.20, 'kyiv_alert_min_prev_12h': 5.4, 'kyiv_under_alert_at_issue': 0.0, 'neighbours_under_alert_at_issue': 1.0, 'neighbour_starts_prev_12h': 12.0}
point-in-time correct (returned the earlier row A): True
```

## 5. Baseline vs model (time split)

Train 2023-01-01 … 2026-02-28, hold-out 2026-03-01 … 2026-08-28 (the official file stops on 2026-08-29). Why a time split and these dates: [design.md §5.4](docs/design.md#54-experimentation-and-validation). Lower is better for Brier, log-loss and pinball.

Training data version `data_version = 183e372e0f43433c3e3e8c35624f1d44.dir` (feature_version v2; batches 2022-03 … 2026-08). Registered as `kyiv-m1-start-risk` v3 and `kyiv-m2-duration` v3 (alias `champion`) from commit `84a7cf5`.

**M1 — start risk**, 4,344 hold-out hours, an alert started in 7.3% of them. Train: 27,720 hours.

| | Brier ↓ (main) | AUC ↑ | log-loss ↓ | F1 ±1 h ↑ |
|---|---|---|---|---|
| Trivial: start rate by hour of day | 0.06748 | 0.568 | 0.2629 | 0.335 |
| Rule: same hour, last 30 days | 0.06817 | 0.549 | 0.3423 | 0.339 |
| Model: HistGradientBoostingClassifier | **0.06701** | **0.572** | **0.2595** | **0.345** |

The model beats both baselines on every metric, but by little: 0.0005 Brier (0.7%) below the hour-of-day profile, and AUC stays below the 0.60 target. With a hold-out that ended on 2026-07-31 the two were tied (0.05918 vs 0.05917), so the edge comes from August, the busiest month (90 Kyiv alerts). It is not yet a robust win. The 30-day rule is worse than the static profile on Brier. The Telegram features (Lab 3) are the bet for a real improvement.

**M2 — duration**, 325 hold-out Kyiv alerts with a known end. Train: 1,413 alerts.

| | pinball P50 ↓ | pinball P90 ↓ (main) | P50 within ±15 min ↑ | P90 kept (target 85–95%) | P90 width ↓ |
|---|---|---|---|---|---|
| Rule: last-12-months P50 / P90 (33 / 224 min) | 16.13 | 21.39 | 48.9% | 97.5% | 224 min |
| Model: quantile HistGradientBoosting | **15.68** | **15.00** | **49.8%** | **92.9%** | **136 min** |

The model beats the rule on every metric: P90 pinball is 30% lower, P90 shrinks from 224 to 136 min and still holds for 92.9% of alerts (inside the target band; the rule's 97.5% is too wide). With v1 features (all 24 oblasts) the model lost to the rule on P50; restricting the regional features to Kyiv's neighbours fixed that.

## 6. Experiment tracking and the registry

MLflow 3.16.1 at http://localhost:5001: backend store PostgreSQL (database `mlflow`), artifact store MinIO (bucket `mlflow`, served through the tracking server). Started by `make up`.

Every write to MLflow goes through one class, [`TrackedRun`](src/kyiv_alerts/tracking.py). It refuses to start a run without `data_version` and `git_commit`, records every param, metric, tag, dataset, artifact and model it writes, marks a crashed run `FAILED`, and ends each run by logging that record as **`logging_manifest.json`** and printing a one-line summary:

```
[mlflow] kyiv-m1-start-risk / m1-84a7cf5 (a5876500…) FINISHED: 6 tags · 13 params · 24 metrics · 2 datasets · 2 artifacts · model HistGradientBoostingClassifier · kyiv-m1-start-risk v3 @champion
```

Every `make train` run logs:

- **params:** split dates, seed, hyperparameters, feature list ([configs/train.yaml](configs/train.yaml));
- **datasets:** the train and hold-out frames as MLflow datasets (schema, rows, source file, digest = DVC hash), in the run's *Datasets* panel;
- **metrics:** every baseline and the model side by side (`holdout_<metric>_<trivial|rule|model>`);
- **artifacts:** `config/train.yaml`, `logging_manifest.json`, the model;
- **tags:** `data_version` (DVC hash of `data/features`), `raw_data_version` (DVC hash of `data/raw`), `git_commit`, `git_dirty`, `feature_version`.

Training refuses to run with uncommitted changes or with `data/features` out of sync with `data/features.dvc`, so the two lineage tags always identify what built the model. Each run registers a version of `kyiv-m1-start-risk` / `kyiv-m2-duration` with `data_version` and `git_commit` copied onto the version; the first version gets the alias **`champion`** (`make train CHAMPION=1` moves it), which Lab 4 loads as `models:/kyiv-m1-start-risk@champion`.

### Reproduce a registered run

```bash
make reproduce MODEL=kyiv-m1-start-risk ALIAS=champion   # or VERSION=3
make reproduce MODEL=kyiv-m2-duration ALIAS=champion
```

[src/kyiv_alerts/reproduce.py](src/kyiv_alerts/reproduce.py) reads `git_commit` and `data_version` from the model version, checks out that commit in a temporary `git worktree` (your working tree is untouched), verifies that its `data/features.dvc` has hash `data_version`, `dvc pull`s it from MinIO, runs that commit's training without registering (tagged `reproduces=<run id>`), and compares the main metric.

**On another machine** the snapshot is not in that machine's MinIO yet: run `make data-seed`, `make backfill` and `make snapshot` first. The export is deterministic (sorted rows, same lock file), so it reproduces the same `data/features.dvc` hash and pushes the data the registered version needs.

**Tolerance:** |Δ Brier| ≤ 1e-4 for M1, |Δ pinball@0.9| ≤ 0.1 min for M2. On the same machine and lock file the difference is 0 (fixed seed, early stopping off); the tolerance covers floating-point differences between CPUs and BLAS builds.

```
$ make reproduce MODEL=kyiv-m1-start-risk ALIAS=champion
kyiv-m1-start-risk v3: run a5876500d37b4f78a6223163f9117ce1, commit 84a7cf5431304916bbcfcd48b11c7e4cd0ef70b2, data 183e372e0f43433c3e3e8c35624f1d44.dir
$ git worktree add --detach /var/folders/.../kyiv-repro-…/repo 84a7cf5431304916bbcfcd48b11c7e4cd0ef70b2
$ dvc pull data/features.dvc
3 files fetched and 2 files added
$ uv run --frozen python -m kyiv_alerts.train --model m1 --no-register --reproduces a5876500d37b4f78a6223163f9117ce1 ...
[mlflow] kyiv-m1-start-risk / m1-84a7cf5 (ac3cdd78…) FINISHED: 7 tags · 13 params · 24 metrics · 2 datasets · 2 artifacts · model HistGradientBoostingClassifier · not registered
$ git worktree remove --force /var/folders/.../kyiv-repro-…/repo
holdout_brier_model: registered 0.067009, reproduced 0.067009, |diff| 0.00e+00 (tolerance 0.0001) -> PASS
reproduction run: ac3cdd7839b341d69f397ffc026b138f

$ make reproduce MODEL=kyiv-m2-duration ALIAS=champion
kyiv-m2-duration v3: run 16eeeb1720414d90a56f2cb117be5530, commit 84a7cf5431304916bbcfcd48b11c7e4cd0ef70b2, data 183e372e0f43433c3e3e8c35624f1d44.dir
...
holdout_pinball_90_model: registered 14.996139, reproduced 14.996139, |diff| 0.00e+00 (tolerance 0.1) -> PASS
reproduction run: 8bdf1893bf014f428a6b67a07f7ce6a3
```

## Repository

```
dags/alerts_etl.py            Airflow DAG (thin wrapper)
src/kyiv_alerts/              pipeline, sources, cleaning, features, gate, training (tracking.py: MLflow writes), reproduction
  sql/warehouse.sql           warehouse schema (dwh, features)
configs/train.yaml            split dates and hyperparameters
feature_repo/                 Feast offline demo (optional)
scripts/                      seeding, bad batch, Airflow run / backfill helpers
docker/                       Airflow and MLflow images, Postgres init
docker-compose.yml            local stack
data/raw.dvc, data/features.dvc   data versions (contents live in the DVC remote)
docs/design.md                design document
tests/                        unit tests (no Docker needed): make test
```

## Maturity scorecard

| Property | Evidence a reviewer checks | Added in |
|---|---|---|
| Repository reproducible from a clean clone | CI is green; `make setup` and `make lint` succeed on a fresh clone | Lab 1 |
| Design document exists and is current | [`docs/design.md`](docs/design.md) follows the template, every section filled or marked as an assumption | Lab 1 |
| Data versioned | `make data` retrieves the raw data on a clean clone with the checksums in [§1](#1-data-versioning) | Lab 2 |
| Pipeline idempotent | the same month run twice gives identical row counts, both pasted in [§2](#idempotency-the-same-interval-run-twice) | Lab 2 |
| Quality gate present | `make bad-batch` fails `quality_check` on purpose; output and screenshot in [§3](#3-quality-gate) | Lab 2 |
| Experiments tracked | every run in MLflow (http://localhost:5001) has params, metrics, `data_version` and `git_commit` ([§6](#6-experiment-tracking-and-the-registry)) | Lab 2 |
| Model registered with lineage | `kyiv-m1-start-risk@champion` / `kyiv-m2-duration@champion` tagged with data version and commit; `make reproduce` passes within the stated tolerance ([§6](#reproduce-a-registered-run)) | Lab 2 |
