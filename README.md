# Kyiv air-alert forecasting

Course project for KSE "Introduction to MLOps and ML System Design" (2026). The full design is in [docs/design.md](docs/design.md).

**Status:** Lab 1, design only. The repository has the environment, linting and CI; data pipelines and models arrive in Lab 2.

## Problem

Kyiv alerts are known only when the siren starts, with no estimate of when the next one comes or how long it lasts.
Twice a day (10:00 and 22:00) the system forecasts, for each of the next 12 hours, the probability that a Kyiv alert starts and the likely threat type.
When a Kyiv alert starts, it predicts how long the alert will last.
It learns from the public alert history of all Ukraine since 2022 and from the Ukrainian Air Force Telegram channel, retrains monthly and checks daily that the data has not changed.
Predictions are served by a REST API for planning only; the official alert always takes precedence.

## What the system predicts

Three models, numbered in the order they were designed:

| Model | Question | When it runs | Output | When we learn the truth |
|---|---|---|---|---|
| **M1** start risk | In which of the next 12 hours is a Kyiv alert likely to **start**? | 10:00 (for 10–22) and 22:00 (for 22–10) | one probability per hour | at the end of each hour |
| **M2** duration | This alert just started: when will it end? | the moment a Kyiv alert starts | **P50** "typically ends by" (half of such alerts end earlier) and **P90** "almost surely ends by" (9 of 10 end earlier) | when the alert ends |
| **M3** threat type | If an alert starts in that hour, what is it likely to be? | together with M1 | per hour: probability of drones, cruise missiles, ballistic | when an Air Force post names the type |

**Why "starts in the hour" and not "alert during the hour".** 62% of Kyiv alerts cross an hour boundary. With "starts in the hour" every alert belongs to exactly one hour, so short alerts are never lost and long ones are never counted twice. How long an hour stays disrupted is M2's job.

**Example: the 22:00 forecast.** The numbers below are what the simplest baseline predicts today (the share of hours in 2025–26 in which an alert started); a trained model moves each hour up or down from these. The type row is illustrative until M3 is trained.

| Hour | 22 | 23 | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P(alert starts) | 9.3% | 10.1% | 11.9% | 10.8% | 6.4% | 8.2% | 6.2% | 4.7% | 5.1% | 4.4% | 4.2% | 6.7% |
| Risk | 🔴 | 🔴 | 🔴 | 🔴 | 🟡 | 🟡 | 🟡 | 🟢 | 🟡 | 🟢 | 🟢 | 🟡 |
| Likely type | drones | drones | drones | drones | drones | drones | mixed | mixed | ballistic | ballistic | ballistic | ballistic |

An average hour has a 6.4% chance that an alert starts, so the risk bands are 🟢 below 5%, 🟡 5–9%, 🔴 9% and above (about 1.5× average). Over the whole night the baseline gives a 57% chance that at least one alert starts.

**Example: duration.** An alert starts at 23:41, right after an Air Force post about drones. M2 returns P50 = 95 min and P90 = 240 min: "typically over by 01:16, almost surely by 03:41".

**API:** `GET /v1/forecast/latest` (M1 + M3), `GET /v1/alert/current` (current alert + M2), `GET /v1/health`.

## How it works

1. **Ingest.** Airflow jobs collect alert history (GitHub dataset, then monthly from the alerts.in.ua API) and Air Force posts (every 2 min). Raw files go to MinIO and are versioned with DVC.
2. **Prepare.** Alerts are de-duplicated and aggregated to oblast level; posts are tagged with a threat type. A quality gate stops the run on bad data.
3. **Train.** A monthly job trains the models on all data up to last month and logs every run to MLflow. A new model replaces the current one only if it beats it on the last 3 months and a person approves.
4. **Serve.** At 10:00 and 22:00 a job writes the M1 + M3 forecast to PostgreSQL. A poller checks alerts.in.ua every 20 s; when a Kyiv alert starts, it asks the API for an M2 prediction.
5. **Monitor.** Daily drift checks and accuracy checks once the truth is known; alarms go to the on-call engineer.

Diagrams (system context, data flow, deployment): [design.md §6.1](docs/design.md#61-high-level-design).

## How predictions are scored

Scores below are on a **hold-out period** the models never saw during training: 2026-03-01 … 2026-09-05. A **baseline** is the simplest rule a model has to beat. The **main** metric is the one we optimise. Targets marked *(assumption)* are our choice, to be revisited.

**M1** is scored on 4,536 hours; an alert started in 7.5% of them. Baseline: the average start rate for that hour of day.

| Metric | What it checks | Target | Baseline | First model |
|---|---|---|---|---|
| Brier score (main) | how close the hourly probabilities are to what happened; lower is better | below baseline | 0.0695 | 0.0687 |
| F1 with ±1 h tolerance | if we flag "high-risk" hours, how many real starts we catch and how many flags are right; an alert at 23:58 counts as caught by a 00:00 flag | ≥ 0.30 *(assumption)* | 0.25 | 0.27 |
| AUC | whether hours with a start get higher probabilities than quiet hours; 0.5 = random | ≥ 0.60 | 0.57 | 0.58 |

The first model uses alert history only and barely beats the baseline: predicting *when* an alert starts is hard. Telegram features are the main bet to improve it.

**M2** is scored on 349 alerts. Baseline: always predict the last 12 months' median and 90th percentile, i.e. P50 = 33 min, P90 = 224 min.

| Metric | What it checks | Target | Baseline |
|---|---|---|---|
| Pinball loss for P50 / P90 (main) | one error number per estimate that balances "P90 holds" against "P90 is short"; lower is better | below baseline | 17.3 / 20.9 |
| P50 within ±15 min | share of alerts where "typically ends by" was at most 15 min off | ≥ 55% *(assumption)* | 47% |
| P90 kept | share of alerts that really ended before "almost surely ends by" | 85–95% | 97% |
| P90 width | median P90; shorter is more useful | < 224 min | 224 min |

Results are also split by threat type and by night/day, because a 15-min ballistic alert and a 4-hour drone attack behave very differently.

**M3** will be scored in Lab 2 with macro-F1 (F1 per type, averaged, so rare ballistic alerts count as much as drones) against the rule "always predict the most common type".

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

## What is monitored

The truth arrives late (up to 12 h for M1, until the alert ends for M2), so the first three checks work without it and act as early warnings.

| Check | What is compared | Alarm when | Needs the truth? |
|---|---|---|---|
| Data health | freshness of each source, share of posts the parser can't read, rows per batch, failed jobs, API latency | alerts.in.ua unreachable > 2 min · newest alert > 2 days old · > 20% unreadable posts · batch size ±50% | no |
| Input drift | last 7 days (30 for the hourly profile) vs the data the current model was trained on: number of oblasts under alert, raw alert records per hour, Kyiv alerts by hour of day, Telegram threat mix | difference above what normal weeks show, 2 days in a row | no |
| Prediction drift | the model's own outputs: average P(start), share of 🔴 hours, average P90 | change > 50% (last 7 days vs last 30) | no |
| Accuracy | M1 Brier vs baseline; share of the last 30 alerts that ended before P90 | M1 worse than baseline 14 days in a row · P90 held for < 80% | yes |

Drift is measured with PSI (numeric features) and Jensen–Shannon distance (distributions); both are 0 when nothing changed. The thresholds come from 2025–26 history: a normal week reaches PSI up to 0.47, while the November 2025 switch to district-level alerts pushed it to 1.00. Alarms go to the on-call engineer through Grafana (Telegram bot and email). Details and runbook: [design.md §6.6](docs/design.md#66-monitoring-and-alarms).

## Quick start

Requires [uv](https://docs.astral.sh/uv/); it installs Python 3.12 if needed.

```bash
make setup                  # create the pinned environment from uv.lock
make lint                   # run ruff (the same check as pre-commit and CI)
uv run pre-commit install   # optional: lint before every commit
```

## Repository

```
docs/design.md              design document (template sections 1–7.4, diagrams)
src/kyiv_alerts/            package for pipeline and service code (from Lab 2)
.github/workflows/lint.yml  CI: make setup + make lint on every push
Makefile, pyproject.toml, uv.lock, .pre-commit-config.yaml
```

## Maturity scorecard

| Property | Evidence a reviewer checks | Added in |
|---|---|---|
| Repository reproducible from a clean clone | CI is green; `make setup` and `make lint` succeed on a fresh clone | Lab 1 |
| Design document exists and is current | [`docs/design.md`](docs/design.md) follows the template, every section filled or marked as an assumption | Lab 1 |
