# Kyiv air-alert forecasting

KSE MLOps 2026 course project. Full design: [docs/design.md](docs/design.md).

## Problem

Kyiv alerts are known only when the siren starts, with no estimate of when the next one comes or how long it lasts.
Twice a day (10:00, 22:00) the system forecasts, for each of the next 12 hours, the probability of a Kyiv alert and its likely threat type.
When a Kyiv alert starts, it predicts the alert's duration (P50, P90).
Inputs: public alert history for all of Ukraine since 2022 (dataset + alerts.in.ua API) and the Ukrainian Air Force Telegram channel; the system retrains monthly and monitors drift daily.
Predictions are served by a REST API for planning only; the official alert always takes precedence.

## What is predicted

| | When | Output | Label known |
|---|---|---|---|
| **M1** hourly risk | 10:00 and 22:00, next 12 h | P(Kyiv alert ≥ 10 min) per hour + risk band 🟢 < 10% · 🟡 10–20% · 🔴 ≥ 20% | end of each hour |
| **M3** threat type | with M1 | P(drones / cruise missiles / ballistic \| alert) per hour | when the alert's type appears in Telegram |
| **M2** duration | at each Kyiv alert start | P50 and P90 minutes | when the alert ends |

Example, 22:00 issue (P(alert) = current baseline: 2025–26 share of alert hours by hour of day; type columns illustrative until M3 is trained):

| Hour | 22 | 23 | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P(alert) | 12% | 18% | 23% | 27% | 23% | 21% | 18% | 15% | 13% | 12% | 8% | 8% |
| Risk | 🟡 | 🟡 | 🔴 | 🔴 | 🔴 | 🔴 | 🟡 | 🟡 | 🟡 | 🟡 | 🟢 | 🟢 |
| Likely type | drones | drones | drones | drones | drones | drones | mixed | mixed | ballistic | ballistic | ballistic | ballistic |

Window: P(≥ 60 min of alert) 0.33 · expected alert-hours 2.0 · P(any drones) 0.28 · P(any missiles/ballistic) 0.10.

M2 example: alert starts 23:41 after an Air Force drone post → duration P50 95 min, P90 240 min → ends by 01:16 (P50) / 03:41 (P90).

API: `GET /v1/forecast/latest`, `GET /v1/alert/current`, `GET /v1/health`. JSON format: [design.md §5.1](docs/design.md#51-problem-statement).

## Baselines (hold-out 2026-03 … 2026-08)

| | Baseline | Score | First model |
|---|---|---|---|
| M1 | alert rate by hour of day | Brier 0.0981, AUC 0.58 | GBM on alert history: Brier 0.0946, AUC 0.64 |
| M2 | last-12-months median / P90 (33 / 224 min) | MAE 34.6 min | quantile GBM (Lab 2) |
| M3 | last month's type mix | log-loss (Lab 2) | multiclass GBM (Lab 2) |

## What is monitored

| Layer | Signal | Alarm | Labels needed |
|---|---|---|---|
| Data health | source freshness, parser failures, batch volume, DAG runs, API latency | alerts.in.ua down > 2 min · newest alert > 2 d old · unparsed posts > 20% · volume ±50% | no |
| Input drift | raw alert records/h, oblasts under alert (PSI, 7 d); Kyiv hour-of-day profile (JS, 30 d); Telegram type mix | above 95th pct of normal variation (PSI 0.5, JS 0.13), 2 days in a row | no |
| Prediction drift | mean P(alert), share of 🔴 hours, mean P90, type mix | change > 50% (7 d vs 30 d) | no |
| Performance | M1 Brier vs baseline · M2 P90 kept over last 30 alerts · M3 log-loss | M1 worse 14 days in a row · M2 < 80% · M3 worse than baseline | yes |

Reference = champion's training data. Thresholds are calibrated on 2025–26 history, e.g. the Nov 2025 switch to raion-level alerts moved raw-record PSI to 1.00 while normal weeks stay ≤ 0.47. Alarms go to the on-call engineer via Grafana → Telegram bot / email; runbook in [design.md §6.6](docs/design.md#66-monitoring-and-alarms).

## Quick start

```bash
make setup   # uv sync from uv.lock (installs Python 3.12 if needed)
make lint    # ruff, same check as pre-commit and CI
uv run pre-commit install
```

## Maturity scorecard

| Property | Evidence a reviewer checks | Added in |
|---|---|---|
| Repository reproducible from a clean clone | CI is green; `make setup` and `make lint` succeed on a fresh clone | Lab 1 |
| Design document exists and is current | [`docs/design.md`](docs/design.md) follows the template, every section filled or marked as an assumption | Lab 1 |
