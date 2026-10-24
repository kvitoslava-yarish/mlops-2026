# Kyiv air-alert forecasting

KSE MLOps 2026 course project. Full design: [docs/design.md](docs/design.md).

## Problem

Kyiv alerts are known only when the siren starts, with no estimate of when the next one comes or how long it lasts.
Twice a day (10:00, 22:00) the system forecasts, for each of the next 12 hours, the probability that a Kyiv alert starts and its likely threat type.
When a Kyiv alert starts, it predicts the alert's duration (P50, P90).
Inputs: public alert history for all of Ukraine since 2022 (dataset + alerts.in.ua API) and the Ukrainian Air Force Telegram channel; the system retrains monthly and monitors drift daily.
Predictions are served by a REST API for planning only; the official alert always takes precedence.

## What is predicted

| | When | Output | Label known |
|---|---|---|---|
| **M1** start risk | 10:00 and 22:00, next 12 h | P(a Kyiv alert **starts** in the hour) + risk band 🟢 < 5% · 🟡 5–9% · 🔴 ≥ 9% | end of each hour |
| **M3** threat type | with M1 | P(drones / cruise missiles / ballistic \| alert) per hour | when the type appears in Telegram |
| **M2** duration | at each Kyiv alert start | P50 "typically ends by", P90 "almost surely ends by" | when the alert ends |

Why "starts in the hour": each alert falls in exactly one hour, so none is lost or double-counted (62% of alerts cross an hour boundary). Details: [design.md §5.1](docs/design.md#51-problem-statement).

Example, 22:00 issue. P(start) = current baseline (2025–26 share of hours in which an alert started); type row illustrative until M3 is trained.

| Hour | 22 | 23 | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P(start) | 9.3% | 10.1% | 11.9% | 10.8% | 6.4% | 8.2% | 6.2% | 4.7% | 5.1% | 4.4% | 4.2% | 6.7% |
| Risk | 🔴 | 🔴 | 🔴 | 🔴 | 🟡 | 🟡 | 🟡 | 🟢 | 🟡 | 🟢 | 🟢 | 🟡 |
| Likely type | drones | drones | drones | drones | drones | drones | mixed | mixed | ballistic | ballistic | ballistic | ballistic |

Window: P(at least one alert starts) 0.57 · expected alerts 0.88.

M2 example: alert starts 23:41 after an Air Force drone post → P50 95 min, P90 240 min → ends by 01:16 (typically) / 03:41 (almost surely).

API: `GET /v1/forecast/latest`, `GET /v1/alert/current`, `GET /v1/health`.

## How predictions are scored

Hold-out 2026-03-01 … 2026-09-05. Thresholds marked *a* are assumptions.

**M1** (4,536 hours, an alert started in 7.5%)

| Metric | Meaning | Target | Baseline (hour of day) | First model |
|---|---|---|---|---|
| Brier (main) | mean of (P − 1/0)² per hour; lower is better | below baseline | 0.0695 | 0.0687 |
| F1 ±1 h | flagged hours vs real starts; a start at 23:58 counts as a hit for a 00:00 flag | ≥ 0.30 *a* | 0.25 | 0.27 |
| AUC | ranks start hours above quiet hours; 0.5 = random | ≥ 0.60 | 0.57 | 0.58 |

**M2** (349 alerts)

| Metric | Meaning | Target | Baseline (last 12 months: 33 / 224 min) |
|---|---|---|---|
| Pinball loss P50 / P90 (main) | one number balancing "P90 kept" and "P90 short"; lower is better | below baseline | 17.3 / 20.9 |
| P50 within ±15 min | share of alerts where "typically ends by" was ≤ 15 min off | ≥ 55% *a* | 47% |
| P90 kept | share of alerts that ended before P90 | 85–95% | 97% |
| P90 width | median P90 | < 224 min | 224 min |

Also reported by threat type and night/day.

**M3**: macro-F1 over the three types (rare types count equally) and log-loss, vs "always the most common type"; measured in Lab 2.

## What is monitored

| Layer | Signal | Alarm | Labels needed |
|---|---|---|---|
| Data health | source freshness, parser failures, batch volume, DAG runs, API latency | alerts.in.ua down > 2 min · newest alert > 2 d old · unparsed posts > 20% · volume ±50% | no |
| Input drift | raw alert records/h, oblasts under alert (PSI, 7 d); Kyiv hour-of-day profile (JS, 30 d); Telegram type mix | above 95th pct of normal variation (PSI 0.5, JS 0.13), 2 days in a row | no |
| Prediction drift | mean P(start), share of 🔴 hours, mean P90, type mix | change > 50% (7 d vs 30 d) | no |
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
