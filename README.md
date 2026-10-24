# Kyiv air-alert forecasting (KSE MLOps 2026)

## Problem

Kyiv residents and businesses plan their nights and days around air-raid alerts, but today they only learn about an alert when the siren starts and never know how long it will last.
This system issues a 12-hour forecast twice a day, at 10:00 and at 22:00 Kyiv time, giving the probability of an alert in each hour of the window, and predicts the duration (median and 90th percentile) of every Kyiv alert the moment it starts.
It learns from the public history of air-raid alerts across Ukraine since 2022 and from the Ukrainian Air Force's public Telegram channel, and it keeps learning as new alerts and posts arrive every day.
Predictions are served through a REST API and are meant for planning only: the official alert always takes precedence over the model.
Success means beating simple baselines (hour-of-day alert rates and the last-12-months median duration) on a held-out recent period, and noticing quickly when the attack patterns change.

Full design: [docs/design.md](docs/design.md).

## Quick start

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 if needed).

```bash
make setup   # create the pinned environment from uv.lock
make lint    # run ruff (same check as pre-commit and CI)
uv run pre-commit install   # optional: lint before every commit
```

## Maturity scorecard

| Property | Evidence a reviewer checks | Added in |
|---|---|---|
| Repository reproducible from a clean clone | CI is green; `make setup` and `make lint` succeed on a fresh clone | Lab 1 |
| Design document exists and is current | [`docs/design.md`](docs/design.md) follows the template, every section filled or marked as an assumption | Lab 1 |
