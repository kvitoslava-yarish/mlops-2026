# Every target runs from the repository root. .env (copied from .env.example) is
# exported to all commands, so DVC, MLflow and the pipeline find the local stack.
-include .env
export

COMPOSE  := docker compose
MONTH    ?= 2025-06
FROM     ?= 2022-03
TO       ?= 2026-09
MODEL    ?= kyiv-m1-start-risk
ALIAS    ?= champion

.PHONY: setup lint test env up down nuke ps logs data data-seed run backfill bad-batch \
        row-counts snapshot train reproduce feast-demo api-smoke

## --- environment ----------------------------------------------------------
setup:            ## pinned Python env from uv.lock
	uv sync

lint:
	uv run ruff check .

test:
	uv run pytest -q

env:
	@test -f .env || (cp .env.example .env && echo "created .env from .env.example")

## --- local stack ------------------------------------------------------------
up: env           ## build and start Postgres, MinIO, MLflow, Airflow; wait until healthy
	$(COMPOSE) up -d --build --wait
	@echo "Airflow  http://localhost:8080"
	@echo "MLflow   http://localhost:5001"
	@echo "MinIO    http://localhost:9001  ($(MINIO_ROOT_USER) / $(MINIO_ROOT_PASSWORD))"

down:
	$(COMPOSE) down

nuke:             ## stop and DELETE all volumes (warehouse, MinIO, MLflow, Airflow state)
	$(COMPOSE) down -v

ps:
	$(COMPOSE) ps

logs:
	$(COMPOSE) logs -f --tail=100

## --- data versioning ----------------------------------------------------------
data:             ## pull the raw data version recorded in data/raw.dvc from MinIO
	uv run dvc pull data/raw.dvc
	uv run python scripts/checksum.py data/raw

data-seed:        ## empty remote (new machine): download pinned upstream, verify, push
	scripts/seed_data.sh

## --- pipeline -------------------------------------------------------------------
run:              ## run alerts_etl for one month: make run MONTH=2025-06
	scripts/airflow_run.sh $(MONTH)

backfill:         ## run every month FROM..TO in order: make backfill FROM=2022-03 TO=2026-09
	scripts/airflow_backfill.sh $(FROM) $(TO)

bad-batch:        ## feed a deliberately broken 2025-06 batch; quality_check must fail
	uv run python scripts/make_bad_batch.py $(MONTH)
	-scripts/airflow_run.sh $(MONTH) /opt/airflow/data/fixtures/bad_batch_$(MONTH).csv
	@echo "expected: quality_check failed. Restore the month with: make run MONTH=$(MONTH)"

row-counts:       ## rows per table for one month: make row-counts MONTH=2025-06
	uv run python -m kyiv_alerts.warehouse $(MONTH)

## --- training -------------------------------------------------------------------
snapshot:         ## export passed batches to data/features, version with DVC, push
	uv run python -m kyiv_alerts.snapshot
	uv run dvc add data/features
	uv run dvc push data/features.dvc
	@echo "now commit data/features.dvc, then: make train"

train:            ## train M1 + M2 with baselines, log to MLflow, register (CHAMPION=1 moves alias)
	uv run python -m kyiv_alerts.train --model all $(if $(CHAMPION),--champion)

reproduce:        ## rerun a registered version from its tags: make reproduce MODEL=... ALIAS=champion
	uv run python -m kyiv_alerts.reproduce --model $(MODEL) $(if $(VERSION),--version $(VERSION),--alias $(ALIAS))

feast-demo:       ## optional: Feast point-in-time retrieval on the exported M1 features
	uv run --group feast python feature_repo/demo.py

api-smoke:        ## check ALERTS_IN_UA_TOKEN: fetch the last 30 days for Kyiv only
	uv run python -m kyiv_alerts.sources.alerts_in_ua
