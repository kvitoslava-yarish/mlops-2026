# Every target runs from the repository root. .env (copied from .env.example) is
# exported to all commands, so DVC, MLflow and the pipeline find the local stack.
-include .env
export

COMPOSE  := docker compose
MONTH    ?= 2025-06
FROM     ?= 2022-03
TO       ?= 2026-08
MODEL    ?= kyiv-m1-start-risk
ALIAS    ?= champion

.PHONY: setup lint test env up down nuke ps logs data data-seed run backfill bad-batch \
        row-counts snapshot train reproduce api-smoke corpus index index-all eval-windows \
        eval-prelabel llm-rule llm-baseline rag-eval rag-eval-all ingest-telegram tg-posts

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
	uv run python -m kyiv_alerts.etl.warehouse $(MONTH)

## --- training -------------------------------------------------------------------
snapshot:         ## export passed batches to data/features, version with DVC, push
	uv run python -m kyiv_alerts.etl.snapshot
	uv run dvc add data/features
	uv run dvc push data/features.dvc
	@echo "now commit data/features.dvc, then: make train"

train:            ## train M1 + M2 with baselines, log to MLflow, register (CHAMPION=1 moves alias)
	uv run python -m kyiv_alerts.training.train --model all $(if $(CHAMPION),--champion)

reproduce:        ## rerun a registered version from its tags: make reproduce MODEL=... ALIAS=champion
	uv run python -m kyiv_alerts.training.reproduce --model $(MODEL) $(if $(VERSION),--version $(VERSION),--alias $(ALIAS))

api-smoke:        ## check ALERTS_IN_UA_TOKEN: fetch the last 30 days for Kyiv only
	uv run python -m kyiv_alerts.sources.alerts_in_ua

## --- Lab 3: LLM + retrieval ------------------------------------------------------------
corpus:           ## scrape @smolii_ukraine for the corpus range, version data/corpus with DVC
	uv run python -m kyiv_alerts.sources.telegram --channel smolii_ukraine \
		--since 2026-07-24 --until 2026-10-24 --out data/corpus/smolii_ukraine.csv
	uv run dvc add data/corpus
	uv run dvc push data/corpus.dvc

index:            ## rebuild the pgvector index from the corpus: make index [MODEL=e5_small]
	uv run python -m kyiv_alerts.llm.index $(if $(MODEL),--model $(MODEL))

index-all:        ## rebuild the index for every embedding model in configs/rag.yaml
	for m in e5_small e5_base minilm; do uv run python -m kyiv_alerts.llm.index --model $$m; done

eval-windows:     ## pick the evaluation windows and questions (seeded) -> data/eval/windows.csv
	uv run python -m kyiv_alerts.llm.windows select

eval-prelabel:    ## pre-label every post in every window (rule + Gemini) -> data/eval/labels.csv
	uv run python -m kyiv_alerts.llm.windows prelabel

llm-rule:         ## keyword-rule baseline (no LLM) on the eval set, logged to MLflow
	uv run python -m kyiv_alerts.llm.evaluate rule

llm-baseline:     ## LLM baseline: Gemini on all window posts, no retrieval, logged to MLflow
	uv run python -m kyiv_alerts.llm.evaluate baseline

rag-eval:         ## retrieval + Gemini on the eval set: make rag-eval [MODEL=e5_small]
	uv run python -m kyiv_alerts.llm.evaluate rag $(if $(MODEL),--model $(MODEL))

rag-eval-all:     ## the embedding-model comparison: rag-eval for every model
	for m in e5_small e5_base minilm; do uv run python -m kyiv_alerts.llm.evaluate rag --model $$m; done

ingest-telegram:  ## run DAG ingest_telegram now (scheduled 03:45/09:45/15:45/21:45 Kyiv) and wait
	scripts/airflow_trigger.sh ingest_telegram

tg-posts:         ## ingested Telegram posts per channel: count, newest post, hours since
	docker compose exec -T postgres psql -U warehouse -d warehouse -c \
	  "SELECT channel, count(*), max(posted_at) AS newest, round(extract(epoch FROM now() - max(posted_at)) / 3600, 1) AS hours_ago FROM telegram.posts GROUP BY channel"
