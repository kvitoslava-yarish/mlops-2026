from datetime import UTC, datetime, timedelta

from airflow.sdk import dag, task
from airflow.sdk.exceptions import AirflowFailException
from airflow.timetables.trigger import CronTriggerTimetable

from kyiv_alerts.config import get_settings
from kyiv_alerts.etl import telegram, warehouse
from kyiv_alerts.llm.index import load_config, update_index
from kyiv_alerts.storage import Lake


@dag(
    dag_id="ingest_telegram",
    schedule=CronTriggerTimetable("45 3,9,15,21 * * *", timezone="Europe/Kyiv"),
    start_date=datetime(2026, 11, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=3)},
    tags=["kyiv-alerts", "telegram"],
)
def ingest_telegram():
    @task
    def fetch() -> list[str]:
        cfg, settings = load_config()["ingest"], get_settings()
        conn = warehouse.connect(settings.warehouse_dsn)
        try:
            warehouse.ensure_schema(conn)
            return telegram.fetch_to_lake(conn, Lake(settings).put, cfg["channels"],
                                          cfg["first_since"], cfg["sleep_s"])  # fmt: skip
        finally:
            conn.close()

    @task
    def load(keys: list[str]) -> dict[str, int]:
        settings = get_settings()
        conn = warehouse.connect(settings.warehouse_dsn)
        try:
            inserted = telegram.load_posts(conn, Lake(settings).get, keys)
        finally:
            conn.close()
        print(f"inserted {inserted} from {len(keys)} raw files")
        return inserted

    @task
    def index(_: dict) -> int:
        cfg = load_config()
        conn = warehouse.connect(get_settings().warehouse_dsn)
        try:
            n = update_index(conn, cfg, cfg["retrieval"]["embedding_model"])
        finally:
            conn.close()
        print(f"embedded {n} new posts into rag.chunks_{cfg['retrieval']['embedding_model']}")
        return n

    @task
    def check(_: dict) -> None:
        limits = load_config()["ingest"]["max_age_hours"]
        conn = warehouse.connect(get_settings().warehouse_dsn)
        try:
            age = telegram.hours_since_newest(conn)
        finally:
            conn.close()
        stale = {ch: round(age.get(ch, float("inf")), 1) for ch, limit in limits.items()
                 if age.get(ch, float("inf")) > limit}  # fmt: skip
        print({ch: round(h, 1) for ch, h in age.items()}, "hours since the newest post")
        if stale:
            raise AirflowFailException(f"stale channels (hours): {stale}, limits {limits}")

    loaded = load(fetch())
    index(loaded)
    check(loaded)


ingest_telegram()
