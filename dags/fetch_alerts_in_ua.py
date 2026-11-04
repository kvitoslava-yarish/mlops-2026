from datetime import UTC, datetime, timedelta

from airflow.sdk import dag, task
from airflow.sdk.exceptions import AirflowFailException
from airflow.timetables.trigger import CronTriggerTimetable

from kyiv_alerts.config import get_settings
from kyiv_alerts.sources import alerts_in_ua
from kyiv_alerts.storage import Lake


@dag(
    dag_id="fetch_alerts_in_ua",
    schedule=CronTriggerTimetable("30 0 1,15 * *", timezone="UTC"),
    start_date=datetime(2026, 11, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=10)},
    tags=["kyiv-alerts"],
)
def fetch_alerts_in_ua():
    @task
    def fetch() -> str:
        settings = get_settings()
        if not settings.alerts_in_ua_token:
            raise AirflowFailException("ALERTS_IN_UA_TOKEN is not set in .env")
        return alerts_in_ua.archive_month_ago(settings.alerts_in_ua_token, Lake(settings).put)

    @task
    def check(folder: str) -> None:
        lake = Lake(get_settings())
        saved = lake.list(folder + "/")
        if len(saved) != len(alerts_in_ua.OBLAST_UIDS):
            raise AirflowFailException(f"{folder}: {len(saved)} of 25 oblasts saved")
        keys = lake.list(alerts_in_ua.ARCHIVE_PREFIX + "/")
        _, windows = alerts_in_ua.read_archive((k, lake.get(k)) for k in keys)
        for lo, hi in windows:
            print(f"complete fetch covers {lo:%Y-%m-%d %H:%M} .. {hi:%Y-%m-%d %H:%M} UTC")

    check(fetch())


fetch_alerts_in_ua()
