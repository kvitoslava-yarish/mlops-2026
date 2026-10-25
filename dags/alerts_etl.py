"""alerts_etl: one calendar month of air-raid alerts -> warehouse -> feature tables.

The batch is the run's data interval: CronDataIntervalTimetable("@monthly") gives
each run [month start, next month start), and the batch is the month of
data_interval_start. A manual run can name the month instead with the
``batch_month`` param ("YYYY-MM"). Nothing reads the wall clock.

Rerunning a month is harmless: the raw file in the lake is reused as is, and
every warehouse write is delete-then-insert of that month inside one transaction.
"""

from datetime import UTC, datetime

from airflow.sdk import Param, dag, task
from airflow.sdk.exceptions import AirflowFailException
from airflow.timetables.interval import CronDataIntervalTimetable

from kyiv_alerts import pipeline
from kyiv_alerts.batch import resolve_batch
from kyiv_alerts.config import get_settings
from kyiv_alerts.quality import QualityGateError
from kyiv_alerts.validate import ValidationError


def _batch(context):
    return resolve_batch(context["params"].get("batch_month"), context.get("data_interval_start"))


@dag(
    dag_id="alerts_etl",
    schedule=CronDataIntervalTimetable("@monthly", timezone="UTC"),
    start_date=datetime(2022, 3, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,  # months build on the previous month's lookback
    default_args={"retries": 0},
    params={
        "batch_month": Param("", type="string", description="YYYY-MM; empty = data interval"),
        "raw_override": Param("", type="string", description="CSV path in the container"),
        "refresh_raw": Param(False, type="boolean", description="re-extract even if raw exists"),
    },
    tags=["kyiv-alerts", "lab2"],
    doc_md=__doc__,
)
def alerts_etl():
    @task
    def extract(**context) -> dict:
        batch, params = _batch(context), context["params"]
        info = pipeline.extract(
            get_settings(), batch, params.get("raw_override", ""), params.get("refresh_raw", False)
        )
        print(f"batch {batch.month} [{batch.start} .. {batch.end}): {info}")
        return {"batch_month": batch.month, **info}

    @task
    def validate(info: dict, **context) -> dict:
        try:
            report = pipeline.validate(get_settings(), _batch(context), info)
        except ValidationError as e:
            raise AirflowFailException(str(e)) from e
        print(report)
        return info

    @task
    def transform(info: dict, **context) -> dict:
        counts = pipeline.transform(get_settings(), _batch(context), info)
        print(counts)
        return {**info, "counts": counts}

    @task
    def load(info: dict, **context) -> dict:
        n = pipeline.load(get_settings(), _batch(context), info, info["counts"], context["run_id"])
        print(f"dwh.alerts: {n} rows for {info['batch_month']}")
        return info

    @task
    def build_features(info: dict, **context) -> dict:
        n = pipeline.build_features(get_settings(), _batch(context))
        print(n)
        return {**info, **n}

    @task
    def quality_check(info: dict, **context) -> None:
        try:
            pipeline.quality_check(get_settings(), _batch(context), info["counts"])
        except QualityGateError as e:
            raise AirflowFailException(str(e)) from e

    quality_check(build_features(load(transform(validate(extract())))))


alerts_etl()
