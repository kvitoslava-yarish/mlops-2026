"""Settings read from the environment.

Defaults point at the local docker compose stack as seen from the host; the
Airflow containers override them (see docker-compose.yml).
"""

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    warehouse_dsn: str
    s3_endpoint_url: str
    aws_access_key_id: str
    aws_secret_access_key: str
    lake_bucket: str
    raw_data_dir: Path
    alerts_in_ua_token: str


def get_settings() -> Settings:
    return Settings(
        warehouse_dsn=os.getenv(
            "WAREHOUSE_DSN", "postgresql://warehouse:warehouse@localhost:5433/warehouse"
        ),
        s3_endpoint_url=os.getenv("S3_ENDPOINT_URL", "http://localhost:9000"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123"),
        lake_bucket=os.getenv("LAKE_BUCKET", "lake"),
        raw_data_dir=Path(os.getenv("RAW_DATA_DIR", str(REPO_ROOT / "data" / "raw"))),
        alerts_in_ua_token=os.getenv("ALERTS_IN_UA_TOKEN", ""),
    )
