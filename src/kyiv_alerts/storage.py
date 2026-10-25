"""Object store (MinIO, S3 API) access for the lake bucket."""

import hashlib
from functools import lru_cache

import boto3
from botocore.exceptions import ClientError

from kyiv_alerts.config import Settings


@lru_cache(maxsize=4)
def _client(endpoint: str, key: str, secret: str):
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        region_name="us-east-1",
    )


class Lake:
    def __init__(self, settings: Settings):
        self.bucket = settings.lake_bucket
        self.s3 = _client(
            settings.s3_endpoint_url, settings.aws_access_key_id, settings.aws_secret_access_key
        )

    def exists(self, key: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def put(self, key: str, data: bytes) -> str:
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data)
        return sha256(data)

    def get(self, key: str) -> bytes:
        return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
