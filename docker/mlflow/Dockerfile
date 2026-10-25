FROM ghcr.io/mlflow/mlflow:v3.16.1

# PostgreSQL backend store and S3 (MinIO) artifact store drivers.
RUN pip install --no-cache-dir "psycopg2-binary==2.9.12" "boto3==1.43.56"

# Apple Silicon + old Docker Desktop: if the container exits with code 132,
# update Docker Desktop first; if that is not possible, uncomment the next line.
# RUN pip install --no-cache-dir "cryptography==46.0.5"
