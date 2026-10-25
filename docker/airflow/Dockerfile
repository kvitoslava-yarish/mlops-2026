FROM apache/airflow:3.3.1-python3.12

# pandas, pyarrow, boto3, psycopg2 and requests already ship with the image;
# the pipeline code is mounted at /opt/airflow/src (PYTHONPATH), not installed.

# Apple Silicon + old Docker Desktop: if containers exit with code 132, update
# Docker Desktop first; if that is not possible, uncomment the next line.
# RUN pip install --no-cache-dir "cryptography==46.0.5"
