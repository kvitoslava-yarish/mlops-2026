FROM apache/airflow:3.3.1-python3.12

# pandas, pyarrow, boto3, psycopg2 and requests already ship with the image;
# the pipeline code is mounted at /opt/airflow/src (PYTHONPATH), not installed.

# DAG ingest_telegram embeds new posts: the same encoder stack and versions as uv.lock, so the
# vectors in pgvector match the ones the host-side evaluation was run with. CPU-only torch;
# apache-airflow is pinned in the same call so pip cannot move Airflow's own dependencies.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==2.15.0" \
 && pip install --no-cache-dir "apache-airflow==3.3.1" "sentence-transformers==6.2.0" \
      "transformers==5.22.2" "pgvector==0.5.1"
# owned by the airflow user, so the named volume mounted here is writable
RUN mkdir -p /opt/airflow/hf-cache

# Apple Silicon + old Docker Desktop: if containers exit with code 132, update
# Docker Desktop first; if that is not possible, uncomment the next line.
# RUN pip install --no-cache-dir "cryptography==46.0.5"
