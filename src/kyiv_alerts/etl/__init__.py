"""ETL for one monthly batch: the steps the Airflow DAG ``alerts_etl`` runs.

pipeline (task functions) -> validate -> clean (transform) -> warehouse (load)
-> features (build_features, shared module) -> quality (gate); snapshot exports
the passed batches for training.
"""
