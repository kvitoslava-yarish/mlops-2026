"""Kyiv air-alert forecasting system.

Layout:
    config, batch, storage   shared basics (settings, the monthly batch, MinIO access)
    sources/                 upstream alert sources
    features                 feature formulas shared by ETL, training and serving
    etl/                     the steps of the Airflow DAG alerts_etl
    training/                baselines, models, MLflow tracking, reproduction
"""

__version__ = "0.2.0"
