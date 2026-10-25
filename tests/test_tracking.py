import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression

from kyiv_alerts.tracking import MANIFEST_FILE, TrackedRun, has_alias

TAGS = {"data_version": "abc.dir", "git_commit": "deadbeef", "feature_version": "v2"}


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A throwaway MLflow tracking server: SQLite backend, artifacts in tmp_path."""
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.create_experiment("exp", artifact_location=(tmp_path / "artifacts").as_uri())
    yield MlflowClient()
    mlflow.set_tracking_uri(None)


def test_refuses_run_without_lineage(store):
    with pytest.raises(ValueError, match="git_commit"):
        TrackedRun("exp", "r", {"data_version": "abc.dir"})
    assert store.search_runs([store.get_experiment_by_name("exp").experiment_id]) == []


def test_every_write_lands_in_mlflow_and_the_manifest(store):
    X = pd.DataFrame({"a": np.arange(20.0), "b": np.arange(20.0) % 3})
    y = (X["a"] > 9).astype(int)
    model = LogisticRegression().fit(X, y)

    with TrackedRun("exp", "r", {**TAGS, "model_family": "m1"}) as run:
        run.log_params({"seed": 42, "split.train_end": "2026-02-28"})
        run.log_dataset(X.assign(y=y), "train", "training", "data/features/x.parquet", "y")
        run.log_metrics({"holdout_brier_model": 0.06, "n_train": 20})
        run.log_dict({"seed": 42}, "config/train.yaml")
        version = run.log_sklearn_model(model, registered_name="m", input_example=X.head(2))
        run.register("m", version, version_tags=TAGS, alias="champion")

    logged = store.get_run(run.run_id)
    assert logged.info.status == "FINISHED"
    assert logged.data.params["seed"] == "42"
    assert logged.data.metrics["holdout_brier_model"] == 0.06
    assert logged.data.tags["git_commit"] == "deadbeef"
    assert logged.inputs.dataset_inputs[0].dataset.digest == "abc.dir"

    path = mlflow.artifacts.download_artifacts(run_id=run.run_id, artifact_path=MANIFEST_FILE)
    manifest = json.loads(Path(path).read_text())
    assert manifest["status"] == "FINISHED"
    assert set(manifest["params"]) == {"seed", "split.train_end"}
    assert manifest["metrics"]["n_train"] == 20.0
    assert manifest["datasets"][0]["rows"] == 20
    assert manifest["artifacts"] == ["config/train.yaml"]
    assert manifest["model"]["class"] == "LogisticRegression"
    assert manifest["registry"] == {"name": "m", "version": "1", "version_tags": TAGS,
                                    "alias": "champion"}  # fmt: skip

    assert has_alias("m", "champion")
    assert store.get_model_version("m", "1").tags["data_version"] == "abc.dir"


def test_failed_run_is_marked_and_still_has_a_manifest(store):
    with pytest.raises(RuntimeError), TrackedRun("exp", "r", TAGS) as run:
        run.log_metrics({"x": 1})
        raise RuntimeError("training crashed")
    assert store.get_run(run.run_id).info.status == "FAILED"
    assert run.manifest["status"] == "FAILED"
