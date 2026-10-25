import json
import warnings

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

MANIFEST_FILE = "logging_manifest.json"


class TrackedRun:
    REQUIRED_TAGS = ("data_version", "git_commit")

    def __init__(self, experiment: str, run_name: str, tags: dict):
        missing = [t for t in self.REQUIRED_TAGS if not tags.get(t)]
        if missing:
            raise ValueError(f"lineage tags missing: {missing}; refusing to start the run")
        self.experiment, self.run_name, self._tags = experiment, run_name, tags
        self.run = None
        self.manifest: dict = {
            "experiment": experiment,
            "run_name": run_name,
            "run_id": None,
            "tags": {},
            "params": {},
            "metrics": {},
            "datasets": [],
            "artifacts": [],
            "model": None,
            "registry": None,
        }

    def __enter__(self) -> "TrackedRun":
        mlflow.set_experiment(self.experiment)
        self.run = mlflow.start_run(run_name=self.run_name)
        self.manifest["run_id"] = self.run.info.run_id
        self.log_tags(self._tags)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        status = "FAILED" if exc_type else "FINISHED"
        self.manifest["status"] = status
        try:
            mlflow.log_dict(self.manifest, MANIFEST_FILE)
        finally:
            mlflow.end_run(status)
        print(self.summary())
        return False  # never swallow the exception

    @property
    def run_id(self) -> str:
        return self.manifest["run_id"]

    # --- writes ------------------------------------------------------------
    def log_tags(self, tags: dict) -> None:
        tags = {k: str(v) for k, v in tags.items()}
        mlflow.set_tags(tags)
        self.manifest["tags"].update(tags)

    def log_params(self, params: dict) -> None:
        mlflow.log_params(params)
        self.manifest["params"].update({k: str(v) for k, v in params.items()})

    def log_metrics(self, metrics: dict) -> None:
        metrics = {k: float(v) for k, v in metrics.items()}
        mlflow.log_metrics(metrics)
        self.manifest["metrics"].update(metrics)

    def log_dict(self, obj, artifact_file: str) -> None:
        mlflow.log_dict(obj, artifact_file)
        self.manifest["artifacts"].append(artifact_file)

    def log_dataset(
        self,
        df: pd.DataFrame,
        name: str,
        context: str,
        source: str,
        targets: str | None = None,
    ) -> None:
        digest = self._tags["data_version"]
        with warnings.catch_warnings():
            # MLflow registers its local-file source type twice and warns about the tie.
            warnings.filterwarnings("ignore", "The specified dataset source can be interpreted")
            dataset = mlflow.data.from_pandas(
                df, source=source, targets=targets, name=name, digest=digest
            )
        mlflow.log_input(dataset, context=context)
        self.manifest["datasets"].append(
            {"name": name, "context": context, "source": source, "rows": len(df),
             "columns": len(df.columns), "targets": targets, "digest": digest}
        )  # fmt: skip

    def log_sklearn_model(
        self, model, name: str = "model", registered_name: str | None = None, **kwargs
    ) -> str | None:
        """Log (and optionally register) a scikit-learn model; returns the registered version."""
        info = mlflow.sklearn.log_model(
            model, name=name, registered_model_name=registered_name, **kwargs
        )
        version = info.registered_model_version if registered_name else None
        self.manifest["model"] = {
            "artifact": name,
            "flavor": "sklearn",
            "class": type(model).__name__,
            "model_uri": info.model_uri,
            "registered_name": registered_name,
            "version": None if version is None else str(version),
        }
        return None if version is None else str(version)

    def register(
        self, name: str, version: str, version_tags: dict, alias: str | None = None
    ) -> None:
        """Copy lineage tags onto the registered version and optionally point an alias at it."""
        client = MlflowClient()
        for key, value in version_tags.items():
            client.set_model_version_tag(name, version, key, str(value))
        if alias:
            client.set_registered_model_alias(name, alias, version)
        self.manifest["registry"] = {
            "name": name,
            "version": str(version),
            "version_tags": {k: str(v) for k, v in version_tags.items()},
            "alias": alias,
        }

    def summary(self) -> str:
        m = self.manifest
        model = m["model"]["class"] if m["model"] else "none"
        reg = m["registry"]
        registered = f"{reg['name']} v{reg['version']}" if reg else "not registered"
        if reg and reg["alias"]:
            registered += f" @{reg['alias']}"
        return (
            f"[mlflow] {m['experiment']} / {m['run_name']} ({m['run_id']}) {m.get('status', '')}: "
            f"{len(m['tags'])} tags · {len(m['params'])} params · {len(m['metrics'])} metrics · "
            f"{len(m['datasets'])} datasets · {len(m['artifacts']) + 1} artifacts · "
            f"model {model} · {registered}"
        )

    def manifest_json(self) -> str:
        return json.dumps(self.manifest, indent=2, default=str)


def has_alias(name: str, alias: str) -> bool:
    try:
        MlflowClient().get_model_version_by_alias(name, alias)
        return True
    except MlflowException:
        return False
