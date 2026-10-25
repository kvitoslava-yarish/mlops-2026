"""Train M1 (start risk) and M2 (duration) against their baselines on a time split.

Every run logs params, the train / hold-out data sets, metrics for baseline(s) and
model side by side, and the two lineage tags: ``data_version`` (DVC hash of
data/features) and ``git_commit``. All writes go through ``tracking.TrackedRun``,
which also logs ``logging_manifest.json``. A registered model version carries the
same tags; ``--champion`` moves the alias.

    uv run python -m kyiv_alerts.training.train --model all [--champion]
"""

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingClassifier

from kyiv_alerts.config import REPO_ROOT
from kyiv_alerts.features import FEATURE_VERSION, M1_FEATURES, M1_LABEL, M2_FEATURES, M2_LABEL
from kyiv_alerts.training import lineage, metrics
from kyiv_alerts.training.models import QuantilePair
from kyiv_alerts.training.tracking import TrackedRun, has_alias

FEATURES_DIR = REPO_ROOT / "data" / "features"
FEATURES_DVC = REPO_ROOT / "data" / "features.dvc"
FEATURE_FILES = {"m1": "m1_hourly.parquet", "m2": "m2_alert.parquet"}
RAW_DVC = REPO_ROOT / "data" / "raw.dvc"
CHAMPION = "champion"
# Main metric per model family (lower is better); reproduce.py compares these.
MAIN_METRIC = {"m1": "holdout_brier_model", "m2": "holdout_pinball_90_model"}
ONE_DAY = timedelta(days=1)
LINEAGE_VERSION_TAGS = ("data_version", "git_commit", "raw_data_version", "feature_version")
# Our own HistGradientBoosting trees; skops (MLflow 3 default format) must be told to load them.
TRUSTED_TYPES = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]


def _ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def time_split(df: pd.DataFrame, col: str, split: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = pd.to_datetime(df[col], utc=True)
    train = df[(t >= _ts(split["train_start"])) & (t < _ts(split["train_end"]) + ONE_DAY)]
    hold = df[(t >= _ts(split["holdout_start"])) & (t < _ts(split["holdout_end"]) + ONE_DAY)]
    return train.reset_index(drop=True), hold.reset_index(drop=True)


def train_m1(cfg: dict) -> tuple:
    df = pd.read_parquet(FEATURES_DIR / FEATURE_FILES["m1"])
    train, hold = time_split(df, "hour_start", cfg["split"])
    model = HistGradientBoostingClassifier(
        early_stopping=False, random_state=cfg["seed"], **cfg["m1"]["model"]
    ).fit(train[M1_FEATURES], train[M1_LABEL])

    hour_rate = train.groupby("hour_local")[M1_LABEL].mean()
    predict = {
        "trivial": lambda d: d["hour_local"].map(hour_rate).to_numpy(),  # start rate by hour
        "rule": lambda d: d["kyiv_rate_same_hour_30d"].to_numpy(),  # same hour, last 30 days
        "model": lambda d: model.predict_proba(d[M1_FEATURES])[:, 1],
    }
    tune = train[train["hour_start"] >= _ts(cfg["split"]["threshold_window_start"])]
    results = {}
    for name, fn in predict.items():
        p = fn(hold)
        scores = metrics.m1_scores(hold[M1_LABEL].to_numpy(), p)
        thr = metrics.best_threshold(tune["hour_start"], tune[M1_LABEL].to_numpy(), fn(tune))
        scores |= metrics.f1_tolerant(hold["hour_start"], hold[M1_LABEL].to_numpy(), p >= thr)
        scores["threshold"] = thr
        results |= {f"holdout_{k}_{name}": v for k, v in scores.items()}
    results |= {
        "n_train": len(train),
        "n_holdout": len(hold),
        "holdout_positive_rate": float(hold[M1_LABEL].mean()),
    }
    data = {"train": train[[*M1_FEATURES, M1_LABEL]], "holdout": hold[[*M1_FEATURES, M1_LABEL]]}
    return model, results, train[M1_FEATURES].head(5).astype("float64"), data


def train_m2(cfg: dict) -> tuple:
    df = pd.read_parquet(FEATURES_DIR / FEATURE_FILES["m2"])
    df = df[df["label_valid"]]
    train, hold = time_split(df, "alert_start", cfg["split"])
    model = QuantilePair(params=cfg["m2"]["model"], random_state=cfg["seed"])
    model.fit(train[M2_FEATURES], train[M2_LABEL])

    window = _ts(cfg["split"]["holdout_start"]) - timedelta(days=cfg["m2"]["rule_window_days"])
    recent = train.loc[train["alert_start"] >= window, M2_LABEL]
    rule = (float(recent.quantile(0.5)), float(recent.quantile(0.9)))
    y = hold[M2_LABEL].to_numpy()
    pred = model.predict(hold[M2_FEATURES])
    results = {}
    for name, (p50, p90) in {
        "rule": (np.full(len(y), rule[0]), np.full(len(y), rule[1])),  # last-12-months P50/P90
        "model": (pred[:, 0], pred[:, 1]),
    }.items():
        results |= {f"holdout_{k}_{name}": v for k, v in metrics.m2_scores(y, p50, p90).items()}
    results |= {
        "n_train": len(train),
        "n_holdout": len(hold),
        "rule_p50_min": rule[0],
        "rule_p90_min": rule[1],
    }
    data = {"train": train[[*M2_FEATURES, M2_LABEL]], "holdout": hold[[*M2_FEATURES, M2_LABEL]]}
    return model, results, train[M2_FEATURES].head(5).astype("float64"), data


def lineage_tags(allow_dirty: bool) -> dict:
    dirty = lineage.git_dirty()
    if dirty and not allow_dirty:
        raise SystemExit("uncommitted changes: commit first, so git_commit identifies the code")
    if not lineage.dvc_in_sync(FEATURES_DVC):
        raise SystemExit("data/features differs from data/features.dvc: run `make snapshot`")
    return {
        "git_commit": lineage.git_commit(),
        "git_dirty": str(dirty).lower(),
        "data_version": lineage.dvc_md5(FEATURES_DVC),
        "raw_data_version": lineage.dvc_md5(RAW_DVC),
        "feature_version": FEATURE_VERSION,
    }


def _params(cfg: dict, family: str) -> dict:
    flat = {"seed": cfg["seed"], **{f"split.{k}": v for k, v in cfg["split"].items()}}
    for k, v in cfg[family].items():
        if isinstance(v, dict):
            flat |= {f"{family}.{k}.{kk}": vv for kk, vv in v.items()}
        else:
            flat[f"{family}.{k}"] = v
    flat["features"] = ",".join(M1_FEATURES if family == "m1" else M2_FEATURES)
    return flat


def run_family(family: str, cfg: dict, tags: dict, args) -> dict:
    name = cfg[family]["registered_model"]
    label = M1_LABEL if family == "m1" else M2_LABEL
    model, results, example, data = (train_m1 if family == "m1" else train_m2)(cfg)

    run_tags = {**tags, "model_family": family}
    if args.reproduces:
        run_tags["reproduces"] = args.reproduces
    # M1 pyfunc returns [P(no start), P(start)], not a class label.
    log_kwargs = {"skops_trusted_types": TRUSTED_TYPES, "pyfunc_predict_fn": "predict_proba"}
    if family == "m2":
        log_kwargs = {
            "code_paths": [str(Path(__file__).parents[1])],  # the whole kyiv_alerts package
            "skops_trusted_types": [*TRUSTED_TYPES, "kyiv_alerts.training.models.QuantilePair"],
        }
    output = model.predict_proba(example) if family == "m1" else model.predict(example)

    with TrackedRun(name, f"{family}-{tags['git_commit'][:7]}", run_tags) as run:
        run.log_params(_params(cfg, family))
        source = f"data/features/{FEATURE_FILES[family]}"
        run.log_dataset(data["train"], f"{family}_train", "training", source, label)
        run.log_dataset(data["holdout"], f"{family}_holdout", "evaluation", source, label)
        run.log_metrics(results)
        run.log_dict(cfg, "config/train.yaml")
        version = run.log_sklearn_model(
            model,
            registered_name=None if args.no_register else name,
            signature=infer_signature(example, output),
            input_example=example,
            **log_kwargs,
        )
        if version is not None:
            promote = args.champion or not has_alias(name, CHAMPION)
            run.register(
                name,
                version,
                version_tags={k: tags[k] for k in LINEAGE_VERSION_TAGS},
                alias=CHAMPION if promote else None,
            )
    summary = {"family": family, "run_id": run.run_id, "registered_model": name,
               "version": version, "metrics": results}  # fmt: skip
    print(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=["m1", "m2", "all"], default="all")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "train.yaml"))
    ap.add_argument("--no-register", action="store_true", help="log the run only")
    ap.add_argument("--champion", action="store_true", help="move the champion alias here")
    ap.add_argument("--allow-dirty", action="store_true", help="allow uncommitted changes")
    ap.add_argument("--reproduces", default="", help="run id this run reproduces (tag)")
    ap.add_argument("--result-file", default="", help="write the run summary as JSON")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    tags = lineage_tags(args.allow_dirty)
    families = ["m1", "m2"] if args.model == "all" else [args.model]
    summaries = [run_family(f, cfg, tags, args) for f in families]
    if args.result_file:
        Path(args.result_file).write_text(json.dumps(summaries, indent=2, default=str))


if __name__ == "__main__":
    main()
