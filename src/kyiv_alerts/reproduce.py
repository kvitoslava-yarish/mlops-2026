"""Reproduce a registered model version from its two lineage tags.

    uv run python -m kyiv_alerts.reproduce --model kyiv-m1-start-risk --alias champion

1. reads ``git_commit`` and ``data_version`` from the model version;
2. checks out that commit in a temporary git worktree (your working tree is untouched);
3. checks that the commit's data/features.dvc has hash ``data_version`` and pulls it from
   the DVC remote;
4. runs that commit's training (``--no-register``, tagged ``reproduces=<run id>``);
5. compares the main metric with the registered run, within ``TOLERANCE``.
Exit code 0 = reproduced, 1 = metric outside tolerance.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from kyiv_alerts.config import REPO_ROOT
from kyiv_alerts.lineage import dvc_md5
from kyiv_alerts.train import MAIN_METRIC

# Same machine and lock file give identical numbers; the tolerance absorbs float
# differences across CPUs / BLAS builds (README, "Reproduce a registered run").
TOLERANCE = {"m1": 1e-4, "m2": 0.1}


def _run(cmd: list[str], cwd: Path) -> None:
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def main() -> int:
    ap = argparse.ArgumentParser(description="Reproduce a registered model version")
    ap.add_argument("--model", required=True, help="registered model name")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--alias", default="champion")
    group.add_argument("--version")
    ap.add_argument("--keep", action="store_true", help="keep the temporary worktree")
    args = ap.parse_args()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    client = MlflowClient()
    mv = (
        client.get_model_version(args.model, args.version)
        if args.version
        else client.get_model_version_by_alias(args.model, args.alias)
    )
    commit, data_version = mv.tags["git_commit"], mv.tags["data_version"]
    original = client.get_run(mv.run_id)
    family = original.data.tags["model_family"]
    metric = MAIN_METRIC[family]
    expected = original.data.metrics[metric]
    print(f"{args.model} v{mv.version}: run {mv.run_id}, commit {commit}, data {data_version}")

    workdir = Path(tempfile.mkdtemp(prefix="kyiv-repro-"))
    tree = workdir / "repo"
    _run(["git", "worktree", "add", "--detach", str(tree), commit], cwd=REPO_ROOT)
    try:
        found = dvc_md5(tree / "data" / "features.dvc")
        if found != data_version:
            raise SystemExit(f"commit {commit} tracks data {found}, expected {data_version}")
        local_cfg = REPO_ROOT / ".dvc" / "config.local"
        if local_cfg.exists():
            shutil.copy(local_cfg, tree / ".dvc" / "config.local")
        _run(["dvc", "pull", "data/features.dvc"], cwd=tree)
        result = workdir / "result.json"
        _run(
            ["uv", "run", "--frozen", "python", "-m", "kyiv_alerts.train", "--model", family,
             "--no-register", "--reproduces", mv.run_id, "--result-file", str(result)],
            cwd=tree,
        )  # fmt: skip
        summary = json.loads(result.read_text())[0]
    finally:
        if not args.keep:
            _run(["git", "worktree", "remove", "--force", str(tree)], cwd=REPO_ROOT)
            shutil.rmtree(workdir, ignore_errors=True)

    got = summary["metrics"][metric]
    diff = abs(got - expected)
    ok = diff <= TOLERANCE[family]
    print(
        f"\n{metric}: registered {expected:.6f}, reproduced {got:.6f}, "
        f"|diff| {diff:.2e} (tolerance {TOLERANCE[family]}) -> {'PASS' if ok else 'FAIL'}\n"
        f"reproduction run: {summary['run_id']}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
