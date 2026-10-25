import subprocess
from pathlib import Path

import yaml

from kyiv_alerts.config import REPO_ROOT


def _git(*args: str, cwd: Path = REPO_ROOT) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def git_commit(cwd: Path = REPO_ROOT) -> str:
    return _git("rev-parse", "HEAD", cwd=cwd)


def git_dirty(cwd: Path = REPO_ROOT) -> bool:
    """Uncommitted changes to tracked files (untracked files do not change the code)."""
    return bool(_git("status", "--porcelain", "--untracked-files=no", cwd=cwd))


def dvc_md5(dvc_file: Path) -> str:
    """The hash DVC recorded for a tracked file or directory (``…​.dir`` for directories)."""
    meta = yaml.safe_load(dvc_file.read_text())
    return meta["outs"][0]["md5"]


def dvc_in_sync(target: Path, cwd: Path = REPO_ROOT) -> bool:
    """True if the workspace data matches its .dvc file (nothing modified or missing)."""
    out = subprocess.run(
        ["dvc", "status", "--json", str(target)], cwd=cwd, check=True,
        capture_output=True, text=True,
    ).stdout  # fmt: skip
    return out.strip() in ("{}", "")
