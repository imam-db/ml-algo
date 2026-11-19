"""Smoke tests for every module's ``sklearn_example.py`` script.

Each script is executed in an isolated subprocess with a non-interactive
matplotlib backend (``MPLBACKEND=Agg``) and a temporary working directory so
that any plots written by the examples do not dirty the repository.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Scripts that take noticeably longer (>20s) and are skipped on pull requests
# when pytest is invoked with ``-m "not slow"``.
SLOW_SCRIPTS: set[str] = {
    "01_supervised_learning/04_random_forest/sklearn_example.py",
    "01_supervised_learning/09_gradient_boosting/sklearn_example.py",
    "02_unsupervised_learning/05_tsne/sklearn_example.py",
    "04_advanced_topics/07_object_detection/sklearn_example.py",
}


def _discover_sklearn_examples() -> list[pytest.param]:
    """Discover all ``*/*/sklearn_example.py`` scripts relative to the repo root."""
    scripts = sorted(REPO_ROOT.glob("*/*/sklearn_example.py"))
    params = []
    for script in scripts:
        rel_path = script.relative_to(REPO_ROOT).as_posix()
        marks = [pytest.mark.slow] if rel_path in SLOW_SCRIPTS else []
        params.append(pytest.param(script, id=rel_path, marks=marks))
    return params


SKLEARN_EXAMPLE_PARAMS = _discover_sklearn_examples()


def test_sklearn_examples_discovered() -> None:
    """Guard against the glob silently matching zero scripts."""
    assert len(SKLEARN_EXAMPLE_PARAMS) >= 18


@pytest.mark.parametrize("script_path", SKLEARN_EXAMPLE_PARAMS)
def test_sklearn_example_runs(script_path: Path, tmp_path: Path) -> None:
    """Verify that each ``sklearn_example.py`` runs cleanly to completion."""
    rel_parent = script_path.parent.relative_to(REPO_ROOT)
    # Mirror the module's relative path inside tmp_path so scripts that call
    # plt.savefig("<category>/<module>/plot.png") succeed without writing into
    # the repository working tree.
    (tmp_path / rel_parent).mkdir(parents=True, exist_ok=True)

    # Track pre-existing PNGs in the script's own folder so any plot written via
    # Path(__file__).parent (e.g. t-SNE) is cleaned up after the smoke test.
    existing_pngs = set(script_path.parent.glob("*.png"))

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONIOENCODING"] = "utf-8"

    rel_path = script_path.relative_to(REPO_ROOT).as_posix()
    timeout = 600 if rel_path in SLOW_SCRIPTS else 180

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    finally:
        for png in script_path.parent.glob("*.png"):
            if png not in existing_pngs:
                png.unlink(missing_ok=True)

    assert result.returncode == 0, (
        f"{script_path.relative_to(REPO_ROOT).as_posix()} exited with code "
        f"{result.returncode}\n"
        f"--- stdout ---\n{result.stdout[-2000:]}\n"
        f"--- stderr ---\n{result.stderr[-2000:]}"
    )
