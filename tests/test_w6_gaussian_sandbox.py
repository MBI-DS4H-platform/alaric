"""Run the A9 CLI with only its declared dependencies in a Seamless sandbox."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from test_w6_gaussian_check import ROOT, _prepare_case
from alaric.gaussian_check import semantic_cache_identity
from alaric.gaussian_filter import load_spec


LOCATOR = Path.home() / ".alaric" / "fraglib.yaml"
RUNNER = ROOT / "tests" / "w2_sandbox_runner.py"
SCRIPT = ROOT / "util" / "gaussian-check.py"
DEPS = ROOT / "util" / "gaussian-check.py.DEPS.txt"


def _clean_env() -> dict[str, str]:
    env = dict(os.environ)
    for name in ("PYTHONPATH", "SEAMLESS_CACHE", "SEAMLESS_DATABASE", "SEAMLESS_HASHSERVER"):
        env.pop(name, None)
    return env


def test_a9_cli_runs_1b7f_job_with_declared_dependencies_in_seamless_sandbox(
    tmp_path: Path,
) -> None:
    if shutil.which("seamless-run") is None:
        pytest.skip("seamless-run is unavailable in this environment")
    assert DEPS.is_file(), f"missing declared dependency list: {DEPS}"
    assert SCRIPT.is_file(), f"missing A9 CLI entry point: {SCRIPT}"
    assert RUNNER.is_file()
    assert "gaussian_stacking_filter.json" not in DEPS.read_text(encoding="utf-8")

    best_fit, proteins, jobs, identities = _prepare_case(tmp_path / "case")
    (tmp_path / "case" / "jobs.json").write_text(json.dumps([jobs[0]]) + "\n")
    semantic_identity = json.dumps(
        semantic_cache_identity(load_spec()),
        sort_keys=True,
        separators=(",", ":"),
    )
    case = tmp_path / "case"
    output = case / "result.json"
    workdir = case / "seamless-workdir"
    workdir.mkdir()
    (workdir / "seamless.profile.yaml").write_text(
        "- clusters:\n"
        "    local_cluster: w6-local\n"
        "    w6-local:\n"
        "      type: local\n"
        "      workers: 8\n"
        "      frontends: []\n"
        "- execution: spawn\n"
        "- persistent: false\n",
        encoding="utf-8",
    )

    command = [
        "seamless-run",
        "--local",
        "-y",
        "-W",
        "-ms",
        "-g1",
        "-I",
        str(DEPS),
        "-i",
        str(RUNNER),
        "-i",
        str(case / "jobs.json"),
        "-i",
        str(best_fit),
        "-i",
        str(proteins),
        "--metafile",
        str(LOCATOR),
        "--metavar",
        f"repo_root={ROOT}",
        "--var",
        f"fraglib_identities={identities}",
        "--var",
        f"semantic_identity={semantic_identity}",
        "--capture",
        f"result.json:{output}",
        "python",
        "-I",
        "-B",
        str(RUNNER),
        f"forbid={ROOT}",
        str(SCRIPT),
        "--jobs",
        "jobs.json",
        "--best-fit",
        "best-fit",
        "--proteins",
        "proteins",
        "--exclude-code",
        "1b7f",
        "--fraglib-identities",
        "fraglib_identities",
        "--semantic-identity",
        "semantic_identity",
        "--output",
        "result.json",
    ]
    completed = subprocess.run(
        command,
        cwd=workdir,
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(output.read_text())
    assert result["schema"] == 1
    assert len(result["jobs"]) == 1
    assert result["jobs"][0]["key"] == jobs[0]["key"]
    assert "error" not in result["jobs"][0]
