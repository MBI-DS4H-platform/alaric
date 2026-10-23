"""Opt-in real-library regression for the standalone Seamless best-fit CLIs.

RUN_SEAMLESS_BEST_FIT_TESTS=1 pytest tests/1b7f/test_refe_best_fit_seamless.py
Requires Seamless and the configured external fragment library (several GB).
Run pytest in an environment without Alaric installed to check that contract.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_SEAMLESS_BEST_FIT_TESTS") != "1",
    reason="set RUN_SEAMLESS_BEST_FIT_TESTS=1 for real-library Seamless tests",
)


@pytest.mark.parametrize("wrapper", ["refe-best-fit-SEAMLESS.sh", "refe-best-fit-SEAMLESS.py"])
def test_standalone_wrapper_matches_goldens(tmp_path: Path, wrapper: str) -> None:
    output = tmp_path / "output with spaces"
    env = dict(os.environ, SEAMLESS_CACHE=str(tmp_path / "cache"))
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [str(ROOT / "util" / wrapper), "--output", str(output),
         str(HERE / "pdbs" / "rna.pdb"), "--exclude-pdb-code", "1b7f"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (output / "best-fits.tsv").read_bytes() == (HERE / "data" / "refe-best-fit.tsv").read_bytes()
    expected = sorted((HERE / "best-fits").glob("*.pdb"))
    actual = sorted((output / "pdbs").glob("*.pdb"))
    assert [p.name for p in actual] == [p.name for p in expected]
    for path in expected:
        assert (output / "pdbs" / path.name).read_bytes() == path.read_bytes()

    # Hints and environments are excluded from identity. Replay must succeed
    # from persistent cache even when these settings could not execute a job.
    hints = tmp_path / "cache-only-hints.yaml"
    hints.write_text("not a usable library configuration\n")
    replay = tmp_path / "cached output"
    cached = subprocess.run(
        [str(ROOT / "util" / wrapper), "--output", str(replay),
         "--fraglib-config", str(hints), "--conda", "deliberately-absent-cache-probe",
         str(HERE / "pdbs" / "rna.pdb"), "--exclude-pdb-code", "1b7f"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )
    assert cached.returncode == 0, cached.stdout + cached.stderr
    assert (replay / "best-fits.tsv").read_bytes() == (output / "best-fits.tsv").read_bytes()
    for path in expected:
        assert (replay / "pdbs" / path.name).read_bytes() == path.read_bytes()
