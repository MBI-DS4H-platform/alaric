"""W4/A5 contract oracles. Run in both alaric and preparic environments.

These tests deliberately use the required CLI rather than an optional factory.
The test launcher is orchestration; scientific code is limited to the eight
declared files. No result is accepted from the CLI's float32 npy output.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "util" / "energetics_seamless_support.py"
DEPS = SUPPORT.with_name(SUPPORT.name + ".DEPS.txt")
RECEPTOR = ROOT / "tests/1b7f/pdbs/1b7f_dom1-aa.pdb"
LIGAND = ROOT / "tests/1b7f/best-fits/frag-1-GU-bestfit.pdb"
EXPECTED_DEPS = [
    "energetics_seamless_support.py",
    "../alaric/parse_pdb.py",
    "reduce-npy.py",
    "../alaric/reduce.dat",
    "../attract-jax/util/jax_scorer.py",
    "../attract-jax/util/numpy_grid_score.py",
    "../attract-jax/util/pdb_io.py",
    "../attract-jax/attract-original/attract-par.npz",
]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _paths():
    assert DEPS.is_file(), f"missing A5 declaration: {DEPS}"
    return [(DEPS.parent / item).resolve() for item in DEPS.read_text().splitlines()]


def _env():
    result = dict(os.environ)
    for key in ("PYTHONPATH", "SEAMLESS_CACHE", "SEAMLESS_DATABASE", "SEAMLESS_HASHSERVER"):
        result.pop(key, None)
    return result


def _unknown_residue(text):
    return "".join(line[:17] + "XXX" + line[20:] if line.startswith(("ATOM  ", "HETATM")) else line
                   for line in text.splitlines(keepends=True))


def _stage(tmp_path, *, bad_receptor=False, count_reductions=False):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    for path in _paths():
        shutil.copyfile(path, sandbox / path.name)
    shutil.copyfile(ROOT / "tests/w4_sandbox_runner.py", sandbox / "runner.py")
    receptor = sandbox / "receptor.pdb"
    text = RECEPTOR.read_text()
    receptor.write_text(_unknown_residue(text) if bad_receptor else text)
    ligands = sandbox / "ligands"
    ligands.mkdir()
    shutil.copyfile(LIGAND, ligands / LIGAND.name)
    (ligands / "unknown.pdb").write_text(_unknown_residue(LIGAND.read_text()))
    if count_reductions:
        reducer = sandbox / "reduce-npy.py"
        source = reducer.read_text()
        # Observe the actual declared reduction routine, including failed calls.
        source, replacements = re.subn(
            r"(def reduce_ppdb\([^\n]+:\n)",
            r'\1    if os.environ.get("W4_REDUCE_CALLS"):\n'
            r'        with open(os.environ["W4_REDUCE_CALLS"], "a") as stream:\n'
            r'            stream.write("reduce\\n")\n',
            source,
        )
        assert replacements == 1, "update test instrumentation for the declared reducer API"
        reducer.write_text(source)
    return sandbox, ligands, receptor


def _run_flat(sandbox, ligands, receptor, *, count=False):
    env = _env()
    if count:
        env["W4_REDUCE_CALLS"] = str(sandbox / "reduction-calls.txt")
    return subprocess.run(
        [sys.executable, "-I", "-B", str(sandbox / "runner.py"), f"forbid={ROOT}",
         str(sandbox / SUPPORT.name), "score", str(ligands), str(receptor)],
        cwd=sandbox, env=env, capture_output=True, text=True, timeout=120,
    )


def _reference_energy(tmp_path):
    import numpy as np

    parser = _load(ROOT / "alaric/parse_pdb.py", "w4_reference_parse_pdb")
    reducer = _load(ROOT / "util/reduce-npy.py", "w4_reference_reducer")
    rules = reducer.read_forcefield(ROOT / "alaric/reduce.dat")
    counts = {}
    for name, path in (("receptor", RECEPTOR), ("ligand", LIGAND)):
        coords, types = reducer.reduce_ppdb(parser.parse_pdb(path.read_text()), rules)
        counts[name] = len(coords)
        np.save(tmp_path / f"{name}-coor.npy", coords)
        np.save(tmp_path / f"{name}-types.npy", types)
    result = subprocess.run([
        sys.executable, "-B", str(ROOT / "attract-jax/util/minfor.py"),
        "--oracle", "numpy", "--no-grid", "--identity", "--score", "--energy-only",
        "--epsilon", "15", "--attract-par-npz", str(ROOT / "attract-jax/attract-original/attract-par.npz"),
        "--receptor-coordinates", str(tmp_path / "receptor-coor.npy"),
        "--receptor-atomtypes", str(tmp_path / "receptor-types.npy"),
        "--ligand-ensemble", str(tmp_path / "ligand-coor.npy"),
        "--ligand-atomtypes", str(tmp_path / "ligand-types.npy"),
    ], cwd=tmp_path, env=_env(), capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    values = re.findall(r"^\s*Energy:\s*([-+0-9.eE]+)", result.stdout, re.MULTILINE)
    assert len(values) == 1, result.stdout + result.stderr
    return float(values[0]), counts


def _assert_result(result, reference, counts):
    import numpy as np

    assert set(result) == {"schema", "epsilon", "receptor", "ligands", "pairs"}
    assert type(result["schema"]) is int and result["schema"] == 1
    assert result["epsilon"] == 15
    assert result["receptor"] == {"beads": counts["receptor"]}
    assert LIGAND.name in result["ligands"]
    assert set(result["ligands"]) <= {LIGAND.name, "unknown.pdb"}
    assert result["ligands"][LIGAND.name] == {"beads": counts["ligand"]}
    rows = {row["ligand"]: row for row in result["pairs"]}
    assert len(rows) == len(result["pairs"]) == 2
    assert set(rows) == {LIGAND.name, "unknown.pdb"}
    good, bad = rows[LIGAND.name], rows["unknown.pdb"]
    assert not good["error"]
    assert type(good["energy"]) is float and math.isfinite(good["energy"])
    assert abs(good["energy"] - reference) <= 1e-6
    # Preserve the Python float64 result instead of the CLI's float32 npy value.
    assert abs(good["energy"] - reference) <= 1e-10
    if reference != float(np.float32(reference)):
        assert good["energy"] != float(np.float32(reference))
    assert isinstance(bad["error"], str) and bad["error"]
    assert "energy" not in bad or bad["energy"] is None
    assert not any(key in result for key in ("range", "range_index", "range_id"))


def test_exact_eight_dependencies_are_existing_unique_and_have_no_checksum_sidecars():
    assert DEPS.read_text().splitlines() == EXPECTED_DEPS
    paths = _paths()
    assert len({path.name for path in paths}) == 8
    for path in paths:
        assert path.is_file(), path
        assert not Path(str(path) + ".CHECKSUM").exists(), path
    full = ROOT / "attract-jax/util/minfor.py.DEPS.txt"
    full_paths = {(full.parent / item).resolve() for item in full.read_text().splitlines()}
    assert {path for path in paths if ROOT / "attract-jax" in path.parents} <= full_paths


def test_input_paths_reads_the_declared_list():
    support = _load(SUPPORT, "w4_support_manifest")
    assert [Path(path).resolve() for path in support.input_paths()] == _paths()


def test_t13_factory_stages_exact_eight_a5_dependencies():
    """Exercise the real T13 factory and its nested-submodule staging paths."""
    preparic_root = ROOT.parent / "preparic"
    runtime_path = preparic_root / "alaric_runtime.py"
    if not runtime_path.is_file():
        pytest.skip("Preparic runtime checkout is unavailable")
    try:
        importlib.util.find_spec("seamless_transformer")
    except ModuleNotFoundError:
        pytest.skip("The Preparic Seamless transformer package is unavailable")
    if importlib.util.find_spec("seamless_transformer") is None:
        pytest.skip("The Preparic Seamless transformer package is unavailable")

    sys.path.insert(0, str(preparic_root))
    try:
        runtime = _load(runtime_path, "w45_alaric_runtime")
        transformer = runtime.energetics_range_transformer(ROOT / "alaric")
    finally:
        sys.path.remove(str(preparic_root))

    dependencies = _paths()
    assert len(dependencies) == 8
    assert len({path.name for path in dependencies}) == 8
    staged_code = str(transformer.code)
    for path in dependencies:
        relative = path.relative_to(ROOT).as_posix()
        assert f"cp {path.name} work/{relative}" in staged_code
        assert transformer.args[path.name].checksum is not None
        assert transformer.celltypes[path.name] == "bytes"
    assert transformer.celltypes["ligands"] == "folder"
    assert transformer.celltypes["receptor"] == "bytes"
    assert "energetics_seamless_support.py.DEPS.txt" not in staged_code
    assert "work/attract-jax/util/" in staged_code


def test_flat_declared_sandbox_float64_identity_and_row_local_unknown_error(tmp_path):
    reference, counts = _reference_energy(tmp_path)
    sandbox, ligands, receptor = _stage(tmp_path, count_reductions=True)
    completed = _run_flat(sandbox, ligands, receptor, count=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    _assert_result(json.loads((sandbox / "energies.json").read_text()), reference, counts)
    assert (sandbox / "reduction-calls.txt").read_text().splitlines() == ["reduce"] * 3


def test_receptor_reduction_failure_fails_the_whole_invocation(tmp_path):
    sandbox, ligands, receptor = _stage(tmp_path, bad_receptor=True)
    completed = _run_flat(sandbox, ligands, receptor)
    assert completed.returncode != 0, completed.stdout + completed.stderr
    assert not (sandbox / "energies.json").exists(), "failed receptor published a partial success result"


def test_declared_seamless_sandbox_matches_identity_cli(tmp_path):
    reference, counts = _reference_energy(tmp_path)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    ligands = inputs / "ligands"
    ligands.mkdir()
    shutil.copyfile(LIGAND, ligands / LIGAND.name)
    (ligands / "unknown.pdb").write_text(_unknown_residue(LIGAND.read_text()))
    output = tmp_path / "energies.json"
    work = tmp_path / "work"
    work.mkdir()
    (work / "seamless.profile.yaml").write_text(
        "- clusters:\n    local_cluster: w4-local\n    w4-local:\n"
        "      type: local\n      workers: 8\n      frontends: []\n"
        "- execution: spawn\n- persistent: false\n")
    runner = ROOT / "tests/w4_sandbox_runner.py"
    command = ["seamless-run", "--local", "-y", "-W", "-ms", "-g1", "-I", str(DEPS),
               "-i", str(runner), "-i", str(ligands), "-i", str(RECEPTOR),
               "--capture", f"energies.json:{output}", "python", "-I", "-B", str(runner),
               f"forbid={ROOT}", str(SUPPORT), "score", str(ligands), str(RECEPTOR)]
    completed = subprocess.run(command, cwd=work, env=_env(), capture_output=True, text=True, timeout=240)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    _assert_result(json.loads(output.read_text()), reference, counts)


def test_numpy_prerequisite_and_accuracy_artifacts_are_committed_and_published():
    submodule = ROOT / "attract-jax"
    def git(*args):
        return subprocess.run(["git", "-C", str(submodule), *args], capture_output=True, text=True)
    for ancestor in ("9453ee6", "22b23e4"):
        completed = git("merge-base", "--is-ancestor", ancestor, "HEAD")
        assert completed.returncode == 0, completed.stderr
    names = ["docs/numpy-grid-free-2048.csv", "docs/numpy-grid-free-2048-summary.json",
             "docs/numpy-grid-free-comparison.csv"]
    for name in names:
        committed = git("show", f"HEAD:{name}")
        assert committed.returncode == 0, committed.stderr
        assert committed.stdout == (submodule / name).read_text()
    assert len(list(csv.DictReader((submodule / names[0]).open()))) == 1221
    assert isinstance(json.loads((submodule / names[1]).read_text()), dict)
    assert len(list(csv.DictReader((submodule / names[2]).open()))) >= 4
    # A2 must fetch after publishing; an ancestor is sufficient when origin has
    # advanced further, and avoids requiring a force-push or exact branch tip.
    published = git("merge-base", "--is-ancestor", "HEAD", "origin/main")
    assert published.returncode == 0, "attract-jax HEAD has not been published to origin/main"
