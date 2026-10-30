"""W2 1b7f scientific and declared-dependency sandbox acceptance.

The ordinal numbering/gap correction is deliberately reserved for W8.  These
oracles use the accepted 1b7f numbering and do not claim gapped-input parity.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "tests" / "1b7f"
RNA = HERE / "pdbs" / "rna.pdb"
FIT = HERE / "data" / "refe-best-fit.tsv"
LOCATOR = Path.home() / ".alaric" / "fraglib.yaml"


def _clean_env():
    env = dict(os.environ)
    for name in ("PYTHONPATH", "SEAMLESS_CACHE", "SEAMLESS_DATABASE", "SEAMLESS_HASHSERVER"):
        env.pop(name, None)
    return env


def _identity():
    from util.refe_best_fit_seamless_support import identities
    return json.dumps(identities(ROOT / "fraglib"), sort_keys=True, separators=(",", ":"))


def _sandbox(deps, script, arguments, output, *, inputs=(), library=False, identity=None, succeeds=True,
             output_name="w2-output", folder=True):
    """Use precisely the CLI's declared file list with flat sandbox mapping."""
    runner = ROOT / "tests" / "w2_sandbox_runner.py"
    command = ["seamless-run", "--local", "-y", "-W", "-ms", "-g1", "-I", str(ROOT / "util" / deps),
               "-i", str(runner)]
    for path in inputs:
        command.extend(["-i", str(path)])
    if library:
        command.extend(["--metafile", str(LOCATOR), "--var", f"fraglib_identities={identity if identity is not None else _identity()}"])
    command.extend(["--capture", f"{output_name}:{output}", "python", "-I", "-B", str(runner),
                    f"forbid={ROOT}", str(ROOT / "util" / script), *map(str, arguments)])
    # `seamless-run` requires project configuration even for `--local`. Give
    # each invocation an isolated local profile with no persistent cache or
    # services, instead of depending on a developer profile or SEAMLESS_CACHE.
    workdir = output.parent / "seamless-workdir"
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "seamless.profile.yaml").write_text(
        "- clusters:\n"
        "    local_cluster: w2-local\n"
        "    w2-local:\n"
        "      type: local\n"
        "      workers: 8\n"
        "      frontends: []\n"
        "- execution: spawn\n"
        "- persistent: false\n",
        encoding="utf-8",
    )
    completed = subprocess.run(command, cwd=workdir, env=_clean_env(), capture_output=True, text=True, timeout=600)
    if not succeeds:
        assert completed.returncode != 0, completed.stdout + completed.stderr
        assert "schema" in (completed.stdout + completed.stderr).lower()
        assert not (output / "pairs.tsv").exists()
        assert not (output / "bases.tsv").exists()
        return output
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (output.is_dir() if folder else output.is_file()), completed.stdout + completed.stderr
    return output


def _pdb_atoms(path):
    atoms = []
    for line in path.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            atoms.append((line[12:16].strip(), line[17:20].strip(), line[21:22], int(line[22:26]),
                          tuple(float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54)))))
    return atoms


@pytest.fixture(scope="module")
def reference_folder(tmp_path_factory):
    output = tmp_path_factory.mktemp("w2-refe") / "result"
    return _sandbox("refe-fragments.DEPS", "refe-fragments.py",
                    [str(RNA), "--templates=.", "--output", "w2-output"],
                    output, inputs=[RNA])


def test_a2_declared_sandbox_matches_a1_keys_atoms_and_in_place_rmsd(reference_folder):
    fitted = {path.name.removesuffix("-bestfit.pdb"): path for path in (HERE / "best-fits").glob("*-bestfit.pdb")}
    reference = {path.name.removesuffix("-refe.pdb"): path for path in reference_folder.glob("*-refe.pdb")}
    assert set(reference) == set(fitted)
    rows = {f"frag-{row['fragment']}-{row['sequence']}": row
            for row in csv.DictReader(io.StringIO(FIT.read_text()), delimiter="\t")}
    assert set(rows) == set(reference)
    for key in sorted(reference):
        refe_atoms, fit_atoms = _pdb_atoms(reference[key]), _pdb_atoms(fitted[key])
        assert [row[:2] for row in refe_atoms] == [row[:2] for row in fit_atoms]
        assert refe_atoms and len(refe_atoms) == len(fit_atoms)
        rmsd = math.sqrt(sum(sum((x - y) ** 2 for x, y in zip(a[4], b[4]))
                             for a, b in zip(refe_atoms, fit_atoms)) / len(refe_atoms))
        assert abs(rmsd - float(rows[key]["rmsd"])) <= 1e-4, (key, rmsd, rows[key]["rmsd"])


@pytest.fixture(scope="module")
def table_folder(tmp_path_factory):
    output = tmp_path_factory.mktemp("w2-tables") / "result"
    return _sandbox("refe-best-fit-tables.DEPS", "refe-best-fit-tables.py",
                    [str(FIT), str(RNA), "--exclude-pdb-code", "1b7f", "--templates-dir=.",
                     "--fraglib-identities", "fraglib_identities", "--fraglib-config", str(LOCATOR),
                     "--output", "w2-output"], output, inputs=[FIT, RNA], library=True)


def test_a3_declared_sandbox_reproduces_exact_library_coordinate_goldens(table_folder):
    assert (table_folder / "pairs.tsv").read_bytes() == (HERE / "data" / "refe-best-fit-pairs.tsv").read_bytes()
    assert (table_folder / "bases.tsv").read_bytes() == (HERE / "data" / "refe-best-fit-bases.tsv").read_bytes()


def test_a3_subset_is_filtered_full_output(tmp_path, table_folder):
    lines = FIT.read_text().splitlines(keepends=True)
    kept = tmp_path / "kept.tsv"
    kept.write_text(lines[0] + "".join(line for line in lines[1:] if 3 <= int(line.split()[0]) <= 8))
    output = _sandbox("refe-best-fit-tables.DEPS", "refe-best-fit-tables.py",
                      [str(kept), str(RNA), "--exclude-pdb-code", "1b7f", "--templates-dir=.",
                       "--fraglib-identities", "fraglib_identities", "--fraglib-config", str(LOCATOR),
                       "--output", "w2-output"], tmp_path / "result", inputs=[kept, RNA], library=True)
    for filename, fields in (("pairs.tsv", ("down", "up")), ("bases.tsv", ("fragment",))):
        full = (table_folder / filename).read_text().splitlines(keepends=True)
        header = full[0].rstrip().split("\t")
        indices = [header.index(field) for field in fields]
        expected = full[0] + "".join(line for line in full[1:]
                                    if all(3 <= int(line.split("\t")[index]) <= 8 for index in indices))
        assert (output / filename).read_text() == expected


@pytest.mark.parametrize("schema", [None, 2])
def test_a3_rejects_unknown_or_missing_identity_schema_without_partial_output(tmp_path, schema):
    identity = json.loads(_identity())
    if schema is None:
        del identity["schema"]
    else:
        identity["schema"] = schema
    _sandbox("refe-best-fit-tables.DEPS", "refe-best-fit-tables.py",
             [str(FIT), str(RNA), "--exclude-pdb-code", "1b7f", "--templates-dir=.",
              "--fraglib-identities", "fraglib_identities", "--fraglib-config", str(LOCATOR),
              "--output", "w2-output"], tmp_path / "result", inputs=[FIT, RNA], library=True,
             identity=json.dumps(identity, sort_keys=True, separators=(",", ":")), succeeds=False)


def test_a4_runs_with_only_declared_standard_library_code_in_sandbox(tmp_path):
    output = _sandbox("refe-thresholds.DEPS", "refe-thresholds.py",
                      [str(HERE / "data" / "refe-best-fit-pairs.tsv"),
                       str(HERE / "data" / "refe-best-fit-bases.tsv"),
                       "--output", "w2-output.json"], tmp_path / "thresholds.json",
                      inputs=[HERE / "data" / "refe-best-fit-pairs.tsv", HERE / "data" / "refe-best-fit-bases.tsv"],
                      output_name="w2-output.json", folder=False)
    result = json.loads(output.read_text())
    assert result["schema"] == 1 and len(result["pairs"]) == 10 and len(result["bases"]) == 11
    assert result["pairs"][5]["ovrmsd"] == "0.52"
    assert result["bases"][3]["nuc2_ovrmsd"] == "0.61"
    assert result["bases"][3]["nuc2_ovrmsd_raw"] == 0.603997


def test_a6_runs_as_staged_standard_library_transformation():
    from seamless_transformer import Transformer
    from seamless_config.select import get_execution, get_persistent, select_execution, select_persistent
    from seamless_transformer.worker import has_spawned, spawn

    previous = get_execution(), get_persistent()
    select_execution("process")
    select_persistent(False)
    if not has_spawned():
        spawn(8)
    try:
        step = Transformer("bash")
        step.code = (
            "python -I -c \"import sys;sys.path.insert(0,'.');"
            "from energy_threshold import energy_threshold;"
            "open('RESULT','w').write(energy_threshold(float(open('raw').read())))\""
        )
        for filename in ("energy_threshold.py", "threshold_rounding.py"):
            step.celltypes[filename] = "bytes"
            step.args[filename] = (ROOT / "alaric" / filename).read_bytes()
        step.celltypes.raw = "str"
        step.args.raw = "-16.59"
        step.celltypes.result = "text"
        assert step.run() == "-16.58"
    finally:
        select_execution(previous[0])
        select_persistent(previous[1])


def _run_a1_factory(tmp_path):
    import seamless
    from seamless_config.select import get_execution, get_persistent, select_execution, select_persistent
    from seamless_transformer.worker import has_spawned, spawn
    from seamless.util.mount_directory import write_to_directory
    from util.seamlessify_refe_best_fit import seamlessify_refe_best_fit

    previous = get_execution(), get_persistent()
    select_execution("process")
    select_persistent(False)
    if not has_spawned():
        spawn(8)
    try:
        transformer = seamlessify_refe_best_fit(fraglib_config=LOCATOR, conda_environment="alaric")
        transformer.args.rna_pdb = RNA.read_bytes()
        transformer.args.exclude_pdb_code = "1b7f"
        transformer.args.fraglib_identities = _identity()
        result = transformer.run()
        output = tmp_path / "result"
        write_to_directory(str(output), result, cleanup=False, deep=True, text_only=False)
        assert (output / "best-fits.tsv").read_bytes() == FIT.read_bytes()
        assert (output / "best-fits.tsv").read_text().splitlines()[0].split() == [
            "fragment", "sequence", "conformer", "rotamer", "grid_x", "grid_y", "grid_z", "rmsd"]
        expected = {path.name: path for path in (HERE / "best-fits").glob("*-bestfit.pdb")}
        actual = {path.name: path for path in (output / "pdbs").glob("*.pdb")}
        assert set(actual) == set(expected)
        for name, path in expected.items():
            assert actual[name].read_bytes() == path.read_bytes()
            atoms = _pdb_atoms(actual[name])
            assert {row[2] for row in atoms} == {"B"}
            assert {row[3] for row in atoms} == {1, 2}
        original = transformer.build()
        original.construct()
        original_identity = original.transformation_checksum
        relocated = tmp_path / "relocated-locator.yaml"
        relocated.write_text(LOCATOR.read_text().replace("/opt/fraglib", "/relocated/immutable/fraglib"))
        other = seamlessify_refe_best_fit(fraglib_config=relocated, conda_environment="alaric")
        other.args.rna_pdb = RNA.read_bytes()
        other.args.exclude_pdb_code = "1b7f"
        other.args.fraglib_identities = _identity()
        relocated = other.build()
        relocated.construct()
        assert relocated.transformation_checksum == original_identity
        changed = json.loads(_identity())
        first = next(iter(changed["sha256"]))
        changed["sha256"][first] = "f" * 64
        other.args.fraglib_identities = json.dumps(changed, sort_keys=True, separators=(",", ":"))
        changed_transformation = other.build()
        changed_transformation.construct()
        assert changed_transformation.transformation_checksum != original_identity
    finally:
        seamless.close()
        select_execution(previous[0])
        select_persistent(previous[1])


def test_a1_factory_runs_on_eight_spawn_workers_with_connectable_inputs(tmp_path):
    # The factory test exercises Seamless' process-global context and closes it
    # during cleanup. Run it in a child so later tests keep the parent's
    # initialized context intact.
    code = (
        "import runpy, sys; from pathlib import Path; "
        "runpy.run_path(sys.argv[1])['_run_a1_factory'](Path(sys.argv[2]))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code, str(Path(__file__).resolve()), str(tmp_path)],
        cwd=ROOT,
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
