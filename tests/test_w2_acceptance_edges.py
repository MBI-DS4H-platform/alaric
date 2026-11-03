"""Additional W2 oracles, independent of fitter/table numerical internals.

Run with: conda run --live-stream -n alaric python -m pytest -q
    tests/test_w2_acceptance_edges.py
These tests construct identities but never execute the expensive A1 fit.
"""
from __future__ import annotations

from decimal import Decimal
import importlib.util
import itertools
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
RNA = ROOT / "tests/1b7f/pdbs/rna.pdb"
TEMPLATES = {f"../fraglib/templates/{seq}-ppdb.npy"
             for length in (1, 2)
             for seq in map("".join, itertools.product("ACGU", repeat=length))}
CODE = {
    "refe-best-fit.DEPS": {
        "refe-best-fit.py", "refe_best_fit_seamless_support.py",
        *{f"../alaric/{name}.py" for name in
          ("library", "parse_pdb", "reference", "backbone", "superimpose", "write_pdb", "rna_pdb", "mutate")}},
    "refe-fragments.DEPS": {
        "refe-fragments.py", "../alaric/parse_pdb.py", "../alaric/rna_pdb.py",
        "../alaric/backbone.py", "../alaric/write_pdb.py"},
    "refe-best-fit-tables.DEPS": {
        "refe-best-fit-tables.py", "refe-best-fit-pairs.py", "refe-best-fit-bases.py",
        *{f"../alaric/{name}.py" for name in
          ("library", "parse_pdb", "reference", "backbone", "rna_pdb", "superimpose", "mutate")}},
    "refe-thresholds.DEPS": {"refe-thresholds.py", "../alaric/threshold_rounding.py"},
}


@pytest.mark.parametrize("filename", CODE)
def test_dependency_scope_is_exact_and_has_unique_flat_names(filename):
    """Prevent the whole package, grid/JAX code or library payload entering a step."""
    entries = (ROOT / "util" / filename).read_text().splitlines()
    expected = CODE[filename] | (set() if filename == "refe-thresholds.DEPS" else TEMPLATES)
    assert set(entries) == expected
    assert len(entries) == len(expected)
    assert len({Path(entry).name for entry in entries}) == len(entries)


@pytest.mark.parametrize("cent", [-1659, -100, -1, 0, 1, 12, 100, 1001])
def test_rounding_immediate_decimal_and_float_neighbours(cent):
    from alaric.energy_threshold import energy_threshold
    from alaric.threshold_rounding import strictly_above_cent

    boundary = Decimal(cent) / 100
    delta = Decimal("0.000000000000000001")
    assert strictly_above_cent(str(boundary - delta)) == format(boundary, ".2f")
    assert strictly_above_cent(str(boundary)) == format(boundary + Decimal("0.01"), ".2f")
    assert strictly_above_cent(str(boundary + delta)) == format(boundary + Decimal("0.01"), ".2f")
    value = float(boundary)
    for raw in (math.nextafter(value, -math.inf), value,
                math.nextafter(value, math.inf)):
        delivered = Decimal(repr(raw))
        rounded = Decimal(energy_threshold(raw))
        # Prove both strictness and minimality without repeating the algorithm.
        assert rounded - Decimal("0.01") <= delivered < rounded
        assert rounded.as_tuple().exponent == -2


@pytest.mark.parametrize("raw", ["NaN", "Infinity", "-Infinity", "", "not-a-number"])
def test_nonfinite_or_malformed_decimal_threshold_is_refused(raw):
    from alaric.threshold_rounding import strictly_above_cent

    with pytest.raises(ValueError):
        strictly_above_cent(raw)


@pytest.mark.parametrize("raw", [math.nan, math.inf, -math.inf])
def test_nonfinite_energy_threshold_is_refused(raw):
    from alaric.energy_threshold import energy_threshold

    with pytest.raises(ValueError):
        energy_threshold(raw)


def _atoms(text):
    return [(int(line[22:26]), line[12:16].strip(), line[17:20].strip(),
             tuple(float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54))))
            for line in text.splitlines() if line.startswith(("ATOM  ", "HETATM"))]


@pytest.mark.parametrize("gap", [False, True], ids=["non-one-start", "numbering-gap"])
def test_a2_ordinal_keys_template_atoms_and_original_coordinates(tmp_path, gap):
    import numpy as np

    original = _atoms(RNA.read_text())
    residues = list(dict.fromkeys(atom[0] for atom in original))
    # Keep four real nucleotides, shifted to 41,42,43,44 or 41,42,44,45.
    selected = residues[:4]
    numbers = dict(zip(selected, [41, 42, 44, 45] if gap else [41, 42, 43, 44]))
    lines = []
    for line in RNA.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")) and int(line[22:26]) in numbers:
            lines.append(line[:22] + f"{numbers[int(line[22:26])]:4d}" + line[26:])
    source = tmp_path / "rna.pdb"
    source.write_text("\n".join(lines) + "\n")
    output = tmp_path / "references"
    completed = subprocess.run(
        [sys.executable, "-B", str(ROOT / "util/refe-fragments.py"), str(source),
         "--templates", str(ROOT / "fraglib/templates"), "--output", str(output)],
        capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    source_atoms = _atoms(source.read_text())
    sequence = "".join(next(atom[2][-1] for atom in source_atoms if atom[0] == number)
                       for number in numbers.values())
    kept = (1, 3) if gap else (1, 2, 3)
    expected_names = {f"frag-{i}-{sequence[i-1:i+1]}-refe.pdb" for i in kept}
    assert {path.name for path in output.iterdir()} == expected_names
    for i in kept:
        seq = sequence[i-1:i+1]
        template = np.load(ROOT / f"fraglib/templates/{seq}-ppdb.npy", allow_pickle=False)
        atoms = _atoms((output / f"frag-{i}-{seq}-refe.pdb").read_text())
        expected = [(int(row["resid"]), row["name"].decode().strip(), row["resname"].decode().strip())
                    for row in template]
        assert [atom[:3] for atom in atoms] == expected
        for resid, name, resname, xyz in atoms:
            original_number = list(numbers.values())[i + resid - 2]
            # Source PDBs can use residue aliases such as RG, while the
            # fragment template writes the canonical one-letter base (G).
            # Coordinates are keyed by the source residue number and atom
            # name; residue-name spelling is not part of coordinate identity.
            matches = [atom[3] for atom in source_atoms
                       if atom[0] == original_number and atom[1] == name]
            assert matches == [xyz]


@pytest.fixture
def isolated_factory(tmp_path, monkeypatch):
    import util.seamlessify_refe_best_fit as factory

    util_dir = tmp_path / "checkout/util"
    util_dir.mkdir(parents=True)
    deps = ROOT / "util/refe-best-fit.DEPS"
    shutil.copy2(deps, util_dir / deps.name)
    for entry in deps.read_text().splitlines():
        target = util_dir / entry
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(deps.parent / entry, target)
    monkeypatch.setattr(factory, "__file__", str(util_dir / "seamlessify_refe_best_fit.py"))
    locator = tmp_path / "locator.yaml"
    locator.write_text("fraglen: 2\nconformers: /old/conformers\nrotamers: /old/rotamers\ncrmsds: /old/crmsds\n")
    identity = json.dumps({"schema": 1, "fraglen": 2,
                           "sha256": {"conformers/AA.npy": "a" * 64}}, sort_keys=True)

    def construct(*, identity_value=identity):
        step = factory.seamlessify_refe_best_fit(fraglib_config=locator)
        step.args.rna_pdb = RNA.read_bytes()
        step.args.exclude_pdb_code = "1b7f"
        step.args.fraglib_identities = identity_value
        built = step.build()
        built.construct()
        return built.transformation_checksum

    return util_dir, locator, identity, construct


@pytest.mark.parametrize("dependency", sorted(CODE["refe-best-fit.DEPS"] | TEMPLATES))
def test_a1_every_declared_file_edit_changes_transformation_identity(isolated_factory, dependency):
    util_dir, _, _, construct = isolated_factory
    original = construct()
    path = util_dir / dependency
    path.write_bytes(path.read_bytes() + b"\n")
    assert construct() != original, dependency


def test_a1_locator_relocation_and_irrelevant_files_do_not_change_identity(isolated_factory):
    util_dir, locator, identity, construct = isolated_factory
    original = construct()
    locator.write_text(locator.read_text().replace("/old/", "/relocated/"))
    assert construct() == original
    (util_dir.parent / "unrelated-grid-source.py").write_text("# unrelated edit\n")
    assert construct() == original
    changed = json.loads(identity)
    changed["sha256"]["conformers/AA.npy"] = "b" * 64
    assert construct(identity_value=json.dumps(changed, sort_keys=True)) != original


@pytest.fixture
def converter():
    spec = importlib.util.spec_from_file_location("w2_edge_converter", ROOT / "util/refe-thresholds.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a4_rounding_preserves_decimal_boundary_beyond_context_precision(tmp_path, converter):
    pairs = tmp_path / "pairs.tsv"
    bases = tmp_path / "bases.tsv"
    below = "0.119999999999999999999999999999"
    above = "0.120000000000000000000000000001"
    pairs.write_text(
        "down\tup\tsequence\tovRMSD\tcRMSD\n"
        f"1\t2\tAAA\t{below}\t{above}\n"
    )
    bases.write_text(
        "fragment\tsequence\tnucleotide1_ovRMSD\tnucleotide2_ovRMSD\n"
        f"1\tAA\t{below}\t{above}\n"
    )

    result = converter.build_thresholds(pairs, bases)

    assert (result["pairs"][0]["ovrmsd"], result["pairs"][0]["crmsd"]) == (
        "0.12",
        "0.13",
    )
    assert (result["bases"][0]["nuc1_ovrmsd"], result["bases"][0]["nuc2_ovrmsd"]) == (
        "0.12",
        "0.13",
    )


def test_a4_preserves_row_order_keys_and_decimal_boundary_text(tmp_path, converter):
    pairs = tmp_path / "pairs.tsv"
    bases = tmp_path / "bases.tsv"
    pairs.write_text("down\tup\tsequence\tovRMSD\tcRMSD\n"
                     "10\t11\tUUU\t0.11999999999999999999\t0.12\n"
                     "2\t3\tUGU\t0.12000000000000000001\t0.12\n")
    bases.write_text("fragment\tsequence\tnucleotide1_ovRMSD\tnucleotide2_ovRMSD\n"
                     "11\tUU\t0.12\t0.11999999999999999999\n"
                     "2\tUG\t0.12\t0.12000000000000000001\n")
    result = converter.build_thresholds(pairs, bases)
    assert [(row["down"], row["up"], row["sequence"]) for row in result["pairs"]] == [
        (10, 11, "UUU"), (2, 3, "UGU")]
    assert [(row["fragment"], row["sequence"]) for row in result["bases"]] == [(11, "UU"), (2, "UG")]
    assert [row["ovrmsd"] for row in result["pairs"]] == ["0.12", "0.13"]
    assert [row["nuc2_ovrmsd"] for row in result["bases"]] == ["0.12", "0.13"]
    # Both decimal inputs collapse to the same binary float: thresholds must
    # still be derived from the original text, before raw JSON conversion.
    assert [row["ovrmsd_raw"] for row in result["pairs"]] == [0.12, 0.12]
