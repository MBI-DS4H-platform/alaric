"""W2 acceptance oracles: G7/G8 and the A1/A4/A6 public boundaries.

These tests deliberately use table text and fixed expected thresholds.  They
are independent of the numerical code that produces the best-fit tables.
"""
from __future__ import annotations

import ast
import csv
import importlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "1b7f" / "data"
RNA = ROOT / "tests" / "1b7f" / "pdbs" / "rna.pdb"
CASES = [
    (0.123, "0.13"), (0.12, "0.13"), (-16.5913, "-16.59"),
    (-16.59, "-16.58"), (-3.141, "-3.14"), (0.0, "0.01"),
    (-0.001, "0.00"), (0.11999999999999998, "0.12"),
    (0.12000000000000001, "0.13"), (-0.010000000000000002, "-0.01"),
]
PAIR_THRESHOLDS = [
    ("1.70", "0.73"), ("1.57", "1.12"), ("0.73", "0.28"),
    ("0.65", "0.29"), ("1.21", "0.83"), ("0.52", "0.24"),
    ("0.84", "0.23"), ("0.60", "0.33"), ("0.80", "0.39"),
    ("1.35", "1.05"),
]
BASE_THRESHOLDS = [
    ("0.32", "1.18"), ("1.12", "0.77"), ("1.27", "0.59"),
    ("0.61", "0.61"), ("0.25", "0.57"), ("0.86", "0.50"),
    ("0.34", "0.66"), ("0.38", "0.77"), ("0.59", "0.61"),
    ("0.67", "0.47"), ("0.87", "2.25"),
]


def _rows(path):
    return list(csv.DictReader(io.StringIO(path.read_text()), delimiter="\t"))


def _convert(pairs, bases, output):
    completed = subprocess.run(
        [sys.executable, "-B", str(ROOT / "util" / "refe-thresholds.py"),
         str(pairs), str(bases), "--output", str(output)],
        cwd=output.parent, capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(output.read_text())


@pytest.mark.parametrize("raw,expected", CASES)
def test_a6_g7_strictly_above_shortest_float64_text(raw, expected):
    from alaric.energy_threshold import energy_threshold

    result = energy_threshold(raw)
    assert result == expected
    assert isinstance(result, str)
    assert raw < float(result)


def test_a4_and_a6_use_only_standard_library_transitively():
    """The only permitted local dependency is the shared rounding primitive."""
    allowed_local = {"threshold_rounding", "energy_threshold"}
    paths = [ROOT / "alaric" / "threshold_rounding.py",
             ROOT / "alaric" / "energy_threshold.py",
             ROOT / "util" / "refe-thresholds.py"]
    for path in paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                pieces = name.split(".")
                assert pieces[0] in sys.stdlib_module_names or pieces[-1] in allowed_local, (path, name)
        if path.name != "threshold_rounding.py":
            assert any((isinstance(node, ast.ImportFrom)
                        and (node.module or "").split(".")[-1] == "threshold_rounding")
                       or (isinstance(node, ast.Import)
                           and any(alias.name.split(".")[-1] == "threshold_rounding" for alias in node.names))
                       for node in ast.walk(tree)), path
            assert any(isinstance(node, ast.Call)
                       and ((isinstance(node.func, ast.Name) and node.func.id in {"strictly_above_cent", "strictly_above_float"})
                            or (isinstance(node.func, ast.Attribute) and node.func.attr in {"strictly_above_cent", "strictly_above_float"}))
                       for node in ast.walk(tree)), path


def test_a4_fixture_has_exact_schema_raw_values_and_rounded_goldens(tmp_path):
    pairs = _rows(DATA / "refe-best-fit-pairs.tsv")
    bases = _rows(DATA / "refe-best-fit-bases.tsv")
    result = _convert(DATA / "refe-best-fit-pairs.tsv", DATA / "refe-best-fit-bases.tsv", tmp_path / "thresholds.json")
    assert set(result) == {"schema", "pairs", "bases"}
    assert type(result["schema"]) is int and result["schema"] == 1
    assert len(result["pairs"]) == len(pairs) == 10
    assert len(result["bases"]) == len(bases) == 11
    for actual, raw, rounded in zip(result["pairs"], pairs, PAIR_THRESHOLDS):
        assert set(actual) == {"down", "up", "sequence", "ovrmsd", "crmsd", "ovrmsd_raw", "crmsd_raw"}
        assert (actual["down"], actual["up"], actual["sequence"]) == (int(raw["down"]), int(raw["up"]), raw["sequence"])
        assert (actual["ovrmsd"], actual["crmsd"]) == rounded
        assert actual["ovrmsd_raw"] == float(raw["ovRMSD"])
        assert actual["crmsd_raw"] == float(raw["cRMSD"])
        assert type(actual["ovrmsd_raw"]) in (int, float)
        assert actual["ovrmsd_raw"] < float(actual["ovrmsd"])
        assert actual["crmsd_raw"] < float(actual["crmsd"])
    for actual, raw, rounded in zip(result["bases"], bases, BASE_THRESHOLDS):
        assert set(actual) == {"fragment", "sequence", "nuc1_ovrmsd", "nuc2_ovrmsd", "nuc1_ovrmsd_raw", "nuc2_ovrmsd_raw"}
        assert (actual["fragment"], actual["sequence"]) == (int(raw["fragment"]), raw["sequence"])
        assert (actual["nuc1_ovrmsd"], actual["nuc2_ovrmsd"]) == rounded
        for n in (1, 2):
            assert actual[f"nuc{n}_ovrmsd_raw"] == float(raw[f"nucleotide{n}_ovRMSD"])
            assert actual[f"nuc{n}_ovrmsd_raw"] < float(actual[f"nuc{n}_ovrmsd"])


def test_a4_decimal_table_text_exact_multiple_negative_and_below_boundary(tmp_path):
    pairs = tmp_path / "pairs.tsv"
    bases = tmp_path / "bases.tsv"
    pairs.write_text("down\tup\tsequence\tovRMSD\tcRMSD\n1\t2\tAAA\t0.120000\t-16.590000\n2\t3\tAAA\t0.11999999999999998\t0\n")
    bases.write_text("fragment\tsequence\tnucleotide1_ovRMSD\tnucleotide2_ovRMSD\n1\tAA\t-0.001\t0.123\n")
    result = _convert(pairs, bases, tmp_path / "thresholds.json")
    assert (result["pairs"][0]["ovrmsd"], result["pairs"][0]["crmsd"]) == ("0.13", "-16.58")
    assert (result["pairs"][1]["ovrmsd"], result["pairs"][1]["crmsd"]) == ("0.12", "0.01")
    assert (result["bases"][0]["nuc1_ovrmsd"], result["bases"][0]["nuc2_ovrmsd"]) == ("0.00", "0.13")


def test_eighteen_existing_yaml_thresholds_match_a4_fixture(tmp_path):
    import yaml

    result = _convert(DATA / "refe-best-fit-pairs.tsv", DATA / "refe-best-fit-bases.tsv", tmp_path / "thresholds.json")
    bases = {row["fragment"]: row for row in result["bases"]}
    pairs = {(row["down"], row["up"]): row for row in result["pairs"]}
    fixtures = {
        "frag4-anchor-fwd": ("base", 4, 1), "frag4-anchor-bwd": ("base", 4, 2),
        "frag5-anchor-fwd": ("base", 5, 1), "frag5-anchor-bwd": ("base", 5, 2),
        "frag6-anchor": ("base", 6, 1), "frag10-anchor": ("base", 10, 2),
        "frag5-fwd-restrict": ("pair", 4, 5), "frag6-fwd": ("pair", 5, 6),
        "frag7-fwd": ("pair", 6, 7), "frag8-fwd": ("pair", 7, 8),
        "frag9-fwd": ("pair", 8, 9), "frag9-bwd-restrict": ("pair", 9, 10),
    }
    checked = 0
    for name, (kind, a, b) in fixtures.items():
        value = yaml.safe_load((ROOT / "tests" / "1b7f-anchor-refe" / name / "alaric.yaml").read_text())
        if kind == "base":
            assert float(bases[a][f"nuc{b}_ovrmsd"]) == value["ovrmsd"]
            checked += 1
        else:
            for field in ("ovrmsd", "crmsd"):
                assert float(pairs[a, b][field]) == value[field]
                checked += 1
    assert checked == 18


def test_a1_factory_leaves_project_inputs_unbound_and_never_executes(tmp_path, monkeypatch):
    import seamless_config
    import util.seamlessify_refe_best_fit as factory
    from seamless_transformer.transformer_class import BashTransformer

    def fail(*args, **kwargs):
        pytest.fail("factory configured or executed Seamless")

    monkeypatch.setattr(seamless_config, "init", fail)
    monkeypatch.setattr(BashTransformer, "run", fail)
    calls = []

    def fake_files(paths, **kwargs):
        calls.extend(Path(path) for path in paths)
        return {str(path): "a" * 64 for path in paths}, {}

    monkeypatch.setattr(factory, "files_to_checksums", fake_files)
    config = tmp_path / "fraglib.yaml"
    config.write_text("fraglen: 2\nconformers: /immutable/conformers\nrotamers: /immutable/rotamers\ncrmsds: /immutable/crmsds\n")
    transformer = factory.seamlessify_refe_best_fit(fraglib_config=config)
    deps = ROOT / "util" / "refe-best-fit.DEPS"
    declared = {(deps.parent / line).resolve() for line in deps.read_text().splitlines()}
    staged = {path.resolve() for path in calls} - {config.resolve()}
    assert staged == declared or staged == declared | {deps.resolve()}
    for name, celltype in (("rna_pdb", "bytes"), ("exclude_pdb_code", "str"), ("fraglib_identities", "str")):
        assert transformer.celltypes[name] == celltype
        assert transformer.args[name].checksum is None, f"project pin {name} was bound by factory"
    assert RNA not in calls
    code = transformer.code.content.decode()
    assert str(RNA) not in code and "1b7f" not in code.lower()
    assert transformer.celltypes.result == "folder"
    transformer.args.rna_pdb = RNA.read_bytes()
    transformer.args.exclude_pdb_code = "1b7f"
    transformer.args.fraglib_identities = '{"schema":1,"fraglen":2,"sha256":{}}'
    assert transformer.args.rna_pdb.value == RNA.read_bytes()
    assert transformer.args.exclude_pdb_code.value == "1b7f"


@pytest.mark.parametrize("filename", ["refe-best-fit.DEPS", "refe-fragments.DEPS", "refe-best-fit-tables.DEPS", "refe-thresholds.DEPS"])
def test_cli_dependency_lists_are_relative_exact_files_without_sidecars(filename):
    path = ROOT / "util" / filename
    lines = path.read_text().splitlines()
    assert lines and all(line and line.strip() == line and not line.startswith("#") for line in lines)
    assert len(lines) == len(set(lines))
    for line in lines:
        assert not Path(line).is_absolute(), (filename, line)
        dependency = path.parent / line
        assert dependency.is_file(), dependency
        assert not Path(str(dependency) + ".CHECKSUM").exists(), dependency
