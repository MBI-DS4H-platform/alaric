"""W8 follow-up oracles for deployment, fail-closed export, and RNA breaks.

Run explicitly from the repository root:
  conda run --live-stream -n alaric python -m pytest -q tests/test_w8_contract_followup.py

Reuses the X1–X7 snapshot fixture; does not require the optional slow pose pool.
"""
from copy import deepcopy
from itertools import product
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "alaric"), str(ROOT / "tests")]

from test_w8_export_seam_contract import snapshot  # noqa: E402,F401


def _bundle(snapshot):
    from alaric.export_bundle import build_export_bundle

    return build_export_bundle(snapshot)


def _assert_refused(snapshot):
    # Missing contract evidence may be refused at validation or represented by
    # explicit non-overridable hard blocks. Incidental TypeError/KeyError fails.
    snapshot["proceed_anyway_required"] = True
    snapshot["proceed_anyway"] = True
    try:
        bundle = _bundle(snapshot)
    except ValueError:
        return
    assert bundle["hard_blocks"], "missing required evidence must be a hard block"
    assert bundle["ready"] is False


@pytest.mark.parametrize("deployer", ["local", "local-chunk"])
def test_energy_filter_backend_context_renders_real_deployed_template(tmp_path, deployer):
    from alaric.energy_filter_action import build_parser
    from alaric.middle.backend import template_context
    from alaric.middle.deploy import render_action_template
    from alaric.middle.project import Project
    from alaric.middle.resolve import ResolvedAction

    source_dir = tmp_path / "frag3-bestfit"
    source_dir.mkdir()
    (source_dir / "sigil.txt").write_text("source-pose-sigil\n")
    source = ResolvedAction("frag3-bestfit", source_dir, "anchor-refe",
                            {"fragment": 3, "sequence": "GU"}, {})
    action = ResolvedAction("frag3-energy-range-2", tmp_path / "filter", "energy-filter", {
        "input": source, "__protein": "range-2-aa.pdb", "protein": "range-2",
        "fragment": 3, "sequence": "GU", "exclude": ["1b7f"],
        "threshold": -2.0, "epsilon": 15, "cascade": [], "min_cascade": 0,
        "nprocs": 2, "chunk_poses": 999,
    }, {})
    arguments = dict(alaric_dir=str(ROOT / "alaric"), output_dir="../CACHE/results/filtered",
                     location="local")
    context = template_context(action, **arguments)
    assert shlex.split(context["protein_range"]) == ["range-2"]
    # Use deployment's real template discovery and renderer, with no injected
    # context that could accidentally conceal a missing backend field.
    lines = render_action_template(Project(tmp_path), action, deployer=deployer, **arguments)
    script = "\n".join(lines)
    assert "{{" not in script and "}}" not in script
    command = shlex.split(script.replace("\\\n", " "), comments=True)
    assert Path(command[1]).name == "energy_filter_action.py"
    parsed = build_parser().parse_args(command[2:])
    assert parsed.protein_range == "range-2"
    assert parsed.protein.name == "range-2-aa.pdb"
    assert parsed.pose_dir.as_posix() == "../CACHE/results/source-pose-sigil"
    assert (parsed.nprocs, parsed.chunk_poses) == (2, 999)


def test_export_multi_range_energy_actions_form_successive_and_chain(snapshot):
    snapshot["energy_matrix"]["frag-1-AC"]["range-3"] = True
    snapshot["edited_energy_thresholds"]["frag-1-AC"]["range-3"] = "-3.00"
    bundle = _bundle(snapshot)
    assert bundle["ready"]
    actions = bundle["energy_filter_actions"]
    chain = [row for row in actions if row["name"].startswith("energy-filter-frag1-")]
    assert [row["protein"] for row in chain] == ["range-2", "range-3"]
    assert chain[0]["input"] == "frag-1-bestfit"
    assert chain[1]["input"] == chain[0]["name"]
    other = next(row for row in actions if row["name"] == "energy-filter-frag3-range-2")
    assert other["input"] == "frag-3-bestfit", "chains must stay fragment-local"


@pytest.mark.parametrize("path", [
    ("validations",), ("validations", "results"), ("validations", "decisions"),
    ("validations", "results", "fit"), ("validations", "results", "energetics"),
    ("validations", "results", "gaussian"), ("validations", "decisions", "thresholds"),
    ("handoff",), ("class_table",), ("selected_anchors",), ("reference_candidates",),
], ids=lambda path: ".".join(path))
@pytest.mark.parametrize("replacement", ["absent", "null", "empty"])
def test_export_missing_required_evidence_fails_closed(snapshot, path, replacement):
    assert _bundle(snapshot)["ready"], "baseline must satisfy the export fixture"
    parent = snapshot
    for key in path[:-1]:
        parent = parent[key]
    key = path[-1]
    if replacement == "absent":
        del parent[key]
    elif replacement == "null":
        parent[key] = None
    else:
        parent[key] = [] if isinstance(parent[key], list) else {}
    _assert_refused(snapshot)


@pytest.mark.parametrize("path", [("handoff", "fragment_rows"), ("class_table", "rows")],
                         ids=["handoff-fragment-rows", "class-rows"])
def test_export_empty_handoff_or_class_rows_fail_closed(snapshot, path):
    assert _bundle(snapshot)["ready"]
    snapshot[path[0]][path[1]] = []
    _assert_refused(snapshot)


@pytest.mark.parametrize("field", ["energy_matrix", "kept_pairs"])
@pytest.mark.parametrize("replacement", ["absent", "null"])
def test_export_missing_matrix_or_pair_decisions_fail_closed(snapshot, field, replacement):
    assert _bundle(snapshot)["ready"]
    if replacement == "absent":
        del snapshot[field]
    else:
        snapshot[field] = None
    _assert_refused(snapshot)


def test_reference_fragment_deps_pin_only_minimal_backbone_helper():
    entries = (ROOT / "util/refe-fragments.DEPS").read_text().splitlines()
    templates = {f"../fraglib/templates/{''.join(seq)}-ppdb.npy"
                 for length in (1, 2) for seq in product("ACGU", repeat=length)}
    code = {"refe-fragments.py", "../alaric/parse_pdb.py", "../alaric/rna_pdb.py",
            "../alaric/write_pdb.py", "../alaric/backbone.py"}
    assert set(entries) == code | templates
    assert len(entries) == len(set(entries)), "dependency lines must be unique"
    assert len({Path(entry).name for entry in entries}) == len(entries)
    assert "../alaric/reference.py" not in entries


@pytest.mark.parametrize("manifest,script_entries,module_names", [
    ("refe-best-fit.DEPS", {"refe-best-fit.py", "refe_best_fit_seamless_support.py"},
     {"library", "parse_pdb", "reference", "superimpose", "write_pdb", "rna_pdb", "mutate", "backbone"}),
    ("refe-best-fit-tables.DEPS", {"refe-best-fit-tables.py", "refe-best-fit-pairs.py", "refe-best-fit-bases.py"},
     {"library", "parse_pdb", "reference", "rna_pdb", "superimpose", "mutate", "backbone"}),
], ids=["best-fit", "best-fit-tables"])
def test_reference_consumers_declare_exact_backbone_dependency_closure(manifest, script_entries, module_names):
    entries = (ROOT / "util" / manifest).read_text().splitlines()
    templates = {f"../fraglib/templates/{''.join(seq)}-ppdb.npy"
                 for length in (1, 2) for seq in product("ACGU", repeat=length)}
    expected = script_entries | {f"../alaric/{name}.py" for name in module_names} | templates
    assert "../alaric/backbone.py" in entries
    assert set(entries) == expected
    assert len(entries) == len(expected)
    assert len({Path(entry).name for entry in entries}) == len(entries)


@pytest.mark.parametrize("anchor_fragment,nucleotide,neighbor_fragment,neighbor_nucleotide", [
    (1, "second", 2, "first"), (3, "first", 2, "second"),
], ids=["second-needs-next-first", "first-needs-previous-second"])
@pytest.mark.parametrize("change", ["absent", "clipped", "not-unclipped"])
def test_selected_residue_requires_each_kept_unclipped_neighbor_candidate(
    snapshot, anchor_fragment, nucleotide, neighbor_fragment, neighbor_nucleotide, change,
):
    snapshot["selected_anchors"] = [row for row in snapshot["selected_anchors"]
                                    if (row["fragment"], row["nucleotide"]) == (anchor_fragment, nucleotide)]
    assert len(snapshot["selected_anchors"]) == 1
    assert _bundle(snapshot)["ready"]
    selected_residue = anchor_fragment + (nucleotide == "second")
    assert neighbor_fragment + (neighbor_nucleotide == "second") == selected_residue
    matching = [row for row in snapshot["reference_candidates"]
                if (row["fragment"], row["nucleotide"]) == (neighbor_fragment, neighbor_nucleotide)]
    assert len(matching) == 1
    if change == "absent":
        snapshot["reference_candidates"].remove(matching[0])
    else:
        matching[0].update({"clipped": True} if change == "clipped" else {"unclipped": False})
    # A clipped row cannot satisfy a still-kept, unclipped fragment obligation.
    _assert_refused(snapshot)


@pytest.mark.parametrize("neighbor_scope", ["not-kept", "clipped", "not-unclipped"])
def test_clipped_or_nonkept_neighbor_fragment_does_not_require_candidate(snapshot, neighbor_scope):
    assert _bundle(snapshot)["ready"]
    if neighbor_scope == "not-kept":
        del snapshot["kept_fragments"]["frag-2-CG"]
    else:
        snapshot["kept_fragments"]["frag-2-CG"].update(
            {"clipped": True} if neighbor_scope == "clipped" else {"unclipped": False})
    snapshot["reference_candidates"] = [row for row in snapshot["reference_candidates"] if row["fragment"] != 2]
    snapshot["kept_pairs"] = []
    bundle = _bundle(snapshot)
    assert not bundle["hard_blocks"]
    assert bundle["ready"]
    assert {row["fragment"] for row in bundle["anchor_refe_actions"]} == {1, 3}


@pytest.mark.parametrize("candidate_change", ["absent", "clipped", "not-unclipped"])
def test_selected_anchor_requires_its_own_unclipped_reference_candidate(snapshot, candidate_change):
    assert _bundle(snapshot)["ready"]
    candidate = next(row for row in snapshot["reference_candidates"]
                     if (row["fragment"], row["nucleotide"]) == (1, "second"))
    snapshot["reference_candidates"] = [
        row for row in snapshot["reference_candidates"]
        if (row["fragment"], row["nucleotide"]) != (1, "second")
    ]
    # Other selected anchors still have candidates, including their overlap
    # duplicate: a nonempty global candidate list cannot prove completeness.
    assert snapshot["reference_candidates"]
    if candidate_change != "absent":
        candidate = deepcopy(candidate)
        candidate.update({"clipped": True} if candidate_change == "clipped"
                         else {"unclipped": False})
        snapshot["reference_candidates"].append(candidate)
    _assert_refused(snapshot)


@pytest.fixture
def broken_backbone():
    from alaric.parse_pdb import parse_pdb
    from alaric.reference import Reference

    atoms = parse_pdb((ROOT / "tests/1b7f/pdbs/rna.pdb").read_text())
    resids = list(dict.fromkeys(int(resid) for resid in atoms["resid"]))[:4]
    atoms = atoms[np.isin(atoms["resid"], resids)].copy()
    # Renumber to a non-one start with consecutive labels. Moving the complete
    # downstream block breaks only link 2→3, preserving links 1→2 and 3→4.
    for index, resid in enumerate(resids):
        atoms["resid"][atoms["resid"] == resid] = 101 + index
    kwargs = dict(rna=True, ignore_unknown=False, ignore_missing=False, ignore_reordered=False)
    intact = Reference(atoms, **kwargs)
    assert intact.get_fragment_positions(2) == [1, 2, 3]
    broken = atoms.copy()
    broken["x"][broken["resid"] >= 103] += 100.0
    reference = Reference(broken, **kwargs)
    assert reference.get_fragment_positions(2) == [1, 3]
    return broken, reference


def test_rmsd_rejects_consecutive_label_chemical_break_consistently(broken_backbone):
    from alaric.rmsd import _infer_fragment

    broken, reference = broken_backbone
    with pytest.raises(ValueError):
        _infer_fragment(broken, 2)
    # The two-residue implicit selector must enforce the same chemical rule.
    with pytest.raises(ValueError):
        _infer_fragment(broken[np.isin(broken["resid"], [102, 103])], None)
    for position in reference.get_fragment_positions(2):
        sequence, selected = _infer_fragment(broken, position)
        assert sequence == reference.get_sequence(position, 2)
        np.testing.assert_array_equal(np.unique(selected["resid"]), [1, 2])
        coords = np.column_stack((selected["x"], selected["y"], selected["z"]))
        np.testing.assert_allclose(coords, reference.get_coordinates(position, 2), rtol=0, atol=0)


def test_reference_fragment_cli_omits_only_chemical_break_window(broken_backbone, tmp_path):
    from alaric.parse_pdb import parse_pdb
    from alaric.write_pdb import write_pdb

    broken, reference = broken_backbone
    source = tmp_path / "broken-reference.pdb"
    source.write_text(write_pdb(broken))
    output = tmp_path / "fragments"
    result = subprocess.run([
        sys.executable, str(ROOT / "util/refe-fragments.py"), str(source),
        "--templates", str(ROOT / "fraglib/templates"), "--output", str(output),
    ], check=False, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    expected = {f"frag-{pos}-{reference.get_sequence(pos, 2)}-refe.pdb"
                for pos in reference.get_fragment_positions(2)}
    assert {path.name for path in output.iterdir()} == expected
    for position in reference.get_fragment_positions(2):
        path = output / f"frag-{position}-{reference.get_sequence(position, 2)}-refe.pdb"
        emitted = parse_pdb(path.read_text())
        coords = np.column_stack((emitted["x"], emitted["y"], emitted["z"]))
        np.testing.assert_allclose(coords, reference.get_coordinates(position, 2), rtol=0, atol=0.001)
