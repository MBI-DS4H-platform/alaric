"""Sigils hash extracted values, not the DATA files they were extracted from.

An ``energy-filter`` with ``threshold: auto`` or ``cascade/min_cascade: auto`` takes its
values from DATA/energy-thresholds.json and DATA/score-filter.yaml. The sigil must hold
those values only: an explicit value and an ``auto`` that resolves to it give the same
sigil, and editing an unrelated entry of the file changes nothing.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from alaric.middle.graph import ActionGraph
from alaric.middle.project import Project
from alaric.middle.sigil import compute_project_sigils, parameter_record

THRESHOLD = -15.93
CASCADE = [
    {"plateau_distance": 5, "neighbour_distance": 7, "max_error": 1.0},
    {"plateau_distance": 10, "neighbour_distance": 12, "max_error": 0.05},
]
MIN_CASCADE = 1000

THRESHOLDS = {
    "schema": 1,
    "frag-4-GU": {"range-2": "-15.93", "range-3": "-12.50"},
    "frag-5-UU": {"range-2": "-9.10"},
}
SCORE_FILTER = {"cascade": CASCADE, "min_cascade": MIN_CASCADE}


def _action(root: Path, name: str, spec: dict) -> None:
    path = root / name
    path.mkdir()
    (path / "alaric.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))


def _write_thresholds(root: Path, data: dict) -> None:
    (root / "DATA" / "energy-thresholds.json").write_text(json.dumps(data, indent=1) + "\n")


def _write_score_filter(root: Path, data: dict) -> None:
    (root / "DATA" / "score-filter.yaml").write_text(yaml.safe_dump(data, sort_keys=False))


def _project(root: Path) -> Project:
    data = root / "DATA"
    data.mkdir()
    (data / "constraints.json").write_text(
        json.dumps({"schema": 1, "frag4": {"sequence": "GU"}, "pairs": []})
    )
    (data / "pdbcode.txt").write_text("1abc\n")
    (data / "reference.pdb").write_text("HEADER reference\n")
    (data / "range-2-aa.pdb").write_text("HEADER range-2\n")
    (data / "range-3-aa.pdb").write_text("HEADER range-3\n")
    _write_thresholds(root, THRESHOLDS)
    _write_score_filter(root, SCORE_FILTER)

    _action(root, "frag4-anchor", {
        "action": "anchor-refe", "fragment": 4, "sequence": "GU", "exclude": [],
        "nucleotide": "first", "ovrmsd": 0.75,
    })

    _action(root, "frag5-anchor", {
        "action": "anchor-refe", "fragment": 5, "sequence": "UU", "exclude": [],
        "nucleotide": "first", "ovrmsd": 0.75,
    })

    def energy_filter(name: str, **overrides) -> None:
        spec = {
            "action": "energy-filter", "input": "frag4-anchor", "fragment": 4,
            "sequence": "GU", "exclude": [], "protein": "range-2", "threshold": "auto",
            "epsilon": 15, "cascade": "auto", "min_cascade": "auto",
        }
        _action(root, name, {**spec, **overrides})

    # Everything auto.
    energy_filter("f4-auto")
    # The same values given explicitly.
    energy_filter("f4-explicit", threshold=THRESHOLD, cascade=CASCADE, min_cascade=MIN_CASCADE)
    # Other cells of energy-thresholds.json: same fragment other range, other fragment.
    energy_filter("f4-r3-auto", protein="range-3")
    energy_filter("f5-auto", input="frag5-anchor", fragment=5, sequence="UU")
    # Only one of cascade / min_cascade is auto.
    energy_filter("f4-cascade-auto", threshold=THRESHOLD, min_cascade=MIN_CASCADE)
    energy_filter("f4-min-auto", threshold=THRESHOLD, cascade=CASCADE)
    return Project.discover(root)


def _sigils(project: Project) -> dict[str, str]:
    # force: the sigil.txt of the previous call is stale after a DATA edit
    return compute_project_sigils(project, force=True)


def _part1(sigils: dict[str, str], name: str) -> str:
    return sigils[name].split("-")[0]


@pytest.fixture
def project(tmp_path: Path) -> Project:
    return _project(tmp_path)


def test_explicit_and_auto_values_give_the_same_sigil(project: Project) -> None:
    sigils = _sigils(project)
    assert sigils["f4-auto"] == sigils["f4-explicit"]
    # Mixed explicit/auto routes agree as well.
    assert sigils["f4-cascade-auto"] == sigils["f4-explicit"]
    assert sigils["f4-min-auto"] == sigils["f4-explicit"]


def test_explicit_and_auto_records_have_identical_values(project: Project) -> None:
    resolved = ActionGraph(project).build()
    records = {
        name: parameter_record(resolved[name], {"input": "x"})
        for name in ("f4-auto", "f4-explicit")
    }
    auto, explicit = records["f4-auto"], records["f4-explicit"]
    for key in ("threshold", "cascade", "min_cascade"):
        assert auto[key] == explicit[key]
        assert type(auto[key]) is type(explicit[key])
    assert type(auto["threshold"]) is float and auto["threshold"] == THRESHOLD
    assert type(auto["min_cascade"]) is int
    assert auto["cascade"] == [
        {"plateau_distance": 5.0, "neighbour_distance": 7.0, "max_error": 1.0},
        {"plateau_distance": 10.0, "neighbour_distance": 12.0, "max_error": 0.05},
    ]
    assert all(type(v) is float for step in auto["cascade"] for v in step.values())


def test_cascade_key_order_in_explicit_yaml_does_not_matter(tmp_path: Path) -> None:
    project = _project(tmp_path)
    reordered = [{"max_error": 1.0, "neighbour_distance": 7, "plateau_distance": 5},
                 {"max_error": 0.05, "plateau_distance": 10.0, "neighbour_distance": 12.0}]
    _action(tmp_path, "f4-reordered", {
        "action": "energy-filter", "input": "frag4-anchor", "fragment": 4, "sequence": "GU",
        "exclude": [], "protein": "range-2", "threshold": THRESHOLD, "epsilon": 15,
        "cascade": reordered, "min_cascade": MIN_CASCADE,
    })
    sigils = _sigils(Project.discover(tmp_path))
    assert sigils["f4-reordered"] == sigils["f4-auto"]


def test_data_files_are_not_hashed_for_extracted_values(project: Project) -> None:
    resolved = ActionGraph(project).build()
    action = resolved["f4-auto"]
    # Only the protein, which the action consumes whole, is a hashed file.
    assert set(action.file_fields) == {"protein"}
    record = parameter_record(action, {"input": "x"})
    assert "energy-thresholds" not in record and "score-filter" not in record
    assert "__energy-thresholds" not in record and "__score-filter" not in record


def test_unrelated_threshold_entries_do_not_change_the_sigil(project: Project) -> None:
    root = project.root
    before = _sigils(project)
    edited = json.loads(json.dumps(THRESHOLDS))
    edited["frag-5-UU"]["range-2"] = "-8.00"        # another fragment
    edited["frag-4-GU"]["range-3"] = "-13.00"       # another range of the same fragment
    edited["frag-6-UU"] = {"range-2": "-7.00"}      # a new fragment
    _write_thresholds(root, edited)
    after = _sigils(project)
    assert after["f4-auto"] == before["f4-auto"]
    assert after["f4-explicit"] == before["f4-explicit"]
    assert after["f4-cascade-auto"] == before["f4-cascade-auto"]
    # The entries that were edited are the ones their own actions use.
    assert after["f4-r3-auto"] != before["f4-r3-auto"]
    assert after["f5-auto"] != before["f5-auto"]


def test_reformatting_the_threshold_file_does_not_change_the_sigil(project: Project) -> None:
    before = _sigils(project)
    reordered = {"frag-5-UU": THRESHOLDS["frag-5-UU"], "schema": 1,
                 "frag-4-GU": dict(reversed(THRESHOLDS["frag-4-GU"].items()))}
    (project.root / "DATA" / "energy-thresholds.json").write_text(json.dumps(reordered))
    assert _sigils(project) == before


def test_the_used_threshold_entry_changes_the_sigil(project: Project) -> None:
    before = _sigils(project)
    edited = json.loads(json.dumps(THRESHOLDS))
    edited["frag-4-GU"]["range-2"] = "-15.94"
    _write_thresholds(project.root, edited)
    after = _sigils(project)
    assert _part1(after, "f4-auto") != _part1(before, "f4-auto")
    assert _part1(after, "f4-cascade-auto") == _part1(before, "f4-cascade-auto")
    assert after["f4-explicit"] == before["f4-explicit"]
    # Same fragment and range as an explicit threshold of the new value.
    _action(project.root, "f4-explicit-new", {
        "action": "energy-filter", "input": "frag4-anchor", "fragment": 4, "sequence": "GU",
        "exclude": [], "protein": "range-2", "threshold": -15.94, "epsilon": 15,
        "cascade": CASCADE, "min_cascade": MIN_CASCADE,
    })
    again = _sigils(Project.discover(project.root))
    assert again["f4-explicit-new"] == again["f4-auto"]


def test_unrelated_score_filter_edits_do_not_change_the_sigil(project: Project) -> None:
    before = _sigils(project)
    # Formatting and comments are not content.
    (project.root / "DATA" / "score-filter.yaml").write_text(
        "# tuned by hand\nmin_cascade: 1000\ncascade:\n"
        "  - {max_error: 1.0, neighbour_distance: 7, plateau_distance: 5}\n"
        "  - {max_error: 0.05, neighbour_distance: 12, plateau_distance: 10}\n"
    )
    assert _sigils(project) == before


def test_score_filter_edits_change_only_the_actions_that_use_the_value(project: Project) -> None:
    before = _sigils(project)

    _write_score_filter(project.root, {**SCORE_FILTER, "min_cascade": 500})
    after = _sigils(project)
    assert after["f4-auto"] != before["f4-auto"]
    assert after["f4-min-auto"] != before["f4-min-auto"]
    assert after["f4-cascade-auto"] == before["f4-cascade-auto"]
    assert after["f4-explicit"] == before["f4-explicit"]
    assert after["f4-r3-auto"] != before["f4-r3-auto"]

    steps = json.loads(json.dumps(CASCADE))
    steps[1]["max_error"] = 0.1
    _write_score_filter(project.root, {"cascade": steps, "min_cascade": MIN_CASCADE})
    after = _sigils(project)
    assert after["f4-auto"] != before["f4-auto"]
    assert after["f4-cascade-auto"] != before["f4-cascade-auto"]
    assert after["f4-min-auto"] == before["f4-min-auto"]
    assert after["f4-explicit"] == before["f4-explicit"]


def test_protein_file_is_still_hashed_by_content(project: Project) -> None:
    before = _sigils(project)
    (project.root / "DATA" / "range-2-aa.pdb").write_text("HEADER range-2 edited\n")
    after = _sigils(project)
    for name in ("f4-auto", "f4-explicit", "f5-auto"):
        assert _part1(after, name) != _part1(before, name)
    assert after["f4-r3-auto"] == before["f4-r3-auto"]


def test_generated_scripts_pass_values_and_never_read_the_data_files(project: Project) -> None:
    from alaric.middle.deploy import generate_check_sh, generate_chunk_files, generate_run_sh

    _sigils(project)
    action = ActionGraph(project).build()["f4-auto"]
    scripts: dict[str, str] = {}
    for deployer in ("local", "remote"):
        scripts[deployer] = generate_run_sh(project, action, deployer)
        scripts[f"{deployer}-check"] = generate_check_sh(project, action, location=deployer)
    for deployer in ("local-chunk", "remote-chunk"):
        for name, body in generate_chunk_files(project, action, deployer, nchunks=2).items():
            scripts[f"{deployer}/{name}"] = body
    for name, body in scripts.items():
        assert "energy-thresholds" not in body, name
        assert "score-filter" not in body, name
    for deployer in ("local", "remote"):
        body = scripts[deployer]
        assert "--threshold -15.93" in body
        assert '"plateau_distance":5.0' in body
        assert "--min-cascade 1000" in body
    chunk_bodies = [b for n, b in scripts.items() if n.startswith(("local-chunk/", "remote-chunk/"))]
    assert any("--threshold -15.93" in b for b in chunk_bodies)


def test_remote_push_ships_only_the_protein_file(project: Project, monkeypatch) -> None:
    import alaric.middle.deploy as deploy_module

    _sigils(project)
    resolved = ActionGraph(project).build()["f4-auto"]
    monkeypatch.setenv("ALARIC_REMOTE_DEPLOYMENT_DIR", "/remote/deploy")
    monkeypatch.setenv("ALARIC_REMOTE_RESULT_DIR", "/remote/results")
    calls: list[list[str]] = []

    class Done:
        returncode = 1  # "blob not present": upload everything that is a file field

    monkeypatch.setattr(deploy_module.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or Done())
    deploy_module._remote_push(
        host="remote", action=resolved, sigil="s", files={},
        file_fields=resolved.file_fields,
    )
    shipped = [c[1] for c in calls if c[0] == "scp" and c[1].startswith(str(project.data_dir))]
    assert shipped == [str(project.data_dir / "range-2-aa.pdb")]


# --- explicit grow crmsd/ovrmsd and explicit sequence vs the auto route -------------------


def _flow_project(root: Path, *, crmsd: str, ovrmsd: str) -> Project:
    """anchor-refe (frag 3) -> grow / grow-test (frag 4) -> score, all with sequence AA."""
    data = root / "DATA"
    data.mkdir()
    (data / "constraints.json").write_text(json.dumps({
        "schema": 1,
        "frag3": {"sequence": "AA"},
        "frag4": {"sequence": "AA"},
        "pairs": [{"down": 3, "up": 4, "sequence": "AAA", "crmsd": crmsd, "ovrmsd": ovrmsd}],
    }))
    (data / "pdbcode.txt").write_text("1abc\n")
    (data / "reference.pdb").write_text("HEADER reference\n")
    (data / "range-2-aa.pdb").write_text("HEADER range-2\n")
    _action(root, "frag3-pose", {
        "action": "anchor-refe", "fragment": 3, "sequence": "auto", "exclude": "auto",
        "nucleotide": "first", "ovrmsd": 0.75,
    })
    return Project.discover(root)


def _add_grow(root: Path, name: str, action: str = "grow", **overrides) -> None:
    spec = {
        "action": action, "input": "frag3-pose", "fragment": 4, "sequence": "auto",
        "exclude": "auto", "direction": "forward", "crmsd": "auto", "ovrmsd": "auto",
    }
    if action == "grow-test":
        spec["conformer"] = 1
    _action(root, name, {**spec, **overrides})


def _record(project: Project, name: str) -> dict:
    return parameter_record(ActionGraph(project).build()[name], {"input": "x"})


@pytest.mark.parametrize("action", ["grow", "grow-test"])
@pytest.mark.parametrize(
    "explicit, pair_value",
    [("0.25", "0.25"), (0.25, "0.25"), (0.25, 0.25), (1, "1.00"), ("1", "1.00"), (1.0, 1), (0.1, "0.10")],
)
def test_explicit_grow_thresholds_match_auto(tmp_path: Path, action: str, explicit, pair_value) -> None:
    project = _flow_project(tmp_path, crmsd=pair_value, ovrmsd=pair_value)
    _add_grow(tmp_path, "frag4-auto", action)
    _add_grow(tmp_path, "frag4-explicit", action, crmsd=explicit, ovrmsd=explicit)
    # One field explicit, the other auto.
    _add_grow(tmp_path, "frag4-crmsd", action, crmsd=explicit)
    _add_grow(tmp_path, "frag4-ovrmsd", action, ovrmsd=explicit)
    sigils = _sigils(Project.discover(tmp_path))
    assert sigils["frag4-explicit"] == sigils["frag4-auto"]
    assert sigils["frag4-crmsd"] == sigils["frag4-auto"]
    assert sigils["frag4-ovrmsd"] == sigils["frag4-auto"]
    auto, explicit_record = _record(project, "frag4-auto"), _record(project, "frag4-explicit")
    for key in ("crmsd", "ovrmsd"):
        assert type(explicit_record[key]) is float
        assert type(auto[key]) is type(explicit_record[key]) and auto[key] == explicit_record[key]


def test_explicit_grow_threshold_that_differs_from_auto_changes_the_sigil(tmp_path: Path) -> None:
    _flow_project(tmp_path, crmsd="0.25", ovrmsd="0.75")
    _add_grow(tmp_path, "frag4-auto")
    _add_grow(tmp_path, "frag4-other", crmsd="0.3", ovrmsd=0.75)
    sigils = _sigils(Project.discover(tmp_path))
    assert _part1(sigils, "frag4-other") != _part1(sigils, "frag4-auto")


@pytest.mark.parametrize("bad", [True, float("inf"), float("nan"), "inf", "x", [0.25]])
def test_grow_rejects_bad_explicit_thresholds(tmp_path: Path, bad) -> None:
    from alaric.middle.errors import SchemaError
    from alaric.middle.schema import load_action_spec

    _flow_project(tmp_path, crmsd="0.25", ovrmsd="0.75")
    for key in ("crmsd", "ovrmsd"):
        _add_grow(tmp_path, f"frag4-bad-{key}", **{key: bad})
        with pytest.raises(SchemaError):
            load_action_spec("x", tmp_path / f"frag4-bad-{key}" / "alaric.yaml")


def test_grow_ovrmsd_above_sqrt3_still_reaches_the_runtime_check(tmp_path: Path) -> None:
    # The sqrt(3) limit is enforced by grow.py, not by the schema; normalization must keep
    # passing the value through as a float.
    _flow_project(tmp_path, crmsd="0.25", ovrmsd="0.75")
    _add_grow(tmp_path, "frag4-big", ovrmsd="1.8")
    assert _record(Project.discover(tmp_path), "frag4-big")["ovrmsd"] == 1.8


@pytest.mark.parametrize("written", ["aa", " AA", "Aa\n", "AA"])
def test_explicit_sequence_matches_auto(tmp_path: Path, written: str) -> None:
    _flow_project(tmp_path, crmsd="0.25", ovrmsd="0.75")
    _action(tmp_path, "frag3-pose-auto", {
        "action": "anchor-refe", "fragment": 3, "sequence": "auto", "exclude": "auto",
        "nucleotide": "first", "ovrmsd": 0.75,
    })
    _action(tmp_path, "frag3-pose-explicit", {
        "action": "anchor-refe", "fragment": 3, "sequence": written, "exclude": "auto",
        "nucleotide": "first", "ovrmsd": 0.75,
    })
    _add_grow(tmp_path, "frag4-auto")
    _add_grow(tmp_path, "frag4-explicit", sequence=written)
    _add_grow(tmp_path, "frag4t-auto", "grow-test")
    _add_grow(tmp_path, "frag4t-explicit", "grow-test", sequence=written)
    score = {"action": "score", "input": "frag3-pose", "exclude": "auto", "protein": "range-2"}
    _action(tmp_path, "frag3-score-auto", {**score, "sequence": "auto"})
    _action(tmp_path, "frag3-score-explicit", {**score, "sequence": written})
    _action(tmp_path, "frag3-filter-explicit", {
        "action": "energy-filter", "input": "frag3-pose", "fragment": 3, "sequence": written,
        "exclude": "auto", "protein": "range-2", "threshold": -1.0, "epsilon": 15,
        "cascade": [], "min_cascade": 0,
    })
    _action(tmp_path, "frag3-filter-auto", {
        "action": "energy-filter", "input": "frag3-pose", "fragment": 3, "sequence": "auto",
        "exclude": "auto", "protein": "range-2", "threshold": -1.0, "epsilon": 15,
        "cascade": [], "min_cascade": 0,
    })
    sigils = _sigils(Project.discover(tmp_path))
    assert sigils["frag3-pose-explicit"] == sigils["frag3-pose-auto"]
    assert sigils["frag4-explicit"] == sigils["frag4-auto"]
    assert sigils["frag4t-explicit"] == sigils["frag4t-auto"]
    assert sigils["frag3-score-explicit"] == sigils["frag3-score-auto"]
    assert sigils["frag3-filter-explicit"] == sigils["frag3-filter-auto"]
    project = Project.discover(tmp_path)
    for name in ("frag3-pose-explicit", "frag4-explicit", "frag3-score-explicit"):
        assert _record(project, name)["sequence"] == "AA"


def test_explicit_sequence_normalization_keeps_a_different_sequence_different(tmp_path: Path) -> None:
    _flow_project(tmp_path, crmsd="0.25", ovrmsd="0.75")
    _add_grow(tmp_path, "frag4-auto")
    _add_grow(tmp_path, "frag4-other", sequence="ac")
    sigils = _sigils(Project.discover(tmp_path))
    assert _part1(sigils, "frag4-other") != _part1(sigils, "frag4-auto")
