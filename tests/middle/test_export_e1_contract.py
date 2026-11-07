"""Export E1 action/constraints boundary acceptance (C1--C3)."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import yaml

from alaric.middle.deploy import generate_run_sh
from alaric.middle.errors import GraphError, ResolveError, SchemaError
from alaric.middle.graph import ActionGraph
from alaric.middle.project import Project
from alaric.middle.resolve import ResolvedAction, _pair_thresholds, final_fragment, final_sequence, resolve_sequence
from alaric.middle.schema import DEPENDENCY_FIELDS, OUTPUT_KIND, normalize_action
from alaric.middle.sigil import action_sigil, compute_project_sigils


GOLDEN = Path(__file__).with_name("data") / "export-e1-two-input.json"


def constraints():
    return {"schema": 1, "frag1": {"sequence": "GU"}, "frag2": {"sequence": "UA"},
            "pairs": [{"down": 1, "up": 2, "sequence": "GUA", "crmsd": "0.25", "ovrmsd": "0.75"}]}


def project(root, payload=None):
    (root / "DATA").mkdir(exist_ok=True)
    (root / "DATA" / "constraints.json").write_text(json.dumps(constraints() if payload is None else payload))
    (root / "DATA" / "pdbcode.txt").write_text("1abc\n")
    (root / "DATA" / "reference.pdb").write_bytes(b"HEADER exact reference\n")
    return Project(root)


def action(root, name, data):
    directory = root / name
    directory.mkdir(exist_ok=True)
    (directory / "alaric.yaml").write_text(yaml.safe_dump(data, sort_keys=False))


def mask_project(root, *, input="source-one", side=1):
    result = project(root)
    for name, threshold in (("source-one", .5), ("source-two", .6), ("unrelated", .7)):
        action(root, name, {"action": "anchor-refe", "fragment": 5, "sequence": "GU",
                           "exclude": ["1abc"], "nucleotide": "first", "ovrmsd": threshold})
    action(root, "paired", {"action": "mask-common-conformer", "input1": "source-one", "input2": "source-two"})
    action(root, "filtered", {"action": "apply-mask", "input": input, "mask_input": "paired", "mask": side})
    return result


@pytest.mark.parametrize("side", [1, 2])
def test_apply_mask_schema_graph_metadata_and_backend(tmp_path, side):
    source = "source-one" if side == 1 else "source-two"
    proj = mask_project(tmp_path, input=source, side=side)
    graph = ActionGraph(proj)
    resolved = graph.build()
    out = resolved["filtered"]
    assert DEPENDENCY_FIELDS["apply-mask"] == {"input": "pose", "mask_input": "mask"}
    assert OUTPUT_KIND["apply-mask"] == "pose"
    assert graph.dependency_names(graph.specs["filtered"]) == {"input": source, "mask_input": "paired"}
    assert final_fragment(out) == 5
    assert final_sequence(out) == "GU"
    compute_project_sigils(proj)
    for location in ("local", "remote"):
        body = generate_run_sh(proj, out, location)
        assert "apply_mask" in body or "apply-mask" in body
        assert f"--mask {side}" in body or f"mask{side}.npy" in body
        assert "/mask.npy" not in body
        assert "{{" not in body


@pytest.mark.parametrize("value", [True, False, 0, 3, -1, 1.0, "1", None])
def test_apply_mask_selector_is_strict_integer(tmp_path, value):
    with pytest.raises(SchemaError):
        normalize_action("out", tmp_path, {"action": "apply-mask", "input": "one", "mask_input": "pair", "mask": value})


@pytest.mark.parametrize("field", ["input", "mask_input", "mask"])
def test_apply_mask_required_keys(tmp_path, field):
    data = {"action": "apply-mask", "input": "one", "mask_input": "pair", "mask": 1}
    del data[field]
    with pytest.raises(SchemaError):
        normalize_action("out", tmp_path, data)


@pytest.mark.parametrize("input,side", [("source-two", 1), ("source-one", 2), ("unrelated", 1)])
def test_apply_mask_refuses_wrong_source_or_side(tmp_path, input, side):
    with pytest.raises(GraphError):
        ActionGraph(mask_project(tmp_path, input=input, side=side)).build()


def test_apply_mask_refuses_pose_as_mask_dependency(tmp_path):
    proj = mask_project(tmp_path)
    action(tmp_path, "filtered", {"action": "apply-mask", "input": "source-one", "mask_input": "source-two", "mask": 1})
    with pytest.raises(GraphError):
        ActionGraph(proj).build()


def test_apply_mask_refuses_ordinary_mask_instead_of_paired_mask(tmp_path):
    proj = mask_project(tmp_path)
    action(tmp_path, "score", {"action": "rmsd", "input": "source-one", "fragment": 5, "exclude": ["1abc"]})
    action(tmp_path, "ordinary-mask", {"action": "mask", "input": "source-one", "score_input": "score", "threshold": 1.0})
    action(tmp_path, "filtered", {"action": "apply-mask", "input": "source-one", "mask_input": "ordinary-mask", "mask": 1})
    with pytest.raises(GraphError):
        ActionGraph(proj).build()


def test_apply_mask_refuses_incompatible_fragment_inputs(tmp_path):
    proj = mask_project(tmp_path)
    action(tmp_path, "source-two", {"action": "anchor-refe", "fragment": 6, "sequence": "UA", "exclude": ["1abc"], "nucleotide": "first", "ovrmsd": .6})
    with pytest.raises(GraphError):
        ActionGraph(proj).build()


def test_apply_mask_sigil_contains_selector_and_both_dependencies(tmp_path):
    out = ActionGraph(mask_project(tmp_path)).build()["filtered"]
    deps = {"input": "source-sigil", "mask_input": "mask-sigil"}
    base, record = action_sigil(out, deps)
    assert record["mask"] == 1
    assert record["dependencies"] == deps
    assert action_sigil(replace(out, params={**out.params, "mask": 2}), deps)[0] != base
    for field in deps:
        assert action_sigil(out, {**deps, field: "changed"})[0] != base


def test_single_identity_has_one_dependency_and_inherits_metadata(tmp_path):
    from alaric.middle.pool_graph import _pool_kind

    proj = project(tmp_path)
    action(tmp_path, "source", {"action": "anchor-refe", "fragment": 5, "sequence": "GU", "exclude": ["1abc"], "nucleotide": "first", "ovrmsd": .5})
    action(tmp_path, "unique", {"action": "identity", "input1": "source"})
    graph = ActionGraph(proj)
    out = graph.build()["unique"]
    assert graph.dependency_names(graph.specs["unique"]) == {"input1": "source"}
    assert "input2" not in out.params
    assert final_fragment(out) == 5 and final_sequence(out) == "GU"
    assert _pool_kind(out) == "dedup"
    sigils = compute_project_sigils(proj)
    record = json.loads((proj.parameters_dir / sigils["unique"]).read_text())
    assert set(record["dependencies"]) == {"input1"}
    assert "input2" not in record
    for location in ("local", "remote"):
        body = generate_run_sh(proj, out, location)
        assert "--single-input" in body
        assert "{{" not in body


def test_two_input_identity_saved_sigil_fixture_is_unchanged():
    golden = json.loads(GOLDEN.read_text())
    dep1 = ResolvedAction("one", Path("/ignored/one"), "anchor-refe", {"fragment": 5, "sequence": "GU"}, {})
    dep2 = replace(dep1, name="two", path=Path("/ignored/two"))
    out = ResolvedAction("frag5-merged", Path("/ignored/merged"), "identity", {"action": "identity", "input1": dep1, "input2": dep2}, {})
    actual, record = action_sigil(out, {"input1": "1111111111-null", "input2": "2222222222-null"})
    assert actual == golden["sigil"]
    assert record == golden["parameter_record"]


def test_two_input_same_source_identity_has_versioned_semantics_only_for_equal_dependency_sigils():
    golden = json.loads(GOLDEN.read_text())
    dep1 = ResolvedAction("one", Path("/ignored/one"), "anchor-refe", {"fragment": 5, "sequence": "GU"}, {})
    # Names/paths differ deliberately: identical dependency identities, rather
    # than source directory spelling, define whether cached semantics change.
    dep2 = replace(dep1, name="two", path=Path("/ignored/two"))
    out = ResolvedAction("frag5-merged", Path("/ignored/merged"), "identity", {"action": "identity", "input1": dep1, "input2": dep2}, {})
    equal = {"input1": "1111111111-null", "input2": "1111111111-null"}
    same_sigil, same_record = action_sigil(out, equal)
    assert same_record == {"action": "identity", "same_input_dedup_version": 1, "dependencies": equal}

    distinct_sigil, distinct_record = action_sigil(
        out, {"input1": "1111111111-null", "input2": "2222222222-null"})
    assert distinct_sigil == golden["sigil"]
    assert distinct_record == golden["parameter_record"]
    assert same_sigil.split("-", 1)[0] != distinct_sigil.split("-", 1)[0]

    single = replace(out, params={"action": "identity", "input1": dep1})
    _, single_record = action_sigil(single, {"input1": "1111111111-null"})
    assert single_record == {"action": "identity", "dependencies": {"input1": "1111111111-null"}}


def test_schema1_constraints_lowercase_and_uppercase_aliases(tmp_path):
    payload = constraints()
    proj = project(tmp_path, payload)
    assert resolve_sequence(proj, 1) == "GU"
    assert _pair_thresholds(proj, 1, 2) == (.25, .75)
    assert _pair_thresholds(proj, 2, 1) == (.25, .75)
    row = payload["pairs"][0]
    row["cRMSD"], row["ovRMSD"] = row.pop("crmsd"), row.pop("ovrmsd")
    project(tmp_path, payload)
    assert _pair_thresholds(proj, 1, 2) == (.25, .75)
    row["crmsd"], row["ovrmsd"] = ".25", ".75"
    project(tmp_path, payload)
    assert _pair_thresholds(proj, 1, 2) == (.25, .75)


def bad_constraints():
    cases = []
    for schema in (None, 0, 2, True, "1"):
        value = constraints()
        if schema is None:
            value.pop("schema")
        else:
            value["schema"] = schema
        cases.append(value)
    for key in ("frag0", "frag01", "frag-1", "frag1-SEQ", "unexpected"):
        value = constraints(); value[key] = {"sequence": "GU"}; cases.append(value)
    for sequence in ("G", "GUA", "GX", None, 12):
        value = constraints(); value["frag1"]["sequence"] = sequence; cases.append(value)
    for field, bad_values in (("down", [True, 0, 1.5]), ("up", [1, 3, False]),
                              ("sequence", ["GUU", "GU", None]),
                              ("crmsd", [True, "nan", "inf", "0", "-0.01", "0.123"]),
                              ("ovrmsd", [False, "NaN", "Infinity", 0, -.1])):
        for bad in bad_values:
            value = constraints(); value["pairs"][0][field] = bad; cases.append(value)
    for field in ("down", "up", "sequence", "crmsd", "ovrmsd"):
        value = constraints(); del value["pairs"][0][field]; cases.append(value)
    value = constraints(); value["pairs"].append(deepcopy(value["pairs"][0])); cases.append(value)
    value = constraints(); value["pairs"] = {}; cases.append(value)
    value = constraints(); value["pairs"][0]["cRMSD"] = ".26"; cases.append(value)
    value = constraints(); value["pairs"][0]["ovRMSD"] = ".76"; cases.append(value)
    return cases


@pytest.mark.parametrize("payload", bad_constraints())
def test_invalid_constraints_refused_by_both_consumers(tmp_path, payload):
    proj = project(tmp_path, payload)
    for consume in (lambda: resolve_sequence(proj, 1), lambda: _pair_thresholds(proj, 1, 2)):
        with pytest.raises(ResolveError):
            consume()


def test_missing_requested_constraints_entry_does_not_use_fallback(tmp_path):
    payload = constraints(); del payload["frag2"]; payload["pairs"] = []
    proj = project(tmp_path, payload)
    (proj.data_dir / "sequence.txt").write_text("GUA\n")
    with pytest.raises(ResolveError):
        resolve_sequence(proj, 2)
    with pytest.raises(ResolveError):
        _pair_thresholds(proj, 1, 2)
