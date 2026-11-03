"""Final W8 X4/X5 export admission evidence and kept-grow-pair oracles.

Run from the repository root:
  conda run --live-stream -n alaric python -m pytest -q tests/test_w8_export_admission_followup.py
"""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from test_w8_export_seam_contract import snapshot  # noqa: E402,F401


def _bundle(snapshot):
    from alaric.export_bundle import build_export_bundle

    return build_export_bundle(snapshot)


def _refused(snapshot):
    # Preserve the required flag being tested: setting it to True here would
    # accidentally erase missing/non-bool evidence and make this oracle vacuous.
    snapshot["proceed_anyway"] = True
    try:
        bundle = _bundle(snapshot)
    except ValueError:
        return
    assert bundle["hard_blocks"], "invalid export evidence must be non-overridable"
    assert bundle["ready"] is False


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false", "", [], {}],
                         ids=["null", "zero", "one", "string-true", "string-false", "empty-string", "list", "mapping"])
def test_proceed_anyway_required_rejects_non_boolean_evidence(snapshot, value):
    assert _bundle(snapshot)["ready"]
    snapshot["proceed_anyway_required"] = value
    _refused(snapshot)


def test_proceed_anyway_required_rejects_missing_evidence(snapshot):
    assert _bundle(snapshot)["ready"]
    del snapshot["proceed_anyway_required"]
    _refused(snapshot)


@pytest.mark.parametrize("required,acknowledged,ready", [
    (False, False, True), (False, True, True), (True, False, False), (True, True, True),
])
def test_real_boolean_admission_evidence_preserves_explicit_acknowledgment_gate(
    snapshot, required, acknowledged, ready,
):
    snapshot.update(proceed_anyway_required=required, proceed_anyway=acknowledged)
    bundle = _bundle(snapshot)
    assert not bundle["hard_blocks"]
    assert bundle["proceed_anyway_required"] is required
    assert bundle["ready"] is ready


@pytest.mark.parametrize("field", ["crmsd", "ovrmsd", "cRMSD", "ovRMSD"])
@pytest.mark.parametrize("value", [None, "auto", float("nan"), float("inf"), -float("inf"),
                                  "NaN", "inf", 0, -0.1, True, False, "invalid", [], {}],
                         ids=["null", "auto", "nan", "inf", "minus-inf", "string-nan", "string-inf",
                              "zero", "negative", "true", "false", "invalid-string", "list", "mapping"])
def test_every_kept_pair_threshold_must_be_finite_positive_numeric(snapshot, field, value):
    assert _bundle(snapshot)["ready"]
    pair = snapshot["kept_pairs"][0]
    canonical = "crmsd" if field in {"crmsd", "cRMSD"} else "ovrmsd"
    del pair[canonical]
    pair[field] = value
    _refused(snapshot)


@pytest.mark.parametrize("field", ["crmsd", "ovrmsd"])
def test_every_kept_pair_requires_both_thresholds(snapshot, field):
    assert _bundle(snapshot)["ready"]
    del snapshot["kept_pairs"][0][field]
    _refused(snapshot)


@pytest.mark.parametrize("camel_case", [False, True])
@pytest.mark.parametrize("numeric", [False, True])
def test_valid_finite_positive_kept_pair_thresholds_and_aliases_pass(snapshot, camel_case, numeric):
    pair = snapshot["kept_pairs"][0]
    crmsd, ovrmsd = pair.pop("crmsd"), pair.pop("ovrmsd")
    pair["cRMSD" if camel_case else "crmsd"] = float(crmsd) if numeric else crmsd
    pair["ovRMSD" if camel_case else "ovrmsd"] = float(ovrmsd) if numeric else ovrmsd
    bundle = _bundle(snapshot)
    assert not bundle["hard_blocks"]
    assert bundle["ready"]
    assert len(bundle["grow_thresholds"]) == 2


@pytest.mark.parametrize("field", ["down", "up"])
@pytest.mark.parametrize("missing", [False, True], ids=["null", "absent"])
def test_kept_pair_requires_complete_addressing(snapshot, field, missing):
    assert _bundle(snapshot)["ready"]
    if missing:
        del snapshot["kept_pairs"][0][field]
    else:
        snapshot["kept_pairs"][0][field] = None
    _refused(snapshot)


@pytest.mark.parametrize("changes", [
    {"down": True}, {"up": True}, {"down": 1.0}, {"up": 2.0},
    {"down": "1"}, {"up": "2"}, {"down": 0}, {"up": 0},
    {"down": -1}, {"up": -2}, {"up": 1}, {"down": 2, "up": 1}, {"up": 3},
    {"sequence": None}, {"sequence": ""}, {"sequence": "AC"}, {"sequence": "ACGU"},
    {"sequence": "AXG"}, {"sequence": ["A", "C", "G"]},
    {"sequence": "UUU"},
], ids=["bool-down", "bool-up", "float-down", "float-up", "string-down", "string-up",
        "zero-down", "zero-up", "negative-down", "negative-up", "same-fragment", "reversed",
        "non-adjacent", "null-sequence", "empty-sequence", "short-sequence", "long-sequence", "invalid-base",
        "sequence-list", "kept-map-sequence-mismatch"])
def test_kept_pair_addressing_is_valid_adjacent_and_matches_kept_fragments(snapshot, changes):
    assert _bundle(snapshot)["ready"]
    snapshot["kept_pairs"][0].update(changes)
    _refused(snapshot)


def test_all_kept_rows_are_validated_even_after_a_valid_pair(snapshot):
    invalid = deepcopy(snapshot["kept_pairs"][0])
    invalid["crmsd"] = "auto"
    snapshot["kept_pairs"].append(invalid)
    _refused(snapshot)


def test_kept_pair_sequence_is_optional_when_addresses_and_thresholds_are_valid(snapshot):
    del snapshot["kept_pairs"][0]["sequence"]
    bundle = _bundle(snapshot)
    assert not bundle["hard_blocks"]
    assert bundle["ready"]
    assert len(bundle["grow_thresholds"]) == 2


def test_explicit_unkept_pair_does_not_require_export_thresholds(snapshot):
    snapshot["kept_pairs"].append({"kept": False, "down": 90, "up": 91,
                                   "sequence": "AAA", "crmsd": "auto", "ovrmsd": None})
    bundle = _bundle(snapshot)
    assert not bundle["hard_blocks"]
    assert bundle["ready"]
    assert len(bundle["grow_thresholds"]) == 2


def test_export_fraction_must_match_runtime_constant_even_when_preparation_agrees(snapshot):
    snapshot["min_cell_fraction"] = 0.5
    snapshot["prepared_min_cell_fraction"] = 0.5
    with pytest.raises(ValueError, match="canonical default"):
        _bundle(snapshot)


@pytest.mark.parametrize("down,up", [(90, 91), (1, 2)])
def test_kept_pair_addresses_must_exist_in_kept_fragments_without_sequence(snapshot, down, up):
    pair = snapshot["kept_pairs"][0]
    pair.pop("sequence")
    pair.update(down=down, up=up)
    if (down, up) == (1, 2):
        del snapshot["kept_fragments"]["frag-2-CG"]
    _refused(snapshot)
