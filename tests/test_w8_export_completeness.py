"""X3/X4 completeness: retained identities determine required export inputs.

No numerical engine is needed: Preparic's attract-jax mode is NumPy energy-only.
"""
from copy import deepcopy

import pytest

from test_w8_export_seam_contract import api, snapshot  # noqa: F401


@pytest.fixture
def complete_snapshot(snapshot):
    # Start with every §1 adjacent pair, so a removed row is the only defect.
    return snapshot


def _assert_non_overridable(snapshot):
    snapshot.update(proceed_anyway_required=True, proceed_anyway=True)
    try:
        bundle = api().build_export_bundle(snapshot)
    except ValueError:
        return
    assert bundle["hard_blocks"], "incomplete retained evidence needs a hard block"
    assert bundle["ready"] is False


def test_complete_adjacent_kept_pair_evidence_is_ready(complete_snapshot):
    bundle = api().build_export_bundle(complete_snapshot)
    assert not bundle["hard_blocks"]
    assert bundle["ready"]
    assert {(row["down"], row["up"]) for row in bundle["grow_thresholds"]} == {
        (1, 2), (2, 3)
    }


@pytest.mark.parametrize("omitted", [(1, 2), (2, 3), "all"],
                         ids=["first-pair", "last-pair", "all-pairs"])
def test_omitted_pair_between_explicitly_kept_adjacent_ids_blocks_export(
    complete_snapshot, omitted,
):
    complete_snapshot["kept_pairs"] = [
        row for row in complete_snapshot["kept_pairs"]
        if omitted != "all" and (row["down"], row["up"]) != omitted
    ]
    _assert_non_overridable(complete_snapshot)


def test_false_pair_flag_cannot_hide_pair_between_kept_fragments(complete_snapshot):
    # §1 defines retention from the fragment set, not from a pair-row override.
    complete_snapshot["kept_pairs"][1]["kept"] = False
    _assert_non_overridable(complete_snapshot)


@pytest.mark.parametrize("evidence", ["missing", None, {}, [], "all-clipped"],
                         ids=["absent", "null", "empty-map", "empty-list", "all-clipped"])
def test_empty_or_missing_kept_evidence_cannot_admit_selected_candidates(
    complete_snapshot, evidence,
):
    # Remove pair rows deliberately: rejection must not depend on validating an
    # otherwise unrelated supplied pair against the now-empty retained set.
    complete_snapshot["kept_pairs"] = []
    if evidence == "missing":
        del complete_snapshot["kept_fragments"]
    elif evidence == "all-clipped":
        for row in complete_snapshot["kept_fragments"].values():
            row["clipped"] = True
    else:
        complete_snapshot["kept_fragments"] = deepcopy(evidence)
    _assert_non_overridable(complete_snapshot)


@pytest.mark.parametrize("fragment,nucleotide", [(1, "first"), (3, "second")],
                         ids=["clip-right-to-first-fragment", "clip-left-to-last-fragment"])
def test_single_terminal_kept_fragment_allows_empty_pair_set(
    complete_snapshot, fragment, nucleotide,
):
    # With n=3, clip=(0,2) or (2,0) keeps one fragment. The selected
    # terminal RNA residue has no second candidate beyond the RNA endpoint.
    key = "frag-1-AC" if fragment == 1 else "frag-3-GU"
    complete_snapshot["kept_fragments"] = {
        key: complete_snapshot["kept_fragments"][key]
    }
    anchor = deepcopy(complete_snapshot["selected_anchors"][0 if fragment == 1 else 1])
    anchor["nucleotide"] = nucleotide
    complete_snapshot["selected_anchors"] = [anchor]
    complete_snapshot["reference_candidates"] = [dict(
        fragment=fragment, sequence=anchor["sequence"],
        nucleotide=nucleotide, ovrmsd="0.80",
    )]
    complete_snapshot["kept_pairs"] = []
    complete_snapshot["handoff"]["fragment_rows"] = [key]
    bundle = api().build_export_bundle(complete_snapshot)
    assert not bundle["hard_blocks"]
    assert bundle["ready"]
    assert bundle["grow_thresholds"] == []
    assert len(bundle["anchor_refe_actions"]) == 1


def test_nonadjacent_kept_ids_do_not_invent_a_grow_pair(complete_snapshot):
    # §1's Alaric numbering permits fragment holes. This is not a claim that
    # Preparic permits gapped RNA ranges: its P2.1 restriction still applies.
    del complete_snapshot["kept_fragments"]["frag-2-CG"]
    complete_snapshot["kept_pairs"] = []
    for anchor, nucleotide in zip(complete_snapshot["selected_anchors"], ["first", "second"]):
        anchor["nucleotide"] = nucleotide
    complete_snapshot["reference_candidates"] = [dict(
        fragment=anchor["fragment"], sequence=anchor["sequence"],
        nucleotide=anchor["nucleotide"], ovrmsd="0.80",
    ) for anchor in complete_snapshot["selected_anchors"]]
    complete_snapshot["handoff"]["fragment_rows"] = list(complete_snapshot["kept_fragments"])
    bundle = api().build_export_bundle(complete_snapshot)
    assert not bundle["hard_blocks"]
    assert bundle["ready"]
    assert bundle["grow_thresholds"] == []
