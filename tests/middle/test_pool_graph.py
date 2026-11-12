from __future__ import annotations

from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from alaric.middle.errors import PoolGraphError
from alaric.middle.pool_graph import PoolGraph
from alaric.middle.project import Project


def _mk(root: Path, name: str, spec: dict) -> None:
    path = root / name
    path.mkdir()
    (path / "alaric.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))


def _anchor(fragment: int, nucleotide: str = "first") -> dict:
    return {
        "action": "anchor-refe", "fragment": fragment, "sequence": "GU", "exclude": "1abc",
        "nucleotide": nucleotide, "ovrmsd": 1.0,
    }


def _grow(input_name: str, fragment: int, direction: str = "forward") -> dict:
    return {
        "action": "grow", "input": input_name, "fragment": fragment, "sequence": "UU",
        "exclude": "1abc", "direction": direction, "crmsd": 0.25, "ovrmsd": 0.75,
    }


def _filter(input_name: str, fragment: int, threshold: float = -1.0,
            sequence: str = "UU") -> dict:
    return {
        "action": "energy-filter",
        "input": input_name,
        "fragment": fragment,
        "sequence": sequence,
        "exclude": "1abc",
        "protein": "range-2",
        "threshold": threshold,
        "epsilon": 15,
        "cascade": [],
        "min_cascade": 0,
    }


def _data(root: Path) -> None:
    data = root / "DATA"
    data.mkdir()
    (data / "range-2-aa.pdb").write_text("HEADER aa\n")
    (data / "reference.pdb").write_text("HEADER reference\n")
    (data / "anchor.yaml").write_text("ovrmsd: 1.0\n")


def _full_project(root: Path) -> None:
    """Mirror the 1b7f-RUN topology: grow chain with a merge, a dedup, and a
    disconnected branch, all hermetic (explicit params, no shared fixture)."""
    _data(root)
    # frag4: two anchors -> merge -> filter (the grow source for frag5)
    _mk(root, "frag4-anchor-fwd", _anchor(4, "first"))
    _mk(root, "frag4-anchor-bwd", _anchor(4, "second"))
    _mk(root, "frag4-merged", {"action": "identity", "input1": "frag4-anchor-fwd", "input2": "frag4-anchor-bwd"})
    _mk(root, "frag4-merged-filter", _filter("frag4-merged", 4, sequence="GU"))
    # frag5: grow -> filter, plus a frag5 anchor, merged together
    _mk(root, "frag5-fwd", _grow("frag4-merged-filter", 5))
    _mk(root, "frag5-fwd-filter", _filter("frag5-fwd", 5))
    _mk(root, "frag5-anchor", _anchor(5, "second"))
    _mk(root, "frag5-merged", {"action": "identity", "input1": "frag5-fwd-filter", "input2": "frag5-anchor"})
    # frag6: grow -> filter
    _mk(root, "frag6-fwd", _grow("frag5-merged", 6))
    _mk(root, "frag6-fwd-filter", _filter("frag6-fwd", 6))
    # frag7: grow -> filter -> dedup (identity self-merge)
    _mk(root, "frag7-fwd", _grow("frag6-fwd-filter", 7))
    _mk(root, "frag7-fwd-filter", _filter("frag7-fwd", 7))
    _mk(root, "frag7-fwd-filter-unique", {"action": "identity", "input1": "frag7-fwd-filter", "input2": "frag7-fwd-filter"})
    # frag8: grow endpoint (no filter)
    _mk(root, "frag8-fwd", _grow("frag7-fwd-filter-unique", 8))
    # frag10: anchor + filter, not wired into any grow -> disconnected
    _mk(root, "frag10-anchor", _anchor(10, "second"))
    _mk(root, "frag10-anchor-filter", _filter("frag10-anchor", 10, sequence="GU"))


def test_representatives_match_lineage(tmp_path):
    _full_project(tmp_path)
    graph = PoolGraph(Project(tmp_path))
    assert graph.representatives == {
        4: "frag4-merged-filter",
        5: "frag5-merged",
        6: "frag6-fwd-filter",
        7: "frag7-fwd-filter-unique",
        8: "frag8-fwd",
    }
    # frag10 is an anchor+filter not wired into any grow -> skipped, not an error.
    assert 10 in graph.skipped


def test_frag4_to_frag5_composition_routes_through_merge_input1(tmp_path):
    _full_project(tmp_path)
    graph = PoolGraph(Project(tmp_path))
    link = next(l for l in graph.links() if l["source_fragment"] == 4 and l["target_fragment"] == 5)
    assert link["join_pool"] == "frag4-merged-filter"
    assert link["source_to_join"] == []  # source rep IS the grow source
    # target rep maps up through the merge's input1 branch (not the frag5-anchor branch)
    steps = [(s["pool"], s["to"]) for s in link["target_to_join"]]
    assert steps == [
        ("frag5-merged", "frag5-fwd-filter"),
        ("frag5-fwd-filter", "frag5-fwd"),
        ("frag5-fwd", "frag4-merged-filter"),
    ]
    assert link["target_to_join"][0]["array"] == "map-1.npy"  # merge step
    assert link["target_to_join"][1]["array"] == "provenance.npy"  # filter step


def test_dedup_records_both_maps_but_one_lineage(tmp_path):
    _full_project(tmp_path)
    graph = PoolGraph(Project(tmp_path))
    node = graph.pools["frag7-fwd-filter-unique"]
    assert node.kind == "dedup"
    # on disk a dedup owns both map-1 and map-2 (redundant columns of a self-merge)
    assert {e.array for e in node.parents} == {"map-1.npy", "map-2.npy"}
    # but it collapses to a single lineage step, so the fragment is unambiguous
    assert graph.representatives[7] == "frag7-fwd-filter-unique"


def test_every_link_lands_in_join_pool_space(tmp_path):
    _full_project(tmp_path)
    graph = PoolGraph(Project(tmp_path))
    links = graph.links()
    # one link per grow that connects two represented fragments (frag5..frag8)
    assert {(l["source_fragment"], l["target_fragment"]) for l in links} == {
        (4, 5), (5, 6), (6, 7), (7, 8),
    }
    for link in links:
        # both paths must terminate at the join pool
        if link["target_to_join"]:
            assert link["target_to_join"][-1]["to"] == link["join_pool"]
        for path in (link["target_to_join"], link["source_to_join"]):
            assert all(s["array"] in {"provenance.npy", "map-1.npy", "map-2.npy"} for s in path)


def _ambiguous_project(root: Path) -> None:
    """A frag4 root grown to frag5, then frag5 filtered two independent ways."""
    _data(root)
    _mk(root, "frag4-anchor", _anchor(4))
    _mk(root, "frag5-fwd", _grow("frag4-anchor", 5))
    _mk(root, "frag5-filter-a", _filter("frag5-fwd", 5, -1.0))
    _mk(root, "frag5-filter-b", _filter("frag5-fwd", 5, -2.0))


def test_ambiguous_fragment_raises(tmp_path):
    _ambiguous_project(tmp_path)
    with pytest.raises(PoolGraphError, match="multiple chain lineages"):
        PoolGraph(Project(tmp_path))


def test_ambiguity_message_points_at_select(tmp_path):
    _ambiguous_project(tmp_path)
    with pytest.raises(PoolGraphError, match="Use --select"):
        PoolGraph(Project(tmp_path))


def test_select_within_one_lineage_picks_a_sink(tmp_path):
    """Two sinks, one lineage: --select just names the representative, dropping nothing."""
    _ambiguous_project(tmp_path)
    graph = PoolGraph(Project(tmp_path), select=["frag5-filter-a"])
    assert graph.representatives == {4: "frag4-anchor", 5: "frag5-filter-a"}
    assert graph.dropped == {}


def test_select_sibling_sink_that_cannot_compose_raises(tmp_path):
    """frag6 is grown from the *other* sink, so selecting this one is uncomposable."""
    _ambiguous_project(tmp_path)
    _mk(tmp_path, "frag6-fwd", _grow("frag5-filter-b", 6))
    with pytest.raises(PoolGraphError, match="does not descend from"):
        PoolGraph(Project(tmp_path), select=["frag5-filter-a"])


# -- fragments visited twice (sweeps) -------------------------------------


def _roundtrip_project(root: Path) -> None:
    """anchor frag2 -> grow backward into frag1 -> grow forward into frag2 again.

    frag2 ends up with two disjoint lineages: A from the anchor, B regrown from frag1.
    """
    _data(root)
    _mk(root, "frag2-anchor", _anchor(2))                       # lineage A
    _mk(root, "frag2-filtered", _filter("frag2-anchor", 2, sequence="GU"))
    _mk(root, "frag1-bwd", _grow("frag2-filtered", 1, "backward"))
    _mk(root, "frag1-bwd-filter", _filter("frag1-bwd", 1))
    _mk(root, "frag1-filtered-uniq", {"action": "identity", "input1": "frag1-bwd-filter",
                                      "input2": "frag1-bwd-filter"})
    _mk(root, "frag2-fwd", _grow("frag1-filtered-uniq", 2))     # lineage B


def test_roundtrip_is_ambiguous_without_select(tmp_path):
    _roundtrip_project(tmp_path)
    with pytest.raises(PoolGraphError, match="frag2-filtered, frag2-fwd"):
        PoolGraph(Project(tmp_path))


def test_select_drops_the_other_lineage(tmp_path):
    _roundtrip_project(tmp_path)
    graph = PoolGraph(Project(tmp_path), select=["frag2-fwd"])
    assert graph.representatives == {1: "frag1-filtered-uniq", 2: "frag2-fwd"}
    assert sorted(graph.dropped) == ["frag2-anchor", "frag2-filtered"]
    # frag1-bwd is orphaned by the drop (its grow source is gone) but its lineage is
    # still the source side of the forward grow, so it must survive
    assert "frag1-bwd" in graph.pools
    assert graph.pools["frag1-bwd"].parents == []
    links = graph.links()
    assert [(l["source_rep"], l["target_rep"]) for l in links] == [
        ("frag1-filtered-uniq", "frag2-fwd")
    ]
    assert links[0]["join_pool"] == "frag1-filtered-uniq"


def test_select_orphans_a_competing_downstream_lineage(tmp_path):
    """Something else grown off the dropped lineage falls out without being named."""
    _roundtrip_project(tmp_path)
    _mk(tmp_path, "frag3-alt", _grow("frag2-filtered", 3))   # off lineage A
    _mk(tmp_path, "frag3-fwd", _grow("frag2-fwd", 3))        # off lineage B
    graph = PoolGraph(Project(tmp_path), select=["frag2-fwd"])
    assert graph.representatives == {
        1: "frag1-filtered-uniq", 2: "frag2-fwd", 3: "frag3-fwd",
    }
    # frag3-alt is not dropped, just no longer anchored -> loses to frag3-fwd
    assert "frag3-alt" not in graph.dropped


# -- orphans left behind by a drop ----------------------------------------


def _two_lineage_project(root: Path, *, downstream: str | None) -> None:
    """frag5 reached both by a grow from frag4 and by its own anchor.

    ``downstream`` optionally grows something off the anchor-lineage side of frag5,
    directly ("grow") or through a filter ("filter").
    """
    _data(root)
    _mk(root, "frag4-anchor", _anchor(4))
    _mk(root, "frag5-fwd", _grow("frag4-anchor", 5))       # lineage A
    _mk(root, "frag5-anchor", _anchor(5, "second"))        # lineage B
    _mk(root, "frag6-fwd", _grow("frag5-anchor", 6))       # grown off lineage B
    if downstream == "grow":
        _mk(root, "frag7-fwd", _grow("frag6-fwd", 7))
    elif downstream == "filter":
        _mk(root, "frag6-fwd-filter", _filter("frag6-fwd", 6))
        _mk(root, "frag7-fwd", _grow("frag6-fwd-filter", 7))


def test_orphan_with_nothing_downstream_is_skipped(tmp_path):
    _two_lineage_project(tmp_path, downstream=None)
    graph = PoolGraph(Project(tmp_path), select=["frag5-fwd"])
    assert graph.representatives == {4: "frag4-anchor", 5: "frag5-fwd"}
    assert "orphaned by --select" in graph.skipped[6]


def test_orphan_grown_onward_leaves_a_gap(tmp_path):
    _two_lineage_project(tmp_path, downstream="grow")
    with pytest.raises(PoolGraphError, match="the chain has a gap"):
        PoolGraph(Project(tmp_path), select=["frag5-fwd"])


def test_gap_is_caught_through_an_intervening_filter(tmp_path):
    """The orphan itself grows nothing -- a filter sits between it and the next grow."""
    _two_lineage_project(tmp_path, downstream="filter")
    with pytest.raises(PoolGraphError, match="the chain has a gap"):
        PoolGraph(Project(tmp_path), select=["frag5-fwd"])


def test_target_is_the_other_remedy_for_a_gap(tmp_path):
    _two_lineage_project(tmp_path, downstream="filter")
    graph = PoolGraph(Project(tmp_path), targets=["frag5-fwd"])
    assert graph.representatives == {4: "frag4-anchor", 5: "frag5-fwd"}
    assert graph.dropped == {}


# -- selection validation and provenance ----------------------------------


def test_select_unknown_pool_raises(tmp_path):
    _roundtrip_project(tmp_path)
    with pytest.raises(PoolGraphError, match="unknown pool"):
        PoolGraph(Project(tmp_path), select=["nope"])


def test_select_two_pools_in_one_fragment_raises(tmp_path):
    _roundtrip_project(tmp_path)
    with pytest.raises(PoolGraphError, match="select at most one pool per fragment"):
        PoolGraph(Project(tmp_path), select=["frag2-fwd", "frag2-filtered"])


def test_json_records_the_selection(tmp_path):
    _roundtrip_project(tmp_path)
    data = PoolGraph(Project(tmp_path), select=["frag2-fwd"]).to_dict()
    assert data["selected"] == ["frag2-fwd"]
    assert sorted(data["dropped_pools"]) == ["frag2-anchor", "frag2-filtered"]
    assert "fragment 2" in data["dropped_pools"]["frag2-filtered"]
    assert "frag2-filtered" not in data["pools"]


def test_unselected_projects_carry_empty_provenance(tmp_path):
    _full_project(tmp_path)
    data = PoolGraph(Project(tmp_path)).to_dict()
    assert data["selected"] == []
    assert data["dropped_pools"] == {}


# -- common-conformer masks (unfixed: range-2 / range-3 molecules) ---------


def _common_conformer_project(root: Path) -> None:
    """Two molecules sharing frag4: range-3 grows forward from a frag3 anchor,
    range-2 grows backward from a frag5 anchor. A common-conformer mask pairs the
    two frag4 pools; range-2 grows on (into frag3) from its masked side."""
    _data(root)
    _mk(root, "frag3-anchor", _anchor(3, "first"))
    _mk(root, "frag4-range3", _grow("frag3-anchor", 4))
    _mk(root, "frag5-anchor", _anchor(5, "second"))
    _mk(root, "frag4-range2", _grow("frag5-anchor", 4, "backward"))
    _mk(root, "frag4-mask", {
        "action": "mask-common-conformer", "input1": "frag4-range2", "input2": "frag4-range3",
    })
    _mk(root, "frag4-range2-cc", {
        "action": "apply-mask", "input": "frag4-range2", "mask_input": "frag4-mask", "mask": 1,
    })
    _mk(root, "frag4-range3-cc", {
        "action": "apply-mask", "input": "frag4-range3", "mask_input": "frag4-mask", "mask": 2,
    })
    _mk(root, "frag3-range2", _grow("frag4-range2-cc", 3, "backward"))


def test_common_conformer_mask_is_a_root(tmp_path):
    """Targeting one side of a paired mask must not pull in the other molecule."""
    _common_conformer_project(tmp_path)
    graph = PoolGraph(Project(tmp_path), targets=["frag3-range2"])
    assert sorted(graph.pools) == [
        "frag3-range2", "frag4-range2", "frag4-range2-cc", "frag5-anchor",
    ]
    assert graph.representatives == {
        3: "frag3-range2", 4: "frag4-range2-cc", 5: "frag5-anchor",
    }


def test_common_conformer_other_side(tmp_path):
    _common_conformer_project(tmp_path)
    graph = PoolGraph(Project(tmp_path), targets=["frag4-range3-cc"])
    assert sorted(graph.pools) == ["frag3-anchor", "frag4-range3", "frag4-range3-cc"]
    assert graph.representatives == {3: "frag3-anchor", 4: "frag4-range3-cc"}
