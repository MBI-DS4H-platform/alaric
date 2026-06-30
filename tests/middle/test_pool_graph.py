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
        "action": "anchor", "fragment": fragment, "sequence": "GU", "exclude": "1abc",
        "protein": "dom", "resid": 1, "nucleotide": nucleotide,
        "dihedral": "45 -45", "angle": 30,
    }


def _grow(input_name: str, fragment: int) -> dict:
    return {
        "action": "grow", "input": input_name, "fragment": fragment, "sequence": "UU",
        "exclude": "1abc", "direction": "forward", "crmsd": 0.25, "ovrmsd": 0.75,
    }


def _score(input_name: str) -> dict:
    return {"action": "score", "input": input_name, "sequence": "UU",
            "exclude": "1abc", "protein": "dom"}


def _filter(input_name: str, score_name: str, threshold: float = -1.0) -> dict:
    return {"action": "filter", "input": input_name, "score_input": score_name,
            "threshold": threshold}


def _data(root: Path) -> None:
    data = root / "DATA"
    data.mkdir()
    (data / "dom.pdb").write_text("HEADER plain\n")
    (data / "dom-aa.pdb").write_text("HEADER aa\n")


def _full_project(root: Path) -> None:
    """Mirror the 1b7f-RUN topology: grow chain with a merge, a dedup, and a
    disconnected branch, all hermetic (explicit params, no shared fixture)."""
    _data(root)
    # frag4: two anchors -> merge -> filter (the grow source for frag5)
    _mk(root, "frag4-anchor-fwd", _anchor(4, "first"))
    _mk(root, "frag4-anchor-bwd", _anchor(4, "second"))
    _mk(root, "frag4-merged", {"action": "identity", "input1": "frag4-anchor-fwd", "input2": "frag4-anchor-bwd"})
    _mk(root, "frag4-merged-score", _score("frag4-merged"))
    _mk(root, "frag4-merged-filter", _filter("frag4-merged", "frag4-merged-score"))
    # frag5: grow -> filter, plus a frag5 anchor, merged together
    _mk(root, "frag5-fwd", _grow("frag4-merged-filter", 5))
    _mk(root, "frag5-fwd-score", _score("frag5-fwd"))
    _mk(root, "frag5-fwd-filter", _filter("frag5-fwd", "frag5-fwd-score"))
    _mk(root, "frag5-anchor", _anchor(5, "second"))
    _mk(root, "frag5-merged", {"action": "identity", "input1": "frag5-fwd-filter", "input2": "frag5-anchor"})
    # frag6: grow -> filter
    _mk(root, "frag6-fwd", _grow("frag5-merged", 6))
    _mk(root, "frag6-fwd-score", _score("frag6-fwd"))
    _mk(root, "frag6-fwd-filter", _filter("frag6-fwd", "frag6-fwd-score"))
    # frag7: grow -> filter -> dedup (identity self-merge)
    _mk(root, "frag7-fwd", _grow("frag6-fwd-filter", 7))
    _mk(root, "frag7-fwd-score", _score("frag7-fwd"))
    _mk(root, "frag7-fwd-filter", _filter("frag7-fwd", "frag7-fwd-score"))
    _mk(root, "frag7-fwd-filter-unique", {"action": "identity", "input1": "frag7-fwd-filter", "input2": "frag7-fwd-filter"})
    # frag8: grow endpoint (no filter)
    _mk(root, "frag8-fwd", _grow("frag7-fwd-filter-unique", 8))
    # frag10: anchor + filter, not wired into any grow -> disconnected
    _mk(root, "frag10-anchor", _anchor(10, "second"))
    _mk(root, "frag10-anchor-score", _score("frag10-anchor"))
    _mk(root, "frag10-anchor-filter", _filter("frag10-anchor", "frag10-anchor-score"))


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
    _mk(root, "frag5-score", _score("frag5-fwd"))
    _mk(root, "frag5-filter-a", _filter("frag5-fwd", "frag5-score", -1.0))
    _mk(root, "frag5-filter-b", _filter("frag5-fwd", "frag5-score", -2.0))


def test_ambiguous_fragment_raises(tmp_path):
    _ambiguous_project(tmp_path)
    with pytest.raises(PoolGraphError, match="multiple chain lineages"):
        PoolGraph(Project(tmp_path))
