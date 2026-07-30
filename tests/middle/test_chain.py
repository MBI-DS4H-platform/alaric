from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "alaric"))

from poses import pack_pool, write_arc_file  # noqa: E402

import sys as _sys
_sys.path.insert(0, str(ROOT / "alaric"))
from poses import PoseReader, select_pose_indices  # noqa: E402

from alaric.middle.chain import (  # noqa: E402
    CHAINS_FILE,
    CHAINS_METADATA_FILE,
    ChainError,
    ChainGraph,
    _identities,
    _write_unique_pose_dir,
    build_chains,
    count_chains,
    resolve_selection,
    write_chain_outputs,
)


# -- fixture helpers ------------------------------------------------------


def _write_pose_dir(path: Path, poses: list[tuple[int, int, tuple[int, int, int]]]) -> None:
    """Write one organized poses-1.arc; global pose index == list order."""
    path.mkdir(parents=True, exist_ok=True)
    confs = np.array([p[0] for p in poses], dtype=np.uint16)
    rots = np.array([p[1] for p in poses], dtype=np.uint16)
    trans = np.array([p[2] for p in poses], dtype=np.int32)
    packed = pack_pool(confs, rots, trans, bucket_size=16, sort_offsets=False)
    assert len(packed) == 1, "test poses must fit one bucket"
    M, O, C, P = packed[0]
    write_arc_file(path / "poses-1.arc", M, O, C, P, bucket_size=16)


def _prov(path: Path, values: list[int]) -> None:
    np.save(path / "provenance.npy", np.array(values, dtype=np.uint32))


def _map(path: Path, name: str, pairs: list[tuple[int, int]]) -> None:
    np.save(path / name, np.array(pairs, dtype=np.uint64).reshape(-1, 2))


def _dense_step(pool: str, to: str) -> dict:
    return {"pool": pool, "array": "provenance.npy", "orientation": "row=this,value=parent", "to": to}


def _map_step(pool: str, array: str, to: str) -> dict:
    return {"pool": pool, "array": array, "orientation": "cols=(parent,this)", "to": to}


def _write_json(root: Path, reps: dict[int, str], pools: dict[str, int], links: list[dict]) -> Path:
    data = {
        "project": str(root),
        "representatives": {str(k): v for k, v in reps.items()},
        "skipped_fragments": {},
        "pools": {name: {"fragment": frag, "result_dir": name} for name, frag in pools.items()},
        "links": links,
    }
    path = root / "graph.json"
    path.write_text(json.dumps(data))
    return path


def _graph(root: Path) -> ChainGraph:
    data = json.loads((root / "graph.json").read_text())
    return ChainGraph(data, root)


# -- direct grow chain: prune dead ends, unique < kept --------------------


def _basic_project(root: Path) -> None:
    # frag1: A0,A1,A2 ; frag2: B0,B1,B2 (grown, prov->A) ; frag3: C0,C1,C2 (C1==C2)
    _write_pose_dir(root / "r1", [(0, 0, (0, 0, 0)), (0, 0, (1, 0, 0)), (0, 0, (2, 0, 0))])
    _write_pose_dir(root / "r2", [(0, 0, (0, 1, 0)), (0, 0, (1, 1, 0)), (0, 0, (2, 1, 0))])
    _write_pose_dir(root / "r3", [(0, 0, (0, 2, 0)), (0, 0, (1, 2, 0)), (0, 0, (1, 2, 0))])
    _prov(root / "r2", [0, 0, 2])  # B0<-A0, B1<-A0, B2<-A2  (A1 has no child)
    _prov(root / "r3", [1, 2, 2])  # C0<-B1, C1<-B2, C2<-B2  (B0 has no child)
    _write_json(
        root,
        {1: "r1", 2: "r2", 3: "r3"},
        {"r1": 1, "r2": 2, "r3": 3},
        [
            {"source_fragment": 1, "source_rep": "r1", "target_fragment": 2, "target_rep": "r2",
             "join_pool": "r1", "target_to_join": [_dense_step("r2", "r1")], "source_to_join": []},
            {"source_fragment": 2, "source_rep": "r2", "target_fragment": 3, "target_rep": "r3",
             "join_pool": "r2", "target_to_join": [_dense_step("r3", "r2")], "source_to_join": []},
        ],
    )


def test_basic_count(tmp_path):
    _basic_project(tmp_path)
    g = _graph(tmp_path)
    res = count_chains(g, resolve_selection(g, None, None))
    assert res.order == ["r1", "r2", "r3"]
    assert res.kept == [2, 2, 2]      # A1 and B0 are dead ends, pruned; C1/C2 dedup
    assert res.unique == [2, 2, 2]
    assert res.total_chains == 2      # A0-B1-C0, A2-B2-C1/C2


def _read_ids(pose_dir: Path) -> list[tuple[int, int, int, int, int]]:
    n = PoseReader.get_nposes(pose_dir)
    ch = select_pose_indices(pose_dir, np.arange(n))
    return [
        (int(c), int(r), int(t[0]), int(t[1]), int(t[2]))
        for c, r, t in zip(ch.conformers, ch.rotamers, ch.translations_grid)
    ]


def test_build_basic(tmp_path):
    _basic_project(tmp_path)
    g = _graph(tmp_path)
    sel = resolve_selection(g, None, None)
    cnt = count_chains(g, sel)
    out = tmp_path / "out"
    layers, table = build_chains(g, sel, out)

    assert layers.order == ["r1", "r2", "r3"]
    # one pose dir per fragment, each holding exactly the unique kept poses
    for pool, uniq in zip(layers.order, cnt.unique):
        assert PoseReader.get_nposes(out / pool) == uniq

    # the table is every chain as 1-based indices into those pose dirs
    assert table.shape == (cnt.total_chains, 3)
    assert int(table.min()) >= 1
    for j, pool in enumerate(layers.order):
        assert int(table[:, j].max()) <= PoseReader.get_nposes(out / pool)
    assert sorted(table.tolist()) == sorted([[1, 1, 1], [2, 2, 2]])

    # indices resolve to the right physical poses
    assert _read_ids(out / "r1") == [(0, 0, 0, 0, 0), (0, 0, 2, 0, 0)]  # A0, A2
    assert _read_ids(out / "r2") == [(0, 0, 1, 1, 0), (0, 0, 2, 1, 0)]  # B1, B2
    assert _read_ids(out / "r3") == [(0, 0, 0, 2, 0), (0, 0, 1, 2, 0)]  # C0, C1(=C2)


def test_build_writes_chains_file_and_metadata(tmp_path):
    _basic_project(tmp_path)
    g = _graph(tmp_path)
    sel = resolve_selection(g, None, None)
    out = tmp_path / "out"
    layers, table = build_chains(g, sel, out)
    metadata = write_chain_outputs(
        g, layers, table, out, graph_json=tmp_path / "graph.json"
    )

    # the index table is the pool header plus one row per chain
    lines = (out / CHAINS_FILE).read_text().splitlines()
    assert lines[0].split("\t") == layers.order
    written = np.loadtxt(out / CHAINS_FILE, dtype=np.int64, skiprows=1, ndmin=2)
    assert written.tolist() == table.tolist()

    assert json.loads((out / CHAINS_METADATA_FILE).read_text()) == metadata
    assert metadata["nchains"] == len(table)
    assert metadata["chains_file"] == CHAINS_FILE
    assert metadata["graph"] == str(tmp_path / "graph.json")
    assert [c["pool"] for c in metadata["columns"]] == layers.order
    assert [c["fragment"] for c in metadata["columns"]] == [1, 2, 3]
    assert [c["nposes"] for c in metadata["columns"]] == [
        PoseReader.get_nposes(out / pool) for pool in layers.order
    ]
    # the fixture project has no DATA/, so these stay unresolved rather than failing
    assert [c["sequence"] for c in metadata["columns"]] == [None, None, None]
    assert metadata["exclude"] is None


def test_build_matches_count(tmp_path):
    # multi-step composition (through the filter intermediate) round-trips
    _filter_step_project(tmp_path)
    g = _graph(tmp_path)
    sel = resolve_selection(g, None, None)
    cnt = count_chains(g, sel)
    layers, table = build_chains(g, sel, tmp_path / "out")
    assert len(table) == cnt.total_chains
    for pool, uniq in zip(layers.order, cnt.unique):
        assert PoseReader.get_nposes(tmp_path / "out" / pool) == uniq


def test_written_pose_indices_follow_numeric_identity_order(tmp_path):
    """Packing reorders buckets; returned 1-based indices must still be exact."""
    source = tmp_path / "source"
    source.mkdir()
    poses = [
        (1, 0, (32, 0, 0)),
        (2, 0, (0, 0, 0)),
        (256, 0, (0, 0, 0)),
        (3, 0, (48, 0, 0)),
    ]
    packed = pack_pool(
        np.array([p[0] for p in poses], dtype=np.uint16),
        np.array([p[1] for p in poses], dtype=np.uint16),
        np.array([p[2] for p in poses], dtype=np.int32),
        bucket_size=16,
        sort_offsets=False,
    )
    for number, (M, O, C, P) in enumerate(packed, start=1):
        write_arc_file(source / f"poses-{number}.arc", M, O, C, P, bucket_size=16)

    class Graph:
        def _result_dir(self, _pool):
            return source

    source_ids = np.arange(PoseReader.get_nposes(source), dtype=np.int64)
    one_based = _write_unique_pose_dir(Graph(), "pool", source_ids, tmp_path / "out")
    source_rows = _identities(select_pose_indices(source, source_ids))
    written_rows = _identities(
        select_pose_indices(tmp_path / "out", one_based - 1)
    )
    assert np.array_equal(written_rows, source_rows)


def test_select_subset_two_layers(tmp_path):
    _basic_project(tmp_path)
    g = _graph(tmp_path)
    res = count_chains(g, resolve_selection(g, ["r1", "r2"], None))
    assert res.order == ["r1", "r2"]
    # over only frag1-2: A0->{B0,B1}, A2->B2 ; A1 dead. all of A0,A2,B0,B1,B2 on a chain
    assert res.kept == [2, 3]
    assert res.total_chains == 3


def test_exclude_endpoint(tmp_path):
    _basic_project(tmp_path)
    g = _graph(tmp_path)
    res = count_chains(g, resolve_selection(g, None, ["r3"]))
    assert res.order == ["r1", "r2"]
    assert res.total_chains == 3


def test_exclude_middle_breaks_chain(tmp_path):
    _basic_project(tmp_path)
    g = _graph(tmp_path)
    with pytest.raises(ChainError, match="not adjacent in the chain"):
        count_chains(g, resolve_selection(g, None, ["r2"]))


def test_selection_validation_lists_representatives(tmp_path):
    _basic_project(tmp_path)
    g = _graph(tmp_path)
    with pytest.raises(ChainError, match="r1, r2, r3"):
        resolve_selection(g, ["nope"], None)


# -- multi-step composition through a filter ------------------------------


def _filter_step_project(root: Path) -> None:
    # r2 (filter) <- g2 (grow) <- r1 ; g2 is a non-representative intermediate
    _write_pose_dir(root / "r1", [(0, 0, (0, 0, 0)), (0, 0, (1, 0, 0)), (0, 0, (2, 0, 0))])
    _write_pose_dir(root / "g2", [(0, 0, (0, 1, 0)), (0, 0, (1, 1, 0)),
                                  (0, 0, (2, 1, 0)), (0, 0, (3, 1, 0))])
    _write_pose_dir(root / "r2", [(0, 0, (0, 1, 0)), (0, 0, (3, 1, 0))])  # filter keeps G0,G3
    _prov(root / "g2", [0, 1, 1, 2])   # G0<-A0, G1<-A1, G2<-A1, G3<-A2
    _prov(root / "r2", [0, 3])         # r2_0<-G0, r2_1<-G3
    _write_json(
        root,
        {1: "r1", 2: "r2"},
        {"r1": 1, "g2": 2, "r2": 2},
        [{"source_fragment": 1, "source_rep": "r1", "target_fragment": 2, "target_rep": "r2",
          "join_pool": "r1",
          "target_to_join": [_dense_step("r2", "g2"), _dense_step("g2", "r1")],
          "source_to_join": []}],
    )


def test_filter_step_composition(tmp_path):
    _filter_step_project(tmp_path)
    g = _graph(tmp_path)
    res = count_chains(g, resolve_selection(g, None, None))
    # r2_0 -> G0 -> A0 ; r2_1 -> G3 -> A2 ; A1 has no surviving child
    assert res.kept == [2, 2]          # A0,A2 kept (A1 pruned)
    assert res.total_chains == 2


def test_intermediate_poses_obsoleted_but_provenance_kept(tmp_path):
    # The crux: g2's *poses* are gone but its provenance array survives.
    # Chain building must still work -- intermediates need provenance, not poses.
    _filter_step_project(tmp_path)
    (tmp_path / "g2" / "poses-1.arc").unlink()
    assert list((tmp_path / "g2").glob("provenance.npy"))  # provenance retained
    g = _graph(tmp_path)
    assert not g.is_materialized("g2")
    res = count_chains(g, resolve_selection(g, None, None))
    assert res.kept == [2, 2]
    assert res.total_chains == 2


def test_intermediate_fully_gone_gives_up(tmp_path):
    _filter_step_project(tmp_path)
    shutil.rmtree(tmp_path / "g2")  # neither materialized nor any provenance
    g = _graph(tmp_path)
    with pytest.raises(ChainError, match="neither materialized nor has filter provenance"):
        count_chains(g, resolve_selection(g, None, None))


def test_unmaterialized_representative_errors(tmp_path):
    _filter_step_project(tmp_path)
    (tmp_path / "r2" / "poses-1.arc").unlink()  # representative has no poses
    g = _graph(tmp_path)
    with pytest.raises(ChainError, match="representative 'r2'.*not materialized"):
        count_chains(g, resolve_selection(g, None, None))


# -- merge (map array) routing through input1 -----------------------------


def test_merge_step_composition(tmp_path):
    root = tmp_path
    _write_pose_dir(root / "rs", [(0, 0, (0, 0, 0)), (0, 0, (1, 0, 0)), (0, 0, (2, 0, 0))])
    _write_pose_dir(root / "gfwd", [(0, 0, (0, 1, 0)), (0, 0, (1, 1, 0)), (0, 0, (2, 1, 0))])
    _write_pose_dir(root / "rt", [(0, 0, (1, 1, 0)), (0, 0, (2, 1, 0))])  # merged poses M0,M1
    _prov(root / "gfwd", [0, 1, 2])    # G0<-S0, G1<-S1, G2<-S2
    # rt = identity-merge(gfwd, anc): M0<-gfwd1, M1<-gfwd2 (only input1 route matters)
    _map(root / "rt", "map-1.npy", [(1, 0), (2, 1)])   # (parent=gfwd, this=rt)
    _map(root / "rt", "map-2.npy", [(0, 0), (1, 1)])   # the anchor route (unused here)
    _write_json(
        root,
        {1: "rs", 2: "rt"},
        {"rs": 1, "gfwd": 2, "rt": 2},
        [{"source_fragment": 1, "source_rep": "rs", "target_fragment": 2, "target_rep": "rt",
          "join_pool": "rs",
          "target_to_join": [_map_step("rt", "map-1.npy", "gfwd"),
                             _dense_step("gfwd", "rs")],
          "source_to_join": []}],
    )
    g = _graph(root)
    res = count_chains(g, resolve_selection(g, None, None))
    # M0->gfwd1->S1 ; M1->gfwd2->S2 ; S0 has no child
    assert res.kept == [2, 2]
    assert res.total_chains == 2


# -- dedup one-to-many collapses duplicate edges --------------------------


def test_dedup_collapses_parallel_edges_before_counting_chains(tmp_path):
    root = tmp_path
    _write_pose_dir(root / "rs", [(0, 0, (0, 0, 0)), (0, 0, (1, 0, 0)), (0, 0, (2, 0, 0))])
    # gfwd: G0,G1 identical (both grown from S0), G2<-S1, G3<-S2
    _write_pose_dir(root / "gfwd", [(0, 0, (5, 5, 5)), (0, 0, (5, 5, 5)),
                                    (0, 0, (1, 1, 0)), (0, 0, (2, 1, 0))])
    _write_pose_dir(root / "rt", [(0, 0, (5, 5, 5)), (0, 0, (1, 1, 0)), (0, 0, (2, 1, 0))])
    _prov(root / "gfwd", [0, 0, 1, 2])  # G0,G1<-S0 ; G2<-S1 ; G3<-S2
    # dedup: M0<-{G0,G1}, M1<-G2, M2<-G3  (map columns parent=gfwd, this=rt)
    _map(root / "rt", "map-1.npy", [(0, 0), (1, 0), (2, 1), (3, 2)])
    _map(root / "rt", "map-2.npy", [(0, 0), (1, 0), (2, 1), (3, 2)])
    _write_json(
        root,
        {1: "rs", 2: "rt"},
        {"rs": 1, "gfwd": 2, "rt": 2},
        [{"source_fragment": 1, "source_rep": "rs", "target_fragment": 2, "target_rep": "rt",
          "join_pool": "rs",
          "target_to_join": [_map_step("rt", "map-1.npy", "gfwd"),
                             _dense_step("gfwd", "rs")],
          "source_to_join": []}],
    )
    g = _graph(root)
    res = count_chains(g, resolve_selection(g, None, None))
    # Deduplicating the pose pool before path enumeration also merges the two
    # identical (S0, M0) edges, so there are three physical chains.
    assert res.kept == [3, 3]
    assert res.total_chains == 3
