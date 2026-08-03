from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from alaric.middle import propagate_provenance as module  # noqa: E402
from alaric.middle.propagate_provenance import (  # noqa: E402
    PropagationError,
    _lineages,
    graph_mode,
    propagate,
)
from alaric.npy_io import load_npy, save_npy  # noqa: E402
from alaric.poses import pack_pool, write_arc_file  # noqa: E402


def _poses(path: Path, nposes: int) -> None:
    path.mkdir(parents=True)
    conf = np.zeros(nposes, dtype=np.uint16)
    rot = np.zeros(nposes, dtype=np.uint16)
    trans = np.zeros((nposes, 3), dtype=np.int32)
    M, O, C, P = pack_pool(conf, rot, trans, bucket_size=16, sort_offsets=False)[0]
    write_arc_file(path / "poses-1.arc", M, O, C, P, bucket_size=16)


def test_propagates_filter_provenance_to_compressed_output(tmp_path: Path) -> None:
    _poses(tmp_path / "representative", 2)
    (tmp_path / "grow").mkdir()
    np.save(tmp_path / "representative" / "provenance.npy", np.array([0, 3], dtype=np.uint32))
    np.save(tmp_path / "grow" / "provenance.npy", np.array([8, 7, 6, 5], dtype=np.uint32))

    output = propagate([tmp_path / "representative", tmp_path / "grow"], 3)

    assert output.name == "prop-frag3-provenance.npy.zst"
    assert load_npy(output).tolist() == [8, 5]


def test_merge_map_is_selected_by_next_directory_sigil(tmp_path: Path) -> None:
    _poses(tmp_path / "representative", 1)
    for name in ("merge", "parent-sigil", "other-sigil", "grow"):
        (tmp_path / name).mkdir()
    np.save(tmp_path / "representative" / "provenance.npy", np.array([0], dtype=np.uint32))
    (tmp_path / "merge" / "identity-filter.json").write_text(
        json.dumps(
            {
                "pose_dir_1": "../CACHE/results/parent-sigil",
                "pose_dir_2": "../CACHE/results/other-sigil",
            }
        )
    )
    # A merge is relational even where a particular map happens to be one-to-one.
    np.save(tmp_path / "merge" / "map-1.npy", np.array([[0, 0], [1, 0]], dtype=np.uint64))
    np.save(tmp_path / "merge" / "map-2.npy", np.array([[9, 0]], dtype=np.uint64))
    np.save(tmp_path / "parent-sigil" / "provenance.npy", np.array([0, 1], dtype=np.uint32))
    save_npy(tmp_path / "grow" / "provenance.npy", np.array([42, 99], dtype=np.uint32))

    output = propagate(
        [tmp_path / "representative", tmp_path / "merge", tmp_path / "parent-sigil", tmp_path / "grow"],
        6,
    )

    assert output.name == "prop-frag6-map.npy.zst"
    assert load_npy(output).tolist() == [[42, 0], [99, 0]]
    assert not (tmp_path / "representative" / "prop-frag6-provenance.npy.zst").exists()


def test_writing_one_route_keeps_another_fragments_file(tmp_path: Path) -> None:
    _poses(tmp_path / "representative", 2)
    for name in ("grow-a", "grow-b"):
        (tmp_path / name).mkdir()
    np.save(tmp_path / "representative" / "provenance.npy", np.array([0, 1], dtype=np.uint32))
    np.save(tmp_path / "grow-a" / "provenance.npy", np.array([4, 5], dtype=np.uint32))
    np.save(tmp_path / "grow-b" / "provenance.npy", np.array([7, 8], dtype=np.uint32))
    # A stale file from the pre-fragment-keyed naming must not survive a rewrite.
    save_npy(tmp_path / "representative" / "prop-provenance.npy", np.array([0], dtype=np.uint32))

    first = propagate([tmp_path / "representative", tmp_path / "grow-a"], 8)
    second = propagate([tmp_path / "representative", tmp_path / "grow-b"], 10)

    assert first.is_file() and second.is_file()
    assert load_npy(first).tolist() == [4, 5]
    assert load_npy(second).tolist() == [7, 8]
    assert not (tmp_path / "representative" / "prop-provenance.npy").exists()


def _reconnect_graph() -> dict:
    """A merge whose two arms carry provenance from different fragments."""
    return {
        "pools": {
            "rep": {
                "kind": "merge",
                "fragment": 9,
                "parents": [
                    {"pool": "fwd", "edge": "merge", "array": "map-1.npy"},
                    {"pool": "bwd", "edge": "merge", "array": "map-2.npy"},
                ],
            },
            "fwd": {"kind": "grow", "fragment": 9, "parents": [{"pool": "frag8", "edge": "grow", "array": "provenance.npy"}]},
            "bwd": {"kind": "grow", "fragment": 9, "parents": [{"pool": "frag10", "edge": "grow", "array": "provenance.npy"}]},
            "frag8": {"kind": "filter", "fragment": 8, "parents": []},
            "frag10": {"kind": "filter", "fragment": 10, "parents": []},
        }
    }


def test_a_reconnect_yields_one_lineage_per_source_fragment() -> None:
    assert _lineages(_reconnect_graph(), "rep", []) == [
        (["rep", "fwd"], 8),
        (["rep", "bwd"], 10),
    ]


def test_branch_restricts_a_reconnect_to_one_arm() -> None:
    assert _lineages(_reconnect_graph(), "rep", ["bwd"]) == [(["rep", "bwd"], 10)]
    with pytest.raises(PropagationError, match="--branch does not select"):
        _lineages(_reconnect_graph(), "rep", ["absent"])


def test_lineages_clashing_on_one_fragment_require_a_branch() -> None:
    data = _reconnect_graph()
    data["pools"]["bwd"]["parents"] = [{"pool": "frag8", "edge": "grow", "array": "provenance.npy"}]
    with pytest.raises(PropagationError, match="carry fragment 8 provenance"):
        _lineages(data, "rep", [])
    assert _lineages(data, "rep", ["fwd"]) == [(["rep", "fwd"], 8)]


def _reconnect_project(tmp_path: Path) -> Path:
    """Lay out the reconnect graph's result dirs under a project's CACHE."""
    results = tmp_path / "CACHE" / "results"
    _poses(results / "rep-sigil", 2)
    for name in ("fwd-sigil", "bwd-sigil"):
        (results / name).mkdir(parents=True)
    # rep merges the two arms; each arm's map is chosen by the next dir's sigil.
    (results / "rep-sigil" / "identity-filter.json").write_text(
        json.dumps({"pose_dir_1": "../CACHE/results/fwd-sigil", "pose_dir_2": "../CACHE/results/bwd-sigil"})
    )
    # An identity filter keeps what both inputs contain, so both maps cover every
    # output pose -- each arm propagates a complete route.
    np.save(results / "rep-sigil" / "map-1.npy", np.array([[0, 0], [1, 1]], dtype=np.uint64))
    np.save(results / "rep-sigil" / "map-2.npy", np.array([[3, 0], [2, 1]], dtype=np.uint64))
    np.save(results / "fwd-sigil" / "provenance.npy", np.array([40, 41], dtype=np.uint32))
    np.save(results / "bwd-sigil" / "provenance.npy", np.array([50, 51, 52, 53], dtype=np.uint32))

    data = _reconnect_graph()
    for pool, sigil in (("rep", "rep-sigil"), ("fwd", "fwd-sigil"), ("bwd", "bwd-sigil")):
        data["pools"][pool]["sigil"] = sigil
    data["representatives"] = {"9": "rep"}
    graph = tmp_path / "graph.json"
    graph.write_text(json.dumps(data))
    (tmp_path / "project").mkdir()  # ../CACHE/results is resolved from here
    return graph


def test_local_graph_mode_propagates_every_route(tmp_path: Path, monkeypatch) -> None:
    graph = _reconnect_project(tmp_path)
    monkeypatch.chdir(tmp_path / "project")

    written = graph_mode(graph, "rep", "local", [])

    assert [path.name for path in written] == ["prop-frag8-map.npy.zst", "prop-frag10-map.npy.zst"]
    assert load_npy(written[0]).tolist() == [[40, 0], [41, 1]]
    assert load_npy(written[1]).tolist() == [[52, 1], [53, 0]]


def test_remote_graph_mode_writes_one_command_per_route(tmp_path: Path, monkeypatch) -> None:
    graph = _reconnect_project(tmp_path)
    monkeypatch.chdir(tmp_path / "project")
    monkeypatch.setenv("ALARIC_REMOTE_HOST", "remote")
    monkeypatch.setenv("ALARIC_REMOTE_RESULT_DIR", "/remote/results")
    monkeypatch.setenv("ALARIC_REMOTE_DEPLOYMENT_DIR", "/remote/deploy")
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: None)

    (script,) = graph_mode(graph, "rep", "remote", [])

    commands = [line for line in script.read_text().splitlines() if line.startswith("alaric-")]
    assert commands == [
        "alaric-propagate-provenance --source-fragment 8 --pool-dirs "
        "/remote/results/rep-sigil /remote/results/fwd-sigil",
        "alaric-propagate-provenance --source-fragment 10 --pool-dirs "
        "/remote/results/rep-sigil /remote/results/bwd-sigil",
    ]


def test_a_representative_without_grow_lineage_writes_nothing(tmp_path: Path, capsys) -> None:
    data = _reconnect_graph()
    for pool in ("fwd", "bwd"):
        data["pools"][pool] = {"kind": "anchor", "fragment": 9, "parents": []}
    data["representatives"] = {"9": "rep"}
    graph = tmp_path / "graph.json"
    graph.write_text(json.dumps(data))

    assert graph_mode(graph, "rep", "local", []) == []
    assert "nothing to propagate" in capsys.readouterr().out


def test_an_arm_without_grow_provenance_is_dropped() -> None:
    data = _reconnect_graph()
    # An anchored pool sources its own poses; that arm carries nothing to propagate.
    data["pools"]["bwd"] = {"kind": "anchor", "fragment": 9, "parents": []}
    assert _lineages(data, "rep", []) == [(["rep", "fwd"], 8)]
    # ... and a representative whose every arm is anchored propagates nothing at all.
    data["pools"]["fwd"] = {"kind": "anchor", "fragment": 9, "parents": []}
    assert _lineages(data, "rep", []) == []
