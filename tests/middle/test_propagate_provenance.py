from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from alaric.middle.propagate_provenance import (  # noqa: E402
    PropagationError,
    _lineage,
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

    output = propagate([tmp_path / "representative", tmp_path / "grow"])

    assert output.name == "prop-provenance.npy.zst"
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
    np.save(tmp_path / "merge" / "map-1.npy", np.array([[0, 0]], dtype=np.uint64))
    np.save(tmp_path / "merge" / "map-2.npy", np.array([[9, 0]], dtype=np.uint64))
    np.save(tmp_path / "parent-sigil" / "provenance.npy", np.array([0], dtype=np.uint32))
    save_npy(tmp_path / "grow" / "provenance.npy", np.array([42], dtype=np.uint32))

    output = propagate(
        [tmp_path / "representative", tmp_path / "merge", tmp_path / "parent-sigil", tmp_path / "grow"]
    )

    assert load_npy(output).tolist() == [42]


def test_graph_lineage_requires_a_merge_branch() -> None:
    data = {
        "pools": {
            "rep": {"kind": "filter", "parents": [{"pool": "merge", "array": "provenance.npy", "cross_fragment": False}]},
            "merge": {
                "kind": "merge",
                "parents": [
                    {"pool": "left", "edge": "merge", "array": "map-1.npy"},
                    {"pool": "right", "edge": "merge", "array": "map-2.npy"},
                ],
            },
            "left": {"kind": "grow", "parents": [{"pool": "source", "edge": "grow", "array": "provenance.npy"}]},
        }
    }
    with pytest.raises(PropagationError, match="--branch"):
        _lineage(data, "rep", [])
    assert _lineage(data, "rep", ["left"]) == ["rep", "merge", "left"]
