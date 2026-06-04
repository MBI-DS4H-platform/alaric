from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / ".alaric"))
sys.path.insert(0, str(HERE / ".util"))

from poses import PoseReader, pack_pool, write_arc_file  # noqa: E402
from rmsd import GRID_SPACING  # noqa: E402


class FakeLibrary:
    def __init__(self) -> None:
        self.coordinates = np.array(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            ],
            dtype=np.float32,
        )
        self.template = np.empty((2,), dtype=np.float32)
        self._rotamers = np.eye(3, dtype=np.float32)[None, :, :]

    def get_rotamers(self, conformer: int) -> np.ndarray:
        assert conformer == 0
        return self._rotamers


def _load_pairwise_module():
    path = HERE / ".util" / "pairwise-rmsd.py"
    spec = importlib.util.spec_from_file_location("pairwise_rmsd_script", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pairwise_rmsd_stores_upper_triangle_order(tmp_path: Path) -> None:
    pairwise = _load_pairwise_module()
    library = FakeLibrary()
    conformers = np.array([0, 0, 0], dtype=np.uint16)
    rotamers = np.array([0, 0, 0], dtype=np.uint16)
    translations = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 2, 0],
        ],
        dtype=np.int16,
    )
    packed = pack_pool(
        conformers,
        rotamers,
        translations,
        bucket_size=16,
    )
    assert len(packed) == 1
    M, O, C, P = packed[0]
    pose_dir = tmp_path / "poses"
    write_arc_file(pose_dir / "poses-1.arc", M, O, C, P, bucket_size=16)

    actual = pairwise.compute_pairwise_rmsd_with_library(
        PoseReader(pose_dir),
        library=library,
        pair_chunk_size=2,
        nprocs=1,
    )
    expected = GRID_SPACING * np.array([1.0, 2.0, np.sqrt(5.0)], dtype=np.float32)
    assert np.allclose(actual, expected, atol=1e-3, rtol=0)


def test_pairwise_npy_output_accepts_unordered_indexed_chunks(tmp_path: Path) -> None:
    pairwise = _load_pairwise_module()
    output = tmp_path / "pairwise-rmsd.npy"
    chunks = iter(
        [
            (2, np.array([3.0], dtype=np.float32)),
            (0, np.array([1.0, 2.0], dtype=np.float32)),
        ]
    )
    pairwise.write_npy_output_chunks(output, chunks, total_pairs=3)
    assert np.array_equal(
        np.load(output),
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
    )


def test_order_array_remaps_pairwise_upper_triangle(tmp_path: Path) -> None:
    pairwise = _load_pairwise_module()
    output = tmp_path / "ordered-pairwise-rmsd.npy"
    order_array = np.array([2, 0, 1], dtype=np.int64)
    chunks = iter(
        [
            (0, np.array([1.0, 2.0], dtype=np.float32)),
            (2, np.array([3.0], dtype=np.float32)),
        ]
    )
    pairwise.write_ordered_npy_output_chunks(
        output,
        chunks,
        total_pairs=3,
        nposes=3,
        order_array=order_array,
    )
    assert np.array_equal(
        np.load(output),
        np.array([3.0, 1.0, 2.0], dtype=np.float32),
    )
