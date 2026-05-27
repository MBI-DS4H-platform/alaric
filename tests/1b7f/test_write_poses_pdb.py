from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "code"))

from parse_pdb import atomic_dtype  # noqa: E402
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
        self.template = _template()
        self._rotamers = np.eye(3, dtype=np.float32)[None, :, :]

    def get_rotamers(self, conformer: int) -> np.ndarray:
        assert conformer == 0
        return self._rotamers


def _load_write_poses_pdb_module():
    path = REPO / "code" / "write-poses-pdb.py"
    spec = importlib.util.spec_from_file_location("write_poses_pdb_script", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _template() -> np.ndarray:
    template = np.zeros((2,), dtype=atomic_dtype)
    template["hetero"] = b""
    template["name"] = [b"P", b"C1'"]
    template["altloc"] = b" "
    template["resname"] = b"G"
    template["chain"] = b"A"
    template["index"] = [10, 11]
    template["icode"] = b" "
    template["resid"] = [1, 1]
    template["occupancy"] = 1.0
    template["segid"] = b""
    template["element"] = [b"P", b"C"]
    return template


def _write_test_poses(pose_dir: Path) -> None:
    packed = pack_pool(
        np.array([0, 0], dtype=np.uint16),
        np.array([0, 0], dtype=np.uint16),
        np.array([[0, 0, 0], [3, 0, 0]], dtype=np.int16),
        bucket_size=16,
    )
    assert len(packed) == 1
    M, O, C, P = packed[0]
    write_arc_file(pose_dir / "poses-1.arc", M, O, C, P, bucket_size=16)


def test_materialize_pose_pdb_fills_template_from_pose_coordinates(tmp_path: Path) -> None:
    write_poses_pdb = _load_write_poses_pdb_module()
    pose_dir = tmp_path / "poses"
    _write_test_poses(pose_dir)

    atoms = write_poses_pdb.materialize_pose_pdb(
        PoseReader(pose_dir, rows_per_chunk=1),
        library=FakeLibrary(),
    )

    assert atoms.shape == (2, 2)
    assert np.array_equal(atoms["model"][:, 0], np.array([1, 2], dtype=np.uint16))
    assert np.array_equal(atoms["index"][0], np.array([1, 2], dtype=np.uint32))
    assert np.allclose(atoms["x"][0], np.array([0.0, 1.0], dtype=np.float32))
    assert np.allclose(
        atoms["x"][1],
        np.array([3 * GRID_SPACING, 1 + 3 * GRID_SPACING], dtype=np.float32),
    )


def test_write_pdb_text_with_library_uses_shared_multi_model_writer(
    tmp_path: Path,
) -> None:
    write_poses_pdb = _load_write_poses_pdb_module()
    pose_dir = tmp_path / "poses"
    _write_test_poses(pose_dir)

    pdb_text = write_poses_pdb.write_pdb_text_with_library(
        PoseReader(pose_dir),
        library=FakeLibrary(),
    )

    assert pdb_text.count("MODEL") == 2
    assert pdb_text.count("ENDMDL") == 2
    assert "MODEL 1\n" in pdb_text
    assert "MODEL 2\n" in pdb_text
