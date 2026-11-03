"""The filter action's two routes must write compressed poses and provenance."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ALARIC = HERE / ".alaric"
sys.path.insert(0, str(ALARIC))

from npy_io import load_npy  # noqa: E402
from poses import (  # noqa: E402
    PoseReader,
    discover_organized,
    pack_pool,
    write_arc_file,
)

POSES = [
    (0, 0, (0, 0, 0)),
    (0, 1, (1, 0, 0)),
    (1, 0, (17, 0, 0)),
    (1, 1, (33, 0, 0)),
]


def _write_input_pose_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    packed = pack_pool(
        np.array([p[0] for p in POSES], dtype=np.uint16),
        np.array([p[1] for p in POSES], dtype=np.uint16),
        np.array([p[2] for p in POSES], dtype=np.int32),
        bucket_size=16,
    )
    for index, (M, O, C, P) in enumerate(packed, start=1):
        write_arc_file(path / f"poses-{index}.arc", M, O, C, P, bucket_size=16)


def _run(script: str, *args: str) -> None:
    subprocess.run(
        [sys.executable, str(ALARIC / script), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _organized_names(pose_dir: Path) -> list[str]:
    return [path.name for path in discover_organized(pose_dir)]


def test_filter_poses_writes_compressed_poses_and_provenance(tmp_path: Path) -> None:
    pose_dir = tmp_path / "in"
    _write_input_pose_dir(pose_dir)
    energies = tmp_path / "score.npy"
    np.save(energies, np.array([-1.0, 5.0, -3.0, 5.0], dtype=np.float32))
    out_dir = tmp_path / "out"

    _run("filter-poses.py", str(pose_dir), str(energies), "0.0", str(out_dir), "--compress")

    assert all(name.endswith(".arc.zst") for name in _organized_names(out_dir))
    assert not (out_dir / "provenance.npy").exists()
    np.testing.assert_array_equal(
        load_npy(out_dir / "provenance.npy.zst"),
        np.array([0, 2], dtype=np.uint64),
    )


def test_filter_poses_uncompressed_by_default(tmp_path: Path) -> None:
    pose_dir = tmp_path / "in"
    _write_input_pose_dir(pose_dir)
    energies = tmp_path / "score.npy"
    np.save(energies, np.array([-1.0, 5.0, -3.0, 5.0], dtype=np.float32))
    out_dir = tmp_path / "out"

    _run("filter-poses.py", str(pose_dir), str(energies), "0.0", str(out_dir))

    assert all(name.endswith(".arc") for name in _organized_names(out_dir))
    assert (out_dir / "provenance.npy").is_file()


def test_select_poses_mask_route_writes_compressed_poses(tmp_path: Path) -> None:
    pose_dir = tmp_path / "in"
    _write_input_pose_dir(pose_dir)
    mask = tmp_path / "mask.npy"
    np.save(mask, np.array([True, False, True, False]))
    out_dir = tmp_path / "out"

    _run("select-poses.py", str(pose_dir), str(mask), str(out_dir), "--force", "--compress")

    assert _organized_names(out_dir)
    assert all(name.endswith(".arc.zst") for name in _organized_names(out_dir))
    # a boolean mask preserves pose order, so no order array is produced
    assert not list(out_dir.glob("order-array.npy*"))


def test_select_poses_index_route_compresses_the_order_array(tmp_path: Path) -> None:
    pose_dir = tmp_path / "in"
    _write_input_pose_dir(pose_dir)
    indices = tmp_path / "mask.npy"
    np.save(indices, np.array([3, 0, 2], dtype=np.uint32))
    out_dir = tmp_path / "out"

    _run("select-poses.py", str(pose_dir), str(indices), str(out_dir), "--force", "--compress")

    assert all(name.endswith(".arc.zst") for name in _organized_names(out_dir))
    assert not (out_dir / "order-array.npy").exists()
    order = load_npy(out_dir / "order-array.npy.zst")
    assert sorted(order.tolist()) == [0, 1, 2]


def test_select_poses_accepts_a_compressed_index_array(tmp_path: Path) -> None:
    """The index file is referenced by its logical name even when compressed."""
    from npy_io import save_npy

    pose_dir = tmp_path / "in"
    _write_input_pose_dir(pose_dir)
    save_npy(tmp_path / "mask.npy", np.array([True, False, True, False]))
    out_dir = tmp_path / "out"

    _run("select-poses.py", str(pose_dir), str(tmp_path / "mask.npy"), str(out_dir), "--force")

    assert PoseReader.get_nposes(out_dir) == 2
