from __future__ import annotations

import io
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import zstandard as zstd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / ".alaric"))

from npy_io import load_npy, save_npy  # noqa: E402
from organize import organize_pose_dir  # noqa: E402
from poses import (  # noqa: E402
    HEADER_SIZE,
    PoseReader,
    discover_organized,
    pack_pool,
    read_arc_file,
    select_pose_indices,
    write_arc_file,
)
from identity_filter import run_identity_filter, write_identity_pose_dir  # noqa: E402


def test_compressed_organized_arc_records_frame_content_size(tmp_path: Path) -> None:
    pose_dir = tmp_path / "poses"
    pose_dir.mkdir()
    packed = pack_pool(
        np.array([0, 1, 1], dtype=np.uint16),
        np.array([0, 0, 2], dtype=np.uint16),
        np.array([[0, 0, 0], [3, 0, 0], [17, 0, 0]], dtype=np.int16),
        bucket_size=16,
    )
    for index, (M, O, C, P) in enumerate(packed, start=1):
        write_arc_file(
            pose_dir / f"unorganized-test-{index}.arc.zst",
            M,
            O,
            C,
            P,
            bucket_size=16,
            zstd=True,
        )

    organize_pose_dir(pose_dir, compress=True, nprocs=1, max_poses_per_file=100)

    organized = discover_organized(pose_dir)
    assert organized
    for path in organized:
        _M, O, _C, P, _bucket_size = read_arc_file(path)
        expected_size = HEADER_SIZE + 10 * len(O) + 6 * len(P)
        assert zstd.frame_content_size(path.read_bytes()) == expected_size


def test_identity_filter_can_write_compressed_pose_output(tmp_path: Path) -> None:
    pose_dir = tmp_path / "identity"
    identity = {
        ((0, 0, 0), (0, 0, 0)): np.array([0, 1], dtype=np.uint32),
        ((0, 0, 0), (1, 0, 0)): np.array([2], dtype=np.uint32),
    }

    lookup = write_identity_pose_dir(
        identity,
        pose_dir,
        bucket_size=16,
        max_poses_per_file=100,
        compress=True,
    )

    organized = discover_organized(pose_dir)
    assert [path.name for path in organized] == ["poses-1.arc.zst"]
    M, O, C, P, bucket_size = read_arc_file(organized[0])
    np.testing.assert_array_equal(M, np.array([0, 0, 0], dtype=np.int16))
    np.testing.assert_array_equal(
        O,
        np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int16),
    )
    np.testing.assert_array_equal(C, np.array([2, 1], dtype=np.uint32))
    np.testing.assert_array_equal(
        P,
        np.array([[0, 0, 0], [0, 1, 0], [0, 2, 1]], dtype=np.uint16),
    )
    assert bucket_size == 16
    assert lookup[((0, 0, 0), (1, 0, 0))][0] == 2


def _write_organized_pose_dir(path: Path, poses: list[tuple[int, int, tuple[int, int, int]]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    packed = pack_pool(
        np.array([p[0] for p in poses], dtype=np.uint16),
        np.array([p[1] for p in poses], dtype=np.uint16),
        np.array([p[2] for p in poses], dtype=np.int32),
        bucket_size=16,
    )
    for index, (M, O, C, P) in enumerate(packed, start=1):
        write_arc_file(path / f"poses-{index}.arc", M, O, C, P, bucket_size=16)


def test_identity_filter_compresses_its_map_arrays(tmp_path: Path) -> None:
    shared = [(0, 0, (0, 0, 0)), (1, 1, (1, 0, 0))]
    _write_organized_pose_dir(tmp_path / "one", shared + [(2, 2, (2, 0, 0))])
    _write_organized_pose_dir(tmp_path / "two", shared + [(3, 3, (3, 0, 0))])
    out_dir = tmp_path / "identity"

    manifest = run_identity_filter(
        tmp_path / "one", tmp_path / "two", out_dir, compress=True
    )

    assert manifest["identity_poses"] == len(shared)
    assert not (out_dir / "map-1.npy").exists()
    assert not (out_dir / "map-2.npy").exists()
    for name in ("map-1.npy.zst", "map-2.npy.zst"):
        rows = load_npy(out_dir / name)
        assert rows.shape == (len(shared), 2)
        assert rows.dtype == np.uint64


def _write_provenance_shards(pose_dir: Path, *, compress_sidecars: bool) -> None:
    """Two unorganized shards whose provenance interleaves once reordered."""
    pose_dir.mkdir(exist_ok=True)
    M = np.array([0, 0, 0], dtype=np.int16)
    O = np.array([[0, 0, 0]], dtype=np.int16)
    C = np.array([2], dtype=np.uint32)

    for index, (P, provenance) in enumerate(
        (
            ([[2, 0, 0], [1, 1, 0]], [20, 11]),
            ([[1, 0, 0], [2, 0, 0]], [10, 21]),
        ),
        start=1,
    ):
        shard = pose_dir / f"unorganized-test-{index}.arc.zst"
        write_arc_file(
            shard,
            M,
            O,
            C,
            np.array(P, dtype=np.uint16),
            bucket_size=16,
            zstd=True,
        )
        save_npy(
            shard.with_name(shard.name + ".provenance.npy"),
            np.array(provenance, dtype=np.uint32),
            compress=compress_sidecars,
        )


def _assert_reordered(pose_dir: Path, provenance_path: Path) -> None:
    organized = discover_organized(pose_dir)
    assert [path.name for path in organized] == ["poses-1.arc.zst"]
    _M, _O, _C, P, _bucket_size = read_arc_file(organized[0])
    np.testing.assert_array_equal(
        P,
        np.array(
            [[1, 0, 0], [1, 1, 0], [2, 0, 0], [2, 0, 0]],
            dtype=np.uint16,
        ),
    )
    np.testing.assert_array_equal(
        load_npy(provenance_path),
        np.array([10, 11, 20, 21], dtype=np.uint32),
    )
    assert not list(pose_dir.glob("*.provenance.npy*"))


def test_organize_reorders_provenance_sidecars(tmp_path: Path) -> None:
    pose_dir = tmp_path / "poses"
    _write_provenance_shards(pose_dir, compress_sidecars=True)

    organize_pose_dir(pose_dir, compress=True, nprocs=1, max_poses_per_file=100)

    assert not (pose_dir / "provenance.npy").exists()
    _assert_reordered(pose_dir, pose_dir / "provenance.npy.zst")


def test_organize_reads_uncompressed_provenance_sidecars(tmp_path: Path) -> None:
    """Pools written before the sidecars were compressed still organize."""
    pose_dir = tmp_path / "poses"
    _write_provenance_shards(pose_dir, compress_sidecars=False)

    organize_pose_dir(pose_dir, compress=True, nprocs=1, max_poses_per_file=100)

    _assert_reordered(pose_dir, pose_dir / "provenance.npy.zst")


def test_organize_without_compress_leaves_provenance_uncompressed(tmp_path: Path) -> None:
    pose_dir = tmp_path / "poses"
    _write_provenance_shards(pose_dir, compress_sidecars=True)

    organize_pose_dir(pose_dir, compress=False, nprocs=1, max_poses_per_file=100)

    assert not (pose_dir / "provenance.npy.zst").exists()
    organized = discover_organized(pose_dir)
    assert [path.name for path in organized] == ["poses-1.arc"]
    np.testing.assert_array_equal(
        np.load(pose_dir / "provenance.npy"),
        np.array([10, 11, 20, 21], dtype=np.uint32),
    )


@pytest.mark.skipif(shutil.which("zstd") is None, reason="staging needs the zstd CLI")
def test_local_tempdir_staging_carries_compressed_sidecars(tmp_path: Path) -> None:
    pose_dir = tmp_path / "poses"
    _write_provenance_shards(pose_dir, compress_sidecars=True)

    organize_pose_dir(
        pose_dir,
        compress=True,
        nprocs=1,
        max_poses_per_file=100,
        local_tempdir=True,
        local_stagedir=True,
    )

    _assert_reordered(pose_dir, pose_dir / "provenance.npy.zst")


def test_compressed_provenance_matches_uncompressed_bytes(tmp_path: Path) -> None:
    """Compression is checksum-transparent: same .npy bytes under the .zst frame."""
    compressed_dir = tmp_path / "compressed"
    plain_dir = tmp_path / "plain"
    _write_provenance_shards(compressed_dir, compress_sidecars=True)
    _write_provenance_shards(plain_dir, compress_sidecars=True)

    organize_pose_dir(compressed_dir, compress=True, nprocs=1, max_poses_per_file=100)
    organize_pose_dir(plain_dir, compress=False, nprocs=1, max_poses_per_file=100)

    payload = (compressed_dir / "provenance.npy.zst").read_bytes()
    assert zstd.ZstdDecompressor().stream_reader(io.BytesIO(payload)).read() == (
        plain_dir / "provenance.npy"
    ).read_bytes()


def _write_one_pose_per_file(pose_dir: Path, nposes: int) -> None:
    """Organized pool of ``nposes`` poses, each in its own bucket hence its own file."""
    pose_dir.mkdir(parents=True, exist_ok=True)
    packed = pack_pool(
        np.arange(1, nposes + 1, dtype=np.uint16),
        np.zeros(nposes, dtype=np.uint16),
        np.array([[64 * i, 0, 0] for i in range(nposes)], dtype=np.int32),
        bucket_size=16,
        sort_offsets=False,
    )
    assert len(packed) == nposes
    for index, (M, O, C, P) in enumerate(packed, start=1):
        write_arc_file(pose_dir / f"poses-{index}.arc", M, O, C, P, bucket_size=16)


def test_a_decompressed_copy_beside_its_zst_is_one_pose_file(tmp_path: Path) -> None:
    """Both forms of one file must not be listed twice.

    A result dir can end up holding a decompressed copy next to the .arc.zst it
    came from.  Counting both doubles every pose and, across several files,
    shifts every global pose id onto a different pose -- so a provenance array
    resolves to real but wrong poses, silently.
    """
    pose_dir = tmp_path / "poses"
    _write_one_pose_per_file(pose_dir, 4)
    before = discover_organized(pose_dir)
    assert PoseReader.get_nposes(pose_dir) == 4
    identities = select_pose_indices(pose_dir, np.arange(4)).conformers.tolist()

    for path in list(pose_dir.glob("poses-*.arc")):
        path.with_name(path.name + ".zst").write_bytes(
            zstd.ZstdCompressor().compress(path.read_bytes())
        )

    assert discover_organized(pose_dir) == before
    assert PoseReader.get_nposes(pose_dir) == 4
    assert select_pose_indices(pose_dir, np.arange(4)).conformers.tolist() == identities


def test_the_compressed_form_is_used_when_it_is_the_only_one(tmp_path: Path) -> None:
    pose_dir = tmp_path / "poses"
    _write_one_pose_per_file(pose_dir, 3)
    for path in list(pose_dir.glob("poses-*.arc")):
        path.with_name(path.name + ".zst").write_bytes(
            zstd.ZstdCompressor().compress(path.read_bytes())
        )
        path.unlink()

    assert [path.name for path in discover_organized(pose_dir)] == [
        "poses-1.arc.zst",
        "poses-2.arc.zst",
        "poses-3.arc.zst",
    ]
    assert PoseReader.get_nposes(pose_dir) == 3
