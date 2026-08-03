"""Grow's restrict-pose index.

The index is grouped in one global pass rather than per read chunk, which is a pure
performance change -- so what these pin down is the contract that change has to keep:
one entry per (conformer, rotamer), translations sorted and deduplicated, whatever the
pool's file/chunk layout happens to be.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "alaric"))

from alaric.grow import _load_restrict_index, _pack_translations  # noqa: E402
from organize import organize_pose_dir  # noqa: E402
from poses import PoseWriter  # noqa: E402


def _build_pool(
    path: Path,
    poses: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    max_poses_per_file: int = 64,
    cache_poses: int = 50,
) -> None:
    """Organize the poses into a pool spanning many files (and so many read chunks)."""
    conformers, rotamers, translations = poses
    writer = PoseWriter(path, bucket_size=16, cache_poses=cache_poses)
    for start in range(0, len(conformers), 37):
        stop = start + 37
        writer.add_chunk(
            conformers[start:stop], rotamers[start:stop], translations[start:stop]
        )
    writer.finish()
    organize_pose_dir(
        path, compress=True, nprocs=1, max_poses_per_file=max_poses_per_file
    )


def _expected(poses) -> dict[int, dict[int, np.ndarray]]:
    """The index, computed the obvious slow way."""
    conformers, rotamers, translations = poses
    packed = _pack_translations(translations)
    expected: dict[int, dict[int, set]] = {}
    for conformer, rotamer, value in zip(
        conformers.tolist(), rotamers.tolist(), packed.tolist()
    ):
        expected.setdefault(int(conformer), {}).setdefault(int(rotamer), set()).add(value)
    return {
        conformer: {
            rotamer: np.array(sorted(values), dtype=np.uint64)
            for rotamer, values in by_rotamer.items()
        }
        for conformer, by_rotamer in expected.items()
    }


def _assert_index_equals(actual, expected) -> None:
    assert sorted(actual) == sorted(expected)
    for conformer in expected:
        assert sorted(actual[conformer]) == sorted(expected[conformer])
        for rotamer, values in expected[conformer].items():
            found = actual[conformer][rotamer]
            assert found.dtype == np.uint64
            np.testing.assert_array_equal(found, values)


def _poses(n: int, *, conformers: int, rotamers: int, duplicate_every: int = 0):
    rng = np.random.default_rng(1234)
    conf = rng.integers(0, conformers, n).astype(np.uint16)
    rot = rng.integers(0, rotamers, n).astype(np.uint16)
    trans = rng.integers(-20, 20, (n, 3)).astype(np.int16)
    if duplicate_every:
        # exact duplicate poses: same conformer, rotamer *and* translation
        conf[::duplicate_every] = conf[0]
        rot[::duplicate_every] = rot[0]
        trans[::duplicate_every] = trans[0]
    return conf, rot, trans


@pytest.mark.parametrize(
    "conformers,rotamers,duplicate_every",
    [
        (3, 3, 0),  # few pairs, many poses each
        (40, 40, 4),  # many pairs, with exact duplicates
        (200, 200, 0),  # mostly-unique pairs: one group per pose
    ],
)
def test_index_matches_the_obvious_grouping(
    tmp_path, conformers, rotamers, duplicate_every
):
    poses = _poses(900, conformers=conformers, rotamers=rotamers, duplicate_every=duplicate_every)
    _build_pool(tmp_path / "pool", poses)
    # the fixture must really span many files, or chunk-boundary grouping is untested
    assert len(list((tmp_path / "pool").glob("poses-*.arc.zst"))) > 5

    _assert_index_equals(_load_restrict_index(tmp_path / "pool"), _expected(poses))


def test_index_is_independent_of_the_read_chunk_size(tmp_path):
    poses = _poses(500, conformers=20, rotamers=20, duplicate_every=5)
    _build_pool(tmp_path / "pool", poses)

    reference = _load_restrict_index(tmp_path / "pool", chunk_poses=1_000_000)
    for chunk_poses in (1, 7, 64):
        _assert_index_equals(
            _load_restrict_index(tmp_path / "pool", chunk_poses=chunk_poses), reference
        )


def test_translations_are_sorted_for_searchsorted(tmp_path):
    """_restrict_output_mask binary-searches these, so order is load-bearing."""
    poses = _poses(400, conformers=8, rotamers=8, duplicate_every=3)
    _build_pool(tmp_path / "pool", poses)

    index = _load_restrict_index(tmp_path / "pool")
    for by_rotamer in index.values():
        for values in by_rotamer.values():
            assert np.all(np.diff(values.astype(np.int64)) > 0)  # sorted and unique


def test_empty_pool_is_rejected(tmp_path):
    pool = tmp_path / "pool"
    pool.mkdir()
    with pytest.raises((ValueError, FileNotFoundError)):
        _load_restrict_index(pool)
