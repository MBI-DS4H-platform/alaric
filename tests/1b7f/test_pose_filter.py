"""Streaming semantics of the filter action's two routes.

The scripts never see the whole pool, so what has to be pinned down is that the *result*
is unchanged by that: the kept poses come out in input order, provenance points back at
the input poses, and the order array points back at the request -- all independent of how
many workers split the work.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
ALARIC = HERE / ".alaric"
sys.path.insert(0, str(ALARIC))

from npy_io import load_npy, save_npy  # noqa: E402
from organize import organize_pose_dir  # noqa: E402
from poses import PoseReader, PoseWriter, select_pose_indices  # noqa: E402

BUCKET_SIZE = 16
# Small enough that the pool spans many organized files -- i.e. many parallel tasks.
MAX_POSES_PER_FILE = 23


def _pool_poses(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A pool spanning several buckets, with duplicate poses among the candidates."""
    rng = np.random.default_rng(20260803)
    conformers = rng.integers(0, 4, n).astype(np.uint16)
    rotamers = rng.integers(0, 3, n).astype(np.uint16)
    translations = rng.integers(-40, 40, (n, 3)).astype(np.int16)
    # force exact duplicates (same conformer/rotamer/translation) into the pool
    translations[1::7] = translations[0]
    conformers[1::7] = conformers[0]
    rotamers[1::7] = rotamers[0]
    return conformers, rotamers, translations


def _make_pool(path: Path, poses) -> None:
    conformers, rotamers, translations = poses
    writer = PoseWriter(path, bucket_size=BUCKET_SIZE, cache_poses=64)
    for start in range(0, len(conformers), 50):
        stop = start + 50
        writer.add_chunk(
            conformers[start:stop], rotamers[start:stop], translations[start:stop]
        )
    writer.finish()
    organize_pose_dir(
        path, compress=False, nprocs=1, max_poses_per_file=MAX_POSES_PER_FILE
    )


def _identities(pose_dir: Path) -> list[tuple[int, ...]]:
    """(conformer, rotamer, tx, ty, tz) per pose, in pose-dir order."""
    n = PoseReader.get_nposes(pose_dir)
    chunk = select_pose_indices(pose_dir, np.arange(n, dtype=np.int64))
    return [
        (int(c), int(r), int(t[0]), int(t[1]), int(t[2]))
        for c, r, t in zip(chunk.conformers, chunk.rotamers, chunk.translations_grid)
    ]


def _run(script: str, *args: str) -> str:
    proc = subprocess.run(
        [sys.executable, str(ALARIC / script), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout


@pytest.fixture(scope="module")
def pool(tmp_path_factory) -> tuple[Path, list[tuple[int, ...]]]:
    path = tmp_path_factory.mktemp("pool") / "in"
    _make_pool(path, _pool_poses(400))
    identities = _identities(path)
    assert len(identities) == 400
    # the fixture is only meaningful if it really spans several files and buckets
    assert len(list(path.glob("poses-*.arc"))) > 4
    return path, identities


@pytest.mark.parametrize("nprocs", [1, 3])
def test_filter_keeps_input_order_and_records_provenance(pool, tmp_path, nprocs):
    pose_dir, input_ids = pool
    rng = np.random.default_rng(7)
    scores = rng.normal(size=len(input_ids)).astype(np.float32)
    score_path = tmp_path / "score.npy"
    np.save(score_path, scores)
    expected_keep = np.flatnonzero(scores < 0.0)
    assert 0 < len(expected_keep) < len(input_ids)
    out_dir = tmp_path / f"out{nprocs}"

    _run(
        "filter-poses.py",
        str(pose_dir),
        str(score_path),
        "0.0",
        str(out_dir),
        "--force",
        "--compress",
        "--nprocs",
        str(nprocs),
        "--chunk-poses",
        "7",
    )

    provenance = load_npy(out_dir / "provenance.npy.zst")
    assert provenance.dtype == np.uint64
    # provenance is the kept input poses, in input order ...
    assert provenance.tolist() == expected_keep.tolist()
    # ... and the organized output is those same poses, in that same order
    assert _identities(out_dir) == [input_ids[i] for i in expected_keep]


def test_filter_is_independent_of_worker_count(pool, tmp_path):
    pose_dir, input_ids = pool
    scores = np.linspace(-1.0, 1.0, len(input_ids), dtype=np.float32)
    score_path = tmp_path / "score.npy"
    np.save(score_path, scores)

    outputs = []
    for nprocs in (1, 4):
        out_dir = tmp_path / f"out-{nprocs}"
        _run(
            "filter-poses.py",
            str(pose_dir),
            str(score_path),
            "0.25",
            str(out_dir),
            "--force",
            "--nprocs",
            str(nprocs),
        )
        outputs.append(
            (
                _identities(out_dir),
                load_npy(out_dir / "provenance.npy").tolist(),
                sorted(p.name for p in out_dir.glob("poses-*.arc")),
            )
        )
    assert outputs[0] == outputs[1]


def test_filter_rejects_a_score_file_of_the_wrong_length(pool, tmp_path):
    pose_dir, input_ids = pool
    score_path = tmp_path / "short.npy"
    np.save(score_path, np.zeros(len(input_ids) - 1, dtype=np.float32))

    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        _run("filter-poses.py", str(pose_dir), str(score_path), "0.0", str(tmp_path / "o"), "--force")
    assert "selector covers" in excinfo.value.stderr


def test_filter_accepts_a_compressed_score_file(pool, tmp_path):
    pose_dir, input_ids = pool
    scores = np.linspace(-1.0, 1.0, len(input_ids), dtype=np.float32)
    save_npy(tmp_path / "score.npy", scores)
    out_dir = tmp_path / "out"

    _run("filter-poses.py", str(pose_dir), str(tmp_path / "score.npy"), "0.0", str(out_dir), "--force")

    assert load_npy(out_dir / "provenance.npy").tolist() == np.flatnonzero(scores < 0).tolist()


@pytest.mark.parametrize("nprocs", [1, 3])
def test_select_order_array_maps_organized_poses_back_to_the_request(
    pool, tmp_path, nprocs
):
    pose_dir, input_ids = pool
    rng = np.random.default_rng(11)
    request = rng.choice(len(input_ids), size=120, replace=False).astype(np.uint32)
    request[3] = request[0]  # a duplicated request entry
    rng.shuffle(request)  # ... and an out-of-order one
    np.save(tmp_path / "select.npy", request)
    out_dir = tmp_path / f"out{nprocs}"

    _run(
        "select-poses.py",
        str(pose_dir),
        str(tmp_path / "select.npy"),
        str(out_dir),
        "--force",
        "--nprocs",
        str(nprocs),
    )

    order = load_npy(out_dir / "order-array.npy")
    out_ids = _identities(out_dir)
    assert len(out_ids) == len(request)
    # the order array is a permutation of request positions ...
    assert sorted(order.tolist()) == list(range(len(request)))
    # ... and each organized pose is the pose the request asked for at that position
    for organized_position, request_position in enumerate(order.tolist()):
        assert out_ids[organized_position] == input_ids[int(request[request_position])]
    # the narrowest dtype that holds a request position, as before the rewrite
    assert order.dtype == np.uint8


def test_select_with_a_boolean_mask_needs_no_order_array(pool, tmp_path):
    pose_dir, input_ids = pool
    mask = np.zeros(len(input_ids), dtype=bool)
    mask[::5] = True
    np.save(tmp_path / "mask.npy", mask)
    out_dir = tmp_path / "out"

    _run(
        "select-poses.py",
        str(pose_dir),
        str(tmp_path / "mask.npy"),
        str(out_dir),
        "--force",
    )

    assert not list(out_dir.glob("order-array.npy*"))
    assert not list(out_dir.glob("provenance.npy*"))
    assert _identities(out_dir) == [input_ids[i] for i in np.flatnonzero(mask)]
