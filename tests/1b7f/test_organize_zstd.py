from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import zstandard as zstd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / ".alaric"))

from organize import organize_pose_dir  # noqa: E402
from poses import HEADER_SIZE, discover_organized, pack_pool, read_arc_file, write_arc_file  # noqa: E402


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
