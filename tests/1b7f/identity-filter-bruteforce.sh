#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d frag4-fwd ]; then
  bash stack-frag4-fwd.sh
fi
if [ ! -d frag4-bwd ]; then
  bash stack-frag4-bwd.sh
fi

rm -rf frag4-identity
python3 code/identity_filter.py frag4-fwd frag4-bwd frag4-identity

python3 - <<'PY'
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path("code").resolve()))
from poses import discover_organized, read_arc_file


def read_pose_keys(pose_dir):
    rows = []
    bucket_size = None
    for path in discover_organized(pose_dir):
        M, O, C, P, file_bucket_size = read_arc_file(path)
        if bucket_size is None:
            bucket_size = file_bucket_size
        elif file_bucket_size != bucket_size:
            raise AssertionError(f"inconsistent bucket_size in {pose_dir}")
        I = np.empty(len(C) + 1, dtype=np.uint64)
        I[0] = 0
        I[1:] = np.cumsum(C, dtype=np.uint64)
        M_key = tuple(int(x) for x in M)
        for offset_index, offset in enumerate(O):
            offset_key = tuple(int(x) for x in offset)
            start = int(I[offset_index])
            stop = int(I[offset_index + 1])
            for pose in P[start:stop]:
                rows.append(
                    (
                        *M_key,
                        *offset_key,
                        int(pose[0]),
                        int(pose[1]),
                    )
                )
    return rows


rows1 = read_pose_keys("frag4-fwd")
rows2 = read_pose_keys("frag4-bwd")
common = sorted(set(rows1) & set(rows2))
common_index = {key: i for i, key in enumerate(common)}

actual = read_pose_keys("frag4-identity")
if actual != common:
    raise AssertionError(
        f"identity output mismatch: expected {len(common)} rows, got {len(actual)}"
    )


def expected_mapping(rows):
    return np.array(
        [[source_index, common_index[key]] for source_index, key in enumerate(rows) if key in common_index],
        dtype=np.uint64,
    ).reshape((-1, 2))


for name, rows in (("map-1.npy", rows1), ("map-2.npy", rows2)):
    expected = expected_mapping(rows)
    actual_map = np.load(Path("frag4-identity") / name)
    if actual_map.shape != expected.shape or not np.array_equal(actual_map, expected):
        raise AssertionError(
            f"{name} mismatch: expected shape {expected.shape}, got {actual_map.shape}"
        )

print(f"identity-filter: OK ({len(common)} shared poses)")
PY
