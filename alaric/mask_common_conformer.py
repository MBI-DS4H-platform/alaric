#!/usr/bin/env python3
"""Mask poses whose conformer occurs in both input pose pools."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from .npy_io import save_npy
    from .poses import discover_organized, iter_arc_pose_chunks, read_arc_header
except ImportError:  # direct script execution with ALARIC_DIR on PYTHONPATH
    from npy_io import save_npy
    from poses import discover_organized, iter_arc_pose_chunks, read_arc_header


def _load_pool(pool: Path, *, retain_conformers: bool = True) -> tuple[np.ndarray | None, str, int]:
    """Read conformers and fingerprint the ordered pose records in one pass."""
    paths = discover_organized(pool)
    if not paths:
        raise FileNotFoundError(f"No organized poses-*.arc files found in {pool}")
    total = sum(read_arc_header(path)[2] for path in paths)
    conformers = np.empty(total, dtype=np.uint16) if retain_conformers else None
    digest = hashlib.sha256()
    offset = 0
    for path in paths:
        for _M, _O, _C, poses, _bucket_size in iter_arc_pose_chunks(path):
            # The order identity represents decoded pose records rather than file bytes,
            # so compression and archive rechunking do not change it.
            M, O, _C, bucket_size = _M, _O, _C, _bucket_size
            translations = O.astype(np.int32) + int(bucket_size) * M.astype(np.int32)
            records = np.empty((len(poses), 5), dtype="<i8")
            records[:, :2] = poses[:, :2]
            records[:, 2:] = translations[poses[:, 2].astype(np.int64)]
            digest.update(records.tobytes(order="C"))
            stop = offset + len(poses)
            if conformers is not None:
                conformers[offset:stop] = poses[:, 0]
            offset = stop
    assert offset == total
    return conformers, digest.hexdigest(), total


def _load_conformers(pool: Path) -> np.ndarray:
    """Compatibility helper returning just the conformer column."""
    conformers, _digest, _count = _load_pool(pool)
    assert conformers is not None
    return conformers


def pose_order_identity(pool: str | Path) -> tuple[int, str]:
    """Return (pose count, ordered record digest) for mask-pair validation."""
    _conformers, digest, count = _load_pool(Path(pool), retain_conformers=False)
    return count, digest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write per-pool masks for conformers common to two pose pools."
    )
    parser.add_argument("input1", help="First organized pose pool")
    parser.add_argument("input2", help="Second organized pose pool")
    parser.add_argument("output", help="Output directory for mask1.npy.zst and mask2.npy.zst")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    conformers1, identity1, count1 = _load_pool(Path(args.input1))
    conformers2, identity2, count2 = _load_pool(Path(args.input2))
    assert conformers1 is not None and conformers2 is not None
    identities = {
        "schema": 1,
        "sources": {
            "1": {"poses": count1, "order_sha256": identity1},
            "2": {"poses": count2, "order_sha256": identity2},
        },
    }
    common = np.intersect1d(conformers1, conformers2, assume_unique=False)

    # Process one pool at a time: only its conformer array and mask remain live while its
    # output is written, which keeps the peak below two masks plus both conformer arrays.
    mask1 = np.isin(conformers1, common, assume_unique=False)
    save_npy(output / "mask1.npy", mask1, compress=True)
    del mask1, conformers1

    mask2 = np.isin(conformers2, common, assume_unique=False)
    save_npy(output / "mask2.npy", mask2, compress=True)
    del mask2, conformers2, common
    (output / "mask-pair.json").write_text(
        json.dumps(identities, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
