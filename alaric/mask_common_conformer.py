#!/usr/bin/env python3
"""Mask poses whose conformer occurs in both input pose pools."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from .npy_io import save_npy
    from .poses import discover_organized, iter_arc_pose_chunks, read_arc_header
except ImportError:  # direct script execution with ALARIC_DIR on PYTHONPATH
    from npy_io import save_npy
    from poses import discover_organized, iter_arc_pose_chunks, read_arc_header


def _load_conformers(pool: Path) -> np.ndarray:
    """Read just the conformer column, leaving all other pose data streamed."""
    paths = discover_organized(pool)
    if not paths:
        raise FileNotFoundError(f"No organized poses-*.arc files found in {pool}")
    conformers = np.empty(sum(read_arc_header(path)[2] for path in paths), dtype=np.uint16)
    offset = 0
    for path in paths:
        for _M, _O, _C, poses, _bucket_size in iter_arc_pose_chunks(path):
            stop = offset + len(poses)
            conformers[offset:stop] = poses[:, 0]
            offset = stop
    assert offset == len(conformers)
    return conformers


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

    conformers1 = _load_conformers(Path(args.input1))
    conformers2 = _load_conformers(Path(args.input2))
    common = np.intersect1d(conformers1, conformers2, assume_unique=False)

    # Process one pool at a time: only its conformer array and mask remain live while its
    # output is written, which keeps the peak below two masks plus both conformer arrays.
    mask1 = np.isin(conformers1, common, assume_unique=False)
    save_npy(output / "mask1.npy", mask1, compress=True)
    del mask1, conformers1

    mask2 = np.isin(conformers2, common, assume_unique=False)
    save_npy(output / "mask2.npy", mask2, compress=True)
    del mask2, conformers2, common
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
