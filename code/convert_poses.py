#!/usr/bin/env python3
"""Convert alaric arc pose dir + fragment library → rotvec DOFs.

Output:
  <output-prefix>.rotvec.npy      -- (N, 6) float64  Rodrigues rotvec (0-2) +
                                      world-frame translation in Angstrom (3-5)
  <output-prefix>.conformers.npy  -- (N,) int32  0-based conformer index
"""
from __future__ import annotations

import argparse
import sys
from math import sqrt
from pathlib import Path

import numpy as np

GRID_SPACING = sqrt(3) / 3


def _existing_dir(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"directory does not exist: {path}")
    if not p.is_dir():
        raise argparse.ArgumentTypeError(f"not a directory: {path}")
    return path


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer: {value}") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("index must be positive")
    return result


def _dinucleotide_sequence(seq: str) -> str:
    seq = seq.strip().upper()
    if len(seq) != 2:
        raise argparse.ArgumentTypeError("--sequence must be a dinucleotide (length 2)")
    if any(ch not in {"A", "C", "G", "U"} for ch in seq):
        raise argparse.ArgumentTypeError("--sequence must contain only A/C/G/U")
    return seq


def _rotvecs_for(rotaconformers: np.ndarray) -> np.ndarray:
    """Return (N, 3) float64 ATTRACT-JAX/SciPy rotvecs from rotamer data."""
    if rotaconformers.ndim == 2 and rotaconformers.shape[1] == 3:
        return -rotaconformers.astype(np.float64, copy=False)
    if rotaconformers.ndim == 3 and rotaconformers.shape[1:] == (3, 3):
        from scipy.spatial.transform import Rotation
        return Rotation.from_matrix(
            np.swapaxes(rotaconformers, -1, -2)
        ).as_rotvec().astype(np.float64)
    raise ValueError(f"Unsupported rotamer shape {rotaconformers.shape}")


def _load_library(sequence: str):
    from library import config
    libraries, _ = config(verify_checksums=False)
    if sequence not in libraries:
        raise ValueError(f"Sequence {sequence} not in fragment library")
    lib_source = libraries[sequence]
    lib_source.load_rotaconformers()
    return lib_source.create(None, with_rotaconformers=True)


def _process_chunk(
    chunk,
    lib,
    rotvec_cache: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one PoseChunk → (rotvec_dofs, conformer_ids)."""
    n = len(chunk)
    conformer_ids = chunk.conformers.astype(np.int32)
    rotamer_ids = chunk.rotamers.astype(np.intp)

    rotvec_dofs = np.empty((n, 6), dtype=np.float64)
    rotvec_dofs[:, 3:] = chunk.translations_grid.astype(np.float64) * GRID_SPACING

    sort_idx = np.argsort(conformer_ids, kind="stable")
    sorted_conf = conformer_ids[sort_idx]
    boundaries = np.flatnonzero(np.diff(sorted_conf)) + 1
    group_starts = np.empty(len(boundaries) + 2, dtype=np.intp)
    group_starts[0] = 0
    group_starts[-1] = n
    group_starts[1:-1] = boundaries

    for start, end in zip(group_starts[:-1], group_starts[1:]):
        rows = sort_idx[start:end]
        conf = int(sorted_conf[start])
        rv = rotvec_cache.get(conf)
        if rv is None:
            rv = _rotvecs_for(lib.get_rotamers(conf))
            rotvec_cache[conf] = rv
        group_rots = rotamer_ids[rows]
        bad = group_rots >= len(rv)
        if np.any(bad):
            bad_pos = int(np.flatnonzero(bad)[0])
            raise ValueError(
                f"Rotamer index {int(group_rots[bad_pos])} out of range "
                f"for conformer {conf} (n_rotamers={len(rv)})"
            )
        rotvec_dofs[rows, :3] = rv[group_rots]

    return rotvec_dofs, conformer_ids


def convert_arc_dir(
    pose_dir: Path,
    pose_range: tuple[int, int] | None,
    sequence: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (rotvec_dofs, conformers).

    rotvec_dofs : (N, 6) float64 — (v0, v1, v2, tx, ty, tz)
      tx/ty/tz are world-frame translations in Angstrom
    conformers  : (N,) int32 — 0-based conformer index into the library
    """
    from poses import PoseReader

    lib = _load_library(sequence)
    rotvec_cache: dict[int, np.ndarray] = {}
    rotvec_chunks: list[np.ndarray] = []
    conf_chunks: list[np.ndarray] = []

    reader = PoseReader(pose_dir, pose_range=pose_range)
    for chunk in reader.iter_chunks():
        if len(chunk) == 0:
            continue
        rv_chunk, conf_chunk = _process_chunk(chunk, lib, rotvec_cache)
        rotvec_chunks.append(rv_chunk)
        conf_chunks.append(conf_chunk)

    if not rotvec_chunks:
        return np.empty((0, 6), dtype=np.float64), np.empty((0,), dtype=np.int32)

    return (
        np.concatenate(rotvec_chunks, axis=0),
        np.concatenate(conf_chunks, axis=0),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pose-dir", required=True, type=_existing_dir,
                    help="Directory containing poses-*.arc[.zst] files")
    ap.add_argument("--pose-range", nargs=2, metavar=("START", "END"),
                    type=_positive_int,
                    help="1-based end-inclusive global pose range (default: all poses)")
    ap.add_argument("--sequence", required=True, type=_dinucleotide_sequence,
                    help="Dinucleotide sequence (e.g. GU)")
    ap.add_argument("--output-prefix", required=True,
                    help="Output prefix (no extension)")
    args = ap.parse_args()

    pose_range = None
    if args.pose_range is not None:
        start, end = args.pose_range
        if start > end:
            ap.error("--pose-range START must be <= END")
        pose_range = (start, end)

    rotvec_dofs, conformers = convert_arc_dir(
        Path(args.pose_dir),
        pose_range,
        args.sequence,
    )

    prefix = args.output_prefix
    np.save(prefix + ".rotvec.npy", rotvec_dofs)
    np.save(prefix + ".conformers.npy", conformers)
    print(
        f"Wrote {prefix}.rotvec.npy ({rotvec_dofs.shape}) "
        f"and {prefix}.conformers.npy ({conformers.shape})"
    )


if __name__ == "__main__":
    main()
