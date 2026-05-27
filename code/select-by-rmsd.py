#!/usr/bin/env python3
"""Write pose indices whose RMSD is below a threshold, sorted by RMSD."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEFAULT_CHUNK_SIZE = 10_000_000


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _uint_dtype_for_count(count: int) -> np.dtype:
    max_value = max(0, int(count) - 1)
    if max_value <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if max_value <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if max_value <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    return np.dtype(np.uint64)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select pose indices with RMSD below a threshold.",
    )
    parser.add_argument("rmsd_file", type=Path)
    parser.add_argument("threshold", type=float)
    parser.add_argument("output_file", type=Path)
    parser.add_argument(
        "--chunksize",
        type=_positive_int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Rows per scan chunk for .npy input (default: {DEFAULT_CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--preserve-order",
        action="store_true",
        help="Keep original pose order instead of sorting selected indices by RMSD.",
    )
    return parser


def _load_rmsd(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path, mmap_mode="r")
    return np.loadtxt(path)


def select_by_rmsd(
    rmsd: np.ndarray,
    threshold: float,
    *,
    chunksize: int = DEFAULT_CHUNK_SIZE,
    sort_by_rmsd: bool = True,
) -> np.ndarray:
    if rmsd.ndim != 1:
        raise ValueError("RMSD array must be 1D")

    index_parts = []
    value_parts = []
    for start in range(0, len(rmsd), chunksize):
        stop = min(start + chunksize, len(rmsd))
        values = np.asarray(rmsd[start:stop])
        local = np.flatnonzero(values < threshold)
        if len(local) == 0:
            continue
        index_parts.append(local.astype(np.uint64) + np.uint64(start))
        if sort_by_rmsd:
            value_parts.append(values[local].astype(np.float32, copy=False))

    if not index_parts:
        return np.empty((0,), dtype=_uint_dtype_for_count(len(rmsd)))

    indices = np.concatenate(index_parts)
    if sort_by_rmsd:
        values = np.concatenate(value_parts)
        order = np.argsort(values, kind="stable")
        indices = indices[order]
    return indices.astype(_uint_dtype_for_count(len(rmsd)), copy=False)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rmsd = _load_rmsd(args.rmsd_file)
    indices = select_by_rmsd(
        rmsd,
        args.threshold,
        chunksize=args.chunksize,
        sort_by_rmsd=not args.preserve_order,
    )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_file, indices)
    print(
        f"Wrote {len(indices):,} pose index/indices to {args.output_file} "
        f"({indices.dtype})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
