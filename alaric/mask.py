#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from .npy_io import save_npy
except ImportError:  # direct script execution with ALARIC_DIR on PYTHONPATH
    from npy_io import save_npy


def _uint_dtype_for_count(count: int) -> np.dtype:
    max_value = max(0, count - 1)
    if max_value <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if max_value <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if max_value <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    return np.dtype(np.uint64)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a compact pose mask from scores.")
    parser.add_argument("score_input")
    parser.add_argument("threshold", type=float)
    parser.add_argument("output", help="Output path; --compress appends .zst to it.")
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Write OUTPUT.zst (zstd) instead of OUTPUT. A boolean mask is one byte "
        "per pose and compresses by orders of magnitude.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scores = np.load(Path(args.score_input))
    if scores.ndim != 1:
        raise ValueError("score array must be 1D")
    mask = scores < float(args.threshold)
    indices = np.where(mask)[0].astype(_uint_dtype_for_count(len(scores)), copy=False)
    chosen = mask if mask.nbytes <= indices.nbytes else indices
    save_npy(Path(args.output), chosen, compress=bool(args.compress))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
