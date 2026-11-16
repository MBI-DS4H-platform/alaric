#!/usr/bin/env python3
"""Add two one-dimensional per-pose score arrays in input order."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np

try:
    from .npy_io import NpyWriter, find_npy, open_npy_mmap
except ImportError:  # direct script execution with ALARIC_DIR on PYTHONPATH
    from npy_io import NpyWriter, find_npy, open_npy_mmap


COPY_CHUNK = 4_000_000


def add_score_arrays(score_input1: str | Path, score_input2: str | Path, output: str | Path) -> Path:
    """Add two numeric score arrays without materializing pool-scale inputs."""
    path1 = find_npy(score_input1)
    path2 = find_npy(score_input2)
    if path1 is None:
        raise FileNotFoundError(f"no such score array: {score_input1}")
    if path2 is None:
        raise FileNotFoundError(f"no such score array: {score_input2}")

    output = Path(output)
    with ExitStack() as stack:
        arr1 = stack.enter_context(open_npy_mmap(path1))
        arr2 = stack.enter_context(open_npy_mmap(path2))
        if arr1.ndim != 1 or arr2.ndim != 1:
            raise ValueError("score arrays must be 1D")
        if arr1.shape != arr2.shape:
            raise ValueError(f"score array shape mismatch: {arr1.shape} != {arr2.shape}")
        for array, path in ((arr1, path1), (arr2, path2)):
            if array.dtype.kind not in "iuf":
                raise ValueError(f"{path}: score array must have a real numeric dtype")
        dtype = np.result_type(arr1.dtype, arr2.dtype)
        with NpyWriter(output, dtype=dtype, shape=arr1.shape, compress=False) as writer:
            for start in range(0, len(arr1), COPY_CHUNK):
                stop = min(start + COPY_CHUNK, len(arr1))
                writer.write(arr1[start:stop] + arr2[start:stop])
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add two score arrays.")
    parser.add_argument("score_input1")
    parser.add_argument("score_input2")
    parser.add_argument("output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    add_score_arrays(args.score_input1, args.score_input2, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
