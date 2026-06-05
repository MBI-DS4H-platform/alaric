#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


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
    parser.add_argument("output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scores = np.load(Path(args.score_input))
    if scores.ndim != 1:
        raise ValueError("score array must be 1D")
    mask = scores < float(args.threshold)
    indices = np.where(mask)[0].astype(_uint_dtype_for_count(len(scores)), copy=False)
    chosen = mask if mask.nbytes <= indices.nbytes else indices
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, chosen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
