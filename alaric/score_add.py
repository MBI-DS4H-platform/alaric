#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add two score arrays.")
    parser.add_argument("score_input1")
    parser.add_argument("score_input2")
    parser.add_argument("output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    arr1 = np.load(Path(args.score_input1))
    arr2 = np.load(Path(args.score_input2))
    if arr1.ndim != 1 or arr2.ndim != 1:
        raise ValueError("score arrays must be 1D")
    if arr1.shape != arr2.shape:
        raise ValueError(f"score array shape mismatch: {arr1.shape} != {arr2.shape}")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, arr1 + arr2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
