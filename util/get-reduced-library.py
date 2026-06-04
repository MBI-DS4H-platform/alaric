#!/usr/bin/env python3
"""Load a dinucleotide fragment library, optionally exclude PDB codes,
reduce atomic coordinates to ATTRACT bead coordinates, and save the result.

Output shape: (C, B, 3) float32  — C conformers, B beads per conformer, 3 coords.

Usage:
    python3 get-reduced-library.py GU output.npy [-x 1b7f] [-x 3sxl]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_CODE = Path(__file__).with_name("alaric")
sys.path.insert(0, str(_CODE))

from library import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sequence",
                        help="Dinucleotide sequence (e.g. GU).")
    parser.add_argument("output", type=Path,
                        help="Output .npy file for reduced coordinates.")
    parser.add_argument("-x", "--exclude", action="append", default=[],
                        metavar="PDB",
                        help="PDB code to exclude (repeatable, e.g. -x 1b7f -x 3sxl).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence = args.sequence.strip().upper()
    if len(sequence) != 2:
        raise ValueError("Sequence must be a dinucleotide (length 2)")

    libraries, _ = config(verify_checksums=False)
    if sequence not in libraries:
        valid = ", ".join(sorted(libraries.keys()))
        raise ValueError(f"Unknown dinucleotide '{sequence}'. Valid: {valid}")

    excluded = [code.lower() for code in args.exclude]
    if not excluded:
        pdb_code = None
    elif len(excluded) == 1:
        pdb_code = excluded[0]
    else:
        pdb_code = excluded

    library = libraries[sequence].create(pdb_code)
    reduced = library.reduce()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, reduced)
    print(f"Saved {reduced.shape} ({reduced.dtype}) → {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
