#!/usr/bin/env python3
"""Extract ATTRACT atom types from a reduced template ppdb.npy.

Atom types are stored in the 'occupancy' field of the ppdb array.

Usage:
    python3 ligand-atomtypes.py --sequence GU [OUTPUT_NPY]
    python3 ligand-atomtypes.py PPDB_NPY [OUTPUT_NPY]

Without an output path the atom-type array is printed to stdout.
ATTRACT templates are resolved from fraglib/templates/ATTRACT/ relative to this repo.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_FRAGLIB_ATTRACT = _HERE.parent / "fraglib" / "templates" / "ATTRACT"


def extract_atomtypes(ppdb_path: Path) -> np.ndarray:
    pdb = np.load(ppdb_path, allow_pickle=False)
    if "occupancy" not in pdb.dtype.names:
        raise ValueError(f"{ppdb_path}: expected structured array with 'occupancy' field")
    return pdb["occupancy"].astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", "-s", metavar="SEQ",
                    help="Dinucleotide sequence (e.g. GU); resolves ppdb from ATTRACT templates")
    ap.add_argument("ppdb_or_output", nargs="?", metavar="PPDB_NPY_OR_OUTPUT",
                    help="Either a ppdb.npy path (when --sequence is not given) "
                         "or the output atomtypes.npy path (when --sequence is given)")
    ap.add_argument("output_only", nargs="?", metavar="OUTPUT_NPY",
                    help="Output atomtypes.npy (when ppdb_or_output is the ppdb path)")

    args = ap.parse_args()

    if args.sequence is not None:
        seq = args.sequence.strip().upper()
        ppdb_path = _FRAGLIB_ATTRACT / f"{seq}-ppdb.npy"
        if not ppdb_path.exists():
            print(f"Error: {ppdb_path} not found", file=sys.stderr)
            sys.exit(1)
        output = args.ppdb_or_output  # first positional = output when --sequence given
    else:
        if args.ppdb_or_output is None:
            ap.error("provide either --sequence SEQ or a positional PPDB_NPY path")
        ppdb_path = Path(args.ppdb_or_output)
        output = args.output_only

    atomtypes = extract_atomtypes(ppdb_path)
    print(f"Extracted {len(atomtypes)} atom types from {ppdb_path}", file=sys.stderr)

    if output:
        out = Path(output)
        np.save(out, atomtypes)
        print(f"Saved → {out}", file=sys.stderr)
    else:
        print(atomtypes)


if __name__ == "__main__":
    main()
