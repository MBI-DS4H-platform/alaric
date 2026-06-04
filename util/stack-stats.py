#!/usr/bin/env python3
from __future__ import annotations

import argparse
from math import degrees
from pathlib import Path
import sys

import numpy as np

ALARIC_DIR = Path(__file__).with_name("alaric")
if str(ALARIC_DIR) not in sys.path:
    sys.path.insert(0, str(ALARIC_DIR))

from parse_pdb import parse_pdb  # noqa: E402
from stack_geometry import (  # noqa: E402
    protein_ring_coordinates,
    rna_ring_coordinates,
    stacking_angle_dihedral,
)


def _existing_file(path: str) -> Path:
    result = Path(path)
    if not result.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    return result


def _first_model(atoms: np.ndarray) -> np.ndarray:
    if np.any(atoms["model"] == 1):
        return atoms[atoms["model"] == 1]
    return atoms


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Report the stacking angle and dihedral for one RNA nucleotide and "
            "one protein aromatic residue, using the same convention as anchor.py."
        )
    )
    parser.add_argument("rna_pdb", type=_existing_file, help="RNA PDB file.")
    parser.add_argument(
        "nucleotide_resid",
        type=int,
        help="Residue ID of the nucleotide whose base stacks.",
    )
    parser.add_argument("protein_pdb", type=_existing_file, help="Protein PDB file.")
    parser.add_argument(
        "protein_resid",
        type=int,
        help="Residue ID of the protein aromatic residue that stacks.",
    )
    parser.add_argument(
        "--radians",
        action="store_true",
        help="Print radians instead of degrees.",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Do not print the TSV column header.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    rna_atoms = _first_model(parse_pdb(args.rna_pdb.read_text()))
    protein_atoms = _first_model(parse_pdb(args.protein_pdb.read_text()))

    nucleotide_ring = rna_ring_coordinates(rna_atoms, args.nucleotide_resid)
    protein_ring = protein_ring_coordinates(protein_atoms, args.protein_resid)
    angle, dihedral = stacking_angle_dihedral(protein_ring, nucleotide_ring)

    if args.radians:
        columns = ("angle_rad", "dihedral_rad")
        values = (angle, dihedral)
    else:
        columns = ("angle_deg", "dihedral_deg")
        values = (degrees(angle), degrees(dihedral))

    if not args.no_header:
        print("\t".join(columns))
    print(f"{values[0]:.6f}\t{values[1]:.6f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (argparse.ArgumentTypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
