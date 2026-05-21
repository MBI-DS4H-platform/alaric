#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from library import config  # noqa: E402
from parse_pdb import parse_pdb  # noqa: E402
from reference import Reference  # noqa: E402
from superimpose import superimpose_array  # noqa: E402


def _existing_file(path: str) -> Path:
    result = Path(path)
    if not result.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    return result


def _pdb_code(value: str) -> str:
    code = value.strip()
    if len(code) != 4 or not code[0].isdigit() or not code[1:].isalnum():
        raise argparse.ArgumentTypeError(
            "PDB codes must be 4 chars: one digit + 3 alphanumeric characters"
        )
    return code.lower()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cut an RNA PDB into dinucleotide fragments and report the best-fitting "
            "conformer for each fragment."
        )
    )
    parser.add_argument("rna_pdb", type=_existing_file, help="RNA PDB file to fit.")
    parser.add_argument(
        "--exclude-pdb-code",
        "--exclude",
        dest="exclude_pdb_code",
        type=_pdb_code,
        help="PDB code whose origin conformers should be excluded/replaced.",
    )
    parser.add_argument(
        "--ignore-unknown",
        action="store_true",
        help="Skip non-canonical or unknown residues.",
    )
    parser.add_argument(
        "--ignore-missing",
        action="store_true",
        help="Skip nucleotides with atoms missing from the mononucleotide template.",
    )
    parser.add_argument(
        "--ignore-reordered",
        action="store_true",
        help="Skip nucleotides whose atom order differs from the template.",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Do not print the column header.",
    )
    return parser


def best_conformer(
    reference_coordinates: np.ndarray, library_coordinates: np.ndarray
) -> tuple[int, float]:
    _, rmsds = superimpose_array(library_coordinates, reference_coordinates)
    conformer = int(rmsds.argmin())
    return conformer, float(rmsds[conformer])


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    dinucleotide_libraries, _ = config(verify_checksums=False)
    ppdb = parse_pdb(args.rna_pdb.read_text())
    reference = Reference(
        ppdb,
        rna=True,
        ignore_unknown=args.ignore_unknown,
        ignore_missing=args.ignore_missing,
        ignore_reordered=args.ignore_reordered,
    )

    if not args.no_header:
        print("fragment\tsequence\tconformer\trmsd")

    libraries_by_sequence = {}
    for fragment_position in reference.get_fragment_positions(2):
        sequence = reference.get_sequence(fragment_position, 2)
        reference_coordinates = reference.get_coordinates(fragment_position, 2)
        if sequence not in libraries_by_sequence:
            libraries_by_sequence[sequence] = dinucleotide_libraries[sequence].create(
                pdb_code=args.exclude_pdb_code,
                prune_conformers=True,
            )
        library = libraries_by_sequence[sequence]
        conformer0, rmsd = best_conformer(reference_coordinates, library.coordinates)
        if library.conformer_mapping is not None:
            conformer0 = int(library.conformer_mapping[conformer0])
        print(f"{fragment_position}\t{sequence}\t{conformer0 + 1}\t{rmsd:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
