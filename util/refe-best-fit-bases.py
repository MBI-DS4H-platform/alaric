#!/usr/bin/env python3
"""Report per-nucleotide base overlap RMSDs for refe-best-fit poses."""

from __future__ import annotations

import argparse
import gc
from collections import defaultdict
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ALARIC_DIR = Path(__file__).with_name("alaric")
if str(ALARIC_DIR) not in sys.path:
    sys.path.insert(0, str(ALARIC_DIR))

from library import Library, LibraryFactory, config  # noqa: E402
from parse_pdb import parse_pdb  # noqa: E402
from reference import Reference  # noqa: E402


GRID_SPACING = sqrt(3) / 3


@dataclass
class Fragment:
    position: int
    sequence: str
    conformer: int  # 0-based index into the unpruned conformer array
    rotamer: int  # 0-based index into the conformer's rotamer array
    grid: np.ndarray  # integer translation grid offset [x, y, z]


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
            "Read a refe-best-fit.tsv (produced by refe-best-fit.py) and report "
            "the base overlap RMSD of each nucleotide in every fitted fragment. "
            "For example, fragment 3 reports nucleotides 3 and 4."
        )
    )
    parser.add_argument("best_fit_tsv", type=_existing_file, help="TSV file to read.")
    parser.add_argument(
        "rna_pdb",
        type=_existing_file,
        help="Reference RNA PDB used to produce the TSV.",
    )
    parser.add_argument(
        "--exclude-pdb-code",
        "--exclude",
        dest="exclude_pdb_code",
        type=_pdb_code,
        help=(
            "PDB code whose origin conformers were excluded/replaced when "
            "producing the TSV. Must match refe-best-fit.py."
        ),
    )
    parser.add_argument(
        "--grid-spacing",
        type=float,
        default=GRID_SPACING,
        help=(
            "Translation grid spacing in Angstrom used to produce the TSV "
            f"(default: {GRID_SPACING:.8f})."
        ),
    )
    parser.add_argument(
        "--ignore-unknown",
        action="store_true",
        help="Skip non-canonical or unknown residues in the reference RNA.",
    )
    parser.add_argument(
        "--ignore-missing",
        action="store_true",
        help="Skip nucleotides missing atoms from their mononucleotide template.",
    )
    parser.add_argument(
        "--ignore-reordered",
        action="store_true",
        help="Skip nucleotides whose atom order differs from the template.",
    )
    parser.add_argument(
        "--no-header", action="store_true", help="Do not print the column header."
    )
    return parser


def read_fragments(path: Path) -> list[Fragment]:
    table = np.genfromtxt(path, names=True, dtype=None, encoding=None)
    if table.size == 0:
        return []
    if table.ndim == 0:
        table = table.reshape(1)
    fragments = [
        Fragment(
            position=int(row["fragment"]),
            sequence=str(row["sequence"]),
            conformer=int(row["conformer"]) - 1,
            rotamer=int(row["rotamer"]) - 1,
            grid=np.array(
                [int(row["grid_x"]), int(row["grid_y"]), int(row["grid_z"])],
                dtype=np.int64,
            ),
        )
        for row in table
    ]
    fragments.sort(key=lambda fragment: fragment.position)
    return fragments


def _rotamer_to_matrix(rotamer: np.ndarray) -> np.ndarray:
    rotamer = np.asarray(rotamer)
    if rotamer.shape == (3, 3):
        return rotamer
    if rotamer.shape == (3,):
        return Rotation.from_rotvec(rotamer).as_matrix()
    raise ValueError(
        "Unsupported rotaconformer representation; expected shape [3] or [3,3]"
    )


def _rotamer_group_key(factory: LibraryFactory) -> tuple[str | None, str | None]:
    return (factory.rotaconformers_file, factory.rotaconformers_extension_file)


def _base_library(factory: LibraryFactory, first: bool, pdb_code: str | None) -> Library:
    return factory.create(
        pdb_code=pdb_code,
        nucleotide_mask=np.array([first, not first]),
        only_base=True,
        with_rotaconformers=True,
    )


def base_rmsds(
    fragments: list[Fragment],
    reference: Reference,
    dinucleotide_libraries: dict[str, LibraryFactory],
    *,
    exclude_pdb_code: str | None,
    grid_spacing: float,
) -> dict[int, tuple[float, float]]:
    """Return the unaligned base RMSD of each fitted nucleotide by fragment."""
    fragments_by_sequence: dict[str, list[Fragment]] = defaultdict(list)
    for fragment in fragments:
        fragments_by_sequence[fragment.sequence].append(fragment)

    sequences_by_rotamers: dict[
        tuple[str | None, str | None], list[str]
    ] = defaultdict(list)
    for sequence in fragments_by_sequence:
        if sequence not in dinucleotide_libraries:
            raise ValueError(f"Unsupported sequence in TSV: {sequence}")
        sequences_by_rotamers[_rotamer_group_key(dinucleotide_libraries[sequence])].append(
            sequence
        )

    result: dict[int, tuple[float, float]] = {}
    for sequences in sequences_by_rotamers.values():
        donor = dinucleotide_libraries[sequences[0]]
        donor.load_rotaconformers()
        try:
            for sequence in sequences:
                factory = dinucleotide_libraries[sequence]
                factory.rotaconformers = donor.rotaconformers
                factory.rotaconformers_index = donor.rotaconformers_index
                first_library = _base_library(factory, first=True, pdb_code=exclude_pdb_code)
                second_library = _base_library(factory, first=False, pdb_code=exclude_pdb_code)
                assert first_library.atom_mask is not None
                assert second_library.atom_mask is not None

                for fragment in fragments_by_sequence[sequence]:
                    if reference.get_sequence(fragment.position, 2) != sequence:
                        raise ValueError(
                            f"Fragment {fragment.position} sequence {sequence} does not "
                            "match the reference RNA"
                        )
                    rotation = _rotamer_to_matrix(
                        first_library.get_rotamers(fragment.conformer)[fragment.rotamer]
                    )
                    offset = fragment.grid * grid_spacing
                    reference_coordinates = reference.get_coordinates(fragment.position, 2)
                    rmsds = []
                    for library in (first_library, second_library):
                        fitted = library.coordinates[fragment.conformer].dot(rotation) + offset
                        target = reference_coordinates[library.atom_mask]
                        if fitted.shape != target.shape:
                            raise ValueError(
                                f"Base-coordinate shape mismatch for fragment {fragment.position}: "
                                f"{fitted.shape} versus {target.shape}"
                            )
                        difference = fitted - target
                        rmsds.append(float(np.sqrt((difference * difference).sum() / len(difference))))
                    result[fragment.position] = (rmsds[0], rmsds[1])
        finally:
            for sequence in sequences:
                factory = dinucleotide_libraries[sequence]
                factory.rotaconformers = None
                factory.rotaconformers_index = None
            donor.unload_rotaconformers()
            gc.collect()

    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fragments = read_fragments(args.best_fit_tsv)
    reference = Reference(
        parse_pdb(args.rna_pdb.read_text()),
        rna=True,
        ignore_unknown=args.ignore_unknown,
        ignore_missing=args.ignore_missing,
        ignore_reordered=args.ignore_reordered,
    )
    dinucleotide_libraries, _ = config(verify_checksums=False)
    rmsds = base_rmsds(
        fragments,
        reference,
        dinucleotide_libraries,
        exclude_pdb_code=args.exclude_pdb_code,
        grid_spacing=args.grid_spacing,
    )

    if not args.no_header:
        print("fragment\tsequence\tnucleotide1_ovRMSD\tnucleotide2_ovRMSD")
    for fragment in fragments:
        first, second = rmsds[fragment.position]
        print(f"{fragment.position}\t{fragment.sequence}\t{first:.6f}\t{second:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
