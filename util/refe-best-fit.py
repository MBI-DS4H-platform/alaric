#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
from collections import defaultdict
from math import sqrt
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
CODE_DIR = HERE / ".alaric"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from library import Library, LibraryFactory, config  # noqa: E402
from parse_pdb import parse_pdb  # noqa: E402
from reference import Reference  # noqa: E402
from superimpose import superimpose_array  # noqa: E402
from write_pdb import write_pdb  # noqa: E402


GRID_SPACING = sqrt(3) / 3
ROTAMER_PRECISION = 0.5


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
            "Cut an RNA PDB into dinucleotide fragments and report the best "
            "discrete pose using the best-fitting conformer."
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
        "--grid-spacing",
        type=float,
        default=GRID_SPACING,
        help=f"Translation grid spacing in Angstrom (default: {GRID_SPACING:.8f}).",
    )
    parser.add_argument(
        "--rotamer-precision",
        type=float,
        default=ROTAMER_PRECISION,
        help=(
            "Initial conformer search margin before keeping the best conformer "
            f"(default: {ROTAMER_PRECISION:g})."
        ),
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
    parser.add_argument(
        "--pdb-output",
        metavar="DIR",
        type=Path,
        help="Write the best fitted pose for each fragment as PDB files in DIR.",
    )
    return parser


def _rotamers_to_matrices(rotamers: np.ndarray) -> np.ndarray:
    if rotamers.ndim == 3 and rotamers.shape[1:] == (3, 3):
        return rotamers
    if rotamers.ndim == 2 and rotamers.shape[1] == 3:
        return Rotation.from_rotvec(rotamers).as_matrix()
    raise ValueError(
        "Unsupported rotaconformer representation; expected shape [N,3] or [N,3,3]"
    )


def _rotamer_group_key(factory: LibraryFactory) -> tuple[str | None, str | None]:
    return (factory.rotaconformers_file, factory.rotaconformers_extension_file)


def best_conformer(
    reference_coordinates: np.ndarray,
    library: Library,
    *,
    rotamer_precision: float,
    grid_spacing: float,
) -> int:
    _, rmsds = superimpose_array(library.coordinates, reference_coordinates)
    threshold = float(rmsds.min()) + rotamer_precision + grid_spacing
    candidates = np.flatnonzero(rmsds < threshold)
    if not len(candidates):
        raise ValueError("No conformer candidates found")
    best_local = int(candidates[np.argmin(rmsds[candidates])])
    if library.conformer_mapping is not None:
        return int(library.conformer_mapping[best_local])
    return best_local


def best_pose(
    reference_coordinates: np.ndarray,
    library: Library,
    conformer: int,
    *,
    grid_spacing: float,
) -> tuple[int, np.ndarray, np.ndarray, float]:
    conformer_coordinates = library.coordinates[conformer]
    rotamers = _rotamers_to_matrices(library.get_rotamers(conformer))
    superpositions = np.einsum("kj,ijl->ikl", conformer_coordinates, rotamers)
    offsets = reference_coordinates.mean(axis=0) - superpositions.mean(axis=1)
    offset_grid = np.rint(offsets / grid_spacing).astype(np.int32)
    offsets_disc = offset_grid * grid_spacing
    dif = superpositions + offsets_disc[:, None] - reference_coordinates
    rmsds = np.sqrt(np.einsum("ijk,ijk->i", dif, dif) / len(reference_coordinates))
    rotamer = int(rmsds.argmin())
    best_coordinates = superpositions[rotamer] + offsets_disc[rotamer]
    return rotamer, offset_grid[rotamer], best_coordinates, float(rmsds[rotamer])


def write_pose_pdb(
    output_dir: Path,
    fragment_position: int,
    sequence: str,
    template: np.ndarray,
    coordinates: np.ndarray,
) -> None:
    pdb_template = template.copy()
    pdb_template["x"] = coordinates[:, 0]
    pdb_template["y"] = coordinates[:, 1]
    pdb_template["z"] = coordinates[:, 2]
    path = output_dir / f"frag-{fragment_position}-{sequence}-bestfit.pdb"
    path.write_text(write_pdb(pdb_template))


def collect_fragments(reference: Reference) -> dict[str, list[int]]:
    fragments = defaultdict(list)
    for fragment_position in reference.get_fragment_positions(2):
        fragments[reference.get_sequence(fragment_position, 2)].append(fragment_position)
    return dict(fragments)


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
    fragments = collect_fragments(reference)
    if args.pdb_output is not None:
        args.pdb_output.mkdir(parents=True, exist_ok=True)

    if not args.no_header:
        print("fragment\tsequence\tconformer\trotamer\tgrid_x\tgrid_y\tgrid_z\trmsd")

    results = {}
    sequences_by_rotamers = defaultdict(list)
    for sequence in fragments:
        key = _rotamer_group_key(dinucleotide_libraries[sequence])
        sequences_by_rotamers[key].append(sequence)

    for sequences in sequences_by_rotamers.values():
        donor = dinucleotide_libraries[sequences[0]]
        donor.load_rotaconformers()
        try:
            libraries_by_sequence = {}
            for sequence in sequences:
                factory = dinucleotide_libraries[sequence]
                factory.rotaconformers = donor.rotaconformers
                factory.rotaconformers_index = donor.rotaconformers_index
                libraries_by_sequence[sequence] = (
                    factory.create(
                        pdb_code=args.exclude_pdb_code,
                        prune_conformers=True,
                    ),
                    factory.create(
                        pdb_code=args.exclude_pdb_code,
                        prune_conformers=False,
                        with_rotaconformers=True,
                    ),
                )

            for sequence in sequences:
                conformer_library, pose_library = libraries_by_sequence[sequence]
                for fragment_position in fragments[sequence]:
                    reference_coordinates = reference.get_coordinates(fragment_position, 2)
                    conformer = best_conformer(
                        reference_coordinates,
                        conformer_library,
                        rotamer_precision=args.rotamer_precision,
                        grid_spacing=args.grid_spacing,
                    )
                    rotamer, offset_grid, pose_coordinates, rmsd = best_pose(
                        reference_coordinates,
                        pose_library,
                        conformer,
                        grid_spacing=args.grid_spacing,
                    )
                    if args.pdb_output is not None:
                        write_pose_pdb(
                            args.pdb_output,
                            fragment_position,
                            sequence,
                            pose_library.template,
                            pose_coordinates,
                        )
                    results[fragment_position] = (
                        sequence,
                        conformer + 1,
                        rotamer + 1,
                        offset_grid,
                        rmsd,
                    )
        finally:
            for sequence in sequences:
                factory = dinucleotide_libraries[sequence]
                factory.rotaconformers = None
                factory.rotaconformers_index = None
            donor.unload_rotaconformers()
            gc.collect()

    for fragment_position in reference.get_fragment_positions(2):
        sequence, conformer, rotamer, offset_grid, rmsd = results[fragment_position]
        grid_x, grid_y, grid_z = offset_grid
        print(
            f"{fragment_position}\t{sequence}\t{conformer}\t{rotamer}\t"
            f"{grid_x}\t{grid_y}\t{grid_z}\t{rmsd:.6f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
