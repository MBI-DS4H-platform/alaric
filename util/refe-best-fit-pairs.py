#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
from collections import defaultdict
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
import sys

import numpy as np

ALARIC_DIR = Path(__file__).with_name("alaric")
if str(ALARIC_DIR) not in sys.path:
    sys.path.insert(0, str(ALARIC_DIR))

from library import LibraryFactory, config  # noqa: E402
from superimpose import superimpose  # noqa: E402


GRID_SPACING = sqrt(3) / 3


@dataclass
class Fragment:
    position: int
    sequence: str
    conformer: int  # 0-based index into the (non-pruned) conformer array
    rotamer: int  # 0-based index into the conformer's rotamers
    grid: np.ndarray  # integer translation grid offset [x, y, z]


@dataclass
class Pose:
    sequence: str
    rotation: np.ndarray  # 3x3 rotation matrix for the chosen rotamer
    offset: np.ndarray  # discrete translation (grid * grid_spacing)
    coor_down: np.ndarray  # coordinates of the last (downstream) nucleotide
    coor_up: np.ndarray  # coordinates of the first (upstream) nucleotide


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
            "the overlap RMSD (ovRMSD) and the conformer RMSD (cRMSD) of the "
            "shared nucleotide for each consecutive fragment pair."
        )
    )
    parser.add_argument(
        "best_fit_tsv",
        type=_existing_file,
        help="refe-best-fit.tsv file to read.",
    )
    parser.add_argument(
        "--exclude-pdb-code",
        "--exclude",
        dest="exclude_pdb_code",
        type=_pdb_code,
        help=(
            "PDB code whose origin conformers were excluded/replaced when "
            "producing the TSV. Must match refe-best-fit.py for the "
            "coordinates to be reconstructed identically."
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
        "--no-header",
        action="store_true",
        help="Do not print the column header.",
    )
    return parser


def read_fragments(path: Path) -> list[Fragment]:
    table = np.genfromtxt(path, names=True, dtype=None, encoding=None)
    if table.ndim == 0:
        table = table.reshape(1)
    fragments = []
    for row in table:
        fragments.append(
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
        )
    fragments.sort(key=lambda frag: frag.position)
    return fragments


def _rotamer_to_matrix(rotamer: np.ndarray) -> np.ndarray:
    """Convert a single rotaconformer to a 3x3 rotation matrix.

    Rotaconformers are stored either as scaled axis-angle 3-vectors or already
    as 3x3 matrices (mirrors refe-best-fit.py).
    """
    rotamer = np.asarray(rotamer, dtype=float)
    if rotamer.shape == (3, 3):
        return rotamer
    if rotamer.shape != (3,):
        raise ValueError(
            "Unsupported rotaconformer representation; expected shape [3] or [3,3]"
        )
    # Rodrigues' rotation formula (equivalent to scipy Rotation.from_rotvec).
    angle = float(np.linalg.norm(rotamer))
    if angle == 0.0:
        return np.eye(3)
    axis = rotamer / angle
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return (
        np.eye(3)
        + np.sin(angle) * cross
        + (1.0 - np.cos(angle)) * (cross @ cross)
    )


def _rotamer_group_key(factory: LibraryFactory) -> tuple[str | None, str | None]:
    return (factory.rotaconformers_file, factory.rotaconformers_extension_file)


def build_poses(
    fragments: list[Fragment],
    dinucleotide_libraries: dict[str, LibraryFactory],
    *,
    exclude_pdb_code: str | None,
    grid_spacing: float,
) -> dict[int, Pose]:
    """Reconstruct the fitted pose of the shared nucleotide for each fragment.

    Fragments are processed per group of sequences that share the same
    rotaconformer files, so the rotamers only need to be loaded once per group.
    """
    fragments_by_sequence: dict[str, list[Fragment]] = defaultdict(list)
    for fragment in fragments:
        fragments_by_sequence[fragment.sequence].append(fragment)

    sequences_by_rotamers: dict[
        tuple[str | None, str | None], list[str]
    ] = defaultdict(list)
    for sequence in fragments_by_sequence:
        key = _rotamer_group_key(dinucleotide_libraries[sequence])
        sequences_by_rotamers[key].append(sequence)

    poses: dict[int, Pose] = {}
    for sequences in sequences_by_rotamers.values():
        donor = dinucleotide_libraries[sequences[0]]
        donor.load_rotaconformers()
        try:
            for sequence in sequences:
                factory = dinucleotide_libraries[sequence]
                factory.rotaconformers = donor.rotaconformers
                factory.rotaconformers_index = donor.rotaconformers_index
                lib_down = factory.create(
                    pdb_code=exclude_pdb_code,
                    nucleotide_mask=np.array([False, True]),
                    with_rotaconformers=True,
                )
                lib_up = factory.create(
                    pdb_code=exclude_pdb_code,
                    nucleotide_mask=np.array([True, False]),
                    with_rotaconformers=True,
                )
                for fragment in fragments_by_sequence[sequence]:
                    rotation = lib_down.get_rotamers(fragment.conformer)[
                        fragment.rotamer
                    ]
                    poses[fragment.position] = Pose(
                        sequence=sequence,
                        rotation=_rotamer_to_matrix(rotation),
                        offset=fragment.grid * grid_spacing,
                        coor_down=lib_down.coordinates[fragment.conformer].copy(),
                        coor_up=lib_up.coordinates[fragment.conformer].copy(),
                    )
        finally:
            for sequence in sequences:
                factory = dinucleotide_libraries[sequence]
                factory.rotaconformers = None
                factory.rotaconformers_index = None
            donor.unload_rotaconformers()
            gc.collect()

    return poses


def pair_rmsds(down: Pose, up: Pose) -> tuple[float, float]:
    """Compute (ovRMSD, cRMSD) for the nucleotide shared by two fragments.

    The downstream nucleotide of the first fragment overlaps the upstream
    nucleotide of the second fragment.
    """
    coor_conf_down = down.coor_down
    coor_conf_up = up.coor_up

    _, crmsd = superimpose(coor_conf_down, coor_conf_up)

    coor_down = coor_conf_down.dot(down.rotation) + down.offset
    coor_up = coor_conf_up.dot(up.rotation) + up.offset
    dif = coor_down - coor_up
    ovrmsd = float(np.sqrt((dif * dif).sum() / len(dif)))
    return ovrmsd, float(crmsd)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    fragments = read_fragments(args.best_fit_tsv)
    dinucleotide_libraries, _ = config(verify_checksums=False)
    poses = build_poses(
        fragments,
        dinucleotide_libraries,
        exclude_pdb_code=args.exclude_pdb_code,
        grid_spacing=args.grid_spacing,
    )

    if not args.no_header:
        print("down\tup\tsequence\tovRMSD\tcRMSD")

    for first, second in zip(fragments, fragments[1:]):
        if second.position != first.position + 1:
            continue
        seq1 = first.sequence
        seq2 = second.sequence
        if seq1[-1] != seq2[0]:
            raise ValueError(
                f"Fragments {first.position} ({seq1}) and {second.position} "
                f"({seq2}) do not overlap in their shared nucleotide"
            )
        down = poses[first.position]
        up = poses[second.position]
        ovrmsd, crmsd = pair_rmsds(down, up)
        triseq = seq1 + seq2[1:]
        print(
            f"{first.position}\t{second.position}\t{triseq}\t"
            f"{ovrmsd:.6f}\t{crmsd:.6f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
