#!/usr/bin/env python3
"""Calculate fragment and approximate chain RMSDs for an ``alaric-chain`` output.

The per-fragment columns are ordinary dinucleotide RMSDs.  The leading column
approximates the RMSD of ``alaric-chain-coordinates`` output without writing
coordinates: each nucleotide shared by two fragments is assigned the arithmetic
mean of the two corresponding mononucleotide RMSDs.

``--pairwise`` instead materializes those averaged chain coordinates and writes
the all-versus-all, same-frame RMSD matrix.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_ALARIC_DIR = _ROOT / "alaric"
if str(_ALARIC_DIR) not in sys.path:
    sys.path.insert(0, str(_ALARIC_DIR))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from parse_pdb import parse_pdb  # noqa: E402
from poses import DEFAULT_POSE_CHUNK_SIZE, PoseReader  # noqa: E402
from rmsd import (  # noqa: E402
    RMSD_DECIMALS,
    _infer_fragment,
    _load_library,
    _load_reference_coordinates,
    _pdb_code,
    _positive_int,
    _rmsd_for_pose_chunk,
)

from alaric.middle.chain_coordinates import (  # noqa: E402
    ChainCoordinatesError,
    chain_coordinates as materialize_chain_coordinates,
    load_metadata,
    read_chain_table,
)

OUTPUT_FILE = "chain_rmsd.txt"
MAX_PAIRWISE_CHAINS = 50_000
PAIRWISE_BLOCK_SIZE = 512


class ChainRmsdError(RuntimeError):
    """Invalid chain RMSD input."""


@dataclass(frozen=True)
class Column:
    pool: str
    fragment: int
    pose_dir: Path
    nposes: int
    sequence: str


class _CoordinateView:
    """Library facade for ``_rmsd_for_pose_chunk`` over an atom subset."""

    def __init__(self, source, coordinates: np.ndarray) -> None:
        self._source = source
        self.coordinates = coordinates

    def get_rotamers(self, conformer: int) -> np.ndarray:
        return self._source.get_rotamers(conformer)


def _reference_sequence(reference_path: Path, fragment: int) -> str:
    reference = parse_pdb(reference_path.read_text())
    if len(reference) == 0:
        raise ChainRmsdError(f"Reference PDB contains no atoms: {reference_path}")
    models = np.unique(reference["model"])
    if len(models) > 1:
        reference = reference[reference["model"] == models[0]]
    return _infer_fragment(reference, fragment)[0]


def _nucleotide_masks(template: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    resids = np.unique(template["resid"])
    if len(resids) != 2:
        raise ChainRmsdError(
            f"dinucleotide template has {len(resids)} residues, expected 2"
        )
    return template["resid"] == resids[0], template["resid"] == resids[1]


def _rmsd_vectors(
    pose_dir: Path,
    reference_coordinates: np.ndarray,
    library,
    *,
    chunksize: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return full, first-nucleotide, and second-nucleotide RMSDs per pose."""
    coordinates = library.coordinates.astype(np.float32, copy=False)
    first_mask, second_mask = _nucleotide_masks(library.template)
    first_library = _CoordinateView(library, coordinates[:, first_mask])
    second_library = _CoordinateView(library, coordinates[:, second_mask])
    reader = PoseReader(pose_dir, rows_per_chunk=chunksize)
    nposes = reader.selected_poses
    full = np.empty(nposes, dtype=np.float32)
    first = np.empty(nposes, dtype=np.float32)
    second = np.empty(nposes, dtype=np.float32)
    offset = 0
    for chunk in reader.iter_chunks():
        stop = offset + len(chunk)
        full[offset:stop] = _rmsd_for_pose_chunk(
            chunk, reference_coordinates, coordinates=coordinates, library=library
        )
        first[offset:stop] = _rmsd_for_pose_chunk(
            chunk,
            reference_coordinates[first_mask],
            coordinates=first_library.coordinates,
            library=first_library,
        )
        second[offset:stop] = _rmsd_for_pose_chunk(
            chunk,
            reference_coordinates[second_mask],
            coordinates=second_library.coordinates,
            library=second_library,
        )
        offset = stop
    if offset != nposes:
        raise ChainRmsdError(f"{pose_dir}: expected {nposes} poses, read {offset}")
    return full, first, second


def _excluded_pdb_code(chain_dir: Path) -> str:
    data_dir = chain_dir.absolute().parent / "DATA"
    if not data_dir.is_dir():
        raise ChainRmsdError(f"required sibling DATA directory not found: {data_dir}")
    pdbcode = data_dir / "pdbcode.txt"
    if not pdbcode.is_file():
        raise ChainRmsdError(f"required excluded PDB code not found: {pdbcode}")
    code = pdbcode.read_text().strip()
    if not code:
        raise ChainRmsdError(f"excluded PDB code is empty: {pdbcode}")
    try:
        return _pdb_code(code)
    except argparse.ArgumentTypeError as exc:
        raise ChainRmsdError(f"{pdbcode}: {exc}") from None


def _data_inputs(chain_dir: Path) -> tuple[Path, str]:
    data_dir = chain_dir.absolute().parent / "DATA"
    if not data_dir.is_dir():
        raise ChainRmsdError(f"required sibling DATA directory not found: {data_dir}")
    reference = data_dir / "reference.pdb"
    if not reference.is_file():
        raise ChainRmsdError(f"required reference PDB not found: {reference}")
    return reference, _excluded_pdb_code(chain_dir)


def _columns(chain_dir: Path, metadata: dict) -> list[Column]:
    result = []
    for entry in metadata["columns"]:
        pool = str(entry["pool"])
        pose_dir = chain_dir / entry.get("pose_dir", pool)
        if not pose_dir.is_dir():
            raise ChainRmsdError(f"{pool}: pose dir not found at {pose_dir}")
        try:
            nposes = PoseReader.get_nposes(pose_dir)
        except (OSError, ValueError) as exc:
            raise ChainRmsdError(
                f"{pool}: cannot read poses at {pose_dir}: {exc}"
            ) from None
        if nposes == 0:
            raise ChainRmsdError(f"{pool}: pose dir contains no poses")
        result.append(
            Column(
                pool=pool,
                fragment=int(entry["fragment"]),
                pose_dir=pose_dir,
                nposes=nposes,
                sequence=(
                    str(entry["sequence"]).upper() if entry.get("sequence") else ""
                ),
            )
        )
    return result


def calculate_chain_rmsds(
    chain_dir: Path,
    *,
    chunksize: int = DEFAULT_POSE_CHUNK_SIZE,
    verify_checksums: bool = False,
) -> tuple[list[str], np.ndarray]:
    """Return output headers and the per-chain RMSD table.

    Each non-leading output column is the full dinucleotide RMSD for the pose
    selected by that chain.  The first column is the approximation described in
    this module's docstring.
    """
    chain_dir = Path(chain_dir)
    try:
        metadata = load_metadata(chain_dir)
        if int(metadata.get("nchains", 0)) <= 0:
            raise ChainRmsdError("chain table contains no chains")
        table, _start = read_chain_table(chain_dir, metadata, None)
    except ChainCoordinatesError as exc:
        raise ChainRmsdError(str(exc)) from None

    columns = _columns(chain_dir, metadata)
    if table.shape[1] != len(columns):
        raise ChainRmsdError("chain table column count does not match metadata")
    for j, column in enumerate(columns):
        ids = table[:, j]
        if int(ids.min()) < 1 or int(ids.max()) > column.nposes:
            raise ChainRmsdError(
                f"{column.pool}: chain pose index out of range 1...{column.nposes} "
                f"(got {int(ids.min())}...{int(ids.max())})"
            )

    reference_path, pdb_code = _data_inputs(chain_dir)
    full_values = np.empty((len(table), len(columns)), dtype=np.float32)
    nucleotide_values: list[np.ndarray] = []
    prior_second: np.ndarray | None = None

    for j, column in enumerate(columns):
        sequence = _reference_sequence(reference_path, column.fragment)
        if column.sequence and column.sequence != sequence:
            raise ChainRmsdError(
                f"{column.pool}: metadata sequence {column.sequence!r} does not "
                f"match reference sequence {sequence!r} for frag{column.fragment}"
            )
        library, factory = _load_library(
            sequence,
            verify_checksums=verify_checksums,
            excluded_pdb_codes={pdb_code},
        )
        try:
            _sequence, reference_coordinates = _load_reference_coordinates(
                reference_path, column.fragment, sequence, library.template
            )
            full, first, second = _rmsd_vectors(
                column.pose_dir, reference_coordinates, library, chunksize=chunksize
            )
        finally:
            factory.unload_rotaconformers()

        ids = table[:, j] - 1
        full_values[:, j] = full[ids]
        first_values = first[ids]
        second_values = second[ids]
        if prior_second is None:
            nucleotide_values.append(first_values)
        else:
            nucleotide_values.append((prior_second + first_values) / 2.0)
        prior_second = second_values
    assert prior_second is not None
    nucleotide_values.append(prior_second)
    chain_rmsd = np.sqrt(np.mean(np.square(np.stack(nucleotide_values)), axis=0))
    output = np.column_stack((chain_rmsd, full_values))
    return ["chain_rmsd"] + [column.pool for column in columns], output


def write_table(output_path: Path, headers: list[str], values: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        handle.write("\t".join(headers) + "\n")
        np.savetxt(handle, values, delimiter="\t", fmt=f"%.{RMSD_DECIMALS}f")


def write_pairwise_matrix(
    output_path: Path,
    coordinates: np.ndarray,
    *,
    block_size: int = PAIRWISE_BLOCK_SIZE,
) -> None:
    """Write a float32 matrix of same-frame RMSDs between averaged chains."""
    coordinates = np.asarray(coordinates, dtype=np.float32)
    if coordinates.ndim != 3 or coordinates.shape[2] != 3:
        raise ChainRmsdError("chain coordinates must have shape (nchains, natoms, 3)")
    nchains, natoms, _ = coordinates.shape
    if nchains > MAX_PAIRWISE_CHAINS:
        raise ChainRmsdError(
            f"--pairwise supports at most {MAX_PAIRWISE_CHAINS:,} chains, got {nchains:,}"
        )
    if natoms == 0:
        raise ChainRmsdError("chain coordinates contain no atoms")
    if block_size <= 0:
        raise ValueError("pairwise block size must be positive")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(nchains, nchains),
    )
    flat = np.ascontiguousarray(coordinates.reshape(nchains, natoms * 3))
    norms = np.einsum("ij,ij->i", flat, flat, dtype=np.float32)
    try:
        for row_start in range(0, nchains, block_size):
            row_stop = min(row_start + block_size, nchains)
            row_coords = flat[row_start:row_stop]
            for col_start in range(row_start, nchains, block_size):
                col_stop = min(col_start + block_size, nchains)
                cross = row_coords @ flat[col_start:col_stop].T
                squared = (
                    norms[row_start:row_stop, None]
                    + norms[None, col_start:col_stop]
                    - 2.0 * cross
                )
                values = np.sqrt(np.maximum(squared, 0.0) / natoms)
                values = np.round(values, RMSD_DECIMALS).astype(np.float32, copy=False)
                matrix[row_start:row_stop, col_start:col_stop] = values
                if col_start != row_start:
                    matrix[col_start:col_stop, row_start:row_stop] = values.T
        np.fill_diagonal(matrix, 0.0)
        matrix.flush()
    finally:
        del matrix


def write_pairwise_chain_rmsd(
    chain_dir: Path,
    output_path: Path,
    *,
    verify_checksums: bool = False,
) -> None:
    """Materialize averaged chains and write their pairwise RMSD matrix."""
    chain_dir = Path(chain_dir)
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".npy":
        raise ChainRmsdError("--pairwise output must have a .npy suffix")
    try:
        metadata = load_metadata(chain_dir)
    except ChainCoordinatesError as exc:
        raise ChainRmsdError(str(exc)) from None
    nchains = int(metadata.get("nchains", 0))
    if nchains <= 0:
        raise ChainRmsdError("chain table contains no chains")
    if nchains > MAX_PAIRWISE_CHAINS:
        raise ChainRmsdError(
            f"--pairwise supports at most {MAX_PAIRWISE_CHAINS:,} chains, got {nchains:,}"
        )

    excluded = _excluded_pdb_code(chain_dir)
    try:
        atoms, start = materialize_chain_coordinates(
            chain_dir,
            exclude=[excluded],
            verify_checksums=verify_checksums,
        )
    except ChainCoordinatesError as exc:
        raise ChainRmsdError(str(exc)) from None
    if start != 0 or len(atoms) != nchains:
        raise ChainRmsdError(
            f"expected {nchains} materialized chains, got {len(atoms)}"
        )
    coordinates = np.stack((atoms["x"], atoms["y"], atoms["z"]), axis=-1)
    write_pairwise_matrix(output_path, coordinates)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chain_rmsd",
        description="Calculate per-fragment RMSDs and an approximate RMSD per chain.",
    )
    parser.add_argument(
        "chain_dir", type=Path, help="Build-mode output directory of alaric-chain."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output table (default: CHAIN_DIR/chain_rmsd.txt).",
    )
    parser.add_argument(
        "--pairwise",
        type=Path,
        metavar="OUTPUT.npy",
        help=(
            "Write an all-versus-all float32 RMSD matrix for averaged chains "
            f"(maximum {MAX_PAIRWISE_CHAINS:,} chains), instead of the reference RMSD table."
        ),
    )
    parser.add_argument(
        "--chunksize",
        type=_positive_int,
        default=DEFAULT_POSE_CHUNK_SIZE,
        help=f"Poses to process per RMSD chunk (default: {DEFAULT_POSE_CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--verify-checksums",
        action="store_true",
        help="Enable fraglib checksum verification when loading the library config.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.pairwise is not None and args.output is not None:
        raise SystemExit("error: --pairwise cannot be combined with --output")
    if args.pairwise is not None:
        try:
            write_pairwise_chain_rmsd(
                args.chain_dir,
                args.pairwise,
                verify_checksums=args.verify_checksums,
            )
        except (ChainRmsdError, OSError, ValueError) as exc:
            raise SystemExit(f"error: {exc}") from None
        print(f"wrote pairwise RMSD matrix to {args.pairwise}")
        return 0

    output = args.output or args.chain_dir / OUTPUT_FILE
    try:
        headers, values = calculate_chain_rmsds(
            args.chain_dir,
            chunksize=args.chunksize,
            verify_checksums=args.verify_checksums,
        )
    except (ChainRmsdError, OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from None
    write_table(output, headers, values)
    print(f"wrote {len(values)} chains to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
