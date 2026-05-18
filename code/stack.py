from __future__ import annotations

import argparse
from dataclasses import dataclass
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
from math import sqrt
import sys
import time
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from library import config
from offsets import get_discrete_offsets
from parse_pdb import parse_pdb
from poses import PoseWriter, discover_organized
from rna_pdb import ppdb2nucseq
from stack_geometry import (
    RINGS,
    as_coordinates,
    calc_plane,
    protein_ring_coordinates,
    residue_ring_atom_mask,
)


GRID_SPACING = sqrt(3) / 3
LATERAL_SLOPE = 1.78966


@dataclass(frozen=True)
class _StackWorkConfig:
    coordinates: np.ndarray
    ring_mask: np.ndarray
    rotaconformers: np.ndarray
    rotaconformers_index: np.ndarray
    selected_rotamer_positions: np.ndarray | None
    protein_center: np.ndarray
    protein_plane: np.ndarray
    protein_x: np.ndarray
    protein_y: np.ndarray
    max_angle_deg: float
    margin: float
    dihedral_min_deg: float | None
    dihedral_max_deg: float | None
    output_dir: Path
    cache_size: int
    center_batch_size: int
    offset_chunk_size: int
    writer_memory_lock: object | None


@dataclass(frozen=True)
class _StackBatchResult:
    conformers: int
    candidates: int
    surviving: int
    poses: int
    files: int


_WORK_CONFIG: _StackWorkConfig | None = None
_PROGRESS_QUEUE = None
_THREAD_LIMITS = None
_PROGRESS_FILTER = "filter"
_PROGRESS_WRITER_START = "writer_start"
_PROGRESS_WRITER_DONE = "writer_done"


def _existing_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    if not p.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {path}")
    return path


def _pdb_code(code: str) -> str:
    code = code.strip()
    if len(code) != 4 or not code[0].isdigit() or not code[1:].isalnum():
        raise argparse.ArgumentTypeError(
            "PDB codes must be 4 chars: one digit + 3 alphanumeric characters (e.g. 1B7F)"
        )
    return code.upper()


def _dinucleotide_sequence(seq: str) -> str:
    seq = seq.strip().upper()
    if len(seq) != 2:
        raise argparse.ArgumentTypeError("--sequence must be a dinucleotide (length 2)")
    allowed = set("ACGU")
    if any(ch not in allowed for ch in seq):
        raise argparse.ArgumentTypeError("--sequence must contain only A/C/G/U")
    return seq


def _dihedral_pair(values: Sequence[str]) -> tuple[float, float]:
    if len(values) != 2:
        raise argparse.ArgumentTypeError(
            "--dihedral requires 2 floats: MIN MAX or MAX MIN"
        )
    try:
        a = float(values[0])
        b = float(values[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--dihedral values must be floats") from exc
    return (a, b)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stack",
        description="Create stacking-based initial poses from a protein and a dinucleotide library.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show a full traceback on errors.",
    )

    parser.add_argument(
        "--protein",
        required=True,
        type=_existing_file,
        help="PDB file of the protein.",
    )
    parser.add_argument(
        "--resid",
        required=True,
        type=int,
        help="Residue ID (numeric) of the protein aromatic residue that stacks.",
    )
    parser.add_argument(
        "--sequence",
        required=True,
        type=_dinucleotide_sequence,
        help="Dinucleotide sequence (e.g. AA, AC, GU).",
    )
    parser.add_argument(
        "--dihedral",
        nargs=2,
        metavar=("MIN_DEG", "MAX_DEG"),
        required=False,
        help="""Minimum and maximum dihedral angle in degrees.
If the maximum is smaller than the minimum, only the tails are kept, 
and the values in between are eliminated.""",
    )
    parser.add_argument(
        "--angle",
        type=float,
        required=True,
        help="Maximum stacking angle (in degrees).",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.5,
        help="Distance vector slack in angstroms (default: 0.5).",
    )
    nuc_group = parser.add_mutually_exclusive_group(required=True)
    nuc_group.add_argument(
        "--first",
        action="store_true",
        help="First nucleotide stacks (exactly one of --first/--second required).",
    )
    nuc_group.add_argument(
        "--second",
        action="store_true",
        help="Second nucleotide stacks (exactly one of --first/--second required).",
    )
    parser.add_argument(
        "--output",
        metavar="POSE_DIR",
        required=True,
        help="Output directory where unorganized .arc.zst pose shards are written.",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=50_000_000,
        metavar="N",
        help=(
            "Approximate weighted pose cache before flushing; unsorted poses count "
            "as 7 and sorted poses count as 1 (default: 50000000)."
        ),
    )
    parser.add_argument(
        "--nprocs",
        type=int,
        default=os.cpu_count() or 1,
        metavar="N",
        help="Number of worker processes used to evaluate conformers (default: CPU count).",
    )
    parser.add_argument(
        "--poselock",
        type=int,
        default=None,
        metavar="N",
        help=(
            "If set, limit concurrent PoseWriter sorting/packing sections to N "
            "worker processes (default: no limit)."
        ),
    )
    conformer_group = parser.add_mutually_exclusive_group()
    conformer_group.add_argument(
        "--conformer-range",
        nargs=2,
        type=int,
        default=None,
        metavar=("FIRST", "LAST"),
        help="Restrict the search to available conformers with 1-based indices FIRST..LAST inclusive.",
    )
    conformer_group.add_argument(
        "--conformer",
        type=int,
        default=None,
        metavar="INDEX",
        help="Restrict the search to a single 0-based conformer index.",
    )
    parser.add_argument(
        "--rotamer-range",
        nargs=2,
        type=int,
        default=None,
        metavar=("FIRST", "LAST"),
        help=(
            "Restrict every selected conformer to local rotamer positions FIRST..LAST inclusive. "
            "Indices are 1-based. "
            "LAST must fit within the smallest selected rotamer list."
        ),
    )
    parser.add_argument(
        "--pdb-exclude",
        nargs="+",
        default=[],
        type=_pdb_code,
        metavar="PDB",
        help="One or more PDB codes to exclude from the fragment library.",
    )
    return parser


def _calc_planes(coordinates: np.ndarray) -> np.ndarray:
    centered = coordinates - coordinates.mean(axis=1)[:, None, :]
    covar = np.einsum("ijk,ijl->ikl", centered, centered)
    _, _, wt = np.linalg.svd(covar)
    directionless = wt[:, 2, :] / np.linalg.norm(wt[:, 2, :], axis=-1)[:, None]
    directed = np.cross(
        centered[:, 1, :] - centered[:, 0, :],
        centered[:, 2, :] - centered[:, 1, :],
        axis=1,
    )
    flip = np.sign(np.einsum("ij,ij->i", directionless, directed))
    return directionless * flip[:, None]


def _select_library(
    sequence: str,
    first: bool,
    excluded_pdb_codes: set[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dinucleotide_libraries, dinucleotide_templates = config(verify_checksums=False)
    if sequence not in dinucleotide_libraries:
        raise ValueError(f"Unsupported sequence: {sequence}")

    factory = dinucleotide_libraries[sequence]
    template = dinucleotide_templates[sequence]
    nucleotide_mask = np.array([first, not first], dtype=bool)

    factory.load_rotaconformers()
    pdb_code_filter: str | list[str] | None = None
    if excluded_pdb_codes:
        pdb_code_filter = sorted(excluded_pdb_codes)
    library = factory.create(
        pdb_code=pdb_code_filter,
        nucleotide_mask=nucleotide_mask,
        only_base=True,
        with_rotaconformers=True,
    )
    if library.rotaconformers is None or library.rotaconformers_index is None:
        raise RuntimeError("Rotaconformers were not loaded")

    coordinates = library.coordinates
    conformer_mask = np.ones(len(coordinates), dtype=bool)
    if library.conformer_mask is not None:
        conformer_mask &= library.conformer_mask

    sequence_from_template, nucleotide_indices = ppdb2nucseq(
        template,
        rna=True,
        return_index=True,
    )
    if sequence_from_template != sequence:
        raise ValueError(
            f"Template sequence mismatch: expected {sequence}, got {sequence_from_template}"
        )

    nucleotide_index = 0 if first else 1
    atom_start, atom_end = nucleotide_indices[nucleotide_index]
    residue_ring_mask, _ = residue_ring_atom_mask(template[atom_start:atom_end])

    ring_mask_full = np.zeros(len(template), dtype=bool)
    ring_mask_full[atom_start:atom_end] = residue_ring_mask
    if library.atom_mask is not None:
        ring_mask = ring_mask_full[library.atom_mask]
    else:
        ring_mask = ring_mask_full
    if not np.any(ring_mask):
        raise ValueError(f"No nucleobase ring atoms selected for {sequence}")

    factory.unload_rotaconformers()
    result = (
        coordinates,
        conformer_mask,
        ring_mask,
        library.rotaconformers,
        library.rotaconformers_index,
    )
    del library
    return result


def _rotamers_to_matrices(rotamers: np.ndarray) -> np.ndarray:
    if rotamers.ndim == 3 and rotamers.shape[1:] == (3, 3):
        return rotamers.astype(np.float32, copy=False)
    if rotamers.ndim == 2 and rotamers.shape[1] == 3:
        return Rotation.from_rotvec(rotamers).as_matrix().astype(np.float32, copy=False)
    raise ValueError(
        "Unsupported rotamer representation; expected shape [N,3] rotvec or [N,3,3] matrix"
    )


def _select_conformers(
    conformer_mask: np.ndarray,
    index_range: Sequence[int] | None,
    specific_index: int | None,
) -> np.ndarray:
    available_indices = np.nonzero(conformer_mask)[0]
    if len(available_indices) == 0:
        raise ValueError("No conformers available after exclusions")
    if specific_index is not None:
        if specific_index < 0:
            raise ValueError("--conformer must be non-negative")
        if specific_index >= len(conformer_mask):
            raise ValueError(
                f"--conformer index {specific_index} out of range for library of size {len(conformer_mask)}"
            )
        if not conformer_mask[specific_index]:
            raise ValueError(
                f"--conformer index {specific_index} is not available after exclusions"
            )
        return np.array([specific_index], dtype=available_indices.dtype)
    if index_range is None:
        return available_indices.copy()

    first, last = index_range
    if first <= 0:
        raise ValueError("--conformer-range FIRST must be positive")
    if last < first:
        raise ValueError("--conformer-range LAST must be greater than or equal to FIRST")
    if last > len(conformer_mask):
        raise ValueError(
            f"--conformer-range LAST index {last} out of range for library of size {len(conformer_mask)}"
        )

    requested = np.arange(first - 1, last, dtype=available_indices.dtype)
    selected = requested[conformer_mask[requested]]
    if len(selected) == 0:
        raise ValueError("No conformers available in --conformer-range after exclusions")
    return selected


def _select_rotamer_positions(
    index_range: Sequence[int] | None,
    selected_conformers: np.ndarray,
    rotaconformers_index: np.ndarray,
) -> np.ndarray | None:
    if index_range is None:
        return None

    first, last = index_range
    if first <= 0:
        raise ValueError("--rotamer-range FIRST must be positive")
    if last < first:
        raise ValueError("--rotamer-range LAST must be greater than or equal to FIRST")

    rotamer_counts = np.diff(rotaconformers_index, prepend=0)
    smallest_count = int(rotamer_counts[selected_conformers].min())
    if last > smallest_count:
        raise ValueError(
            f"--rotamer-range LAST={last} is larger than the smallest selected rotamer list ({smallest_count})"
        )
    return np.arange(first - 1, last, dtype=np.int64)


def _dihedral_filter(
    angles: np.ndarray,
    dihedral: np.ndarray,
    max_angle_deg: float,
    dihedral_min_deg: float | None,
    dihedral_max_deg: float | None,
) -> np.ndarray:
    mask = angles <= max_angle_deg / 180.0 * np.pi
    if dihedral_min_deg is not None:
        assert dihedral_max_deg is not None
        dihedral_min = dihedral_min_deg / 180.0 * np.pi
        dihedral_max = dihedral_max_deg / 180.0 * np.pi
        if dihedral_max < dihedral_min:
            dihedral_mask = (dihedral <= dihedral_max) | (dihedral >= dihedral_min)
        else:
            dihedral_mask = (dihedral >= dihedral_min) & (dihedral <= dihedral_max)
        mask &= dihedral_mask
    return mask


def _build_reverse_table(reverse_map: dict) -> np.ndarray:
    reverse_table = np.empty((max(reverse_map.keys()) + 1, 3), dtype=np.int16)
    for p_index, xyz in reverse_map.items():
        reverse_table[p_index] = xyz
    return reverse_table


def _process_conformer(
    conformer_index: int,
    config: _StackWorkConfig,
    pose_writer: PoseWriter,
    reverse_table: np.ndarray | None,
) -> tuple[int, int, np.ndarray | None]:
    total_candidates = 0
    total_surviving = 0

    ring_coordinates = config.coordinates[conformer_index, config.ring_mask]
    if len(ring_coordinates) < 3:
        return total_candidates, total_surviving, reverse_table

    rotamer_end = int(config.rotaconformers_index[conformer_index])
    rotamer_start = (
        0
        if conformer_index == 0
        else int(config.rotaconformers_index[conformer_index - 1])
    )
    rotamer_count = rotamer_end - rotamer_start
    if rotamer_count <= 0:
        return total_candidates, total_surviving, reverse_table

    if config.selected_rotamer_positions is None:
        local_rotamer_indices = np.arange(rotamer_count, dtype=np.int64)
        rotamer_block = config.rotaconformers[rotamer_start:rotamer_end]
    else:
        local_rotamer_indices = config.selected_rotamer_positions.astype(
            np.int64, copy=False
        )
        rotamer_block = config.rotaconformers[rotamer_start + local_rotamer_indices]

    rotamer_matrices = _rotamers_to_matrices(rotamer_block)
    base_center = ring_coordinates.mean(axis=0)
    base_plane = calc_plane(ring_coordinates)
    base_x = ring_coordinates[0] - base_center
    ring_planes = base_plane @ rotamer_matrices

    cross_norm = np.linalg.norm(
        np.cross(config.protein_plane[None, :], ring_planes), axis=1
    )
    cross_norm = np.minimum(cross_norm, 1.0)
    angles = np.arcsin(cross_norm)

    ring_centers = base_center @ rotamer_matrices
    ring_x = base_x @ rotamer_matrices
    ring_x -= (
        np.einsum("ij,j->i", ring_x, config.protein_plane)[:, None]
        * config.protein_plane[None, :]
    )
    ring_x_norm = np.linalg.norm(ring_x, axis=1)
    degenerate = ring_x_norm < 0.0001
    safe_norm = np.where(degenerate, 1.0, ring_x_norm)
    normalized_ring_x = ring_x / safe_norm[:, None]
    dihedral_x = np.einsum("ij,j->i", normalized_ring_x, config.protein_x)
    dihedral_y = np.einsum("ij,j->i", normalized_ring_x, config.protein_y)
    dihedral = np.angle(dihedral_x + 1.0j * dihedral_y)
    dihedral[degenerate] = np.pi / 2

    angle_mask = _dihedral_filter(
        angles,
        dihedral,
        max_angle_deg=config.max_angle_deg,
        dihedral_min_deg=config.dihedral_min_deg,
        dihedral_max_deg=config.dihedral_max_deg,
    )
    if not np.any(angle_mask):
        return total_candidates, total_surviving, reverse_table

    if config.selected_rotamer_positions is None:
        passing_rotamers_raw = np.flatnonzero(angle_mask).astype(
            np.int64, copy=False
        )
    else:
        passing_rotamers_raw = local_rotamer_indices[angle_mask]
    if (
        len(passing_rotamers_raw)
        and passing_rotamers_raw.max() > np.iinfo(np.uint16).max
    ):
        raise ValueError(
            "Rotamer index exceeds uint16 range; use --rotamer-range to constrain per-conformer rotamers"
        )
    if conformer_index > np.iinfo(np.uint16).max:
        raise ValueError("Conformer index exceeds uint16 range")

    passing_rotamers = passing_rotamers_raw.astype(np.uint16, copy=False)
    passing_centers = ring_centers[angle_mask].astype(np.float32, copy=False)
    if len(passing_centers) == 0:
        return total_candidates, total_surviving, reverse_table

    for batch_start in range(0, len(passing_centers), config.center_batch_size):
        batch_end = min(batch_start + config.center_batch_size, len(passing_centers))
        centers_batch = passing_centers[batch_start:batch_end]
        rot_batch = passing_rotamers[batch_start:batch_end]

        displacement_world = config.protein_center[None, :] - centers_batch
        displacement_grid = displacement_world / GRID_SPACING
        radius_grid = np.full(
            (len(displacement_grid),),
            (5.0 + config.margin) / GRID_SPACING,
            dtype=np.float32,
        )
        disp_indices, p_indices, rounded, reverse_map = get_discrete_offsets(
            displacement_grid,
            radius_grid,
        )
        if len(disp_indices) == 0:
            continue

        total_candidates += len(disp_indices)

        if reverse_table is None:
            reverse_table = _build_reverse_table(reverse_map)

        rounded16 = rounded.astype(np.int16, copy=False)
        for offset_start in range(0, len(disp_indices), config.offset_chunk_size):
            offset_end = min(
                offset_start + config.offset_chunk_size,
                len(disp_indices),
            )
            disp_chunk = disp_indices[offset_start:offset_end]
            p_chunk = p_indices[offset_start:offset_end]

            offset_grid = rounded16[disp_chunk] + reverse_table[p_chunk]
            center_vec = (
                offset_grid.astype(np.float32) * GRID_SPACING
                - displacement_world[disp_chunk]
            )

            center_z = np.abs(center_vec[:, 2])
            center_dis = np.linalg.norm(center_vec, axis=1)
            axial = np.abs(center_vec.dot(config.protein_plane))
            axial = np.maximum(0.0, np.minimum(center_dis - 0.001, axial))
            lateral = np.sqrt(
                np.maximum(0.0, center_dis * center_dis - axial * axial)
            )
            center_dis_corrected = center_dis - lateral / LATERAL_SLOPE

            mask_a = (center_z >= 2.3 - config.margin) & (
                center_z <= 4.5 + config.margin
            )
            mask_b = (center_dis_corrected >= 2.3 - config.margin) & (
                center_dis_corrected <= 3.8 + config.margin
            )
            mask_c = center_dis <= 5.0 + config.margin
            keep_mask = mask_a & mask_b & mask_c
            if not np.any(keep_mask):
                continue

            kept_disp = disp_chunk[keep_mask]
            kept_p = p_chunk[keep_mask]
            total_surviving += len(kept_disp)

            translations = rounded16[kept_disp] + reverse_table[kept_p]
            conformers_chunk = np.full(
                len(kept_disp),
                conformer_index,
                dtype=np.uint16,
            )
            rotamers_chunk = rot_batch[kept_disp]
            pose_writer.add_chunk(
                conformers_chunk,
                rotamers_chunk,
                translations,
            )

    return total_candidates, total_surviving, reverse_table


def _process_conformer_batch(
    conformer_indices: Sequence[int],
    config: _StackWorkConfig | None = None,
    progress_callback=None,
    writer_start_callback=None,
    writer_done_callback=None,
) -> _StackBatchResult:
    if config is None:
        if _WORK_CONFIG is None:
            raise RuntimeError("Worker configuration has not been initialized")
        config = _WORK_CONFIG

    pose_writer = PoseWriter(
        config.output_dir,
        cache_poses=config.cache_size,
        memory_lock=config.writer_memory_lock,
    )
    total_candidates = 0
    total_surviving = 0
    reverse_table: np.ndarray | None = None
    pending_conformers = 0
    pending_candidates = 0
    last_progress = time.monotonic()

    def emit_progress() -> None:
        nonlocal pending_conformers, pending_candidates
        if not pending_conformers and not pending_candidates:
            return
        if progress_callback is not None:
            progress_callback(pending_conformers, pending_candidates)
        elif _PROGRESS_QUEUE is not None:
            _PROGRESS_QUEUE.put(
                (_PROGRESS_FILTER, pending_conformers, pending_candidates)
            )
        pending_conformers = 0
        pending_candidates = 0

    try:
        for conformer_index in conformer_indices:
            candidates, surviving, reverse_table = _process_conformer(
                int(conformer_index),
                config,
                pose_writer,
                reverse_table,
            )
            total_candidates += candidates
            total_surviving += surviving
            pending_conformers += 1
            pending_candidates += candidates
            now = time.monotonic()
            if now - last_progress >= 5.0:
                emit_progress()
                last_progress = now
        emit_progress()
        if writer_start_callback is not None:
            writer_start_callback()
        elif _PROGRESS_QUEUE is not None:
            _PROGRESS_QUEUE.put((_PROGRESS_WRITER_START, 1, 0))
        written = pose_writer.finish()
        if writer_done_callback is not None:
            writer_done_callback()
        elif _PROGRESS_QUEUE is not None:
            _PROGRESS_QUEUE.put((_PROGRESS_WRITER_DONE, 1, 0))
        return _StackBatchResult(
            conformers=len(conformer_indices),
            candidates=total_candidates,
            surviving=total_surviving,
            poses=pose_writer.total_poses,
            files=len(written),
        )
    finally:
        pose_writer.cleanup()


def _init_stack_worker(
    config: _StackWorkConfig,
    progress_queue,
    limit_native_threads: bool,
) -> None:
    global _WORK_CONFIG, _PROGRESS_QUEUE, _THREAD_LIMITS
    _WORK_CONFIG = config
    _PROGRESS_QUEUE = progress_queue
    if limit_native_threads:
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:
            return
        _THREAD_LIMITS = threadpool_limits(limits=1)


def _drain_progress_queue(
    progress_queue,
    conformer_pbar,
    distance_pbar,
    writer_pbar,
    writer_started: list[bool],
) -> None:
    while True:
        try:
            kind, value_a, value_b = progress_queue.get_nowait()
        except Empty:
            return
        if kind == _PROGRESS_FILTER:
            conformer_pbar.update(value_a)
            distance_pbar.update(value_b)
        elif kind == _PROGRESS_WRITER_START:
            if not writer_started[0]:
                writer_pbar.reset()
                writer_started[0] = True
        elif kind == _PROGRESS_WRITER_DONE:
            writer_pbar.update(value_a)
        else:
            raise RuntimeError(f"Unknown worker progress message: {kind!r}")


def _chunk_conformers(
    conformers: np.ndarray,
    nprocs: int,
) -> list[list[int]]:
    if len(conformers) == 0:
        return []
    return [
        [int(x) for x in conformers[worker_index::nprocs]]
        for worker_index in range(min(nprocs, len(conformers)))
    ]


def _run(args: argparse.Namespace) -> int:
    dihedral_min_deg, dihedral_max_deg = None, None
    if args.dihedral is not None:
        dihedral_min_deg, dihedral_max_deg = _dihedral_pair(args.dihedral)
    if args.margin < 0:
        raise ValueError("--margin must be non-negative")
    if args.cache_size <= 0:
        raise ValueError("--cache-size must be positive")
    if args.nprocs <= 0:
        raise ValueError("--nprocs must be positive")
    if args.poselock is not None and args.poselock <= 0:
        raise ValueError("--poselock must be positive when set")
    if args.poselock is not None and args.poselock > args.nprocs:
        raise ValueError("--poselock must be smaller than or equal to --nprocs")

    output_dir = Path(args.output)
    if discover_organized(output_dir):
        raise ValueError(
            f"--output directory already contains organized poses-*.arc: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading protein structure...", file=sys.stderr)
    protein_atoms = parse_pdb(Path(args.protein).read_text())
    if np.any(protein_atoms["model"] == 1):
        protein_atoms = protein_atoms[protein_atoms["model"] == 1]

    protein_ring = protein_ring_coordinates(protein_atoms, args.resid)
    protein_center = protein_ring.mean(axis=0)
    protein_plane = calc_plane(protein_ring)
    protein_x = protein_ring[0] - protein_center
    protein_x /= np.linalg.norm(protein_x)
    protein_y = np.cross(protein_plane, protein_x)
    protein_y /= np.linalg.norm(protein_y)

    print("[2/7] Loading fragment library...", file=sys.stderr)
    coordinates, conformer_mask, ring_mask, rotaconformers, rotaconformers_index = (
        _select_library(
            sequence=args.sequence,
            first=args.first,
            excluded_pdb_codes=set(args.pdb_exclude),
        )
    )

    selected_conformers = _select_conformers(
        conformer_mask,
        args.conformer_range,
        args.conformer,
    )
    print(
        f"[3/7] Filtering angle/dihedral for {len(selected_conformers)} conformers...",
        file=sys.stderr,
    )

    selected_rotamer_positions = _select_rotamer_positions(
        args.rotamer_range,
        selected_conformers,
        rotaconformers_index,
    )

    print("[4/7] Generating offset candidates...", file=sys.stderr)
    print("[5/7] Applying distance-vector filter...", file=sys.stderr)
    center_batch_size = 512 if args.rotamer_range is not None else 256
    offset_chunk_size = 2_000_000
    total_candidates = 0
    total_surviving = 0
    total_poses = 0
    total_files = 0
    nprocs = min(args.nprocs, len(selected_conformers))
    pool_context = None
    writer_memory_lock = None
    if nprocs > 1:
        ctx_name = "fork" if "fork" in mp.get_all_start_methods() else None
        pool_context = (
            mp.get_context(ctx_name)
            if ctx_name is not None
            else mp.get_context()
        )
        if args.poselock is not None:
            writer_memory_lock = pool_context.Semaphore(args.poselock)

    work_config = _StackWorkConfig(
        coordinates=coordinates,
        ring_mask=ring_mask,
        rotaconformers=rotaconformers,
        rotaconformers_index=rotaconformers_index,
        selected_rotamer_positions=selected_rotamer_positions,
        protein_center=protein_center,
        protein_plane=protein_plane,
        protein_x=protein_x,
        protein_y=protein_y,
        max_angle_deg=args.angle,
        margin=args.margin,
        dihedral_min_deg=dihedral_min_deg,
        dihedral_max_deg=dihedral_max_deg,
        output_dir=output_dir,
        cache_size=args.cache_size,
        center_batch_size=center_batch_size,
        offset_chunk_size=offset_chunk_size,
        writer_memory_lock=writer_memory_lock,
    )

    conformer_pbar = tqdm(
        total=len(selected_conformers),
        desc="Conformer-angle filter",
        unit="conf",
        mininterval=2.0,
    )
    distance_pbar = tqdm(desc="Offset-distance filter", unit="cand", mininterval=2.0)
    writer_pbar = None
    writer_started = [False]
    try:
        batches = _chunk_conformers(selected_conformers, max(1, nprocs))
        writer_pbar = tqdm(
            total=len(batches),
            desc="Pose-writer flush",
            unit="worker",
            mininterval=2.0,
        )
        if nprocs == 1:
            def update_progress(conformers: int, candidates: int) -> None:
                conformer_pbar.update(conformers)
                distance_pbar.update(candidates)

            def update_writer_start() -> None:
                assert writer_pbar is not None
                if not writer_started[0]:
                    writer_pbar.reset()
                    writer_started[0] = True

            def update_writer_done() -> None:
                assert writer_pbar is not None
                writer_pbar.update(1)

            for batch in batches:
                result = _process_conformer_batch(
                    batch,
                    work_config,
                    update_progress,
                    update_writer_start,
                    update_writer_done,
                )
                total_candidates += result.candidates
                total_surviving += result.surviving
                total_poses += result.poses
                total_files += result.files
        else:
            assert pool_context is not None
            ctx = pool_context
            progress_queue = ctx.Queue()
            with ctx.Pool(
                processes=nprocs,
                initializer=_init_stack_worker,
                initargs=(work_config, progress_queue, True),
            ) as pool:
                async_results = [
                    pool.apply_async(_process_conformer_batch, (batch,))
                    for batch in batches
                ]
                pending = list(async_results)
                while pending:
                    for async_result in pending[:]:
                        if not async_result.ready():
                            continue
                        result = async_result.get()
                        total_candidates += result.candidates
                        total_surviving += result.surviving
                        total_poses += result.poses
                        total_files += result.files
                        pending.remove(async_result)
                    _drain_progress_queue(
                        progress_queue,
                        conformer_pbar,
                        distance_pbar,
                        writer_pbar,
                        writer_started,
                    )
                    if pending:
                        try:
                            kind, value_a, value_b = progress_queue.get(timeout=0.2)
                        except Empty:
                            continue
                        if kind == _PROGRESS_FILTER:
                            conformer_pbar.update(value_a)
                            distance_pbar.update(value_b)
                        elif kind == _PROGRESS_WRITER_START:
                            if not writer_started[0]:
                                writer_pbar.reset()
                                writer_started[0] = True
                        elif kind == _PROGRESS_WRITER_DONE:
                            writer_pbar.update(value_a)
                        else:
                            raise RuntimeError(
                                f"Unknown worker progress message: {kind!r}"
                            )
                _drain_progress_queue(
                    progress_queue,
                    conformer_pbar,
                    distance_pbar,
                    writer_pbar,
                    writer_started,
                )
        if conformer_pbar.n < len(selected_conformers):
            conformer_pbar.update(len(selected_conformers) - conformer_pbar.n)
        if distance_pbar.n < total_candidates:
            distance_pbar.update(total_candidates - distance_pbar.n)
        if writer_pbar.n < len(batches):
            writer_pbar.update(len(batches) - writer_pbar.n)
        conformer_pbar.close()
        distance_pbar.close()
        writer_pbar.close()
        del rotaconformers, rotaconformers_index

        print(
            f"[6/7] Candidate offset assignments: {total_candidates}",
            file=sys.stderr,
        )
        print(
            f"[6/7] Surviving offset assignments: {total_surviving}",
            file=sys.stderr,
        )
        print("[7/7] Finished packing and writing poses.", file=sys.stderr)
        if total_files > 1:
            print(
                f"[7/7] Output split into {total_files} unorganized .arc.zst shards.",
                file=sys.stderr,
            )

        print(f"{total_poses} total solutions", file=sys.stderr)
        return 0
    finally:
        if hasattr(conformer_pbar, "disable") and not conformer_pbar.disable:
            conformer_pbar.close()
        if hasattr(distance_pbar, "disable") and not distance_pbar.disable:
            distance_pbar.close()
        if (
            writer_pbar is not None
            and hasattr(writer_pbar, "disable")
            and not writer_pbar.disable
        ):
            writer_pbar.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return int(_run(args))
    except SystemExit:
        raise
    except Exception as exc:
        if getattr(args, "debug", False):
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
