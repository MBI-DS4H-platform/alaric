from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import sys
import time
from typing import Sequence

import numpy as np

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from anchor_constants import MIN_CELL_FRACTION as _DEFAULT_MIN_CELL_FRACTION
from anchor_precalc import (
    GRID_SPACING,
    HOPF_CELLS,
    HOPF_N,
    HOPF_NPSI,
    _PrecalculatedAnchor,
    _build_capacity_table,
    _build_cell_lookup,
    _deduplicate_owner_grid,
    _find_precalculated_array,
    _hopf_cell_axes,
    _hopf_cell_indices,
    _hopf_cell_matrices,
    _lattice_geometry,
    _load_precalculated_anchor,
    _load_reference_coordinates,
    _pose_cell_capacity,
    _resolve_precalculated_directory,
    _rotamers_to_matrices,
    _snap_cache_rotations,
    _superimpose_rotation,
    _translation_segments,
    _validate_precalculated_arrays,
    candidate_cells,
    cell_fill,
)
from tqdm import tqdm

from library import config
from nprocs import default_nprocs
from parse_pdb import parse_pdb
from poses import PoseWriter, discover_organized
from rna_pdb import ppdb2nucseq
from stack_geometry import (
    as_coordinates,
    residue_ring_coordinates,
    residue_ring_atom_mask,
)


def _action_min_cell_fraction() -> float:
    value = os.environ.get("ALARIC_ANCHOR_MIN_CELL_FRACTION")
    if value is None:
        return _DEFAULT_MIN_CELL_FRACTION
    try:
        fraction = float(value)
    except ValueError as exc:
        raise ValueError(
            "ALARIC_ANCHOR_MIN_CELL_FRACTION must be a number in [0, 1]"
        ) from exc
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("ALARIC_ANCHOR_MIN_CELL_FRACTION must be in [0, 1]")
    return fraction


# Standalone anchor runs may use the historical override. The A9 validator
# imports the pinned default directly from anchor_constants instead.
MIN_CELL_FRACTION = _action_min_cell_fraction()


@dataclass(frozen=True)
class _StackWorkConfig:
    coordinates: np.ndarray
    ring_mask: np.ndarray
    rotaconformers: np.ndarray
    rotaconformers_index: np.ndarray
    selected_rotamer_positions: np.ndarray | None
    protein_center: np.ndarray
    output_dir: Path
    cache_size: int
    bucket_size: int
    unorganized_subdirs: bool
    center_batch_size: int
    writer_memory_lock: object | None
    precalculated_anchor: _PrecalculatedAnchor
    min_cell_fraction: float = MIN_CELL_FRACTION


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anchor",
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
        "--bucket-size",
        type=int,
        default=16,
        metavar="N",
        help=(
            "Grid bucket edge length (1..65535) written into each .arc header. "
            "Smaller buckets give organize finer-grained parallelism at the cost "
            "of more shards; larger buckets do the reverse (default: 16)."
        ),
    )
    parser.add_argument(
        "--unorganized-subdirs",
        action="store_true",
        help=(
            "Write unorganized pose shards as unorganized-PID/RANDOM.arc.zst "
            "instead of flat unorganized-PID-RANDOM.arc.zst files."
        ),
    )
    parser.add_argument(
        "--nprocs",
        type=int,
        default=default_nprocs(),
        metavar="N",
        help=(
            "Number of worker processes used to evaluate conformers "
            "(default: the SLURM CPU allocation if any, else the CPU count)."
        ),
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
        help="Restrict the search to a single 1-based conformer index.",
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
    parser.add_argument(
        "--precalculated-anchor",
        metavar="DIR",
        required=True,
        help=(
            "Directory with the ring-level precalculated filter arrays "
            "(e.g. precomputed-filters/gaussian/rrm95)."
        ),
    )
    parser.add_argument(
        "--precalculated-filter-name",
        required=True,
        help="A7 Gaussian filter name associated with --precalculated-anchor.",
    )
    parser.add_argument(
        "--anchor-reference-ring",
        required=True,
        type=_existing_file,
        metavar="FILE",
        help="Canonical A/C nucleotide ring coordinates used to build the cache.",
    )
    parser.add_argument(
        "--anchor-reference-protein-ring",
        type=_existing_file,
        metavar="FILE",
        help=(
            "Canonical protein ring coordinates. By default refe-RESNAME.npy is "
            "loaded beside --anchor-reference-ring (TYR uses refe-PHE.npy)."
        ),
    )
    return parser










def _protein_reference_name(resname: str) -> str:
    return "PHE" if resname == "TYR" else resname




















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




def _select_conformers(
    conformer_mask: np.ndarray,
    index_range: Sequence[int] | None,
    specific_index: int | None,
) -> np.ndarray:
    available_indices = np.nonzero(conformer_mask)[0]
    if len(available_indices) == 0:
        raise ValueError("No conformers available after exclusions")
    if specific_index is not None:
        if specific_index < 1:
            raise ValueError("--conformer must be positive")
        if specific_index > len(conformer_mask):
            raise ValueError(
                f"--conformer index {specific_index} out of range for library of size {len(conformer_mask)}"
            )
        if not conformer_mask[specific_index - 1]:
            raise ValueError(
                f"--conformer index {specific_index} is not available after exclusions"
            )
        return np.array([specific_index - 1], dtype=available_indices.dtype)
    if index_range is None:
        return available_indices.copy()

    first, last = index_range
    if first <= 0:
        raise ValueError("--conformer-range FIRST must be positive")
    if last < first:
        raise ValueError(
            "--conformer-range LAST must be greater than or equal to FIRST"
        )
    if last > len(conformer_mask):
        raise ValueError(
            f"--conformer-range LAST index {last} out of range for library of size {len(conformer_mask)}"
        )

    requested = np.arange(first - 1, last, dtype=available_indices.dtype)
    selected = requested[conformer_mask[requested]]
    if len(selected) == 0:
        raise ValueError(
            "No conformers available in --conformer-range after exclusions"
        )
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








def _process_conformer_precalculated(
    conformer_index: int,
    config: _StackWorkConfig,
    pose_writer: PoseWriter,
) -> tuple[int, int]:
    precalculated = config.precalculated_anchor
    if precalculated is None:
        raise RuntimeError("precalculated anchor configuration is missing")

    ring_coordinates = config.coordinates[conformer_index, config.ring_mask]
    if ring_coordinates.shape != precalculated.nucleotide_reference.shape:
        raise ValueError(
            "selected nucleotide ring and --anchor-reference-ring have different "
            f"shapes: {ring_coordinates.shape} versus {precalculated.nucleotide_reference.shape}"
        )

    rotamer_end = int(config.rotaconformers_index[conformer_index])
    rotamer_start = (
        0
        if conformer_index == 0
        else int(config.rotaconformers_index[conformer_index - 1])
    )
    rotamer_count = rotamer_end - rotamer_start
    if rotamer_count <= 0:
        return 0, 0
    if config.selected_rotamer_positions is None:
        local_rotamer_indices = np.arange(rotamer_count, dtype=np.int64)
        rotamer_block = config.rotaconformers[rotamer_start:rotamer_end]
    else:
        local_rotamer_indices = config.selected_rotamer_positions.astype(
            np.int64, copy=False
        )
        rotamer_block = config.rotaconformers[rotamer_start + local_rotamer_indices]
    if len(local_rotamer_indices) and local_rotamer_indices.max() > np.iinfo(
        np.uint16
    ).max:
        raise ValueError(
            "Rotamer index exceeds uint16 range; use --rotamer-range to constrain per-conformer rotamers"
        )
    if conformer_index > np.iinfo(np.uint16).max:
        raise ValueError("Conformer index exceeds uint16 range")

    rotamer_matrices = _rotamers_to_matrices(rotamer_block)

    total_candidates = 0
    total_surviving = 0
    for batch_start in range(0, len(local_rotamer_indices), config.center_batch_size):
        batch_end = min(
            batch_start + config.center_batch_size, len(local_rotamer_indices)
        )
        owners, grid, counts, capacity = candidate_cells(
            precalculated,
            ring_coordinates,
            rotamer_matrices[batch_start:batch_end],
            config.protein_center,
        )
        if not len(owners):
            continue
        total_candidates += int(counts.sum(dtype=np.int64))
        keep = counts >= config.min_cell_fraction * capacity
        if not np.any(keep):
            continue
        owners = owners[keep]
        grid = grid[keep]
        total_surviving += len(owners)
        pose_writer.add_chunk(
            np.full(len(owners), conformer_index, dtype=np.uint16),
            local_rotamer_indices[batch_start:batch_end][owners].astype(
                np.uint16, copy=False
            ),
            grid.astype(np.int16, copy=False),
        )
    return total_candidates, total_surviving


def _process_conformer(
    conformer_index: int,
    config: _StackWorkConfig,
    pose_writer: PoseWriter,
) -> tuple[int, int]:
    return _process_conformer_precalculated(conformer_index, config, pose_writer)


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
        bucket_size=config.bucket_size,
        memory_lock=config.writer_memory_lock,
        unorganized_subdirs=config.unorganized_subdirs,
    )
    total_candidates = 0
    total_surviving = 0
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
            candidates, surviving = _process_conformer(
                int(conformer_index),
                config,
                pose_writer,
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
    if args.cache_size <= 0:
        raise ValueError("--cache-size must be positive")
    if args.bucket_size < 1 or args.bucket_size > 65535:
        raise ValueError("--bucket-size must be in 1..65535")
    if args.nprocs <= 0:
        raise ValueError("--nprocs must be positive")
    if args.poselock is not None and args.poselock <= 0:
        raise ValueError("--poselock must be positive when set")
    if args.poselock is not None and args.poselock > args.nprocs:
        raise ValueError("--poselock must be smaller than or equal to --nprocs")

    filter_name = getattr(args, "precalculated_filter_name", None)
    if not filter_name:
        raise ValueError("--precalculated-filter-name is required for A7 validation")
    spec_path = Path(__file__).with_name("gaussian_stacking_filter.json")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    thresholds = spec.get("thresholds")
    if spec.get("schema") != 1 or not isinstance(thresholds, dict) or len(thresholds) != 12:
        raise ValueError(f"invalid A7 Gaussian filter specification: {spec_path}")
    if filter_name not in thresholds:
        raise ValueError(f"unsupported Gaussian filter name {filter_name!r} in A7")
    relative_dir = Path(thresholds[filter_name]["precalc"]["dir"])
    actual_dir = Path(args.precalculated_anchor).resolve()
    expected_tail = relative_dir.parts
    if actual_dir.parts[-len(expected_tail) :] != expected_tail:
        raise ValueError(
            f"precalculated filter directory {actual_dir} does not match "
            f"A7 {filter_name!r} path suffix {relative_dir}"
        )

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

    protein_ring, protein_resname = residue_ring_coordinates(
        protein_atoms,
        args.resid,
        structure_label="protein",
    )
    supported_protein_rings = {"PHE", "TYR", "HIS", "ARG", "TRP"}
    if protein_resname not in supported_protein_rings:
        raise ValueError(
            f"Residue {args.resid} ({protein_resname}) is not an aromatic supported protein ring"
        )
    protein_center = protein_ring.mean(axis=0)

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
        f"[3/7] Selected {len(selected_conformers)} conformers...",
        file=sys.stderr,
    )

    selected_rotamer_positions = _select_rotamer_positions(
        args.rotamer_range,
        selected_conformers,
        rotaconformers_index,
    )

    protein_reference_path = args.anchor_reference_protein_ring
    if protein_reference_path is None:
        reference_name = _protein_reference_name(protein_resname)
        protein_reference_path = str(
            Path(args.anchor_reference_ring).parent / f"refe-{reference_name}.npy"
        )
        if not Path(protein_reference_path).is_file():
            raise ValueError(
                "could not locate the canonical protein ring automatically: "
                f"{protein_reference_path}; pass --anchor-reference-protein-ring"
            )
    print("[3/7] Loading precalculated anchor filter...", file=sys.stderr)
    precalculated_anchor = _load_precalculated_anchor(
        args.precalculated_anchor,
        args.anchor_reference_ring,
        protein_reference_path,
        protein_ring,
    )
    selected_base = args.sequence[0 if args.first else 1]
    expected_ring_size = 9 if selected_base in {"A", "G"} else 6
    if len(precalculated_anchor.nucleotide_reference) != expected_ring_size:
        canonical = "A/purine" if expected_ring_size == 9 else "C/pyrimidine"
        raise ValueError(
            f"selected base {selected_base} requires the {canonical} reference "
            f"with {expected_ring_size} ring atoms, got {len(precalculated_anchor.nucleotide_reference)}"
        )

    print("[4/7] Generating offset candidates...", file=sys.stderr)
    print("[5/7] Collapsing cache candidates onto the pose grid...", file=sys.stderr)
    center_batch_size = 128
    total_candidates = 0
    total_surviving = 0
    total_poses = 0
    total_files = 0
    nprocs = min(args.nprocs, len(selected_conformers))
    pool_context = None
    writer_memory_lock = None
    if nprocs > 1:
        if "fork" not in mp.get_all_start_methods():
            raise ValueError(
                "--nprocs > 1 requires the 'fork' start method (Linux/macOS). "
                "Use --nprocs 1 on this platform."
            )
        pool_context = mp.get_context("fork")
        if args.poselock is not None:
            writer_memory_lock = pool_context.Semaphore(args.poselock)

    work_config = _StackWorkConfig(
        coordinates=coordinates,
        ring_mask=ring_mask,
        rotaconformers=rotaconformers,
        rotaconformers_index=rotaconformers_index,
        selected_rotamer_positions=selected_rotamer_positions,
        protein_center=protein_center,
        output_dir=output_dir,
        cache_size=args.cache_size,
        bucket_size=args.bucket_size,
        unorganized_subdirs=bool(args.unorganized_subdirs),
        center_batch_size=center_batch_size,
        writer_memory_lock=writer_memory_lock,
        precalculated_anchor=precalculated_anchor,
    )

    conformer_pbar = tqdm(
        total=len(selected_conformers),
        desc="Conformer-cache filter",
        unit="conf",
        mininterval=2.0,
    )
    distance_pbar = tqdm(
        desc="Offset-cache filter",
        unit="cand",
        mininterval=2.0,
    )
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
