#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import multiprocessing as mp
import os
from pathlib import Path
import sys
from typing import Iterator

import numpy as np
from tqdm import tqdm

from poses import DEFAULT_POSE_CHUNK_SIZE, PoseChunk, PoseReader
from rmsd import (
    DEFAULT_PARALLEL_MIN_POSES,
    DEFAULT_ROTAMER_MATRIX_BUILD_CHUNK,
    GRID_SPACING,
    _convert_library_rotaconformers_to_matrices,
    _dinucleotide_sequence,
    _existing_dir,
    _existing_file,
    _load_order_array,
    _load_library,
    _pdb_code,
    _positive_int,
    _rotamers_to_matrices,
    _validate_pose_range,
)


DEFAULT_PAIRWISE_CHUNK_SIZE = 5_000_000


def _rotamer_batch_to_matrices(rotamers: np.ndarray) -> np.ndarray:
    if rotamers.ndim == 3 and rotamers.shape[1:] == (3, 3):
        return rotamers.astype(np.float32, copy=False)
    return _rotamers_to_matrices(rotamers)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pairwise-rmsd",
        description=(
            "Calculate an upper-triangular pairwise RMSD vector for an organized "
            "pose directory."
        ),
    )
    parser.add_argument(
        "pose_dir",
        type=_existing_dir,
        help="Input organized pose directory containing poses-*.arc(.zst).",
    )
    parser.add_argument(
        "--sequence",
        required=True,
        type=_dinucleotide_sequence,
        help="Dinucleotide sequence for the pose directory.",
    )
    parser.add_argument(
        "--outputfile",
        type=Path,
        help="Optional output path. Use .npy for NumPy output; otherwise write text.",
    )
    parser.add_argument(
        "--order-array",
        type=_existing_file,
        help=(
            "NumPy array mapping organized pose indices to output pose indices. "
            "Used to restore pairwise results after organize reordered selected poses."
        ),
    )
    parser.add_argument(
        "--pose-range",
        nargs=2,
        metavar=("START", "END"),
        type=_positive_int,
        help="1-based, end-inclusive global pose range to process.",
    )
    parser.add_argument(
        "--chunksize",
        type=_positive_int,
        default=DEFAULT_PAIRWISE_CHUNK_SIZE,
        help=(
            "Maximum number of pairwise RMSD values per output chunk "
            f"(default: {DEFAULT_PAIRWISE_CHUNK_SIZE})."
        ),
    )
    parser.add_argument(
        "--pose-chunksize",
        type=_positive_int,
        default=DEFAULT_POSE_CHUNK_SIZE,
        help=(
            "Number of poses per input chunk while building coordinates "
            f"(default: {DEFAULT_POSE_CHUNK_SIZE})."
        ),
    )
    parser.add_argument(
        "--nprocs",
        type=_positive_int,
        default=os.cpu_count() or 1,
        help="Number of worker processes for pairwise RMSD evaluation.",
    )
    parser.add_argument(
        "--parallel-min-poses",
        type=_positive_int,
        default=DEFAULT_PARALLEL_MIN_POSES,
        help=(
            "Minimum selected pose count required before using worker "
            f"processes (default: {DEFAULT_PARALLEL_MIN_POSES})."
        ),
    )
    parser.add_argument(
        "--max-pending-chunks",
        type=_positive_int,
        help=(
            "Maximum pairwise chunks queued to worker processes "
            "(default: 2 * --nprocs)."
        ),
    )
    parser.add_argument(
        "--rotamer-matrix-build-chunksize",
        type=_positive_int,
        default=DEFAULT_ROTAMER_MATRIX_BUILD_CHUNK,
        help=(
            "Number of rotaconformers to convert per build batch when "
            "parallel pairwise RMSD precomputes the shared rotation-matrix table."
        ),
    )
    parser.add_argument(
        "--verify-checksums",
        action="store_true",
        help="Enable fraglib checksum verification when loading the library config.",
    )
    parser.add_argument(
        "--pdb-exclude",
        "--exclude",
        nargs="+",
        default=[],
        type=_pdb_code,
        metavar="PDB",
        help="One or more PDB codes to exclude from the fragment library.",
    )
    return parser


def upper_triangular_size(nposes: int) -> int:
    nposes = int(nposes)
    return nposes * (nposes - 1) // 2


def upper_triangular_offset(row: int, nposes: int) -> int:
    row = int(row)
    nposes = int(nposes)
    return row * (2 * nposes - row - 1) // 2


def _upper_triangular_indices(
    rows: np.ndarray,
    cols: np.ndarray,
    nposes: int,
) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    return rows * (2 * int(nposes) - rows - 1) // 2 + (cols - rows - 1)


def _row_from_upper_triangular_offset(offset: int, nposes: int) -> int:
    low = 0
    high = max(0, int(nposes) - 1)
    while low < high:
        mid = (low + high + 1) // 2
        if upper_triangular_offset(mid, nposes) <= offset:
            low = mid
        else:
            high = mid - 1
    if upper_triangular_offset(low, nposes) != offset:
        raise ValueError(f"Pairwise chunk offset {offset} is not row-aligned")
    return low


def _pair_count_for_rows(start: int, stop: int, nposes: int) -> int:
    return upper_triangular_offset(stop, nposes) - upper_triangular_offset(
        start,
        nposes,
    )


def _iter_row_ranges(nposes: int, max_pairs: int) -> Iterator[tuple[int, int]]:
    start = 0
    last_row = max(0, int(nposes) - 1)
    max_pairs = int(max_pairs)
    while start < last_row:
        low = start + 1
        high = last_row
        while low < high:
            mid = (low + high + 1) // 2
            if _pair_count_for_rows(start, mid, nposes) <= max_pairs:
                low = mid
            else:
                high = mid - 1
        stop = low
        yield start, stop
        start = stop


def _coordinates_for_pose_chunk(
    chunk: PoseChunk,
    *,
    coordinates: np.ndarray,
    library,
    pose_offset: int = 0,
) -> np.ndarray:
    natoms = coordinates.shape[1]
    conf_ids = chunk.conformers
    rot_ids = chunk.rotamers
    translations = chunk.translations_grid.astype(np.float32, copy=False) * GRID_SPACING
    result = np.empty((len(conf_ids), natoms, 3), dtype=np.float32)

    conf_order = np.argsort(conf_ids, kind="stable")
    sorted_conf = conf_ids[conf_order]
    boundaries = np.flatnonzero(np.diff(sorted_conf)) + 1
    starts = np.concatenate((np.array([0], dtype=np.intp), boundaries))
    stops = np.concatenate((boundaries, np.array([len(conf_order)], dtype=np.intp)))

    for start, stop in zip(starts, stops):
        rows = conf_order[start:stop]
        conf = int(sorted_conf[start])
        if conf >= len(coordinates):
            raise ValueError(f"Conformer index {conf} out of range")

        rotamers = library.get_rotamers(conf)

        for batch_start in range(0, len(rows), DEFAULT_POSE_CHUNK_SIZE):
            batch_rows = rows[batch_start : batch_start + DEFAULT_POSE_CHUNK_SIZE]
            batch_rot = rot_ids[batch_rows].astype(np.int64, copy=False)
            if batch_rot.size and int(batch_rot.max()) >= len(rotamers):
                bad_pos = int(np.flatnonzero(batch_rot >= len(rotamers))[0])
                bad_row = int(batch_rows[bad_pos])
                bad_rot = int(batch_rot[bad_pos])
                raise ValueError(
                    f"Pose row {pose_offset + bad_row}: rotamer index {bad_rot} "
                    f"out of range for conformer {conf} (n={len(rotamers)})"
                )
            rotations = _rotamer_batch_to_matrices(rotamers[batch_rot])
            transformed = np.einsum(
                "aj,njk->nak",
                coordinates[conf],
                rotations,
            )
            transformed += translations[batch_rows, None, :]
            result[batch_rows] = transformed

    return result


def load_pose_coordinates(
    reader: PoseReader,
    *,
    library,
    show_progress: bool = False,
) -> np.ndarray:
    coordinates = library.coordinates.astype(np.float32, copy=False)
    chunks = []
    offset = 0
    progress = tqdm(
        total=reader.selected_poses,
        desc="Build coordinates",
        unit="pose",
        unit_scale=True,
        disable=not show_progress,
    )
    try:
        for chunk in reader.iter_chunks():
            transformed = _coordinates_for_pose_chunk(
                chunk,
                coordinates=coordinates,
                library=library,
                pose_offset=offset,
            )
            chunks.append(transformed)
            offset += len(chunk)
            progress.update(len(chunk))
    finally:
        progress.close()

    if not chunks:
        return np.empty((0, len(library.template), 3), dtype=np.float32)
    return np.concatenate(chunks)


def _pairwise_for_rows(
    coordinates: np.ndarray,
    start: int,
    stop: int,
) -> np.ndarray:
    nposes = len(coordinates)
    natoms = coordinates.shape[1]
    result = np.empty((_pair_count_for_rows(start, stop, nposes),), dtype=np.float32)
    offset = 0
    for row in range(start, stop):
        cols = coordinates[row + 1 :]
        count = len(cols)
        dif = cols - coordinates[row][None, :, :]
        result[offset : offset + count] = np.sqrt(
            np.einsum("nij,nij->n", dif, dif) / natoms
        )
        offset += count
    return result


def iter_pairwise_rmsd_chunks(
    coordinates: np.ndarray,
    *,
    pair_chunk_size: int = DEFAULT_PAIRWISE_CHUNK_SIZE,
    nprocs: int = 1,
    max_pending_chunks: int | None = None,
) -> Iterator[np.ndarray]:
    indexed_chunks = iter_indexed_pairwise_rmsd_chunks(
        coordinates,
        pair_chunk_size=pair_chunk_size,
        nprocs=nprocs,
        max_pending_chunks=max_pending_chunks,
        ordered=True,
    )
    for _offset, chunk in indexed_chunks:
        yield chunk


def iter_indexed_pairwise_rmsd_chunks(
    coordinates: np.ndarray,
    *,
    pair_chunk_size: int = DEFAULT_PAIRWISE_CHUNK_SIZE,
    nprocs: int = 1,
    max_pending_chunks: int | None = None,
    ordered: bool = False,
) -> Iterator[tuple[int, np.ndarray]]:
    if nprocs <= 1:
        yield from _iter_indexed_pairwise_rmsd_chunks_serial(
            coordinates,
            pair_chunk_size=pair_chunk_size,
        )
        return
    yield from _iter_indexed_pairwise_rmsd_chunks_parallel(
        coordinates,
        pair_chunk_size=pair_chunk_size,
        nprocs=nprocs,
        max_pending_chunks=max_pending_chunks,
        ordered=ordered,
    )


def _iter_indexed_pairwise_rmsd_chunks_serial(
    coordinates: np.ndarray,
    *,
    pair_chunk_size: int,
) -> Iterator[tuple[int, np.ndarray]]:
    nposes = len(coordinates)
    for start, stop in _iter_row_ranges(nposes, pair_chunk_size):
        yield upper_triangular_offset(start, nposes), _pairwise_for_rows(
            coordinates,
            start,
            stop,
        )


_WORKER_COORDINATES: np.ndarray | None = None


def _init_pairwise_worker(coordinates: np.ndarray) -> None:
    global _WORKER_COORDINATES
    _WORKER_COORDINATES = coordinates


def _run_pairwise_chunk_worker(row_range: tuple[int, int]) -> np.ndarray:
    if _WORKER_COORDINATES is None:
        raise RuntimeError("Pairwise RMSD worker was not initialized")
    start, stop = row_range
    return _pairwise_for_rows(_WORKER_COORDINATES, start, stop)


def _iter_indexed_pairwise_rmsd_chunks_parallel(
    coordinates: np.ndarray,
    *,
    pair_chunk_size: int,
    nprocs: int,
    max_pending_chunks: int | None,
    ordered: bool,
) -> Iterator[tuple[int, np.ndarray]]:
    ctx = mp.get_context("fork")
    nposes = len(coordinates)
    row_ranges = iter(_iter_row_ranges(nposes, pair_chunk_size))
    window = (
        int(max_pending_chunks)
        if max_pending_chunks is not None
        else 2 * int(nprocs)
    )
    window = max(1, window)

    with cf.ProcessPoolExecutor(
        max_workers=int(nprocs),
        mp_context=ctx,
        initializer=_init_pairwise_worker,
        initargs=(coordinates,),
    ) as executor:
        pending: dict[cf.Future, tuple[int, int, int]] = {}
        completed: dict[int, tuple[int, np.ndarray]] = {}
        next_submit = 0
        next_yield = 0
        exhausted = False

        def submit_until_full() -> None:
            nonlocal next_submit, exhausted
            while not exhausted and len(pending) < window:
                try:
                    start, stop = next(row_ranges)
                except StopIteration:
                    exhausted = True
                    break
                offset = upper_triangular_offset(start, nposes)
                pending[executor.submit(_run_pairwise_chunk_worker, (start, stop))] = (
                    next_submit,
                    offset,
                    stop - start,
                )
                next_submit += 1

        submit_until_full()
        while pending or completed:
            if ordered:
                while next_yield in completed:
                    offset, result = completed.pop(next_yield)
                    next_yield += 1
                    yield offset, result

            if not pending:
                if completed:
                    raise RuntimeError("Ordered pairwise RMSD chunk stream stalled")
                break

            done, _not_done = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            ready = []
            for future in done:
                chunk_index, offset, _nrows = pending.pop(future)
                result = future.result()
                if ordered:
                    completed[chunk_index] = (offset, result)
                else:
                    ready.append((offset, result))
            submit_until_full()
            for item in ready:
                yield item


def compute_pairwise_rmsd_with_library(
    reader: PoseReader,
    *,
    library,
    pair_chunk_size: int = DEFAULT_PAIRWISE_CHUNK_SIZE,
    nprocs: int = 1,
    max_pending_chunks: int | None = None,
) -> np.ndarray:
    coordinates = load_pose_coordinates(reader, library=library)
    chunks = list(
        iter_pairwise_rmsd_chunks(
            coordinates,
            pair_chunk_size=pair_chunk_size,
            nprocs=nprocs,
            max_pending_chunks=max_pending_chunks,
        )
    )
    if not chunks:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(chunks)


def iter_pairwise_rmsd_chunks_with_library(
    reader: PoseReader,
    *,
    library,
    pair_chunk_size: int = DEFAULT_PAIRWISE_CHUNK_SIZE,
    nprocs: int = 1,
    max_pending_chunks: int | None = None,
) -> Iterator[np.ndarray]:
    coordinates = load_pose_coordinates(reader, library=library)
    yield from iter_pairwise_rmsd_chunks(
        coordinates,
        pair_chunk_size=pair_chunk_size,
        nprocs=nprocs,
        max_pending_chunks=max_pending_chunks,
    )


def iter_indexed_pairwise_rmsd_chunks_with_library(
    reader: PoseReader,
    *,
    library,
    pair_chunk_size: int = DEFAULT_PAIRWISE_CHUNK_SIZE,
    nprocs: int = 1,
    max_pending_chunks: int | None = None,
    ordered: bool = False,
) -> Iterator[tuple[int, np.ndarray]]:
    coordinates = load_pose_coordinates(reader, library=library)
    yield from iter_indexed_pairwise_rmsd_chunks(
        coordinates,
        pair_chunk_size=pair_chunk_size,
        nprocs=nprocs,
        max_pending_chunks=max_pending_chunks,
        ordered=ordered,
    )


def compute_pairwise_rmsd(
    pose_dir: Path,
    sequence: str,
    *,
    verify_checksums: bool,
    excluded_pdb_codes: set[str],
    pose_range: tuple[int, int] | None = None,
    pair_chunk_size: int = DEFAULT_PAIRWISE_CHUNK_SIZE,
    nprocs: int = 1,
    max_pending_chunks: int | None = None,
) -> np.ndarray:
    library, factory = _load_library(
        sequence,
        verify_checksums=verify_checksums,
        excluded_pdb_codes=excluded_pdb_codes,
    )
    try:
        reader = PoseReader(pose_dir, pose_range=pose_range)
        return compute_pairwise_rmsd_with_library(
            reader,
            library=library,
            pair_chunk_size=pair_chunk_size,
            nprocs=nprocs,
            max_pending_chunks=max_pending_chunks,
        )
    finally:
        factory.unload_rotaconformers()


def write_output_chunks(
    outputfile: Path | None,
    chunks: Iterator[np.ndarray],
    *,
    total_pairs: int,
    stdout_limit: int | None = None,
    show_progress: bool = False,
) -> None:
    if outputfile is None and stdout_limit is not None and total_pairs > stdout_limit:
        raise RuntimeError(
            "Refusing to stream more than one pairwise RMSD chunk to stdout; "
            "use --outputfile or --pose-range"
        )

    progress = tqdm(
        total=total_pairs,
        desc="Pairwise RMSD",
        unit="pair",
        unit_scale=True,
        disable=not show_progress,
    )

    def track(chunk: np.ndarray) -> np.ndarray:
        progress.update(len(chunk))
        return chunk

    if outputfile is None:
        try:
            for chunk_index, chunk in enumerate(chunks):
                if chunk_index > 0:
                    raise RuntimeError(
                        "Refusing to stream more than one pairwise RMSD chunk to stdout; "
                        "use --outputfile or --pose-range"
                    )
                np.savetxt(sys.stdout, track(chunk), fmt="%.6f")
        finally:
            progress.close()
        return

    outputfile.parent.mkdir(parents=True, exist_ok=True)
    try:
        with outputfile.open("w") as handle:
            for chunk in chunks:
                np.savetxt(handle, track(chunk), fmt="%.6f")
    finally:
        progress.close()


def write_npy_output_chunks(
    outputfile: Path,
    chunks: Iterator[tuple[int, np.ndarray]],
    *,
    total_pairs: int,
    show_progress: bool = False,
) -> None:
    outputfile.parent.mkdir(parents=True, exist_ok=True)
    progress = tqdm(
        total=total_pairs,
        desc="Pairwise RMSD",
        unit="pair",
        unit_scale=True,
        disable=not show_progress,
    )
    try:
        out = np.lib.format.open_memmap(
            outputfile,
            mode="w+",
            dtype=np.float32,
            shape=(total_pairs,),
        )
        written = 0
        for offset, chunk in chunks:
            stop = offset + len(chunk)
            if offset < 0 or stop > total_pairs:
                raise RuntimeError(
                    f"Pairwise RMSD chunk [{offset}:{stop}] is outside output "
                    f"size {total_pairs}"
                )
            out[offset:stop] = chunk
            written += len(chunk)
            progress.update(len(chunk))
        if written != total_pairs:
            raise RuntimeError(
                f"Expected {total_pairs} pairwise RMSDs, wrote {written}"
            )
        out.flush()
    finally:
        progress.close()


def write_ordered_npy_output_chunks(
    outputfile: Path,
    chunks: Iterator[tuple[int, np.ndarray]],
    *,
    total_pairs: int,
    nposes: int,
    order_array: np.ndarray,
    show_progress: bool = False,
) -> None:
    outputfile.parent.mkdir(parents=True, exist_ok=True)
    progress = tqdm(
        total=total_pairs,
        desc="Pairwise RMSD",
        unit="pair",
        unit_scale=True,
        disable=not show_progress,
    )
    try:
        out = np.lib.format.open_memmap(
            outputfile,
            mode="w+",
            dtype=np.float32,
            shape=(total_pairs,),
        )
        written = 0
        for offset, chunk in chunks:
            stop = offset + len(chunk)
            if offset < 0 or stop > total_pairs:
                raise RuntimeError(
                    f"Pairwise RMSD chunk [{offset}:{stop}] is outside output "
                    f"size {total_pairs}"
                )
            row = _row_from_upper_triangular_offset(offset, nposes)
            source_pos = 0
            while source_pos < len(chunk):
                count = nposes - row - 1
                source_stop = source_pos + count
                if source_stop > len(chunk):
                    raise RuntimeError("Pairwise RMSD chunk ended mid-row")
                out_row = int(order_array[row])
                out_cols = np.asarray(order_array[row + 1 :], dtype=np.int64)
                target_rows = np.minimum(out_row, out_cols)
                target_cols = np.maximum(out_row, out_cols)
                if np.any(target_rows == target_cols):
                    raise ValueError("--order-array must be a permutation")
                targets = _upper_triangular_indices(target_rows, target_cols, nposes)
                out[targets] = chunk[source_pos:source_stop]
                written += count
                progress.update(count)
                source_pos = source_stop
                row += 1
        if written != total_pairs:
            raise RuntimeError(
                f"Expected {total_pairs} pairwise RMSDs, wrote {written}"
            )
        out.flush()
    finally:
        progress.close()


def collect_ordered_output_chunks(
    chunks: Iterator[tuple[int, np.ndarray]],
    *,
    total_pairs: int,
    nposes: int,
    order_array: np.ndarray,
    show_progress: bool = False,
) -> np.ndarray:
    progress = tqdm(
        total=total_pairs,
        desc="Pairwise RMSD",
        unit="pair",
        unit_scale=True,
        disable=not show_progress,
    )
    out = np.empty((total_pairs,), dtype=np.float32)
    written = 0
    try:
        for offset, chunk in chunks:
            stop = offset + len(chunk)
            if offset < 0 or stop > total_pairs:
                raise RuntimeError(
                    f"Pairwise RMSD chunk [{offset}:{stop}] is outside output "
                    f"size {total_pairs}"
                )
            row = _row_from_upper_triangular_offset(offset, nposes)
            source_pos = 0
            while source_pos < len(chunk):
                count = nposes - row - 1
                source_stop = source_pos + count
                if source_stop > len(chunk):
                    raise RuntimeError("Pairwise RMSD chunk ended mid-row")
                out_row = int(order_array[row])
                out_cols = np.asarray(order_array[row + 1 :], dtype=np.int64)
                target_rows = np.minimum(out_row, out_cols)
                target_cols = np.maximum(out_row, out_cols)
                if np.any(target_rows == target_cols):
                    raise ValueError("--order-array must be a permutation")
                targets = _upper_triangular_indices(target_rows, target_cols, nposes)
                out[targets] = chunk[source_pos:source_stop]
                written += count
                progress.update(count)
                source_pos = source_stop
                row += 1
        if written != total_pairs:
            raise RuntimeError(
                f"Expected {total_pairs} pairwise RMSDs, wrote {written}"
            )
        return out
    finally:
        progress.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        pose_range = _validate_pose_range(args.pose_range)
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(str(exc)) from None

    library, factory = _load_library(
        args.sequence,
        verify_checksums=args.verify_checksums,
        excluded_pdb_codes=set(args.pdb_exclude),
    )
    try:
        reader = PoseReader(
            args.pose_dir,
            pose_range=pose_range,
            rows_per_chunk=args.pose_chunksize,
        )
        if args.order_array is not None and pose_range is not None:
            raise ValueError("--order-array cannot be combined with --pose-range")
        order_array = (
            None
            if args.order_array is None
            else _load_order_array(args.order_array, reader.selected_poses)
        )
        effective_nprocs = (
            args.nprocs if reader.selected_poses > args.parallel_min_poses else 1
        )
        if effective_nprocs > 1:
            _convert_library_rotaconformers_to_matrices(
                library,
                factory,
                rows_per_chunk=args.rotamer_matrix_build_chunksize,
            )
        coordinates = load_pose_coordinates(
            reader,
            library=library,
            show_progress=reader.selected_poses > args.pose_chunksize,
        )
        total_pairs = upper_triangular_size(len(coordinates))
        show_pair_progress = total_pairs > args.chunksize
        if order_array is not None:
            indexed_chunks = iter_indexed_pairwise_rmsd_chunks(
                coordinates,
                pair_chunk_size=args.chunksize,
                nprocs=effective_nprocs,
                max_pending_chunks=args.max_pending_chunks,
                ordered=False,
            )
            if args.outputfile is not None and args.outputfile.suffix.lower() == ".npy":
                write_ordered_npy_output_chunks(
                    args.outputfile,
                    indexed_chunks,
                    total_pairs=total_pairs,
                    nposes=len(coordinates),
                    order_array=order_array,
                    show_progress=show_pair_progress,
                )
            else:
                ordered = collect_ordered_output_chunks(
                    indexed_chunks,
                    total_pairs=total_pairs,
                    nposes=len(coordinates),
                    order_array=order_array,
                    show_progress=show_pair_progress,
                )
                write_output_chunks(
                    args.outputfile,
                    iter([ordered]),
                    total_pairs=total_pairs,
                    stdout_limit=args.chunksize,
                    show_progress=False,
                )
        elif args.outputfile is not None and args.outputfile.suffix.lower() == ".npy":
            indexed_chunks = iter_indexed_pairwise_rmsd_chunks(
                coordinates,
                pair_chunk_size=args.chunksize,
                nprocs=effective_nprocs,
                max_pending_chunks=args.max_pending_chunks,
                ordered=False,
            )
            write_npy_output_chunks(
                args.outputfile,
                indexed_chunks,
                total_pairs=total_pairs,
                show_progress=show_pair_progress,
            )
        else:
            chunks = iter_pairwise_rmsd_chunks(
                coordinates,
                pair_chunk_size=args.chunksize,
                nprocs=effective_nprocs,
                max_pending_chunks=args.max_pending_chunks,
            )
            write_output_chunks(
                args.outputfile,
                chunks,
                total_pairs=total_pairs,
                stdout_limit=args.chunksize,
                show_progress=show_pair_progress,
            )
    finally:
        factory.unload_rotaconformers()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
