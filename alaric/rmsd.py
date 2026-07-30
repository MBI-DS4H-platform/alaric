#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import gc
from math import sqrt
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Iterator

import numpy as np
from tqdm import tqdm

_ALARIC_DIR = Path(__file__).resolve().parent
if str(_ALARIC_DIR) not in sys.path:
    sys.path.insert(0, str(_ALARIC_DIR))

from nprocs import default_nprocs
from poses import DEFAULT_POSE_CHUNK_SIZE, PoseChunk, PoseReader


GRID_SPACING = sqrt(3) / 3
DEFAULT_ROTAMER_MATRIX_BUILD_CHUNK = 1_000_000
DEFAULT_PARALLEL_MIN_POSES = 10_000
RMSD_DECIMALS = 3

def _existing_file(path: str) -> Path:
    result = Path(path)
    if not result.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    return result


def _existing_dir(path: str) -> Path:
    result = Path(path)
    if not result.is_dir():
        raise argparse.ArgumentTypeError(f"directory does not exist: {path}")
    return result


def _load_order_array(path: Path, total_poses: int) -> np.ndarray:
    order = np.load(path, mmap_mode="r")
    if order.ndim != 1:
        raise ValueError("--order-array must be a 1D NumPy array")
    if len(order) != total_poses:
        raise ValueError(
            f"--order-array length {len(order)} does not match pose count {total_poses}"
        )
    if not np.issubdtype(order.dtype, np.integer):
        raise ValueError("--order-array must contain integer indices")
    if (
        np.issubdtype(order.dtype, np.signedinteger)
        and order.size
        and int(order.min()) < 0
    ):
        raise ValueError("--order-array must contain non-negative indices")
    if order.size and int(order.max()) >= total_poses:
        raise ValueError("--order-array contains indices outside the pose array")
    return order


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer: {value}") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _validate_pose_range(values: list[int] | None) -> tuple[int, int] | None:
    if values is None:
        return None
    start, end = values
    if start > end:
        raise argparse.ArgumentTypeError("--pose-range START must be <= END")
    return start, end


def _dinucleotide_sequence(seq: str) -> str:
    seq = seq.strip().upper()
    if len(seq) != 2:
        raise argparse.ArgumentTypeError("--sequence must be a dinucleotide (length 2)")
    if any(ch not in {"A", "C", "G", "U"} for ch in seq):
        raise argparse.ArgumentTypeError("--sequence must contain only A/C/G/U")
    return seq


def _pdb_code(code: str) -> str:
    code = code.strip()
    if len(code) != 4 or not code[0].isdigit() or not code[1:].isalnum():
        raise argparse.ArgumentTypeError(
            "PDB codes must be 4 chars: one digit + 3 alphanumeric characters"
        )
    return code.lower()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rmsd",
        description="Calculate RMSDs between an organized pose directory and a reference PDB fragment.",
    )
    parser.add_argument(
        "pose_dir",
        type=_existing_dir,
        help="Input organized pose directory containing poses-*.arc(.zst).",
    )
    parser.add_argument(
        "--reference",
        required=True,
        type=_existing_file,
        help="Reference PDB file containing the selected dinucleotide fragment.",
    )
    parser.add_argument(
        "--fragment",
        type=_positive_int,
        metavar="N",
        help=(
            "Select residues N and N+1 from the reference PDB. "
            "If omitted, the reference must contain exactly two residues."
        ),
    )
    parser.add_argument(
        "--sequence",
        type=_dinucleotide_sequence,
        help="Expected dinucleotide sequence. If provided, it must match the reference.",
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
            "Used to restore results after organize reordered selected poses."
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
        default=DEFAULT_POSE_CHUNK_SIZE,
        help=f"Number of poses per RMSD chunk (default: {DEFAULT_POSE_CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--nprocs",
        type=_positive_int,
        default=default_nprocs(),
        help=(
            "Number of worker processes for RMSD evaluation "
            "(default: the SLURM CPU allocation if any, else the CPU count)."
        ),
    )
    parser.add_argument(
        "--parallel-min-poses",
        type=_positive_int,
        default=DEFAULT_PARALLEL_MIN_POSES,
        help=(
            "Minimum selected pose count required before using worker "
            f"processes or pre-fork matrix conversion (default: {DEFAULT_PARALLEL_MIN_POSES})."
        ),
    )
    parser.add_argument(
        "--max-pending-chunks",
        type=_positive_int,
        help=(
            "Maximum RMSD chunks queued to worker processes "
            "(default: 2 * --nprocs)."
        ),
    )
    parser.add_argument(
        "--rotamer-matrix-build-chunksize",
        type=_positive_int,
        default=DEFAULT_ROTAMER_MATRIX_BUILD_CHUNK,
        help=(
            "Number of rotaconformers to convert per build batch when "
            "parallel RMSD precomputes the shared rotation-matrix table."
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


def _decode(value) -> str:
    if isinstance(value, (bytes, bytearray, memoryview, np.bytes_)):
        return bytes(value).decode().strip()
    if hasattr(value, "item"):
        return _decode(value.item())
    return str(value).strip()


def _base_from_resname(resname: str) -> str:
    resname = resname.strip().upper()
    mapping = {base: base for base in "ACGU"}
    mapping.update({f"R{base}": base for base in "ACGU"})
    try:
        return mapping[resname]
    except KeyError:
        raise ValueError(f"Unsupported RNA residue name {resname!r}") from None


def _rotamers_to_matrices(rotamers: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    if rotamers.ndim == 3 and rotamers.shape[1:] == (3, 3):
        return rotamers.astype(np.float32, copy=False)
    if rotamers.ndim == 2 and rotamers.shape[1] == 3:
        return Rotation.from_rotvec(rotamers).as_matrix().astype(np.float32, copy=False)
    raise ValueError(
        "Unsupported rotamer representation; expected shape [N,3] rotvec or [N,3,3] matrix"
    )


def _convert_library_rotaconformers_to_matrices(
    library,
    factory,
    *,
    rows_per_chunk: int = DEFAULT_ROTAMER_MATRIX_BUILD_CHUNK,
):
    rotaconformers = library.rotaconformers
    if rotaconformers is None:
        raise RuntimeError("Rotaconformers were not loaded")
    if rotaconformers.ndim == 3 and rotaconformers.shape[1:] == (3, 3):
        matrices = rotaconformers.astype(np.float32, copy=False)
    elif rotaconformers.ndim == 2 and rotaconformers.shape[1] == 3:
        from scipy.spatial.transform import Rotation

        rows_per_chunk = int(rows_per_chunk)
        matrices = np.empty((len(rotaconformers), 3, 3), dtype=np.float32)
        with tqdm(
            total=len(rotaconformers),
            desc="Build rotamer matrices",
            unit="rot",
            unit_scale=True,
            mininterval=2.0,
        ) as progress:
            for start in range(0, len(rotaconformers), rows_per_chunk):
                stop = min(start + rows_per_chunk, len(rotaconformers))
                matrices[start:stop] = Rotation.from_rotvec(
                    rotaconformers[start:stop]
                ).as_matrix()
                progress.update(stop - start)
    else:
        raise ValueError(
            "Unsupported rotaconformer representation; expected shape [N,3] "
            "rotvec or [N,3,3] matrix"
        )

    object.__setattr__(library, "rotaconformers", matrices)
    factory.rotaconformers = matrices
    del rotaconformers
    gc.collect()


def _infer_fragment(
    reference: np.ndarray, first_resid: int | None
) -> tuple[str, np.ndarray]:
    if first_resid is None:
        selected_resids = [int(resid) for resid in np.unique(reference["resid"])]
        if len(selected_resids) != 2:
            raise ValueError(
                "--fragment is required unless the reference contains exactly two residues"
            )
    else:
        selected_resids = [first_resid, first_resid + 1]
    mask = np.isin(reference["resid"], selected_resids)
    fragment = reference[mask]
    if len(fragment) == 0:
        raise ValueError(
            f"Reference contains no atoms for selected residues {selected_resids}"
        )

    present_resids = [int(resid) for resid in np.unique(fragment["resid"])]
    if present_resids != selected_resids:
        raise ValueError(
            f"Expected selected residues {selected_resids}, found {present_resids}"
        )

    sequence = []
    normalized = fragment.copy()
    normalized["index"] = np.arange(
        1,
        len(normalized) + 1,
        dtype=normalized["index"].dtype,
    )
    for local_resid, resid in enumerate(selected_resids, start=1):
        curr_mask = normalized["resid"] == resid
        curr = normalized[curr_mask]
        resnames = sorted({_decode(value) for value in curr["resname"]})
        if len(resnames) != 1:
            raise ValueError(f"Residue {resid} has multiple resnames: {resnames}")
        base = _base_from_resname(resnames[0])
        sequence.append(base)
        normalized["resid"][curr_mask] = local_resid
        normalized["resname"][curr_mask] = base

    return "".join(sequence), normalized


def _load_library(
    sequence: str,
    *,
    verify_checksums: bool,
    excluded_pdb_codes: set[str],
):
    from library import config

    libraries, _ = config(verify_checksums=verify_checksums)
    if sequence not in libraries:
        raise ValueError(f"Sequence {sequence} not available in fragment library")
    factory = libraries[sequence]
    factory.load_rotaconformers()
    pdb_code_filter: str | list[str] | None = None
    if excluded_pdb_codes:
        pdb_code_filter = sorted(excluded_pdb_codes)
    library = factory.create(pdb_code_filter, with_rotaconformers=True)
    if library.rotaconformers is None or library.rotaconformers_index is None:
        raise RuntimeError("Rotaconformers were not loaded")
    return library, factory


def _load_reference_coordinates(
    reference_path: Path,
    first_resid: int | None,
    sequence: str | None,
    template: np.ndarray,
) -> tuple[str, np.ndarray]:
    from parse_pdb import parse_pdb

    reference = parse_pdb(reference_path.read_text())
    if len(reference) == 0:
        raise ValueError("Reference PDB contains no atoms")
    model_ids = np.unique(reference["model"])
    if len(model_ids) > 1:
        reference = reference[reference["model"] == model_ids[0]]
        if len(reference) == 0:
            raise ValueError("Reference PDB first model contains no atoms")

    inferred_sequence, fragment = _infer_fragment(reference, first_resid)
    if sequence is not None and sequence != inferred_sequence:
        raise ValueError(
            f"--sequence {sequence} does not match reference sequence {inferred_sequence}"
        )

    if fragment.shape != template.shape:
        raise ValueError(
            f"Reference/template shape mismatch: {fragment.shape} vs {template.shape}"
        )

    for field in template.dtype.names:
        if field in {"x", "y", "z", "occupancy", "segid"}:
            continue
        if field == "chain":
            continue
        equal = np.array_equal(fragment[field], template[field])
        if not equal:
            raise ValueError(
                f"Reference topology does not match template in field '{field}'"
            )

    coordinates = np.stack((fragment["x"], fragment["y"], fragment["z"]), axis=-1)
    return inferred_sequence, coordinates.astype(np.float32, copy=False)


def compute_rmsd(
    pose_dir: Path,
    reference_path: Path,
    *,
    fragment: int | None,
    sequence: str | None,
    verify_checksums: bool,
    excluded_pdb_codes: set[str],
    pose_range: tuple[int, int] | None = None,
    nprocs: int = 1,
    max_pending_chunks: int | None = None,
) -> np.ndarray:
    if sequence is None:
        from parse_pdb import parse_pdb

        inferred_sequence, _ = _infer_fragment(
            parse_pdb(reference_path.read_text()), fragment
        )
        sequence = inferred_sequence

    library, factory = _load_library(
        sequence,
        verify_checksums=verify_checksums,
        excluded_pdb_codes=excluded_pdb_codes,
    )
    try:
        reader = PoseReader(pose_dir, pose_range=pose_range)
        return compute_rmsd_with_library(
            reader,
            reference_path,
            fragment=fragment,
            sequence=sequence,
            library=library,
            nprocs=nprocs,
            max_pending_chunks=max_pending_chunks,
        )
    finally:
        factory.unload_rotaconformers()


def compute_rmsd_with_library(
    reader: PoseReader,
    reference_path: Path,
    *,
    fragment: int | None,
    sequence: str,
    library,
    nprocs: int = 1,
    max_pending_chunks: int | None = None,
) -> np.ndarray:
    inferred_sequence, reference_coords = _load_reference_coordinates(
        reference_path, fragment, sequence, library.template
    )
    if sequence != inferred_sequence:
        raise ValueError(
            f"Library sequence {sequence} does not match reference {inferred_sequence}"
        )

    chunks = list(
        iter_rmsd_chunks(
            reader,
            reference_coords,
            library=library,
            nprocs=nprocs,
            max_pending_chunks=max_pending_chunks,
        )
    )
    if not chunks:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(chunks)


def iter_rmsd_chunks_with_library(
    reader: PoseReader,
    reference_path: Path,
    *,
    fragment: int | None,
    sequence: str,
    library,
    nprocs: int = 1,
    max_pending_chunks: int | None = None,
) -> Iterator[np.ndarray]:
    inferred_sequence, reference_coords = _load_reference_coordinates(
        reference_path, fragment, sequence, library.template
    )
    if sequence != inferred_sequence:
        raise ValueError(
            f"Library sequence {sequence} does not match reference {inferred_sequence}"
        )
    yield from iter_rmsd_chunks(
        reader,
        reference_coords,
        library=library,
        nprocs=nprocs,
        max_pending_chunks=max_pending_chunks,
    )


def iter_indexed_rmsd_chunks_with_library(
    reader: PoseReader,
    reference_path: Path,
    *,
    fragment: int | None,
    sequence: str,
    library,
    nprocs: int = 1,
    max_pending_chunks: int | None = None,
    ordered: bool = False,
) -> Iterator[tuple[int, np.ndarray]]:
    inferred_sequence, reference_coords = _load_reference_coordinates(
        reference_path, fragment, sequence, library.template
    )
    if sequence != inferred_sequence:
        raise ValueError(
            f"Library sequence {sequence} does not match reference {inferred_sequence}"
        )
    yield from iter_indexed_rmsd_chunks(
        reader,
        reference_coords,
        library=library,
        nprocs=nprocs,
        max_pending_chunks=max_pending_chunks,
        ordered=ordered,
    )


def iter_rmsd_chunks(
    reader: PoseReader,
    reference_coords: np.ndarray,
    *,
    library,
    nprocs: int = 1,
    max_pending_chunks: int | None = None,
) -> Iterator[np.ndarray]:
    indexed_chunks = iter_indexed_rmsd_chunks(
        reader,
        reference_coords,
        library=library,
        nprocs=nprocs,
        max_pending_chunks=max_pending_chunks,
        ordered=True,
    )
    for _offset, rmsd in indexed_chunks:
        yield rmsd


def iter_indexed_rmsd_chunks(
    reader: PoseReader,
    reference_coords: np.ndarray,
    *,
    library,
    nprocs: int = 1,
    max_pending_chunks: int | None = None,
    ordered: bool = False,
) -> Iterator[tuple[int, np.ndarray]]:
    if nprocs <= 1:
        yield from _iter_indexed_rmsd_chunks_serial(
            reader,
            reference_coords,
            library=library,
        )
        return
    yield from _iter_indexed_rmsd_chunks_parallel(
        reader,
        reference_coords,
        library=library,
        nprocs=nprocs,
        max_pending_chunks=max_pending_chunks,
        ordered=ordered,
    )


def _iter_indexed_rmsd_chunks_serial(
    reader: PoseReader,
    reference_coords: np.ndarray,
    *,
    library,
) -> Iterator[tuple[int, np.ndarray]]:
    coordinates = library.coordinates.astype(np.float32, copy=False)
    offset = 0

    while True:
        chunk = reader.read(reader.rows_per_chunk)
        if len(chunk) == 0:
            break
        rmsd = _rmsd_for_pose_chunk(
            chunk,
            reference_coords,
            coordinates=coordinates,
            library=library,
        )
        yield offset, rmsd
        offset += len(chunk)


_WORKER_REFERENCE_COORDS: np.ndarray | None = None
_WORKER_COORDINATES: np.ndarray | None = None
_WORKER_LIBRARY = None


def _init_rmsd_worker(reference_coords: np.ndarray) -> None:
    global _WORKER_REFERENCE_COORDS, _WORKER_COORDINATES
    _WORKER_REFERENCE_COORDS = reference_coords
    _WORKER_COORDINATES = _WORKER_LIBRARY.coordinates.astype(np.float32, copy=False)


def _run_rmsd_chunk_worker(chunk: PoseChunk) -> np.ndarray:
    if _WORKER_REFERENCE_COORDS is None or _WORKER_COORDINATES is None:
        raise RuntimeError("RMSD worker was not initialized")
    return _rmsd_for_pose_chunk(
        chunk,
        _WORKER_REFERENCE_COORDS,
        coordinates=_WORKER_COORDINATES,
        library=_WORKER_LIBRARY,
    )


def _iter_indexed_rmsd_chunks_parallel(
    reader: PoseReader,
    reference_coords: np.ndarray,
    *,
    library,
    nprocs: int,
    max_pending_chunks: int | None,
    ordered: bool,
) -> Iterator[tuple[int, np.ndarray]]:
    global _WORKER_LIBRARY
    ctx = mp.get_context("fork")
    window = (
        int(max_pending_chunks)
        if max_pending_chunks is not None
        else 2 * int(nprocs)
    )
    window = max(1, window)
    _WORKER_LIBRARY = library
    try:
        with cf.ProcessPoolExecutor(
            max_workers=int(nprocs),
            mp_context=ctx,
            initializer=_init_rmsd_worker,
            initargs=(reference_coords,),
        ) as executor:
            chunk_iter = reader.iter_chunks()
            pending: dict[cf.Future, tuple[int, int]] = {}
            completed: dict[int, tuple[int, np.ndarray]] = {}
            next_submit = 0
            next_yield = 0
            next_offset = 0
            exhausted = False

            def submit_until_full() -> None:
                nonlocal next_submit, next_offset, exhausted
                while not exhausted and len(pending) < window:
                    try:
                        chunk = next(chunk_iter)
                    except StopIteration:
                        exhausted = True
                        break
                    pending[executor.submit(_run_rmsd_chunk_worker, chunk)] = (
                        next_submit,
                        next_offset,
                    )
                    next_submit += 1
                    next_offset += len(chunk)

            submit_until_full()
            while pending or completed:
                if ordered:
                    while next_yield in completed:
                        offset, result = completed.pop(next_yield)
                        next_yield += 1
                        yield offset, result

                if not pending:
                    if completed:
                        raise RuntimeError("Ordered RMSD chunk stream stalled")
                    break

                done, _not_done = cf.wait(
                    pending,
                    return_when=cf.FIRST_COMPLETED,
                )
                ready = []
                for future in done:
                    chunk_index, offset = pending.pop(future)
                    result = future.result()
                    if ordered:
                        completed[chunk_index] = (offset, result)
                    else:
                        ready.append((offset, result))
                submit_until_full()
                for item in ready:
                    yield item
    finally:
        _WORKER_LIBRARY = None


def _rmsd_for_pose_chunk(
    chunk: PoseChunk,
    reference_coords: np.ndarray,
    *,
    coordinates: np.ndarray,
    library,
) -> np.ndarray:
    natoms = len(reference_coords)
    pose_batch_size = 4096

    conf_ids = chunk.conformers
    rot_ids = chunk.rotamers
    translations = chunk.translations_grid.astype(np.float32, copy=False) * GRID_SPACING
    rmsd = np.empty((len(conf_ids),), dtype=np.float32)

    ref_mean = reference_coords.mean(axis=0).astype(np.float32, copy=False)
    ref_centered = (reference_coords - ref_mean).astype(np.float32, copy=False)
    ref_trace = float(np.einsum("ij,ij->", ref_centered, ref_centered))

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

        conf_coords = coordinates[conf].astype(np.float32, copy=False)
        mean_coords = conf_coords.mean(axis=0).astype(np.float32, copy=False)
        centered = (conf_coords - mean_coords).astype(np.float32, copy=False)
        pose_trace = float(np.einsum("ij,ij->", centered, centered))
        cross = centered.T @ ref_centered  # (3, 3)

        for batch_start in range(0, len(rows), pose_batch_size):
            batch_rows = rows[batch_start : batch_start + pose_batch_size]
            batch_rot = rot_ids[batch_rows].astype(np.int64, copy=False)
            if batch_rot.size and int(batch_rot.max()) >= len(rotamers):
                bad_pos = int(np.flatnonzero(batch_rot >= len(rotamers))[0])
                bad_row = int(batch_rows[bad_pos])
                bad_rot = int(batch_rot[bad_pos])
                raise ValueError(
                    f"Pose row {bad_row}: rotamer index {bad_rot} out of range "
                    f"for conformer {conf} (n={len(rotamers)})"
                )

            unique_rot, inverse = np.unique(batch_rot, return_inverse=True)
            rotations = _rotamers_to_matrices(rotamers[unique_rot])
            trace_scores = np.einsum("njk,jk->n", rotations, cross)
            rc_sd_unique = pose_trace + ref_trace - 2.0 * trace_scores
            mean_rotated_unique = np.einsum("j,njk->nk", mean_coords, rotations)

            rc_sd = rc_sd_unique[inverse]
            mean_rotated = mean_rotated_unique[inverse]
            delta = mean_rotated + translations[batch_rows] - ref_mean
            total_sd = rc_sd + natoms * np.einsum("ij,ij->i", delta, delta)
            rmsd[batch_rows] = np.sqrt(np.maximum(total_sd, 0.0) / natoms)

    return np.round(rmsd, RMSD_DECIMALS).astype(np.float32, copy=False)


def write_output_chunks(
    outputfile: Path | None,
    chunks: Iterator[np.ndarray],
    *,
    total_poses: int,
    stdout_limit: int | None = None,
    show_progress: bool = False,
) -> None:
    if outputfile is None and stdout_limit is not None and total_poses > stdout_limit:
        raise RuntimeError(
            "Refusing to stream more than one RMSD chunk to stdout; "
            "use --outputfile or --pose-range"
        )

    progress = tqdm(
        total=total_poses,
        desc="RMSD",
        unit="pose",
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
                        "Refusing to stream more than one RMSD chunk to stdout; "
                        "use --outputfile or --pose-range"
                    )
                np.savetxt(sys.stdout, track(chunk), fmt=f"%.{RMSD_DECIMALS}f")
        finally:
            progress.close()
        return

    outputfile.parent.mkdir(parents=True, exist_ok=True)
    if outputfile.suffix.lower() == ".npy":
        try:
            out = np.lib.format.open_memmap(
                outputfile,
                mode="w+",
                dtype=np.float32,
                shape=(total_poses,),
            )
            offset = 0
            for chunk in chunks:
                chunk = track(chunk)
                out[offset : offset + len(chunk)] = chunk
                offset += len(chunk)
            if offset != total_poses:
                raise RuntimeError(f"Expected {total_poses} RMSDs, wrote {offset}")
            out.flush()
        finally:
            progress.close()
        return

    try:
        with outputfile.open("w") as handle:
            for chunk in chunks:
                np.savetxt(handle, track(chunk), fmt=f"%.{RMSD_DECIMALS}f")
    finally:
        progress.close()


def write_npy_output_chunks(
    outputfile: Path,
    chunks: Iterator[tuple[int, np.ndarray]],
    *,
    total_poses: int,
    show_progress: bool = False,
) -> None:
    outputfile.parent.mkdir(parents=True, exist_ok=True)
    progress = tqdm(
        total=total_poses,
        desc="RMSD",
        unit="pose",
        unit_scale=True,
        disable=not show_progress,
    )
    try:
        out = np.lib.format.open_memmap(
            outputfile,
            mode="w+",
            dtype=np.float32,
            shape=(total_poses,),
        )
        written = 0
        for offset, chunk in chunks:
            stop = offset + len(chunk)
            if offset < 0 or stop > total_poses:
                raise RuntimeError(
                    f"RMSD chunk [{offset}:{stop}] is outside output size {total_poses}"
                )
            out[offset:stop] = chunk
            written += len(chunk)
            progress.update(len(chunk))
        if written != total_poses:
            raise RuntimeError(f"Expected {total_poses} RMSDs, wrote {written}")
        out.flush()
    finally:
        progress.close()


def write_ordered_npy_output_chunks(
    outputfile: Path,
    chunks: Iterator[tuple[int, np.ndarray]],
    *,
    total_poses: int,
    order_array: np.ndarray,
    show_progress: bool = False,
) -> None:
    outputfile.parent.mkdir(parents=True, exist_ok=True)
    progress = tqdm(
        total=total_poses,
        desc="RMSD",
        unit="pose",
        unit_scale=True,
        disable=not show_progress,
    )
    try:
        out = np.lib.format.open_memmap(
            outputfile,
            mode="w+",
            dtype=np.float32,
            shape=(total_poses,),
        )
        written = 0
        for offset, chunk in chunks:
            stop = offset + len(chunk)
            if offset < 0 or stop > total_poses:
                raise RuntimeError(
                    f"RMSD chunk [{offset}:{stop}] is outside output size {total_poses}"
                )
            targets = np.asarray(order_array[offset:stop], dtype=np.intp)
            out[targets] = chunk
            written += len(chunk)
            progress.update(len(chunk))
        if written != total_poses:
            raise RuntimeError(f"Expected {total_poses} RMSDs, wrote {written}")
        out.flush()
    finally:
        progress.close()


def collect_ordered_output_chunks(
    chunks: Iterator[tuple[int, np.ndarray]],
    *,
    total_poses: int,
    order_array: np.ndarray,
    show_progress: bool = False,
) -> np.ndarray:
    progress = tqdm(
        total=total_poses,
        desc="RMSD",
        unit="pose",
        unit_scale=True,
        disable=not show_progress,
    )
    out = np.empty((total_poses,), dtype=np.float32)
    written = 0
    try:
        for offset, chunk in chunks:
            stop = offset + len(chunk)
            targets = np.asarray(order_array[offset:stop], dtype=np.intp)
            out[targets] = chunk
            written += len(chunk)
            progress.update(len(chunk))
        if written != total_poses:
            raise RuntimeError(f"Expected {total_poses} RMSDs, wrote {written}")
        return out
    finally:
        progress.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        pose_range = _validate_pose_range(args.pose_range)
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(str(exc)) from None
    sequence = args.sequence
    if sequence is None:
        from parse_pdb import parse_pdb

        sequence, _ = _infer_fragment(
            parse_pdb(args.reference.read_text()),
            args.fragment,
        )

    library, factory = _load_library(
        sequence,
        verify_checksums=args.verify_checksums,
        excluded_pdb_codes=set(args.pdb_exclude),
    )
    try:
        reader = PoseReader(
            args.pose_dir,
            pose_range=pose_range,
            rows_per_chunk=args.chunksize,
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
        if order_array is not None:
            indexed_chunks = iter_indexed_rmsd_chunks_with_library(
                reader,
                args.reference,
                fragment=args.fragment,
                sequence=sequence,
                library=library,
                nprocs=effective_nprocs,
                max_pending_chunks=args.max_pending_chunks,
                ordered=False,
            )
            if args.outputfile is not None and args.outputfile.suffix.lower() == ".npy":
                write_ordered_npy_output_chunks(
                    args.outputfile,
                    indexed_chunks,
                    total_poses=reader.selected_poses,
                    order_array=order_array,
                    show_progress=reader.selected_poses > args.chunksize,
                )
            else:
                ordered = collect_ordered_output_chunks(
                    indexed_chunks,
                    total_poses=reader.selected_poses,
                    order_array=order_array,
                    show_progress=reader.selected_poses > args.chunksize,
                )
                write_output_chunks(
                    args.outputfile,
                    iter([ordered]),
                    total_poses=reader.selected_poses,
                    stdout_limit=args.chunksize,
                    show_progress=False,
                )
        elif args.outputfile is not None and args.outputfile.suffix.lower() == ".npy":
            indexed_chunks = iter_indexed_rmsd_chunks_with_library(
                reader,
                args.reference,
                fragment=args.fragment,
                sequence=sequence,
                library=library,
                nprocs=effective_nprocs,
                max_pending_chunks=args.max_pending_chunks,
                ordered=False,
            )
            write_npy_output_chunks(
                args.outputfile,
                indexed_chunks,
                total_poses=reader.selected_poses,
                show_progress=reader.selected_poses > args.chunksize,
            )
        else:
            chunks = iter_rmsd_chunks_with_library(
                reader,
                args.reference,
                fragment=args.fragment,
                sequence=sequence,
                library=library,
                nprocs=effective_nprocs,
                max_pending_chunks=args.max_pending_chunks,
            )
            write_output_chunks(
                args.outputfile,
                chunks,
                total_poses=reader.selected_poses,
                stdout_limit=args.chunksize,
                show_progress=reader.selected_poses > args.chunksize,
            )
    finally:
        factory.unload_rotaconformers()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
