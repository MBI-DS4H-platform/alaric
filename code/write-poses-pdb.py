#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

from parse_pdb import atomic_dtype
from poses import DEFAULT_POSE_CHUNK_SIZE, PoseChunk, PoseReader
from rmsd import (
    GRID_SPACING,
    _dinucleotide_sequence,
    _existing_dir,
    _load_library,
    _pdb_code,
    _positive_int,
    _rotamers_to_matrices,
    _validate_pose_range,
)
from write_pdb import write_multi_pdb


MAX_MODELS = 9999


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="write-poses-pdb",
        description=(
            "Materialize organized pose coordinates and write them as a "
            "multi-model PDB to stdout."
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
        "--pose-range",
        nargs=2,
        metavar=("START", "END"),
        type=_positive_int,
        help="1-based, end-inclusive global pose range to write.",
    )
    parser.add_argument(
        "--chunksize",
        type=_positive_int,
        default=DEFAULT_POSE_CHUNK_SIZE,
        help=f"Number of poses per input chunk (default: {DEFAULT_POSE_CHUNK_SIZE}).",
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


def _pose_chunk_coordinates(
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
            raise ValueError(
                f"Pose row {pose_offset + int(rows[0])}: "
                f"conformer index {conf} out of range"
            )

        rotamers = library.get_rotamers(conf)
        batch_rot = rot_ids[rows].astype(np.int64, copy=False)
        if batch_rot.size and int(batch_rot.max()) >= len(rotamers):
            bad_pos = int(np.flatnonzero(batch_rot >= len(rotamers))[0])
            bad_row = int(rows[bad_pos])
            bad_rot = int(batch_rot[bad_pos])
            raise ValueError(
                f"Pose row {pose_offset + bad_row}: rotamer index {bad_rot} "
                f"out of range for conformer {conf} (n={len(rotamers)})"
            )

        rotations = _rotamers_to_matrices(rotamers[batch_rot])
        transformed = np.einsum(
            "aj,njk->nak",
            coordinates[conf],
            rotations,
        )
        transformed += translations[rows, None, :]
        result[rows] = transformed

    return result


def _coordinates_to_parsed_pdb(
    pose_coordinates: np.ndarray,
    *,
    template: np.ndarray,
    model_offset: int = 0,
) -> np.ndarray:
    if pose_coordinates.ndim != 3 or pose_coordinates.shape[1:] != (len(template), 3):
        raise ValueError(
            "Pose coordinates must have shape "
            f"(nposes, {len(template)}, 3), got {pose_coordinates.shape}"
        )

    atoms = np.empty((len(pose_coordinates), len(template)), dtype=template.dtype)
    atom_indices = np.arange(1, len(template) + 1, dtype=template["index"].dtype)
    for i, coords in enumerate(pose_coordinates):
        model_index = model_offset + i + 1
        model_atoms = template.copy()
        model_atoms["model"] = model_index
        model_atoms["index"] = atom_indices
        model_atoms["x"] = coords[:, 0]
        model_atoms["y"] = coords[:, 1]
        model_atoms["z"] = coords[:, 2]
        atoms[i] = model_atoms
    return atoms


def _as_parsed_pdb_template(template: np.ndarray) -> np.ndarray:
    if (
        template.dtype == atomic_dtype
        and template.dtype.isalignedstruct == atomic_dtype.isalignedstruct
    ):
        return template
    if template.dtype.names != atomic_dtype.names:
        raise TypeError("Library template does not match parsed PDB fields")
    parsed = np.empty(len(template), dtype=atomic_dtype)
    for field in atomic_dtype.names:
        parsed[field] = template[field]
    return parsed


def materialize_pose_pdb(reader: PoseReader, *, library) -> np.ndarray:
    template = _as_parsed_pdb_template(library.template)
    coordinates = library.coordinates.astype(np.float32, copy=False)
    chunks = []
    offset = 0
    progress = tqdm(
        total=reader.selected_poses,
        desc="Build PDB models",
        unit="pose",
        unit_scale=True,
        disable=reader.selected_poses <= reader.rows_per_chunk,
        file=sys.stderr,
    )
    try:
        for chunk in reader.iter_chunks():
            pose_coordinates = _pose_chunk_coordinates(
                chunk,
                coordinates=coordinates,
                library=library,
                pose_offset=offset,
            )
            chunks.append(
                _coordinates_to_parsed_pdb(
                    pose_coordinates,
                    template=template,
                    model_offset=offset,
                )
            )
            offset += len(chunk)
            progress.update(len(chunk))
    finally:
        progress.close()

    if not chunks:
        return np.empty((0, len(template)), dtype=template.dtype)
    return np.concatenate(chunks, axis=0)


def write_pdb_text_with_library(reader: PoseReader, *, library) -> str:
    if reader.selected_poses > MAX_MODELS:
        raise ValueError(
            f"Refusing to write more than {MAX_MODELS} PDB models to stdout; "
            "use --pose-range to select fewer poses"
        )
    return write_multi_pdb(materialize_pose_pdb(reader, library=library))


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
            rows_per_chunk=args.chunksize,
        )
        sys.stdout.write(write_pdb_text_with_library(reader, library=library))
    finally:
        factory.unload_rotaconformers()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
