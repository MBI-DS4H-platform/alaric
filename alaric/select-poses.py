#!/usr/bin/env python3
"""Select poses by global pose index into a new organized pose directory."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import numpy as np

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from npy_io import compress_npy_file, find_npy, load_npy
from organize import organize_pose_dir
from poses import DEFAULT_BUCKET_SIZE, PoseReader, PoseWriter, select_pose_indices


def _load_pose_indices(
    path: Path,
    *,
    base: int | None,
    nposes: int,
) -> tuple[np.ndarray, bool]:
    is_numpy = path.name.endswith(".npy") or path.name.endswith(".npy.zst")
    default_base = 0 if is_numpy else 1
    base = default_base if base is None else int(base)
    if base not in {0, 1}:
        raise ValueError("--base must be 0 or 1")

    arr = load_npy(path) if is_numpy else np.loadtxt(path)
    arr = np.asarray(arr)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError("pose index file must contain a 1D array")

    if is_numpy and arr.dtype == np.bool_:
        if len(arr) != nposes:
            raise ValueError(
                f"boolean pose mask length {len(arr)} does not match pose count {nposes}"
            )
        return (
            np.where(arr)[0].astype(_uint_dtype_for_count(nposes), copy=False),
            True,
        )

    if is_numpy and not np.issubdtype(arr.dtype, np.integer):
        raise ValueError("NumPy pose index files must have bool or integer dtype")

    if not np.issubdtype(arr.dtype, np.integer):
        if np.any(arr != np.floor(arr)):
            raise ValueError("pose indices must be integers")
        arr = arr.astype(np.int64)

    if not (is_numpy and np.issubdtype(arr.dtype, np.unsignedinteger)):
        minimum = 0 if base == 0 else 1
        if arr.size and int(arr.min()) < minimum:
            raise ValueError(f"pose indices must be >= {minimum} for base {base}")

    indices = arr.astype(np.uint64, copy=False)
    if base == 1:
        indices = indices - np.uint64(1)
    if indices.size and int(indices.max()) >= nposes:
        raise ValueError(
            f"pose index {int(indices.max()) + base} exceeds number of poses ({nposes})"
        )
    return indices, False


def _uint_dtype_for_count(count: int) -> np.dtype:
    max_value = max(0, int(count) - 1)
    if max_value <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if max_value <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if max_value <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    return np.dtype(np.uint64)


def _prepare_output_dir(out_dir: Path, *, force: bool) -> None:
    if out_dir.exists():
        if not out_dir.is_dir():
            raise ValueError(f"output path is not a directory: {out_dir}")
        if any(out_dir.iterdir()):
            if not force:
                raise ValueError(
                    f"output directory is not empty: {out_dir}; use --force to replace it"
                )
            shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select indexed poses into a new organized pose directory.",
    )
    parser.add_argument("pose_dir", help="Source organized pose directory")
    parser.add_argument("pose_index_file", help="Pose index array/text file")
    parser.add_argument("out_dir", help="Output pose directory")
    parser.add_argument(
        "--base",
        type=int,
        choices=(0, 1),
        help="Index base. Defaults to 0 for NumPy files and 1 for text files.",
    )
    parser.add_argument(
        "--order-array",
        type=Path,
        help="Order-array output path. Defaults to OUT_DIR/order-array.npy.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove OUT_DIR first if it already exists and is non-empty.",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help=(
            "Write zstd-compressed output: organized poses-*.arc.zst instead of "
            "poses-*.arc, and a compressed order array."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pose_dir = Path(args.pose_dir)
    # Accept the logical name of a compressed index array (e.g. a mask written as
    # mask.npy.zst but referenced as mask.npy by the generated scripts).
    index_path = find_npy(args.pose_index_file) or Path(args.pose_index_file)
    out_dir = Path(args.out_dir)
    _prepare_output_dir(out_dir, force=bool(args.force))

    nposes = PoseReader.get_nposes(pose_dir)
    indices, input_was_mask = _load_pose_indices(
        index_path,
        base=args.base,
        nposes=nposes,
    )
    print(f"Selecting {len(indices):,} pose(s) from {nposes:,}", flush=True)

    chunk = select_pose_indices(pose_dir, indices)
    writer = PoseWriter(
        out_dir,
        bucket_size=DEFAULT_BUCKET_SIZE,
        cache_poses=max(1, len(indices)),
    )
    writer.add_chunk(chunk.conformers, chunk.rotamers, chunk.translations_grid)
    written = writer.finish()
    print(f"Wrote {len(written)} unorganized shard(s)", flush=True)

    order_array_path = None
    if not input_was_mask:
        order_array_path = args.order_array or (out_dir / "order-array.npy")
    organize_pose_dir(
        out_dir,
        compress=bool(args.compress),
        return_order_array=order_array_path is not None,
        order_array_path=order_array_path,
    )
    if order_array_path is not None:
        # organize builds the order array as a scattered memmap (several processes write
        # into it), so it can only be compressed once it is complete.
        if args.compress:
            order_array_path = compress_npy_file(order_array_path)
        print(f"Wrote order array: {order_array_path}", flush=True)
    print(f"Done: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
