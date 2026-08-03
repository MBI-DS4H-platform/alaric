#!/usr/bin/env python3
"""Select poses by global pose index into a new organized pose directory.

The pool and the index/mask file are read piecemeal (see ``pose_filter``): at pool scale
a boolean mask is one byte per pose (15 GB for 15 gigaposes) and the pool itself is far
larger, so neither is loaded whole.

For an index array the *order array* -- organized pose to its position in the request --
falls out of organize: each kept pose is tagged with its request position and organize
maps the tags through the same permutation it applies to the poses. That is also what
makes the result independent of how the filtering work was split.
"""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
import shutil
import sys

import numpy as np

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from npy_io import (
    NpyWriter,
    compressed_path,
    find_npy,
    load_npy,
    open_npy_mmap,
    read_npy_header,
)
from nprocs import default_nprocs
from organize import PROVENANCE_NAME, organize_pose_dir
from pose_filter import (
    DEFAULT_CACHE_POSES,
    DEFAULT_CHUNK_POSES,
    BoolMaskSelector,
    IndexSelector,
    filter_pose_dir,
    open_range_source,
)
from poses import PoseReader

_CONVERT_CHUNK = 4_000_000


def _uint_dtype_for_count(count: int) -> np.dtype:
    max_value = max(0, int(count) - 1)
    if max_value <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if max_value <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if max_value <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    return np.dtype(np.uint64)


def _load_index_array(path: Path, *, is_numpy: bool, base: int) -> np.ndarray:
    """The request as an index array; memory-mapped when it can be used as-is.

    A base-1 request has to be rebased, and a text request has to be parsed, so those
    are materialized -- both are caller-authored inputs, not pool-scale ones.
    """
    if not is_numpy:
        arr = np.asarray(np.loadtxt(path))
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if not np.issubdtype(arr.dtype, np.integer):
            if np.any(arr != np.floor(arr)):
                raise ValueError("pose indices must be integers")
            arr = arr.astype(np.int64)
    else:
        # Mapped, not loaded: the request can be pool-scale in its own right, and the
        # selector only ever binary-searches it.
        arr = load_npy(path, mmap=True)
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError("NumPy pose index files must have bool or integer dtype")

    if arr.ndim != 1:
        raise ValueError("pose index file must contain a 1D array")
    if not np.issubdtype(arr.dtype, np.unsignedinteger):
        minimum = 0 if base == 0 else 1
        if arr.size and int(np.asarray(arr).min()) < minimum:
            raise ValueError(f"pose indices must be >= {minimum} for base {base}")
    if base == 1:
        arr = np.asarray(arr).astype(np.int64, copy=False) - 1
    return arr


def build_selector(
    path: Path, *, base: int | None, nposes: int, stack: contextlib.ExitStack
):
    """Return ``(selector, input_was_mask)`` for the request in ``path``."""
    is_numpy = path.name.endswith(".npy") or path.name.endswith(".npy.zst")
    default_base = 0 if is_numpy else 1
    base = default_base if base is None else int(base)
    if base not in {0, 1}:
        raise ValueError("--base must be 0 or 1")

    if is_numpy:
        shape, dtype = read_npy_header(path)
        if dtype == np.dtype(np.bool_):
            if len(shape) != 1:
                raise ValueError("pose index file must contain a 1D array")
            if shape[0] != nposes:
                raise ValueError(
                    f"boolean pose mask length {shape[0]} does not match "
                    f"pose count {nposes}"
                )
            # A mask is one byte per pose, so it is range-read rather than loaded.
            return BoolMaskSelector(open_range_source(path, stack)), True

    indices = _load_index_array(path, is_numpy=is_numpy, base=base)
    return IndexSelector(indices, nposes), False


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


def _publish_order_array(
    out_dir: Path, target: Path, *, kept: int, compress: bool
) -> Path | None:
    """Turn organize's tag provenance into the order array.

    organize records the tags as ``provenance.npy`` in uint32; the order array is the
    same values in the narrowest dtype that fits, which is what this wrote before the
    tags were routed through organize -- so the result stays byte-identical.
    """
    produced = find_npy(out_dir / PROVENANCE_NAME)
    if produced is None:
        return None
    dtype = _uint_dtype_for_count(kept)
    destination = compressed_path(target) if compress else Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if dtype == np.dtype(np.uint32):
        # shutil.move, not Path.replace: --order-array may point at another filesystem.
        shutil.move(str(produced), str(destination))
        return destination
    with open_npy_mmap(produced) as tags:
        with NpyWriter(
            target, dtype=dtype, shape=tags.shape, compress=compress
        ) as writer:
            for start in range(0, len(tags), _CONVERT_CHUNK):
                writer.write(tags[start : start + _CONVERT_CHUNK].astype(dtype))
    produced.unlink()
    return destination


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
    parser.add_argument(
        "--nprocs",
        type=int,
        default=default_nprocs(),
        help="Worker processes; the pool is split per organized input file.",
    )
    parser.add_argument(
        "--chunk-poses",
        type=int,
        default=DEFAULT_CHUNK_POSES,
        help="Pose rows worked on per step. The filesystem request size is set "
        "separately (poses.ARC_STREAM_READ_SIZE, npy_io.RANGE_READ_BLOCK).",
    )
    parser.add_argument(
        "--cache-poses",
        type=int,
        default=DEFAULT_CACHE_POSES,
        help="Selected poses buffered before a shard is flushed, shared across workers.",
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
    with contextlib.ExitStack() as stack:
        selector, input_was_mask = build_selector(
            index_path, base=args.base, nposes=nposes, stack=stack
        )
        print(f"Selecting from {nposes:,} pose(s)", flush=True)

        stats = filter_pose_dir(
            pose_dir,
            out_dir,
            selector,
            compress=bool(args.compress),
            nprocs=int(args.nprocs),
            chunk_poses=int(args.chunk_poses),
            cache_poses=int(args.cache_poses),
        )
    print(f"Selected {stats.kept_poses:,} pose(s)", flush=True)
    print(f"Wrote {stats.shards} unorganized shard(s)", flush=True)

    organize_pose_dir(out_dir, compress=bool(args.compress))

    if not input_was_mask:
        target = args.order_array or (out_dir / "order-array.npy")
        written = _publish_order_array(
            out_dir, Path(target), kept=stats.kept_poses, compress=bool(args.compress)
        )
        if written is not None:
            print(f"Wrote order array: {written}", flush=True)
    print(f"Done: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
