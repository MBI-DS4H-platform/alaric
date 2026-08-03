#!/usr/bin/env python3
"""Filter poses by energy threshold.

Usage:
  python filter-poses.py [--force] [--compress] POSE_DIR ENERGY_FILE THRESHOLD OUT_DIR

ENERGY_FILE must contain one float per pose, in pose order.
Writes filtered poses and provenance.npy (uint64, 0-based indices into POSE_DIR),
zstd-compressed with --compress.

Both the pool and the energy file are read piecemeal (see ``pose_filter``): at pool
scale neither fits in memory -- 15 gigaposes is a 60 GB float32 energy file -- so peak
memory depends on the chunk size and the number of poses *kept*, not on the pool size.
"""

import sys
import argparse
import contextlib
import shutil
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from nprocs import default_nprocs  # noqa: E402
from organize import main as organize_main  # noqa: E402
from pose_filter import (  # noqa: E402
    DEFAULT_CACHE_POSES,
    DEFAULT_CHUNK_POSES,
    ScoreThresholdSelector,
    filter_pose_dir,
    open_range_source,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("pose_dir")
    p.add_argument("energy_file")
    p.add_argument("threshold", type=float)
    p.add_argument("out_dir")
    p.add_argument(
        "--force",
        action="store_true",
        help="Remove OUT_DIR first if it already exists and is non-empty.",
    )
    p.add_argument(
        "--compress",
        action="store_true",
        help=(
            "Write zstd-compressed output: organized poses-*.arc.zst instead of "
            "poses-*.arc, and provenance.npy.zst instead of provenance.npy."
        ),
    )
    p.add_argument(
        "--nprocs",
        type=int,
        default=default_nprocs(),
        help="Worker processes; the pool is split per organized input file.",
    )
    p.add_argument(
        "--chunk-poses",
        type=int,
        default=DEFAULT_CHUNK_POSES,
        help="Pose rows worked on per step. The filesystem request size is set "
        "separately (poses.ARC_STREAM_READ_SIZE, npy_io.RANGE_READ_BLOCK).",
    )
    p.add_argument(
        "--cache-poses",
        type=int,
        default=DEFAULT_CACHE_POSES,
        help="Kept poses buffered before a shard is flushed, shared across workers.",
    )
    return p.parse_args()


def prepare_output_dir(out_dir: Path, *, force: bool) -> None:
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


def main():
    args = parse_args()
    pose_dir = Path(args.pose_dir)
    out_dir = Path(args.out_dir)
    prepare_output_dir(out_dir, force=bool(args.force))

    with contextlib.ExitStack() as stack:
        energies = open_range_source(args.energy_file, stack)
        selector = ScoreThresholdSelector(energies, args.threshold)
        print(f"Total poses: {len(selector):,}", flush=True)
        print(f"Threshold:   {args.threshold}", flush=True)

        stats = filter_pose_dir(
            pose_dir,
            out_dir,
            selector,
            provenance_path=out_dir / "provenance.npy",
            compress=bool(args.compress),
            nprocs=int(args.nprocs),
            chunk_poses=int(args.chunk_poses),
            cache_poses=int(args.cache_poses),
        )
    print(f"Passing:     {stats.kept_poses:,}", flush=True)
    print(f"Wrote {stats.shards} arc shard(s)", flush=True)

    # Organize the unorganized shards produced by the filter pass.
    organize_argv = [str(out_dir)]
    if args.compress:
        organize_argv.append("--compress")
    organize_rc = organize_main(organize_argv)
    if organize_rc:
        raise RuntimeError(f"organize failed with exit code {organize_rc}")

    print(f"Done → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
