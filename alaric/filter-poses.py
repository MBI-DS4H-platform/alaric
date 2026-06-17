#!/usr/bin/env python3
"""Filter poses by energy threshold.

Usage:
  python filter-poses.py [--force] POSE_DIR ENERGY_FILE THRESHOLD OUT_DIR

ENERGY_FILE must contain one float per pose, in pose order.
Writes filtered poses and provenance.npy (uint64, 0-based indices into POSE_DIR).
"""

import sys
import argparse
import shutil
import numpy as np
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from organize import main as organize_main
from poses import PoseReader, PoseWriter

BUCKET_SIZE = 16


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

    energies = np.lib.format.open_memmap(Path(args.energy_file))
    mask = energies < args.threshold
    keep_idx = np.where(mask)[0]  # 0-based

    print(f"Total poses: {len(energies):,}", flush=True)
    print(f"Threshold:   {args.threshold}", flush=True)
    print(f"Passing:     {len(keep_idx):,}", flush=True)

    # Read and filter poses
    reader = PoseReader(pose_dir, rows_per_chunk=200_000)
    writer = PoseWriter(out_dir, bucket_size=BUCKET_SIZE, cache_poses=5_000_000)

    keep_set = set(keep_idx.tolist())
    out_conformers = []
    out_rotamers = []
    out_translations = []
    out_provenance = []

    cur = 0
    for chunk in reader.iter_chunks():
        n = len(chunk.conformers)
        for local_i in range(n):
            global_i = cur + local_i
            if global_i in keep_set:
                out_conformers.append(int(chunk.conformers[local_i]))
                out_rotamers.append(int(chunk.rotamers[local_i]))
                out_translations.append(chunk.translations_grid[local_i].tolist())
                out_provenance.append(global_i)
        cur += n

    conformers   = np.array(out_conformers,   dtype=np.uint16)
    rotamers     = np.array(out_rotamers,     dtype=np.uint16)
    translations = np.array(out_translations, dtype=np.int16)
    provenance   = np.array(out_provenance,   dtype=np.uint64)

    writer.add_chunk(conformers, rotamers, translations)
    written = writer.finish()
    print(f"Wrote {len(written)} arc shard(s)", flush=True)

    # Organize the unorganized shards produced by PoseWriter.
    organize_rc = organize_main([str(out_dir)])
    if organize_rc:
        raise RuntimeError(f"organize failed with exit code {organize_rc}")

    np.save(out_dir / "provenance.npy", provenance)
    print(f"Wrote provenance.npy: shape={provenance.shape}, dtype={provenance.dtype}", flush=True)
    print(f"Done → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
