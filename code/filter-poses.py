#!/usr/bin/env python3
"""Filter poses by energy threshold.

Usage:
  python filter-poses.py POSE_DIR THRESHOLD OUT_DIR

POSE_DIR must contain energies.npy (one float per pose, in pose order).
Writes filtered poses and provenance.npy (uint64, 0-based indices into POSE_DIR).
"""

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "code"))

from poses import PoseReader, PoseWriter
import subprocess, os

BUCKET_SIZE = 16


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("pose_dir")
    p.add_argument("threshold", type=float)
    p.add_argument("out_dir")
    return p.parse_args()


def main():
    args = parse_args()
    pose_dir = Path(args.pose_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    energies = np.load(pose_dir / "energies.npy")
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

    # Organize
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent / "code")
    subprocess.run(
        [sys.executable, str(Path(__file__).parent / "code" / "organize.py"), str(out_dir)],
        env=env, check=True
    )

    np.save(out_dir / "provenance.npy", provenance)
    print(f"Wrote provenance.npy: shape={provenance.shape}, dtype={provenance.dtype}", flush=True)
    print(f"Done → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
