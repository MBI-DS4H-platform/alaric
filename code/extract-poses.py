#!/usr/bin/env python3
"""Extract specific poses (by 1-based global index) into a new pose directory."""

import sys
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

from poses import PoseReader, PoseWriter

BUCKET_SIZE = 16
CHUNK_SIZE = 1_000_000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("pose_dir", help="Source pose directory")
    p.add_argument("out_dir", help="Output pose directory")
    p.add_argument("--pose-indices", nargs="+", type=int, required=True,
                   help="1-based global pose indices to extract")
    return p.parse_args()


def main():
    args = parse_args()
    pose_dir = Path(args.pose_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    indices_1based = sorted(args.pose_indices)
    print(f"Extracting {len(indices_1based)} poses from {pose_dir} → {out_dir}", flush=True)

    by_chunk = defaultdict(list)
    for p in indices_1based:
        by_chunk[(p - 1) // CHUNK_SIZE].append(p)

    total_poses_in_source = PoseReader.get_nposes(pose_dir)
    writer = PoseWriter(out_dir, bucket_size=BUCKET_SIZE)

    all_conformers = []
    all_rotamers = []
    all_translations = []

    for chunk_idx in sorted(by_chunk):
        chunk_start = chunk_idx * CHUNK_SIZE + 1
        chunk_end = min((chunk_idx + 1) * CHUNK_SIZE, total_poses_in_source)
        wanted = sorted(by_chunk[chunk_idx])
        wanted_set = set(wanted)
        print(f"  chunk {chunk_idx} ({chunk_start}-{chunk_end}): {len(wanted)} poses", flush=True)

        reader = PoseReader(pose_dir, pose_range=(chunk_start, chunk_end))
        cur_global = chunk_start
        found = 0
        for pchunk in reader.iter_chunks():
            n = len(pchunk.conformers)
            for local_i in range(n):
                global_i = cur_global + local_i
                if global_i in wanted_set:
                    all_conformers.append(int(pchunk.conformers[local_i]))
                    all_rotamers.append(int(pchunk.rotamers[local_i]))
                    all_translations.append(pchunk.translations_grid[local_i].tolist())
                    found += 1
            cur_global += n
            if found == len(wanted):
                break

        print(f"    found {found}/{len(wanted)}", flush=True)

    conformers = np.array(all_conformers, dtype=np.uint16)
    rotamers = np.array(all_rotamers, dtype=np.uint16)
    translations = np.array(all_translations, dtype=np.int16)

    print(f"Writing {len(conformers)} poses to {out_dir}...", flush=True)
    writer.add_chunk(conformers, rotamers, translations)
    written = writer.finish()
    print(f"Wrote {len(written)} arc file(s): {[p.name for p in written]}", flush=True)
    print(f"Total poses written: {writer.total_poses}", flush=True)


if __name__ == "__main__":
    main()
