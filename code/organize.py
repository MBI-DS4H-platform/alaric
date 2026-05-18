from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

from poses import (
    MAX_NO,
    MAX_NP,
    discover_organized,
    discover_unorganized,
    read_arc_file,
    write_arc_file,
)


DONE_MARKER = ".ORGANIZED-DONE"


MiniBuckets = dict[
    tuple[int, int, int], dict[tuple[int, int, int], list[np.ndarray]]
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="organize",
        description="Canonicalize unorganized alaric .arc pose shards.",
    )
    parser.add_argument("pose_dir", metavar="POSE_DIR")
    parser.add_argument("--capacity", type=int, default=2_000_000_000)
    parser.add_argument("--max-poses-per-file", type=int, default=100_000_000)
    parser.add_argument("--nprocs", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    return parser


def _load_unorganized(paths: list[Path]) -> MiniBuckets:
    buckets: MiniBuckets = defaultdict(lambda: defaultdict(list))
    for path in paths:
        M, O, C, P = read_arc_file(path)
        if int(C.max(initial=0)) > MAX_NP:
            raise ValueError(f"mini-bucket count exceeds uint32 in {path}")
        if len(P) == 0:
            continue

        offset_indices = P[:, 2].astype(np.int64, copy=False)
        order = np.argsort(offset_indices, kind="stable")
        sorted_indices = offset_indices[order]
        boundaries = np.flatnonzero(sorted_indices[1:] != sorted_indices[:-1]) + 1
        starts = np.concatenate(([0], boundaries))
        stops = np.concatenate((boundaries, [len(order)]))
        m_key = tuple(int(x) for x in M)

        for start, stop in zip(starts, stops):
            pose_rows = order[start:stop]
            offset = O[int(sorted_indices[start])]
            rows = P[pose_rows, 0:2].astype(np.uint16, copy=True)
            buckets[m_key][tuple(int(x) for x in offset)].append(rows)
    return buckets


def _sorted_conf_rot(parts: list[np.ndarray]) -> np.ndarray:
    rows = np.concatenate(parts) if len(parts) > 1 else parts[0]
    if len(rows) <= 1:
        return rows.astype(np.uint16, copy=False)
    order = np.lexsort((rows[:, 1], rows[:, 0]))
    return rows[order].astype(np.uint16, copy=False)


def _write_current(
    pose_dir: Path,
    file_index: int,
    M: tuple[int, int, int],
    offsets: list[tuple[int, int, int]],
    counts: list[int],
    pose_parts: list[np.ndarray],
) -> int:
    if not pose_parts:
        return file_index
    O = np.array(offsets, dtype=np.int8)
    C = np.array(counts, dtype=np.uint32)
    P = np.concatenate(pose_parts).astype(np.uint16, copy=False)
    write_arc_file(pose_dir / f"poses-{file_index}.arc", np.array(M, dtype=np.int8), O, C, P)
    return file_index + 1


def _write_organized(
    pose_dir: Path,
    buckets: MiniBuckets,
    *,
    max_poses_per_file: int,
) -> int:
    file_index = 1
    for M in sorted(buckets):
        current_offsets: list[tuple[int, int, int]] = []
        current_counts: list[int] = []
        current_parts: list[np.ndarray] = []
        current_nposes = 0

        for offset in sorted(buckets[M]):
            rows = _sorted_conf_rot(buckets[M][offset])
            if len(rows) > MAX_NP:
                raise ValueError(f"offset {offset} in M {M} has more than 2**32 - 1 poses")

            start = 0
            while start < len(rows):
                if (
                    current_parts
                    and (
                        current_nposes >= max_poses_per_file
                        or len(current_offsets) >= MAX_NO
                    )
                ):
                    file_index = _write_current(
                        pose_dir,
                        file_index,
                        M,
                        current_offsets,
                        current_counts,
                        current_parts,
                    )
                    current_offsets = []
                    current_counts = []
                    current_parts = []
                    current_nposes = 0

                room_by_pose = max_poses_per_file - current_nposes
                room_by_u32 = MAX_NP - current_nposes
                room = min(room_by_pose, room_by_u32)
                if room <= 0:
                    file_index = _write_current(
                        pose_dir,
                        file_index,
                        M,
                        current_offsets,
                        current_counts,
                        current_parts,
                    )
                    current_offsets = []
                    current_counts = []
                    current_parts = []
                    current_nposes = 0
                    continue

                if len(current_offsets) >= MAX_NO:
                    file_index = _write_current(
                        pose_dir,
                        file_index,
                        M,
                        current_offsets,
                        current_counts,
                        current_parts,
                    )
                    current_offsets = []
                    current_counts = []
                    current_parts = []
                    current_nposes = 0
                    continue

                take = min(room, len(rows) - start)
                offset_index = len(current_offsets)
                P = np.empty((take, 3), dtype=np.uint16)
                P[:, 0:2] = rows[start : start + take]
                P[:, 2] = offset_index
                current_offsets.append(offset)
                current_counts.append(take)
                current_parts.append(P)
                current_nposes += take
                start += take

                if start < len(rows):
                    file_index = _write_current(
                        pose_dir,
                        file_index,
                        M,
                        current_offsets,
                        current_counts,
                        current_parts,
                    )
                    current_offsets = []
                    current_counts = []
                    current_parts = []
                    current_nposes = 0

        file_index = _write_current(
            pose_dir, file_index, M, current_offsets, current_counts, current_parts
        )
    return file_index - 1


def _run(args: argparse.Namespace) -> int:
    if args.capacity <= 0:
        raise ValueError("--capacity must be positive")
    if args.max_poses_per_file <= 0 or args.max_poses_per_file > MAX_NP:
        raise ValueError("--max-poses-per-file must be in 1..2**32-1")
    if args.nprocs <= 0:
        raise ValueError("--nprocs must be positive")

    pose_dir = Path(args.pose_dir)
    pose_dir.mkdir(parents=True, exist_ok=True)
    marker = pose_dir / DONE_MARKER
    unorganized = discover_unorganized(pose_dir)
    organized = discover_organized(pose_dir)

    if marker.exists():
        for path in unorganized:
            path.unlink()
        marker.unlink()
        return 0

    if not unorganized:
        return 0
    if organized:
        raise ValueError(
            f"{pose_dir} contains both organized poses-*.arc and unorganized-*.arc* files"
        )

    buckets = _load_unorganized(unorganized)
    if buckets:
        _write_organized(
            pose_dir,
            buckets,
            max_poses_per_file=min(int(args.capacity), int(args.max_poses_per_file)),
        )

    marker.touch()
    for path in unorganized:
        path.unlink()
    marker.unlink()
    return 0


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
