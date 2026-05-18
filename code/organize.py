from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import os
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
WriteTask = tuple[
    int,
    tuple[int, int, int],
    list[tuple[tuple[int, int, int], int, int]],
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="organize",
        description="Canonicalize unorganized alaric .arc pose shards.",
    )
    parser.add_argument("pose_dir", metavar="POSE_DIR")
    parser.add_argument("--capacity", type=int, default=2_000_000_000)
    parser.add_argument("--max-poses-per-file", type=int, default=100_000_000)
    parser.add_argument("--nprocs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--debug", action="store_true")
    return parser


def _load_unorganized_path(path: Path) -> MiniBuckets:
    buckets: MiniBuckets = defaultdict(lambda: defaultdict(list))
    M, O, C, P = read_arc_file(path)
    if int(C.max(initial=0)) > MAX_NP:
        raise ValueError(f"mini-bucket count exceeds uint32 in {path}")
    if len(P) == 0:
        return buckets

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


def _merge_buckets(target: MiniBuckets, source: MiniBuckets) -> None:
    for M, offsets in source.items():
        target_offsets = target[M]
        for offset, parts in offsets.items():
            target_offsets[offset].extend(parts)


def _load_unorganized(paths: list[Path], *, nprocs: int = 1) -> MiniBuckets:
    buckets: MiniBuckets = defaultdict(lambda: defaultdict(list))
    if nprocs <= 1 or len(paths) <= 1:
        for path in paths:
            _merge_buckets(buckets, _load_unorganized_path(path))
        return buckets

    with ThreadPoolExecutor(max_workers=min(nprocs, len(paths))) as executor:
        for loaded in executor.map(_load_unorganized_path, paths):
            _merge_buckets(buckets, loaded)
    return buckets


def _sorted_conf_rot(parts: list[np.ndarray]) -> np.ndarray:
    rows = np.concatenate(parts) if len(parts) > 1 else parts[0]
    if len(rows) <= 1:
        return rows.astype(np.uint16, copy=False)
    packed = (
        (rows[:, 0].astype(np.uint32, copy=False) << 16)
        | rows[:, 1].astype(np.uint32, copy=False)
    )
    packed.sort()
    sorted_rows = np.empty((len(packed), 2), dtype=np.uint16)
    sorted_rows[:, 0] = (packed >> 16).astype(np.uint16, copy=False)
    sorted_rows[:, 1] = packed.astype(np.uint16, copy=False)
    return sorted_rows


def _plan_write_tasks(
    buckets: MiniBuckets,
    *,
    max_poses_per_file: int,
) -> list[WriteTask]:
    tasks: list[WriteTask] = []
    file_index = 1
    for M in sorted(buckets):
        current_items: list[tuple[tuple[int, int, int], int, int]] = []
        current_nposes = 0

        for offset in sorted(buckets[M]):
            row_count = sum(len(part) for part in buckets[M][offset])
            if row_count > MAX_NP:
                raise ValueError(f"offset {offset} in M {M} has more than 2**32 - 1 poses")

            start = 0
            while start < row_count:
                if (
                    current_items
                    and (
                        current_nposes >= max_poses_per_file
                        or len(current_items) >= MAX_NO
                    )
                ):
                    tasks.append((file_index, M, current_items))
                    file_index += 1
                    current_items = []
                    current_nposes = 0

                room_by_pose = max_poses_per_file - current_nposes
                room_by_u32 = MAX_NP - current_nposes
                room = min(room_by_pose, room_by_u32)
                if room <= 0:
                    tasks.append((file_index, M, current_items))
                    file_index += 1
                    current_items = []
                    current_nposes = 0
                    continue

                if len(current_items) >= MAX_NO:
                    tasks.append((file_index, M, current_items))
                    file_index += 1
                    current_items = []
                    current_nposes = 0
                    continue

                take = min(room, row_count - start)
                current_items.append((offset, start, take))
                current_nposes += take
                start += take

                if start < row_count:
                    tasks.append((file_index, M, current_items))
                    file_index += 1
                    current_items = []
                    current_nposes = 0

        if current_items:
            tasks.append((file_index, M, current_items))
            file_index += 1
    return tasks


def _write_task(pose_dir: Path, buckets: MiniBuckets, task: WriteTask) -> int:
    file_index, M, items = task
    offsets: list[tuple[int, int, int]] = []
    counts: list[int] = []
    pose_parts: list[np.ndarray] = []

    for offset_index, (offset, start, take) in enumerate(items):
        rows = _sorted_conf_rot(buckets[M][offset])
        P = np.empty((take, 3), dtype=np.uint16)
        P[:, 0:2] = rows[start : start + take]
        P[:, 2] = offset_index
        offsets.append(offset)
        counts.append(take)
        pose_parts.append(P)

    _write_current(pose_dir, file_index, M, offsets, counts, pose_parts)
    return file_index


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
    nprocs: int = 1,
) -> int:
    tasks = _plan_write_tasks(buckets, max_poses_per_file=max_poses_per_file)
    if not tasks:
        return 0

    if nprocs <= 1:
        for task in tasks:
            _write_task(pose_dir, buckets, task)
    else:
        with ThreadPoolExecutor(max_workers=min(nprocs, len(tasks))) as executor:
            for _ in executor.map(lambda task: _write_task(pose_dir, buckets, task), tasks):
                pass
    return len(tasks)


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

    buckets = _load_unorganized(unorganized, nprocs=int(args.nprocs))
    if buckets:
        _write_organized(
            pose_dir,
            buckets,
            max_poses_per_file=min(int(args.capacity), int(args.max_poses_per_file)),
            nprocs=int(args.nprocs),
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
