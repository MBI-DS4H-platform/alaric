from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
from typing import Sequence

import numpy as np

from poses import (
    MAX_NO,
    MAX_NP,
    discover_organized,
    discover_unorganized,
    read_arc_file,
    read_arc_offsets,
    write_arc_file,
)


DONE_MARKER = ".ORGANIZED-DONE"


@dataclass(frozen=True)
class SourceMeta:
    path: Path
    M: tuple[int, int, int]
    O: np.ndarray
    C: np.ndarray
    nP: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="organize",
        description="Canonicalize unorganized alaric .arc pose shards.",
    )
    parser.add_argument("pose_dir", metavar="POSE_DIR")
    parser.add_argument("--capacity", type=int, default=2_000_000_000)
    parser.add_argument("--max-poses-per-file", type=int, default=100_000_000)
    parser.add_argument("--nprocs", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Write organized poses-*.arc.zst files by streaming completed temp .arc files through zstd.",
    )
    parser.add_argument("--debug", action="store_true")
    return parser


@dataclass(frozen=True)
class OutputLayout:
    file_index: int
    M: tuple[int, int, int]
    offsets: list[tuple[int, int, int]]
    counts: list[int]


@dataclass(frozen=True)
class Segment:
    layout_id: int
    offset_id: int
    start: int
    count: int


def _read_source_meta(path: Path) -> SourceMeta:
    M, O, C, nP = read_arc_offsets(path)
    if int(C.max(initial=0)) > MAX_NP:
        raise ValueError(f"mini-bucket count exceeds uint32 in {path}")
    return SourceMeta(path=path, M=tuple(int(x) for x in M), O=O, C=C, nP=nP)


def _read_sources(paths: list[Path], *, nprocs: int = 1) -> list[SourceMeta]:
    if nprocs <= 1 or len(paths) <= 1:
        return [_read_source_meta(path) for path in paths]

    with ThreadPoolExecutor(max_workers=min(nprocs, len(paths))) as executor:
        return list(executor.map(_read_source_meta, paths))


def _build_layouts(
    sources: list[SourceMeta],
    *,
    max_poses_per_file: int,
) -> tuple[
    list[OutputLayout],
    dict[tuple[tuple[int, int, int], tuple[int, int, int]], list[Segment]],
]:
    counts_by_offset: dict[
        tuple[int, int, int],
        dict[tuple[int, int, int], int],
    ] = defaultdict(lambda: defaultdict(int))
    for source in sources:
        offsets = counts_by_offset[source.M]
        for offset_row, count in zip(source.O, source.C):
            offset = tuple(int(x) for x in offset_row)
            total = offsets[offset] + int(count)
            if total > MAX_NP:
                raise ValueError(
                    f"offset {offset} in M {source.M} has more than 2**32 - 1 poses"
                )
            offsets[offset] = total

    layouts: list[OutputLayout] = []
    destination: dict[
        tuple[tuple[int, int, int], tuple[int, int, int]],
        list[Segment],
    ] = {}
    file_index = 1
    for M in sorted(counts_by_offset):
        current_offsets: list[tuple[int, int, int]] = []
        current_counts: list[int] = []
        current_nposes = 0

        def close_current() -> None:
            nonlocal file_index, current_offsets, current_counts, current_nposes
            if not current_offsets:
                return
            layouts.append(OutputLayout(file_index, M, current_offsets, current_counts))
            file_index += 1
            current_offsets = []
            current_counts = []
            current_nposes = 0

        for offset in sorted(counts_by_offset[M]):
            count = counts_by_offset[M][offset]
            key = (M, offset)
            segments: list[Segment] = []
            if count <= max_poses_per_file:
                if (
                    current_offsets
                    and (
                        current_nposes + count > max_poses_per_file
                        or len(current_offsets) >= MAX_NO
                    )
                ):
                    close_current()

                if len(current_offsets) >= MAX_NO:
                    close_current()

                offset_id = len(current_offsets)
                current_offsets.append(offset)
                current_counts.append(count)
                segments.append(
                    Segment(
                        layout_id=len(layouts),
                        offset_id=offset_id,
                        start=0,
                        count=count,
                    )
                )
                current_nposes += count
            else:
                close_current()
                start = 0
                while start < count:
                    take = min(max_poses_per_file, count - start)
                    current_offsets.append(offset)
                    current_counts.append(take)
                    segments.append(
                        Segment(
                            layout_id=len(layouts),
                            offset_id=0,
                            start=start,
                            count=take,
                        )
                    )
                    current_nposes = take
                    start += take
                    close_current()

            destination[key] = segments

        close_current()
    return layouts, destination


def _scatter_source(
    source: SourceMeta,
    destination: dict[
        tuple[tuple[int, int, int], tuple[int, int, int]],
        list[Segment],
    ],
    temp_paths: list[Path],
    temp_locks: list[threading.Lock],
    split_ids: dict[tuple[tuple[int, int, int], tuple[int, int, int]], int],
    split_paths: list[Path],
    split_locks: list[threading.Lock],
) -> None:
    M_arr, O, C, P = read_arc_file(source.path)
    M = tuple(int(x) for x in M_arr)
    if M != source.M:
        raise ValueError(f"M changed while reading {source.path}")
    if len(P) == 0:
        return

    output_file = np.empty(len(O), dtype=np.uint16)
    output_offset = np.empty(len(O), dtype=np.uint16)
    split_local_ids = np.full(len(O), -1, dtype=np.int32)
    for local_index, offset_row in enumerate(O):
        key = (M, tuple(int(x) for x in offset_row))
        segments = destination[key]
        if len(segments) == 1:
            segment = segments[0]
            output_file[local_index] = segment.layout_id
            output_offset[local_index] = segment.offset_id
        else:
            split_local_ids[local_index] = split_ids[key]

    row_split_ids = split_local_ids[P[:, 2]]
    split_mask = row_split_ids >= 0
    for split_id in np.unique(row_split_ids[split_mask]):
        mask = row_split_ids == split_id
        records = P[mask, 0:2].astype(np.uint16, copy=True)
        split_index = int(split_id)
        with split_locks[split_index]:
            with split_paths[split_index].open("ab") as handle:
                handle.write(records.tobytes(order="C"))

    keep_mask = ~split_mask
    if not np.any(keep_mask):
        return
    kept_offsets = P[keep_mask, 2]
    file_ids = output_file[kept_offsets]
    offset_ids = output_offset[kept_offsets]
    conf_rot = P[keep_mask, 0:2]
    for file_id in np.unique(file_ids):
        mask = file_ids == file_id
        n = int(mask.sum())
        if n == 0:
            continue
        records = np.empty((n, 3), dtype=np.uint16)
        records[:, 0] = offset_ids[mask]
        records[:, 1:3] = conf_rot[mask]
        file_index = int(file_id)
        with temp_locks[file_index]:
            with temp_paths[file_index].open("ab") as handle:
                handle.write(records.tobytes(order="C"))


def _distribute_split_offset(
    split_path: Path,
    segments: list[Segment],
    temp_paths: list[Path],
    temp_locks: list[threading.Lock],
) -> None:
    rows = np.fromfile(split_path, dtype=np.uint16).reshape(-1, 2)
    expected = sum(segment.count for segment in segments)
    if len(rows) != expected:
        raise ValueError(f"{split_path} has {len(rows)} rows, expected {expected}")
    packed = (
        (rows[:, 0].astype(np.uint32, copy=False) << 16)
        | rows[:, 1].astype(np.uint32, copy=False)
    )
    del rows
    packed.sort()

    for segment in segments:
        chunk = packed[segment.start : segment.start + segment.count]
        records = np.empty((len(chunk), 3), dtype=np.uint16)
        records[:, 0] = segment.offset_id
        records[:, 1] = (chunk >> 16).astype(np.uint16, copy=False)
        records[:, 2] = chunk.astype(np.uint16, copy=False)
        with temp_locks[segment.layout_id]:
            with temp_paths[segment.layout_id].open("ab") as handle:
                handle.write(records.tobytes(order="C"))


def _compress_arc_file(source: Path, dest: Path) -> None:
    try:
        import zstandard as zstd
    except ImportError as exc:
        raise ImportError("zstandard is required for --compress") from exc

    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f"{dest.name}.",
        suffix=".tmp",
        dir=dest.parent,
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
        with source.open("rb") as arc:
            with zstd.ZstdCompressor().stream_writer(handle) as compressor:
                shutil.copyfileobj(arc, compressor, length=1024 * 1024)
    try:
        tmp_path.replace(dest)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _write_layout_from_temp(
    pose_dir: Path,
    layout: OutputLayout,
    temp_path: Path,
    arc_temp_dir: Path,
    *,
    compress: bool,
) -> int:
    expected = int(sum(layout.counts))
    records = np.fromfile(temp_path, dtype=np.uint16).reshape(-1, 3)
    if len(records) != expected:
        raise ValueError(
            f"{temp_path} has {len(records)} records, expected {expected}"
        )
    keys = (
        (records[:, 0].astype(np.uint64, copy=False) << 32)
        | (records[:, 1].astype(np.uint64, copy=False) << 16)
        | records[:, 2].astype(np.uint64, copy=False)
    )
    del records
    keys.sort()

    P = np.empty((len(keys), 3), dtype=np.uint16)
    P[:, 0] = ((keys >> 16) & 0xFFFF).astype(np.uint16, copy=False)
    P[:, 1] = (keys & 0xFFFF).astype(np.uint16, copy=False)
    P[:, 2] = (keys >> 32).astype(np.uint16, copy=False)
    O = np.array(layout.offsets, dtype=np.int8)
    C = np.array(layout.counts, dtype=np.uint32)
    arc_path = (
        arc_temp_dir / f"poses-{layout.file_index}.arc"
        if compress
        else pose_dir / f"poses-{layout.file_index}.arc"
    )
    write_arc_file(
        arc_path,
        np.array(layout.M, dtype=np.int8),
        O,
        C,
        P,
        zstd=False,
    )
    if compress:
        _compress_arc_file(arc_path, pose_dir / f"poses-{layout.file_index}.arc.zst")
    return layout.file_index


def _organize_streaming(
    pose_dir: Path,
    sources: list[SourceMeta],
    *,
    max_poses_per_file: int,
    nprocs: int,
    compress: bool,
) -> int:
    layouts, destination = _build_layouts(
        sources,
        max_poses_per_file=max_poses_per_file,
    )
    if not layouts:
        return 0

    temp_dir = pose_dir / ".organize-tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    temp_paths = [temp_dir / f"poses-{layout.file_index}.records" for layout in layouts]
    for path in temp_paths:
        path.touch()
    temp_locks = [threading.Lock() for _ in temp_paths]
    split_items = [
        (key, segments)
        for key, segments in destination.items()
        if len(segments) > 1
    ]
    split_ids = {key: index for index, (key, _) in enumerate(split_items)}
    split_paths = [
        temp_dir / f"split-{index}.records" for index in range(len(split_items))
    ]
    for path in split_paths:
        path.touch()
    split_locks = [threading.Lock() for _ in split_paths]
    arc_temp_dir = temp_dir / "arc"
    arc_temp_dir.mkdir()

    try:
        workers = min(nprocs, max(len(sources), len(layouts)))
        if workers <= 1:
            for source in sources:
                _scatter_source(
                    source,
                    destination,
                    temp_paths,
                    temp_locks,
                    split_ids,
                    split_paths,
                    split_locks,
                )
            for split_path, (_, segments) in zip(split_paths, split_items):
                _distribute_split_offset(split_path, segments, temp_paths, temp_locks)
            for layout, temp_path in zip(layouts, temp_paths):
                _write_layout_from_temp(
                    pose_dir,
                    layout,
                    temp_path,
                    arc_temp_dir,
                    compress=compress,
                )
        else:
            with ThreadPoolExecutor(max_workers=min(workers, len(sources))) as executor:
                futures = [
                    executor.submit(
                        _scatter_source,
                        source,
                        destination,
                        temp_paths,
                        temp_locks,
                        split_ids,
                        split_paths,
                        split_locks,
                    )
                    for source in sources
                ]
                for future in futures:
                    future.result()

            if split_items:
                with ThreadPoolExecutor(
                    max_workers=min(workers, len(split_items))
                ) as executor:
                    futures = [
                        executor.submit(
                            _distribute_split_offset,
                            split_path,
                            segments,
                            temp_paths,
                            temp_locks,
                        )
                        for split_path, (_, segments) in zip(split_paths, split_items)
                    ]
                    for future in futures:
                        future.result()

            with ThreadPoolExecutor(max_workers=min(workers, len(layouts))) as executor:
                futures = [
                    executor.submit(
                        _write_layout_from_temp,
                        pose_dir,
                        layout,
                        temp_path,
                        arc_temp_dir,
                        compress=compress,
                    )
                    for layout, temp_path in zip(layouts, temp_paths)
                ]
                for future in futures:
                    future.result()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return len(layouts)


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

    sources = _read_sources(unorganized, nprocs=int(args.nprocs))
    if sources:
        _organize_streaming(
            pose_dir,
            sources,
            max_poses_per_file=min(int(args.capacity), int(args.max_poses_per_file)),
            nprocs=int(args.nprocs),
            compress=bool(args.compress),
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
