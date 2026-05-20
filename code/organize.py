from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import contextlib
from dataclasses import dataclass
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Sequence

import numpy as np
from tqdm import tqdm

from poses import (
    ARC_SUFFIX,
    ARC_ZSTD_SUFFIX,
    MAGIC,
    MAX_NO,
    MAX_NP,
    discover_organized,
    discover_unorganized,
    iter_arc_pose_chunks,
    read_arc_offsets,
)


DONE_MARKER = ".ORGANIZED-DONE"


@dataclass(frozen=True)
class SourceMeta:
    path: Path
    M: tuple[int, int, int]
    O: np.ndarray
    C: np.ndarray
    nP: int
    bucket_size: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="organize",
        description="Canonicalize unorganized alaric .arc pose shards.",
    )
    parser.add_argument("pose_dir", metavar="POSE_DIR")
    parser.add_argument(
        "--capacity",
        type=int,
        default=2_000_000_000,
        help="Per-worker memory budget in poses; peak in-memory ~ capacity * nprocs * 4 bytes.",
    )
    parser.add_argument("--max-poses-per-file", type=int, default=100_000_000)
    parser.add_argument(
        "--chunk-poses",
        type=int,
        default=1_000_000,
        help="Maximum pose rows to hold in memory for streaming scatter/write steps.",
    )
    parser.add_argument("--nprocs", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Write organized poses-*.arc.zst files by streaming completed temp .arc files through zstd.",
    )
    parser.add_argument(
        "--local-tempdir",
        action="store_true",
        help="Copy unorganized shards into a local tempdir, decompressing .arc.zst to .arc, and read sources from there.",
    )
    parser.add_argument(
        "--local-stagedir",
        action="store_true",
        help=(
            "Write organized output to a local tempdir, then delete the "
            "unorganized shards in the pose dir and move the staged output "
            "in. Use when the pose dir's partition cannot hold the organized "
            "and unorganized data simultaneously. WARNING: a process crash "
            "between the unorganized deletion and the staged commit destroys "
            "both the organized output (still in a local tempdir that will be "
            "cleaned up) and the unorganized inputs; recovery requires "
            "regenerating the unorganized shards from upstream."
        ),
    )
    parser.add_argument("--debug", action="store_true")
    return parser


@dataclass(frozen=True)
class OutputLayout:
    file_index: int
    M: tuple[int, int, int]
    offsets: list[tuple[int, int, int]]
    counts: list[int]
    bucket_size: int


@dataclass(frozen=True)
class Segment:
    layout_id: int
    offset_id: int
    start: int
    count: int


@dataclass(frozen=True)
class LayoutEntry:
    M: tuple[int, int, int]
    offset: tuple[int, int, int]
    segment: Segment


def _read_source_meta(path: Path) -> SourceMeta:
    M, O, C, nP, bucket_size = read_arc_offsets(path)
    if int(C.max(initial=0)) > MAX_NP:
        raise ValueError(f"mini-bucket count exceeds uint32 in {path}")
    return SourceMeta(
        path=path,
        M=tuple(int(x) for x in M),
        O=O,
        C=C,
        nP=nP,
        bucket_size=bucket_size,
    )


def _read_sources(paths: list[Path], *, nprocs: int = 1) -> list[SourceMeta]:
    if not paths:
        return []
    first = _read_source_meta(paths[0])
    expected_bucket_size = first.bucket_size

    def _read_and_verify(path: Path) -> SourceMeta:
        meta = _read_source_meta(path)
        if meta.bucket_size != expected_bucket_size:
            raise ValueError(
                f"inconsistent bucket_size: {paths[0]} has {expected_bucket_size}, "
                f"{path} has {meta.bucket_size}"
            )
        return meta

    remaining = paths[1:]
    if not remaining:
        return [first]
    if nprocs <= 1 or len(remaining) <= 1:
        return [first] + [_read_and_verify(p) for p in remaining]
    with ThreadPoolExecutor(max_workers=min(nprocs, len(remaining))) as executor:
        return [first] + list(executor.map(_read_and_verify, remaining))


def _build_layouts(
    sources: list[SourceMeta],
    *,
    max_poses_per_file: int,
) -> tuple[
    list[OutputLayout],
    dict[tuple[tuple[int, int, int], tuple[int, int, int]], list[Segment]],
    int,
]:
    if not sources:
        raise ValueError("no sources")
    bucket_size = sources[0].bucket_size

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
            layouts.append(
                OutputLayout(
                    file_index, M, current_offsets, current_counts, bucket_size
                )
            )
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
    return layouts, destination, bucket_size


def _write_arc_header(
    handle,
    M: np.ndarray,
    O: np.ndarray,
    C: np.ndarray,
    nP: int,
    bucket_size: int,
) -> None:
    if M.shape != (3,):
        raise ValueError("M must have shape (3,)")
    if O.ndim != 2 or O.shape[1] != 3:
        raise ValueError("O must be a [nO,3] array")
    if C.ndim != 1 or len(C) != len(O):
        raise ValueError("C length must equal nO")
    if len(O) > MAX_NO:
        raise ValueError("nO must be <= 65536")
    if nP <= 0 or nP > MAX_NP:
        raise ValueError("nP must be in 1..2**32-1")
    if int(C.sum(dtype=np.uint64)) != nP:
        raise ValueError("sum(C) must equal nP")

    nO_raw = 0 if len(O) == MAX_NO else len(O)
    handle.write(MAGIC)
    handle.write(np.array([bucket_size], dtype="<u2").tobytes())
    handle.write(M.astype("<i2", copy=False).tobytes(order="C"))
    handle.write(np.array([nO_raw], dtype="<u2").tobytes())
    handle.write(np.array([nP], dtype="<u4").tobytes())
    handle.write(O.astype("<i2", copy=False).tobytes(order="C"))
    handle.write(C.astype("<u4", copy=False).tobytes(order="C"))


def _build_layout_entries(
    layouts: list[OutputLayout],
    destination: dict[
        tuple[tuple[int, int, int], tuple[int, int, int]],
        list[Segment],
    ],
) -> list[list[LayoutEntry]]:
    entries: list[list[LayoutEntry | None]] = [
        [None] * len(layout.offsets) for layout in layouts
    ]
    for (M, offset), segments in destination.items():
        for segment in segments:
            entries[segment.layout_id][segment.offset_id] = LayoutEntry(
                M=M,
                offset=offset,
                segment=segment,
            )
    finalized = []
    for layout_id, layout_entries in enumerate(entries):
        missing = [i for i, entry in enumerate(layout_entries) if entry is None]
        if missing:
            raise ValueError(f"layout {layout_id} is missing entries for {missing[:5]}")
        finalized.append([entry for entry in layout_entries if entry is not None])
    return finalized


def _write_bucket(handle, offset_id: int, bucket: np.ndarray, *, chunk_poses: int) -> None:
    bucket.sort()
    for start in range(0, len(bucket), chunk_poses):
        stop = min(start + chunk_poses, len(bucket))
        chunk = bucket[start:stop]
        P = np.empty((len(chunk), 3), dtype=np.uint16)
        P[:, 0] = (chunk >> 16).astype(np.uint16, copy=False)
        P[:, 1] = chunk.astype(np.uint16, copy=False)
        P[:, 2] = offset_id
        handle.write(P.tobytes(order="C"))


@dataclass(frozen=True)
class LayoutGroup:
    M: tuple[int, int, int]
    layout_ids: list[int]
    nposes: int


def _build_layout_groups(
    layouts: list[OutputLayout],
    *,
    capacity: int,
) -> list[LayoutGroup]:
    groups: list[LayoutGroup] = []
    current_M: tuple[int, int, int] | None = None
    current_ids: list[int] = []
    current_nposes = 0

    def close_current() -> None:
        nonlocal current_M, current_ids, current_nposes
        if current_M is None or not current_ids:
            return
        groups.append(LayoutGroup(current_M, current_ids, current_nposes))
        current_M = None
        current_ids = []
        current_nposes = 0

    for layout_id, layout in enumerate(layouts):
        nposes = int(sum(layout.counts))
        if (
            current_M is not None
            and (
                layout.M != current_M
                or (current_nposes and current_nposes + nposes > capacity)
            )
        ):
            close_current()
        if current_M is None:
            current_M = layout.M
        current_ids.append(layout_id)
        current_nposes += nposes
    close_current()
    return groups


def _write_group_layout(
    pose_dir: Path,
    layout: OutputLayout,
    entries: list[LayoutEntry],
    buffers: dict[tuple[tuple[int, int, int], tuple[int, int, int]], np.ndarray],
    *,
    compress: bool,
    chunk_poses: int,
    progress=None,
    overrides: dict[int, tuple[int, int]] | None = None,
) -> int:
    try:
        import zstandard as zstd
    except ImportError as exc:
        if compress:
            raise ImportError("zstandard is required for --compress") from exc
        zstd = None

    M = np.array(layout.M, dtype=np.int16)
    O = np.array(layout.offsets, dtype=np.int16)
    if overrides:
        effective_counts = [
            overrides[i][1] if i in overrides else c
            for i, c in enumerate(layout.counts)
        ]
    else:
        effective_counts = list(layout.counts)
    C = np.array(effective_counts, dtype=np.uint32)
    if int(C.sum(dtype=np.uint64)) == 0:
        # A boundary-adjusted split slice collapsed to zero poses (e.g. an
        # entire run of duplicates at the tail). Skip writing an empty file;
        # discover_organized tolerates gaps in file_index.
        return layout.file_index
    dest = (
        pose_dir / f"poses-{layout.file_index}.arc.zst"
        if compress
        else pose_dir / f"poses-{layout.file_index}.arc"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f"{dest.name}.",
        suffix=".tmp",
        dir=dest.parent,
        delete=False,
    ) as raw_handle:
        tmp_path = Path(raw_handle.name)
        output_handle = raw_handle
        compressor = None
        if compress:
            assert zstd is not None
            compressor = zstd.ZstdCompressor().stream_writer(raw_handle)
            output_handle = compressor
        try:
            _write_arc_header(
                output_handle,
                M,
                O,
                C,
                int(C.sum(dtype=np.uint64)),
                layout.bucket_size,
            )
            for offset_id, entry in enumerate(entries):
                key = (entry.M, entry.offset)
                segment = entry.segment
                bucket = buffers[key]
                if overrides and offset_id in overrides:
                    bucket.sort()
                    start, count = overrides[offset_id]
                    out_bucket = bucket[start : start + count]
                elif len(bucket) == segment.count and segment.start == 0:
                    out_bucket = bucket
                    out_bucket.sort()
                else:
                    bucket.sort()
                    out_bucket = bucket[
                        segment.start : segment.start + segment.count
                    ]
                if len(out_bucket) != int(C[offset_id]):
                    raise ValueError(f"layout count mismatch for {key}")
                _write_bucket(
                    output_handle,
                    offset_id,
                    out_bucket,
                    chunk_poses=chunk_poses,
                )
                if progress is not None:
                    progress.update(len(out_bucket))
            if compressor is not None:
                compressor.close()
                compressor = None
        finally:
            if compressor is not None:
                compressor.close()
    try:
        tmp_path.replace(dest)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return layout.file_index


class _CounterSink:
    __slots__ = ("counter",)

    def __init__(self, counter) -> None:
        self.counter = counter

    def update(self, n: int) -> None:
        if n <= 0:
            return
        with self.counter.get_lock():
            self.counter.value += int(n)


def _process_layout_group(
    pose_dir: Path,
    group: LayoutGroup,
    layouts: list[OutputLayout],
    layout_entries: list[list[LayoutEntry]],
    sources: list[SourceMeta],
    destination: dict[
        tuple[tuple[int, int, int], tuple[int, int, int]],
        list[Segment],
    ],
    *,
    compress: bool,
    chunk_poses: int,
    progress: bool = True,
    progress_counter=None,
) -> None:
    group_keys: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
    collect_counts: dict[
        tuple[tuple[int, int, int], tuple[int, int, int]],
        int,
    ] = {}
    for layout_id in group.layout_ids:
        for entry in layout_entries[layout_id]:
            key = (entry.M, entry.offset)
            if key in collect_counts:
                continue
            segments = destination[key]
            if len(segments) == 1:
                collect_count = entry.segment.count
            else:
                collect_count = sum(segment.count for segment in segments)
            collect_counts[key] = int(collect_count)
            group_keys.append(key)

    key_to_index = {key: index for index, key in enumerate(group_keys)}
    buffers = {
        key: np.empty(count, dtype=np.uint32)
        for key, count in collect_counts.items()
    }
    cursors = np.zeros(len(group_keys), dtype=np.uint64)
    group_sources = [source for source in sources if source.M == group.M]
    if progress_counter is not None:
        sink = _CounterSink(progress_counter)
        read_cm = contextlib.nullcontext(sink)
        write_cm = contextlib.nullcontext(sink)
    else:
        read_cm = tqdm(
            total=group.nposes,
            desc=f"Organize {group.M} read",
            unit="pose",
            unit_scale=True,
            mininterval=2.0,
            disable=not progress,
        )
        write_cm = tqdm(
            total=group.nposes,
            desc=f"Organize {group.M} write",
            unit="pose",
            unit_scale=True,
            mininterval=2.0,
            disable=not progress,
        )
    with read_cm as read_pbar:
        for source in group_sources:
            local_to_group = np.full(len(source.O), -1, dtype=np.int32)
            for local_index, offset_row in enumerate(source.O):
                key = (source.M, tuple(int(x) for x in offset_row))
                group_index = key_to_index.get(key)
                if group_index is not None:
                    local_to_group[local_index] = group_index
            if not np.any(local_to_group >= 0):
                continue
            saw_chunk = False
            checked_meta = False
            for M_arr, O, C, P, _bs in iter_arc_pose_chunks(
                source.path,
                rows_per_chunk=chunk_poses,
            ):
                saw_chunk = True
                if not checked_meta:
                    M = tuple(int(x) for x in M_arr)
                    if M != source.M:
                        raise ValueError(f"M changed while reading {source.path}")
                    if len(O) != len(source.O) or not np.array_equal(O, source.O):
                        raise ValueError(f"O changed while reading {source.path}")
                    if len(C) != len(source.C) or not np.array_equal(C, source.C):
                        raise ValueError(f"C changed while reading {source.path}")
                    checked_meta = True
                if len(P) == 0:
                    continue
                group_ids = local_to_group[P[:, 2]]
                keep = group_ids >= 0
                n_kept = int(np.count_nonzero(keep))
                read_pbar.update(n_kept)
                if n_kept == 0:
                    continue
                kept_group_ids = group_ids[keep]
                kept_rows = P[keep, 0:2]
                packed = (
                    (kept_rows[:, 0].astype(np.uint32, copy=False) << 16)
                    | kept_rows[:, 1].astype(np.uint32, copy=False)
                )
                order = np.argsort(kept_group_ids, kind="stable")
                sorted_group_ids = kept_group_ids[order]
                sorted_packed = packed[order]
                run_starts = np.r_[
                    0,
                    np.flatnonzero(sorted_group_ids[1:] != sorted_group_ids[:-1]) + 1,
                ]
                run_stops = np.r_[run_starts[1:], len(sorted_group_ids)]
                for run_start, run_stop in zip(run_starts, run_stops):
                    group_index = int(sorted_group_ids[run_start])
                    key = group_keys[group_index]
                    n = int(run_stop - run_start)
                    cursor = int(cursors[group_index])
                    buffers[key][cursor : cursor + n] = sorted_packed[run_start:run_stop]
                    cursors[group_index] += n
            if not saw_chunk and source.nP != 0:
                raise ValueError(f"no pose chunks read from {source.path}")

    for index, key in enumerate(group_keys):
        expected = len(buffers[key])
        if int(cursors[index]) != expected:
            raise ValueError(
                f"group {group.M} offset {key[1]} has {int(cursors[index])} "
                f"poses, expected {expected}"
            )

    # For split mini-buckets, sort and shift slice boundaries forward past any
    # duplicate run that straddles them so identical poses never split across
    # output files.
    layout_overrides: dict[int, dict[int, tuple[int, int]]] = {}
    for key in group_keys:
        segments = destination[key]
        if len(segments) <= 1:
            continue
        bucket = buffers[key]
        bucket.sort()
        n = len(bucket)
        cur = 0
        last_idx = len(segments) - 1
        for i, seg in enumerate(segments):
            if i == last_idx:
                new_start = cur
                new_count = n - cur
            else:
                planned_end = min(cur + seg.count, n)
                while planned_end < n and bucket[planned_end] == bucket[planned_end - 1]:
                    planned_end += 1
                new_start = cur
                new_count = planned_end - cur
                cur = planned_end
            if new_start != seg.start or new_count != seg.count:
                layout_overrides.setdefault(seg.layout_id, {})[seg.offset_id] = (
                    new_start,
                    new_count,
                )

    with write_cm as write_pbar:
        for layout_id in group.layout_ids:
            _write_group_layout(
                pose_dir,
                layouts[layout_id],
                layout_entries[layout_id],
                buffers,
                compress=compress,
                chunk_poses=chunk_poses,
                progress=write_pbar,
                overrides=layout_overrides.get(layout_id),
            )


_WORKER_STATE: dict = {}


def _init_group_worker(state: dict) -> None:
    _WORKER_STATE.clear()
    _WORKER_STATE.update(state)


def _run_group_in_worker(group: LayoutGroup) -> None:
    state = _WORKER_STATE
    _process_layout_group(
        state["pose_dir"],
        group,
        state["layouts"],
        state["layout_entries"],
        state["sources"],
        state["destination"],
        compress=state["compress"],
        chunk_poses=state["chunk_poses"],
        progress=False,
        progress_counter=state.get("progress_counter"),
    )


def _poll_progress(counter, total: int, stop_event: threading.Event) -> None:
    with tqdm(
        total=total,
        desc="Organize poses",
        unit="pose",
        unit_scale=True,
        mininterval=2.0,
    ) as bar:
        last = 0
        while not stop_event.wait(0.5):
            with counter.get_lock():
                current = counter.value
            if current > last:
                bar.update(current - last)
                last = current
        with counter.get_lock():
            current = counter.value
        if current > last:
            bar.update(current - last)


def _organize_streaming(
    pose_dir: Path,
    sources: list[SourceMeta],
    *,
    capacity: int,
    max_poses_per_file: int,
    nprocs: int,
    compress: bool,
    chunk_poses: int,
) -> int:
    layouts, destination, _bucket_size = _build_layouts(
        sources,
        max_poses_per_file=max_poses_per_file,
    )
    if not layouts:
        return 0

    layout_entries = _build_layout_entries(layouts, destination)
    effective_nprocs = max(1, nprocs)
    groups = _build_layout_groups(layouts, capacity=capacity)
    n_workers = min(effective_nprocs, len(groups))
    if n_workers <= 1:
        for group in tqdm(groups, desc="Organize ranges", unit="range"):
            _process_layout_group(
                pose_dir,
                group,
                layouts,
                layout_entries,
                sources,
                destination,
                compress=compress,
                chunk_poses=chunk_poses,
            )
        return len(layouts)

    ctx = mp.get_context("fork")
    counter = ctx.Value("Q", 0)
    total_work = 2 * sum(group.nposes for group in groups)
    state = {
        "pose_dir": pose_dir,
        "layouts": layouts,
        "layout_entries": layout_entries,
        "sources": sources,
        "destination": destination,
        "compress": compress,
        "chunk_poses": chunk_poses,
        "progress_counter": counter,
    }
    stop_event = threading.Event()
    poll_thread = threading.Thread(
        target=_poll_progress,
        args=(counter, total_work, stop_event),
        daemon=True,
    )
    poll_thread.start()
    try:
        with ctx.Pool(
            processes=n_workers,
            initializer=_init_group_worker,
            initargs=(state,),
        ) as pool:
            for _ in pool.imap_unordered(_run_group_in_worker, groups):
                pass
    finally:
        stop_event.set()
        poll_thread.join()
    return len(layouts)


def _staged_unorganized_arc_name(src: Path) -> str:
    if src.name.endswith(ARC_ZSTD_SUFFIX):
        stem = src.name[: -len(ARC_ZSTD_SUFFIX)]
    elif src.name.endswith(ARC_SUFFIX):
        stem = src.name[: -len(ARC_SUFFIX)]
    else:
        raise ValueError(f"unexpected unorganized file: {src}")

    if src.parent.name.startswith("unorganized-") and not src.name.startswith(
        "unorganized-"
    ):
        stem = f"{src.parent.name}-{stem}"
    return f"{stem}{ARC_SUFFIX}"


def _unlink_unorganized(paths: Sequence[Path], root: Path) -> None:
    parents: set[Path] = set()
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        if path.parent != root and path.parent.name.startswith("unorganized-"):
            parents.add(path.parent)

    for parent in sorted(parents, key=lambda p: len(p.parts), reverse=True):
        with contextlib.suppress(OSError):
            parent.rmdir()


def _stage_unorganized_uncompressed(unorganized: list[Path], tempdir: Path) -> None:
    if not unorganized:
        return
    has_zst = any(p.name.endswith(ARC_ZSTD_SUFFIX) for p in unorganized)
    zstd_bin = shutil.which("zstd") if has_zst else None
    if has_zst and zstd_bin is None:
        raise FileNotFoundError(
            "zstd CLI is required to decompress .arc.zst sources for --local-tempdir"
        )

    ncores = os.cpu_count() or 1
    workers = max(1, min(8, ncores, len(unorganized)))
    threads_per_worker = max(1, ncores // workers)

    def stage_one(src: Path) -> None:
        dst = tempdir / _staged_unorganized_arc_name(src)
        if src.name.endswith(ARC_ZSTD_SUFFIX):
            assert zstd_bin is not None
            subprocess.run(
                [
                    zstd_bin,
                    "-d",
                    "-q",
                    "-f",
                    f"-T{threads_per_worker}",
                    "-o",
                    str(dst),
                    str(src),
                ],
                check=True,
            )
        elif src.name.endswith(ARC_SUFFIX):
            shutil.copy2(src, dst)
        else:
            raise ValueError(f"unexpected unorganized file: {src}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(
            tqdm(
                executor.map(stage_one, unorganized),
                total=len(unorganized),
                desc="Staging sources",
                unit="file",
            )
        )


def _commit_staged_organized(stagedir: Path, pose_dir: Path) -> None:
    organized = discover_organized(stagedir)
    if not organized:
        return
    for src in organized:
        shutil.move(str(src), str(pose_dir / src.name))


def _run(args: argparse.Namespace) -> int:
    if args.capacity <= 0:
        raise ValueError("--capacity must be positive")
    if args.max_poses_per_file <= 0 or args.max_poses_per_file > MAX_NP:
        raise ValueError("--max-poses-per-file must be in 1..2**32-1")
    if args.chunk_poses <= 0:
        raise ValueError("--chunk-poses must be positive")
    if args.nprocs <= 0:
        raise ValueError("--nprocs must be positive")

    pose_dir = Path(args.pose_dir)
    pose_dir.mkdir(parents=True, exist_ok=True)
    marker = pose_dir / DONE_MARKER
    unorganized = discover_unorganized(pose_dir)
    organized = discover_organized(pose_dir)

    if marker.exists():
        _unlink_unorganized(unorganized, pose_dir)
        marker.unlink()
        return 0

    if not unorganized:
        return 0
    if organized:
        raise ValueError(
            f"{pose_dir} contains both organized poses-*.arc and unorganized .arc files"
        )

    local_tempdir: Path | None = None
    local_stagedir: Path | None = None
    try:
        if args.local_tempdir:
            local_tempdir = Path(tempfile.mkdtemp(prefix="alaric-organize-src-"))
            _stage_unorganized_uncompressed(unorganized, local_tempdir)
            source_paths = discover_unorganized(local_tempdir)
        else:
            source_paths = unorganized

        if args.local_stagedir:
            local_stagedir = Path(tempfile.mkdtemp(prefix="alaric-organize-stage-"))
            output_dir = local_stagedir
        else:
            output_dir = pose_dir

        sources = _read_sources(source_paths, nprocs=int(args.nprocs))
        if sources:
            _organize_streaming(
                output_dir,
                sources,
                capacity=int(args.capacity),
                max_poses_per_file=int(args.max_poses_per_file),
                nprocs=int(args.nprocs),
                compress=bool(args.compress),
                chunk_poses=int(args.chunk_poses),
            )

        marker.touch()
        _unlink_unorganized(unorganized, pose_dir)
        if local_stagedir is not None:
            _commit_staged_organized(local_stagedir, pose_dir)
        marker.unlink()
    finally:
        if local_tempdir is not None and local_tempdir.exists():
            shutil.rmtree(local_tempdir)
        if local_stagedir is not None and local_stagedir.exists():
            shutil.rmtree(local_stagedir)
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
