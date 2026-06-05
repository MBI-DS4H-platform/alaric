"""Filter identity-overlapping organized pose directories.

This module intentionally bypasses PoseReader because it compares and rewrites
raw .arc bucket structure: M/O/C metadata, local offset indices, and packed
conformer/rotamer keys are part of the algorithm.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm
import numpy as np

from poses import (
    ARC_ZSTD_SUFFIX,
    HEADER_SIZE,
    MAX_NP,
    _parse_arc_header,
    _validate_arc_arrays,
    discover_organized,
    read_arc_file,
    read_arc_offsets,
    write_arc_file,
)


_ZSTD_CLI = shutil.which("zstd")
_LOADER_WORKERS = 10
_ZSTD_THREADS = 4


PoseKey = tuple[tuple[int, int, int], tuple[int, int, int]]


@dataclass
class FileIndexEntry:
    path: Path
    order: int
    M: tuple[int, int, int]
    O: np.ndarray
    C: np.ndarray
    I: np.ndarray
    nP: int
    bucket_size: int
    global_start: int
    offset_to_index: dict[tuple[int, int, int], int]
    a_first: int | None = None
    a_last: int | None = None


@dataclass
class LoadedFile:
    entry: FileIndexEntry
    P: np.ndarray

    def pose_keys(self, offset_index: int) -> np.ndarray:
        start = int(self.entry.I[offset_index])
        stop = int(self.entry.I[offset_index + 1])
        poses = self.P[start:stop]
        return _pack_pose_keys(poses)


@dataclass
class IdentityTask:
    small: FileIndexEntry
    other: FileIndexEntry
    offsets: list[tuple[tuple[int, int, int], int, int]]


def _row_tuple(row: np.ndarray) -> tuple[int, int, int]:
    return tuple(int(x) for x in row)


def _pack_pose_keys(P: np.ndarray) -> np.ndarray:
    conf = P[:, 0].astype(np.uint32, copy=False)
    rot = P[:, 1].astype(np.uint32, copy=False)
    return (conf << np.uint32(16)) | rot


def _unpack_pose_keys(keys: np.ndarray, offset_index: int) -> np.ndarray:
    P = np.empty((len(keys), 3), dtype=np.uint16)
    P[:, 0] = (keys >> np.uint32(16)).astype(np.uint16, copy=False)
    P[:, 1] = keys.astype(np.uint16, copy=False)
    P[:, 2] = np.uint16(offset_index)
    return P


def _read_file_index(pose_dir: Path) -> list[FileIndexEntry]:
    paths = discover_organized(pose_dir)
    if not paths:
        return []

    entries: list[FileIndexEntry] = []
    bucket_size: int | None = None
    global_start = 0
    for order, path in enumerate(paths):
        M, O, C, nP, file_bucket_size = read_arc_offsets(path)
        if bucket_size is None:
            bucket_size = file_bucket_size
        elif file_bucket_size != bucket_size:
            raise ValueError(
                f"inconsistent bucket_size in {pose_dir}: "
                f"expected {bucket_size}, got {file_bucket_size} in {path}"
            )
        I = np.empty(len(C) + 1, dtype=np.uint64)
        I[0] = 0
        I[1:] = np.cumsum(C, dtype=np.uint64)
        entries.append(
            FileIndexEntry(
                path=path,
                order=order,
                M=_row_tuple(M),
                O=O,
                C=C,
                I=I,
                nP=int(nP),
                bucket_size=int(file_bucket_size),
                global_start=global_start,
                offset_to_index={_row_tuple(row): i for i, row in enumerate(O)},
            )
        )
        global_start += int(nP)
    return entries


def _index_by_bucket(
    entries: list[FileIndexEntry],
) -> dict[tuple[int, int, int], list[FileIndexEntry]]:
    by_bucket: dict[tuple[int, int, int], list[FileIndexEntry]] = {}
    for entry in entries:
        by_bucket.setdefault(entry.M, []).append(entry)
    return by_bucket


def _bucket_offsets(entries: list[FileIndexEntry]) -> set[tuple[int, int, int]]:
    offsets: set[tuple[int, int, int]] = set()
    for entry in entries:
        offsets.update(entry.offset_to_index)
    return offsets


def _prepare_common_offsets(
    entries1: list[FileIndexEntry],
    entries2: list[FileIndexEntry],
) -> dict[tuple[int, int, int], list[tuple[int, int, int]]]:
    by_bucket1 = _index_by_bucket(entries1)
    by_bucket2 = _index_by_bucket(entries2)
    common: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
    for M in sorted(set(by_bucket1) & set(by_bucket2)):
        offsets = sorted(_bucket_offsets(by_bucket1[M]) & _bucket_offsets(by_bucket2[M]))
        if offsets:
            common[M] = offsets
    return common


def _assign_a_ranges(
    entries: list[FileIndexEntry],
    common_offsets: dict[tuple[int, int, int], list[tuple[int, int, int]]],
) -> None:
    for entry in entries:
        offsets = common_offsets.get(entry.M)
        if not offsets:
            entry.a_first = None
            entry.a_last = None
            continue
        offset_to_a = {offset: i for i, offset in enumerate(offsets)}
        positions = [offset_to_a[offset] for offset in entry.offset_to_index if offset in offset_to_a]
        if positions:
            entry.a_first = min(positions)
            entry.a_last = max(positions)
        else:
            entry.a_first = None
            entry.a_last = None


def _build_offset_entries(
    entries: list[FileIndexEntry],
    common_offsets: dict[tuple[int, int, int], list[tuple[int, int, int]]],
) -> dict[PoseKey, list[FileIndexEntry]]:
    offset_entries: dict[PoseKey, list[FileIndexEntry]] = {}
    for entry in entries:
        offsets = common_offsets.get(entry.M)
        if not offsets:
            continue
        common_for_bucket = set(offsets)
        for offset in entry.offset_to_index:
            if offset in common_for_bucket:
                offset_entries.setdefault((entry.M, offset), []).append(entry)
    for holders in offset_entries.values():
        holders.sort(key=lambda e: e.order)
    return offset_entries


def _ranges_overlap(left: FileIndexEntry, right: FileIndexEntry) -> bool:
    if left.a_first is None or left.a_last is None:
        return False
    if right.a_first is None or right.a_last is None:
        return False
    return left.a_first <= right.a_last and right.a_first <= left.a_last


def _decompress_arc_bytes(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Read an .arc file using the multi-threaded zstd CLI when available."""
    if _ZSTD_CLI and str(path).endswith(ARC_ZSTD_SUFFIX):
        proc = subprocess.run(
            [_ZSTD_CLI, "-d", "-c", f"-T{_ZSTD_THREADS}", "-q", "--", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        raw = proc.stdout
        M, nO, nP, bucket_size = _parse_arc_header(raw[:HEADER_SIZE], path)
        expected = HEADER_SIZE + nO * 6 + nO * 4 + nP * 6
        if len(raw) != expected:
            raise ValueError(
                f"bad .arc length for {path}: expected {expected} bytes, got {len(raw)}"
            )
        pos = HEADER_SIZE
        O = np.frombuffer(raw[pos : pos + nO * 6], dtype="<i2").copy().reshape(nO, 3)
        pos += nO * 6
        C = np.frombuffer(raw[pos : pos + nO * 4], dtype="<u4").copy()
        pos += nO * 4
        P = np.frombuffer(raw[pos : pos + nP * 6], dtype="<u2").copy().reshape(nP, 3)
        M, O, C, P = _validate_arc_arrays(M, O, C, P, bucket_size)
        return M, O, C, P, bucket_size
    return read_arc_file(path)


def _read_loaded(entry: FileIndexEntry) -> LoadedFile:
    M, O, C, P, bucket_size = _decompress_arc_bytes(entry.path)
    if tuple(int(x) for x in M) != entry.M:
        raise ValueError(f"bucket changed while reading {entry.path}")
    if bucket_size != entry.bucket_size:
        raise ValueError(f"bucket_size changed while reading {entry.path}")
    if len(O) != len(entry.O) or not np.array_equal(O, entry.O):
        raise ValueError(f"offsets changed while reading {entry.path}")
    if not np.array_equal(C, entry.C):
        raise ValueError(f"counts changed while reading {entry.path}")
    return LoadedFile(entry, P)


import sortednp

def _intersect_sorted_unique(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if len(left) == 0 or len(right) == 0:
        return np.empty(0, dtype=np.uint32)
    return sortednp.intersect(
        left,
        right,
        duplicates=sortednp.IntersectDuplicates.DROP,
    )


class _PrefetchingLoader:
    """Refcount-based cache with a thread pool that prefetches .arc files.

    Refholders are indexed 0..n-1. At any time the holders for indices in a
    sliding window [current, current+window_ahead] are active; the loader
    keeps in cache every path with refcount>0 and evicts the rest. A caller can
    alternatively cap future readahead by unique path count.
    """

    def __init__(
        self,
        refholder_paths_fn,
        n_items: int,
        path_to_entry: dict[Path, FileIndexEntry],
        *,
        window_ahead: int = 20,
        max_readahead_paths: int | None = None,
        workers: int = _LOADER_WORKERS,
    ) -> None:
        self._refholder_paths = refholder_paths_fn
        self._n_items = n_items
        self._path_to_entry = path_to_entry
        self._window_ahead = window_ahead
        self._max_readahead_paths = max_readahead_paths
        self._cache: dict[Path, LoadedFile] = {}
        self._refcount: dict[Path, int] = {}
        self._in_flight: set[Path] = set()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._exc: list[BaseException] = []
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="arc-loader"
        )
        with self._lock:
            if self._max_readahead_paths is None:
                for j in range(min(window_ahead + 1, n_items)):
                    self._add_refs(j)
            else:
                self._refresh_capped_refs(0)
            self._cond.notify_all()
        self._thread = threading.Thread(target=self._loader, daemon=True)
        self._thread.start()

    def _add_refs(self, idx: int) -> None:
        for p in self._refholder_paths(idx):
            self._refcount[p] = self._refcount.get(p, 0) + 1

    def _remove_refs(self, idx: int) -> None:
        for p in self._refholder_paths(idx):
            c = self._refcount.get(p, 0) - 1
            if c <= 0:
                self._refcount.pop(p, None)
            else:
                self._refcount[p] = c

    def _refresh_capped_refs(self, current_idx: int) -> None:
        target: dict[Path, int] = {}

        def add_paths(paths: list[Path]) -> None:
            for p in paths:
                target[p] = target.get(p, 0) + 1

        if current_idx >= self._n_items:
            self._refcount = target
            return

        current_paths = self._refholder_paths(current_idx)
        current_path_set = set(current_paths)
        add_paths(current_paths)

        assert self._max_readahead_paths is not None
        future_paths: set[Path] = set()
        for idx in range(current_idx + 1, self._n_items):
            paths = self._refholder_paths(idx)
            new_future_paths = [
                p for p in paths
                if p not in current_path_set and p not in future_paths
            ]
            if len(future_paths) + len(new_future_paths) > self._max_readahead_paths:
                break
            add_paths(paths)
            future_paths.update(new_future_paths)

        self._refcount = target

    def _make_callback(self, p: Path):
        def callback(future: concurrent.futures.Future) -> None:
            with self._lock:
                self._in_flight.discard(p)
                try:
                    loaded = future.result()
                except BaseException as exc:
                    self._exc.append(exc)
                    self._cond.notify_all()
                    return
                if self._refcount.get(p, 0) > 0 and p not in self._cache:
                    self._cache[p] = loaded
                self._cond.notify_all()
        return callback

    def _loader(self) -> None:
        try:
            while not self._stop.is_set():
                with self._lock:
                    stale = [
                        p for p in self._cache
                        if self._refcount.get(p, 0) == 0 and p not in self._in_flight
                    ]
                    for p in stale:
                        del self._cache[p]
                    pending = [
                        p for p in self._refcount
                        if p not in self._cache and p not in self._in_flight
                    ]
                    for p in pending:
                        self._in_flight.add(p)
                    entries_to_load = [(p, self._path_to_entry[p]) for p in pending]
                for p, entry_to_load in entries_to_load:
                    if self._stop.is_set():
                        break
                    future = self._pool.submit(_read_loaded, entry_to_load)
                    future.add_done_callback(self._make_callback(p))
                with self._lock:
                    if self._stop.is_set():
                        return
                    if not [
                        p for p in self._refcount
                        if p not in self._cache and p not in self._in_flight
                    ]:
                        self._cond.wait(timeout=0.1)
        except BaseException as exc:
            with self._lock:
                self._exc.append(exc)
                self._cond.notify_all()

    def wait_for(self, path: Path) -> LoadedFile:
        with self._lock:
            while path not in self._cache:
                if self._exc:
                    raise self._exc[0]
                self._cond.wait()
            return self._cache[path]

    def advance(self, idx: int) -> None:
        """After processing idx, refresh the active current/prefetch refs."""
        with self._lock:
            if self._max_readahead_paths is None:
                self._remove_refs(idx)
                nxt = idx + self._window_ahead + 1
                if nxt < self._n_items:
                    self._add_refs(nxt)
            else:
                self._refresh_capped_refs(idx + 1)
            self._cond.notify_all()

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            self._cond.notify_all()
        self._thread.join()
        self._pool.shutdown(wait=True)

    def __enter__(self) -> "_PrefetchingLoader":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

def build_identity_set(
    entries1: list[FileIndexEntry],
    entries2: list[FileIndexEntry],
    common_offsets: dict[tuple[int, int, int], list[tuple[int, int, int]]],
    *,
    readahead: int = 20,
) -> dict[PoseKey, np.ndarray]:
    if readahead < 0:
        raise ValueError("--readahead must be non-negative")
    nposes1 = sum(e.nP for e in entries1 if e.M in common_offsets)
    nposes2 = sum(e.nP for e in entries2 if e.M in common_offsets)
    small_entries, other_entries = (
        (entries1, entries2) if nposes1 <= nposes2 else (entries2, entries1)
    )
    other_by_bucket = _index_by_bucket(other_entries)
    other_offset_entries = _build_offset_entries(other_entries, common_offsets)

    tasks: list[IdentityTask] = []
    for i, entry in enumerate(small_entries):
        if entry.M not in common_offsets or entry.a_first is None:
            continue
        candidates = [
            other
            for other in other_by_bucket.get(entry.M, [])
            if _ranges_overlap(entry, other)
        ]
        if not candidates:
            continue
        candidates_set = {other.path for other in candidates}
        common_for_bucket = set(common_offsets[entry.M])
        groups: dict[
            Path,
            tuple[FileIndexEntry, list[tuple[tuple[int, int, int], int, int]]],
        ] = {}
        for offset in entry.offset_to_index:
            if offset not in common_for_bucket:
                continue
            for other in other_offset_entries.get((entry.M, offset), []):
                if other.path not in candidates_set:
                    continue
                group = groups.setdefault(other.path, (other, []))
                group[1].append(
                    (offset, entry.offset_to_index[offset], other.offset_to_index[offset])
                )
        for other, offsets in groups.values():
            tasks.append(IdentityTask(entry, other, offsets))

    path_to_entry: dict[Path, FileIndexEntry] = {}
    for entry in small_entries:
        path_to_entry[entry.path] = entry
    for entry in other_entries:
        path_to_entry[entry.path] = entry

    def refholder_paths(idx: int) -> list[Path]:
        if idx < 0 or idx >= len(tasks):
            return []
        task = tasks[idx]
        return [task.small.path, task.other.path]

    identity_parts: dict[PoseKey, list[np.ndarray]] = {}
    with _PrefetchingLoader(
        refholder_paths,
        len(tasks),
        path_to_entry,
        max_readahead_paths=readahead,
    ) as loader:
        for i, task in enumerate(tqdm(tasks, desc="Build identity set")):
            try:
                loaded_entry = loader.wait_for(task.small.path)
                loaded_other = loader.wait_for(task.other.path)

                for offset, offset_index, other_offset_index in task.offsets:
                    left_keys = loaded_entry.pose_keys(offset_index)
                    right_keys = loaded_other.pose_keys(other_offset_index)
                    overlap = _intersect_sorted_unique(left_keys, right_keys)
                    if len(overlap):
                        identity_parts.setdefault((task.small.M, offset), []).append(overlap)
            finally:
                loader.advance(i)
    return {
        key: parts[0] if len(parts) == 1 else np.concatenate(parts)
        for key, parts in identity_parts.items()
    }


def _clean_output_dir(output_dir: Path, *, force: bool) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            if not force:
                raise ValueError(f"output directory is not empty: {output_dir}")
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def write_identity_pose_dir(
    identity: dict[PoseKey, np.ndarray],
    output_dir: Path,
    *,
    bucket_size: int,
    max_poses_per_file: int,
    compress: bool = False,
) -> dict[PoseKey, tuple[int, np.ndarray]]:
    if max_poses_per_file <= 0 or max_poses_per_file > MAX_NP:
        raise ValueError("--max-poses-per-file must be in 1..2**32-1")

    global_lookup: dict[PoseKey, tuple[int, np.ndarray]] = {}
    file_index = 1
    global_start = 0
    for M in sorted({key[0] for key in identity}):
        offsets = sorted(offset for bucket, offset in identity if bucket == M)
        current_offsets: list[tuple[int, int, int]] = []
        current_counts: list[int] = []
        current_parts: list[tuple[np.ndarray, int]] = []
        current_nposes = 0

        def close_current() -> None:
            nonlocal file_index, current_offsets, current_counts, current_parts, current_nposes
            if not current_offsets:
                return
            P_parts = [
                _unpack_pose_keys(keys, offset_index)
                for keys, offset_index in current_parts
            ]
            P = np.concatenate(P_parts, axis=0) if P_parts else np.empty((0, 3), dtype=np.uint16)
            suffix = ".arc.zst" if compress else ".arc"
            write_arc_file(
                output_dir / f"poses-{file_index}{suffix}",
                np.array(M, dtype=np.int16),
                np.array(current_offsets, dtype=np.int16),
                np.array(current_counts, dtype=np.uint32),
                P,
                bucket_size=bucket_size,
                zstd=compress,
            )
            file_index += 1
            current_offsets = []
            current_counts = []
            current_parts = []
            current_nposes = 0

        for offset in offsets:
            keys = np.asarray(identity[(M, offset)], dtype=np.uint32)
            if len(keys) == 0:
                continue
            global_lookup[(M, offset)] = (global_start, keys)
            global_start += len(keys)

            start = 0
            while start < len(keys):
                if current_offsets and (
                    current_nposes >= max_poses_per_file
                    or len(current_offsets) >= 65536
                ):
                    close_current()
                room = max_poses_per_file - current_nposes
                take = min(room, len(keys) - start)
                if take <= 0:
                    close_current()
                    continue
                offset_index = len(current_offsets)
                current_offsets.append(offset)
                current_counts.append(take)
                current_parts.append((keys[start : start + take], offset_index))
                current_nposes += take
                start += take
                if current_nposes >= max_poses_per_file:
                    close_current()
        close_current()
    return global_lookup


def build_mapping(
    entries: list[FileIndexEntry],
    global_lookup: dict[PoseKey, tuple[int, np.ndarray]],
    *,
    readahead: int = 20,
    desc: str = "Build mapping",
) -> np.ndarray:
    if readahead < 0:
        raise ValueError("--readahead must be non-negative")
    relevant_per_entry: list[list[tuple[int, tuple[int, np.ndarray]]]] = []
    for entry in entries:
        rel = [
            (offset_index, global_lookup[(entry.M, offset)])
            for offset, offset_index in entry.offset_to_index.items()
            if (entry.M, offset) in global_lookup
        ]
        relevant_per_entry.append(rel)

    path_to_entry = {entry.path: entry for entry in entries}

    def refholder_paths(idx: int) -> list[Path]:
        if 0 <= idx < len(entries) and relevant_per_entry[idx]:
            return [entries[idx].path]
        return []

    chunks: list[np.ndarray] = []
    with _PrefetchingLoader(
        refholder_paths,
        len(entries),
        path_to_entry,
        window_ahead=readahead,
    ) as loader:
        for i, entry in enumerate(tqdm(entries, desc=desc)):
            try:
                relevant = relevant_per_entry[i]
                if not relevant:
                    continue
                loaded = loader.wait_for(entry.path)
                for offset_index, (dest_start, dest_keys) in relevant:
                    keys = loaded.pose_keys(offset_index)
                    if len(keys) == 0 or len(dest_keys) == 0:
                        continue
                    _, (idx_keys, idx_dest) = sortednp.intersect(
                        keys,
                        dest_keys,
                        duplicates=sortednp.IntersectDuplicates.KEEP_MAX_N,
                        indices=True,
                    )
                    if len(idx_keys) == 0:
                        continue
                    local_start = int(entry.I[offset_index])
                    source_ids = (
                        np.uint64(entry.global_start + local_start)
                        + idx_keys.astype(np.uint64, copy=False)
                    )
                    dest_ids = (
                        np.uint64(dest_start) + idx_dest.astype(np.uint64, copy=False)
                    )
                    chunks.append(np.column_stack((source_ids, dest_ids)).astype(np.uint64, copy=False))
            finally:
                loader.advance(i)
    if not chunks:
        return np.empty((0, 2), dtype=np.uint64)
    return np.concatenate(chunks, axis=0)


def run_identity_filter(
    pose_dir1: Path,
    pose_dir2: Path,
    output_dir: Path,
    *,
    force: bool = False,
    max_poses_per_file: int = 100_000_000,
    readahead: int = 20,
    compress: bool = False,
) -> dict[str, int]:
    print("Indexing pose directories...", file=sys.stderr)
    entries1 = _read_file_index(pose_dir1)
    entries2 = _read_file_index(pose_dir2)
    if not entries1:
        raise ValueError(f"no organized pose files found in {pose_dir1}")
    if not entries2:
        raise ValueError(f"no organized pose files found in {pose_dir2}")
    bucket_size1 = entries1[0].bucket_size
    bucket_size2 = entries2[0].bucket_size
    if bucket_size1 != bucket_size2:
        raise ValueError(
            f"inconsistent bucket_size: {pose_dir1} has {bucket_size1}, "
            f"{pose_dir2} has {bucket_size2}"
        )
    input_poses_1 = int(sum(entry.nP for entry in entries1))
    input_poses_2 = int(sum(entry.nP for entry in entries2))
    print(f"Bucket size: {bucket_size1}", file=sys.stderr)
    print(f"Number of input poses in pose directory 1: {input_poses_1}", file=sys.stderr)
    print(f"Number of input poses in pose directory 2: {input_poses_2}", file=sys.stderr)

    common_buckets = {entry.M for entry in entries1} & {entry.M for entry in entries2}
    kept1 = [entry for entry in entries1 if entry.M in common_buckets]
    kept2 = [entry for entry in entries2 if entry.M in common_buckets]
    common_offsets = _prepare_common_offsets(kept1, kept2)
    _assign_a_ranges(kept1, common_offsets)
    _assign_a_ranges(kept2, common_offsets)
    kept_poses_1 = int(sum(entry.nP for entry in kept1))
    kept_poses_2 = int(sum(entry.nP for entry in kept2))
    common_mini_buckets = int(sum(len(v) for v in common_offsets.values()))
    print(f"Number of common buckets: {len(common_buckets)}", file=sys.stderr)
    print(f"Number of common mini-buckets: {common_mini_buckets}", file=sys.stderr)
    print(f"Number of poses kept from pose directory 1: {kept_poses_1}", file=sys.stderr)
    print(f"Number of poses kept from pose directory 2: {kept_poses_2}", file=sys.stderr)

    _clean_output_dir(output_dir, force=force)
    identity = build_identity_set(
        kept1,
        kept2,
        common_offsets,
        readahead=readahead,
    )
    total_identity_poses = int(sum(len(keys) for keys in identity.values()))
    print(f"Number of unique poses: {total_identity_poses}", file=sys.stderr)
    print("Writing output pose directory...", file=sys.stderr)
    global_lookup = write_identity_pose_dir(
        identity,
        output_dir,
        bucket_size=bucket_size1,
        max_poses_per_file=max_poses_per_file,
        compress=compress,
    )
    print("Building mapping array 1...", file=sys.stderr)
    map1 = build_mapping(kept1, global_lookup, readahead=readahead)
    print(f"Number of mapping rows for pose directory 1: {len(map1)}", file=sys.stderr)
    print("Building mapping array 2...", file=sys.stderr)
    map2 = build_mapping(kept2, global_lookup, readahead=readahead)
    print(f"Number of mapping rows for pose directory 2: {len(map2)}", file=sys.stderr)
    np.save(output_dir / "map-1.npy", map1)
    np.save(output_dir / "map-2.npy", map2)

    manifest = {
        "pose_dir_1": str(pose_dir1),
        "pose_dir_2": str(pose_dir2),
        "bucket_size": bucket_size1,
        "input_poses_1": input_poses_1,
        "input_poses_2": input_poses_2,
        "kept_poses_1": kept_poses_1,
        "kept_poses_2": kept_poses_2,
        "common_buckets": len(common_buckets),
        "common_mini_buckets": common_mini_buckets,
        "identity_poses": total_identity_poses,
        "map_1_rows": int(len(map1)),
        "map_2_rows": int(len(map2)),
    }
    (output_dir / "identity-filter.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identity-filter two organized alaric pose directories."
    )
    parser.add_argument("pose_dir_1", type=Path)
    parser.add_argument("pose_dir_2", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete an existing non-empty output directory before writing.",
    )
    parser.add_argument(
        "--max-poses-per-file",
        type=int,
        default=100_000_000,
        help="Maximum number of output poses per organized .arc file.",
    )
    parser.add_argument(
        "--readahead",
        type=int,
        default=20,
        help=(
            "Maximum number of additional future .arc files to prefetch while "
            "building the identity set. Lower this on memory-constrained machines."
        ),
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Write output organized poses-*.arc.zst files instead of poses-*.arc.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_identity_filter(
        args.pose_dir_1,
        args.pose_dir_2,
        args.output_dir,
        force=args.force,
        max_poses_per_file=args.max_poses_per_file,
        readahead=args.readahead,
        compress=args.compress,
    )


if __name__ == "__main__":
    main()
