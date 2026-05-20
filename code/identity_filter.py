from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
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


def _load(entry: FileIndexEntry, cache: dict[Path, LoadedFile]) -> LoadedFile:
    loaded = cache.get(entry.path)
    if loaded is None:
        loaded = _read_loaded(entry)
        cache[entry.path] = loaded
    return loaded

import sortednp

def _intersect_sorted_unique(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if len(left) == 0 or len(right) == 0:
        return np.empty(0, dtype=np.uint32)
    return sortednp.intersect(
        left,
        right,
        duplicates=sortednp.IntersectDuplicates.DROP,
    )

def build_identity_set(
    entries1: list[FileIndexEntry],
    entries2: list[FileIndexEntry],
    common_offsets: dict[tuple[int, int, int], list[tuple[int, int, int]]],
) -> dict[PoseKey, np.ndarray]:
    nposes1 = sum(e.nP for e in entries1 if e.M in common_offsets)
    nposes2 = sum(e.nP for e in entries2 if e.M in common_offsets)
    small_entries, other_entries = (
        (entries1, entries2) if nposes1 <= nposes2 else (entries2, entries1)
    )
    other_by_bucket = _index_by_bucket(other_entries)
    other_offset_entries = _build_offset_entries(other_entries, common_offsets)

    WINDOW_AHEAD = 20

    holders_for_entry: list[list[FileIndexEntry]] = [[] for _ in small_entries]
    holder_paths_for_entry: list[set[Path]] = [set() for _ in small_entries]
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
        seen: set[Path] = set()
        holders: list[FileIndexEntry] = []
        for offset in entry.offset_to_index:
            if offset not in common_for_bucket:
                continue
            for other in other_offset_entries.get((entry.M, offset), []):
                if other.path in candidates_set and other.path not in seen:
                    seen.add(other.path)
                    holders.append(other)
        holders_for_entry[i] = holders
        holder_paths_for_entry[i] = seen

    path_to_entry: dict[Path, FileIndexEntry] = {}
    for entry in small_entries:
        path_to_entry[entry.path] = entry
    for entry in other_entries:
        path_to_entry[entry.path] = entry

    cache: dict[Path, LoadedFile] = {}
    refcount: dict[Path, int] = {}
    in_flight: set[Path] = set()
    lock = threading.Lock()
    cond = threading.Condition(lock)
    stop_event = threading.Event()
    loader_exc: list[BaseException] = []

    def refholder_paths(idx: int) -> list[Path]:
        if idx < 0 or idx >= len(small_entries):
            return []
        if not holders_for_entry[idx]:
            return []
        paths = [small_entries[idx].path]
        for other in holders_for_entry[idx]:
            paths.append(other.path)
        return paths

    def add_refs(idx: int) -> None:
        for p in refholder_paths(idx):
            refcount[p] = refcount.get(p, 0) + 1

    def remove_refs(idx: int) -> None:
        for p in refholder_paths(idx):
            c = refcount.get(p, 0) - 1
            if c <= 0:
                refcount.pop(p, None)
            else:
                refcount[p] = c

    def make_callback(p: Path):
        def callback(future: concurrent.futures.Future) -> None:
            with lock:
                in_flight.discard(p)
                try:
                    loaded = future.result()
                except BaseException as exc:
                    loader_exc.append(exc)
                    cond.notify_all()
                    return
                if refcount.get(p, 0) > 0 and p not in cache:
                    cache[p] = loaded
                cond.notify_all()
        return callback

    def loader() -> None:
        pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=_LOADER_WORKERS, thread_name_prefix="arc-loader"
        )
        try:
            while not stop_event.is_set():
                with lock:
                    stale = [
                        p for p in cache
                        if refcount.get(p, 0) == 0 and p not in in_flight
                    ]
                    for p in stale:
                        del cache[p]
                    pending = [
                        p for p in refcount
                        if p not in cache and p not in in_flight
                    ]
                    for p in pending:
                        in_flight.add(p)
                    entries_to_load = [(p, path_to_entry[p]) for p in pending]
                for p, entry_to_load in entries_to_load:
                    if stop_event.is_set():
                        break
                    future = pool.submit(_read_loaded, entry_to_load)
                    future.add_done_callback(make_callback(p))
                with lock:
                    if stop_event.is_set():
                        return
                    if not [
                        p for p in refcount
                        if p not in cache and p not in in_flight
                    ]:
                        cond.wait(timeout=0.1)
        except BaseException as exc:
            with lock:
                loader_exc.append(exc)
                cond.notify_all()
        finally:
            pool.shutdown(wait=True)

    def wait_for(path: Path) -> LoadedFile:
        with lock:
            while path not in cache:
                if loader_exc:
                    raise loader_exc[0]
                cond.wait()
            return cache[path]

    with lock:
        for j in range(min(WINDOW_AHEAD + 1, len(small_entries))):
            add_refs(j)
        cond.notify_all()

    loader_thread = threading.Thread(target=loader, daemon=True)
    loader_thread.start()

    identity: dict[PoseKey, np.ndarray] = {}
    try:
        for i, entry in enumerate(tqdm(small_entries, desc="Build identity set")):
            try:
                holders = holders_for_entry[i]
                if not holders:
                    continue

                loaded_entry = wait_for(entry.path)
                common_for_bucket = set(common_offsets[entry.M])
                holder_path_set = holder_paths_for_entry[i]

                for offset, offset_index in tqdm(entry.offset_to_index.items(), desc="Filter mini-bucket"):
                    if offset not in common_for_bucket:
                        continue
                    other_holders = [
                        other
                        for other in other_offset_entries.get((entry.M, offset), [])
                        if other.path in holder_path_set
                    ]
                    if not other_holders:
                        continue
                    left_keys = loaded_entry.pose_keys(offset_index)
                    other_parts = []
                    for other in other_holders:
                        loaded_other = wait_for(other.path)
                        other_parts.append(loaded_other.pose_keys(other.offset_to_index[offset]))
                    if len(other_parts) == 1:
                        right_keys = other_parts[0]
                    else:
                        right_keys = np.concatenate(other_parts)
                    overlap = _intersect_sorted_unique(left_keys, right_keys)
                    if len(overlap):
                        key = (entry.M, offset)
                        previous = identity.get(key)
                        identity[key] = (
                            overlap if previous is None else np.union1d(previous, overlap)
                        )
            finally:
                with lock:
                    remove_refs(i)
                    nxt = i + WINDOW_AHEAD + 1
                    if nxt < len(small_entries):
                        add_refs(nxt)
                    cond.notify_all()
    finally:
        stop_event.set()
        with lock:
            cond.notify_all()
        loader_thread.join()
    return identity


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
            write_arc_file(
                output_dir / f"poses-{file_index}.arc",
                np.array(M, dtype=np.int16),
                np.array(current_offsets, dtype=np.int16),
                np.array(current_counts, dtype=np.uint32),
                P,
                bucket_size=bucket_size,
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
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for entry in entries:
        relevant = [
            (offset, offset_index, global_lookup[(entry.M, offset)])
            for offset, offset_index in entry.offset_to_index.items()
            if (entry.M, offset) in global_lookup
        ]
        if not relevant:
            continue
        loaded = _load(entry, {})
        for _offset, offset_index, (dest_start, dest_keys) in relevant:
            keys = loaded.pose_keys(offset_index)
            positions = np.searchsorted(dest_keys, keys)
            in_range = positions < len(dest_keys)
            mask = np.zeros(len(keys), dtype=bool)
            if np.any(in_range):
                checked = positions[in_range]
                mask[in_range] = dest_keys[checked] == keys[in_range]
            if not np.any(mask):
                continue
            local_start = int(entry.I[offset_index])
            local_indices = np.nonzero(mask)[0].astype(np.uint64, copy=False)
            source_ids = (
                np.uint64(entry.global_start + local_start) + local_indices
            )
            dest_ids = np.uint64(dest_start) + positions[mask].astype(np.uint64, copy=False)
            chunks.append(np.column_stack((source_ids, dest_ids)).astype(np.uint64, copy=False))
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
) -> dict[str, int]:
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

    common_buckets = {entry.M for entry in entries1} & {entry.M for entry in entries2}
    kept1 = [entry for entry in entries1 if entry.M in common_buckets]
    kept2 = [entry for entry in entries2 if entry.M in common_buckets]
    common_offsets = _prepare_common_offsets(kept1, kept2)
    _assign_a_ranges(kept1, common_offsets)
    _assign_a_ranges(kept2, common_offsets)

    _clean_output_dir(output_dir, force=force)
    identity = build_identity_set(kept1, kept2, common_offsets)
    global_lookup = write_identity_pose_dir(
        identity,
        output_dir,
        bucket_size=bucket_size1,
        max_poses_per_file=max_poses_per_file,
    )
    map1 = build_mapping(kept1, global_lookup)
    map2 = build_mapping(kept2, global_lookup)
    np.save(output_dir / "map-1.npy", map1)
    np.save(output_dir / "map-2.npy", map2)

    total_identity_poses = int(sum(len(keys) for keys in identity.values()))
    manifest = {
        "pose_dir_1": str(pose_dir1),
        "pose_dir_2": str(pose_dir2),
        "bucket_size": bucket_size1,
        "input_poses_1": int(sum(entry.nP for entry in entries1)),
        "input_poses_2": int(sum(entry.nP for entry in entries2)),
        "kept_poses_1": int(sum(entry.nP for entry in kept1)),
        "kept_poses_2": int(sum(entry.nP for entry in kept2)),
        "common_buckets": len(common_buckets),
        "common_mini_buckets": int(sum(len(v) for v in common_offsets.values())),
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_identity_filter(
        args.pose_dir_1,
        args.pose_dir_2,
        args.output_dir,
        force=args.force,
        max_poses_per_file=args.max_poses_per_file,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
