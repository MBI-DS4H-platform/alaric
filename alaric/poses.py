from __future__ import annotations

import io
import os
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Iterable, Iterator

import numpy as np

from npy_io import save_npy


MAGIC = b"alaric1"
HEADER_SIZE = 21
MAX_NP = 2**32 - 1
MAX_NO = 2**16
MAX_BUCKET_SIZE = 2**16 - 1
DEFAULT_BUCKET_SIZE = 16
ARC_SUFFIX = ".arc"
ARC_ZSTD_SUFFIX = ".arc.zst"
# Per-shard provenance sidecar, written next to an unorganized shard and consumed by
# organize.py (which folds the sidecars into the pose dir's single provenance array).
# Stored zstd-compressed; readers resolve either form through npy_io.find_npy.
PROVENANCE_SUFFIX = ".provenance.npy"
ARC_ZSTD_FRAME_BYTES = 4 * 1024 * 1024
DEFAULT_POSE_CHUNK_SIZE = 10_000


def _require_zstandard(action: str):
    try:
        import zstandard as zstd
    except ImportError as exc:
        raise ImportError(f"zstandard is required to {action} .arc.zst files") from exc
    return zstd


def _as_uint16_array(name: str, arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if arr.size and (arr.min() < 0 or arr.max() > np.iinfo(np.uint16).max):
        raise ValueError(f"{name} values must fit in uint16")
    return arr.astype(np.uint16, copy=False)


def _as_grid_array(translations: np.ndarray) -> np.ndarray:
    translations = np.asarray(translations)
    if translations.ndim != 2 or translations.shape[1] != 3:
        raise ValueError("translations must be a [N,3] array")
    if translations.size and (
        translations.min() < -32768 or translations.max() > 32767
    ):
        raise ValueError("grid coordinates must fit in int16")
    return translations.astype(np.int32, copy=False)


def _check_bucket_size(bucket_size: int) -> int:
    bucket_size = int(bucket_size)
    if bucket_size < 1 or bucket_size > MAX_BUCKET_SIZE:
        raise ValueError(f"bucket_size must be in 1..{MAX_BUCKET_SIZE}")
    return bucket_size


def split_M_O(grid: np.ndarray, bucket_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Split absolute grid coordinates into .arc M and O components."""
    bucket_size = _check_bucket_size(bucket_size)
    half = bucket_size // 2
    grid = _as_grid_array(grid)
    M = ((grid + half) // bucket_size).astype(np.int32, copy=False)
    O = grid - bucket_size * M
    if M.size and (M.min() < -32768 or M.max() > 32767):
        raise ValueError("M values must fit in signed int16")
    if O.size and (O.min() < -32768 or O.max() > 32767):
        raise ValueError("O values must fit in signed int16")
    return M.astype(np.int16, copy=False), O.astype(np.int16, copy=False)


def _validate_arc_arrays(
    M: np.ndarray,
    O: np.ndarray,
    C: np.ndarray,
    P: np.ndarray,
    bucket_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bucket_size = _check_bucket_size(bucket_size)
    half = bucket_size // 2

    M = np.asarray(M)
    if M.shape != (3,):
        raise ValueError("M must have shape (3,)")
    if M.size and (M.min() < -32768 or M.max() > 32767):
        raise ValueError("M values must fit in signed int16")
    M = M.astype(np.int16, copy=False)

    O = np.asarray(O)
    if O.ndim != 2 or O.shape[1] != 3:
        raise ValueError("O must be a [nO,3] array")
    if len(O) > MAX_NO:
        raise ValueError("nO must be <= 65536")
    max_offset = bucket_size - half - 1
    if O.size and (O.min() < -half or O.max() > max_offset):
        raise ValueError(
            f"O values must fit in [-{half}, {max_offset}] for bucket_size {bucket_size}"
        )
    O = O.astype(np.int16, copy=False)

    C = np.asarray(C)
    if C.ndim != 1:
        raise ValueError("C must be a 1D array")
    if len(C) != len(O):
        raise ValueError("C length must equal nO")
    if C.size and (C.min() < 0 or C.max() > MAX_NP):
        raise ValueError("C values must fit in uint32")
    C = C.astype(np.uint32, copy=False)

    P = np.asarray(P)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("P must be a [nP,3] array")
    if len(P) > MAX_NP:
        raise ValueError("nP must be <= 2**32 - 1")
    if P.size and (P.min() < 0 or P.max() > np.iinfo(np.uint16).max):
        raise ValueError("P values must fit in uint16")
    P = P.astype(np.uint16, copy=False)

    if int(C.sum(dtype=np.uint64)) != len(P):
        raise ValueError("sum(C) must equal nP")
    if len(P) and (len(O) == 0 or int(P[:, 2].max()) >= len(O)):
        raise ValueError("P offset indices must be < nO")
    return M, O, C, P


def _open_arc_bytes(path: str | Path) -> bytes:
    path = Path(path)
    if path.name.endswith(ARC_ZSTD_SUFFIX):
        zstd = _require_zstandard("read")
        with path.open("rb") as compressed:
            with zstd.ZstdDecompressor().stream_reader(compressed) as reader:
                return reader.read()
    return path.read_bytes()


def _read_arc_prefix(path: str | Path, size: int) -> bytes:
    path = Path(path)
    if path.name.endswith(ARC_ZSTD_SUFFIX):
        zstd = _require_zstandard("read")
        with path.open("rb") as compressed:
            with zstd.ZstdDecompressor().stream_reader(compressed) as reader:
                data = reader.read(size)
    else:
        with path.open("rb") as handle:
            data = handle.read(size)
    if len(data) != size:
        raise ValueError(f".arc file is too small: {path}")
    return data


def _read_exact(handle, size: int, path: str | Path) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = handle.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    data = b"".join(chunks)
    if len(data) != size:
        raise ValueError(f".arc file is too small: {path}")
    return data


def _parse_arc_header(raw: bytes, path: str | Path) -> tuple[np.ndarray, int, int, int]:
    if len(raw) < HEADER_SIZE:
        raise ValueError(f".arc file is too small: {path}")
    if raw[:7] != MAGIC:
        raise ValueError(f"bad .arc magic in {path}")
    bucket_size = int(np.frombuffer(raw[7:9], dtype="<u2")[0])
    if bucket_size < 1:
        raise ValueError(f"bucket_size must be >= 1 in {path}")
    M = np.frombuffer(raw[9:15], dtype="<i2").copy()
    nO_raw = int(np.frombuffer(raw[15:17], dtype="<u2")[0])
    nO = MAX_NO if nO_raw == 0 else nO_raw
    nP = int(np.frombuffer(raw[17:21], dtype="<u4")[0])
    return M, nO, nP, bucket_size


def read_arc_header(path: str | Path) -> tuple[np.ndarray, int, int, int]:
    return _parse_arc_header(_read_arc_prefix(path, HEADER_SIZE), path)


def read_arc_offsets(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    M, nO, nP, bucket_size = read_arc_header(path)
    raw = _read_arc_prefix(path, HEADER_SIZE + nO * 6 + nO * 4)
    pos = HEADER_SIZE
    O = np.frombuffer(raw[pos : pos + nO * 6], dtype="<i2").copy().reshape(nO, 3)
    pos += nO * 6
    C = np.frombuffer(raw[pos : pos + nO * 4], dtype="<u4").copy()
    if int(C.sum(dtype=np.uint64)) != nP:
        raise ValueError(f"sum(C) must equal nP in {path}")
    return M, O, C, nP, bucket_size


def read_arc_file(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    raw = _open_arc_bytes(path)
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


def iter_arc_pose_chunks(
    path: str | Path,
    *,
    rows_per_chunk: int = 1_000_000,
) -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]]:
    """Yield one .arc file as bounded pose chunks.

    The metadata arrays are copied once and reused across chunks. Pose chunks are
    copied out of the input stream so callers can safely keep them after the next
    read.
    """
    if rows_per_chunk <= 0:
        raise ValueError("rows_per_chunk must be positive")

    path = Path(path)
    if path.name.endswith(ARC_ZSTD_SUFFIX):
        zstd = _require_zstandard("read")
        compressed = path.open("rb")
        stream_context = zstd.ZstdDecompressor().stream_reader(compressed)
    else:
        compressed = None
        stream_context = path.open("rb")

    try:
        with stream_context as stream:
            header = _read_exact(stream, HEADER_SIZE, path)
            M, nO, nP, bucket_size = _parse_arc_header(header, path)
            O_raw = _read_exact(stream, nO * 6, path)
            O = np.frombuffer(O_raw, dtype="<i2").copy().reshape(nO, 3)
            C_raw = _read_exact(stream, nO * 4, path)
            C = np.frombuffer(C_raw, dtype="<u4").copy()
            if int(C.sum(dtype=np.uint64)) != nP:
                raise ValueError(f"sum(C) must equal nP in {path}")

            remaining = nP
            while remaining:
                take = min(rows_per_chunk, remaining)
                raw = _read_exact(stream, take * 6, path)
                P = np.frombuffer(raw, dtype="<u2").copy().reshape(take, 3)
                if len(P) and (len(O) == 0 or int(P[:, 2].max()) >= len(O)):
                    raise ValueError(f"P offset indices must be < nO in {path}")
                yield M, O, C, P, bucket_size
                remaining -= take

            extra = stream.read(1)
            if extra:
                raise ValueError(f"bad .arc length for {path}: trailing data")
    finally:
        if compressed is not None:
            compressed.close()


def write_arc_file(
    path: str | Path,
    M: np.ndarray,
    O: np.ndarray,
    C: np.ndarray,
    P: np.ndarray,
    *,
    bucket_size: int,
    zstd: bool | None = None,
) -> None:
    bucket_size = _check_bucket_size(bucket_size)
    M, O, C, P = _validate_arc_arrays(M, O, C, P, bucket_size)
    if len(P) == 0:
        raise ValueError("empty .arc files are not written")

    path = Path(path)
    if zstd is None:
        zstd = path.name.endswith(ARC_ZSTD_SUFFIX)
    nO_raw = 0 if len(O) == MAX_NO else len(O)

    payload = io.BytesIO()
    payload.write(MAGIC)
    payload.write(np.array([bucket_size], dtype="<u2").tobytes())
    payload.write(M.astype("<i2", copy=False).tobytes(order="C"))
    payload.write(np.array([nO_raw], dtype="<u2").tobytes())
    payload.write(np.array([len(P)], dtype="<u4").tobytes())
    payload.write(O.astype("<i2", copy=False).tobytes(order="C"))
    payload.write(C.astype("<u4", copy=False).tobytes(order="C"))
    payload.write(P.astype("<u2", copy=False).tobytes(order="C"))
    data = payload.getvalue()

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f"{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        tmp_path = Path(handle.name)
        if zstd:
            zstd_mod = _require_zstandard("write")
            compressor = zstd_mod.ZstdCompressor()
            view = memoryview(data)
            for offset in range(0, len(data), ARC_ZSTD_FRAME_BYTES):
                handle.write(
                    compressor.compress(view[offset : offset + ARC_ZSTD_FRAME_BYTES])
                )
        else:
            handle.write(data)
    try:
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def pack_pool(
    conformer_indices: np.ndarray,
    rotamer_indices: np.ndarray,
    translations: np.ndarray,
    *,
    bucket_size: int,
    sort_offsets: bool = True,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Pack expanded pose rows into one .arc tuple per M bucket."""
    bucket_size = _check_bucket_size(bucket_size)
    conf = _as_uint16_array("conformer_indices", conformer_indices)
    rot = _as_uint16_array("rotamer_indices", rotamer_indices)
    if conf.shape != rot.shape:
        raise ValueError("conformer_indices and rotamer_indices must have same shape")
    grid = _as_grid_array(translations)
    if len(grid) != len(conf):
        raise ValueError("translations length must match conformer/rotamer indices")
    if len(grid) == 0:
        return []

    Ms, Os = split_M_O(grid, bucket_size)
    buckets: dict[tuple[int, int, int], list[np.ndarray]] = defaultdict(list)
    for m in np.unique(Ms, axis=0):
        mask = np.all(Ms == m, axis=1)
        rows = np.column_stack((Os[mask], conf[mask], rot[mask]))
        buckets[tuple(int(x) for x in m)].append(rows)

    packed = []
    for m_key in sorted(buckets):
        rows = np.concatenate(buckets[m_key], axis=0)
        offsets = rows[:, :3].astype(np.int16, copy=False)
        conf_rot = rows[:, 3:5].astype(np.uint16, copy=False)
        O, inverse = np.unique(offsets, axis=0, return_inverse=True)
        if sort_offsets:
            order = np.lexsort((O[:, 2], O[:, 1], O[:, 0]))
            remap = np.empty(len(order), dtype=np.uint16)
            remap[order] = np.arange(len(order), dtype=np.uint16)
            O = O[order]
            inverse = remap[inverse]
        C = np.bincount(inverse, minlength=len(O)).astype(np.uint32)
        P = np.column_stack((conf_rot, inverse.astype(np.uint16, copy=False)))
        packed.append((np.array(m_key, dtype=np.int16), O, C, P))
    return packed


def decode_pool(
    M: np.ndarray, O: np.ndarray, C: np.ndarray, P: np.ndarray, bucket_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bucket_size = _check_bucket_size(bucket_size)
    M, O, C, P = _validate_arc_arrays(M, O, C, P, bucket_size)
    grid = O.astype(np.int32) + bucket_size * M.astype(np.int32)
    translations = grid[P[:, 2].astype(np.int64)]
    return (
        P[:, 0].astype(np.uint16, copy=False),
        P[:, 1].astype(np.uint16, copy=False),
        translations.astype(np.int16, copy=False),
    )


def select_pose_indices(pose_dir: str | Path, pose_indices: np.ndarray) -> PoseChunk:
    """Return selected 0-based pose indices from an organized pose directory.

    The requested indices are sorted for efficient .arc access, then the
    returned rows are restored to the original request order.
    """
    indices = np.asarray(pose_indices)
    if indices.ndim != 1:
        raise ValueError("pose indices must be a 1D array")
    if np.issubdtype(indices.dtype, np.signedinteger) and indices.size:
        if int(indices.min()) < 0:
            raise ValueError("pose indices must be non-negative")
    indices = indices.astype(np.uint64, copy=False)

    paths = discover_organized(pose_dir)
    if not paths:
        raise FileNotFoundError(f"No organized poses-*.arc files found in {pose_dir}")

    files: list[_PoseFile] = []
    global_start = 0
    for path in paths:
        _, _, nposes, _ = read_arc_header(path)
        files.append(_PoseFile(path=path, nposes=int(nposes), global_start=global_start))
        global_start += int(nposes)

    if len(indices) == 0:
        return PoseReader._empty_chunk()
    if int(indices.max()) >= global_start:
        raise ValueError(
            f"pose index {int(indices.max())} exceeds number of poses ({global_start})"
        )

    sort_order = np.argsort(indices, kind="stable")
    sorted_indices = indices[sort_order]
    restore_order = np.empty(len(sort_order), dtype=np.intp)
    restore_order[sort_order] = np.arange(len(sort_order), dtype=np.intp)

    conformers: list[np.ndarray] = []
    rotamers: list[np.ndarray] = []
    translations: list[np.ndarray] = []

    for file_entry in files:
        file_start = file_entry.global_start
        file_stop = file_start + file_entry.nposes
        first = int(np.searchsorted(sorted_indices, file_start, side="left"))
        last = int(np.searchsorted(sorted_indices, file_stop, side="left"))
        if first == last:
            continue

        M, O, _C, P, bucket_size = read_arc_file(file_entry.path)
        rel = (sorted_indices[first:last] - np.uint64(file_start)).astype(
            np.intp,
            copy=False,
        )
        P_selected = P[rel]
        grid = O.astype(np.int32) + bucket_size * M.astype(np.int32)
        selected_translations = grid[P_selected[:, 2].astype(np.int64)]
        conformers.append(P_selected[:, 0].astype(np.uint16, copy=False))
        rotamers.append(P_selected[:, 1].astype(np.uint16, copy=False))
        translations.append(selected_translations.astype(np.int16, copy=False))

    if not conformers:
        raise RuntimeError("no selected poses were read")

    return PoseChunk(
        conformers=np.concatenate(conformers)[restore_order],
        rotamers=np.concatenate(rotamers)[restore_order],
        translations_grid=np.concatenate(translations)[restore_order],
    )


def discover_unorganized(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    paths: set[Path] = set()
    for pattern in (
        "unorganized-*.arc",
        "unorganized-*.arc.zst",
        "unorganized-*/*.arc",
        "unorganized-*/*.arc.zst",
    ):
        paths.update(directory.glob(pattern))
    return sorted(paths)


def _pose_index_from_arc_name(name: str) -> int | None:
    if name.startswith("poses-") and name.endswith(ARC_ZSTD_SUFFIX):
        text = name[len("poses-") : -len(ARC_ZSTD_SUFFIX)]
        return int(text) if text.isdigit() else None
    if not name.startswith("poses-") or not name.endswith(ARC_SUFFIX):
        return None
    text = name[len("poses-") : -len(ARC_SUFFIX)]
    return int(text) if text.isdigit() else None


def discover_organized(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    indexed = []
    for path in list(directory.glob("poses-*.arc")) + list(
        directory.glob("poses-*.arc.zst")
    ):
        index = _pose_index_from_arc_name(path.name)
        if index is not None:
            indexed.append((index, path))
    return [path for _, path in sorted(indexed)]


@dataclass(frozen=True)
class PoseChunk:
    conformers: np.ndarray
    rotamers: np.ndarray
    translations_grid: np.ndarray

    def __len__(self) -> int:
        return len(self.conformers)


@dataclass(frozen=True)
class _PoseFile:
    path: Path
    nposes: int
    global_start: int


class PoseReader:
    """Streaming reader for organized pose directories."""

    @staticmethod
    def get_nposes(pose_dir: str | Path) -> int:
        paths = discover_organized(pose_dir)
        if not paths:
            raise FileNotFoundError(f"No organized poses-*.arc files found in {pose_dir}")
        total = 0
        for path in paths:
            _, _, nposes, _ = read_arc_header(path)
            total += int(nposes)
        return total

    def __init__(
        self,
        pose_dir: str | Path,
        *,
        pose_range: tuple[int, int] | None = None,
        rows_per_chunk: int = DEFAULT_POSE_CHUNK_SIZE,
    ) -> None:
        if rows_per_chunk <= 0:
            raise ValueError("rows_per_chunk must be positive")

        self.pose_dir = Path(pose_dir)
        self.rows_per_chunk = int(rows_per_chunk)
        self.paths = discover_organized(self.pose_dir)
        if not self.paths:
            raise FileNotFoundError(
                f"No organized poses-*.arc files found in {self.pose_dir}"
            )

        files = []
        global_start = 0
        for path in self.paths:
            _, _, nposes, _ = read_arc_header(path)
            files.append(
                _PoseFile(
                    path=path,
                    nposes=int(nposes),
                    global_start=global_start,
                )
            )
            global_start += int(nposes)
        self.files = files
        self.total_poses = global_start

        if pose_range is None:
            self.start0 = 0
            self.stop0 = self.total_poses
        else:
            start, end = pose_range
            if start <= 0 or end <= 0:
                raise ValueError("pose range values must be positive")
            if start > end:
                raise ValueError("pose range START must be <= END")
            self.start0 = start - 1
            self.stop0 = end
            if self.start0 >= self.total_poses:
                raise ValueError(
                    f"pose range starts after the last pose ({self.total_poses})"
                )
            if self.stop0 > self.total_poses:
                raise ValueError(
                    f"pose range END={end} exceeds number of poses ({self.total_poses})"
                )
        self._pos0 = self.start0
        self._stream: Iterator[PoseChunk] | None = None
        self._pending: PoseChunk | None = None

    @property
    def selected_poses(self) -> int:
        return max(0, self.stop0 - self.start0)

    def tell(self) -> int:
        return self._pos0 - self.start0

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            relative = offset
        elif whence == 1:
            relative = self.tell() + offset
        elif whence == 2:
            relative = self.selected_poses + offset
        else:
            raise ValueError("whence must be 0, 1, or 2")
        if relative < 0 or relative > self.selected_poses:
            raise ValueError("seek position out of range")
        self._pos0 = self.start0 + relative
        self._stream = None
        self._pending = None
        return self.tell()

    def read(self, size: int = -1) -> PoseChunk:
        remaining = self.stop0 - self._pos0
        if size is None or size < 0:
            size = remaining
        else:
            size = min(int(size), remaining)
        if size < 0:
            raise ValueError("read size must be non-negative")
        if size == 0:
            return self._empty_chunk()

        chunks = []
        needed = size
        while needed:
            if self._pending is None or len(self._pending) == 0:
                if self._stream is None:
                    self._stream = self._iter_absolute_chunks(self._pos0, self.stop0)
                try:
                    self._pending = next(self._stream)
                except StopIteration:
                    break

            take = min(needed, len(self._pending))
            chunks.append(self._slice_chunk(self._pending, 0, take))
            self._pending = self._slice_chunk(self._pending, take, len(self._pending))
            self._pos0 += take
            needed -= take

        if not chunks:
            return self._empty_chunk()
        return self._concat_chunks(chunks)

    def iter_chunks(self) -> Iterator[PoseChunk]:
        while True:
            chunk = self.read(self.rows_per_chunk)
            if len(chunk) == 0:
                break
            yield chunk

    def _iter_absolute_chunks(self, start0: int, stop0: int) -> Iterator[PoseChunk]:
        for file_entry in self.files:
            file_start = file_entry.global_start
            file_stop = file_start + file_entry.nposes
            if file_stop <= start0:
                continue
            if file_start >= stop0:
                break

            chunk_start = file_start
            for M, O, _C, P, bucket_size in iter_arc_pose_chunks(
                file_entry.path,
                rows_per_chunk=self.rows_per_chunk,
            ):
                chunk_stop = chunk_start + len(P)
                take_start = max(start0, chunk_start)
                take_stop = min(stop0, chunk_stop)
                if take_start < take_stop:
                    rel_start = take_start - chunk_start
                    rel_stop = take_stop - chunk_start
                    P_slice = P[rel_start:rel_stop]
                    grid = O.astype(np.int32) + bucket_size * M.astype(np.int32)
                    translations = grid[P_slice[:, 2].astype(np.int64)]
                    yield PoseChunk(
                        conformers=P_slice[:, 0].astype(np.uint16, copy=False),
                        rotamers=P_slice[:, 1].astype(np.uint16, copy=False),
                        translations_grid=translations.astype(np.int16, copy=False),
                    )
                chunk_start = chunk_stop

    @staticmethod
    def _empty_chunk() -> PoseChunk:
        return PoseChunk(
            conformers=np.empty((0,), dtype=np.uint16),
            rotamers=np.empty((0,), dtype=np.uint16),
            translations_grid=np.empty((0, 3), dtype=np.int16),
        )

    @staticmethod
    def _slice_chunk(chunk: PoseChunk, start: int, stop: int) -> PoseChunk:
        return PoseChunk(
            conformers=chunk.conformers[start:stop],
            rotamers=chunk.rotamers[start:stop],
            translations_grid=chunk.translations_grid[start:stop],
        )

    @staticmethod
    def _concat_chunks(chunks: list[PoseChunk]) -> PoseChunk:
        if len(chunks) == 1:
            return chunks[0]
        return PoseChunk(
            conformers=np.concatenate([chunk.conformers for chunk in chunks]),
            rotamers=np.concatenate([chunk.rotamers for chunk in chunks]),
            translations_grid=np.concatenate(
                [chunk.translations_grid for chunk in chunks]
            ),
        )


class PoseWriter:
    """Process-local unorganized .arc.zst shard writer."""

    def __init__(
        self,
        outdir: str | Path,
        *,
        cache_poses: int = 50_000_000,
        bucket_size: int = DEFAULT_BUCKET_SIZE,
        memory_lock=None,
        unorganized_subdirs: bool = False,
    ) -> None:
        self.outdir = Path(outdir)
        self.cache_poses = int(cache_poses)
        if self.cache_poses <= 0:
            raise ValueError("cache_poses must be positive")
        self.bucket_size = _check_bucket_size(bucket_size)
        self._memory_lock = memory_lock
        self.unorganized_subdirs = bool(unorganized_subdirs)
        self._unsorted_chunks: list[
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]
        ] = []
        self._buckets: dict[
            tuple[int, int, int],
            list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]],
        ] = defaultdict(list)
        self._bucket_counts: dict[tuple[int, int, int], int] = defaultdict(int)
        self.total_poses = 0
        self._unsorted_poses = 0
        self._sorted_poses = 0
        self._written: list[Path] = []

    @property
    def _cached_poses(self) -> int:
        return 7 * self._unsorted_poses + self._sorted_poses

    def _memory_guard(self):
        if self._memory_lock is None:
            return nullcontext()
        return self._memory_lock

    def add_chunk(
        self,
        conformer_indices: np.ndarray,
        rotamer_indices: np.ndarray,
        translations: np.ndarray,
        provenance: np.ndarray | None = None,
    ) -> None:
        conf = _as_uint16_array("conformer_indices", conformer_indices)
        rot = _as_uint16_array("rotamer_indices", rotamer_indices)
        if conf.shape != rot.shape:
            raise ValueError("conformer_indices and rotamer_indices must have same shape")
        grid = _as_grid_array(translations)
        if len(grid) != len(conf):
            raise ValueError("translations length must match conformer/rotamer indices")
        provenance_arr = None
        if provenance is not None:
            provenance_arr = np.asarray(provenance, dtype=np.uint32)
            if provenance_arr.ndim != 1:
                raise ValueError("provenance must be a 1D array")
            if len(provenance_arr) != len(conf):
                raise ValueError("provenance length must match conformer/rotamer indices")
        if len(grid) == 0:
            return

        self._unsorted_chunks.append(
            (
                conf.copy(),
                rot.copy(),
                grid.astype(np.int16, copy=True),
                None if provenance_arr is None else provenance_arr.copy(),
            )
        )
        n = len(grid)
        self._unsorted_poses += n
        self.total_poses += n

        while self._cached_poses >= self.cache_poses:
            self._sort_unsorted_chunks()
            if self._cached_poses < self.cache_poses:
                break
            largest = max(self._bucket_counts, key=self._bucket_counts.get)
            self._flush_bucket(largest)

    def _next_unorganized_path(self) -> Path:
        self.outdir.mkdir(parents=True, exist_ok=True)
        while True:
            prefix = f"unorganized-{os.getpid():x}"
            token = np.random.bytes(4).hex()
            if self.unorganized_subdirs:
                path = self.outdir / prefix / f"{token}.arc.zst"
            else:
                path = self.outdir / f"{prefix}-{token}.arc.zst"
            if not path.exists():
                return path

    def _sort_unsorted_chunks(self) -> None:
        with self._memory_guard():
            chunks = self._unsorted_chunks
            if not chunks:
                return
            self._unsorted_chunks = []
            sorted_count = sum(len(chunk[0]) for chunk in chunks)

            conf = np.concatenate([chunk[0] for chunk in chunks])
            rot = np.concatenate([chunk[1] for chunk in chunks])
            grid = np.concatenate([chunk[2] for chunk in chunks])
            provenance_parts = [chunk[3] for chunk in chunks]
            has_provenance = [part is not None for part in provenance_parts]
            if any(has_provenance) and not all(has_provenance):
                raise ValueError("cannot mix chunks with and without provenance")
            provenance = (
                None
                if not any(has_provenance)
                else np.concatenate(
                    [part for part in provenance_parts if part is not None]
                )
            )
            Ms, Os = split_M_O(grid, self.bucket_size)
            unique_M = np.unique(Ms, axis=0)
            for M in unique_M:
                mask = np.all(Ms == M, axis=1)
                key = tuple(int(x) for x in M)
                self._buckets[key].append(
                    (
                        conf[mask].copy(),
                        rot[mask].copy(),
                        Os[mask].astype(np.int16, copy=True),
                        None if provenance is None else provenance[mask].copy(),
                    )
                )
                self._bucket_counts[key] += int(mask.sum())
            self._unsorted_poses -= sorted_count
            self._sorted_poses += sorted_count

    def _flush_bucket(self, key: tuple[int, int, int]) -> None:
        parts = self._buckets.pop(key, [])
        count = self._bucket_counts.pop(key, 0)
        if count == 0:
            return
        self._sorted_poses -= count
        with self._memory_guard():
            conf = np.concatenate([p[0] for p in parts])
            rot = np.concatenate([p[1] for p in parts])
            O_rows = np.concatenate([p[2] for p in parts])
            provenance_parts = [p[3] for p in parts]
            has_provenance = [part is not None for part in provenance_parts]
            if any(has_provenance) and not all(has_provenance):
                raise ValueError("cannot mix buckets with and without provenance")
            provenance = (
                None
                if not any(has_provenance)
                else np.concatenate(
                    [part for part in provenance_parts if part is not None]
                )
            )
            O, inverse = np.unique(O_rows, axis=0, return_inverse=True)
            C = np.bincount(inverse, minlength=len(O)).astype(np.uint32)
            P = np.column_stack((conf, rot, inverse.astype(np.uint16, copy=False)))
            M = np.array(key, dtype=np.int16)

        path = self._next_unorganized_path()
        write_arc_file(path, M, O, C, P, bucket_size=self.bucket_size, zstd=True)
        if provenance is not None:
            # Compressed like the shard itself: at 4 bytes/pose an uncompressed
            # sidecar would outweigh the .arc.zst it belongs to.
            save_npy(path.with_name(path.name + PROVENANCE_SUFFIX), provenance)
        self._written.append(path)

    def finish(self) -> list[Path]:
        self._sort_unsorted_chunks()
        for key in list(self._bucket_counts):
            self._flush_bucket(key)
        return list(self._written)

    def cleanup(self) -> None:
        self._unsorted_chunks.clear()
        self._buckets.clear()
        self._bucket_counts.clear()
        self._unsorted_poses = 0
        self._sorted_poses = 0


def iter_decoded_rows(paths: Iterable[str | Path]) -> Iterable[np.ndarray]:
    for path in paths:
        M, O, C, P, bucket_size = read_arc_file(path)
        conf, rot, translations = decode_pool(M, O, C, P, bucket_size)
        yield np.column_stack((translations, conf, rot))
