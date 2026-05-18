from __future__ import annotations

import io
import os
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
import tempfile
from typing import Iterable

import numpy as np


MAGIC = b"alaric1"
HEADER_SIZE = 16
MAX_NP = 2**32 - 1
MAX_NO = 2**16
ARC_SUFFIX = ".arc"
ARC_ZSTD_SUFFIX = ".arc.zst"


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


def split_M_O(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split absolute grid coordinates into .arc M and O components."""
    grid = _as_grid_array(grid)
    M = ((grid + 128) // 256).astype(np.int32, copy=False)
    O = grid - 256 * M
    if M.size and (M.min() < -128 or M.max() > 127):
        raise ValueError("M values must fit in signed int8")
    if O.size and (O.min() < -128 or O.max() > 127):
        raise ValueError("O values must fit in signed int8")
    return M.astype(np.int8, copy=False), O.astype(np.int8, copy=False)


def _validate_arc_arrays(
    M: np.ndarray, O: np.ndarray, C: np.ndarray, P: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    M = np.asarray(M)
    if M.shape != (3,):
        raise ValueError("M must have shape (3,)")
    if M.size and (M.min() < -128 or M.max() > 127):
        raise ValueError("M values must fit in signed int8")
    M = M.astype(np.int8, copy=False)

    O = np.asarray(O)
    if O.ndim != 2 or O.shape[1] != 3:
        raise ValueError("O must be a [nO,3] array")
    if len(O) > MAX_NO:
        raise ValueError("nO must be <= 65536")
    if O.size and (O.min() < -128 or O.max() > 127):
        raise ValueError("O values must fit in signed int8")
    O = O.astype(np.int8, copy=False)

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


def read_arc_header(path: str | Path) -> tuple[np.ndarray, int, int]:
    raw = _open_arc_bytes(path)
    if len(raw) < HEADER_SIZE:
        raise ValueError(f".arc file is too small: {path}")
    if raw[:7] != MAGIC:
        raise ValueError(f"bad .arc magic in {path}")
    M = np.frombuffer(raw[7:10], dtype=np.int8).copy()
    nO_raw = int(np.frombuffer(raw[10:12], dtype="<u2")[0])
    nO = MAX_NO if nO_raw == 0 else nO_raw
    nP = int(np.frombuffer(raw[12:16], dtype="<u4")[0])
    return M, nO, nP


def read_arc_file(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = _open_arc_bytes(path)
    M, nO, nP = read_arc_header(path)
    expected = HEADER_SIZE + nO * 3 + nO * 4 + nP * 6
    if len(raw) != expected:
        raise ValueError(
            f"bad .arc length for {path}: expected {expected} bytes, got {len(raw)}"
        )
    pos = HEADER_SIZE
    O = np.frombuffer(raw[pos : pos + nO * 3], dtype=np.int8).copy().reshape(nO, 3)
    pos += nO * 3
    C = np.frombuffer(raw[pos : pos + nO * 4], dtype="<u4").copy()
    pos += nO * 4
    P = np.frombuffer(raw[pos : pos + nP * 6], dtype="<u2").copy().reshape(nP, 3)
    return _validate_arc_arrays(M, O, C, P)


def write_arc_file(
    path: str | Path,
    M: np.ndarray,
    O: np.ndarray,
    C: np.ndarray,
    P: np.ndarray,
    *,
    zstd: bool | None = None,
) -> None:
    M, O, C, P = _validate_arc_arrays(M, O, C, P)
    if len(P) == 0:
        raise ValueError("empty .arc files are not written")

    path = Path(path)
    if zstd is None:
        zstd = path.name.endswith(ARC_ZSTD_SUFFIX)
    nO_raw = 0 if len(O) == MAX_NO else len(O)

    payload = io.BytesIO()
    payload.write(MAGIC)
    payload.write(M.tobytes(order="C"))
    payload.write(np.array([nO_raw], dtype="<u2").tobytes())
    payload.write(np.array([len(P)], dtype="<u4").tobytes())
    payload.write(O.tobytes(order="C"))
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
            handle.write(zstd_mod.ZstdCompressor().compress(data))
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
    sort_offsets: bool = True,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Pack expanded pose rows into one .arc tuple per M bucket."""
    conf = _as_uint16_array("conformer_indices", conformer_indices)
    rot = _as_uint16_array("rotamer_indices", rotamer_indices)
    if conf.shape != rot.shape:
        raise ValueError("conformer_indices and rotamer_indices must have same shape")
    grid = _as_grid_array(translations)
    if len(grid) != len(conf):
        raise ValueError("translations length must match conformer/rotamer indices")
    if len(grid) == 0:
        return []

    Ms, Os = split_M_O(grid)
    buckets: dict[tuple[int, int, int], list[np.ndarray]] = defaultdict(list)
    for m in np.unique(Ms, axis=0):
        mask = np.all(Ms == m, axis=1)
        rows = np.column_stack((Os[mask], conf[mask], rot[mask]))
        buckets[tuple(int(x) for x in m)].append(rows)

    packed = []
    for m_key in sorted(buckets):
        rows = np.concatenate(buckets[m_key], axis=0)
        offsets = rows[:, :3].astype(np.int8, copy=False)
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
        packed.append((np.array(m_key, dtype=np.int8), O, C, P))
    return packed


def decode_pool(
    M: np.ndarray, O: np.ndarray, C: np.ndarray, P: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    M, O, C, P = _validate_arc_arrays(M, O, C, P)
    grid = O.astype(np.int16) + 256 * M.astype(np.int16)
    translations = grid[P[:, 2].astype(np.int64)]
    return (
        P[:, 0].astype(np.uint16, copy=False),
        P[:, 1].astype(np.uint16, copy=False),
        translations.astype(np.int16, copy=False),
    )


def discover_unorganized(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(directory.glob("unorganized-*.arc")) + sorted(
        directory.glob("unorganized-*.arc.zst")
    )


def _pose_index_from_arc_name(name: str) -> int | None:
    if not name.startswith("poses-") or not name.endswith(ARC_SUFFIX):
        return None
    text = name[len("poses-") : -len(ARC_SUFFIX)]
    return int(text) if text.isdigit() else None


def discover_organized(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    indexed = []
    for path in directory.glob("poses-*.arc"):
        index = _pose_index_from_arc_name(path.name)
        if index is not None:
            indexed.append((index, path))
    return [path for _, path in sorted(indexed)]


class PoseWriter:
    """Process-local unorganized .arc.zst shard writer."""

    def __init__(
        self,
        outdir: str | Path,
        *,
        cache_poses: int = 50_000_000,
        memory_lock=None,
    ) -> None:
        self.outdir = Path(outdir)
        self.cache_poses = int(cache_poses)
        if self.cache_poses <= 0:
            raise ValueError("cache_poses must be positive")
        self._memory_lock = memory_lock
        self._unsorted_chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._buckets: dict[
            tuple[int, int, int],
            list[tuple[np.ndarray, np.ndarray, np.ndarray]],
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
    ) -> None:
        conf = _as_uint16_array("conformer_indices", conformer_indices)
        rot = _as_uint16_array("rotamer_indices", rotamer_indices)
        if conf.shape != rot.shape:
            raise ValueError("conformer_indices and rotamer_indices must have same shape")
        grid = _as_grid_array(translations)
        if len(grid) != len(conf):
            raise ValueError("translations length must match conformer/rotamer indices")
        if len(grid) == 0:
            return

        self._unsorted_chunks.append(
            (
                conf.copy(),
                rot.copy(),
                grid.astype(np.int16, copy=True),
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
            name = f"unorganized-{os.getpid():x}-{np.random.bytes(4).hex()}.arc.zst"
            path = self.outdir / name
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
            Ms, Os = split_M_O(grid)
            unique_M = np.unique(Ms, axis=0)
            for M in unique_M:
                mask = np.all(Ms == M, axis=1)
                key = tuple(int(x) for x in M)
                self._buckets[key].append(
                    (
                        conf[mask].copy(),
                        rot[mask].copy(),
                        Os[mask].astype(np.int8, copy=True),
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
            O, inverse = np.unique(O_rows, axis=0, return_inverse=True)
            C = np.bincount(inverse, minlength=len(O)).astype(np.uint32)
            P = np.column_stack((conf, rot, inverse.astype(np.uint16, copy=False)))
            M = np.array(key, dtype=np.int8)

        path = self._next_unorganized_path()
        write_arc_file(path, M, O, C, P, zstd=True)
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
        M, O, C, P = read_arc_file(path)
        conf, rot, translations = decode_pool(M, O, C, P)
        yield np.column_stack((translations, conf, rot))
