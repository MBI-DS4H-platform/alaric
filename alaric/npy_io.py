"""Compression-transparent .npy I/O for alaric's bulk index arrays.

Poses are not the only per-pose data alaric writes: every pool also carries index
arrays -- grow/filter ``provenance.npy``, the identity-filter ``map-*.npy``, the
organize order array, the anchor precalculation caches. At gigapose scale those are
4-8 bytes per pose, i.e. *larger* than the ~1.5 bytes/pose the compressed poses
themselves take, so they are stored as zstd-compressed ``.npy.zst``.

Compression is transparent everywhere it matters:

* The *logical* name is always the uncompressed ``.npy`` name. That is what the pool
  graph JSON records and what a Seamless deepfolder INDEX entry is keyed on
  (``strip_compression_suffix``), so compressing an array changes neither the graph
  nor the result checksum.
* Readers resolve either form (:func:`find_npy`), so pose dirs produced before
  compression was enabled keep working.

Writes are published by rename, matching ``poses.write_arc_file``: a reader either
sees the previous file or the complete new one.
"""

from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
import tempfile

import numpy as np


NPY_SUFFIX = ".npy"
ZSTD_SUFFIX = ".zst"
NPY_ZSTD_SUFFIX = ".npy.zst"

# Chunk size for streaming (de)compression of array payloads.
_STREAM_CHUNK_BYTES = 4 * 1024 * 1024


def _require_zstandard(action: str):
    try:
        import zstandard as zstd
    except ImportError as exc:
        raise ImportError(f"zstandard is required to {action} {NPY_ZSTD_SUFFIX} files") from exc
    return zstd


def is_compressed(path: str | Path) -> bool:
    return str(path).endswith(ZSTD_SUFFIX)


def compressed_path(path: str | Path) -> Path:
    """The ``.zst`` name for a logical ``.npy`` path (idempotent)."""
    path = Path(path)
    if is_compressed(path):
        return path
    return path.with_name(path.name + ZSTD_SUFFIX)


def find_npy(path: str | Path) -> Path | None:
    """Resolve a logical ``.npy`` path to the form present on disk, or None.

    Callers pass the uncompressed name they expect; a ``.npy.zst`` sibling counts.
    """
    path = Path(path)
    if path.is_file():
        return path
    candidate = compressed_path(path)
    if candidate != path and candidate.is_file():
        return candidate
    return None


@contextlib.contextmanager
def _open_stream(path: Path):
    with path.open("rb") as handle:
        if is_compressed(path):
            zstd = _require_zstandard("read")
            with zstd.ZstdDecompressor().stream_reader(handle) as reader:
                yield reader
        else:
            yield handle


def _read_header(stream, path: Path):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version in {(2, 0), (3, 0)}:
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(f"unsupported .npy version {version} in {path}")
    if dtype.hasobject:
        raise ValueError(f"object arrays are not supported in {path}")
    return shape, fortran_order, dtype


def read_npy_header(path: str | Path) -> tuple[tuple[int, ...], np.dtype]:
    """Return ``(shape, dtype)`` without reading the payload.

    Validating a large array's dtype/length is otherwise a full decompression.
    """
    path = Path(path)
    with _open_stream(path) as stream:
        shape, _fortran_order, dtype = _read_header(stream, path)
    return shape, dtype


def load_npy(path: str | Path, *, mmap: bool = False) -> np.ndarray:
    """Load an array from either the compressed or the uncompressed form.

    ``mmap`` is honoured for an uncompressed file only -- a compressed array is
    always materialized in memory. Use :func:`open_npy_mmap` when the mapping is
    required (large array, random access).
    """
    path = Path(path)
    if not is_compressed(path):
        return np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)

    with _open_stream(path) as stream:
        shape, fortran_order, dtype = _read_header(stream, path)
        order = "F" if fortran_order else "C"
        result = np.empty(shape, dtype=dtype, order=order)
        target = memoryview(result.ravel(order=order)).cast("B")
        position = 0
        while position < len(target):
            count = stream.readinto(target[position:])
            if not count:
                raise ValueError(f"truncated compressed .npy array: {path}")
            position += count
        if stream.read(1):
            raise ValueError(f"trailing data after compressed .npy array: {path}")
        return result


@contextlib.contextmanager
def open_npy_mmap(path: str | Path, *, tempdir: str | Path | None = None):
    """Memory-map an array, decompressing to a temp file first when compressed.

    Keeps peak memory bounded where an array is consumed by slices rather than as a
    whole (organize's per-shard provenance). The temp file goes to ``tempdir`` or
    ``$TMPDIR`` -- node-local scratch under the remote deployers -- and is removed on
    exit, so the mapped array must not outlive the context.
    """
    path = Path(path)
    if not is_compressed(path):
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            yield array
        finally:
            del array
        return

    zstd = _require_zstandard("read")
    with tempfile.NamedTemporaryFile(
        prefix="alaric-npy.", suffix=NPY_SUFFIX, dir=tempdir, delete=False
    ) as handle:
        tmp_path = Path(handle.name)
        try:
            with path.open("rb") as compressed:
                zstd.ZstdDecompressor().copy_stream(
                    compressed, handle, write_size=_STREAM_CHUNK_BYTES
                )
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
    try:
        array = np.load(tmp_path, mmap_mode="r", allow_pickle=False)
        try:
            yield array
        finally:
            del array
    finally:
        tmp_path.unlink(missing_ok=True)


# Filesystem request size for a range-read array. Consumers ask for whatever range
# they are working on; the reader fetches at least this much, so the request size seen
# by the filesystem does not follow the caller's chunk size. One buffer per reader per
# process, so the cost is this times the worker count.
RANGE_READ_BLOCK = 32 * 1024 * 1024


class NpyRangeReader:
    """Positioned range reads from a large uncompressed 1-D .npy.

    For a per-pose array that cannot be held in memory -- a 15-gigapose float32 score
    file is 60 GB -- read only the range currently being processed. Deliberately not
    ``mmap``: this is a one-shot sequential scan, often of a file on a network
    filesystem and often from several processes at once, which is the case mapping
    serves worst.

    Access is ascending, so ranges are served from a read-ahead block: the filesystem
    sees ``block`` -sized requests however small the caller's ranges are. Reads go
    through ``os.pread``, so the descriptor stays correct when the reader is inherited
    by forked workers (a shared file *offset* would not be).
    """

    def __init__(self, path: str | Path, *, block: int = RANGE_READ_BLOCK) -> None:
        path = Path(path)
        if is_compressed(path):
            raise ValueError(f"{path}: a compressed array cannot be range-read")
        with path.open("rb") as handle:
            shape, fortran_order, dtype = _read_header(handle, path)
            self._payload_offset = handle.tell()
        if len(shape) != 1:
            raise ValueError(f"{path}: expected a 1-D array, got shape {shape}")
        if fortran_order:
            raise ValueError(f"{path}: Fortran-ordered arrays are not supported")
        self.path = path
        self.dtype = dtype
        self._length = int(shape[0])
        self._items_per_block = max(1, int(block) // dtype.itemsize)
        self._fd: int | None = None
        self._buffer: np.ndarray | None = None
        self._buffer_start = 0
        self._buffer_stop = 0

    def __len__(self) -> int:
        return self._length

    def _descriptor(self) -> int:
        if self._fd is None:
            self._fd = os.open(self.path, os.O_RDONLY)
        return self._fd

    def _pread(self, start: int, stop: int) -> np.ndarray:
        itemsize = self.dtype.itemsize
        offset = self._payload_offset + start * itemsize
        remaining = (stop - start) * itemsize
        fd = self._descriptor()
        parts: list[bytes] = []
        while remaining:
            block = os.pread(fd, remaining, offset)
            if not block:
                raise ValueError(f"{self.path}: truncated at byte {offset}")
            parts.append(block)
            offset += len(block)
            remaining -= len(block)
        raw = parts[0] if len(parts) == 1 else b"".join(parts)
        return np.frombuffer(raw, dtype=self.dtype)

    def read(self, start: int, stop: int) -> np.ndarray:
        start = max(0, int(start))
        stop = min(self._length, int(stop))
        if stop <= start:
            return np.empty(0, dtype=self.dtype)
        if self._buffer is not None and self._buffer_start <= start and stop <= self._buffer_stop:
            return self._buffer[start - self._buffer_start : stop - self._buffer_start]
        # Refill: at least the requested range, at least one block, never past the end.
        buffer_stop = min(self._length, max(stop, start + self._items_per_block))
        self._buffer = self._pread(start, buffer_stop)
        self._buffer_start = start
        self._buffer_stop = buffer_stop
        return self._buffer[: stop - start]

    def close(self) -> None:
        self._buffer = None
        self._buffer_start = self._buffer_stop = 0
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __getstate__(self) -> dict:
        # An fd is meaningless in another process, and the read-ahead is per-process
        # state; both are re-established on first use there.
        state = dict(self.__dict__)
        state["_fd"] = None
        state["_buffer"] = None
        state["_buffer_start"] = 0
        state["_buffer_stop"] = 0
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)


class NpyWriter:
    """Write a .npy sequentially, optionally zstd-compressed, published by rename.

    ``np.lib.format.open_memmap`` cannot write into a compression stream, so code
    that fills an array chunk by chunk in order (organize's provenance reorder) goes
    through this instead. ``shape``/``dtype`` are declared up front because a .npy
    header must precede the payload.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        dtype,
        shape: tuple[int, ...],
        compress: bool = True,
    ) -> None:
        self.path = compressed_path(path) if compress else Path(path)
        self.dtype = np.dtype(dtype)
        self.shape = tuple(int(n) for n in shape)
        self._remaining = int(np.prod(self.shape)) if self.shape else 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = tempfile.NamedTemporaryFile(
            prefix=f"{self.path.name}.",
            suffix=".tmp",
            dir=self.path.parent,
            delete=False,
        )
        self._tmp_path = Path(self._tmp.name)
        # Version 1.0 specifically: it is what ``np.save``/``open_memmap`` emit for these
        # (numeric, few-dimensional) arrays, and the uncompressed bytes are what result
        # checksums are taken over -- a 2.0 header would silently change them.
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(
            header,
            {
                "descr": np.lib.format.dtype_to_descr(self.dtype),
                "fortran_order": False,
                "shape": self.shape,
            },
        )
        header_bytes = header.getvalue()
        self._compressor = None
        handle = self._tmp
        if compress:
            zstd = _require_zstandard("write")
            # Declare the uncompressed size in the frame header, as the .arc.zst writer
            # does, so `zstd -l` and friends can report it without decompressing.
            self._compressor = zstd.ZstdCompressor().stream_writer(
                self._tmp,
                size=len(header_bytes) + self._remaining * self.dtype.itemsize,
            )
            handle = self._compressor
        self._handle = handle
        handle.write(header_bytes)

    def write(self, chunk: np.ndarray) -> None:
        chunk = np.ascontiguousarray(chunk, dtype=self.dtype)
        if chunk.size > self._remaining:
            raise ValueError(f"{self.path}: more elements written than declared")
        self._remaining -= chunk.size
        self._handle.write(chunk.tobytes(order="C"))

    def close(self) -> None:
        try:
            # Before closing the stream: a short write also trips zstd's own declared-size
            # check, and its error says nothing about which array is incomplete.
            if self._remaining:
                raise ValueError(
                    f"{self.path}: {self._remaining} declared elements were never written"
                )
            if self._compressor is not None:
                self._compressor.close()
            self._tmp.close()
            self._tmp_path.replace(self.path)
        finally:
            with contextlib.suppress(Exception):
                self._tmp.close()
            self._tmp_path.unlink(missing_ok=True)

    def __enter__(self) -> "NpyWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            with contextlib.suppress(Exception):
                if self._compressor is not None:
                    self._compressor.close()
                self._tmp.close()
            self._tmp_path.unlink(missing_ok=True)
            return
        self.close()


def save_npy(path: str | Path, array: np.ndarray, *, compress: bool = True) -> Path:
    """Write ``array`` to the logical ``.npy`` ``path``, or to ``path.zst``.

    Returns the path actually written.
    """
    array = np.ascontiguousarray(array)
    with NpyWriter(
        path, dtype=array.dtype, shape=array.shape, compress=compress
    ) as writer:
        flat = array.reshape(-1)
        step = max(1, _STREAM_CHUNK_BYTES // max(1, array.dtype.itemsize))
        for start in range(0, len(flat), step):
            writer.write(flat[start : start + step])
        return writer.path


def compress_npy_file(path: str | Path) -> Path:
    """Replace an existing uncompressed ``.npy`` with its ``.npy.zst`` form.

    For arrays that have to be *built* uncompressed (the organize order array is a
    scattered, multi-process memmap) but are only ever read back whole.
    """
    path = Path(path)
    if is_compressed(path):
        return path
    zstd = _require_zstandard("write")
    dest = compressed_path(path)
    with tempfile.NamedTemporaryFile(
        prefix=f"{dest.name}.", suffix=".tmp", dir=dest.parent, delete=False
    ) as handle:
        tmp_path = Path(handle.name)
        try:
            with path.open("rb") as raw:
                zstd.ZstdCompressor().copy_stream(
                    raw, handle, size=path.stat().st_size, write_size=_STREAM_CHUNK_BYTES
                )
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
    try:
        tmp_path.replace(dest)
    finally:
        tmp_path.unlink(missing_ok=True)
    path.unlink()
    return dest
