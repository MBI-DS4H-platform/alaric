"""Stream an organized pose dir through a selector into a new organized pose dir.

Both routes of the middle-level ``filter`` action land here: filtering on a score
threshold (``filter-poses.py``) and selecting by mask/index array (``select-poses.py``).

Scale is the whole design. A pool holds tens of gigaposes with a per-pose score array to
match -- 15 gigaposes is a 60 GB float32 score file -- so nothing here may build a
per-pose array of the *input*, in Python or in NumPy. Instead:

* poses are read in bounded chunks straight off the ``.arc`` stream;
* the selector is asked only about the pose range currently in hand, so it reads only
  the matching slice of its score/mask file;
* only the *kept* poses are ever materialized, and they go straight into a
  ``PoseWriter`` shard.

Memory is therefore O(chunk + kept-per-shard), not O(nposes). Work is split per
organized input file, which is also the unit of parallelism: an organized file holds one
bucket, so each worker's kept poses flush as a single shard.

**Output order.** ``organize`` canonicalizes the output by (bucket, mini-bucket offset,
packed conformer/rotamer) -- the same key the input dir is already sorted by. A filtered
subsequence is therefore emitted in input order no matter what order the shards were
written in, which is what lets the input pose ids be recorded as a plain ascending array
instead of being tracked through organize (they are uint64: past 4.29 gigaposes they do
not fit organize's uint32 provenance sidecars). This holds only while the output bucket
size equals the input's, so the bucket size is read from the input rather than assumed.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
import tempfile

import numpy as np
from tqdm import tqdm

from npy_io import (
    NpyRangeReader,
    NpyWriter,
    find_npy,
    is_compressed,
    open_npy_mmap,
)
from nprocs import default_nprocs
from poses import (
    PoseWriter,
    discover_organized,
    iter_arc_pose_chunks,
    read_arc_header,
)


# Working-set size per step, not a filesystem request size: the pose stream and the
# score/mask array each read ahead in much larger blocks (poses.ARC_STREAM_READ_SIZE,
# npy_io.RANGE_READ_BLOCK), so this can stay small enough to keep the per-chunk arrays
# cheap.
DEFAULT_CHUNK_POSES = 1_000_000
# Kept poses buffered before a shard is flushed, shared across workers. Generous on
# purpose: a shard per flush means a small cache turns one input file into a dozen
# little shards for organize to merge, and a shared filesystem prefers fewer, bigger
# files. ~10 bytes per pose, so this is a few hundred MB in total.
DEFAULT_CACHE_POSES = 50_000_000
# PoseWriter provenance sidecars are uint32 (organize validates the dtype).
MAX_TAG = 2**32 - 1
_PROVENANCE_COPY_CHUNK = 4_000_000


@dataclass(frozen=True)
class FileTask:
    """One organized input file, and where it sits in the pool's pose numbering."""

    index: int
    path: Path
    global_start: int
    nposes: int


@dataclass(frozen=True)
class FilterStats:
    total_poses: int
    kept_poses: int
    shards: int


# -- selectors ------------------------------------------------------------
#
# A selector answers "which poses in [start, stop) do you keep?" with local offsets into
# that range, plus optional uint32 tags to record as PoseWriter provenance sidecars. It
# must read only the range it is asked about.


class _RangeSource:
    """Range reads from a per-pose array given either as a .npy path or as an array.

    A path is read with ``os.pread`` and never mapped -- that is the pool-scale case (a
    60 GB score file). An array (typically a mapping of a decompressed one, see
    :func:`open_range_source`) is simply sliced.
    """

    def __init__(self, source) -> None:
        if isinstance(source, (str, Path)):
            reader = NpyRangeReader(source)
            self._reader = reader
            self._array = None
            self.dtype = reader.dtype
            self._length = len(reader)
            self.name = str(source)
        else:
            array = np.asarray(source)
            if array.ndim != 1:
                raise ValueError("per-pose array must be 1D")
            self._reader = None
            self._array = array
            self.dtype = array.dtype
            self._length = len(array)
            self.name = "<array>"

    def __len__(self) -> int:
        return self._length

    def read(self, start: int, stop: int) -> np.ndarray:
        if self._reader is not None:
            return self._reader.read(start, stop)
        return self._array[start:stop]


def open_range_source(path: str | Path, stack: contextlib.ExitStack):
    """A per-pose .npy the selectors can range-read, named logically.

    Uncompressed files are read in place. A compressed one cannot be seeked into, so it
    is decompressed once into a temp file and mapped, for as long as ``stack`` lives --
    which must outlast the filtering, and must be the *parent's* stack so that forked
    workers never own the cleanup.
    """
    resolved = find_npy(path)
    if resolved is None:
        raise FileNotFoundError(f"no such array: {path}")
    if not is_compressed(resolved):
        return resolved
    return stack.enter_context(open_npy_mmap(resolved))


class ScoreThresholdSelector:
    """Keep poses whose score is below a threshold."""

    tagged = False

    def __init__(self, scores, threshold: float) -> None:
        self._source = _RangeSource(scores)
        if not np.issubdtype(self._source.dtype, np.number):
            raise ValueError(f"{self._source.name}: score array must be numeric")
        self.threshold = float(threshold)

    def __len__(self) -> int:
        return len(self._source)

    def kept(self, start: int, stop: int) -> tuple[np.ndarray, np.ndarray | None]:
        scores = self._source.read(start, stop)
        return np.flatnonzero(scores < self.threshold), None


class BoolMaskSelector:
    """Keep poses whose mask entry is true."""

    tagged = False

    def __init__(self, mask) -> None:
        self._source = _RangeSource(mask)
        if self._source.dtype != np.dtype(np.bool_):
            raise ValueError(f"{self._source.name}: expected a boolean mask")

    def __len__(self) -> int:
        return len(self._source)

    def kept(self, start: int, stop: int) -> tuple[np.ndarray, np.ndarray | None]:
        return np.flatnonzero(self._source.read(start, stop)), None


class IndexSelector:
    """Keep the poses named by a 0-based index array, tagged with request position.

    The tag is what makes the *order array* fall out of organize: each kept pose carries
    the position it had in the request, so organize's provenance step maps every
    organized pose back to it. Duplicate requests keep their multiplicity.

    An ascending index array (what ``mask.py`` emits) is used in place: only the range
    being processed is touched. An arbitrarily ordered one has to be sorted first, which
    costs memory proportional to the *selection* -- still bounded by the array the caller
    already holds.
    """

    tagged = True

    def __init__(self, indices: np.ndarray, nposes: int) -> None:
        indices = np.asarray(indices)
        if indices.ndim != 1:
            raise ValueError("pose index array must be 1D")
        if len(indices) > MAX_TAG:
            raise ValueError(
                f"cannot select more than {MAX_TAG} poses at once "
                f"({len(indices)} requested)"
            )
        self._nposes = int(nposes)
        if _is_ascending(indices):
            self._sorted = indices
            self._request_position = None
        else:
            order = np.argsort(indices, kind="stable")
            self._sorted = indices[order]
            self._request_position = order
        if len(self._sorted) and int(self._sorted[-1]) >= self._nposes:
            raise ValueError(
                f"pose index {int(self._sorted[-1])} exceeds number of poses ({nposes})"
            )

    def __len__(self) -> int:
        return self._nposes

    @property
    def nselected(self) -> int:
        return len(self._sorted)

    def kept(self, start: int, stop: int) -> tuple[np.ndarray, np.ndarray | None]:
        first = int(np.searchsorted(self._sorted, start, side="left"))
        last = int(np.searchsorted(self._sorted, stop, side="left"))
        if first == last:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.uint32)
        local = np.asarray(self._sorted[first:last], dtype=np.int64) - int(start)
        if self._request_position is None:
            tags = np.arange(first, last, dtype=np.uint32)
        else:
            tags = self._request_position[first:last].astype(np.uint32, copy=False)
        return local, tags


def _is_ascending(values: np.ndarray, *, chunk: int = 10_000_000) -> bool:
    """Non-decreasing check that never materializes a whole-array diff."""
    previous = None
    for start in range(0, len(values), chunk):
        block = np.asarray(values[start : start + chunk])
        if previous is not None and block[0] < previous:
            return False
        if len(block) > 1 and np.any(block[1:] < block[:-1]):
            return False
        previous = block[-1]
    return True


# -- per-file filtering ---------------------------------------------------


def file_tasks(pose_dir: Path) -> tuple[list[FileTask], int, int]:
    """Enumerate the organized input files with their global pose offsets."""
    paths = discover_organized(pose_dir)
    if not paths:
        raise FileNotFoundError(f"No organized poses-*.arc files found in {pose_dir}")
    tasks: list[FileTask] = []
    bucket_size: int | None = None
    global_start = 0
    for index, path in enumerate(paths):
        _M, _nO, nposes, file_bucket_size = read_arc_header(path)
        if bucket_size is None:
            bucket_size = int(file_bucket_size)
        elif int(file_bucket_size) != bucket_size:
            raise ValueError(
                f"inconsistent bucket_size in {pose_dir}: expected {bucket_size}, "
                f"got {file_bucket_size} in {path}"
            )
        tasks.append(FileTask(index, path, global_start, int(nposes)))
        global_start += int(nposes)
    assert bucket_size is not None
    return tasks, bucket_size, global_start


def _filter_one_file(
    task: FileTask,
    *,
    out_dir: Path,
    selector,
    bucket_size: int,
    chunk_poses: int,
    cache_poses: int,
    provenance_dir: Path | None,
) -> tuple[int, int, int]:
    writer = PoseWriter(out_dir, bucket_size=bucket_size, cache_poses=cache_poses)
    kept_ids: list[np.ndarray] | None = [] if provenance_dir is not None else None
    kept_total = 0
    cursor = task.global_start
    for M, O, _C, P, file_bucket_size in iter_arc_pose_chunks(
        task.path, rows_per_chunk=chunk_poses
    ):
        if int(file_bucket_size) != bucket_size:
            raise ValueError(f"bucket_size changed while reading {task.path}")
        stop = cursor + len(P)
        local, tags = selector.kept(cursor, stop)
        if len(local):
            rows = P[local]
            grid = O.astype(np.int32) + bucket_size * M.astype(np.int32)
            translations = grid[rows[:, 2].astype(np.int64)]
            writer.add_chunk(rows[:, 0], rows[:, 1], translations, provenance=tags)
            if kept_ids is not None:
                kept_ids.append(local.astype(np.uint64) + np.uint64(cursor))
            kept_total += len(local)
        cursor = stop
    if cursor != task.global_start + task.nposes:
        raise ValueError(
            f"{task.path}: read {cursor - task.global_start} poses, "
            f"expected {task.nposes}"
        )
    shards = writer.finish()
    if provenance_dir is not None:
        part = (
            np.concatenate(kept_ids)
            if kept_ids
            else np.empty(0, dtype=np.uint64)
        )
        np.save(provenance_dir / f"{task.index:08d}.npy", part)
    return task.index, kept_total, len(shards)


_WORKER_STATE: dict = {}


def _init_worker(state: dict) -> None:
    _WORKER_STATE.clear()
    _WORKER_STATE.update(state)


def _run_file_task(task: FileTask) -> tuple[int, int, int]:
    return _filter_one_file(task, **_WORKER_STATE)


# -- provenance -----------------------------------------------------------


def _write_provenance(
    provenance_dir: Path,
    tasks: list[FileTask],
    kept_total: int,
    output_path: Path,
    *,
    compress: bool,
) -> Path:
    """Concatenate the per-file kept-id parts, in file order, into one array.

    File order is ascending pose id, and the organized output is in input order (see the
    module docstring), so the concatenation is already the provenance of the result. The
    parts are spilled to disk rather than kept in memory because a .npy header needs the
    final length up front, and the length is only known once the scan is done.
    """
    with NpyWriter(
        output_path, dtype=np.uint64, shape=(kept_total,), compress=compress
    ) as out:
        for task in tasks:
            part = np.load(provenance_dir / f"{task.index:08d}.npy", mmap_mode="r")
            try:
                for start in range(0, len(part), _PROVENANCE_COPY_CHUNK):
                    out.write(part[start : start + _PROVENANCE_COPY_CHUNK])
            finally:
                del part
        return out.path


# -- driver ---------------------------------------------------------------


def filter_pose_dir(
    pose_dir: str | Path,
    out_dir: str | Path,
    selector,
    *,
    provenance_path: str | Path | None = None,
    compress: bool = False,
    nprocs: int | None = None,
    chunk_poses: int = DEFAULT_CHUNK_POSES,
    cache_poses: int = DEFAULT_CACHE_POSES,
    progress: bool = True,
) -> FilterStats:
    """Filter ``pose_dir`` into unorganized shards in ``out_dir``.

    The caller organizes ``out_dir`` afterwards (this only writes the shards, so the
    caller keeps control of the organize knobs). ``provenance_path`` names the logical
    (uncompressed) path for the kept input pose ids; pass None to skip recording them.
    """
    pose_dir = Path(pose_dir)
    out_dir = Path(out_dir)
    tasks, bucket_size, total_poses = file_tasks(pose_dir)
    if len(selector) != total_poses:
        raise ValueError(
            f"selector covers {len(selector)} poses but {pose_dir} holds {total_poses}"
        )
    if chunk_poses <= 0:
        raise ValueError("chunk_poses must be positive")
    if cache_poses <= 0:
        raise ValueError("cache_poses must be positive")

    nprocs = default_nprocs() if nprocs is None else max(1, int(nprocs))
    workers = max(1, min(nprocs, len(tasks)))
    out_dir.mkdir(parents=True, exist_ok=True)

    kept_total = 0
    shards = 0
    with contextlib.ExitStack() as stack:
        provenance_dir = None
        if provenance_path is not None:
            provenance_dir = Path(
                stack.enter_context(
                    tempfile.TemporaryDirectory(prefix="alaric-filter-provenance-")
                )
            )
        state = {
            "out_dir": out_dir,
            "selector": selector,
            "bucket_size": bucket_size,
            "chunk_poses": int(chunk_poses),
            # Each worker runs its own PoseWriter, so share the budget out.
            "cache_poses": max(1, int(cache_poses) // workers),
            "provenance_dir": provenance_dir,
        }
        bar = stack.enter_context(
            tqdm(
                total=len(tasks),
                desc="Filter poses",
                unit="file",
                disable=not progress,
            )
        )
        if workers == 1:
            results = (_filter_one_file(task, **state) for task in tasks)
        else:
            ctx = mp.get_context("fork")
            pool = stack.enter_context(
                ctx.Pool(workers, initializer=_init_worker, initargs=(state,))
            )
            # Hand out *contiguous* runs of files. Tasks are in pose order, so a run is
            # one contiguous span of the score/mask file: each worker then reads it as a
            # long sequential stream instead of interleaving with the others, which is
            # what a network filesystem serves best. Several batches per worker so a
            # straggler still leaves work for the others to pick up.
            batch = max(1, -(-len(tasks) // (workers * 4)))
            results = pool.imap_unordered(_run_file_task, tasks, chunksize=batch)
        for _index, kept, n_shards in results:
            kept_total += kept
            shards += n_shards
            bar.update(1)

        if provenance_path is not None:
            assert provenance_dir is not None
            _write_provenance(
                provenance_dir,
                tasks,
                kept_total,
                Path(provenance_path),
                compress=compress,
            )

    return FilterStats(total_poses=total_poses, kept_poses=kept_total, shards=shards)
