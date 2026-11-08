#!/usr/bin/env python3
"""Chunked energy-filter: per-chunk speculative cascade, then a whole-action replay.

The energy-filter contract counts the cascade's undecided poses over the whole action,
never per chunk, so that the route through the cascade -- and therefore the published
statistics, which are part of the result checksum -- does not depend on chunking.
Independently submitted chunk jobs cannot see each other's counts, so:

``chunk``
    Scores one global, 1-based pose range. Unless the whole input pool is already below
    ``min_cascade`` (a global fact every chunk can check), it runs *every* grid step on
    its local undecided poses, ignoring ``min_cascade``, and then the gridless pass on
    whatever it has left undecided. Each step's white, pending and precision-fallback
    ids are recorded as global 0-based ids under ``CHUNKS_DIR/chunk-<N>/``.

``assemble``
    Replays ``energy_filter_action._cascade_minfor`` -- the very function a whole-action
    run uses -- with an oracle that answers each grid step from the chunk records. The
    replay takes the whole-action exit decisions on the summed counts. The chunks'
    speculative gridless set is always a subset of the replay's gridless set: an early
    whole-action exit only adds poses (fewer than ``min_cascade`` in total), and only
    those are scored here. The published pool and statistics are the whole-action ones.

Speculation beyond an early exit only costs grid passes on fewer than ``min_cascade``
poses in total. Each chunk builds its own grids.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

import energy_filter_action as efa

RECORD_SCHEMA = 1
_EMPTY = np.empty(0, dtype=np.uint64)


def chunk_range(total: int, index: int, nchunks: int) -> tuple[int, int]:
    """Inclusive 1-based range of chunk ``index`` (1..nchunks); empty when first > last."""
    return (total * (index - 1)) // nchunks + 1, (total * index) // nchunks


def _settings(args, threshold, steps, minimum, exclude) -> dict:
    return {
        "threshold": threshold, "epsilon": float(args.epsilon), "cascade": steps,
        "min_cascade": minimum, "sequence": args.sequence, "exclude": exclude,
        "fragment": int(args.fragment), "protein_range": str(args.protein_range),
        "protein_sha256": hashlib.sha256(Path(args.protein).read_bytes()).hexdigest(),
    }


def _validated(args):
    threshold, steps, minimum, exclude = efa._validate_settings(
        args.threshold, args.epsilon, json.loads(args.cascade), args.min_cascade,
        args.pdb_exclude, args.nprocs, args.chunk_poses,
    )
    if args.nchunks <= 0:
        raise ValueError("nchunks must be positive")
    return threshold, steps, minimum, exclude


def _total_poses(pose_dir: Path) -> int:
    from poses import PoseReader

    return int(PoseReader.get_nposes(pose_dir))


def _save(directory: Path, name: str, values: np.ndarray) -> None:
    np.save(directory / name, np.asarray(values, dtype=np.uint64), allow_pickle=False)


def run_chunk(args) -> dict:
    threshold, steps, minimum, exclude = _validated(args)
    if not 1 <= args.chunk_index <= args.nchunks:
        raise ValueError("chunk index must be within 1..nchunks")
    total = _total_poses(args.pose_dir)
    first, last = chunk_range(total, args.chunk_index, args.nchunks)
    count = max(0, last - first + 1)
    offset = np.uint64(first - 1)
    final = args.chunks_dir / f"chunk-{args.chunk_index}"
    partial = args.chunks_dir / f"chunk-{args.chunk_index}.partial"
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    record = {
        "schema": RECORD_SCHEMA, "nchunks": args.nchunks, "chunk_index": args.chunk_index,
        "total": total, "first": first, "last": last,
        "settings": _settings(args, threshold, steps, minimum, exclude),
        "steps": 0, "gridless_all": False, "minfor_invocations": 0,
    }
    if count:
        with tempfile.TemporaryDirectory(prefix="energy-filter-chunk-") as temporary:
            work = Path(temporary)
            efa._prepare_minfor_inputs(args.pose_dir, args.protein, args.sequence, exclude, work,
                                       pose_range=(first, last))
            prepared = np.load(work / "poses.rotvec.npy", mmap_mode="r", allow_pickle=False)
            if len(prepared) != count:
                raise ValueError(f"converted {len(prepared)} poses for range {first}-{last}")
            del prepared
            invoke = efa._MinforInvoker(work, count, protein_range=args.protein_range,
                                        epsilon=float(args.epsilon), threshold=threshold,
                                        chunk_poses=args.chunk_poses)
            pending, precision = None, []
            # A pool below min_cascade exits before the first step whatever the chunking;
            # every later exit depends on whole-action counts, so speculate through them.
            if total >= minimum:
                for index, step in enumerate(steps):
                    if pending is not None and not len(pending):
                        break
                    white, grey, unsafe = invoke(pending, step)
                    pending, fallback = grey[~unsafe], grey[unsafe]
                    precision.append(fallback)
                    _save(partial, f"step-{index}.white.npy", white + offset)
                    _save(partial, f"step-{index}.pending.npy", pending + offset)
                    _save(partial, f"step-{index}.precision.npy", fallback + offset)
                    record["steps"] = index + 1
            if pending is None:
                exact = None
                record["gridless_all"] = True
            else:
                exact = np.sort(np.concatenate([pending, *precision]))
                _save(partial, "gridless.input.npy", exact + offset)
            white = _EMPTY
            if exact is None or len(exact):
                white, grey, _ = invoke(exact, None)
                if len(grey):
                    raise ValueError("zero-tolerance gridless filtering returned grey poses")
            _save(partial, "gridless.white.npy", white + offset)
            record["minfor_invocations"] = invoke.count
    else:
        _save(partial, "gridless.white.npy", _EMPTY)
        record["gridless_all"] = True
    (partial / "record.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    shutil.rmtree(final, ignore_errors=True)
    partial.rename(final)
    return record


class _ChunkRecord:
    def __init__(self, directory: Path, record: dict):
        self.directory, self.record = directory, record
        self.first, self.last, self.steps = record["first"], record["last"], record["steps"]
        self.count = max(0, self.last - self.first + 1)

    def load(self, name: str) -> np.ndarray:
        values = np.load(self.directory / name, allow_pickle=False)
        if values.ndim != 1 or values.dtype != np.dtype("uint64"):
            raise ValueError(f"{self.directory / name}: invalid chunk record array")
        if len(values) and (values.min() < self.first - 1 or values.max() >= self.last):
            raise ValueError(f"{self.directory / name}: ids outside chunk range")
        return values

    def pending_before(self, index: int) -> np.ndarray | None:
        """This chunk's undecided ids before grid step ``index`` (None: all of them)."""
        if index == 0:
            return None
        return self.load(f"step-{index - 1}.pending.npy")

    def step(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if index < self.steps:
            return tuple(self.load(f"step-{index}.{kind}.npy")
                         for kind in ("white", "pending", "precision"))
        # The chunk stopped early: valid only if it had nothing left to send to this step.
        before = None if index == 0 else (
            self.load(f"step-{index - 1}.pending.npy") if index - 1 < self.steps else _EMPTY)
        if (self.count if before is None else len(before)):
            raise ValueError(f"{self.directory}: chunk did not run cascade step {index}")
        return _EMPTY, _EMPTY, _EMPTY

    def gridless(self) -> tuple[np.ndarray, np.ndarray]:
        if self.record["gridless_all"]:
            scored = np.arange(self.first - 1, self.last, dtype=np.uint64)
        else:
            scored = self.load("gridless.input.npy")
        return scored, self.load("gridless.white.npy")


def _load_records(args, settings, total) -> list[_ChunkRecord]:
    records = []
    missing = []
    for index in range(1, args.nchunks + 1):
        directory = args.chunks_dir / f"chunk-{index}"
        path = directory / "record.json"
        if not path.is_file():
            missing.append(str(directory))
            continue
        record = json.loads(path.read_text())
        first, last = chunk_range(total, index, args.nchunks)
        expected = {"schema": RECORD_SCHEMA, "nchunks": args.nchunks, "chunk_index": index,
                    "total": total, "first": first, "last": last, "settings": settings}
        stale = sorted(key for key, value in expected.items() if record.get(key) != value)
        if stale:
            # The chunks dir is keyed on the sigil alone, so a chunk left by an earlier
            # deploy at another nchunks (or a never-started chunk job) shows up here.
            raise ValueError(f"{directory}: stale chunk record (mismatched {', '.join(stale)}); "
                             "re-run that chunk")
        records.append(_ChunkRecord(directory, record))
    if missing:
        sample = ", ".join(missing[:5])
        extra = "" if len(missing) <= 5 else f" ... ({len(missing)} missing)"
        raise FileNotFoundError(f"missing energy-filter chunk(s): {sample}{extra}")
    return records


class _ReplayInvoker:
    """Answer ``_cascade_minfor``'s minfor calls from the chunk records."""

    def __init__(self, records, total, args, threshold, exclude):
        self.records, self.total, self.args = records, total, args
        self.threshold, self.exclude = threshold, exclude
        self.count = 0
        self.grid_calls = 0
        self.rescored = 0

    def __call__(self, indices, step):
        self.count += 1
        if step is not None:
            return self._grid(indices)
        return self._gridless(indices)

    def _grid(self, indices):
        index = self.grid_calls
        self.grid_calls += 1
        if index == 0:
            if indices is not None:
                raise ValueError("replay: first cascade step must cover the whole pool")
        else:
            expected = np.concatenate([r.pending_before(index) if r.steps >= index else _EMPTY
                                       for r in self.records])
            if not np.array_equal(np.asarray(indices, dtype=np.uint64), expected):
                raise ValueError(f"replay: undecided poses before step {index} differ from the chunk records")
        parts = [r.step(index) for r in self.records]
        white = np.concatenate([p[0] for p in parts])
        pending = np.concatenate([p[1] for p in parts])
        fallback = np.concatenate([p[2] for p in parts])
        grey = np.concatenate([pending, fallback])
        unsafe = np.zeros(len(grey), dtype=bool)
        unsafe[len(pending):] = True
        return white, grey, unsafe

    def _gridless(self, indices):
        exact = np.arange(self.total, dtype=np.uint64) if indices is None else np.asarray(indices, dtype=np.uint64)
        parts = [r.gridless() for r in self.records]
        scored = np.concatenate([p[0] for p in parts])
        white = np.concatenate([p[1] for p in parts])
        if not np.all(np.isin(scored, exact, assume_unique=True)):
            raise ValueError("replay: a chunk scored gridless poses that the cascade decided")
        missing = np.setdiff1d(exact, scored, assume_unique=True)
        if len(missing):
            white = np.concatenate([white, self._score_missing(missing)])
        return np.sort(white), _EMPTY, _EMPTY

    def _score_missing(self, missing: np.ndarray) -> np.ndarray:
        """Gridless-score the poses an early whole-action exit added (< min_cascade)."""
        args = self.args
        self.rescored = len(missing)
        print(f"Energy filter: gridless rescoring of {len(missing)} poses after a "
              "whole-action cascade exit", file=sys.stderr, flush=True)
        with tempfile.TemporaryDirectory(prefix="energy-filter-exit-") as temporary:
            work = Path(temporary)
            ids_path = work / "exit-ids.npy"
            np.save(ids_path, missing, allow_pickle=False)
            efa._prepare_minfor_inputs(args.pose_dir, args.protein, args.sequence, self.exclude,
                                       work, pose_indices=ids_path)
            invoke = efa._MinforInvoker(work, len(missing), protein_range=args.protein_range,
                                        epsilon=float(args.epsilon), threshold=self.threshold,
                                        chunk_poses=args.chunk_poses)
            white, grey, _ = invoke(None, None)
            if len(grey):
                raise ValueError("zero-tolerance gridless filtering returned grey poses")
        return missing[white.astype(np.intp)]


def run_assemble(args) -> dict:
    threshold, steps, minimum, exclude = _validated(args)
    total = _total_poses(args.pose_dir)
    settings = _settings(args, threshold, steps, minimum, exclude)
    records = _load_records(args, settings, total)
    replay = _ReplayInvoker(records, total, args, threshold, exclude)
    kept_ids, stats = efa._cascade_minfor(total, steps, minimum, replay)
    print(f"Energy filter: assembled {args.nchunks} chunks; "
          f"{replay.rescored} poses rescored after the whole-action exit", file=sys.stderr)
    output_dir = Path(args.output)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="energy-filter-assemble-") as temporary:
        # minfor_invocations reports the whole-action route, as a single run would, so the
        # published statistics (part of the result checksum) do not depend on chunking.
        return efa._publish(args.pose_dir, Path(args.protein), output_dir, kept_ids, stats,
                            Path(temporary), epsilon=float(args.epsilon), threshold=threshold,
                            fragment=args.fragment, protein_range=args.protein_range,
                            nposes=total, invocations=replay.count,
                            nprocs=args.nprocs, chunk_poses=args.chunk_poses)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    chunk = efa.add_filter_arguments(sub.add_parser("chunk", help="score one chunk"))
    chunk.add_argument("--chunk-index", required=True, type=int)
    assemble = efa.add_filter_arguments(sub.add_parser("assemble", help="replay and publish"))
    assemble.add_argument("--output", required=True, type=Path)
    for p in (chunk, assemble):
        p.add_argument("--nchunks", required=True, type=int)
        p.add_argument("--chunks-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.mode == "chunk":
            run_chunk(args)
        else:
            run_assemble(args)
    except Exception as exc:
        print(f"energy-filter {args.mode} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
