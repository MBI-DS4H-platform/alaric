#!/usr/bin/env python3
"""Energy filtering with an independent minimum-count cutoff in each chunk.

Each chunk applies the cascade to its own grid-eligible poses. When fewer than
min_cascade remain, it scores them together with precision fallbacks gridless.
Completed records are reused on retry; unpublished partial records are recomputed.
Assembly combines the accepted global pose IDs and sums the actual route counts.
Routing and statistics can depend on chunking.
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

RECORD_SCHEMA = 2


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
    record = {
        "schema": RECORD_SCHEMA, "nchunks": args.nchunks, "chunk_index": args.chunk_index,
        "total": total, "first": first, "last": last,
        "settings": _settings(args, threshold, steps, minimum, exclude),
    }
    # Only the final directory is published as complete. A killed job may leave even a
    # fully written record in .partial; that directory must never trigger this skip.
    try:
        completed = json.loads((final / "record.json").read_text())
    except (OSError, ValueError):
        completed = None
    if isinstance(completed, dict) and all(completed.get(key) == value
                                           for key, value in record.items()):
        print(f"Energy filter: chunk {args.chunk_index} already complete; skipping", file=sys.stderr)
        return completed

    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="energy-filter-chunk-") as temporary:
        work = Path(temporary)
        if count:
            efa._prepare_minfor_inputs(args.pose_dir, args.protein, args.sequence, exclude, work,
                                       pose_range=(first, last))
            prepared = np.load(work / "poses.rotvec.npy", mmap_mode="r", allow_pickle=False)
            if len(prepared) != count:
                raise ValueError(f"converted {len(prepared)} poses for range {first}-{last}")
            del prepared
        invoke = (efa._MinforInvoker(work, count, protein_range=args.protein_range,
                                     epsilon=float(args.epsilon), threshold=threshold,
                                     chunk_poses=args.chunk_poses) if count else None)
        kept, stats = efa._cascade_minfor(count, steps, minimum, invoke)
        _save(partial, "kept.npy", kept + offset)
        record.update(stats=stats, minfor_invocations=invoke.count if count else 0)
    (partial / "record.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    shutil.rmtree(final, ignore_errors=True)
    partial.rename(final)
    return record


class _ChunkRecord:
    def __init__(self, directory: Path, record: dict):
        self.directory, self.record = directory, record
        self.first, self.last = record["first"], record["last"]

    def load(self, name: str) -> np.ndarray:
        values = np.load(self.directory / name, allow_pickle=False)
        if values.ndim != 1 or values.dtype != np.dtype("uint64"):
            raise ValueError(f"{self.directory / name}: invalid chunk record array")
        if len(values) and (values.min() < self.first - 1 or values.max() >= self.last):
            raise ValueError(f"{self.directory / name}: ids outside chunk range")
        return values


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


def run_assemble(args) -> dict:
    threshold, steps, minimum, exclude = _validated(args)
    total = _total_poses(args.pose_dir)
    settings = _settings(args, threshold, steps, minimum, exclude)
    records = _load_records(args, settings, total)
    kept_ids = np.concatenate([r.load("kept.npy") for r in records])
    chunk_stats = [r.record["stats"] for r in records]
    stats = {}
    for key in ("grid_counts", "precision_fallback_counts"):
        length = max((len(s[key]) for s in chunk_stats), default=0)
        stats[key] = [sum(s[key][i] for s in chunk_stats if i < len(s[key]))
                      for i in range(length)]
    for key in ("gridless_count", "precision_fallback_count"):
        stats[key] = sum(s[key] for s in chunk_stats)
    exits = [{"chunk_index": r.record["chunk_index"], **r.record["stats"]["cascade_exit"]}
             for r in records if r.record["stats"]["cascade_exit"] is not None]
    stats["cascade_exit"] = {"chunks": exits} if exits else None
    invocations = sum(r.record["minfor_invocations"] for r in records)
    print(f"Energy filter: assembled {args.nchunks} chunks", file=sys.stderr)
    output_dir = Path(args.output)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="energy-filter-assemble-") as temporary:
        return efa._publish(args.pose_dir, Path(args.protein), output_dir, kept_ids, stats,
                            Path(temporary), epsilon=float(args.epsilon), threshold=threshold,
                            fragment=args.fragment, protein_range=args.protein_range,
                            nposes=total, invocations=invocations,
                            nprocs=args.nprocs, chunk_poses=args.chunk_poses)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    chunk = efa.add_filter_arguments(sub.add_parser("chunk", help="score one chunk"))
    chunk.add_argument("--chunk-index", required=True, type=int)
    assemble = efa.add_filter_arguments(sub.add_parser("assemble", help="combine chunk results and publish"))
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
