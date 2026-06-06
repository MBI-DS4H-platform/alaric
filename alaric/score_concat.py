#!/usr/bin/env python3
"""Concatenate per-chunk ``score.npy`` files into one ``score.npy``, memory-safely.

The chunk deployer produces one ``chunks/chunk-<N>/score.npy`` per chunk. Naively loading
them all and ``np.concatenate``-ing holds the entire score array (plus a copy) in RAM, which
is dangerous past a few gigaposes (a 20-Gpose float64 score is ~160 GB).

Instead this:
  - reads every chunk via ``mmap_mode="r"`` (no full load),
  - streams them block-by-block into a memmapped output ``.npy`` written to a **local**
    staging dir (``$TMPDIR`` -> node-local scratch on the cluster), so peak RAM is one block,
  - moves the finished file to its final destination at the end (avoids memmap-over-NFS).

The output bytes are identical to ``np.save(np.concatenate(chunks))`` for the same data, so
the result checksum is unchanged.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np


def _chunk_score_files(chunks_dir: Path) -> list[Path]:
    dirs = sorted(
        (d for d in chunks_dir.glob("chunk-*") if d.is_dir()),
        key=lambda d: int(d.name.split("-", 1)[1]),
    )
    return [d / "score.npy" for d in dirs]


def concatenate(chunks_dir: Path, output: Path, *, block: int = 10_000_000) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    score_files = _chunk_score_files(chunks_dir)

    mmaps = [np.load(path, mmap_mode="r") for path in score_files]
    if not mmaps:
        np.save(output, np.empty((0,), dtype=float))
        return

    dtype = mmaps[0].dtype
    total = 0
    for path, m in zip(score_files, mmaps):
        if m.ndim != 1:
            raise ValueError(f"score chunk is not 1-D: {path}")
        if m.dtype != dtype:
            raise ValueError(f"inconsistent score dtype in {path}: {m.dtype} != {dtype}")
        total += int(m.shape[0])

    # Stage on node-local scratch ($TMPDIR), then move to the final destination.
    stagedir = Path(tempfile.mkdtemp(prefix="alaric-score-concat-"))
    staged = stagedir / "score.npy"
    try:
        out = np.lib.format.open_memmap(staged, mode="w+", dtype=dtype, shape=(total,))
        pos = 0
        for m in mmaps:
            n = int(m.shape[0])
            for start in range(0, n, block):
                stop = min(start + block, n)
                out[pos + start : pos + stop] = m[start:stop]
            pos += n
        out.flush()
        del out
        shutil.move(str(staged), str(output))
    finally:
        shutil.rmtree(stagedir, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chunks_dir", type=Path, help="Directory containing chunk-<N>/score.npy")
    parser.add_argument("output", type=Path, help="Final concatenated score.npy path")
    parser.add_argument(
        "--block",
        type=int,
        default=10_000_000,
        help="Elements copied per block (bounds peak memory; default 10000000).",
    )
    args = parser.parse_args(argv)
    concatenate(args.chunks_dir, args.output, block=args.block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
