#!/usr/bin/env python3
"""Concatenate per-chunk ``score.npy`` files into one ``score.npy``, memory-safely.

The chunk deployer produces one ``chunk-<N>/score.npy`` per chunk. Naively loading them all
and ``np.concatenate``-ing holds the entire score array (plus a copy) in RAM, which is
dangerous past a few gigaposes (a 20-Gpose float64 score is ~160 GB).

Instead this:
  - verifies every expected chunk score exists, and -- given ``--nposes`` -- that the chunks
    together cover exactly the input pool,
  - reads every chunk via ``mmap_mode="r"`` (no full load),
  - streams them block-by-block into a memmapped output ``.npy`` written to a **local**
    staging dir (``$TMPDIR`` -> node-local scratch on the cluster), so peak RAM is one block,
  - computes the SHA-256 of the final ``.npy`` byte stream during that copy pass and
    writes ``score.npy.CHECKSUM``;
  - copies the finished local file to ``score.npy.partial`` and renames to ``score.npy``.

The output bytes are identical to ``np.save(np.concatenate(chunks))`` for the same data, so
the result checksum is unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
from pathlib import Path

import numpy as np


def _chunk_score_files(chunks_dir: Path, nchunks: int | None) -> list[Path]:
    if nchunks is None:
        dirs = sorted(
            (d for d in chunks_dir.glob("chunk-*") if d.is_dir()),
            key=lambda d: int(d.name.split("-", 1)[1]),
        )
        score_files = [d / "score.npy" for d in dirs]
    else:
        score_files = [chunks_dir / f"chunk-{idx}" / "score.npy" for idx in range(1, nchunks + 1)]
    missing = [path for path in score_files if not path.is_file()]
    if missing:
        sample = ", ".join(str(path) for path in missing[:5])
        extra = "" if len(missing) <= 5 else f" ... ({len(missing)} missing)"
        raise FileNotFoundError(f"missing score chunk(s): {sample}{extra}")
    return score_files


def concatenate(
    chunks_dir: Path,
    output: Path,
    *,
    nchunks: int | None = None,
    nposes: int | None = None,
    block: int = 10_000_000,
) -> str:
    """Concatenate chunks and return the checksum of the resulting ``.npy`` file.

    The output header is hashed first, followed by each source block in chunk order.
    This is exactly the byte sequence written by ``open_memmap`` for the 1-D score
    arrays, so no post-copy read of the (potentially network-resident) result is needed.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    score_files = _chunk_score_files(chunks_dir, nchunks)

    mmaps = [np.load(path, mmap_mode="r") for path in score_files]
    dtype = mmaps[0].dtype if mmaps else np.dtype(float)
    total = 0
    for path, m in zip(score_files, mmaps):
        if m.ndim != 1:
            raise ValueError(f"score chunk is not 1-D: {path}")
        if m.dtype != dtype:
            raise ValueError(f"inconsistent score dtype in {path}: {m.dtype} != {dtype}")
        total += int(m.shape[0])
    if nposes is not None and total != int(nposes):
        # The chunk dir is keyed on the sigil alone, so it outlives a failed attempt and is
        # shared by every re-deploy of this action -- including one at a different --nchunks,
        # whose chunk-<N>/score.npy files cover different pose ranges. A chunk job that never
        # started (cancelled in the queue, node failure before exec) leaves its predecessor's
        # file in place, and it would otherwise be folded in as if it were this run's. The
        # concatenation has to cover the input pool exactly, so that is caught here rather
        # than becoming a silently wrong -- but self-consistently checksummed -- result.
        raise ValueError(
            f"{chunks_dir}: chunk scores cover {total} poses, expected {nposes}; "
            "a chunk score from an earlier deploy at a different --nchunks is the likely "
            "cause -- remove the chunk dir and rerun the chunks"
        )

    # Stage on node-local scratch ($TMPDIR), then move to the final destination.
    stagedir = Path(tempfile.mkdtemp(prefix="alaric-score-concat-"))
    staged = stagedir / "score.npy"
    try:
        out = np.lib.format.open_memmap(staged, mode="w+", dtype=dtype, shape=(total,))
        # ``open_memmap`` has emitted the final NPY header. Include precisely that header
        # before streaming the payload; ``out.offset`` is the first payload byte.
        out.flush()
        hasher = hashlib.sha256()
        with staged.open("rb") as handle:
            hasher.update(handle.read(out.offset))
        pos = 0
        for m in mmaps:
            n = int(m.shape[0])
            for start in range(0, n, block):
                stop = min(start + block, n)
                values = m[start:stop]
                out[pos + start : pos + stop] = values
                # A score chunk is 1-D and its dtype was checked above, so these are the
                # same contiguous bytes just stored in the final NPY payload.
                hasher.update(memoryview(values).cast("B"))
            pos += n
        out.flush()
        del out
        partial = output.with_name(output.name + ".partial")
        shutil.copyfile(staged, partial)
        partial.replace(output)
        checksum = hasher.hexdigest()
        output.with_name(output.name + ".CHECKSUM").write_text(checksum + "\n")
        return checksum
    finally:
        shutil.rmtree(stagedir, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chunks_dir", type=Path, help="Directory containing chunk-<N>/score.npy")
    parser.add_argument("output", type=Path, help="Final concatenated score.npy path")
    parser.add_argument("--nchunks", type=int, help="Expected number of chunk score files.")
    parser.add_argument(
        "--nposes",
        type=int,
        help="Expected total number of poses. The concatenated length must match it "
        "exactly, which is what rules out a stale chunk score from an earlier deploy "
        "at a different --nchunks.",
    )
    parser.add_argument(
        "--block",
        type=int,
        default=10_000_000,
        help="Elements copied per block (bounds peak memory; default 10000000).",
    )
    args = parser.parse_args(argv)
    concatenate(
        args.chunks_dir,
        args.output,
        nchunks=args.nchunks,
        nposes=args.nposes,
        block=args.block,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
