"""Generator for the Gaussian stacking filter CSR caches.

A cache is a pure function of the accepted (Hopf cell, lattice translation)
pair set; see ``gaussian_filter`` for the definitions and
``documents/gaussian-filter.md`` for the layout.  The score only depends on
(|R33|, i^2+j^2, k^2), so one table over those keys decides every pair.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

try:  # package import or flat import (tests add alaric/ to sys.path)
    from . import gaussian_filter as gf
    from . import npy_io
except ImportError:  # pragma: no cover
    import gaussian_filter as gf
    import npy_io

BLOCK_KEYS = 2048
BLOCK_CELLS = 1024
AUDIT_MIN = 1e-10
FILE_NAMES = ("rotations", "translations", "filtered-sizes", "filtered-concat")


class DeterminismAuditError(RuntimeError):
    pass


@dataclass
class Build:
    cells: np.ndarray  # int64 [R] hopf cell indices
    rotations: np.ndarray  # float64 [R,3,3]
    translations: np.ndarray  # float64 [M,3]
    sizes: np.ndarray  # int32 [R]
    concat_blocks: Callable[[], Iterator[np.ndarray]]
    audit: dict

    @property
    def pairs(self) -> int:
        return int(self.sizes.sum(dtype=np.int64))


def _keys(ijk, ok, ):
    """Per-point key index (-1 if masked out) and the key (s, k2) arrays."""
    s = ijk[:, 0] ** 2 + ijk[:, 1] ** 2
    k2 = ijk[:, 2] ** 2
    combined = s * 4096 + k2
    uniq, inv = np.unique(combined[ok], return_inverse=True)
    tk = np.full(len(ijk), -1, dtype=np.int64)
    tk[ok] = inv
    mult = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    return tk, uniq // 4096, uniq % 4096, mult


def score_table(model=gf.MODEL, theta=None, keys=None):
    """Score over (theta keys x translation keys).  Defaults: all masked keys."""
    s, k2 = keys
    d = np.sqrt((s + k2) * 0.0625)
    phi = np.arccos(np.sqrt(k2.astype(np.float64)) * 0.25 / d)
    return gf.score(d[None, :], phi[None, :], theta[:, None], model)


def accept_pairs(
    cutoff: float,
    *,
    tilt30: bool = False,
    disp45: bool = False,
    model=gf.MODEL,
    audit_min: float = AUDIT_MIN,
) -> Build:
    R, absr33 = gf.hopf_rotations()
    ijk, t = gf.lattice()
    ok = gf.mask_translation(ijk, disp45=disp45)
    tk, ks, kk2, mult = _keys(ijk, ok)
    d = np.sqrt((ks + kk2) * 0.0625)
    phi = np.arccos(np.sqrt(kk2.astype(np.float64)) * 0.25 / d)

    uk, inv = np.unique(absr33, return_inverse=True)
    row_ok = gf.mask_rotation(uk, tilt30)
    theta = np.arccos(np.minimum(1.0, uk))

    accept = np.zeros((len(uk), len(mult)), dtype=bool)
    size_key = np.zeros(len(uk), dtype=np.int64)
    min_margin = np.inf
    for lo in range(0, len(uk), BLOCK_KEYS):
        hi = min(lo + BLOCK_KEYS, len(uk))
        sc = gf.score(d[None, :], phi[None, :], theta[lo:hi, None], model)
        acc = (sc >= cutoff) & row_ok[lo:hi, None]
        accept[lo:hi] = acc
        size_key[lo:hi] = np.rint(acc.astype(np.float64) @ mult).astype(np.int64)
        rows = row_ok[lo:hi]
        if rows.any():
            min_margin = min(min_margin, float(np.abs(sc[rows] / cutoff - 1.0).min()))
    min_tilt_gap = float(np.abs(4.0 * absr33 * absr33 - 3.0).min())
    audit = {"min_rel_margin": min_margin, "min_tilt_gap": min_tilt_gap}
    if min_margin < audit_min or min_tilt_gap < audit_min:
        raise DeterminismAuditError(f"cache would depend on ULP behaviour: {audit}")

    cell_sizes = size_key[inv]
    cells = np.flatnonzero(cell_sizes > 0)
    keep_pts = np.flatnonzero((tk >= 0) & accept.any(axis=0)[np.maximum(tk, 0)])
    tk_kept = tk[keep_pts]
    cell_rows = inv[cells]

    def blocks() -> Iterator[np.ndarray]:
        for lo in range(0, len(cells), BLOCK_CELLS):
            sub = accept[cell_rows[lo : lo + BLOCK_CELLS]][:, tk_kept]
            yield np.nonzero(sub)[1].astype(np.int32)

    return Build(
        cells=cells,
        rotations=R[cells].copy(),
        translations=t[keep_pts].copy(),
        sizes=cell_sizes[cells].astype(np.int32),
        concat_blocks=blocks,
        audit=audit,
    )


def build_filter(name: str) -> Build:
    _, tilt, disp = gf.parse_name(name)
    return accept_pairs(gf.threshold_spec(name)["cutoff"], tilt30=tilt, disp45=disp)


def concat_all(build: Build) -> np.ndarray:
    parts = list(build.concat_blocks())
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32)


def environment() -> dict:
    env = {"python": platform.python_version(), "numpy": np.__version__}
    try:
        import zstandard

        env["zstandard"] = zstandard.__version__
        env["libzstd"] = ".".join(str(x) for x in zstandard.ZSTD_VERSION)
    except ImportError:  # pragma: no cover
        pass
    env["zstd_level"] = 3
    return env


def write_cache(directory, build: Build, extra: dict | None = None) -> Path:
    """Write the four files (compressed) into ``directory``, published by rename."""
    directory = Path(directory)
    tmp = directory.with_name(directory.name + f".tmp{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        npy_io.save_npy(tmp / "rotations.npy", build.rotations)
        npy_io.save_npy(tmp / "translations.npy", build.translations)
        npy_io.save_npy(tmp / "filtered-sizes.npy", build.sizes)
        with npy_io.NpyWriter(
            tmp / "filtered-concat.npy", dtype=np.int32, shape=(build.pairs,)
        ) as writer:
            for block in build.concat_blocks():
                writer.write(block)
        if directory.exists():
            shutil.rmtree(directory)
        directory.parent.mkdir(parents=True, exist_ok=True)
        tmp.replace(directory)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    return directory


def write_manifest(directory, name, build: Build, cutoff_decimal: str) -> dict:
    try:
        from . import gaussian_cache as gc
    except ImportError:  # pragma: no cover
        import gaussian_cache as gc
    directory = Path(directory)
    files = {}
    for fname in FILE_NAMES:
        desc = gc.describe_file(directory / f"{fname}.npy")
        shape, dtype = npy_io.read_npy_header(
            npy_io.find_npy(directory / f"{fname}.npy")
        )
        files[fname] = {**desc, "shape": list(shape), "dtype": dtype.str}
    manifest = {
        "schema": 1,
        "name": name,
        "cutoff": cutoff_decimal,
        "files": files,
        "counts": {
            "rotations": len(build.sizes),
            "translations": len(build.translations),
            "pairs": build.pairs,
        },
        "audit": build.audit,
        "environment": environment(),
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=True) + "\n"
    )
    return manifest


def generate(name: str, directory) -> dict:
    """Build filter ``name`` into ``directory`` (compressed) with a manifest."""
    build = build_filter(name)
    write_cache(directory, build)
    return write_manifest(directory, name, build, gf.threshold_spec(name)["cutoff_decimal"])
