"""Installation check for the Gaussian filter caches (stdlib + lazy zstandard)."""

from __future__ import annotations

import hashlib
from pathlib import Path

FILE_NAMES = ("rotations", "translations", "filtered-sizes", "filtered-concat")
_CHUNK = 4 * 1024 * 1024


class GaussianCacheInstallationError(RuntimeError):
    pass


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _sha_decompressed(path: Path) -> str:
    import zstandard

    h = hashlib.sha256()
    with path.open("rb") as f, zstandard.ZstdDecompressor().stream_reader(f) as r:
        while chunk := r.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def describe_file(npy_path) -> dict:
    """Digests of the on-disk form of logical ``X.npy``: {sha256, npy_sha256}."""
    npy_path = Path(npy_path)
    if npy_path.is_file():
        digest = _sha_file(npy_path)
        return {"sha256": None, "npy_sha256": digest}
    zst = npy_path.with_name(npy_path.name + ".zst")
    return {"sha256": _sha_file(zst), "npy_sha256": _sha_decompressed(zst)}


def generator_command(name: str, root) -> str:
    root = Path(root)
    return (
        f"python {root}/util/gaussian-precalc.py {name} "
        f"--output {root}/precomputed-filters/gaussian/{name} "
        f"(or {root}/util/gaussian-precalc-all.py)"
    )


def verify_file(npy_path, sha256, npy_sha256, *, command: str = "") -> tuple[Path, str]:
    """Verify logical ``npy_path``; return (resolved path, status).

    Status is "npy", "zst" or "recompressed".  Raises
    GaussianCacheInstallationError otherwise.
    """
    npy_path = Path(npy_path)
    zst = npy_path.with_name(npy_path.name + ".zst")

    def fail(reason: str):
        raise GaussianCacheInstallationError(
            f"{reason}: {npy_path} (expected npy_sha256={npy_sha256}, sha256={sha256}). "
            f"Regenerate with: {command}"
        )

    if npy_path.is_file():
        if _sha_file(npy_path) == npy_sha256:
            return npy_path, "npy"
        fail("uncompressed file has the wrong checksum")
    if zst.is_file():
        if sha256 is not None and _sha_file(zst) == sha256:
            return zst, "zst"
        try:
            ok = _sha_decompressed(zst) == npy_sha256
        except Exception as exc:  # corrupt / truncated stream
            fail(f"cannot decompress ({exc})")
        if ok:
            return zst, "recompressed"
        fail("compressed file has the wrong content")
    fail("file is missing")


def _verify_entry(directory: Path, files: dict, command: str) -> dict:
    return {
        fname: verify_file(
            directory / f"{fname}.npy", e.get("sha256"), e["npy_sha256"], command=command
        )[0]
        for fname, e in files.items()
    }


def verify_filter_cache(spec: dict, name: str, root) -> dict:
    """Verify all four files of filter ``name``; return {file name: resolved path}."""
    root = Path(root)
    pre = spec["thresholds"][name]["precalc"]
    return _verify_entry(root / pre["dir"], pre["files"], generator_command(name, root))


def decompress_dir(directory) -> None:
    """Replace each ``X.npy.zst`` in ``directory`` by its ``X.npy`` (streaming)."""
    import zstandard

    directory = Path(directory)
    for fname in FILE_NAMES:
        zst = directory / f"{fname}.npy.zst"
        out = directory / f"{fname}.npy"
        tmp = directory / f"{fname}.npy.tmp"
        with zst.open("rb") as src, tmp.open("wb") as dst:
            zstandard.ZstdDecompressor().copy_stream(src, dst, write_size=_CHUNK)
        tmp.replace(out)
        zst.unlink()


def compare_manifest(manifest: dict, entry_files: dict) -> tuple[bool, list[str]]:
    """Compare freshly built manifest files with JSON entries.

    Returns (arrays_identical, notes).  A sha256-only difference is a note.
    """
    ok, notes = True, []
    for fname, e in entry_files.items():
        m = manifest["files"][fname]
        if m["npy_sha256"] != e["npy_sha256"]:
            ok = False
            notes.append(f"{fname}: npy_sha256 mismatch (algorithm or environment drift)")
        elif m["sha256"] != e["sha256"]:
            notes.append(f"{fname}: recompressed, arrays identical")
    return ok, notes
