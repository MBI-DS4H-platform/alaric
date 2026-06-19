from __future__ import annotations

import gzip
import hashlib
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from seamless import Buffer
from seamless.compression_utils import strip_compression_suffix
from seamless.checksum.calculate_checksum import calculate_checksum
from seamless.checksum.serialize import serialize_sync as serialize


def sha256_bytes(data: bytes) -> str:
    return calculate_checksum(data)


def json_checksum(value: Any) -> str:
    return str(Buffer(value, celltype="plain").get_checksum())


def transparent_bytes(path: Path) -> bytes:
    return path.read_bytes()


def byte_checksum(path: Path) -> str:
    return streaming_file_checksum(path)


def streaming_file_checksum(path: Path) -> str:
    _base, suffix = strip_compression_suffix(path.name)
    hasher = hashlib.sha256()
    if suffix == ".zst":
        import zstandard as zstd

        with path.open("rb") as handle:
            with zstd.ZstdDecompressor().stream_reader(handle) as reader:
                for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                    hasher.update(chunk)
    elif suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
    else:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
    return hasher.hexdigest()


def npy_payload_checksum(path: Path) -> str:
    return byte_checksum(path)


def write_array_sidecar(path: Path) -> str:
    base_name, _suffix = strip_compression_suffix(path.name)
    checksum = streaming_file_checksum(path)
    path.with_name(base_name + ".CHECKSUM").write_text(checksum + "\n")
    return checksum


def pose_member_checksum(path: Path) -> str:
    # Trust the inline sidecar organize.py writes next to each member (sha256 of the
    # uncompressed .arc content) — this avoids re-reading/re-decompressing the (large) pose
    # file. Fall back to computing it for pose dirs produced without the sidecar.
    base, _suffix = strip_compression_suffix(path.name)
    sidecar = path.with_name(base + ".CHECKSUM")
    if sidecar.is_file():
        return read_sidecar(sidecar)
    return byte_checksum(path)


def pose_index(pose_dir: Path) -> dict[str, str]:
    paths = [
        path
        for path in sorted(pose_dir.rglob("*"), key=lambda p: p.as_posix())
        if path.is_file()
        and path.name.startswith("poses-")
        and (path.name.endswith(".arc") or path.name.endswith(".arc.zst"))
    ]

    def _member(path: Path) -> tuple[str, str]:
        rel = path.relative_to(pose_dir).as_posix()
        logical_name, _suffix = strip_compression_suffix(rel)
        return logical_name, pose_member_checksum(path)

    # The members are independent files; for a large organized pose dir on a network FS this
    # checksum pass dominates organize.sh's post-organize time. zstd decompression and
    # hashlib both release the GIL, so a thread pool gives real parallelism (CPU) and overlaps
    # NFS reads. ``ThreadPoolExecutor.map`` preserves input order, and ``serialize(plain)``
    # sorts keys anyway, so the resulting INDEX is byte-identical to the serial version.
    # Worker count follows os.cpu_count(), which honors the SLURM allocation via the
    # PYTHON_CPU_COUNT exported in the generated scripts.
    workers = min(len(paths), max(1, os.cpu_count() or 1))
    if workers <= 1:
        return dict(_member(path) for path in paths)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return dict(executor.map(_member, paths))


def write_pose_sidecars(pose_dir: Path) -> str:
    checksum_path = pose_dir.with_name(pose_dir.name + ".CHECKSUM")
    subprocess.run(["seamless-checksum-index", str(pose_dir)], check=True)
    checksum = read_sidecar(checksum_path)
    return checksum


def result_checksum(result_dir: Path, kind: str) -> str:
    if kind == "score" or kind == "mask":
        for name in ("score.npy", "score.npy.zst", "mask.npy", "mask.npy.zst"):
            path = result_dir / name
            if path.is_file():
                return write_array_sidecar(path)
        raise FileNotFoundError(f"no array result found in {result_dir}")
    return write_pose_sidecars(result_dir)


def read_sidecar(path: Path) -> str:
    return path.read_text().strip()


def verify_array_sidecar(path: Path) -> bool:
    base_name, _suffix = strip_compression_suffix(path.name)
    sidecar = path.with_name(base_name + ".CHECKSUM")
    return sidecar.is_file() and read_sidecar(sidecar) == npy_payload_checksum(path)


def verify_pose_sidecars(pose_dir: Path) -> bool:
    index_path = pose_dir.with_name(pose_dir.name + ".INDEX")
    checksum_path = pose_dir.with_name(pose_dir.name + ".CHECKSUM")
    if not index_path.is_file() or not checksum_path.is_file():
        return False
    return read_sidecar(checksum_path) == byte_checksum(index_path)
