"""Compression-transparent .npy I/O.

The load-bearing property is byte-equality with ``np.save``: a compressed array's
*uncompressed* bytes are what result checksums are taken over, so if this writer ever
diverged from what NumPy produces (a different header version, different padding), every
pose dir carrying an index array would silently change checksum. Verified against the
installed NumPy so a version bump cannot slip that through.
"""

from __future__ import annotations

import io
from pathlib import Path
import sys

import numpy as np
import pytest
import zstandard as zstd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / ".alaric"))

from npy_io import (  # noqa: E402
    NpyWriter,
    compress_npy_file,
    find_npy,
    load_npy,
    open_npy_mmap,
    read_npy_header,
    save_npy,
)


ARRAYS = {
    "provenance": np.arange(1000, dtype=np.uint32),
    "map": np.arange(40, dtype=np.uint64).reshape(-1, 2),
    "mask": np.arange(64) % 3 == 0,
    "order": np.array([3, 0, 2, 1], dtype=np.uint8),
    "scores": np.linspace(-10, 10, 100, dtype=np.float32),
    "empty": np.empty(0, dtype=np.uint32),
}


def _decompress(path: Path) -> bytes:
    return zstd.ZstdDecompressor().stream_reader(io.BytesIO(path.read_bytes())).read()


@pytest.mark.parametrize("name", sorted(ARRAYS))
def test_compressed_payload_is_byte_identical_to_np_save(tmp_path: Path, name: str) -> None:
    array = ARRAYS[name]
    reference = io.BytesIO()
    np.save(reference, array)

    path = save_npy(tmp_path / f"{name}.npy", array)

    assert path.name.endswith(".npy.zst")
    assert _decompress(path) == reference.getvalue()
    # ... and the .npy the same call writes uncompressed is that same byte stream
    assert save_npy(tmp_path / f"{name}-plain.npy", array, compress=False).read_bytes() == (
        reference.getvalue()
    )


def test_streamed_writes_match_np_save(tmp_path: Path) -> None:
    array = ARRAYS["provenance"]
    reference = io.BytesIO()
    np.save(reference, array)

    with NpyWriter(tmp_path / "p.npy", dtype=array.dtype, shape=array.shape) as writer:
        for start in range(0, len(array), 137):
            writer.write(array[start : start + 137])

    assert _decompress(tmp_path / "p.npy.zst") == reference.getvalue()


def test_streamed_writes_are_length_checked(tmp_path: Path) -> None:
    array = ARRAYS["provenance"]
    with pytest.raises(ValueError, match="never written"):
        with NpyWriter(tmp_path / "short.npy", dtype=array.dtype, shape=(10,)) as writer:
            writer.write(array[:3])
    with pytest.raises(ValueError, match="more elements"):
        with NpyWriter(tmp_path / "long.npy", dtype=array.dtype, shape=(2,)) as writer:
            writer.write(array[:3])
    # nothing half-written is left behind for a reader to pick up
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("name", sorted(ARRAYS))
def test_round_trip(tmp_path: Path, name: str) -> None:
    array = ARRAYS[name]
    path = save_npy(tmp_path / f"{name}.npy", array)

    np.testing.assert_array_equal(load_npy(path), array)
    with open_npy_mmap(path) as mapped:
        np.testing.assert_array_equal(mapped, array)
    assert read_npy_header(path) == (array.shape, array.dtype)


def test_find_npy_resolves_either_form(tmp_path: Path) -> None:
    logical = tmp_path / "provenance.npy"
    assert find_npy(logical) is None

    save_npy(logical, ARRAYS["provenance"], compress=False)
    assert find_npy(logical) == logical

    compressed = compress_npy_file(logical)
    assert compressed.name == "provenance.npy.zst"
    assert not logical.exists()
    # the caller still asks for the logical name
    assert find_npy(logical) == compressed
