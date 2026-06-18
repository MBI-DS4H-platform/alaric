from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Iterable

import numpy as np


KEY_DTYPE = np.dtype(
    [
        ("conformer", "<u2"),
        ("rotamer", "<u2"),
        ("tx", "<i2"),
        ("ty", "<i2"),
        ("tz", "<i2"),
    ]
)


def _as_u64(values) -> np.ndarray:
    return np.asarray(values, dtype=np.uint64)


def _splitmix64(values) -> np.ndarray:
    """Vectorized SplitMix64 finalizer."""
    x = _as_u64(values)
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def pack_pose_keys(
    conformers: np.ndarray,
    rotamers: np.ndarray,
    translations: np.ndarray,
) -> np.ndarray:
    """Pack exact bridge identity keys into a structured array."""
    conformers = np.asarray(conformers)
    rotamers = np.asarray(rotamers)
    translations = np.asarray(translations)
    if conformers.ndim != 1 or rotamers.ndim != 1:
        raise ValueError("conformers and rotamers must be 1D arrays")
    if conformers.shape != rotamers.shape:
        raise ValueError("conformers and rotamers must have the same shape")
    if translations.shape != (len(conformers), 3):
        raise ValueError("translations must have shape (N, 3)")
    if translations.size and (
        int(translations.min()) < -32768 or int(translations.max()) > 32767
    ):
        raise ValueError("translations must fit in int16")

    keys = np.empty(len(conformers), dtype=KEY_DTYPE)
    keys["conformer"] = conformers.astype(np.uint16, copy=False)
    keys["rotamer"] = rotamers.astype(np.uint16, copy=False)
    grid = translations.astype(np.int16, copy=False)
    keys["tx"] = grid[:, 0]
    keys["ty"] = grid[:, 1]
    keys["tz"] = grid[:, 2]
    return keys


def _pack_bucket_seed(M: np.ndarray) -> np.uint64:
    M = np.asarray(M)
    if M.shape != (3,):
        raise ValueError("M must have shape (3,)")
    words = M.astype(np.int16, copy=False).view(np.uint16).astype(np.uint64)
    return np.uint64(words[0] | (words[1] << np.uint64(16)) | (words[2] << np.uint64(32)))


def _pack_local_payload(
    conformers: np.ndarray,
    rotamers: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    conformers = np.asarray(conformers)
    rotamers = np.asarray(rotamers)
    offsets = np.asarray(offsets)
    if conformers.ndim != 1 or rotamers.ndim != 1 or conformers.shape != rotamers.shape:
        raise ValueError("conformers and rotamers must be matching 1D arrays")
    if offsets.shape != (len(conformers), 3):
        raise ValueError("offsets must have shape (N, 3)")
    if offsets.size and (int(offsets.min()) < -128 or int(offsets.max()) > 127):
        raise ValueError("bucket-local offsets must fit in signed 8-bit payload fields")

    off = offsets.astype(np.int8, copy=False).view(np.uint8).astype(np.uint64)
    conf = conformers.astype(np.uint16, copy=False).astype(np.uint64)
    rot = rotamers.astype(np.uint16, copy=False).astype(np.uint64)
    return (
        conf
        | (rot << np.uint64(16))
        | (off[:, 0] << np.uint64(32))
        | (off[:, 1] << np.uint64(40))
        | (off[:, 2] << np.uint64(48))
    )


def hash_bucket_payload(
    M: np.ndarray,
    conformers: np.ndarray,
    rotamers: np.ndarray,
    offsets: np.ndarray,
    *,
    salt: int = 0,
) -> np.ndarray:
    """Hash exact middle keys using bucket seed plus local payload."""
    seed = _pack_bucket_seed(M) ^ np.uint64(int(salt) & 0xFFFFFFFFFFFFFFFF)
    payload = _pack_local_payload(conformers, rotamers, offsets)
    return _splitmix64(payload ^ _splitmix64(seed))


def hash_pose_keys(keys: np.ndarray, *, salt: int = 0) -> np.ndarray:
    """Hash structured exact pose keys independent of .arc bucket size."""
    arr = np.asarray(keys, dtype=KEY_DTYPE)
    conf = arr["conformer"].astype(np.uint64)
    rot = arr["rotamer"].astype(np.uint64)
    tx = arr["tx"].view(np.uint16).astype(np.uint64)
    ty = arr["ty"].view(np.uint16).astype(np.uint64)
    tz = arr["tz"].view(np.uint16).astype(np.uint64)
    lo = conf | (rot << np.uint64(16)) | (tx << np.uint64(32)) | (ty << np.uint64(48))
    hi = tz ^ np.uint64(int(salt) & 0xFFFFFFFFFFFFFFFF)
    return _splitmix64(lo ^ _splitmix64(hi))


def bloom_fpr(n_items: int, n_bits: int, n_hashes: int) -> float:
    if n_items < 0:
        raise ValueError("n_items must be non-negative")
    if n_bits <= 0:
        raise ValueError("n_bits must be positive")
    if n_hashes <= 0:
        raise ValueError("n_hashes must be positive")
    if n_items == 0:
        return 0.0
    return (1.0 - math.exp(-float(n_hashes) * float(n_items) / float(n_bits))) ** n_hashes


def optimal_hash_count(n_items: int, n_bits: int, *, max_hashes: int = 16) -> int:
    if n_items <= 0:
        return 1
    if n_bits <= 0:
        raise ValueError("n_bits must be positive")
    if max_hashes <= 0:
        raise ValueError("max_hashes must be positive")
    candidates = range(1, int(max_hashes) + 1)
    return min(candidates, key=lambda k: bloom_fpr(n_items, n_bits, k))


@dataclass(frozen=True)
class BloomMetadata:
    n_bits: int
    n_hashes: int
    expected_items: int
    expected_fpr: float

    @classmethod
    def from_budget(
        cls,
        *,
        memory_bytes: int,
        expected_items: int,
        max_hashes: int = 16,
    ) -> "BloomMetadata":
        if memory_bytes <= 0:
            raise ValueError("memory_bytes must be positive")
        n_bits = int(memory_bytes) * 8
        n_hashes = optimal_hash_count(expected_items, n_bits, max_hashes=max_hashes)
        return cls(
            n_bits=n_bits,
            n_hashes=n_hashes,
            expected_items=int(expected_items),
            expected_fpr=bloom_fpr(expected_items, n_bits, n_hashes),
        )

    def to_json_dict(self) -> dict[str, int | float]:
        return asdict(self)


class BloomFilter:
    """Single-process exact-key Bloom filter backed by a uint64 bitset."""

    def __init__(
        self,
        n_bits: int,
        n_hashes: int,
        *,
        bits: np.ndarray | None = None,
        block_size: int = 1_000_000,
    ) -> None:
        if n_bits <= 0:
            raise ValueError("n_bits must be positive")
        if n_hashes <= 0:
            raise ValueError("n_hashes must be positive")
        self.n_bits = int(n_bits)
        self.n_hashes = int(n_hashes)
        self.block_size = int(block_size)
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        n_words = (self.n_bits + 63) // 64
        if bits is None:
            self.bits = np.zeros(n_words, dtype=np.uint64)
        else:
            arr = np.asarray(bits, dtype=np.uint64)
            if arr.shape != (n_words,):
                raise ValueError(f"bits must have shape ({n_words},)")
            self.bits = arr

    @classmethod
    def from_metadata(cls, metadata: BloomMetadata, **kwargs) -> "BloomFilter":
        return cls(metadata.n_bits, metadata.n_hashes, **kwargs)

    def _positions(self, hashes: np.ndarray, hash_index: int) -> np.ndarray:
        with np.errstate(over="ignore"):
            salt = np.uint64(hash_index) * np.uint64(0xD6E8FEB86659FD93)
        mixed = _splitmix64(_as_u64(hashes) ^ salt)
        return np.remainder(mixed, np.uint64(self.n_bits)).astype(np.uint64, copy=False)

    def _iter_hash_blocks(self, hashes: np.ndarray) -> Iterable[np.ndarray]:
        hashes = _as_u64(hashes).reshape(-1)
        for start in range(0, len(hashes), self.block_size):
            yield hashes[start : start + self.block_size]

    def add_hashes(self, hashes: np.ndarray) -> None:
        for block in self._iter_hash_blocks(hashes):
            for i in range(self.n_hashes):
                pos = self._positions(block, i)
                word = (pos >> np.uint64(6)).astype(np.intp, copy=False)
                mask = np.left_shift(np.uint64(1), (pos & np.uint64(63)).astype(np.uint64, copy=False))
                np.bitwise_or.at(self.bits, word, mask)

    def probe_hashes(self, hashes: np.ndarray) -> np.ndarray:
        hashes = _as_u64(hashes).reshape(-1)
        result = np.ones(len(hashes), dtype=bool)
        cursor = 0
        for block in self._iter_hash_blocks(hashes):
            keep = np.ones(len(block), dtype=bool)
            for i in range(self.n_hashes):
                pos = self._positions(block, i)
                word = (pos >> np.uint64(6)).astype(np.intp, copy=False)
                mask = np.left_shift(np.uint64(1), (pos & np.uint64(63)).astype(np.uint64, copy=False))
                keep &= (self.bits[word] & mask) != 0
            result[cursor : cursor + len(block)] = keep
            cursor += len(block)
        return result

    def add_keys(self, keys: np.ndarray) -> None:
        self.add_hashes(hash_pose_keys(keys))

    def probe_keys(self, keys: np.ndarray) -> np.ndarray:
        return self.probe_hashes(hash_pose_keys(keys))
