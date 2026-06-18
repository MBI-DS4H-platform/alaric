from __future__ import annotations

from dataclasses import dataclass, asdict
import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Iterable

import numpy as np
from tqdm import tqdm

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import grow
import identity_filter
import organize
from poses import PoseReader, decode_pool, discover_organized, discover_unorganized, pack_pool, read_arc_file, select_pose_indices, write_arc_file


KEY_DTYPE = np.dtype(
    [
        ("conformer", "<u2"),
        ("rotamer", "<u2"),
        ("tx", "<i2"),
        ("ty", "<i2"),
        ("tz", "<i2"),
    ]
)
DEFAULT_BLOOM_BLOCK_BITS = 512


def _report(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _fmt_count(value: int) -> str:
    return f"{int(value):,}"


def _fmt_expected_count(value: float) -> str:
    value = float(value)
    if 0.0 < value < 1.0:
        return "<1"
    return f"{value:,.0f}"


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


def blocked_bloom_fpr(
    n_items: int,
    n_bits: int,
    n_hashes: int,
    block_bits: int,
) -> float:
    if n_items < 0:
        raise ValueError("n_items must be non-negative")
    if n_bits <= 0:
        raise ValueError("n_bits must be positive")
    if n_hashes <= 0:
        raise ValueError("n_hashes must be positive")
    if block_bits <= 0 or block_bits % 64 != 0:
        raise ValueError("block_bits must be a positive multiple of 64")
    if n_bits % block_bits != 0:
        raise ValueError("n_bits must be a multiple of block_bits")
    if n_items == 0:
        return 0.0

    n_blocks = n_bits // block_bits
    mean_items_per_block = float(n_items) / float(n_blocks)

    def local_fpr(items_in_block: float) -> float:
        return (
            1.0
            - math.exp(
                -float(n_hashes) * float(items_in_block) / float(block_bits)
            )
        ) ** n_hashes

    if mean_items_per_block > 700.0:
        return local_fpr(mean_items_per_block)

    # A queried key selects one owner block. Insert counts per owner block are
    # approximately Poisson; averaging local FPR over that distribution captures
    # the occupancy variance that a blocked Bloom filter adds.
    p = math.exp(-mean_items_per_block)
    total_probability = p
    expected = p * local_fpr(0.0)
    stop = int(math.ceil(mean_items_per_block + 12.0 * math.sqrt(mean_items_per_block) + 50.0))
    for count in range(1, max(1, stop) + 1):
        p *= mean_items_per_block / float(count)
        total_probability += p
        expected += p * local_fpr(float(count))
    if total_probability < 1.0:
        expected += max(0.0, 1.0 - total_probability) * local_fpr(mean_items_per_block)
    return min(1.0, max(0.0, expected))


def optimal_hash_count(n_items: int, n_bits: int, *, max_hashes: int = 16) -> int:
    if n_items <= 0:
        return 1
    if n_bits <= 0:
        raise ValueError("n_bits must be positive")
    if max_hashes <= 0:
        raise ValueError("max_hashes must be positive")
    candidates = range(1, int(max_hashes) + 1)
    return min(candidates, key=lambda k: bloom_fpr(n_items, n_bits, k))


def optimal_blocked_hash_count(
    n_items: int,
    n_bits: int,
    block_bits: int,
    *,
    max_hashes: int = 16,
) -> int:
    if n_items <= 0:
        return 1
    if n_bits <= 0:
        raise ValueError("n_bits must be positive")
    if max_hashes <= 0:
        raise ValueError("max_hashes must be positive")
    candidates = range(1, int(max_hashes) + 1)
    return min(
        candidates,
        key=lambda k: blocked_bloom_fpr(n_items, n_bits, k, block_bits),
    )


@dataclass(frozen=True)
class BloomMetadata:
    n_bits: int
    n_hashes: int
    expected_items: int
    expected_fpr: float
    block_bits: int = DEFAULT_BLOOM_BLOCK_BITS

    @classmethod
    def from_budget(
        cls,
        *,
        memory_bytes: int,
        expected_items: int,
        max_hashes: int = 16,
        target_fpr: float = 1e-9,
        block_bits: int = DEFAULT_BLOOM_BLOCK_BITS,
    ) -> "BloomMetadata":
        if memory_bytes <= 0:
            raise ValueError("memory_bytes must be positive")
        block_bits = int(block_bits)
        if block_bits <= 0 or block_bits % 64 != 0:
            raise ValueError("block_bits must be a positive multiple of 64")
        if target_fpr <= 0.0 or target_fpr >= 1.0:
            raise ValueError("target_fpr must be between 0 and 1")
        budget_bits = int(memory_bytes) * 8
        if budget_bits < block_bits:
            raise ValueError("memory budget is smaller than one Bloom block")
        usable_budget_bits = (budget_bits // block_bits) * block_bits
        if expected_items <= 0:
            n_bits = block_bits
        else:
            requested_bits = _bits_for_target_blocked_fpr(
                expected_items,
                target_fpr,
                max_hashes=max_hashes,
                block_bits=block_bits,
                max_bits=usable_budget_bits,
            )
            n_bits = min(requested_bits, usable_budget_bits)
        n_hashes = optimal_blocked_hash_count(
            expected_items,
            n_bits,
            block_bits,
            max_hashes=max_hashes,
        )
        return cls(
            n_bits=n_bits,
            n_hashes=n_hashes,
            expected_items=int(expected_items),
            expected_fpr=blocked_bloom_fpr(
                expected_items,
                n_bits,
                n_hashes,
                block_bits,
            ),
            block_bits=block_bits,
        )

    def to_json_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _bits_for_target_fpr(
    n_items: int,
    target_fpr: float,
    *,
    max_hashes: int,
) -> int:
    if n_items <= 0:
        return 64
    best_bits: int | None = None
    target = float(target_fpr)
    for k in range(1, int(max_hashes) + 1):
        root = target ** (1.0 / k)
        if root >= 1.0:
            continue
        denominator = -math.log1p(-root)
        if denominator <= 0.0:
            continue
        bits = math.ceil(k * float(n_items) / denominator)
        best_bits = bits if best_bits is None else min(best_bits, bits)
    if best_bits is None:
        raise ValueError("could not size Bloom filter for target_fpr")
    return int(best_bits)


def _bits_for_target_blocked_fpr(
    n_items: int,
    target_fpr: float,
    *,
    max_hashes: int,
    block_bits: int,
    max_bits: int,
) -> int:
    if n_items <= 0:
        return block_bits

    def rounded(bits: int) -> int:
        return max(block_bits, ((int(bits) + block_bits - 1) // block_bits) * block_bits)

    def best_fpr(bits: int) -> float:
        return min(
            blocked_bloom_fpr(n_items, bits, k, block_bits)
            for k in range(1, int(max_hashes) + 1)
        )

    low = block_bits
    high = rounded(
        _bits_for_target_fpr(n_items, target_fpr, max_hashes=max_hashes)
    )
    high = min(max_bits, max(block_bits, high))
    while high < max_bits and best_fpr(high) > target_fpr:
        low = high + block_bits
        high = min(max_bits, high * 2)
    if best_fpr(high) > target_fpr:
        return high

    low = block_bits if low < block_bits else low
    while low < high:
        mid_blocks = ((low + high) // 2) // block_bits
        mid = max(block_bits, mid_blocks * block_bits)
        if mid < low:
            mid = low
        if mid >= high:
            break
        if best_fpr(mid) <= target_fpr:
            high = mid
        else:
            low = mid + block_bits
    return rounded(high)


class BloomFilter:
    """Single-process blocked Bloom filter backed by a uint64 bitset."""

    def __init__(
        self,
        n_bits: int,
        n_hashes: int,
        *,
        bits: np.ndarray | None = None,
        block_size: int = 1_000_000,
        block_bits: int = DEFAULT_BLOOM_BLOCK_BITS,
    ) -> None:
        if n_bits <= 0:
            raise ValueError("n_bits must be positive")
        if n_hashes <= 0:
            raise ValueError("n_hashes must be positive")
        block_bits = int(block_bits)
        if block_bits <= 0 or block_bits % 64 != 0:
            raise ValueError("block_bits must be a positive multiple of 64")
        if int(n_bits) % block_bits != 0:
            raise ValueError("n_bits must be a multiple of block_bits")
        self.n_bits = int(n_bits)
        self.n_hashes = int(n_hashes)
        self.block_size = int(block_size)
        self.block_bits = block_bits
        self.n_blocks = self.n_bits // self.block_bits
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
        kwargs.setdefault("block_bits", metadata.block_bits)
        return cls(metadata.n_bits, metadata.n_hashes, **kwargs)

    def block_indices(self, hashes: np.ndarray) -> np.ndarray:
        hashes = _as_u64(hashes).reshape(-1)
        mixed = _splitmix64(hashes ^ np.uint64(0xA0761D6478BD642F))
        return np.remainder(mixed, np.uint64(self.n_blocks)).astype(np.uint64, copy=False)

    def _positions(self, hashes: np.ndarray, hash_index: int) -> np.ndarray:
        with np.errstate(over="ignore"):
            salt = np.uint64(hash_index) * np.uint64(0xD6E8FEB86659FD93)
        block_start = self.block_indices(hashes) * np.uint64(self.block_bits)
        mixed = _splitmix64(_as_u64(hashes) ^ salt)
        local = np.remainder(mixed, np.uint64(self.block_bits))
        return (block_start + local).astype(np.uint64, copy=False)

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


@dataclass(frozen=True)
class RotamerChunk:
    chunk: int
    chunks: int

    def bounds(self, nrotamers: int) -> tuple[int, int]:
        if self.chunks <= 0:
            raise ValueError("rotamer chunk count must be positive")
        if self.chunk < 0 or self.chunk >= self.chunks:
            raise ValueError("rotamer chunk index is out of range")
        first = (int(nrotamers) * self.chunk) // self.chunks
        last = (int(nrotamers) * (self.chunk + 1)) // self.chunks
        return first, last


@dataclass(frozen=True)
class BridgeGrowConfig:
    source_poses: Path
    source_sequence: str
    target_sequence: str
    direction: str
    crmsd: float
    ov_rmsd: float
    pdb_exclude: tuple[str, ...] = ()
    pose_range: tuple[int, int] | None = None
    bucket_size: int = 16
    rotamer_chunk: RotamerChunk | None = None
    test_seed: int = 0
    test_conformers: int | None = None
    test_rotamers: int | None = None


@dataclass(frozen=True)
class BridgeGrowResult:
    generated_poses: int
    emitted_poses: int
    output_dir: Path | None
    emitted_origin_path: Path | None


@dataclass(frozen=True)
class BridgeEstimate:
    lower_generated: int
    upper_generated: int
    lower_sampled: int
    upper_sampled: int
    lower_total_poses: int
    upper_total_poses: int
    expected_lower_first: float
    expected_upper_first: float
    first_side: str


def deterministic_sample_indices(total: int, *, seed: int, sample_size: int = 1000) -> np.ndarray:
    if total < 0:
        raise ValueError("total must be non-negative")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if total <= sample_size:
        return np.arange(total, dtype=np.uint64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=sample_size, replace=False).astype(np.uint64))


def expected_intermediate_size(
    n_probe: int,
    n_insert: int,
    n_bits: int,
    n_hashes: int,
    *,
    block_bits: int | None = None,
) -> float:
    if n_probe < 0 or n_insert < 0:
        raise ValueError("pose counts must be non-negative")
    if n_probe == 0 or n_insert == 0:
        return 0.0
    if block_bits is None:
        fpr = bloom_fpr(n_insert, n_bits, n_hashes)
    else:
        fpr = blocked_bloom_fpr(n_insert, n_bits, n_hashes, block_bits)
    return float(n_probe) * fpr


def choose_bridge_orientation(
    *,
    lower_generated: int,
    upper_generated: int,
    n_bits: int,
    n_hashes: int,
    block_bits: int | None = None,
) -> tuple[str, float, float]:
    lower_first = expected_intermediate_size(
        lower_generated,
        upper_generated,
        n_bits,
        n_hashes,
        block_bits=block_bits,
    )
    upper_first = expected_intermediate_size(
        upper_generated,
        lower_generated,
        n_bits,
        n_hashes,
        block_bits=block_bits,
    )
    first = "lower" if lower_first <= upper_first else "upper"
    return first, lower_first, upper_first


def enforce_intermediate_guardrail(expected: float, maximum: int) -> None:
    if maximum <= 0:
        raise ValueError("maximum intermediate poses must be positive")
    if expected > maximum:
        raise ValueError(
            f"expected intermediate poses {expected:.0f} exceeds guardrail {maximum}"
        )


def estimate_bridge_orientation(
    *,
    lower_config: BridgeGrowConfig,
    upper_config: BridgeGrowConfig,
    lower_total_poses: int,
    upper_total_poses: int,
    bloom_metadata: BloomMetadata,
    estimator_seed: int = 0,
    sample_size: int = 1000,
) -> BridgeEstimate:
    lower_indices = deterministic_sample_indices(
        lower_total_poses,
        seed=estimator_seed,
        sample_size=sample_size,
    )
    upper_indices = deterministic_sample_indices(
        upper_total_poses,
        seed=estimator_seed + 1,
        sample_size=sample_size,
    )

    def contiguous_range(indices: np.ndarray) -> tuple[int, int] | None:
        if len(indices) == 0:
            return None
        return int(indices[0]) + 1, int(indices[-1]) + 1

    lower_range = contiguous_range(lower_indices)
    upper_range = contiguous_range(upper_indices)
    lower_sample_config = BridgeGrowConfig(
        **{**lower_config.__dict__, "pose_range": lower_range}
    )
    upper_sample_config = BridgeGrowConfig(
        **{**upper_config.__dict__, "pose_range": upper_range}
    )
    lower_result = bridge_grow(lower_sample_config)
    upper_result = bridge_grow(upper_sample_config)
    lower_span = 0 if lower_range is None else lower_range[1] - lower_range[0] + 1
    upper_span = 0 if upper_range is None else upper_range[1] - upper_range[0] + 1
    lower_generated = (
        0
        if lower_span == 0
        else int(round(lower_result.generated_poses * lower_total_poses / lower_span))
    )
    upper_generated = (
        0
        if upper_span == 0
        else int(round(upper_result.generated_poses * upper_total_poses / upper_span))
    )
    first, expected_lower_first, expected_upper_first = choose_bridge_orientation(
        lower_generated=lower_generated,
        upper_generated=upper_generated,
        n_bits=bloom_metadata.n_bits,
        n_hashes=bloom_metadata.n_hashes,
        block_bits=bloom_metadata.block_bits,
    )
    return BridgeEstimate(
        lower_generated=lower_generated,
        upper_generated=upper_generated,
        lower_sampled=int(len(lower_indices)),
        upper_sampled=int(len(upper_indices)),
        lower_total_poses=int(lower_total_poses),
        upper_total_poses=int(upper_total_poses),
        expected_lower_first=expected_lower_first,
        expected_upper_first=expected_upper_first,
        first_side=first,
    )


class _MemoryPoseOriginWriter:
    def __init__(self, outdir: Path, *, bucket_size: int) -> None:
        self.outdir = Path(outdir)
        self.bucket_size = int(bucket_size)
        self.conformers: list[np.ndarray] = []
        self.rotamers: list[np.ndarray] = []
        self.translations: list[np.ndarray] = []
        self.origins: list[np.ndarray] = []
        self.total_poses = 0

    def add_chunk(
        self,
        conformers: np.ndarray,
        rotamers: np.ndarray,
        translations: np.ndarray,
        origins: np.ndarray,
    ) -> None:
        if len(conformers) == 0:
            return
        if not (len(conformers) == len(rotamers) == len(translations) == len(origins)):
            raise ValueError("pose and origin chunks must have matching lengths")
        self.conformers.append(np.asarray(conformers, dtype=np.uint16).copy())
        self.rotamers.append(np.asarray(rotamers, dtype=np.uint16).copy())
        self.translations.append(np.asarray(translations, dtype=np.int16).copy())
        self.origins.append(np.asarray(origins, dtype=np.uint64).copy())
        self.total_poses += int(len(conformers))

    def finish(self) -> np.ndarray:
        self.outdir.mkdir(parents=True, exist_ok=True)
        if self.total_poses == 0:
            return np.empty((0,), dtype=np.uint64)
        conformers = np.concatenate(self.conformers)
        rotamers = np.concatenate(self.rotamers)
        translations = np.concatenate(self.translations)
        origins = np.concatenate(self.origins)
        emitted_origins: list[np.ndarray] = []
        for file_index, (M, O, C, P) in enumerate(
            pack_pool(
                conformers,
                rotamers,
                translations,
                bucket_size=self.bucket_size,
                sort_offsets=True,
            ),
            start=1,
        ):
            # Reconstruct the same bucket row selection order used by pack_pool.
            half = self.bucket_size // 2
            Ms = ((translations.astype(np.int32) + half) // self.bucket_size).astype(np.int16)
            mask = np.all(Ms == M, axis=1)
            emitted_origins.append(origins[mask])
            write_arc_file(
                self.outdir / f"unorganized-bridge-{file_index}.arc.zst",
                M,
                O,
                C,
                P,
                bucket_size=self.bucket_size,
                zstd=True,
            )
        return np.concatenate(emitted_origins) if emitted_origins else np.empty((0,), dtype=np.uint64)


@dataclass(frozen=True)
class _SourcePoolWithOrigins:
    pool: grow.SourcePool
    origins: np.ndarray


def _load_source_pool_with_origins(
    source_poses: Path,
    pose_range: tuple[int, int] | None,
) -> _SourcePoolWithOrigins:
    kwargs: dict = {}
    start_origin = 0
    if pose_range is not None:
        kwargs["pose_range"] = pose_range
        start_origin = pose_range[0] - 1
    reader = grow.PoseReader(source_poses, **kwargs)
    conformers: list[np.ndarray] = []
    rotamers: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    origins: list[np.ndarray] = []
    cursor = start_origin
    for chunk in reader.iter_chunks():
        conformers.append(chunk.conformers.copy())
        rotamers.append(chunk.rotamers.copy())
        translations.append(chunk.translations_grid.copy())
        origins.append(np.arange(cursor, cursor + len(chunk), dtype=np.uint64))
        cursor += len(chunk)
    if not conformers:
        raise ValueError("No poses found in source directory")
    conf = np.concatenate(conformers)
    rot = np.concatenate(rotamers)
    trans = np.concatenate(translations)
    origin = np.concatenate(origins)
    order = np.argsort(rot, kind="stable")
    order = order[np.argsort(conf[order], kind="stable")]
    conf = conf[order]
    rot = rot[order]
    trans = trans[order]
    origin = origin[order]
    unique_conformers, conformer_starts, conformer_counts = np.unique(
        conf,
        return_index=True,
        return_counts=True,
    )
    return _SourcePoolWithOrigins(
        pool=grow.SourcePool(
            conformers=conf,
            rotamers=rot,
            translations=trans,
            unique_conformers=unique_conformers.astype(np.int64, copy=False),
            conformer_starts=conformer_starts.astype(np.int64, copy=False),
            conformer_counts=conformer_counts.astype(np.int64, copy=False),
        ),
        origins=origin.astype(np.uint64, copy=False),
    )


def _expand_source_instances_with_indices(
    cache: grow.SourceConformerCache,
    pp_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = cache.instance_counts[pp_rows]
    total = int(counts.sum())
    if total == 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0, 3), dtype=np.int16),
        )
    repeat_idx = np.repeat(np.arange(len(pp_rows), dtype=np.int64), counts)
    repeated_starts = cache.instance_starts[pp_rows][repeat_idx]
    group_offsets = np.cumsum(counts, dtype=np.int64) - counts
    within = np.arange(total, dtype=np.int64) - np.repeat(group_offsets, counts)
    translation_indices = repeated_starts + within
    return repeat_idx, translation_indices, cache.instance_translations[translation_indices]


def _target_rotamer_positions(target_library, target_conformer: int, chunk: RotamerChunk | None) -> np.ndarray:
    nrot = len(target_library.get_rotamers(int(target_conformer)))
    if chunk is None:
        return np.arange(nrot, dtype=np.int64)
    first, last = chunk.bounds(nrot)
    return np.arange(first, last, dtype=np.int64)


def _bridge_target_rotamer_count(
    target_library,
    target_conformer: int,
    fixed_positions: np.ndarray | None,
    chunk: RotamerChunk | None,
) -> int:
    if fixed_positions is None:
        return len(_target_rotamer_positions(target_library, target_conformer, chunk))
    if chunk is None:
        return int(len(fixed_positions))
    first, last = chunk.bounds(len(target_library.get_rotamers(int(target_conformer))))
    return int(np.count_nonzero((fixed_positions >= first) & (fixed_positions < last)))


def _estimate_bridge_trace_work(
    source_caches: dict[int, grow.SourceConformerCache],
    target_library,
    target_to_sources: dict[int, np.ndarray],
    target_conformers: np.ndarray,
    fixed_target_rotamer_positions: np.ndarray | None,
    rotamer_chunk: RotamerChunk | None,
) -> int:
    total = 0
    for target_conformer in target_conformers.tolist():
        sources = target_to_sources.get(int(target_conformer))
        if sources is None:
            continue
        target_rotamer_count = _bridge_target_rotamer_count(
            target_library,
            int(target_conformer),
            fixed_target_rotamer_positions,
            rotamer_chunk,
        )
        if target_rotamer_count == 0:
            continue
        for source_conformer in sources.tolist():
            cache = source_caches.get(int(source_conformer))
            if cache is None:
                continue
            total += len(cache.rotamer_flat) * target_rotamer_count
    return int(total)


def bridge_grow(
    config: BridgeGrowConfig,
    *,
    output_dir: Path | None = None,
    bloom: BloomFilter | None = None,
    build_bloom: BloomFilter | None = None,
    emitted_origin_path: Path | None = None,
    report: bool = True,
    desc: str | None = None,
    bloom_inserted_count: int | None = None,
) -> BridgeGrowResult:
    """Run single-process bridge growth.

    If ``build_bloom`` is provided, all accepted target keys are inserted into it.
    If ``bloom`` and ``output_dir`` are provided, only Bloom hits are materialized
    and ``emitted_origin.npy`` is written in unorganized pose order.
    """
    if config.direction not in {"forward", "backward"}:
        raise ValueError("direction must be forward or backward")
    layout = grow._resolve_growth_layout(
        config.source_sequence,
        config.target_sequence,
        config.direction,
    )
    if layout.crmsd_ab[1] != layout.crmsd_bc[0]:
        raise ValueError(
            f"Source and target sequences do not overlap for {config.direction} growth: "
            f"{config.source_sequence}/{config.target_sequence}"
        )
    if report:
        mode = "build-bloom" if build_bloom is not None else "filter"
        if bloom is None and build_bloom is None:
            mode = "count"
        elif bloom is not None and output_dir is not None:
            mode = "filter+materialize"
        _report(
            f"Bridge grow ({mode}): {config.source_sequence} -> "
            f"{config.target_sequence} {config.direction}"
        )
        _report("  Loading source pose pool...")
    source = _load_source_pool_with_origins(config.source_poses, config.pose_range)
    if report:
        _report(
            f"    source poses={_fmt_count(len(source.pool.conformers))} "
            f"unique_conformers={_fmt_count(len(source.pool.unique_conformers))}"
        )
        _report("  Loading libraries and building cRMSD pivot...")
    factories, _templates = grow.config(verify_checksums=False)
    source_factory = factories[config.source_sequence]
    source_factory.load_rotaconformers()
    source_library = source_factory.create(
        pdb_code=None,
        nucleotide_mask=np.array(layout.source_mask, dtype=bool),
        with_rotaconformers=True,
    )
    source_caches = grow._build_source_caches(source.pool, source_library)
    source_factory.unload_rotaconformers()

    excluded = sorted(config.pdb_exclude) or None
    crmsds = grow.load_crmsds(layout.crmsd_ab, layout.crmsd_bc, pdb_code=excluded)
    target_to_sources = grow._build_target_to_sources(
        source.pool.unique_conformers,
        crmsds,
        config.crmsd,
        source_on_rows=layout.source_on_rows,
    )
    target_conformers = np.array(sorted(target_to_sources), dtype=np.int64)
    target_factory = factories[config.target_sequence]
    target_factory.load_rotaconformers()
    target_library = target_factory.create(
        pdb_code=excluded,
        nucleotide_mask=np.array(layout.target_mask, dtype=bool),
        with_rotaconformers=True,
    )
    if target_library.conformer_mask is not None:
        valid = target_library.conformer_mask.astype(bool)
        target_conformers = target_conformers[valid[target_conformers]]
        target_to_sources = {
            int(target): sources
            for target, sources in target_to_sources.items()
            if target < len(valid) and bool(valid[target])
        }
    target_conformers = grow._select_target_conformers(
        target_conformers,
        config.test_conformers,
        config.test_seed,
    )
    if report:
        _report(f"    target conformers={_fmt_count(len(target_conformers))}")
    fixed_target_rotamer_positions = grow._select_target_rotamer_positions(
        target_library,
        target_conformers,
        config.test_rotamers,
        config.test_seed,
    )
    translation_sets = grow._precompute_translation_sets()
    writer = None if output_dir is None else _MemoryPoseOriginWriter(output_dir, bucket_size=config.bucket_size)
    total_generated = 0
    trace_work_total = _estimate_bridge_trace_work(
        source_caches,
        target_library,
        target_to_sources,
        target_conformers,
        fixed_target_rotamer_positions,
        config.rotamer_chunk,
    )
    if report:
        _report(f"    trace work={_fmt_count(trace_work_total)} rotpair")
        if bloom_inserted_count is not None and bloom is not None:
            _report(
                f"    Bloom probe side inserted={_fmt_count(bloom_inserted_count)} "
                f"fpr={bloom_fpr(bloom_inserted_count, bloom.n_bits, bloom.n_hashes):.6g}"
            )

    try:
        progress = tqdm(
            total=trace_work_total,
            desc=desc or "Bridge grow",
            unit="rotpair",
            unit_scale=True,
            mininterval=2.0,
            disable=not report,
        )

        def update_progress(trace_work: int) -> None:
            if trace_work:
                progress.update(trace_work)
            if not report:
                return
            done = max(1, int(progress.n))
            generated_est = int(round(total_generated * trace_work_total / done))
            emitted_now = 0 if writer is None else writer.total_poses
            postfix = {
                "generated": total_generated,
                "gen_est": generated_est,
            }
            if writer is not None:
                emitted_est = int(round(emitted_now * trace_work_total / done))
                postfix["emitted"] = emitted_now
                postfix["emit_est"] = emitted_est
            if bloom_inserted_count is not None and bloom is not None:
                postfix["fpr_hit_est"] = int(
                    round(
                        expected_intermediate_size(
                            generated_est,
                            bloom_inserted_count,
                            bloom.n_bits,
                            bloom.n_hashes,
                            block_bits=bloom.block_bits,
                        )
                    )
                )
            progress.set_postfix(postfix, refresh=False)

        for target_conformer in target_conformers.tolist():
            if target_conformer not in target_to_sources:
                continue
            coords_t = target_library.coordinates[target_conformer].astype(np.float32, copy=False)
            natoms = coords_t.shape[0]
            total_overlap_sd = natoms * config.ov_rmsd * config.ov_rmsd
            mean_t0 = coords_t.mean(axis=0).astype(np.float32, copy=False)
            centered_t = (coords_t - mean_t0).astype(np.float32, copy=False)
            target_trace = float(np.einsum("ij,ij->", centered_t, centered_t))
            rot_qq_all = grow._rotamers_to_matrices(target_library.get_rotamers(target_conformer))
            if fixed_target_rotamer_positions is None:
                positions = _target_rotamer_positions(target_library, target_conformer, config.rotamer_chunk)
            else:
                positions = fixed_target_rotamer_positions
                if config.rotamer_chunk is not None:
                    first, last = config.rotamer_chunk.bounds(len(rot_qq_all))
                    positions = positions[(positions >= first) & (positions < last)]
            rot_qq = rot_qq_all[positions]
            rotamer_indices = positions
            if len(rot_qq) == 0:
                continue
            rot_qq_flat_t = np.ascontiguousarray(rot_qq.reshape(len(rot_qq), 9).T)
            means_q = np.einsum("j,njk->nk", mean_t0, rot_qq).astype(np.float32, copy=False)

            trace_batches = []
            for source_conformer in target_to_sources[target_conformer].tolist():
                cache = source_caches.get(int(source_conformer))
                if cache is None:
                    continue
                cross_second_moment = cache.centered.T.dot(centered_t).astype(np.float32, copy=False)
                source_trace_vectors = np.einsum(
                    "ij,nik->njk",
                    cross_second_moment,
                    cache.rotamer_matrices,
                    optimize=True,
                ).reshape(len(cache.rotamer_matrices), 9)
                trace_batches.append((cache, np.ascontiguousarray(source_trace_vectors)))
            if not trace_batches:
                continue

            for cache, source_trace_vectors in trace_batches:
                trace_work = len(cache.rotamer_flat) * len(rot_qq)
                trace_scores = source_trace_vectors @ rot_qq_flat_t
                rc_sd = cache.pose_trace + target_trace - 2.0 * trace_scores
                rc_sd = np.maximum(rc_sd, 0.0).astype(np.float32, copy=False)
                pp_rows, qq_cols = np.nonzero(rc_sd < total_overlap_sd)
                if pp_rows.size == 0:
                    update_progress(trace_work)
                    continue
                rc_kept = rc_sd[pp_rows, qq_cols]
                repeat_idx, translation_indices, instance_translations = _expand_source_instances_with_indices(cache, pp_rows)
                if instance_translations.size == 0:
                    update_progress(trace_work)
                    continue
                pp_rows_exp = pp_rows[repeat_idx]
                qq_cols_exp = qq_cols[repeat_idx]
                rc_exp = rc_kept[repeat_idx]
                source_means = cache.mean_rotated[pp_rows_exp] + instance_translations.astype(np.float32) * grow.GRID_SPACING
                continuous_translation = source_means - means_q[qq_cols_exp]
                best_grid = np.rint(continuous_translation / grow.GRID_SPACING).astype(np.int32)
                best_world = best_grid.astype(np.float32) * grow.GRID_SPACING
                delta = best_world - continuous_translation
                grid_discretization_sd = (natoms * np.einsum("ij,ij->i", delta, delta)).astype(np.float32, copy=False)
                remaining2 = total_overlap_sd - grid_discretization_sd
                rot_boundary_rmsd = np.sqrt(np.maximum(remaining2, 0.0) / natoms)
                rot_boundary_tolerance = grow._rmsd_tolerance_to_sd_tolerance(
                    natoms,
                    rot_boundary_rmsd,
                    grow.TRACE_RMSD_BOUNDARY_TOLERANCE,
                )
                rot_boundary = np.abs(rc_exp - remaining2) <= rot_boundary_tolerance
                if np.any(rot_boundary):
                    boundary_rows = np.nonzero(rot_boundary)[0]
                    source_centered = np.einsum(
                        "aj,njk->nak",
                        cache.centered,
                        cache.rotamer_matrices[pp_rows_exp[boundary_rows]],
                        optimize=True,
                    )
                    target_centered = np.einsum(
                        "aj,njk->nak",
                        centered_t,
                        rot_qq[qq_cols_exp[boundary_rows]],
                        optimize=True,
                    )
                    dif = source_centered - target_centered
                    rc_exp[boundary_rows] = np.einsum("nij,nij->n", dif, dif)
                keep_rot = np.nonzero(rc_exp < remaining2)[0]
                if keep_rot.size == 0:
                    continue
                kept_pp_rows = pp_rows_exp[keep_rot]
                kept_qq_cols = qq_cols_exp[keep_rot]
                kept_rc = rc_exp[keep_rot]
                kept_best_grid = best_grid[keep_rot]
                kept_delta = delta[keep_rot]
                kept_instance_translations = instance_translations[keep_rot]
                kept_translation_indices = translation_indices[keep_rot]
                remaining3 = total_overlap_sd - grid_discretization_sd[keep_rot] - kept_rc
                remaining4 = total_overlap_sd - kept_rc
                set_indices = grow._select_translation_set_index(translation_sets, remaining3, natoms)
                set_order = np.argsort(set_indices, kind="stable")
                ordered_set_indices = set_indices[set_order]
                set_starts = np.concatenate(
                    (np.array([0], dtype=np.int64), np.flatnonzero(ordered_set_indices[1:] != ordered_set_indices[:-1]) + 1)
                )
                set_stops = np.concatenate((set_starts[1:], np.array([len(set_order)], dtype=np.int64)))
                pool_start = int(source.pool.conformer_starts[np.searchsorted(source.pool.unique_conformers, cache.conformer)])

                for set_start, set_stop in zip(set_starts, set_stops):
                    set_index = int(ordered_set_indices[set_start])
                    translation_offsets = translation_sets.offsets[set_index]
                    if len(translation_offsets) == 0:
                        continue
                    local_rows = set_order[set_start:set_stop]
                    shifted_delta = kept_delta[local_rows, None, :] + translation_offsets.astype(np.float32, copy=False)[None, :, :] * grow.GRID_SPACING
                    translation_sd = natoms * np.einsum("rgj,rgj->rg", shifted_delta, shifted_delta)
                    keep = translation_sd < remaining4[local_rows, None]
                    combined_sd = translation_sd + kept_rc[local_rows, None]
                    boundary = (
                        np.abs(np.sqrt(combined_sd / natoms) - config.ov_rmsd)
                        <= max(grow.RMSD_BOUNDARY_TOLERANCE, grow.TRACE_RMSD_BOUNDARY_TOLERANCE)
                    )
                    if np.any(boundary):
                        boundary_rows, boundary_offsets = np.nonzero(boundary)
                        exact_rows = local_rows[boundary_rows]
                        source_pose = np.einsum(
                            "aj,njk->nak",
                            cache.coords,
                            cache.rotamer_matrices[kept_pp_rows[exact_rows]],
                            optimize=True,
                        )
                        target_pose = np.einsum(
                            "aj,njk->nak",
                            coords_t,
                            rot_qq[kept_qq_cols[exact_rows]],
                            optimize=True,
                        )
                        source_world = kept_instance_translations[exact_rows].astype(np.float32) * grow.GRID_SPACING
                        target_world = (
                            kept_best_grid[exact_rows]
                            + translation_offsets[boundary_offsets].astype(np.int32)
                        ).astype(np.float32) * grow.GRID_SPACING
                        dif = target_pose + target_world[:, None, :] - source_pose - source_world[:, None, :]
                        boundary_rmsd = np.sqrt(np.einsum("nij,nij->n", dif, dif) / natoms)
                        keep[boundary] = False
                        keep[boundary_rows, boundary_offsets] = boundary_rmsd < config.ov_rmsd
                    kept_rows, kept_offsets = np.nonzero(keep)
                    if kept_rows.size == 0:
                        continue
                    out_translations = kept_best_grid[local_rows[kept_rows]] + translation_offsets[kept_offsets].astype(np.int32)
                    if out_translations.size and (
                        out_translations.min() < np.iinfo(np.int16).min
                        or out_translations.max() > np.iinfo(np.int16).max
                    ):
                        raise ValueError("translation exceeds int16 range")
                    out_conformers = np.full(len(out_translations), target_conformer, dtype=np.uint16)
                    out_rotamers = rotamer_indices[kept_qq_cols[local_rows[kept_rows]]].astype(np.uint16, copy=False)
                    origins = source.origins[pool_start + kept_translation_indices[local_rows[kept_rows]]]
                    emit_order = grow._stable_pose_order(out_rotamers, out_translations)
                    out_conformers = out_conformers[emit_order]
                    out_rotamers = out_rotamers[emit_order]
                    out_translations = out_translations[emit_order].astype(np.int16, copy=False)
                    origins = origins[emit_order]
                    total_generated += len(out_conformers)
                    keys = pack_pose_keys(out_conformers, out_rotamers, out_translations)
                    if build_bloom is not None:
                        build_bloom.add_keys(keys)
                    if bloom is not None:
                        hits = bloom.probe_keys(keys)
                    else:
                        hits = np.ones(len(keys), dtype=bool)
                    if writer is not None and np.any(hits):
                        writer.add_chunk(
                            out_conformers[hits],
                            out_rotamers[hits],
                            out_translations[hits],
                            origins[hits],
                        )
                update_progress(trace_work)
        emitted = 0
        origin_path = emitted_origin_path
        if writer is not None:
            if report:
                _report("  Writing filtered pose shards and provenance...")
            emitted_origin = writer.finish()
            emitted = int(len(emitted_origin))
            if origin_path is None:
                origin_path = writer.outdir / "emitted_origin.npy"
            np.save(origin_path, emitted_origin)
        if report:
            _report(
                f"  Bridge grow done: generated={_fmt_count(total_generated)} "
                f"emitted={_fmt_count(emitted)}"
            )
        return BridgeGrowResult(
            generated_poses=int(total_generated),
            emitted_poses=emitted,
            output_dir=output_dir,
            emitted_origin_path=origin_path,
        )
    finally:
        target_factory.unload_rotaconformers()


def write_bridge_grow_manifest(path: Path, result: BridgeGrowResult, *, metadata: dict) -> None:
    payload = {
        "generated_poses": result.generated_poses,
        "emitted_poses": result.emitted_poses,
        "output_dir": None if result.output_dir is None else str(result.output_dir),
        "emitted_origin": None if result.emitted_origin_path is None else str(result.emitted_origin_path),
        **metadata,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def organize_bridge_intermediate(
    pose_dir: Path,
    *,
    emitted_origin_path: Path | None = None,
    organized_origin_path: Path | None = None,
    order_array_path: Path | None = None,
    nprocs: int = 1,
    max_poses_per_file: int = 100_000_000,
    compress: bool = True,
) -> np.ndarray:
    pose_dir = Path(pose_dir)
    emitted_origin_path = emitted_origin_path or (pose_dir / "emitted_origin.npy")
    organized_origin_path = organized_origin_path or (pose_dir / "organized_origin.npy")
    order_array_path = order_array_path or (pose_dir / "order-array.npy")
    emitted_origin = np.load(emitted_origin_path)
    _report(
        f"Organizing bridge intermediate {pose_dir}: "
        f"emitted_origins={_fmt_count(len(emitted_origin))}"
    )
    organize.organize_pose_dir(
        pose_dir,
        nprocs=nprocs,
        max_poses_per_file=max_poses_per_file,
        compress=compress,
        return_order_array=True,
        order_array_path=order_array_path,
    )
    order_array = np.load(order_array_path)
    if len(order_array) != len(emitted_origin):
        raise ValueError(
            f"order array length {len(order_array)} does not match emitted origins {len(emitted_origin)}"
        )
    organized_origin = emitted_origin[order_array.astype(np.intp, copy=False)]
    np.save(organized_origin_path, organized_origin)
    _report(
        f"Organized bridge intermediate {pose_dir}: "
        f"organized_origins={_fmt_count(len(organized_origin))}"
    )
    return organized_origin


def compose_bridge_connections(
    identity_dir: Path,
    first_origin_path: Path,
    second_origin_path: Path,
    output_dir: Path,
    *,
    first_side: str,
) -> tuple[np.ndarray, np.ndarray]:
    if first_side not in {"lower", "upper"}:
        raise ValueError("first_side must be lower or upper")
    identity_dir = Path(identity_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    map1 = np.load(identity_dir / "map-1.npy")
    map2 = np.load(identity_dir / "map-2.npy")
    origin1 = np.load(first_origin_path)
    origin2 = np.load(second_origin_path)
    if map1.ndim != 2 or map1.shape[1] != 2:
        raise ValueError("map-1.npy must have shape (N, 2)")
    if map2.ndim != 2 or map2.shape[1] != 2:
        raise ValueError("map-2.npy must have shape (N, 2)")

    if len(map1) and int(map1[:, 0].max()) >= len(origin1):
        raise ValueError("map-1.npy references an origin outside first_origin_path")
    if len(map2) and int(map2[:, 0].max()) >= len(origin2):
        raise ValueError("map-2.npy references an origin outside second_origin_path")

    if first_side == "lower":
        lower = np.column_stack((origin1[map1[:, 0].astype(np.intp)], map1[:, 1]))
        upper = np.column_stack((map2[:, 1], origin2[map2[:, 0].astype(np.intp)]))
    else:
        lower = np.column_stack((origin2[map2[:, 0].astype(np.intp)], map2[:, 1]))
        upper = np.column_stack((map1[:, 1], origin1[map1[:, 0].astype(np.intp)]))
    lower = np.asarray(lower, dtype=np.uint64)
    upper = np.asarray(upper, dtype=np.uint64)
    if len(lower):
        lower = lower[np.lexsort((lower[:, 1], lower[:, 0]))]
    if len(upper):
        upper = upper[np.lexsort((upper[:, 1], upper[:, 0]))]
    np.save(output_dir / "connections-lower.npy", lower)
    np.save(output_dir / "connections-upper.npy", upper)
    _report(
        f"Composed bridge connections: lower={_fmt_count(len(lower))} "
        f"upper={_fmt_count(len(upper))}"
    )
    return lower, upper


def run_identity_and_compose_bridge(
    first_pose_dir: Path,
    second_pose_dir: Path,
    output_dir: Path,
    *,
    first_origin_path: Path,
    second_origin_path: Path,
    first_side: str,
    max_poses_per_file: int = 100_000_000,
    compress: bool = True,
) -> dict[str, int]:
    identity_dir = Path(output_dir) / "identity"
    _report("Running exact identity filter for bridge intermediates...")
    manifest = identity_filter.run_identity_filter(
        Path(first_pose_dir),
        Path(second_pose_dir),
        identity_dir,
        force=True,
        max_poses_per_file=max_poses_per_file,
        compress=compress,
    )
    compose_bridge_connections(
        identity_dir,
        first_origin_path,
        second_origin_path,
        output_dir,
        first_side=first_side,
    )
    return manifest


def merge_bridge_chunk_outputs(
    chunk_dirs: list[Path],
    output_dir: Path,
    *,
    bucket_size: int = 16,
    nprocs: int = 1,
    max_poses_per_file: int = 100_000_000,
    compress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _report(f"Merging {_fmt_count(len(chunk_dirs))} bridge rotamer chunk(s)...")
    writer = _MemoryPoseOriginWriter(output_dir, bucket_size=bucket_size)
    lower_parts: list[np.ndarray] = []
    upper_parts: list[np.ndarray] = []
    base = 0
    spans: list[tuple[int, int]] = []
    for chunk_dir in tqdm(chunk_dirs, desc="Merge bridge chunks", unit="chunk"):
        chunk_dir = Path(chunk_dir)
        reader = PoseReader(chunk_dir)
        chunk_start = base
        for pose_chunk in reader.iter_chunks():
            n = len(pose_chunk)
            origins = np.arange(base, base + n, dtype=np.uint64)
            writer.add_chunk(
                pose_chunk.conformers,
                pose_chunk.rotamers,
                pose_chunk.translations_grid,
                origins,
            )
            base += n
        spans.append((chunk_start, base))
        lower = np.load(chunk_dir / "connections-lower.npy")
        upper = np.load(chunk_dir / "connections-upper.npy")
        lower = np.asarray(lower, dtype=np.uint64).copy()
        upper = np.asarray(upper, dtype=np.uint64).copy()
        lower[:, 1] += np.uint64(chunk_start)
        upper[:, 0] += np.uint64(chunk_start)
        lower_parts.append(lower)
        upper_parts.append(upper)

    old_middle_ids = writer.finish()
    np.save(output_dir / "chunk-middle-origin.npy", old_middle_ids)
    _report(f"Organizing merged bridge middle pool: poses={_fmt_count(len(old_middle_ids))}")
    organize.organize_pose_dir(
        output_dir,
        nprocs=nprocs,
        max_poses_per_file=max_poses_per_file,
        compress=compress,
        return_order_array=True,
        order_array_path=output_dir / "order-array.npy",
    )
    order = np.load(output_dir / "order-array.npy")
    if len(order) != len(old_middle_ids):
        raise ValueError("merged order array length does not match merged pose count")
    organized_old_ids = old_middle_ids[order.astype(np.intp, copy=False)]
    old_to_new = np.empty(len(organized_old_ids), dtype=np.uint64)
    old_to_new[organized_old_ids.astype(np.intp, copy=False)] = np.arange(
        len(organized_old_ids),
        dtype=np.uint64,
    )

    lower_all = (
        np.concatenate(lower_parts, axis=0)
        if lower_parts
        else np.empty((0, 2), dtype=np.uint64)
    )
    upper_all = (
        np.concatenate(upper_parts, axis=0)
        if upper_parts
        else np.empty((0, 2), dtype=np.uint64)
    )
    if len(lower_all):
        lower_all[:, 1] = old_to_new[lower_all[:, 1].astype(np.intp, copy=False)]
        lower_all = lower_all[np.lexsort((lower_all[:, 1], lower_all[:, 0]))]
    if len(upper_all):
        upper_all[:, 0] = old_to_new[upper_all[:, 0].astype(np.intp, copy=False)]
        upper_all = upper_all[np.lexsort((upper_all[:, 1], upper_all[:, 0]))]
    np.save(output_dir / "connections-lower.npy", lower_all)
    np.save(output_dir / "connections-upper.npy", upper_all)
    manifest = {
        "chunks": [str(Path(p)) for p in chunk_dirs],
        "chunk_spans": spans,
        "middle_poses": int(len(old_middle_ids)),
        "lower_connections": int(len(lower_all)),
        "upper_connections": int(len(upper_all)),
    }
    (output_dir / "bridge-merge.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    _report(
        f"Merged bridge chunks: middle={_fmt_count(len(old_middle_ids))} "
        f"lower_connections={_fmt_count(len(lower_all))} "
        f"upper_connections={_fmt_count(len(upper_all))}"
    )
    return lower_all, upper_all


def _parse_memory_bytes(text: str) -> int:
    value = str(text).strip().upper()
    multipliers = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    if value[-1:] in multipliers:
        return int(float(value[:-1]) * multipliers[value[-1]])
    return int(value)


def _add_pose_dir_to_bloom(pose_dir: Path, bloom: BloomFilter, *, desc: str = "Build Bloom") -> int:
    paths = discover_unorganized(pose_dir) or discover_organized(pose_dir)
    total = 0
    for path in tqdm(paths, desc=desc, unit="file", mininterval=2.0):
        M, O, C, P, bucket_size = read_arc_file(path)
        conf, rot, translations = decode_pool(M, O, C, P, bucket_size)
        bloom.add_keys(pack_pose_keys(conf, rot, translations))
        total += len(conf)
    _report(f"{desc}: inserted={_fmt_count(total)}")
    return total


def _write_sample_pose_dir(
    source_dir: Path,
    output_dir: Path,
    indices: np.ndarray,
    *,
    bucket_size: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(indices) == 0:
        raise ValueError("cannot build an empty bridge estimate sample")
    chunk = select_pose_indices(source_dir, indices)
    packed = pack_pool(
        chunk.conformers,
        chunk.rotamers,
        chunk.translations_grid,
        bucket_size=bucket_size,
    )
    for file_index, (M, O, C, P) in enumerate(packed, start=1):
        write_arc_file(
            output_dir / f"poses-{file_index}.arc",
            M,
            O,
            C,
            P,
            bucket_size=bucket_size,
        )


def _estimate_chunk_growth(
    *,
    lower_config: BridgeGrowConfig,
    upper_config: BridgeGrowConfig,
    chunk_dir: Path,
    chunk: int,
    rotamer_chunks: int,
    metadata: BloomMetadata,
    estimator_seed: int,
    sample_size: int = 1000,
) -> tuple[str, int, int, float, float]:
    estimate_dir = chunk_dir / "estimate"
    lower_total = PoseReader.get_nposes(lower_config.source_poses)
    upper_total = PoseReader.get_nposes(upper_config.source_poses)
    lower_indices = deterministic_sample_indices(
        lower_total,
        seed=estimator_seed + 2 * chunk,
        sample_size=sample_size,
    )
    upper_indices = deterministic_sample_indices(
        upper_total,
        seed=estimator_seed + 2 * chunk + 1,
        sample_size=sample_size,
    )
    lower_sample_dir = estimate_dir / "lower-sample"
    upper_sample_dir = estimate_dir / "upper-sample"
    _report(
        f"[chunk {chunk + 1}/{rotamer_chunks}] Estimating growth from "
        f"{len(lower_indices):,}/{lower_total:,} lower and "
        f"{len(upper_indices):,}/{upper_total:,} upper source poses..."
    )
    _write_sample_pose_dir(
        lower_config.source_poses,
        lower_sample_dir,
        lower_indices,
        bucket_size=lower_config.bucket_size,
    )
    _write_sample_pose_dir(
        upper_config.source_poses,
        upper_sample_dir,
        upper_indices,
        bucket_size=upper_config.bucket_size,
    )
    lower_sample_config = BridgeGrowConfig(
        **{
            **lower_config.__dict__,
            "source_poses": lower_sample_dir,
            "rotamer_chunk": RotamerChunk(chunk, rotamer_chunks),
        }
    )
    upper_sample_config = BridgeGrowConfig(
        **{
            **upper_config.__dict__,
            "source_poses": upper_sample_dir,
            "rotamer_chunk": RotamerChunk(chunk, rotamer_chunks),
        }
    )
    lower_sample = bridge_grow(
        lower_sample_config,
        report=True,
        desc=f"Estimate lower {chunk + 1}/{rotamer_chunks}",
    )
    upper_sample = bridge_grow(
        upper_sample_config,
        report=True,
        desc=f"Estimate upper {chunk + 1}/{rotamer_chunks}",
    )
    lower_est = int(round(lower_sample.generated_poses * lower_total / len(lower_indices)))
    upper_est = int(round(upper_sample.generated_poses * upper_total / len(upper_indices)))
    lower_first_expected = expected_intermediate_size(
        lower_est,
        upper_est,
        metadata.n_bits,
        metadata.n_hashes,
        block_bits=metadata.block_bits,
    )
    upper_first_expected = expected_intermediate_size(
        upper_est,
        lower_est,
        metadata.n_bits,
        metadata.n_hashes,
        block_bits=metadata.block_bits,
    )
    first_side = "lower" if lower_first_expected <= upper_first_expected else "upper"
    _report(
        f"  estimate lower generated={lower_sample.generated_poses:,} "
        f"from {len(lower_indices):,} sampled source poses; "
        f"full_generated_est={lower_est:,}"
    )
    _report(
        f"  estimate upper generated={upper_sample.generated_poses:,} "
        f"from {len(upper_indices):,} sampled source poses; "
        f"full_generated_est={upper_est:,}"
    )
    _report(
        f"  expected initial Bloom false-positive hits: "
        f"lower-first={_fmt_expected_count(lower_first_expected)} "
        f"upper-first={_fmt_expected_count(upper_first_expected)}; "
        f"choosing {first_side}-first"
    )
    return first_side, lower_est, upper_est, lower_first_expected, upper_first_expected


def run_bridge_pipeline(
    *,
    lower_config: BridgeGrowConfig,
    upper_config: BridgeGrowConfig,
    output_dir: Path,
    memory_bytes: int,
    max_intermediate_poses: int,
    max_final_poses: int,
    nprocs: int = 1,
    rotamer_chunks: int = 1,
    estimator_seed: int = 0,
) -> dict[str, int | str]:
    if rotamer_chunks <= 0:
        raise ValueError("rotamer_chunks must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="alaric-bridge-work-"))
    work_dir.mkdir(parents=True, exist_ok=True)
    _report("Starting bridge pipeline")
    _report(
        f"  output={output_dir} chunks={rotamer_chunks} memory={_fmt_count(memory_bytes)} bytes "
        f"max_intermediate={_fmt_count(max_intermediate_poses)} "
        f"max_final={_fmt_count(max_final_poses)}"
    )
    chunk_dirs: list[Path] = []
    total_final = 0
    for chunk in range(rotamer_chunks):
        chunk_dir = work_dir / f"chunk-{chunk:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        _report(f"[chunk {chunk + 1}/{rotamer_chunks}] Preparing Bloom metadata...")
        lower_chunk = BridgeGrowConfig(**{**lower_config.__dict__, "rotamer_chunk": RotamerChunk(chunk, rotamer_chunks)})
        upper_chunk = BridgeGrowConfig(**{**upper_config.__dict__, "rotamer_chunk": RotamerChunk(chunk, rotamer_chunks)})
        metadata = BloomMetadata.from_budget(memory_bytes=memory_bytes, expected_items=max(1, max_intermediate_poses))
        _report(
            f"  Bloom bits={_fmt_count(metadata.n_bits)} hashes={metadata.n_hashes} "
            f"expected_fpr={metadata.expected_fpr:.6g} "
            f"allocated={_fmt_count((metadata.n_bits + 7) // 8)} bytes "
            f"budget={_fmt_count(memory_bytes)} bytes"
        )
        first_side, lower_est, upper_est, expected_lower_first, expected_upper_first = _estimate_chunk_growth(
            lower_config=lower_config,
            upper_config=upper_config,
            chunk_dir=chunk_dir,
            chunk=chunk,
            rotamer_chunks=rotamer_chunks,
            metadata=metadata,
            estimator_seed=estimator_seed,
        )
        if first_side == "lower":
            bloom_side = "upper"
            bloom_config = upper_chunk
            first_config = lower_chunk
            second_config = upper_chunk
            expected_first_from_sample = expected_lower_first
            bloom_est = upper_est
        else:
            bloom_side = "lower"
            bloom_config = lower_chunk
            first_config = upper_chunk
            second_config = lower_chunk
            expected_first_from_sample = expected_upper_first
            bloom_est = lower_est

        _report(
            f"[chunk {chunk + 1}/{rotamer_chunks}] Growing {bloom_side} side into full Bloom "
            f"(sample full_generated_est={_fmt_count(bloom_est)})..."
        )
        full_bloom = BloomFilter.from_metadata(metadata)
        full_bloom_result = bridge_grow(
            bloom_config,
            build_bloom=full_bloom,
            desc=f"{bloom_side.capitalize()} full Bloom {chunk + 1}/{rotamer_chunks}",
        )
        _report(f"  {bloom_side} full generated={_fmt_count(full_bloom_result.generated_poses)}")

        _report(
            f"[chunk {chunk + 1}/{rotamer_chunks}] Growing {first_side} side through "
            f"{bloom_side} Bloom "
            f"(sample expected_fp={_fmt_expected_count(expected_first_from_sample)})..."
        )
        first_dir = chunk_dir / "intermediate-first"
        first = bridge_grow(
            first_config,
            output_dir=first_dir,
            bloom=full_bloom,
            desc=f"First intermediate {chunk + 1}/{rotamer_chunks}",
            bloom_inserted_count=full_bloom_result.generated_poses,
        )
        expected_first = expected_intermediate_size(
            first.generated_poses,
            full_bloom_result.generated_poses,
            metadata.n_bits,
            metadata.n_hashes,
            block_bits=metadata.block_bits,
        )
        enforce_intermediate_guardrail(first.emitted_poses, max_intermediate_poses)
        _report(
            f"  first intermediate generated={_fmt_count(first.generated_poses)} "
            f"expected_fp={_fmt_expected_count(expected_first)} "
            f"emitted={_fmt_count(first.emitted_poses)}"
        )

        _report(f"[chunk {chunk + 1}/{rotamer_chunks}] Building Bloom from first intermediate...")
        first_bloom = BloomFilter.from_metadata(metadata)
        _add_pose_dir_to_bloom(first_dir, first_bloom, desc=f"First intermediate Bloom {chunk + 1}/{rotamer_chunks}")

        second_side = "upper" if first_side == "lower" else "lower"
        _report(f"[chunk {chunk + 1}/{rotamer_chunks}] Growing {second_side} side through first-intermediate Bloom...")
        expected_second = expected_intermediate_size(
            full_bloom_result.generated_poses,
            first.emitted_poses,
            metadata.n_bits,
            metadata.n_hashes,
            block_bits=metadata.block_bits,
        )
        _report(
            f"  expected second intermediate Bloom false-positive hits="
            f"{_fmt_expected_count(expected_second)}"
        )
        second_dir = chunk_dir / "intermediate-second"
        second = bridge_grow(
            second_config,
            output_dir=second_dir,
            bloom=first_bloom,
            desc=f"Second intermediate {chunk + 1}/{rotamer_chunks}",
            bloom_inserted_count=first.emitted_poses,
        )
        enforce_intermediate_guardrail(second.emitted_poses, max_intermediate_poses)
        _report(
            f"  second intermediate generated={_fmt_count(second.generated_poses)} "
            f"expected_fp={_fmt_expected_count(expected_second)} "
            f"emitted={_fmt_count(second.emitted_poses)}"
        )

        _report(f"[chunk {chunk + 1}/{rotamer_chunks}] Organizing intermediates...")
        first_origin = organize_bridge_intermediate(first_dir, nprocs=nprocs)
        second_origin = organize_bridge_intermediate(second_dir, nprocs=nprocs)
        _report(
            f"  organized origins: first={_fmt_count(len(first_origin))} "
            f"second={_fmt_count(len(second_origin))}"
        )
        _report(f"[chunk {chunk + 1}/{rotamer_chunks}] Running exact identity and composing maps...")
        run_identity_and_compose_bridge(
            first_dir,
            second_dir,
            chunk_dir,
            first_origin_path=first_dir / "organized_origin.npy",
            second_origin_path=second_dir / "organized_origin.npy",
            first_side=first_side,
            max_poses_per_file=max_final_poses,
        )
        final_count = PoseReader.get_nposes(chunk_dir / "identity")
        if final_count > max_final_poses:
            raise ValueError(f"final poses {final_count} exceeds guardrail {max_final_poses}")
        total_final += final_count
        if total_final > max_final_poses:
            raise ValueError(
                f"final poses {total_final} exceeds guardrail {max_final_poses}"
            )
        _report(f"  chunk final middle poses={_fmt_count(final_count)}")
        # Put chunk-local final pose files at chunk root for merge.
        for pose_file in discover_organized(chunk_dir / "identity"):
            target = chunk_dir / pose_file.name
            if not target.exists():
                target.write_bytes(pose_file.read_bytes())
        chunk_dirs.append(chunk_dir)

    if rotamer_chunks == 1:
        _report("Promoting single bridge chunk to final output...")
        identity_dir = chunk_dirs[0] / "identity"
        for path in identity_dir.iterdir():
            if path.name.startswith("poses-") or path.name in {"connections-lower.npy", "connections-upper.npy"}:
                continue
        for pose_file in discover_organized(identity_dir):
            (output_dir / pose_file.name).write_bytes(pose_file.read_bytes())
        for name in ("connections-lower.npy", "connections-upper.npy"):
            (output_dir / name).write_bytes((chunk_dirs[0] / name).read_bytes())
    else:
        _report("Merging bridge chunk outputs into final output...")
        merge_bridge_chunk_outputs(
            chunk_dirs,
            output_dir,
            nprocs=nprocs,
            max_poses_per_file=max_final_poses,
        )
    manifest = {
        "chunks": rotamer_chunks,
        "final_poses": int(total_final),
        "memory_bytes": int(memory_bytes),
        "max_intermediate_poses": int(max_intermediate_poses),
        "max_final_poses": int(max_final_poses),
    }
    (output_dir / "bridge.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    shutil.rmtree(work_dir, ignore_errors=True)
    _report(f"Bridge pipeline done: final_poses={_fmt_count(total_final)}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bridge")
    parser.add_argument("--lower-poses", required=True, type=Path)
    parser.add_argument("--upper-poses", required=True, type=Path)
    parser.add_argument("--lower-sequence", required=True)
    parser.add_argument("--middle-sequence", required=True)
    parser.add_argument("--upper-sequence", required=True)
    parser.add_argument("--lower-crmsd", required=True, type=float)
    parser.add_argument("--lower-ov-rmsd", required=True, type=float)
    parser.add_argument("--upper-crmsd", required=True, type=float)
    parser.add_argument("--upper-ov-rmsd", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--memory", default="600G")
    parser.add_argument("--max-intermediate-poses", type=int, default=100_000_000)
    parser.add_argument("--max-final-poses", type=int, default=1_000_000)
    parser.add_argument("--nprocs", type=int, default=1)
    parser.add_argument("--rotamer-chunks", type=int, default=1)
    parser.add_argument("--estimator-seed", type=int, default=0)
    parser.add_argument("--pdb-exclude", nargs="*", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    lower = BridgeGrowConfig(
        source_poses=args.lower_poses,
        source_sequence=args.lower_sequence,
        target_sequence=args.middle_sequence,
        direction="forward",
        crmsd=args.lower_crmsd,
        ov_rmsd=args.lower_ov_rmsd,
        pdb_exclude=tuple(args.pdb_exclude),
    )
    upper = BridgeGrowConfig(
        source_poses=args.upper_poses,
        source_sequence=args.upper_sequence,
        target_sequence=args.middle_sequence,
        direction="backward",
        crmsd=args.upper_crmsd,
        ov_rmsd=args.upper_ov_rmsd,
        pdb_exclude=tuple(args.pdb_exclude),
    )
    run_bridge_pipeline(
        lower_config=lower,
        upper_config=upper,
        output_dir=args.output,
        memory_bytes=_parse_memory_bytes(args.memory),
        max_intermediate_poses=args.max_intermediate_poses,
        max_final_poses=args.max_final_poses,
        nprocs=args.nprocs,
        rotamer_chunks=args.rotamer_chunks,
        estimator_seed=args.estimator_seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
