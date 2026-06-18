from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import grow
from poses import pack_pool, write_arc_file


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


def expected_intermediate_size(n_probe: int, n_insert: int, n_bits: int, n_hashes: int) -> float:
    if n_probe < 0 or n_insert < 0:
        raise ValueError("pose counts must be non-negative")
    if n_probe == 0 or n_insert == 0:
        return 0.0
    return float(n_probe) * bloom_fpr(n_insert, n_bits, n_hashes)


def choose_bridge_orientation(
    *,
    lower_generated: int,
    upper_generated: int,
    n_bits: int,
    n_hashes: int,
) -> tuple[str, float, float]:
    lower_first = expected_intermediate_size(
        lower_generated,
        upper_generated,
        n_bits,
        n_hashes,
    )
    upper_first = expected_intermediate_size(
        upper_generated,
        lower_generated,
        n_bits,
        n_hashes,
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


def bridge_grow(
    config: BridgeGrowConfig,
    *,
    output_dir: Path | None = None,
    bloom: BloomFilter | None = None,
    build_bloom: BloomFilter | None = None,
    emitted_origin_path: Path | None = None,
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
    source = _load_source_pool_with_origins(config.source_poses, config.pose_range)
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
    fixed_target_rotamer_positions = grow._select_target_rotamer_positions(
        target_library,
        target_conformers,
        config.test_rotamers,
        config.test_seed,
    )
    translation_sets = grow._precompute_translation_sets()
    writer = None if output_dir is None else _MemoryPoseOriginWriter(output_dir, bucket_size=config.bucket_size)
    total_generated = 0

    try:
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
            row_start = 0
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
                row_stop = row_start + len(source_trace_vectors)
                trace_batches.append((cache, row_start, row_stop, np.ascontiguousarray(source_trace_vectors)))
                row_start = row_stop
            if not trace_batches:
                continue
            trace_scores = np.concatenate([batch[3] for batch in trace_batches], axis=0) @ rot_qq_flat_t

            for cache, row_start, row_stop, _source_trace_vectors in trace_batches:
                rc_sd = cache.pose_trace + target_trace - 2.0 * trace_scores[row_start:row_stop]
                rc_sd = np.maximum(rc_sd, 0.0).astype(np.float32, copy=False)
                pp_rows, qq_cols = np.nonzero(rc_sd < total_overlap_sd)
                if pp_rows.size == 0:
                    continue
                rc_kept = rc_sd[pp_rows, qq_cols]
                repeat_idx, translation_indices, instance_translations = _expand_source_instances_with_indices(cache, pp_rows)
                if instance_translations.size == 0:
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
        emitted = 0
        origin_path = emitted_origin_path
        if writer is not None:
            emitted_origin = writer.finish()
            emitted = int(len(emitted_origin))
            if origin_path is None:
                origin_path = writer.outdir / "emitted_origin.npy"
            np.save(origin_path, emitted_origin)
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
