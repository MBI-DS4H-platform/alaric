from __future__ import annotations

import math

import numpy as np

from alaric.bridge import (
    BloomFilter,
    BloomMetadata,
    bloom_fpr,
    hash_bucket_payload,
    hash_pose_keys,
    optimal_hash_count,
    pack_pose_keys,
)


def test_pack_and_hash_pose_keys_are_deterministic() -> None:
    conformers = np.array([1, 2, 1], dtype=np.uint16)
    rotamers = np.array([5, 6, 5], dtype=np.uint16)
    translations = np.array([[10, -2, 3], [0, 1, 2], [10, -2, 3]], dtype=np.int16)

    keys = pack_pose_keys(conformers, rotamers, translations)
    first = hash_pose_keys(keys)
    second = hash_pose_keys(keys.copy())

    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.uint64
    assert first[0] == first[2]
    assert first[0] != first[1]


def test_bucket_payload_hash_uses_bucket_seed_and_local_payload() -> None:
    M = np.array([3, -1, 8], dtype=np.int16)
    conformers = np.array([7, 7], dtype=np.uint16)
    rotamers = np.array([11, 11], dtype=np.uint16)
    offsets = np.array([[1, -2, 0], [1, -2, 0]], dtype=np.int16)

    hashes = hash_bucket_payload(M, conformers, rotamers, offsets)
    shifted_bucket = hash_bucket_payload(M + np.array([1, 0, 0], dtype=np.int16), conformers, rotamers, offsets)

    assert hashes[0] == hashes[1]
    assert hashes[0] != shifted_bucket[0]


def test_bloom_filter_has_no_false_negatives_for_inserted_keys() -> None:
    rng = np.random.default_rng(123)
    conformers = rng.integers(0, 100, size=2000, dtype=np.uint16)
    rotamers = rng.integers(0, 1000, size=2000, dtype=np.uint16)
    translations = rng.integers(-300, 300, size=(2000, 3), dtype=np.int16)
    keys = pack_pose_keys(conformers, rotamers, translations)
    metadata = BloomMetadata.from_budget(memory_bytes=8192, expected_items=len(keys))
    bloom = BloomFilter.from_metadata(metadata, block_size=137)

    bloom.add_keys(keys)

    assert bloom.probe_keys(keys).all()


def test_bloom_formula_and_hash_count_optimization() -> None:
    assert bloom_fpr(0, 1024, 4) == 0.0
    expected = (1.0 - math.exp(-4.0 * 100.0 / 1024.0)) ** 4
    assert bloom_fpr(100, 1024, 4) == expected
    k = optimal_hash_count(100, 1024, max_hashes=16)
    assert 1 <= k <= 16
    assert bloom_fpr(100, 1024, k) == min(bloom_fpr(100, 1024, i) for i in range(1, 17))

