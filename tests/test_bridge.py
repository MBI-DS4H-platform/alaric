from __future__ import annotations

import math

import numpy as np

from alaric.bridge import (
    BloomFilter,
    BloomMetadata,
    RotamerChunk,
    _MemoryPoseOriginWriter,
    bloom_fpr,
    choose_bridge_orientation,
    deterministic_sample_indices,
    enforce_intermediate_guardrail,
    expected_intermediate_size,
    hash_bucket_payload,
    hash_pose_keys,
    optimal_hash_count,
    pack_pose_keys,
)
from alaric.poses import decode_pool, discover_unorganized, read_arc_file


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


def test_memory_pose_origin_writer_emits_matching_unorganized_origins(tmp_path) -> None:
    writer = _MemoryPoseOriginWriter(tmp_path, bucket_size=16)
    conformers = np.array([2, 1, 2, 1], dtype=np.uint16)
    rotamers = np.array([3, 4, 1, 2], dtype=np.uint16)
    translations = np.array(
        [[17, 0, 0], [0, 0, 0], [18, 0, 0], [1, 0, 0]],
        dtype=np.int16,
    )
    origins = np.array([10, 11, 12, 13], dtype=np.uint64)

    emitted = writer.finish()
    assert len(emitted) == 0

    writer.add_chunk(conformers, rotamers, translations, origins)
    emitted = writer.finish()

    np.testing.assert_array_equal(emitted, np.array([11, 13, 10, 12], dtype=np.uint64))
    paths = discover_unorganized(tmp_path)
    decoded = [decode_pool(*read_arc_file(path)) for path in paths]
    decoded_conformers = np.concatenate([part[0] for part in decoded])
    np.testing.assert_array_equal(decoded_conformers, np.array([1, 1, 2, 2], dtype=np.uint16))


def test_rotamer_chunk_partitions_target_rotamers() -> None:
    chunks = [RotamerChunk(i, 3).bounds(10) for i in range(3)]
    assert chunks == [(0, 3), (3, 6), (6, 10)]


def test_deterministic_sampling_is_stable_and_sorted() -> None:
    first = deterministic_sample_indices(10_000, seed=5, sample_size=100)
    second = deterministic_sample_indices(10_000, seed=5, sample_size=100)
    np.testing.assert_array_equal(first, second)
    assert len(first) == 100
    assert np.all(first[:-1] < first[1:])
    np.testing.assert_array_equal(
        deterministic_sample_indices(3, seed=5, sample_size=100),
        np.array([0, 1, 2], dtype=np.uint64),
    )


def test_orientation_uses_expected_first_intermediate() -> None:
    first, lower_first, upper_first = choose_bridge_orientation(
        lower_generated=10_000,
        upper_generated=100,
        n_bits=8192,
        n_hashes=3,
    )
    assert first == ("lower" if lower_first <= upper_first else "upper")
    assert expected_intermediate_size(0, 10, 1024, 2) == 0.0


def test_intermediate_guardrail_fails_loudly() -> None:
    enforce_intermediate_guardrail(10.0, 10)
    try:
        enforce_intermediate_guardrail(11.0, 10)
    except ValueError as exc:
        assert "exceeds guardrail" in str(exc)
    else:
        raise AssertionError("guardrail should have failed")
