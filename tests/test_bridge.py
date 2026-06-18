from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import alaric.bridge as bridge_module
from alaric.bridge import (
    BloomFilter,
    BloomMetadata,
    BridgeGrowConfig,
    BridgeGrowResult,
    RotamerChunk,
    _MemoryPoseOriginWriter,
    bloom_fpr,
    choose_bridge_orientation,
    compose_bridge_connections,
    deterministic_sample_indices,
    enforce_intermediate_guardrail,
    expected_intermediate_size,
    hash_bucket_payload,
    hash_pose_keys,
    merge_bridge_chunk_outputs,
    optimal_hash_count,
    pack_pose_keys,
    run_bridge_pipeline,
)
from alaric.poses import decode_pool, discover_unorganized, read_arc_file, write_arc_file


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


def test_bloom_filter_confines_each_key_to_one_block() -> None:
    conformers = np.array([1, 2, 3], dtype=np.uint16)
    rotamers = np.array([4, 5, 6], dtype=np.uint16)
    translations = np.array([[0, 0, 0], [10, -2, 1], [-5, 3, 7]], dtype=np.int16)
    hashes = hash_pose_keys(pack_pose_keys(conformers, rotamers, translations))
    bloom = BloomFilter(n_bits=4096, n_hashes=6, block_bits=512)

    owner_blocks = bloom.block_indices(hashes)
    for hash_index in range(bloom.n_hashes):
        positions = bloom._positions(hashes, hash_index)
        np.testing.assert_array_equal(positions // bloom.block_bits, owner_blocks)


def test_bloom_formula_and_hash_count_optimization() -> None:
    assert bloom_fpr(0, 1024, 4) == 0.0
    expected = (1.0 - math.exp(-4.0 * 100.0 / 1024.0)) ** 4
    assert bloom_fpr(100, 1024, 4) == expected
    k = optimal_hash_count(100, 1024, max_hashes=16)
    assert 1 <= k <= 16
    assert bloom_fpr(100, 1024, k) == min(bloom_fpr(100, 1024, i) for i in range(1, 17))


def test_bloom_budget_is_a_ceiling_not_a_required_allocation() -> None:
    metadata = BloomMetadata.from_budget(
        memory_bytes=240 * 1024**3,
        expected_items=1_000_000_000,
    )
    assert metadata.n_bits // 8 < 10 * 1024**3
    assert metadata.n_bits % metadata.block_bits == 0
    assert metadata.expected_fpr <= 1e-9


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


def test_compose_bridge_connections_preserves_final_orientation(tmp_path) -> None:
    identity_dir = tmp_path / "identity"
    identity_dir.mkdir()
    np.save(identity_dir / "map-1.npy", np.array([[1, 2], [0, 1]], dtype=np.uint64))
    np.save(identity_dir / "map-2.npy", np.array([[0, 2], [2, 1]], dtype=np.uint64))
    first_origin = tmp_path / "first.npy"
    second_origin = tmp_path / "second.npy"
    np.save(first_origin, np.array([10, 11], dtype=np.uint64))
    np.save(second_origin, np.array([20, 21, 22], dtype=np.uint64))

    lower, upper = compose_bridge_connections(
        identity_dir,
        first_origin,
        second_origin,
        tmp_path / "out",
        first_side="lower",
    )

    np.testing.assert_array_equal(lower, np.array([[10, 1], [11, 2]], dtype=np.uint64))
    np.testing.assert_array_equal(upper, np.array([[1, 22], [2, 20]], dtype=np.uint64))

    lower_rev, upper_rev = compose_bridge_connections(
        identity_dir,
        first_origin,
        second_origin,
        tmp_path / "out-rev",
        first_side="upper",
    )
    np.testing.assert_array_equal(lower_rev, np.array([[20, 2], [22, 1]], dtype=np.uint64))
    np.testing.assert_array_equal(upper_rev, np.array([[1, 10], [2, 11]], dtype=np.uint64))


def test_merge_bridge_chunk_outputs_remaps_middle_indices(tmp_path) -> None:
    def write_chunk(path, conformers, rotamers, translations, lower, upper):
        path.mkdir()
        from alaric.poses import pack_pool

        packed = pack_pool(
            np.array(conformers, dtype=np.uint16),
            np.array(rotamers, dtype=np.uint16),
            np.array(translations, dtype=np.int16),
            bucket_size=16,
        )
        assert len(packed) == 1
        M, O, C, P = packed[0]
        write_arc_file(path / "poses-1.arc", M, O, C, P, bucket_size=16)
        np.save(path / "connections-lower.npy", np.array(lower, dtype=np.uint64))
        np.save(path / "connections-upper.npy", np.array(upper, dtype=np.uint64))

    chunk1 = tmp_path / "chunk1"
    chunk2 = tmp_path / "chunk2"
    write_chunk(chunk1, [1, 1], [0, 1], [[5, 0, 0], [0, 0, 0]], [[100, 0], [101, 1]], [[0, 200], [1, 201]])
    write_chunk(chunk2, [1], [2], [[2, 0, 0]], [[102, 0]], [[0, 202]])

    lower, upper = merge_bridge_chunk_outputs([chunk1, chunk2], tmp_path / "final", bucket_size=16, nprocs=1)

    assert lower.shape == (3, 2)
    assert upper.shape == (3, 2)
    assert set(lower[:, 0].tolist()) == {100, 101, 102}
    assert set(upper[:, 1].tolist()) == {200, 201, 202}
    assert set(lower[:, 1].tolist()) == {0, 1, 2}
    assert set(upper[:, 0].tolist()) == {0, 1, 2}


def test_run_bridge_pipeline_enforces_global_final_guardrail_and_keeps_work_out_of_output(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        bridge_module,
        "_estimate_chunk_growth",
        lambda **kwargs: ("lower", 100, 100, 0.0, 0.0),
    )
    monkeypatch.setattr(
        bridge_module,
        "bridge_grow",
        lambda *args, **kwargs: BridgeGrowResult(
            generated_poses=100,
            emitted_poses=1,
            output_dir=kwargs.get("output_dir"),
            emitted_origin_path=(
                None
                if kwargs.get("output_dir") is None
                else Path(kwargs["output_dir"]) / "emitted_origin.npy"
            ),
        ),
    )
    monkeypatch.setattr(bridge_module, "_add_pose_dir_to_bloom", lambda *args, **kwargs: 1)

    def fake_organize(pose_dir, **kwargs):
        pose_dir = Path(pose_dir)
        pose_dir.mkdir(parents=True, exist_ok=True)
        origin = np.array([0], dtype=np.uint64)
        np.save(pose_dir / "organized_origin.npy", origin)
        return origin

    def fake_identity(first_pose_dir, second_pose_dir, output_dir, **kwargs):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "identity").mkdir(exist_ok=True)
        np.save(output_dir / "connections-lower.npy", np.empty((0, 2), dtype=np.uint64))
        np.save(output_dir / "connections-upper.npy", np.empty((0, 2), dtype=np.uint64))
        return {"identity_poses": 7}

    monkeypatch.setattr(bridge_module, "organize_bridge_intermediate", fake_organize)
    monkeypatch.setattr(bridge_module, "run_identity_and_compose_bridge", fake_identity)
    monkeypatch.setattr(bridge_module, "discover_organized", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        bridge_module,
        "PoseReader",
        type("FakePoseReader", (), {"get_nposes": staticmethod(lambda path: 7)}),
    )

    cfg = BridgeGrowConfig(
        source_poses=tmp_path / "source",
        source_sequence="AA",
        target_sequence="AU",
        direction="forward",
        crmsd=1.0,
        ov_rmsd=1.0,
    )
    out = tmp_path / "out"

    with pytest.raises(ValueError, match="final poses 14 exceeds guardrail 10"):
        run_bridge_pipeline(
            lower_config=cfg,
            upper_config=cfg,
            output_dir=out,
            memory_bytes=1024,
            max_intermediate_poses=10,
            max_final_poses=10,
            rotamer_chunks=2,
        )

    assert not (out / "bridge-work").exists()
