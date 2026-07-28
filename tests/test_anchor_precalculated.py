from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
ALARIC = ROOT / "alaric"
if str(ALARIC) not in sys.path:
    sys.path.insert(0, str(ALARIC))

import anchor  # noqa: E402


def _ring() -> np.ndarray:
    return np.array(
        [
            [1.2, 0.1, 0.0],
            [0.4, 1.0, 0.0],
            [-0.8, 0.7, 0.0],
            [-1.0, -0.5, 0.0],
            [0.6, -0.9, 0.0],
        ],
        dtype=np.float64,
    )


def test_superimpose_rotation_handles_row_vector_rigid_transforms() -> None:
    source = _ring()
    rotation = Rotation.from_euler("xyz", [31.0, -18.0, 67.0], degrees=True).as_matrix()
    target = source @ rotation + np.array([4.0, -3.0, 9.0])

    actual = anchor._superimpose_rotation(source, target)

    np.testing.assert_allclose(actual, rotation, atol=1e-12)
    np.testing.assert_allclose(
        (source - source.mean(axis=0)) @ actual,
        target - target.mean(axis=0),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "bad_ring",
    [
        np.zeros((4, 3)),
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    ],
)
def test_superimpose_rotation_rejects_degenerate_rings(bad_ring: np.ndarray) -> None:
    with pytest.raises(ValueError, match="degenerate"):
        anchor._superimpose_rotation(bad_ring, bad_ring.copy())


def test_validate_precalculated_arrays_and_segment_offsets() -> None:
    rotations = np.repeat(np.eye(3)[None, :, :], 2, axis=0)
    translations = np.array([[0.0, 0.0, 3.5], [0.25, 0.0, 3.5]])
    sizes = np.array([2, 1], dtype=np.int64)
    concat = np.array([0, 1, 1], dtype=np.int64)

    _, _, actual_sizes, offsets, actual_concat = (
        anchor._validate_precalculated_arrays(
            rotations, translations, sizes, concat
        )
    )

    np.testing.assert_array_equal(actual_sizes, sizes)
    np.testing.assert_array_equal(offsets, [0, 2, 3])
    np.testing.assert_array_equal(actual_concat, concat)


@pytest.mark.parametrize(
    ("sizes", "concat", "message"),
    [
        (np.array([2]), np.array([0]), "does not equal"),
        (np.array([1]), np.array([1]), "out-of-range"),
        (np.array([-1]), np.array([], dtype=int), "negative"),
    ],
)
def test_validate_precalculated_arrays_rejects_bad_indices(
    sizes: np.ndarray,
    concat: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        anchor._validate_precalculated_arrays(
            np.eye(3)[None, :, :],
            np.array([[0.0, 0.0, 3.5]]),
            sizes,
            concat,
        )


@pytest.mark.parametrize(
    ("rotations", "translations", "message"),
    [
        (np.empty((0, 3, 3)), np.zeros((1, 3)), "rotations must not be empty"),
        (np.eye(3)[None, :, :], np.empty((0, 3)), "translations must not be empty"),
    ],
)
def test_validate_precalculated_arrays_rejects_empty_samples(
    rotations: np.ndarray,
    translations: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        anchor._validate_precalculated_arrays(
            rotations,
            translations,
            np.zeros(len(rotations), dtype=int),
            np.empty(0, dtype=int),
        )


def test_deduplicate_owner_grid_is_per_real_rotamer() -> None:
    owners = np.array([0, 0, 0, 1, 1], dtype=np.int32)
    grid = np.array(
        [[1, 2, 3], [1, 2, 3], [2, 2, 3], [1, 2, 3], [1, 2, 3]]
    )

    actual_owners, actual_grid, actual_counts = anchor._deduplicate_owner_grid(
        owners, grid
    )

    np.testing.assert_array_equal(actual_owners, [0, 0, 1])
    np.testing.assert_array_equal(
        actual_grid, [[1, 2, 3], [2, 2, 3], [1, 2, 3]]
    )
    np.testing.assert_array_equal(actual_counts, [2, 1, 2])


def test_capacity_table_matches_brute_force_counts() -> None:
    rotation = Rotation.from_euler("xyz", [23.0, -11.0, 47.0], degrees=True).as_matrix()
    step = 0.25
    table = anchor._build_capacity_table(step, rotation)

    rng = np.random.default_rng(0)
    centres = rng.uniform(-2.0, 2.0, size=(200, 3))
    half = anchor.GRID_SPACING / (2 * step)
    axis = np.arange(-8, 9)
    lattice = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    expected = np.array(
        [
            np.count_nonzero(np.all(np.abs((lattice - c) @ rotation) <= half, axis=1))
            for c in centres
        ]
    )

    resolution = table.shape[0]
    index = np.minimum((np.mod(centres, 1.0) * resolution).astype(np.int64), resolution - 1)
    actual = table[index[:, 0], index[:, 1], index[:, 2]]

    assert np.mean(actual == expected) > 0.85
    assert np.abs(actual - expected).max() <= 2
    assert abs(actual.mean() - expected.mean()) < 0.3


def test_lattice_geometry_infers_step_and_rejects_irregular_samples() -> None:
    values = -5.0 + 0.25 * np.arange(41)
    samples = np.stack(np.meshgrid(values[:4], values[:4], values[:4]), axis=-1).reshape(-1, 3)

    origin, step = anchor._lattice_geometry(samples)

    assert step == pytest.approx(0.25)
    np.testing.assert_allclose(origin, [-5.0, -5.0, -5.0])
    with pytest.raises(ValueError, match="regular"):
        anchor._lattice_geometry(np.vstack([samples, [[0.1, 0.0, 0.0]]]))


def _cell_lookup(cells: np.ndarray) -> np.ndarray:
    lookup = np.full(anchor.HOPF_CELLS, -1, dtype=np.int32)
    lookup[np.asarray(cells)] = np.arange(len(cells), dtype=np.int32)
    return lookup


def test_hopf_cell_index_inverts_the_generator_grid() -> None:
    rng = np.random.default_rng(0)
    cells = rng.choice(anchor.HOPF_CELLS, 2000, replace=False)

    matrices = anchor._hopf_cell_matrices(cells)

    np.testing.assert_array_equal(anchor._hopf_cell_indices(matrices), cells)


def test_hopf_cell_index_is_stable_within_a_cell() -> None:
    cells = np.array([0, 12345, 54321, anchor.HOPF_CELLS - 1])
    matrices = anchor._hopf_cell_matrices(cells)
    # A rotation nudged well inside its cell must still resolve to that cell.
    nudge = Rotation.from_rotvec(np.deg2rad([0.4, -0.3, 0.2])).as_matrix()

    np.testing.assert_array_equal(
        anchor._hopf_cell_indices(matrices @ nudge), cells
    )


def test_build_cell_lookup_rejects_rotations_off_the_hopf_grid() -> None:
    with pytest.raises(ValueError, match="Hopf cell centres"):
        anchor._build_cell_lookup(np.eye(3)[None, :, :])


class _CollectingWriter:
    def __init__(self) -> None:
        self.chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def add_chunk(
        self,
        conformers: np.ndarray,
        rotamers: np.ndarray,
        translations: np.ndarray,
    ) -> None:
        self.chunks.append(
            (conformers.copy(), rotamers.copy(), translations.copy())
        )


@pytest.mark.parametrize(
    "protein_rotation",
    [
        Rotation.from_euler("xyz", [37.0, -21.0, 12.0], degrees=True).as_matrix(),
        Rotation.from_euler("xyz", [-54.0, 16.0, 103.0], degrees=True).as_matrix(),
    ],
)
def test_precalculated_pose_rotates_translation_onto_real_protein_ring(
    tmp_path: Path,
    protein_rotation: np.ndarray,
) -> None:
    nucleotide_reference = _ring()
    # Tilted, so the conformer maps to the interior of a Hopf cell rather than
    # onto the identity, which lies exactly on a cell boundary.
    conformer_ring = nucleotide_reference @ Rotation.from_euler(
        "xyz", [9.0, 5.0, 3.0], degrees=True
    ).as_matrix() + np.array([0.7, -1.1, 0.3])
    protein_reference = _ring() * np.array([1.1, 0.9, 1.0])
    protein_center = np.array([12.0, -7.0, 4.0])
    protein_ring = protein_reference @ protein_rotation + protein_center
    protein_reference_to_world = anchor._superimpose_rotation(
        protein_reference, protein_ring
    )

    conformer_to_reference = anchor._superimpose_rotation(
        conformer_ring, nucleotide_reference
    )
    cache_cell = anchor._hopf_cell_indices(
        np.einsum(
            "ij,njk,kl->nil",
            conformer_to_reference.T,
            protein_rotation[None, :, :],
            protein_reference_to_world.T,
        )
    )
    cache_rotations = anchor._hopf_cell_matrices(cache_cell)
    canonical_translation = np.array([[0.0, 0.0, 3.8]])
    precalculated = anchor._PrecalculatedAnchor(
        rotations=cache_rotations,
        translations=canonical_translation,
        filtered_sizes=np.array([1]),
        filtered_offsets=np.array([0, 1]),
        filtered_concat=np.array([0]),
        cell_to_row=_cell_lookup(cache_cell),
        nucleotide_reference=nucleotide_reference,
        protein_reference_to_world=protein_reference_to_world,
        translation_vectors_world=canonical_translation
        @ protein_reference_to_world,
        lattice_origin=canonical_translation.min(axis=0),
        lattice_step=0.25,
        capacity_table=np.full((1, 1, 1), 12, dtype=np.int16),
    )
    protein_plane = anchor.calc_plane(protein_ring)
    protein_x = protein_ring[0] - protein_ring.mean(axis=0)
    protein_x /= np.linalg.norm(protein_x)
    protein_y = np.cross(protein_plane, protein_x)
    protein_y /= np.linalg.norm(protein_y)
    config = anchor._StackWorkConfig(
        coordinates=conformer_ring[None, :, :],
        ring_mask=np.ones(len(conformer_ring), dtype=bool),
        rotaconformers=protein_rotation[None, :, :],
        rotaconformers_index=np.array([1]),
        selected_rotamer_positions=None,
        protein_center=protein_ring.mean(axis=0),
        protein_plane=protein_plane,
        protein_x=protein_x,
        protein_y=protein_y,
        max_angle_deg=30.0,
        margin=0.5,
        dihedral_min_deg=None,
        dihedral_max_deg=None,
        output_dir=tmp_path,
        cache_size=1000,
        bucket_size=16,
        unorganized_subdirs=False,
        center_batch_size=16,
        offset_chunk_size=1000,
        writer_memory_lock=None,
        precalculated_anchor=precalculated,
        min_cell_fraction=0.0,
    )
    writer = _CollectingWriter()

    candidates, surviving = anchor._process_conformer_precalculated(
        0, config, writer
    )

    expected = np.round(
        (
            protein_ring.mean(axis=0)
            + canonical_translation[0] @ protein_rotation
            - conformer_ring.mean(axis=0) @ protein_rotation
        )
        / anchor.GRID_SPACING
    ).astype(np.int16)
    assert candidates == 1
    assert surviving == 1
    assert len(writer.chunks) == 1
    conformers, rotamers, translations = writer.chunks[0]
    np.testing.assert_array_equal(conformers, [0])
    np.testing.assert_array_equal(rotamers, [0])
    np.testing.assert_array_equal(translations, expected[None, :])

    without_protein_rotation = np.round(
        (
            protein_ring.mean(axis=0)
            + canonical_translation[0]
            - conformer_ring.mean(axis=0) @ protein_rotation
        )
        / anchor.GRID_SPACING
    ).astype(np.int16)
    assert not np.array_equal(expected, without_protein_rotation)


def test_tyr_uses_phe_reference_ring() -> None:
    assert anchor._protein_reference_name("TYR") == "PHE"
    assert anchor._protein_reference_name("TRP") == "TRP"


def test_cache_lookup_never_replaces_original_local_rotamer_ids(
    tmp_path: Path,
) -> None:
    ring = _ring()
    cache_cell = anchor._hopf_cell_indices(np.eye(3)[None, :, :])
    rotations = anchor._hopf_cell_matrices(cache_cell)
    translation = np.array([[0.0, 0.0, 3.8]])
    precalculated = anchor._PrecalculatedAnchor(
        rotations=rotations,
        translations=translation,
        filtered_sizes=np.array([1]),
        filtered_offsets=np.array([0, 1]),
        filtered_concat=np.array([0]),
        cell_to_row=_cell_lookup(cache_cell),
        nucleotide_reference=ring,
        protein_reference_to_world=np.eye(3),
        translation_vectors_world=translation,
        lattice_origin=translation.min(axis=0),
        lattice_step=0.25,
        capacity_table=np.full((1, 1, 1), 12, dtype=np.int16),
    )
    config = anchor._StackWorkConfig(
        coordinates=ring[None, :, :],
        ring_mask=np.ones(len(ring), dtype=bool),
        rotaconformers=np.repeat(np.eye(3)[None, :, :], 8, axis=0),
        rotaconformers_index=np.array([8]),
        selected_rotamer_positions=np.array([3, 7]),
        protein_center=np.zeros(3),
        protein_plane=np.array([0.0, 0.0, 1.0]),
        protein_x=np.array([1.0, 0.0, 0.0]),
        protein_y=np.array([0.0, 1.0, 0.0]),
        max_angle_deg=30.0,
        margin=0.5,
        dihedral_min_deg=None,
        dihedral_max_deg=None,
        output_dir=tmp_path,
        cache_size=1000,
        bucket_size=16,
        unorganized_subdirs=False,
        center_batch_size=16,
        offset_chunk_size=1000,
        writer_memory_lock=None,
        precalculated_anchor=precalculated,
        min_cell_fraction=0.0,
    )
    writer = _CollectingWriter()

    anchor._process_conformer_precalculated(0, config, writer)

    emitted_rotamers = np.concatenate([chunk[1] for chunk in writer.chunks])
    np.testing.assert_array_equal(emitted_rotamers, [3, 7])

    # A rotamer whose Hopf cell is absent from the cache is out of the region.
    absent = anchor._PrecalculatedAnchor(
        **{
            **vars(precalculated),
            "cell_to_row": _cell_lookup(np.array([(cache_cell[0] + 1) % anchor.HOPF_CELLS])),
        }
    )
    writer = _CollectingWriter()

    candidates, surviving = anchor._process_conformer_precalculated(
        0, replace(config, precalculated_anchor=absent), writer
    )

    assert (candidates, surviving, writer.chunks) == (0, 0, [])
