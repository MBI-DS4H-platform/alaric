"""A11 cell-fill parity for the W6 anchor refactor."""
from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
ALARIC = ROOT / "alaric"
if str(ALARIC) not in sys.path:
    sys.path.insert(0, str(ALARIC))

import anchor  # noqa: E402
import anchor_precalc  # noqa: E402
from anchor_constants import MIN_CELL_FRACTION  # noqa: E402


class _CollectingWriter:
    def __init__(self) -> None:
        self.chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def add_chunk(self, conformers, rotamers, translations) -> None:
        self.chunks.append((conformers.copy(), rotamers.copy(), translations.copy()))


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


def _precalculated_fixture() -> tuple[anchor_precalc._PrecalculatedAnchor, np.ndarray, np.ndarray, np.ndarray]:
    nucleotide_reference = _ring()
    conformer_ring = nucleotide_reference @ Rotation.from_euler(
        "xyz", [9.0, 5.0, 3.0], degrees=True
    ).as_matrix() + np.array([0.7, -1.1, 0.3])
    protein_reference = _ring() * np.array([1.1, 0.9, 1.0])
    protein_rotation = Rotation.from_euler(
        "xyz", [37.0, -21.0, 12.0], degrees=True
    ).as_matrix()
    protein_center = np.array([12.0, -7.0, 4.0])
    protein_ring = protein_reference @ protein_rotation + protein_center
    protein_reference_to_world = anchor_precalc._superimpose_rotation(
        protein_reference, protein_ring
    )

    conformer_to_reference = anchor_precalc._superimpose_rotation(
        conformer_ring, nucleotide_reference
    )
    cell = anchor_precalc._hopf_cell_indices(
        np.einsum(
            "ij,jk,kl->il",
            conformer_to_reference.T,
            protein_rotation,
            protein_reference_to_world.T,
        )[None, :, :]
    )
    rotations = anchor_precalc._hopf_cell_matrices(cell)
    cell_to_row = np.full(anchor_precalc.HOPF_CELLS, -1, dtype=np.int32)
    cell_to_row[cell] = np.arange(len(cell), dtype=np.int32)
    canonical_translation = np.array([[0.0, 0.0, 3.8]])
    precalc = anchor_precalc._PrecalculatedAnchor(
        rotations=rotations,
        translations=canonical_translation,
        filtered_sizes=np.array([1], dtype=np.int32),
        filtered_offsets=np.array([0, 1], dtype=np.int64),
        filtered_concat=np.array([0], dtype=np.int32),
        cell_to_row=cell_to_row,
        nucleotide_reference=nucleotide_reference,
        protein_reference_to_world=protein_reference_to_world,
        translation_vectors_world=canonical_translation @ protein_reference_to_world,
        lattice_origin=canonical_translation.min(axis=0),
        lattice_step=0.25,
        # One accepted translation out of three lattice points makes the cell
        # fraction exactly the inclusive 1/3 boundary.
        capacity_table=np.full((1, 1, 1), 3, dtype=np.int16),
    )
    return precalc, conformer_ring, protein_rotation, protein_ring.mean(axis=0)


def test_candidate_cells_and_cell_fill_match_anchor_emission_at_one_third(
    tmp_path: Path,
) -> None:
    precalc, ring_coords, rotamer, protein_center = _precalculated_fixture()

    owners, grid, counts, capacity = anchor_precalc.candidate_cells(
        precalc, ring_coords, rotamer[None, :, :], protein_center
    )

    assert len(owners) == len(grid) == len(counts) == len(capacity) == 1
    assert counts[0] / capacity[0] == MIN_CELL_FRACTION
    fill_count, fill_capacity = anchor_precalc.cell_fill(
        precalc, ring_coords, rotamer, protein_center, grid[0]
    )
    assert (fill_count, fill_capacity) == (int(counts[0]), int(capacity[0]))

    config = anchor._StackWorkConfig(
        coordinates=ring_coords[None, :, :],
        ring_mask=np.ones(len(ring_coords), dtype=bool),
        rotaconformers=rotamer[None, :, :],
        rotaconformers_index=np.array([1]),
        selected_rotamer_positions=None,
        protein_center=protein_center,
        output_dir=tmp_path,
        cache_size=1000,
        bucket_size=16,
        unorganized_subdirs=False,
        center_batch_size=16,
        writer_memory_lock=None,
        precalculated_anchor=precalc,
        min_cell_fraction=MIN_CELL_FRACTION,
    )
    writer = _CollectingWriter()
    _, emitted = anchor._process_conformer_precalculated(0, config, writer)

    assert emitted == 1
    assert len(writer.chunks) == 1
    assert writer.chunks[0][2].shape == (1, 3)


def test_lazy_cell_capacity_is_cached_by_phase_bin(monkeypatch) -> None:
    precalc, ring_coords, rotamer, protein_center = _precalculated_fixture()
    lazy_precalc = replace(precalc, capacity_table=None)
    owners, grid, _, _ = anchor_precalc.candidate_cells(
        lazy_precalc, ring_coords, rotamer[None, :, :], protein_center
    )
    assert len(owners) == 1

    calls = 0
    original = anchor_precalc._capacity_for_phase_bin

    def count_capacity(phase_bin, configured, resolution=64):
        nonlocal calls
        calls += 1
        return original(phase_bin, configured, resolution)

    monkeypatch.setattr(anchor_precalc, "_capacity_for_phase_bin", count_capacity)
    cache: dict[tuple[int, int, int], int] = {}
    first = anchor_precalc.cell_fill(
        lazy_precalc,
        ring_coords,
        rotamer,
        protein_center,
        grid[0],
        capacity_cache=cache,
    )
    second = anchor_precalc.cell_fill(
        lazy_precalc,
        ring_coords,
        rotamer,
        protein_center,
        grid[0],
        capacity_cache=cache,
    )

    assert first == second
    assert first[1] > 0
    assert calls == 1
    assert len(cache) == 1
