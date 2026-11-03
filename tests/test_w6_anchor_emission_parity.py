"""Real 1b7f parity between A9 rows and the production anchor action."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
for path in (ROOT, TESTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from alaric import (
    anchor,
    anchor_constants,
    anchor_precalc,
    gaussian_cache,
    gaussian_check,
    gaussian_filter,
    npy_io,
)  # noqa: E402
from test_w6_gaussian_check import _prepare_case  # noqa: E402


class _CollectingWriter:
    def __init__(self) -> None:
        self.chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def add_chunk(self, conformers, rotamers, translations) -> None:
        self.chunks.append(
            (conformers.copy(), rotamers.copy(), translations.copy())
        )


def test_real_1b7f_a9_emission_matches_anchor_action_for_all_filters(tmp_path: Path) -> None:
    best_fit, proteins, jobs, identities = _prepare_case(tmp_path)
    job = jobs[0]
    a9 = gaussian_check.run_gaussian_check(
        [job], best_fit, proteins, "1b7f", identities, ROOT
    )
    a9_job = a9["jobs"][0]
    assert "error" not in a9_job

    spec = gaussian_filter.load_spec()
    ring_references = gaussian_check._ring_reference_paths(ROOT)
    model_data = spec["model"]
    model = gaussian_check.Model(
        mean=tuple(model_data["mean"]),
        covariance=tuple(tuple(row) for row in model_data["covariance"]),
        precision=tuple(tuple(row) for row in model_data["precision"]),
        normalizer=float(model_data["normalizer"]),
    )
    prepared = gaussian_check._prepare_jobs(
        [job], best_fit, proteins, ["1b7f"], ring_references, model
    )[0]["prepared"]

    nucleotide_index = 0 if job["nucleotide"] == "first" else 1
    nucleotide_code = gaussian_check._nucleotide_reference_code(
        job["sequence"][nucleotide_index]
    )
    protein_path = gaussian_check._protein_path(proteins, job["range"])
    protein_atoms = gaussian_check.parse_pdb(protein_path.read_text(encoding="utf-8"))
    _, protein_resname = gaussian_check.residue_ring_coordinates(
        protein_atoms, int(job["protein_resid"]), structure_label=str(protein_path)
    )
    protein_code = gaussian_check._protein_reference_code(protein_resname)
    nucleotide_reference_path = ROOT / "ring-refe" / f"refe-{nucleotide_code}.npy"
    protein_reference_path = ROOT / "ring-refe" / f"refe-{protein_code}.npy"

    shared_geometry = None
    for name in gaussian_filter.all_names():
        verified = gaussian_cache.verify_filter_cache(spec, name, ROOT)
        arrays = {key: npy_io.load_npy(path) for key, path in verified.items()}
        if shared_geometry is None:
            # The actual action loader constructs the full capacity table from
            # the same A10 rings and protein geometry used by the A9 job.
            precalculated = anchor._load_precalculated_anchor(
                ROOT / spec["thresholds"][name]["precalc"]["dir"],
                nucleotide_reference_path,
                protein_reference_path,
                prepared.protein_ring,
                arrays=arrays,
            )
            shared_geometry = (
                precalculated.protein_reference_to_world,
                precalculated.capacity_table,
                precalculated.lattice_step,
            )
        else:
            validated = anchor_precalc.prepare_precalculated_arrays(arrays)
            precalculated = anchor_precalc.build_precalculated_anchor(
                validated,
                prepared.nucleotide_cache_reference,
                prepared.protein_cache_reference,
                prepared.protein_ring,
                shared_geometry=shared_geometry,
            )

        config = anchor._StackWorkConfig(
            coordinates=prepared.ring_coordinates[None, :, :],
            ring_mask=np.ones(len(prepared.ring_coordinates), dtype=bool),
            rotaconformers=prepared.rotamer_matrix[None, :, :],
            rotaconformers_index=np.array([1], dtype=np.int64),
            selected_rotamer_positions=None,
            protein_center=prepared.protein_center,
            output_dir=tmp_path / f"anchor-action-{name}",
            cache_size=16,
            bucket_size=16,
            unorganized_subdirs=False,
            center_batch_size=16,
            writer_memory_lock=None,
            precalculated_anchor=precalculated,
            min_cell_fraction=anchor_constants.MIN_CELL_FRACTION,
        )
        writer = _CollectingWriter()
        _, surviving = anchor._process_conformer_precalculated(0, config, writer)
        target_grid = np.asarray(job["grid"], dtype=np.int16)
        action_emitted = any(
            bool(np.any(np.all(translations == target_grid, axis=1)))
            for _, _, translations in writer.chunks
        )
        assert action_emitted is a9_job["filters"][name]["emitted"], name
        assert bool(writer.chunks) == (surviving > 0), name
        del arrays, precalculated, config, writer
