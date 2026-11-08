"""Optional real 1b7f W8 oracle; never enabled by the normal fast suite.

From the repository root, run explicitly (potentially minutes and substantial RAM):

  ALARIC_W8_INTEGRATION=1 conda run --live-stream -n alaric python -m pytest -q tests/test_w8_energy_filter_integration.py

The ignored source pool must already exist. Its generating recipe and blessed
ARC digest are recorded in fixtures/w8/energy-filter-integration.json. Missing
inputs FAIL after opt-in, rather than silently turning acceptance into a skip.
This tests both actual local/local-chunk shell templates, not a scheduling mock;
local-chunk runs three chunks and the organize replay.
Only the recorded dom2 range is covered; other/clipped receptors need their own
recorded fixtures and finite-sample bounds. Bounds apply only to scores eligible
for grid-only decisions under the production float64 precision guard; every
unsafe score encountered by the cascade must reach exact gridless scoring.
Bounds here are empirical, not proofs.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "tests/fixtures/w8/energy-filter-integration.json"
pytestmark = pytest.mark.skipif(
    os.environ.get("ALARIC_W8_INTEGRATION") != "1",
    reason="explicit slow oracle: set ALARIC_W8_INTEGRATION=1 (module docstring)",
)


def _arc_snapshot(directory):
    from poses import _open_arc_bytes, discover_organized

    return [(p.name.removesuffix(".zst"), _open_arc_bytes(p))
            for p in discover_organized(directory)]


def _rows(directory, ids=None):
    from poses import PoseReader, select_pose_indices

    if ids is None:
        ids = np.arange(PoseReader.get_nposes(directory), dtype=np.int64)
    chunk = select_pose_indices(directory, ids)
    return np.column_stack((chunk.conformers, chunk.rotamers, chunk.translations_grid))


@pytest.fixture(scope="module")
def recorded_pool(tmp_path_factory):
    sys.path.insert(0, str(ROOT / "alaric"))
    from poses import PoseReader

    metadata = json.loads(META.read_text())
    source = ROOT / metadata["source"]
    assert source.is_dir(), (
        f"Missing ignored 1b7f pool {source}; prepare using "
        f"{metadata['source_recipe']} in the alaric environment"
    )
    digest = hashlib.sha256()
    for _, raw in _arc_snapshot(source):
        digest.update(raw)
    assert digest.hexdigest() == metadata["source_arc_sha256"]
    n = PoseReader.get_nposes(source)
    assert n >= metadata["sample_count"]
    ids = np.linspace(0, n - 1, metadata["sample_count"], dtype=np.int64)
    assert len(np.unique(ids)) == metadata["sample_count"]
    work = tmp_path_factory.mktemp("w8-real-1b7f")
    np.save(work / "source-ids.npy", ids)
    pool = work / "sample"
    subprocess.run([
        sys.executable, str(ROOT / "alaric/select-poses.py"), str(source),
        str(work / "source-ids.npy"), str(pool), "--base", "0", "--compress",
        "--nprocs", "1", "--chunk-poses", "999",
    ], check=True, cwd=ROOT)
    assert PoseReader.get_nposes(pool) == len(ids)
    return metadata, pool, work


@pytest.fixture(scope="module")
def measured_ranges(recorded_pool):
    """Measure independent grid and gridless JAX oracles on the recorded sample."""
    import energy_filter_action as action
    from energy_filter import DEFAULT_SCORE_FILTER
    from grid_generator import generate_grid
    from jax_scorer import JaxScoreOracle
    from minfor import rotvec_world_to_attr_np
    from nb_kernel.forcefields.nonbon8.params import load_params

    metadata, pool, work = recorded_pool
    results = []
    for range_meta in metadata["ranges"]:
        protein = ROOT / range_meta["protein"]
        assert hashlib.sha256(protein.read_bytes()).hexdigest() == range_meta["protein_sha256"]
        measure_dir = work / (range_meta["protein_range"] + "-measure")
        measure_dir.mkdir()
        # Use the action's shared preparation path, then call the scorer directly. This
        # keeps the measurement independent of the action's cascade implementation and
        # avoids relying on internal scorer callbacks that no longer exist.
        action._prepare_minfor_inputs(pool, protein, metadata["sequence"], metadata["exclude"], measure_dir)
        poses = np.load(measure_dir / "poses.rotvec.npy", mmap_mode="r", allow_pickle=False)
        conformers = np.load(measure_dir / "poses.conformers.npy", allow_pickle=False).astype(np.int32, copy=False)
        nposes = len(poses)
        receptor = np.asarray(np.load(measure_dir / "receptor-coor.npy"), dtype=np.float64)
        receptor_types = np.asarray(np.load(measure_dir / "receptor-atomtypes.npy"), dtype=np.int32)
        receptor_charges = np.zeros(len(receptor), dtype=np.float64)
        ligand = np.asarray(np.load(measure_dir / "ligand-ensemble.npy"), dtype=np.float64)
        if ligand.ndim == 2:
            ligand = ligand[None, :, :]
        ligand_types = np.asarray(np.load(measure_dir / "ligand-atomtypes.npy"), dtype=np.int32)
        ligand_charges = np.zeros(len(ligand_types), dtype=np.float64)
        pivot = np.mean(ligand[0], axis=0)
        dofs = rotvec_world_to_attr_np(poses, pivot, copy=True)
        ensembles = np.ones(nposes, dtype=np.int32)
        par_path = ROOT / "attract-jax/attract-original/attract-par.npz"
        oracle_params = {
            "receptor_ensemble": receptor,
            "receptor_atomtypes": receptor_types,
            "receptor_charges": receptor_charges,
            "ligand_ensemble": ligand,
            "ligand_atomtypes": ligand_types,
            "ligand_charges": ligand_charges,
            "attract_par_npz": str(par_path),
            "lig_pivot": pivot,
            "epsilon": 15.0,
            "score_mode": "bulk",
            "score_batch_size": 1001,
            "nb_kernel": "compiled",
            "energy_only": True,
            "backend": "jax",
            "dof_type": "rotvec",
            "no_conformer_grouping": True,
        }
        exact_oracle = JaxScoreOracle(**oracle_params, grid_free=True)
        exact, _ = exact_oracle.score_batch(ensembles, dofs, conformers)
        ff_params = load_params(str(par_path))
        grids = []
        for step in DEFAULT_SCORE_FILTER["cascade"]:
            grid = generate_grid(
                rec_coords=receptor,
                rec_atomtypes=receptor_types,
                rec_charges_raw=receptor_charges,
                ff_params=ff_params,
                forcefield="nonbon8",
                ffelec=math.sqrt(332.053986 / 15.0),
                plateaudis=float(step["plateau_distance"]),
                neighbourdis=float(step["neighbour_distance"]),
                lig_atomtypes=ligand_types,
                return_gradients=False,
                force_jax=False,
            )
            grid_oracle = JaxScoreOracle(**oracle_params, grid_object=grid)
            grid_scores, _ = grid_oracle.score_batch(ensembles, dofs, conformers)
            grids.append(np.asarray(grid_scores, dtype=np.float64))
        measured = {"exact": np.asarray(exact, dtype=np.float64), "grids": grids}
        assert len(measured["exact"]) == nposes
        assert all(len(values) == nposes for values in grids)
        assert np.all(np.isfinite(measured["exact"]))
        assert all(np.all(np.isfinite(values)) for values in grids)
        measured["sample_ids"] = np.arange(nposes, dtype=np.int64)
        measured["source_ids"] = np.load(work / "source-ids.npy", allow_pickle=False)
        measured["measurement_dir"] = measure_dir
        np.save(measure_dir / "measured-exact.npy", measured["exact"], allow_pickle=False)
        for index, values in enumerate(grids, start=1):
            np.save(measure_dir / f"measured-grid-step-{index}.npy", values, allow_pickle=False)
        measured.update(range_meta=range_meta)
        results.append(measured)
    return results


def _assert_bounds(measured):
    from energy_filter import DEFAULT_SCORE_FILTER, grid_precision_unsafe

    threshold = float(np.median(measured["exact"]))
    for step, grid in zip(DEFAULT_SCORE_FILTER["cascade"], measured["grids"]):
        unsafe = grid_precision_unsafe(grid, threshold, step["max_error"])
        # Pin the exact production guard, rather than hiding arbitrary large
        # energies behind an unrelated cutoff in this empirical bound check.
        precision_margin = step["max_error"] / 2
        expected_unsafe = ((np.spacing(np.abs(grid)) >= precision_margin)
                           | (np.spacing(abs(threshold)) >= precision_margin))
        np.testing.assert_array_equal(unsafe, expected_unsafe)
        eligible = ~unsafe
        assert np.any(eligible), "recorded sample needs grid-only-eligible scores"
        eligible_positions = np.flatnonzero(eligible)
        errors = np.abs(grid[eligible] - measured["exact"][eligible])
        worst_position = int(eligible_positions[int(np.argmax(errors))])
        error = float(errors.max())
        assert error < step["max_error"] / 2, (
            f"{measured['range_meta']['protein_range']} step={step}: empirical "
            f"eligible max abs error {error} must be strictly < {step['max_error'] / 2}; "
            f"eligible={int(eligible.sum())}, precision-unsafe={int(unsafe.sum())}; "
            f"sample_id={int(measured['sample_ids'][worst_position])}, "
            f"source_pose_id={int(measured['source_ids'][worst_position])}, "
            f"grid_energy={float(grid[worst_position])}, "
            f"exact_energy={float(measured['exact'][worst_position])}, "
            f"measurements={measured['measurement_dir']}"
        )


def _observe_precision_routes(measured):
    """Replay real measured energies through the production cascade, tracing IDs."""
    from energy_filter import DEFAULT_SCORE_FILTER, cascade_filter, grid_precision_unsafe

    exact = measured["exact"]
    threshold = float(np.median(exact))
    grids = {(step["plateau_distance"], step["neighbour_distance"]): grid
             for step, grid in zip(DEFAULT_SCORE_FILTER["cascade"], measured["grids"])}
    unsafe_ids = set()
    fallback_counts = []
    grid_counts = []
    exact_ids = []

    def grid_score(ids, step):
        # Once precision-unsafe, a pose must not regain a grid-only decision
        # from a later step with different precision or error characteristics.
        assert unsafe_ids.isdisjoint(ids.tolist())
        scores = grids[(step["plateau_distance"], step["neighbour_distance"])][ids]
        unsafe = grid_precision_unsafe(scores, threshold, step["max_error"])
        unsafe_ids.update(ids[unsafe].tolist())
        fallback_counts.append(int(np.count_nonzero(unsafe)))
        grid_counts.append(len(ids))
        return scores

    def gridless_score(ids):
        exact_ids.extend(ids.tolist())
        return exact[ids]

    selected, stats = cascade_filter(
        np.arange(len(exact), dtype=np.int64), threshold=threshold,
        cascade=DEFAULT_SCORE_FILTER["cascade"],
        min_cascade=DEFAULT_SCORE_FILTER["min_cascade"],
        grid_score=grid_score, gridless_score=gridless_score,
    )
    assert unsafe_ids.issubset(exact_ids), "precision-unsafe pose bypassed exact scoring"
    assert len(exact_ids) == len(set(exact_ids)), "exact fallback scored a pose twice"
    assert stats["grid_counts"] == grid_counts
    assert stats["precision_fallback_counts"] == fallback_counts
    assert stats["precision_fallback_count"] == len(unsafe_ids) == sum(fallback_counts)
    assert stats["gridless_count"] == len(exact_ids)
    np.testing.assert_array_equal(selected, np.flatnonzero(exact < threshold))
    return stats


def test_recorded_range_finite_sample_grid_error_bounds(measured_ranges):
    for measured in measured_ranges:
        _assert_bounds(measured)


def test_recorded_precision_unsafe_scores_always_take_exact_route(measured_ranges):
    fallback_count = 0
    for measured in measured_ranges:
        stats = _observe_precision_routes(measured)
        fallback_count += stats["precision_fallback_count"]
    assert fallback_count > 0, "recorded fixture must exercise the precision fallback"


def _run_template(mode, values, nchunks=3):
    template = (ROOT / "alaric/middle/templates" / mode / "energy-filter.sh").read_text()
    if mode.endswith("-chunk"):
        # Run each chunkN body, then the organize body, as the generated run.sh does.
        chunk, organize = template.split("### ORGANIZE ###")
        values = {**values, "nchunks": nchunks,
                  "chunks_path": shlex.quote(str(values["output_path"]).strip("'") + "-CHUNKS")}
        scripts = [chunk.replace("{{ chunk_index }}", str(i)) for i in range(1, nchunks + 1)]
        scripts.append(organize)
    else:
        scripts = [template]
    for script in scripts:
        for key, value in values.items():
            script = script.replace("{{ " + key + " }}", str(value))
        assert "{{" not in script, script
        subprocess.run(["bash", "-e", "-c", script], check=True, cwd=ROOT)


def test_actual_energy_filter_bytes_ids_and_stats_across_templates(recorded_pool, measured_ranges):
    from energy_filter import DEFAULT_SCORE_FILTER
    from npy_io import load_npy

    metadata, pool, work = recorded_pool
    # Each template runs each process count; scorer chunks straddle min_cascade.
    configurations = [("local", 1, 999), ("local", 2, 1001),
                      ("local-chunk", 1, 1001), ("local-chunk", 2, 999)]
    for measured in measured_ranges:
        _assert_bounds(measured)  # Refuse cascade acceptance without this range's bound.
        range_meta = measured["range_meta"]
        exact = measured["exact"]
        expected_route = _observe_precision_routes(measured)
        threshold = float(np.median(exact))
        kept = np.flatnonzero(exact < threshold)
        assert 0 < len(kept) < len(exact), "fixture must exercise keep and reject"
        expected_rows = _rows(pool, kept)
        baseline = None
        route_stats = {}
        for mode, nprocs, chunk in configurations:
            for label, cascade in [("gridless", []), ("cascade", DEFAULT_SCORE_FILTER["cascade"])]:
                output = work / f"{range_meta['protein_range']}-{mode}-{nprocs}-{chunk}-{label}"
                q = lambda value: shlex.quote(str(value))
                _run_template(mode, {
                    "python": q(sys.executable), "alaric_dir": q(ROOT / "alaric"),
                    "input_result_path": q(pool), "protein_path": q(ROOT / range_meta["protein"]),
                    "sequence": q(metadata["sequence"]), "fragment": metadata["fragment"],
                    "protein_range": q(range_meta["protein_range"]), "threshold": repr(threshold),
                    "epsilon": 15, "cascade_json": q(json.dumps(cascade)),
                    "min_cascade": DEFAULT_SCORE_FILTER["min_cascade"],
                    "exclude_args": " ".join("--pdb-exclude " + q(x) for x in metadata["exclude"]),
                    "nprocs": nprocs, "chunk_poses": chunk, "output_path": q(output),
                })
                stats = json.loads((output / "energy-filter-stats.json").read_text())
                assert stats["input_count"] == len(exact)
                assert stats["kept_count"] == len(kept)
                order = load_npy(output / "order-array.npy.zst")
                # Order array maps organized output rows to ascending selected-ID
                # request positions. This recovers source IDs even with duplicate poses.
                np.testing.assert_array_equal(np.sort(order), np.arange(len(kept)))
                ordered_ids = kept[order.astype(np.int64)]
                np.testing.assert_array_equal(_rows(output), expected_rows[order.astype(np.int64)])
                # select-poses publishes organize's provenance tags as this
                # order array (and removes the intermediate provenance.npy).
                snapshot = (_arc_snapshot(output), ordered_ids.copy(),
                            (order.dtype.str, order.shape, order.tobytes()))
                if baseline is None:
                    baseline = snapshot
                else:
                    assert snapshot[0] == baseline[0]
                    np.testing.assert_array_equal(snapshot[1], baseline[1])
                    assert snapshot[2] == baseline[2]
                if label in route_stats:
                    assert stats == route_stats[label]
                else:
                    route_stats[label] = stats
                if label == "gridless":
                    assert stats["grid_counts"] == []
                    assert stats["gridless_count"] == len(exact)
                    assert stats["precision_fallback_counts"] == []
                    assert stats["precision_fallback_count"] == 0
                else:
                    assert stats["grid_counts"][0] == len(exact)
                    for key in ("grid_counts", "gridless_count", "cascade_exit",
                                "precision_fallback_counts", "precision_fallback_count"):
                        assert stats[key] == expected_route[key]
        # Route counts differ by algorithm, but externally visible metadata agrees.
        route_keys = {"grid_counts", "gridless_count", "cascade_exit",
                      "precision_fallback_counts", "precision_fallback_count", "minfor_invocations"}
        for key in route_stats["gridless"].keys() - route_keys:
            assert route_stats["gridless"][key] == route_stats["cascade"][key]
