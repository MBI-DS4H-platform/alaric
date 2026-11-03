"""Optional real 1b7f W8 oracle; never enabled by the normal fast suite.

From the repository root, run explicitly (potentially minutes and substantial RAM):

  ALARIC_W8_INTEGRATION=1 conda run --live-stream -n alaric python -m pytest -q tests/test_w8_energy_filter_integration.py

The ignored source pool must already exist. Its generating recipe and blessed
ARC digest are recorded in fixtures/w8/energy-filter-integration.json. Missing
inputs FAIL after opt-in, rather than silently turning acceptance into a skip.
This tests both actual local/local-chunk shell templates, not a scheduling mock.
Only the recorded dom2 range is covered; other/clipped receptors need their own
recorded fixtures and finite-sample bounds. Bounds apply only to scores eligible
for grid-only decisions under the production float64 precision guard; every
unsafe score encountered by the cascade must reach exact gridless scoring.
Bounds here are empirical, not proofs.
"""
from __future__ import annotations

import hashlib
import json
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
    """Observe actual action scorers; no energy, grid or selected-pose mock."""
    import energy_filter_action as action
    from energy_filter import DEFAULT_SCORE_FILTER

    metadata, pool, work = recorded_pool
    results = []
    for range_meta in metadata["ranges"]:
        protein = ROOT / range_meta["protein"]
        assert hashlib.sha256(protein.read_bytes()).hexdigest() == range_meta["protein_sha256"]
        measured = {}
        original = action.cascade_filter

        def observe(ids, **kwargs):
            # Capture the production scorer closures, then evaluate every recorded
            # sample at every default step, including poses an earlier step rejects.
            exact = np.asarray(kwargs["gridless_score"](ids))
            grids = [np.asarray(kwargs["grid_score"](ids, step))
                     for step in DEFAULT_SCORE_FILTER["cascade"]]
            assert np.all(np.isfinite(exact))
            assert all(np.all(np.isfinite(g)) for g in grids)
            measured.update(exact=exact, grids=grids)
            return original(ids, **kwargs)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(action, "cascade_filter", observe)
            action.run_energy_filter(
                pool, protein, sequence=metadata["sequence"],
                fragment=metadata["fragment"], protein_range=range_meta["protein_range"],
                threshold=0.0, cascade=[], exclude=metadata["exclude"],
                output_dir=work / (range_meta["protein_range"] + "-probe"),
                nprocs=1, chunk_poses=1001,
            )
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
        error = float(np.max(np.abs(grid[eligible] - measured["exact"][eligible])))
        assert error < step["max_error"] / 2, (
            f"{measured['range_meta']['protein_range']} step={step}: empirical "
            f"eligible max abs error {error} must be strictly < {step['max_error'] / 2}; "
            f"eligible={int(eligible.sum())}, precision-unsafe={int(unsafe.sum())}"
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


def _run_template(mode, values):
    template = (ROOT / "alaric/middle/templates" / mode / "energy-filter.sh").read_text()
    for key, value in values.items():
        template = template.replace("{{ " + key + " }}", str(value))
    assert "{{" not in template, template
    subprocess.run(["bash", "-e", "-c", template], check=True, cwd=ROOT)


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
                      "precision_fallback_counts", "precision_fallback_count"}
        for key in route_stats["gridless"].keys() - route_keys:
            assert route_stats["gridless"][key] == route_stats["cascade"][key]
