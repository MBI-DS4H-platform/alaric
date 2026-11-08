"""Focused compiled-filter checks, independent of the large recorded-pool suite."""
import base64
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ATTRACT = ROOT / "attract-jax"
sys.path[:0] = [str(ROOT), str(ATTRACT / "util"), str(ATTRACT)]

from minfor_filter import filter_poses


@pytest.fixture
def inputs(tmp_path):
    arrays = {"receptor": [[0., 0., 0.]], "ligand": [[0., 0., 0.]], "types": [1],
              "poses": [[0., 0., 0., r, 0., 0.] for r in (1.1, 1.2, 1.5, 2., 10., 0.)],
              "conformers": [0] * 6}
    for name, values in arrays.items():
        np.save(tmp_path / f"{name}.npy", np.asarray(values))
    np.savez(tmp_path / "params.npz", rc=np.ones((1, 1)), ac=np.ones((1, 1)), ivor=np.ones((1, 1)))
    return ["--filter", "--no-grid", "--input-rotvec", str(tmp_path / "poses.npy"),
            "--input-conformers", str(tmp_path / "conformers.npy"), "--input-world-centered",
            "--receptor-coordinates", str(tmp_path / "receptor.npy"),
            "--receptor-atomtypes", str(tmp_path / "types.npy"),
            "--ligand-ensemble", str(tmp_path / "ligand.npy"),
            "--ligand-atomtypes", str(tmp_path / "types.npy"),
            "--attract-par-npz", str(tmp_path / "params.npz"),
            "--energy-threshold", "-0.05", "--energy-tolerance", "0.02",
            "--output-white", str(tmp_path / "white.npy"), "--output-grey", str(tmp_path / "grey.npy")]


class VectorOracle:
    def __init__(self, energies):
        self.energies = np.asarray(energies, dtype=np.float64)

    def score_batch(self, ens, dofs, conformers=None):
        return self.energies[np.asarray(dofs[:, 3], dtype=int)], np.empty((0, 6))


def test_filter_boundaries_are_float64_and_preserve_sparse_source_ids(tmp_path, capsys):
    threshold, tolerance = -1.0, 0.1
    lower, upper = threshold - tolerance, threshold + tolerance
    energies = np.array([np.nextafter(lower, -np.inf), lower, upper, threshold,
                         np.nextafter(threshold, -np.inf)])
    poses = np.zeros((5, 6)); poses[:, 3] = np.arange(5)
    np.save(tmp_path / "indices.npy", np.array([2, 0, 3, 1], dtype=np.uint64))
    filter_poses(VectorOracle(energies), poses, np.ones(5), None,
                 threshold=threshold, tolerance=tolerance, batch_size=2,
                 pose_indices=tmp_path / "indices.npy",
                 white_output=tmp_path / "white.npy", grey_output=tmp_path / "grey.npy")
    assert np.load(tmp_path / "white.npy").tolist() == [0]
    assert np.load(tmp_path / "grey.npy").tolist() == [3, 1]
    assert np.load(tmp_path / "white.npy").dtype == np.uint64
    assert "Filter progress: 4/4" in capsys.readouterr().err
    filter_poses(VectorOracle(energies), poses, np.ones(5), None,
                 threshold=threshold, tolerance=0., white_output=tmp_path / "exact-white.npy",
                 grey_output=tmp_path / "exact-grey.npy")
    # This energy rounds up to the threshold in float32, but must remain white.
    assert np.float32(energies[4]) == np.float32(threshold)
    assert np.load(tmp_path / "exact-white.npy").tolist() == [0, 1, 4]
    assert not len(np.load(tmp_path / "exact-grey.npy"))


def test_filter_implies_score_energy_only_and_bulk(inputs):
    from minfor import parse_args
    args = parse_args(inputs)
    assert args.score and args.energy_only
    assert args.nb_kernel == "compiled" and args.oracle == "jax"
    assert args.score_mode == "bulk"


def test_empty_and_invalid_filter_selections_and_nonfinite_energies(tmp_path):
    poses = np.zeros((2, 6)); poses[:, 3] = [0, 1]
    output = dict(white_output=tmp_path / "white.npy", grey_output=tmp_path / "grey.npy")
    np.save(tmp_path / "selection.npy", np.empty(0, dtype=np.uint64))
    report = filter_poses(VectorOracle([-2., 0.]), poses, np.ones(2), None,
                          threshold=-1., tolerance=.1, pose_indices=tmp_path / "selection.npy", **output)
    assert report["input_count"] == 0
    assert np.load(output["white_output"]).shape == np.load(output["grey_output"]).shape == (0,)
    for selection in (np.array([True, False]), np.array([-1]), np.array([2])):
        np.save(tmp_path / "selection.npy", selection)
        with pytest.raises(ValueError, match="pose ind"):
            filter_poses(VectorOracle([-2., 0.]), poses, np.ones(2), None,
                         threshold=-1., tolerance=.1, pose_indices=tmp_path / "selection.npy", **output)
    with pytest.raises(ValueError, match="finite float64"):
        filter_poses(VectorOracle([np.nan, 0.]), poses, np.ones(2), None,
                     threshold=-1., tolerance=.1, **output)


@pytest.mark.parametrize("extra", [
    ["--oracle", "numpy"], ["--nb-kernel", "jax"], ["--energy-tolerance", "-1"],
    ["--energy-threshold", "nan"], ["--benchmark-steady-state"],
])
def test_invalid_filter_modes_are_rejected(inputs, extra):
    from minfor import parse_args
    with pytest.raises(SystemExit):
        parse_args(inputs + extra)


def test_real_compiled_filter_cli_negative_threshold_indices_and_progress(inputs, tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    command = [sys.executable, str(ATTRACT / "util/minfor.py"), *inputs, "--score-batch-size", "2"]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert np.load(tmp_path / "white.npy").tolist() == [0, 1]
    assert np.load(tmp_path / "grey.npy").tolist() == [2]
    assert "Filter progress: 6/6" in result.stderr
    assert "Minimization" not in result.stdout
    np.save(tmp_path / "selection.npy", np.array([4, 2, 1], dtype=np.uint64))
    result = subprocess.run(command + ["--pose-indices", str(tmp_path / "selection.npy")],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert np.load(tmp_path / "white.npy").tolist() == [1]
    assert np.load(tmp_path / "grey.npy").tolist() == [2]


@pytest.mark.parametrize("rotation", ["euler", "rotvec"])
def test_compiled_gridless_matches_all_pairs_reference_and_batching(rotation, monkeypatch):
    from jax_scorer import JaxScoreOracle
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    rng = np.random.default_rng(17)
    receptor = rng.normal(size=(2, 4, 3)) * 3
    ligand = rng.normal(size=(3, 5, 3))
    kwargs = dict(receptor_ensemble=receptor, receptor_atomtypes=np.array([1, 2, 3, 99]),
                  receptor_charges=np.array([.2, -.3, .1, 9.]), ligand_ensemble=ligand,
                  ligand_atomtypes=np.array([2, 1, 3, 4, 99]),
                  ligand_charges=np.array([-.3, .2, .1, -.1, 8.]),
                  attract_par_npz=str(ATTRACT / "attract-original/attract-par.npz"),
                  lig_pivot=np.array([.4, -.2, .3]), energy_only=True, grid_free=True,
                  dof_type=rotation)
    dofs = rng.normal(size=(7, 6))
    ens = np.array([1, 2, 2, 1, 2, 1, 2]); conformers = np.array([0, 2, 1, 1, 0, 2, 1])
    dofs[-1, :3] = 0
    dofs[-1, 3:] = receptor[1, 0] - ligand[1, 0]  # exercise the short-distance floor
    compiled = JaxScoreOracle(**kwargs, backend="jax", nb_kernel="compiled")
    reference = JaxScoreOracle(**kwargs, backend="numpy")
    actual, gradients = compiled.score_batch(ens, dofs, conformers)
    expected, _ = reference.score_batch(ens, dofs, conformers)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)
    assert gradients.shape == (0, 6)
    rechunked = np.concatenate([compiled.score_batch(ens[start:start + 2], dofs[start:start + 2],
                                                   conformers[start:start + 2])[0]
                               for start in range(0, len(dofs), 2)])
    np.testing.assert_array_equal(actual, rechunked)


@pytest.mark.parametrize("minimum", [0, 2, 4, 20])
def test_subprocess_cascade_routes_match_existing_contract(tmp_path, minimum):
    from alaric.energy_filter import cascade_filter
    from alaric.energy_filter_action import _cascade_minfor
    steps = [{"plateau_distance": 5., "neighbour_distance": 7., "max_error": 1.},
             {"plateau_distance": 10., "neighbour_distance": 12., "max_error": .05}]
    grids = [np.array([-3., -2., 0., -1., 10., -float(2**53), float(2**53), -.99]),
             np.array([-3., -1.1, 0., -.95, 10., -float(2**53), float(2**53), -1.02])]
    exact = np.array([-3., -2., 0., 1., 10., -2., 1., -1.2])
    ids = np.arange(len(exact), dtype=np.uint64)
    poses = np.zeros((len(ids), 6)); poses[:, 3] = ids
    calls = []

    def invoke(selection, step):
        calls.append(None if step is None else step["plateau_distance"])
        selected_path = None
        if selection is not None:
            selected_path = tmp_path / "selection.npy"
            np.save(selected_path, selection)
        energy = exact if step is None else grids[0 if step["plateau_distance"] == 5. else 1]
        report = filter_poses(VectorOracle(energy), poses, np.ones(len(ids)), None,
                             threshold=-1., tolerance=0. if step is None else step["max_error"],
                             pose_indices=selected_path, batch_size=3, precision_safe=step is not None,
                             white_output=tmp_path / "white.npy", grey_output=tmp_path / "grey.npy")
        white, grey = np.load(tmp_path / "white.npy"), np.load(tmp_path / "grey.npy")
        flags = (np.zeros(len(grey), dtype=bool) if step is None else
                 np.unpackbits(np.frombuffer(base64.b64decode(report["precision_fallback_bits"]), dtype=np.uint8),
                               count=len(grey)).astype(bool))
        return white, grey, flags

    actual, stats = _cascade_minfor(len(ids), steps, minimum, invoke)
    expected, expected_stats = cascade_filter(
        ids, threshold=-1., cascade=steps, min_cascade=minimum,
        grid_score=lambda batch, step: grids[0 if step["plateau_distance"] == 5. else 1][batch],
        gridless_score=lambda batch: exact[batch])
    np.testing.assert_array_equal(actual, expected)
    assert stats == expected_stats
    if minimum == 0:
        assert calls == [5., 10., None]
