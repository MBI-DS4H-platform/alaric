#!/usr/bin/env python3
"""Run one grid-assisted, gridless-correct energy-filter action on a pose pool."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_ALARIC = _ROOT / "alaric"
_ATTRACT = _ROOT / "attract-jax"
_ATTRACT_UTIL = _ATTRACT / "util"
for _path in (str(_ALARIC), str(_ROOT), str(_ATTRACT_UTIL), str(_ATTRACT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from energy_filter import cascade_filter  # noqa: E402
from library import config  # noqa: E402
from parse_pdb import parse_pdb  # noqa: E402


def _load_reducer():
    path = _ROOT / "util" / "reduce-npy.py"
    spec = importlib.util.spec_from_file_location("energy_filter_reduce_npy", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load ATTRACT reducer {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_energy_filter(
    pose_dir: str | Path,
    protein_pdb: str | Path,
    *,
    sequence: str,
    fragment: int,
    protein_range: str,
    threshold: float,
    epsilon: float = 15.0,
    cascade: list[dict[str, Any]] | None = None,
    min_cascade: int = 1000,
    exclude: list[str] | None = None,
    output_dir: str | Path,
    nprocs: int = 1,
    chunk_poses: int = 10_000,
) -> dict[str, Any]:
    """Score one fragment/range and write the passing poses in original order."""
    pose_dir = Path(pose_dir)
    protein_pdb = Path(protein_pdb)
    output_dir = Path(output_dir)
    if epsilon != 15.0:
        raise ValueError("energy-filter requires epsilon: 15")
    if nprocs <= 0 or chunk_poses <= 0:
        raise ValueError("nprocs and chunk_poses must be positive")
    exclude = [str(code).lower() for code in (exclude or [])]

    from convert_poses import convert_arc_dir  # noqa: PLC0415
    from jax_scorer import JaxScoreOracle  # noqa: PLC0415
    from minfor import rotvec_world_to_attr_np  # noqa: PLC0415
    from grid_generator import generate_grid  # noqa: PLC0415
    from nb_kernel.forcefields.nonbon8.params import load_params  # noqa: PLC0415
    from library import config  # noqa: PLC0415
    reducer = _load_reducer()

    reduce_rules = reducer.read_forcefield(_ALARIC / "reduce.dat")
    rec_coordinates, rec_atomtypes = reducer.reduce_ppdb(
        parse_pdb(protein_pdb.read_text(encoding="utf-8")), reduce_rules
    )
    if len(rec_coordinates) == 0:
        raise ValueError(f"{protein_pdb}: reduction produced no ATTRACT beads")
    libraries, _ = config(verify_checksums=False)
    if sequence not in libraries:
        raise ValueError(f"sequence {sequence!r} is not in the fragment library")
    factory = libraries[sequence]
    factory.load_rotaconformers()
    pdb_filter: str | list[str] | None
    if len(exclude) == 1:
        pdb_filter = exclude[0]
    elif exclude:
        pdb_filter = exclude
    else:
        pdb_filter = None
    ligand_library = factory.create(pdb_filter, with_rotaconformers=True)
    ligand_coordinates = ligand_library.reduce().astype(np.float64, copy=False)
    attract_template = np.load(
        _ROOT / "fraglib" / "templates" / "ATTRACT" / f"{sequence}-ppdb.npy",
        allow_pickle=False,
    )
    ligand_atomtypes = np.asarray(attract_template["occupancy"], dtype=np.int32)
    if ligand_coordinates.shape[1] != len(ligand_atomtypes):
        raise ValueError("ligand reduced coordinates and ATTRACT atom types do not match")
    pivot = np.mean(ligand_coordinates[0], axis=0).astype(np.float64, copy=False)
    world_dofs, conformers = convert_arc_dir(
        pose_dir, None, sequence, excluded_pdb_codes=exclude
    )
    if len(world_dofs) != len(conformers):
        raise ValueError("pose conversion returned mismatched arrays")
    atom_dofs = rotvec_world_to_attr_np(world_dofs, pivot)
    parameter_path = _ATTRACT / "attract-original" / "attract-par.npz"
    ff_params = load_params(str(parameter_path))
    ffelec = math.sqrt(332.053986 / epsilon)

    def make_oracle(grid=None, *, grid_free: bool = False):
        return JaxScoreOracle(
            backend="numpy",
            energy_only=True,
            grid_free=grid_free,
            epsilon=epsilon,
            receptor_ensemble=rec_coordinates,
            receptor_atomtypes=rec_atomtypes,
            receptor_charges=np.zeros(len(rec_coordinates), dtype=np.float64),
            ligand_ensemble=ligand_coordinates,
            ligand_atomtypes=ligand_atomtypes,
            ligand_charges=np.zeros(len(ligand_atomtypes), dtype=np.float64),
            grid_object=grid,
            attract_par_npz=str(parameter_path),
            lig_pivot=pivot,
            dof_type="rotvec",
        )

    def score_ids(oracle, ids: np.ndarray) -> np.ndarray:
        output: list[np.ndarray] = []
        for start in range(0, len(ids), chunk_poses):
            batch = ids[start : start + chunk_poses]
            energies, _gradients = oracle.score_batch(
                np.ones(len(batch), dtype=np.int32),
                atom_dofs[batch],
                conformers=conformers[batch],
            )
            output.append(np.asarray(energies, dtype=np.float64))
        if not output:
            return np.empty(0, dtype=np.float64)
        return np.concatenate(output)

    # The reusable grid is a receptor/range product, independent of the RNA
    # fragment.  Use a universal ligand alphabet so the exact key is the
    # contract's (range, plateau distance, neighbour distance) tuple.
    grid_oracles: dict[tuple[str, float, float], Any] = {}

    def grid_score(ids: np.ndarray, step: dict[str, Any]) -> np.ndarray:
        key = (
            str(protein_range),
            float(step["plateau_distance"]),
            float(step["neighbour_distance"]),
        )
        oracle = grid_oracles.get(key)
        if oracle is None:
            grid = generate_grid(
                rec_coordinates,
                rec_atomtypes,
                np.zeros(len(rec_coordinates), dtype=np.float64),
                ff_params,
                "nonbon8",
                ffelec,
                plateaudis=key[1],
                neighbourdis=key[2],
                lig_atomtypes=None,
                return_gradients=False,
            )
            oracle = make_oracle(grid)
            grid_oracles[key] = oracle
        return score_ids(oracle, np.asarray(ids, dtype=np.int64))

    gridless_oracle = make_oracle(grid_free=True)
    def gridless_score(ids: np.ndarray) -> np.ndarray:
        return score_ids(gridless_oracle, np.asarray(ids, dtype=np.int64))

    ids = np.arange(len(atom_dofs), dtype=np.int64)
    kept_ids, stats = cascade_filter(
        ids,
        threshold=threshold,
        cascade=cascade or [],
        min_cascade=min_cascade,
        grid_score=grid_score,
        gridless_score=gridless_score,
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="energy-filter-") as temporary:
        index_path = Path(temporary) / "keep.npy"
        np.save(index_path, kept_ids.astype(np.uint64, copy=False))
        subprocess.run(
            [
                sys.executable,
                str(_ALARIC / "select-poses.py"),
                str(pose_dir),
                str(index_path),
                str(output_dir),
                "--base",
                "0",
                "--force",
                "--compress",
                "--nprocs",
                str(nprocs),
                "--chunk-poses",
                str(chunk_poses),
            ],
            check=True,
        )
    # select-poses publishes request positions as order-array.npy. Here the
    # request is already the ordered list of source pose IDs, so publish the
    # energy-filter lineage using those original IDs for middle-level chaining.
    from npy_io import save_npy  # noqa: PLC0415

    save_npy(
        output_dir / "provenance.npy",
        kept_ids.astype(np.uint64, copy=False),
        compress=True,
    )
    stats.update(
        {
            "schema": 1,
            "epsilon": epsilon,
            "threshold": threshold,
            "fragment": int(fragment),
            "protein_range": str(protein_range),
            "input_count": int(len(ids)),
            "kept_count": int(len(kept_ids)),
            "protein": protein_pdb.name,
        }
    )
    (output_dir / "energy-filter-stats.json").write_text(
        json.dumps(stats, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-dir", required=True, type=Path)
    parser.add_argument("--protein", required=True, type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--fragment", required=True, type=int)
    parser.add_argument("--protein-range", required=True)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--epsilon", required=True, type=float)
    parser.add_argument("--cascade", required=True, help="JSON list of grid steps")
    parser.add_argument("--min-cascade", required=True, type=int)
    parser.add_argument("--pdb-exclude", action="append", default=[])
    parser.add_argument("--nprocs", type=int, default=1)
    parser.add_argument("--chunk-poses", type=int, default=10_000)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cascade = json.loads(args.cascade)
        run_energy_filter(
            args.pose_dir,
            args.protein,
            sequence=args.sequence,
            fragment=args.fragment,
            protein_range=args.protein_range,
            threshold=args.threshold,
            epsilon=args.epsilon,
            cascade=cascade,
            min_cascade=args.min_cascade,
            exclude=args.pdb_exclude,
            output_dir=args.output,
            nprocs=args.nprocs,
            chunk_poses=args.chunk_poses,
        )
    except Exception as exc:
        print(f"energy-filter failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

