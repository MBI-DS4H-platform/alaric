#!/usr/bin/env python3
"""Grid-free, identity-pose energetics for staged Seamless execution.

The adjacent ``energetics_seamless_support.py.DEPS.txt`` declares source-side
dependencies. Workers resolve the corresponding files from either the
complete source checkout or Seamless's flat staged sandbox, where the
manifest itself is not staged.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


_SCRIPT = Path(__file__).resolve()
_DEPS_FILE = _SCRIPT.with_name(_SCRIPT.name + ".DEPS.txt")
_EPSILON = 15.0


def input_paths() -> list[str]:
    """Return the exact byte dependencies listed beside this script."""
    paths = []
    for line in _DEPS_FILE.read_text(encoding="utf-8").splitlines():
        if not line:
            raise ValueError(f"blank dependency entry in {_DEPS_FILE}")
        paths.append(str((_DEPS_FILE.parent / line).resolve()))
    return paths


def _dependency_map() -> dict[str, Path]:
    """Resolve the eight declared files without loading the worker manifest."""
    util_dir = _SCRIPT.parent
    flat_files = {
        _SCRIPT.name: Path(_SCRIPT),
        "parse_pdb.py": util_dir.parent / "alaric" / "parse_pdb.py",
        "reduce-npy.py": util_dir / "reduce-npy.py",
        "reduce.dat": util_dir.parent / "alaric" / "reduce.dat",
        "jax_scorer.py": util_dir.parent / "attract-jax" / "util" / "jax_scorer.py",
        "numpy_grid_score.py": util_dir.parent / "attract-jax" / "util" / "numpy_grid_score.py",
        "pdb_io.py": util_dir.parent / "attract-jax" / "util" / "pdb_io.py",
        "attract-par.npz": util_dir.parent / "attract-jax" / "attract-original" / "attract-par.npz",
    }
    result: dict[str, Path] = {}
    for name, source_path in flat_files.items():
        staged_path = util_dir / name
        path = staged_path if staged_path.is_file() else source_path
        if not path.is_file():
            raise FileNotFoundError(f"missing staged A5 dependency: {name}")
        if name in result:
            raise ValueError(f"duplicate dependency basename: {name}")
        result[name] = path.resolve()
    return result


def _load_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load staged dependency {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_scoring_modules(dependencies: dict[str, Path]) -> tuple[Any, Any, Any]:
    # jax_scorer imports pdb_io by its flat module name and loads
    # numpy_grid_score lazily for backend="numpy".  Adding only declared
    # dependency directories supports both the source tree and Seamless's flat
    # staging layout.
    for parent in reversed([path.parent for path in dependencies.values()]):
        entry = str(parent)
        if entry not in sys.path:
            sys.path.insert(0, entry)
    parser = _load_module(dependencies["parse_pdb.py"], "a5_parse_pdb")
    reducer = _load_module(dependencies["reduce-npy.py"], "a5_reduce_npy")
    scorer = _load_module(dependencies["jax_scorer.py"], "a5_jax_scorer")
    return parser, reducer, scorer


def _reduce(path: Path, parser: Any, reducer: Any, forcefield: dict) -> tuple[np.ndarray, np.ndarray]:
    """Parse and reduce one input PDB exactly once."""
    parsed = parser.parse_pdb(path.read_text(encoding="utf-8"))
    coordinates, atomtypes = reducer.reduce_ppdb(parsed, forcefield)
    if len(coordinates) == 0:
        raise ValueError(f"{path.name}: reduction produced no ATTRACT beads")
    return coordinates, atomtypes


def _identity_energy(
    receptor_coordinates: np.ndarray,
    receptor_atomtypes: np.ndarray,
    ligand_coordinates: np.ndarray,
    ligand_atomtypes: np.ndarray,
    parameter_file: Path,
    scorer_module: Any,
) -> float:
    oracle = scorer_module.JaxScoreOracle(
        backend="numpy",
        energy_only=True,
        grid_free=True,
        epsilon=_EPSILON,
        receptor_ensemble=receptor_coordinates,
        receptor_atomtypes=receptor_atomtypes,
        ligand_ensemble=ligand_coordinates,
        ligand_atomtypes=ligand_atomtypes,
        attract_par_npz=str(parameter_file),
        lig_pivot=np.zeros(3, dtype=np.float64),
        dof_type="euler",
    )
    energies, _gradients = oracle.score_batch(
        np.array([1], dtype=np.int32), np.zeros((1, 6), dtype=np.float64)
    )
    energy = float(energies[0])
    if not np.isfinite(energy):
        raise ValueError("grid-free scorer returned a non-finite energy")
    return energy


def score(ligand_directory: str | Path, receptor_pdb: str | Path) -> dict[str, Any]:
    dependencies = _dependency_map()
    parser, reducer, scorer_module = _load_scoring_modules(dependencies)
    forcefield = reducer.read_forcefield(dependencies["reduce.dat"])

    receptor_path = Path(receptor_pdb)
    # Receptor reduction is invocation-wide: any failure aborts before the
    # result file is written.
    receptor_coordinates, receptor_atomtypes = _reduce(
        receptor_path, parser, reducer, forcefield
    )

    ligand_paths = sorted(
        (path for path in Path(ligand_directory).iterdir()
         if path.is_file() and path.suffix.lower() == ".pdb"),
        key=lambda path: path.name,
    )
    result: dict[str, Any] = {
        "schema": 1,
        "epsilon": _EPSILON,
        "receptor": {"beads": int(len(receptor_coordinates))},
        "ligands": {},
        "pairs": [],
    }

    for ligand_path in ligand_paths:
        row: dict[str, Any] = {"ligand": ligand_path.name, "energy": None, "error": None}
        try:
            ligand_coordinates, ligand_atomtypes = _reduce(
                ligand_path, parser, reducer, forcefield
            )
            result["ligands"][ligand_path.name] = {
                "beads": int(len(ligand_coordinates))
            }
            row["energy"] = _identity_energy(
                receptor_coordinates,
                receptor_atomtypes,
                ligand_coordinates,
                ligand_atomtypes,
                dependencies["attract-par.npz"],
                scorer_module,
            )
        except Exception as exc:
            row["energy"] = None
            row["error"] = f"{type(exc).__name__}: {exc}"
        result["pairs"].append(row)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    score_parser = subparsers.add_parser("score", help="score a folder of PDB ligands")
    score_parser.add_argument("ligands", help="folder containing ligand PDB files")
    score_parser.add_argument("receptor", help="one receptor PDB")
    args = parser.parse_args(argv)

    output_path = Path.cwd() / "energies.json"
    # Remove an old result first so a failed receptor reduction cannot leave a
    # stale success artifact that appears to belong to this invocation.
    output_path.unlink(missing_ok=True)
    try:
        result = score(args.ligands, args.receptor)
    except Exception as exc:
        print(f"energetics failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
