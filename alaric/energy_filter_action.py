#!/usr/bin/env python3
"""Drive compiled minfor filter passes and publish a filtered pose pool."""
from __future__ import annotations

import argparse
import base64
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

from energy_filter import _validate_minimum, _validate_steps  # noqa: E402
from nprocs import default_nprocs  # noqa: E402


def _prepare_minfor_inputs(pose_dir, protein_pdb, sequence, exclude, work, *, pose_range=None, pose_indices=None):
    """Prepare the same receptor, library and pose arrays consumed by score.sh.

    ``pose_range`` (1-based, inclusive) or ``pose_indices`` (an NPY file of 0-based global
    ids) restricts the converted poses; the minfor arrays are then indexed locally.
    """
    commands = [
        [sys.executable, str(_ALARIC / "parse_pdb.py"), str(protein_pdb), str(work / "receptor-ppdb.npy")],
        [sys.executable, str(_ROOT / "util" / "reduce-npy.py"), str(work / "receptor-ppdb.npy"), str(work / "receptor")],
        [sys.executable, str(_ROOT / "util" / "get-reduced-library.py"), sequence, str(work / "ligand-ensemble.npy"),
         *[argument for code in exclude for argument in ("-x", code)]],
        [sys.executable, str(_ROOT / "util" / "ligand-atomtypes.py"), "--sequence", sequence, str(work / "ligand-atomtypes.npy")],
        [sys.executable, str(_ALARIC / "convert_poses.py"), "--pose-dir", str(pose_dir), "--sequence", sequence,
         "--output-prefix", str(work / "poses")],
    ]
    commands[-1].extend(argument for code in exclude for argument in ("--pdb-exclude", code))
    if pose_range is not None:
        commands[-1].extend(["--pose-range", str(pose_range[0]), str(pose_range[1])])
    if pose_indices is not None:
        commands[-1].extend(["--pose-indices", str(pose_indices)])
    for command in commands:
        subprocess.run(command, check=True)


def _cascade_minfor(nposes, steps, minimum, invoke):
    """Apply the grid-eligible minimum between minfor subprocesses.

    None represents the entire original pose array, avoiding a full-size index
    array before the first discriminative filter has reduced the candidate set.
    """
    pending, pending_count = None, nposes
    kept, precision = [], []
    stats = {"grid_counts": [], "gridless_count": 0, "precision_fallback_counts": [],
             "precision_fallback_count": 0, "cascade_exit": None}
    for index, step in enumerate(steps):
        if not pending_count or pending_count < minimum:
            stats["cascade_exit"] = {"before_step": index, "undecided": pending_count, "min_cascade": minimum}
            break
        stats["grid_counts"].append(int(pending_count))
        white, grey, unsafe = invoke(pending, step)
        kept.append(white)
        exact_ids = grey[unsafe]
        precision.append(exact_ids)
        stats["precision_fallback_counts"].append(int(len(exact_ids)))
        stats["precision_fallback_count"] += int(len(exact_ids))
        pending = grey[~unsafe]
        pending_count = len(pending)
        if pending_count < minimum:
            stats["cascade_exit"] = {"after_step": index, "undecided": pending_count, "min_cascade": minimum}
            break
    if pending is None:
        exact_ids, exact_count = None, nposes
    else:
        exact_ids = np.sort(np.concatenate([pending, *precision]))
        exact_count = len(exact_ids)
    stats["gridless_count"] = int(exact_count)
    if exact_count:
        white, grey, _ = invoke(exact_ids, None)
        if len(grey):
            raise ValueError("zero-tolerance gridless filtering returned grey poses")
        kept.append(white)
    return (np.sort(np.concatenate(kept)) if kept else np.empty(0, dtype=np.uint64)), stats


def _validate_settings(threshold, epsilon, cascade, min_cascade, exclude, nprocs, chunk_poses):
    """Validate the action parameters shared by whole-action and chunked runs."""
    if epsilon != 15.0:
        raise ValueError("energy-filter requires epsilon: 15")
    if nprocs <= 0 or chunk_poses <= 0:
        raise ValueError("nprocs and chunk_poses must be positive")
    threshold = float(threshold)
    if not math.isfinite(threshold):
        raise ValueError("energy threshold must be finite")
    steps = _validate_steps(cascade or [])
    minimum = _validate_minimum(min_cascade)
    exclude = [str(code).lower() for code in (exclude or [])]
    return threshold, steps, minimum, exclude


class _MinforInvoker:
    """Run one minfor filter pass over prepared arrays in ``work``.

    ``indices`` are 0-based into the prepared arrays (None means all of them). Grids are
    reused within this invoker by (protein range, plateau distance, neighbour distance).
    """

    def __init__(self, work: Path, nposes: int, *, protein_range: str, epsilon: float,
                 threshold: float, chunk_poses: int, prefix: str = "filter"):
        self.work, self.nposes = work, nposes
        self.protein_range, self.epsilon, self.threshold = protein_range, epsilon, threshold
        self.chunk_poses, self.prefix = chunk_poses, prefix
        self.grid_cache: dict[tuple, Path] = {}
        self.count = 0

    def __call__(self, indices, step):
        work = self.work
        self.count += 1
        prefix = work / f"{self.prefix}-{self.count}"
        white_path = prefix.with_suffix(".white.npy")
        grey_path = prefix.with_suffix(".grey.npy")
        report_path = prefix.with_suffix(".json")
        command = [
            sys.executable, str(_ATTRACT_UTIL / "minfor.py"),
            "--filter", "--oracle", "jax", "--nb-kernel", "compiled",
            "--score-mode", "bulk", "--no-conformer-grouping",
            "--input-rotvec", str(work / "poses.rotvec.npy"),
            "--input-conformers", str(work / "poses.conformers.npy"),
            "--input-world-centered",
            "--receptor-coordinates", str(work / "receptor-coor.npy"),
            "--receptor-atomtypes", str(work / "receptor-atomtypes.npy"),
            "--ligand-ensemble", str(work / "ligand-ensemble.npy"),
            "--ligand-atomtypes", str(work / "ligand-atomtypes.npy"),
            "--attract-par-npz", str(_ATTRACT / "attract-original" / "attract-par.npz"),
            "--epsilon", str(self.epsilon), "--energy-threshold", str(self.threshold),
            "--energy-tolerance", str(0.0 if step is None else step["max_error"]),
            "--output-white", str(white_path), "--output-grey", str(grey_path),
            "--filter-report", str(report_path), "--score-batch-size", str(self.chunk_poses),
        ]
        if indices is not None:
            indices_path = prefix.with_suffix(".input.npy")
            np.save(indices_path, indices, allow_pickle=False)
            command.extend(["--pose-indices", str(indices_path)])
        if step is None:
            command.append("--no-grid")
            label = "gridless"
        else:
            key = (str(self.protein_range), float(step["plateau_distance"]), float(step["neighbour_distance"]))
            grid_path = self.grid_cache.setdefault(key, work / f"grid-{len(self.grid_cache)}.npz")
            if grid_path.exists():
                command.extend(["--grid", str(grid_path)])
            else:
                command.extend(["--plateaudis", str(key[1]), "--neighbourdis", str(key[2]),
                                "--save-grid", str(grid_path)])
            command.append("--filter-precision-safe")
            label = f"grid {key[1]:g}/{key[2]:g}"
        count = self.nposes if indices is None else len(indices)
        print(f"Energy filter: {label}, {count} undecided poses", file=sys.stderr, flush=True)
        # minfor reports score/filter progress and errors on stderr, as score.sh does.
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        white = np.load(white_path, allow_pickle=False)
        grey = np.load(grey_path, allow_pickle=False)
        for values in (white, grey):
            if values.ndim != 1 or values.dtype != np.dtype("uint64") or np.any(values >= self.nposes):
                raise ValueError("minfor returned invalid pose indices")
        report = json.loads(report_path.read_text())
        if step is None:
            precision = np.zeros(len(grey), dtype=bool)
        else:
            packed = base64.b64decode(report["precision_fallback_bits"], validate=True)
            if len(packed) != (len(grey) + 7) // 8:
                raise ValueError("minfor returned invalid precision routing metadata")
            precision = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), count=len(grey)).astype(bool)
            if np.count_nonzero(precision) != report["precision_fallback_count"]:
                raise ValueError("minfor precision routing count does not match its metadata")
        return white, grey, precision


def _publish(pose_dir, protein_pdb, output_dir, kept_ids, stats, work, *, epsilon, threshold,
             fragment, protein_range, nposes, invocations, nprocs, chunk_poses):
    """Select the kept poses into ``output_dir`` and write provenance and statistics."""
    index_path = work / "keep.npy"
    np.save(index_path, kept_ids, allow_pickle=False)
    subprocess.run([
        sys.executable, str(_ALARIC / "select-poses.py"), str(pose_dir), str(index_path),
        str(output_dir), "--base", "0", "--force", "--compress",
        "--nprocs", str(nprocs), "--chunk-poses", str(chunk_poses),
    ], check=True)
    from npy_io import save_npy

    save_npy(output_dir / "provenance.npy", kept_ids, compress=True)
    stats.update({
        "schema": 1, "epsilon": epsilon, "threshold": threshold,
        "fragment": int(fragment), "protein_range": str(protein_range),
        "input_count": int(nposes), "kept_count": int(len(kept_ids)),
        "protein": protein_pdb.name, "backend": "compiled", "frontend": "minfor",
        "minfor_invocations": invocations,
    })
    (output_dir / "energy-filter-stats.json").write_text(
        json.dumps(stats, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return stats


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
    nprocs: int | None = None,
    chunk_poses: int = 10_000,
) -> dict[str, Any]:
    """Run compiled minfor filters and retain passing poses in original order."""
    if nprocs is None:
        nprocs = default_nprocs()
    pose_dir, protein_pdb, output_dir = Path(pose_dir), Path(protein_pdb), Path(output_dir)
    threshold, steps, minimum, exclude = _validate_settings(
        threshold, epsilon, cascade, min_cascade, exclude, nprocs, chunk_poses
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="energy-filter-") as temporary:
        work = Path(temporary)
        _prepare_minfor_inputs(pose_dir, protein_pdb, sequence, exclude, work)
        poses = np.load(work / "poses.rotvec.npy", mmap_mode="r", allow_pickle=False)
        nposes = len(poses)
        del poses
        invoke = _MinforInvoker(work, nposes, protein_range=protein_range, epsilon=epsilon,
                                threshold=threshold, chunk_poses=chunk_poses)
        kept_ids, stats = _cascade_minfor(nposes, steps, minimum, invoke)
        return _publish(pose_dir, protein_pdb, output_dir, kept_ids, stats, work,
                        epsilon=epsilon, threshold=threshold, fragment=fragment,
                        protein_range=protein_range, nposes=nposes, invocations=invoke.count,
                        nprocs=nprocs, chunk_poses=chunk_poses)


def add_filter_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The action parameters shared by the whole-action and chunked command lines."""
    parser.add_argument("--pose-dir", required=True, type=Path)
    parser.add_argument("--protein", required=True, type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--fragment", required=True, type=int)
    parser.add_argument("--protein-range", required=True)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--epsilon", required=True, type=float)
    parser.add_argument("--cascade", required=True, help="JSON list of grid steps")
    parser.add_argument("--min-cascade", required=True, type=int)
    parser.add_argument("--pdb-exclude", action="extend", nargs="+", default=[])
    # Non-load-bearing (selection workers only); unspecified follows default_nprocs().
    parser.add_argument("--nprocs", type=int, default=default_nprocs())
    parser.add_argument("--chunk-poses", type=int, default=10_000)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = add_filter_arguments(argparse.ArgumentParser(description=__doc__))
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

