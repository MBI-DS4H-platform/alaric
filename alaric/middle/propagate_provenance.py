"""Propagate grow provenance through a representative's local pool lineage.

``alaric-propagate-provenance`` has two modes.

Graph mode takes a JSON file produced by :mod:`alaric.middle.pool_graph`, a
representative pool name, and ``local`` or ``remote``.  It validates the
representative (printing the available representatives when it is absent), then
walks that pool upstream through filter and identity/merge provenance until it
reaches a grow pool.  A merge is inherently ambiguous, so ``--branch PARENT``
selects its graph parent.  Local result dirs are ``../CACHE/results/<sigil>``;
remote dirs are ``$ALARIC_REMOTE_RESULT_DIR/<sigil>``.  Local graph mode runs
the second mode immediately.  Remote graph mode writes
``propagate-provenance-<representative>.sh`` and uploads it beside the remote
project deployment scripts.

Mode 2 accepts an ordered ``--pool-dirs`` lineage: the representative first and
the grow pool last.  Every non-final directory must provide a filter
``provenance.npy`` (or ``.zst``) or an identity map.  The final directory must
provide grow ``provenance.npy``.  For an identity result, the next directory's
sigil identifies input 1 or input 2 in ``identity-filter.json``, hence chooses
``map-1.npy`` or ``map-2.npy`` without needing logical pool names.  The composed
grow provenance is written as ``prop-provenance.npy.zst`` in the first pose dir.

The output is a derived convenience array, deliberately excluded from result
sidecar checksums.  It is valid only when every selected representative pose has
one unambiguous origin in the chosen grow provenance; malformed, incomplete, or
one-to-many merge mappings fail rather than silently losing chain provenance.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

import numpy as np

from ..npy_io import find_npy, load_npy, save_npy
from ..poses import PoseReader
from .errors import MiddleError


PROVENANCE = "provenance.npy"
PROP_PROVENANCE = "prop-provenance.npy"


class PropagationError(MiddleError):
    """Invalid provenance propagation request or on-disk lineage."""


def _integer_vector(path: Path, *, label: str) -> np.ndarray:
    found = find_npy(path)
    if found is None:
        raise PropagationError(f"missing {label}: {path} (or its .zst form)")
    value = np.asarray(load_npy(found, mmap=True))
    if value.ndim != 1 or value.dtype.kind not in "iu":
        raise PropagationError(f"{found}: {label} must be a one-dimensional integer array")
    return value.astype(np.int64, copy=False)


def _map_name(pool_dir: Path, next_dir: Path) -> str:
    """Resolve an identity output's map from the following input pool sigil."""
    manifest_path = pool_dir / "identity-filter.json"
    if not manifest_path.is_file():
        raise PropagationError(f"{pool_dir}: merge provenance requires identity-filter.json")
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise PropagationError(f"{manifest_path}: invalid JSON") from exc
    next_sigil = next_dir.name
    matches = [
        index
        for index in (1, 2)
        if Path(str(manifest.get(f"pose_dir_{index}", ""))).name == next_sigil
    ]
    if matches == [1]:
        return "map-1.npy"
    if matches == [2]:
        return "map-2.npy"
    # A dedup has the same input directory twice; its two maps are redundant.
    if matches == [1, 2]:
        return "map-1.npy"
    raise PropagationError(
        f"{pool_dir}: next pool directory {next_sigil!r} is neither input recorded "
        "by identity-filter.json"
    )


def _map_indices(pool_dir: Path, next_dir: Path, ids: np.ndarray) -> np.ndarray:
    name = _map_name(pool_dir, next_dir)
    found = find_npy(pool_dir / name)
    if found is None:
        raise PropagationError(f"{pool_dir}: missing merge provenance {name} (or its .zst form)")
    pairs = np.asarray(load_npy(found, mmap=True))
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in "iu":
        raise PropagationError(f"{found}: {name} must be an integer array with shape (N, 2)")
    parents = pairs[:, 0].astype(np.int64, copy=False)
    children = pairs[:, 1].astype(np.int64, copy=False)
    if np.any(parents < 0) or np.any(children < 0):
        raise PropagationError(f"{found}: {name} contains negative pose indices")
    order = np.argsort(children, kind="stable")
    sorted_children = children[order]
    positions = np.searchsorted(sorted_children, ids)
    valid = positions < len(sorted_children)
    valid[valid] &= sorted_children[positions[valid]] == ids[valid]
    if not np.all(valid):
        missing = int(ids[np.flatnonzero(~valid)[0]])
        raise PropagationError(f"{pool_dir}: {name} has no parent mapping for pose {missing}")
    # A propagated provenance array is dense (one value per output pose).  Do not
    # collapse an identity relation with multiple selected parents invisibly.
    starts = np.searchsorted(sorted_children, ids, side="left")
    stops = np.searchsorted(sorted_children, ids, side="right")
    if np.any(stops - starts != 1):
        duplicate = int(ids[np.flatnonzero(stops - starts != 1)[0]])
        raise PropagationError(
            f"{pool_dir}: {name} maps pose {duplicate} to multiple parents; "
            "cannot write dense propagated provenance"
        )
    return parents[order[starts]]


def propagate(pool_dirs: list[str | Path]) -> Path:
    """Compose a resolved lineage and write its first pool's propagated provenance."""
    dirs = [Path(path) for path in pool_dirs]
    if not dirs:
        raise PropagationError("--pool-dirs requires at least a grow pool")
    for path in dirs:
        if not path.is_dir():
            raise PropagationError(f"pool directory does not exist: {path}")

    try:
        nposes = PoseReader.get_nposes(dirs[0])
    except Exception as exc:
        raise PropagationError(f"{dirs[0]}: cannot determine representative pose count") from exc
    ids = np.arange(nposes, dtype=np.int64)
    for index, pool_dir in enumerate(dirs[:-1]):
        dense = find_npy(pool_dir / PROVENANCE)
        maps = [find_npy(pool_dir / f"map-{n}.npy") for n in (1, 2)]
        maps = [path for path in maps if path is not None]
        if dense is not None and maps:
            raise PropagationError(f"{pool_dir}: contains both filter and merge provenance")
        if dense is not None:
            selector = _integer_vector(pool_dir / PROVENANCE, label="filter provenance")
            if np.any(selector < 0):
                raise PropagationError(f"{pool_dir}: filter provenance contains negative indices")
            if np.any(ids < 0) or np.any(ids >= len(selector)):
                bad = int(ids[np.flatnonzero((ids < 0) | (ids >= len(selector)))[0]])
                raise PropagationError(f"{pool_dir}: filter provenance has no row for pose {bad}")
            ids = selector[ids]
        elif maps:
            ids = _map_indices(pool_dir, dirs[index + 1], ids)
        else:
            raise PropagationError(
                f"{pool_dir}: expected filter provenance.npy or merge map-*.npy"
            )

    grow = _integer_vector(dirs[-1] / PROVENANCE, label="grow provenance")
    if np.any(grow < 0):
        raise PropagationError(f"{dirs[-1]}: grow provenance contains negative indices")
    if np.any(ids < 0) or np.any(ids >= len(grow)):
        bad = int(ids[np.flatnonzero((ids < 0) | (ids >= len(grow)))[0]])
        raise PropagationError(f"{dirs[-1]}: grow provenance has no row for pose {bad}")
    output = grow[ids]
    logical_output = dirs[0] / PROP_PROVENANCE
    written = save_npy(logical_output, output.astype(np.uint32, copy=False), compress=True)
    logical_output.unlink(missing_ok=True)
    return written


def _representatives(data: dict) -> list[str]:
    return [str(value) for _fragment, value in sorted(data.get("representatives", {}).items(), key=lambda x: int(x[0]))]


def _lineage(data: dict, representative: str, branches: list[str]) -> list[str]:
    pools = data.get("pools", {})
    current = representative
    result = [current]
    used: set[str] = set()
    while True:
        node = pools.get(current)
        if not isinstance(node, dict):
            raise PropagationError(f"graph has no pool entry for {current!r}")
        parents = [p for p in node.get("parents", []) if p.get("array")]
        if node.get("kind") == "grow":
            if any(p.get("edge") == "grow" for p in parents):
                unused = set(branches) - used
                if unused:
                    raise PropagationError(
                        "--branch does not select a merge upstream: " + ", ".join(sorted(unused))
                    )
                return result
            raise PropagationError(f"{current}: grow pool has no grow provenance edge")
        if node.get("kind") in {"merge", "dedup"}:
            options = [str(p["pool"]) for p in parents if p.get("edge") in {"merge", "dedup"}]
            unique_options = list(dict.fromkeys(options))
            if node.get("kind") == "dedup" and len(unique_options) == 1:
                current = unique_options[0]
                used.add(current)
            else:
                selected = [name for name in branches if name in unique_options]
                if len(selected) != 1:
                    raise PropagationError(
                        f"{current}: select one upstream with --branch; upstreams are: "
                        + ", ".join(unique_options)
                    )
                current = selected[0]
                used.add(current)
        else:
            options = [str(p["pool"]) for p in parents if not p.get("cross_fragment")]
            if len(options) != 1:
                raise PropagationError(
                    f"{current}: expected exactly one same-fragment provenance parent, got: "
                    + ", ".join(options or ["none"])
                )
            current = options[0]
        if current in result:
            raise PropagationError(f"provenance lineage contains a cycle at {current!r}")
        result.append(current)


def _result_dirs(data: dict, lineage: list[str], deployer: str) -> list[Path]:
    pools = data["pools"]
    if deployer == "local":
        base = Path("../CACHE/results")
    else:
        remote = os.environ.get("ALARIC_REMOTE_RESULT_DIR")
        if not remote:
            raise PropagationError("ALARIC_REMOTE_RESULT_DIR is required for remote propagation")
        base = Path(remote)
    result = []
    for pool in lineage:
        sigil = pools[pool].get("sigil")
        if not sigil:
            raise PropagationError(f"{pool}: graph has no sigil")
        result.append(base / str(sigil))
    return result


def _write_remote_script(representative: str, dirs: list[Path]) -> Path:
    host = os.environ.get("ALARIC_REMOTE_HOST")
    deploy_root = os.environ.get("ALARIC_REMOTE_DEPLOYMENT_DIR")
    if not host or not deploy_root:
        raise PropagationError(
            "ALARIC_REMOTE_HOST and ALARIC_REMOTE_DEPLOYMENT_DIR are required for remote propagation"
        )
    project = os.environ.get("ALARIC_PROJECT", "PROJECT").strip("/") or "PROJECT"
    script = Path.cwd() / f"propagate-provenance-{representative}.sh"
    command = "alaric-propagate-provenance --pool-dirs " + " ".join(
        shlex.quote(str(path)) for path in dirs
    )
    script.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + command + "\n")
    os.chmod(script, 0o755)
    destination = f"{deploy_root.rstrip('/')}/{project}"
    subprocess.run(["ssh", host, "mkdir", "-p", destination], check=True)
    subprocess.run(["scp", str(script), f"{host}:{destination}/"], check=True)
    return script


def graph_mode(graph_path: str | Path, representative: str | None, deployer: str | None, branches: list[str]) -> Path | None:
    data = json.loads(Path(graph_path).read_text())
    representatives = _representatives(data)
    if representative not in representatives:
        print("Representatives:")
        for pool in representatives:
            print(pool)
        return None
    if deployer not in {"local", "remote"}:
        raise PropagationError("deployer must be local or remote")
    lineage = _lineage(data, representative, branches)
    dirs = _result_dirs(data, lineage, deployer)
    if deployer == "local":
        return propagate(dirs)
    return _write_remote_script(representative, dirs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alaric-propagate-provenance", description=__doc__)
    parser.add_argument("pool_graph", nargs="?", help="JSON generated by alaric-pool-graph")
    parser.add_argument("representative", nargs="?", help="Representative pool in the graph")
    parser.add_argument("deployer", nargs="?", choices=("local", "remote"))
    parser.add_argument("--pool-dirs", nargs="+", metavar="DIR", help="Resolved representative-to-grow lineage")
    parser.add_argument("--branch", action="append", default=[], metavar="PARENT_POOL", help="Graph mode: choose this merge parent (repeatable)")
    args = parser.parse_args(argv)
    try:
        if args.pool_dirs:
            if args.pool_graph or args.representative or args.deployer or args.branch:
                parser.error("--pool-dirs cannot be combined with graph-mode arguments or --branch")
            print(propagate(args.pool_dirs))
        else:
            if not args.pool_graph:
                parser.error("provide a pool graph or --pool-dirs")
            result = graph_mode(args.pool_graph, args.representative, args.deployer, args.branch)
            if result is not None:
                print(result)
    except PropagationError as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
