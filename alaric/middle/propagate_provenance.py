"""Propagate grow provenance through a representative's local pool lineage.

``alaric-propagate-provenance`` has two modes.

Graph mode takes a JSON file produced by :mod:`alaric.middle.pool_graph`, a
representative pool name, and ``local`` or ``remote``.  It validates the
representative (printing the available representatives when it is absent), then
walks that pool upstream through filter and identity/merge provenance until it
reaches a grow pool.  A merge forks the walk and *every* arm is followed: a
representative that reconnects two grow routes -- ``frag9-reconnect`` joins a
forward route carrying fragment 8 provenance with a backward one carrying
fragment 10 -- yields one propagated file per route.  ``--branch PARENT``
optionally restricts the walk to particular merge arms.  Local result dirs are
``../CACHE/results/<sigil>``; remote dirs are ``$ALARIC_REMOTE_RESULT_DIR/<sigil>``.
Local graph mode runs the second mode immediately.  Remote graph mode writes
``propagate-provenance-<representative>.sh``, one command per arm, and uploads
it beside the remote project deployment scripts.

Mode 2 accepts an ordered ``--pool-dirs`` lineage -- the representative first and
the grow pool last -- plus the ``--source-fragment`` that lineage grows from.
The fragment has to be passed in because result dirs are named by sigil and
record no fragment of their own.  Every non-final directory must provide a filter
``provenance.npy`` (or ``.zst``) or an identity map.  The final directory must
provide grow ``provenance.npy``.  For an identity result, the next directory's
sigil identifies input 1 or input 2 in ``identity-filter.json``, hence chooses
``map-1.npy`` or ``map-2.npy`` without needing logical pool names.

Output lands in the representative's dir, keyed by source fragment: a
filter-only lineage writes dense ``prop-frag<X>-provenance.npy.zst``, a lineage
containing any map writes relational ``prop-frag<X>-map.npy.zst``.  The key is
what makes a reconnected representative usable: ``alaric-chain`` composes one
route per link, and such a representative has one link per source fragment.

The output is a derived convenience array, deliberately excluded from result
sidecar checksums. ``prop-frag<X>-map`` uses ``(grow_source, representative_pose)``
columns, preserving a merge's one-to-many provenance rather than discarding it.
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
# Superseded by the fragment-keyed names below; removed on rewrite so a stale
# copy cannot be mistaken for one of the routes a reconnected pool now carries.
LEGACY_PROP_NAMES = ("prop-provenance.npy", "prop-pair.npy")


def prop_provenance_name(source_fragment: int) -> str:
    """Dense propagated provenance: row = representative pose, value = grow source."""
    return f"prop-frag{int(source_fragment)}-provenance.npy"


def prop_map_name(source_fragment: int) -> str:
    """Relational propagated provenance: columns ``(grow_source, representative_pose)``."""
    return f"prop-frag{int(source_fragment)}-map.npy"


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


def _map_relation(
    pool_dir: Path, next_dir: Path, representative_ids: np.ndarray, ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a map's this->parent relation, expanding representative rows."""
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
    starts = np.searchsorted(sorted_children, ids, side="left")
    stops = np.searchsorted(sorted_children, ids, side="right")
    counts = stops - starts
    total = int(counts.sum())
    offsets = np.repeat(starts, counts)
    within = np.arange(total, dtype=np.int64) - np.repeat(
        np.cumsum(counts) - counts, counts
    )
    rows = order[offsets + within]
    return np.repeat(representative_ids, counts), parents[rows]


def _remove_alternates(path: Path) -> None:
    path.unlink(missing_ok=True)
    path.with_name(path.name + ".zst").unlink(missing_ok=True)


def propagate(pool_dirs: list[str | Path], source_fragment: int) -> Path:
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
    representative_ids = np.arange(nposes, dtype=np.int64)
    ids = representative_ids.copy()
    has_map = False
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
            has_map = True
            representative_ids, ids = _map_relation(
                pool_dir, dirs[index + 1], representative_ids, ids
            )
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
    # Only this fragment's two forms are alternates of each other: another
    # fragment's file describes another route through the same pool and must survive.
    if has_map:
        # The map orientation is (parent, this), so preserve it after replacing
        # the terminal grow-pool pose with its grow-source provenance.
        pair = np.column_stack((output, representative_ids)).astype(np.uint64, copy=False)
        pair = np.unique(pair, axis=0)  # multiple routes can reach one identical edge
        logical_output = dirs[0] / prop_map_name(source_fragment)
        written = save_npy(logical_output, pair, compress=True)
        _remove_alternates(dirs[0] / prop_provenance_name(source_fragment))
    else:
        logical_output = dirs[0] / prop_provenance_name(source_fragment)
        written = save_npy(logical_output, output.astype(np.uint32, copy=False), compress=True)
        _remove_alternates(dirs[0] / prop_map_name(source_fragment))
    logical_output.unlink(missing_ok=True)
    for name in LEGACY_PROP_NAMES:
        _remove_alternates(dirs[0] / name)
    return written


def _representatives(data: dict) -> list[str]:
    return [str(value) for _fragment, value in sorted(data.get("representatives", {}).items(), key=lambda x: int(x[0]))]


def _lineages(data: dict, representative: str, branches: list[str]) -> list[tuple[list[str], int]]:
    """Enumerate every representative-to-grow lineage, keyed by source fragment.

    A merge does not have to be resolved to a single arm: both are walked, and each
    one that reaches a grow pool contributes its own propagated file.  That is what a
    reconnect pool needs -- one arm reaches back to the forward route's fragment, the
    other to the backward route's.  An arm that ends in a source pool (an anchor has
    no provenance of its own) carries no grow provenance and is simply dropped, so a
    representative merging a grown pool with an anchored one still resolves.
    """
    pools = data.get("pools", {})
    lineages: list[tuple[list[str], int]] = []
    used: set[str] = set()
    queue: list[list[str]] = [[representative]]
    while queue:
        path = queue.pop(0)
        current = path[-1]
        node = pools.get(current)
        if not isinstance(node, dict):
            raise PropagationError(f"graph has no pool entry for {current!r}")
        parents = [p for p in node.get("parents", []) if p.get("array")]
        kind = node.get("kind")
        if kind == "grow":
            grow = [p for p in parents if p.get("edge") == "grow"]
            if not grow:
                raise PropagationError(f"{current}: grow pool has no grow provenance edge")
            source = str(grow[0]["pool"])
            fragment = pools.get(source, {}).get("fragment")
            if fragment is None:
                raise PropagationError(
                    f"{current}: graph records no fragment for grow source {source!r}"
                )
            lineages.append((path, int(fragment)))
            continue
        if kind in {"merge", "dedup"}:
            # A dedup lists the same input twice; its two maps are redundant.
            options = list(
                dict.fromkeys(
                    str(p["pool"]) for p in parents if p.get("edge") in {"merge", "dedup"}
                )
            )
            selected = [name for name in options if name in branches]
            if selected:
                used.update(selected)
                options = selected
        else:
            options = list(
                dict.fromkeys(str(p["pool"]) for p in parents if not p.get("cross_fragment"))
            )
            if len(options) > 1:
                raise PropagationError(
                    f"{current}: expected exactly one same-fragment provenance parent, got: "
                    + ", ".join(options)
                )
        for name in options:
            if name in path:
                raise PropagationError(f"provenance lineage contains a cycle at {name!r}")
            queue.append(path + [name])
    unused = set(branches) - used
    if unused:
        raise PropagationError(
            "--branch does not select a merge upstream: " + ", ".join(sorted(unused))
        )
    lineages.sort(key=lambda item: item[1])
    for fragment in {f for _path, f in lineages}:
        clashing = [path for path, f in lineages if f == fragment]
        if len(clashing) > 1:
            raise PropagationError(
                f"{representative}: {len(clashing)} lineages carry fragment {fragment} "
                "provenance and would write the same file; separate them with --branch: "
                + "; ".join(" <- ".join(path) for path in clashing)
            )
    return lineages


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


def _write_remote_script(representative: str, plans: list[tuple[list[Path], int]]) -> Path:
    host = os.environ.get("ALARIC_REMOTE_HOST")
    deploy_root = os.environ.get("ALARIC_REMOTE_DEPLOYMENT_DIR")
    if not host or not deploy_root:
        raise PropagationError(
            "ALARIC_REMOTE_HOST and ALARIC_REMOTE_DEPLOYMENT_DIR are required for remote propagation"
        )
    project = os.environ.get("ALARIC_PROJECT", "PROJECT").strip("/") or "PROJECT"
    script = Path.cwd() / f"propagate-provenance-{representative}.sh"
    # One command per lineage: a reconnected representative propagates each of its
    # grow routes into its own file, and both are needed to chain through it.
    commands = [
        "alaric-propagate-provenance --source-fragment "
        + str(fragment)
        + " --pool-dirs "
        + " ".join(shlex.quote(str(path)) for path in dirs)
        for dirs, fragment in plans
    ]
    script.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(commands) + "\n")
    os.chmod(script, 0o755)
    destination = f"{deploy_root.rstrip('/')}/{project}"
    subprocess.run(["ssh", host, "mkdir", "-p", destination], check=True)
    subprocess.run(["scp", str(script), f"{host}:{destination}/"], check=True)
    return script


def graph_mode(graph_path: str | Path, representative: str | None, deployer: str | None, branches: list[str]) -> list[Path] | None:
    data = json.loads(Path(graph_path).read_text())
    representatives = _representatives(data)
    if representative not in representatives:
        print("Representatives:")
        for pool in representatives:
            print(pool)
        return None
    if deployer not in {"local", "remote"}:
        raise PropagationError("deployer must be local or remote")
    lineages = _lineages(data, representative, branches)
    if not lineages:
        # A representative that only ever *sources* grow actions (the first
        # fragment of a route, or a purely anchored pool) has nothing upstream to
        # propagate.  alaric-chain never asks it for a propagated route either.
        print(f"{representative}: no grow lineage upstream; nothing to propagate")
        return []
    plans = [
        (_result_dirs(data, lineage, deployer), fragment) for lineage, fragment in lineages
    ]
    if deployer == "local":
        return [propagate(dirs, fragment) for dirs, fragment in plans]
    return [_write_remote_script(representative, plans)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alaric-propagate-provenance", description=__doc__)
    parser.add_argument("pool_graph", nargs="?", help="JSON generated by alaric-pool-graph")
    parser.add_argument("representative", nargs="?", help="Representative pool in the graph")
    parser.add_argument("deployer", nargs="?", choices=("local", "remote"))
    parser.add_argument("--pool-dirs", nargs="+", metavar="DIR", help="Resolved representative-to-grow lineage")
    parser.add_argument("--source-fragment", type=int, metavar="N", help="With --pool-dirs: fragment the grow pool grows from")
    parser.add_argument("--branch", action="append", default=[], metavar="PARENT_POOL", help="Graph mode: restrict the walk to this merge parent (repeatable)")
    args = parser.parse_args(argv)
    try:
        if args.pool_dirs:
            if args.pool_graph or args.representative or args.deployer or args.branch:
                parser.error("--pool-dirs cannot be combined with graph-mode arguments or --branch")
            if args.source_fragment is None:
                parser.error("--pool-dirs requires --source-fragment: a result dir is named by "
                             "sigil and records no fragment of its own")
            print(propagate(args.pool_dirs, args.source_fragment))
        else:
            if not args.pool_graph:
                parser.error("provide a pool graph or --pool-dirs")
            if args.source_fragment is not None:
                parser.error("--source-fragment applies to --pool-dirs; graph mode reads it from the graph")
            for path in graph_mode(args.pool_graph, args.representative, args.deployer, args.branch) or []:
                print(path)
    except PropagationError as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
