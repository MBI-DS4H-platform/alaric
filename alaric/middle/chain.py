"""Build chains from the pose-pool graph JSON produced by ``alaric-pool-graph``.

A *chain* is one pose per representative fragment, consistent with the grow
provenance recorded in the pool graph. This module operates on the pools listed
under ``representatives`` in the JSON, in fragment order.

Counting mode (``--count``):
  1. Connect consecutive representatives by composing their provenance arrays
     into per-pose edges (a chain builder's bipartite layers).
  2. Keep only poses that lie on a complete chain spanning every selected
     representative -- i.e. poses that connect to the next fragment, propagated
     both up and down the chain (forward + backward reachability).
  3. Report, per pool, ``kept`` (surviving poses) and ``unique`` (physically
     distinct surviving poses: conformer + rotamer + translation), plus the
     total number of chains.

``--select`` / ``--exclude`` restrict the representatives used; every named pool
must be a representative (otherwise the error lists the representatives).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ALARIC_DIR = Path(__file__).resolve().parents[1]
if str(_ALARIC_DIR) not in sys.path:
    sys.path.insert(0, str(_ALARIC_DIR))

from poses import PoseReader, discover_organized, select_pose_indices  # noqa: E402

from .errors import MiddleError  # noqa: E402


class ChainError(MiddleError):
    """Invalid chain-building request or data."""


# -- variable-length relational primitives --------------------------------


def _gather_ranges(values: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    """For each row i, expand into ``values[lo[i]:hi[i]]``.

    Returns ``(repeat_counts, gathered)`` where ``gathered`` concatenates the
    per-row slices and ``repeat_counts[i] = hi[i]-lo[i]`` (used to expand the
    paired column).
    """
    counts = (hi - lo).astype(np.int64)
    total = int(counts.sum())
    if total == 0:
        return counts, np.empty(0, dtype=values.dtype)
    starts = np.repeat(lo.astype(np.int64), counts)
    within = np.arange(total, dtype=np.int64) - np.repeat(
        np.cumsum(counts) - counts, counts
    )
    return counts, values[starts + within]


def _join(left_keys, left_vals, right_keys, right_vals):
    """Inner join two (key, value) relations; return cross-product values.

    Returns ``(left_out, right_out)`` such that for every pair with
    ``left_keys[i] == right_keys[j]`` there is one output row
    ``(left_vals[i], right_vals[j])``.
    """
    left_keys = np.asarray(left_keys, dtype=np.int64)
    right_keys = np.asarray(right_keys, dtype=np.int64)
    if left_keys.size == 0 or right_keys.size == 0:
        return (
            np.empty(0, dtype=np.asarray(left_vals).dtype),
            np.empty(0, dtype=np.asarray(right_vals).dtype),
        )
    order = np.argsort(left_keys, kind="stable")
    sk = left_keys[order]
    sv = np.asarray(left_vals)[order]
    lo = np.searchsorted(sk, right_keys, side="left")
    hi = np.searchsorted(sk, right_keys, side="right")
    counts, left_out = _gather_ranges(sv, lo, hi)
    right_out = np.repeat(np.asarray(right_vals), counts)
    return left_out, right_out


# -- provenance composition -----------------------------------------------


@dataclass
class Edges:
    """Bipartite edges between a lower-fragment pool and a higher-fragment pool."""

    low: np.ndarray  # pose ids in the lower-fragment representative
    high: np.ndarray  # pose ids in the higher-fragment representative


class ChainGraph:
    def __init__(self, data: dict, project_root: Path):
        self.data = data
        self.project_root = project_root
        self.pools: dict[str, dict] = data["pools"]
        self.representatives: dict[int, str] = {
            int(k): v for k, v in data["representatives"].items()
        }
        self.links: list[dict] = data["links"]
        self._nposes: dict[str, int] = {}

    # materialization / provenance helpers ---------------------------------

    def _pool_dir(self, pool: str) -> Path:
        rel = self.pools[pool].get("result_dir")
        if not rel:
            raise ChainError(f"pool {pool!r} has no result_dir (run alaric-sigil first)")
        return self.project_root / rel

    def _result_dir(self, pool: str) -> Path:
        path = self._pool_dir(pool)
        if not path.is_dir():
            raise ChainError(f"pool {pool!r} result not materialized at {path}")
        return path

    def is_materialized(self, pool: str) -> bool:
        """A pool is materialized when its concrete poses are present on disk."""
        path = self._pool_dir(pool)
        return path.is_dir() and bool(discover_organized(path))

    def _filter_provenance_arrays(self, pool: str) -> list[str]:
        """Names of this pool's same-fragment (filter/merge/dedup) provenance arrays."""
        return [
            parent["array"]
            for parent in self.pools[pool].get("parents", [])
            if parent.get("array") and not parent.get("cross_fragment")
        ]

    def has_filter_provenance(self, pool: str) -> bool:
        path = self._pool_dir(pool)
        if not path.is_dir():
            return False
        return any((path / name).is_file() for name in self._filter_provenance_arrays(pool))

    def _load_step_array(self, step: dict) -> np.ndarray:
        pool = step["pool"]
        name = step["array"]
        path = self._pool_dir(pool) / name
        if path.is_file():
            return np.load(path, mmap_mode="r")
        # The array is unavailable. Follow the user's rule: a pool is walkable as
        # long as it is materialized or still carries filter provenance; if it is
        # neither, the chain provenance cannot be composed -> give up.
        frag = self.pools[pool].get("fragment")
        if not self.is_materialized(pool) and not self.has_filter_provenance(pool):
            raise ChainError(
                f"frag{frag} pool {pool!r}: neither materialized nor has filter "
                f"provenance; cannot compose chain provenance"
            )
        raise ChainError(
            f"pool {pool!r} (frag{frag}): missing provenance array {name!r} at {path}"
        )

    def nposes(self, pool: str) -> int:
        if pool not in self._nposes:
            path = self._pool_dir(pool)
            if not (path.is_dir() and discover_organized(path)):
                raise ChainError(
                    f"representative {pool!r} (frag{self.pools[pool].get('fragment')}) "
                    f"is not materialized"
                )
            self._nposes[pool] = PoseReader.get_nposes(path)
        return self._nposes[pool]

    # path composition -----------------------------------------------------

    def _compose(self, rep: str, path: list[dict]):
        """Map every pose of ``rep`` up to the join pool's pose-id space.

        Walks the provenance route: same-fragment **filter** provenance steps,
        ending in the single cross-fragment **grow** step. Intermediate pools need
        only their provenance arrays present, not their poses. ``provenance.npy`` is
        a dense function (no row growth); a ``map-*.npy`` step may fan out (dedup).

        Returns ``(rep_ids, join_ids)`` where ``rep_ids[i]`` connects to
        ``join_ids[i]``.
        """
        n = self.nposes(rep)
        rep_ids = np.arange(n, dtype=np.int64)
        cur = rep_ids.copy()
        for step in path:
            arr = self._load_step_array(step)
            if step["orientation"].startswith("row="):
                # dense: arr[this] = parent
                prov = np.asarray(arr, dtype=np.int64)
                cur = prov[cur]
            else:
                # pairs: cols (parent, this); map this -> parent (1:many)
                pairs = np.asarray(arr, dtype=np.int64)
                parents, reps = _join(pairs[:, 1], pairs[:, 0], cur, rep_ids)
                cur, rep_ids = parents, reps
        return rep_ids, cur

    def edges_for_link(self, link: dict) -> Edges:
        join = link["join_pool"]
        target_ids, ts = self._compose(link["target_rep"], link["target_to_join"])
        source_ids, rs = self._compose(link["source_rep"], link["source_to_join"])
        # connect where both map to the same join pose
        s_out, t_out = _join(rs, source_ids, ts, target_ids)
        # Edges are kept raw (not deduplicated): the grow provenance is naturally
        # a tree (each grown pose has one origin), so chains branch like a tree.
        # A merge/dedup is the unavoidable exception that gives a pose >1 parent.
        if link["source_fragment"] < link["target_fragment"]:
            return Edges(low=s_out, high=t_out)
        return Edges(low=t_out, high=s_out)


# -- counting -------------------------------------------------------------


@dataclass
class CountResult:
    order: list[str]  # pool names in fragment order
    fragments: list[int]
    kept: list[int]
    unique: list[int]
    total_chains: int


def _select_link(graph: ChainGraph, lower: str, upper: str) -> dict:
    for link in graph.links:
        if {link["source_rep"], link["target_rep"]} == {lower, upper}:
            return link
    raise ChainError(
        f"no link connects {lower!r} and {upper!r}; they are not adjacent in the "
        f"chain (did you exclude an intermediate representative?)"
    )


def _count_unique(graph: ChainGraph, pool: str, kept_idx: np.ndarray) -> int:
    if kept_idx.size == 0:
        return 0
    chunk = select_pose_indices(graph._result_dir(pool), kept_idx)
    rows = np.column_stack(
        (
            chunk.conformers.astype(np.int64),
            chunk.rotamers.astype(np.int64),
            chunk.translations_grid.astype(np.int64),
        )
    )
    return int(len(np.unique(rows, axis=0)))


def count_chains(graph: ChainGraph, selected: list[str]) -> CountResult:
    order = sorted(selected, key=lambda p: graph.pools[p]["fragment"])
    fragments = [graph.pools[p]["fragment"] for p in order]
    nposes = [graph.nposes(p) for p in order]

    if len(order) == 1:
        # a single representative: every pose is a trivial chain, no pruning
        kept_idx = np.arange(nposes[0], dtype=np.int64)
        return CountResult(
            order=order,
            fragments=fragments,
            kept=[nposes[0]],
            unique=[_count_unique(graph, order[0], kept_idx)],
            total_chains=nposes[0],
        )

    edges: list[Edges] = []
    for lower, upper in zip(order, order[1:]):
        edges.append(graph.edges_for_link(_select_link(graph, lower, upper)))

    k = len(order)
    # forward reachability (layer 0 seeds everything)
    fwd = [np.zeros(n, dtype=bool) for n in nposes]
    fwd[0][:] = True
    for j in range(k - 1):
        alive = fwd[j][edges[j].low]
        fwd[j + 1][edges[j].high[alive]] = True
    # backward reachability (last layer seeds everything)
    bwd = [np.zeros(n, dtype=bool) for n in nposes]
    bwd[-1][:] = True
    for j in range(k - 2, -1, -1):
        alive = bwd[j + 1][edges[j].high]
        bwd[j][edges[j].low[alive]] = True

    kept_mask = [f & b for f, b in zip(fwd, bwd)]

    # count chains: paths over kept poses
    ways = np.where(kept_mask[0], 1, 0).astype(np.int64)
    for j in range(k - 1):
        valid = kept_mask[j][edges[j].low] & kept_mask[j + 1][edges[j].high]
        lo = edges[j].low[valid]
        hi = edges[j].high[valid]
        nxt = np.zeros(nposes[j + 1], dtype=np.int64)
        np.add.at(nxt, hi, ways[lo])
        ways = nxt
    total_chains = int(ways.sum())

    kept = [int(m.sum()) for m in kept_mask]
    unique = [
        _count_unique(graph, pool, np.flatnonzero(mask).astype(np.int64))
        for pool, mask in zip(order, kept_mask)
    ]
    return CountResult(
        order=order,
        fragments=fragments,
        kept=kept,
        unique=unique,
        total_chains=total_chains,
    )


# -- selection ------------------------------------------------------------


def resolve_selection(
    graph: ChainGraph,
    select: list[str] | None,
    exclude: list[str] | None,
) -> list[str]:
    reps = set(graph.representatives.values())

    def check(names: list[str], flag: str) -> None:
        unknown = [n for n in names if n not in reps]
        if unknown:
            listing = ", ".join(sorted(reps))
            raise ChainError(
                f"{flag}: {', '.join(unknown)} not in representatives. "
                f"Representatives are: {listing}"
            )

    chosen = set(reps)
    if select:
        check(select, "--select")
        chosen = set(select)
    if exclude:
        check(exclude, "--exclude")
        chosen -= set(exclude)
    if not chosen:
        raise ChainError("no representatives selected")
    return sorted(chosen, key=lambda p: graph.pools[p]["fragment"])


def _format_count(result: CountResult) -> str:
    width = max((len(p) for p in result.order), default=4)
    lines = [f"{'frag':>4}  {'pool':<{width}}  {'kept':>14}  {'unique':>14}"]
    for frag, pool, kept, unique in zip(
        result.fragments, result.order, result.kept, result.unique
    ):
        lines.append(f"{frag:>4}  {pool:<{width}}  {kept:>14}  {unique:>14}")
    lines.append(f"total chains: {result.total_chains}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="alaric-chain",
        description="Build chains from a pose-pool graph JSON.",
    )
    parser.add_argument("graph_json", help="JSON produced by alaric-pool-graph")
    parser.add_argument("--count", action="store_true", help="Counting mode.")
    parser.add_argument("--select", nargs="+", metavar="POOL", help="Use only these representatives.")
    parser.add_argument("--exclude", nargs="+", metavar="POOL", help="Drop these representatives.")
    parser.add_argument("--project-root", help="Override the project root for result dirs.")
    args = parser.parse_args(argv)

    if not args.count:
        parser.error("only --count is implemented")

    data = json.loads(Path(args.graph_json).read_text())
    project_root = Path(args.project_root) if args.project_root else Path(data["project"])
    graph = ChainGraph(data, project_root)

    try:
        selected = resolve_selection(graph, args.select, args.exclude)
        result = count_chains(graph, selected)
    except ChainError as exc:
        parser.exit(2, f"error: {exc}\n")

    print(_format_count(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
