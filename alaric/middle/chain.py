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

``--target`` / ``--exclude`` restrict the representatives used; every named pool
must be a representative (otherwise the error lists the representatives).
(Choosing *which* pool represents a fragment is not done here -- that is
``alaric-pool-graph --select``, and it is already baked into the graph JSON.)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ALARIC_DIR = Path(__file__).resolve().parents[1]
if str(_ALARIC_DIR) not in sys.path:
    sys.path.insert(0, str(_ALARIC_DIR))

from poses import (  # noqa: E402
    DEFAULT_BUCKET_SIZE,
    PoseReader,
    discover_organized,
    pack_pool,
    select_pose_indices,
    write_arc_file,
)

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


# -- connectivity (shared by count and build) -----------------------------


@dataclass
class Layers:
    order: list[str]  # representative pool per fragment, in fragment order
    fragments: list[int]
    nposes: list[int]
    edges: list[Edges]  # edges[j] connects layer j and j+1 (rep-index space)
    kept_mask: list[np.ndarray]  # poses on at least one complete chain


def _select_link(graph: ChainGraph, lower: str, upper: str) -> dict:
    for link in graph.links:
        if {link["source_rep"], link["target_rep"]} == {lower, upper}:
            return link
    raise ChainError(
        f"no link connects {lower!r} and {upper!r}; they are not adjacent in the "
        f"chain (did you exclude an intermediate representative?)"
    )


def _connectivity(graph: ChainGraph, selected: list[str]) -> Layers:
    order = sorted(selected, key=lambda p: graph.pools[p]["fragment"])
    fragments = [graph.pools[p]["fragment"] for p in order]
    nposes = [graph.nposes(p) for p in order]

    if len(order) == 1:
        return Layers(order, fragments, nposes, [], [np.ones(nposes[0], dtype=bool)])

    edges = [
        graph.edges_for_link(_select_link(graph, lower, upper))
        for lower, upper in zip(order, order[1:])
    ]
    k = len(order)
    # keep only poses on a complete chain: forward AND backward reachable
    fwd = [np.zeros(n, dtype=bool) for n in nposes]
    fwd[0][:] = True
    for j in range(k - 1):
        alive = fwd[j][edges[j].low]
        fwd[j + 1][edges[j].high[alive]] = True
    bwd = [np.zeros(n, dtype=bool) for n in nposes]
    bwd[-1][:] = True
    for j in range(k - 2, -1, -1):
        alive = bwd[j + 1][edges[j].high]
        bwd[j][edges[j].low[alive]] = True
    kept_mask = [f & b for f, b in zip(fwd, bwd)]
    return Layers(order, fragments, nposes, edges, kept_mask)


def _kept_edges(layers: Layers, j: int) -> tuple[np.ndarray, np.ndarray]:
    low = layers.edges[j].low
    high = layers.edges[j].high
    valid = layers.kept_mask[j][low] & layers.kept_mask[j + 1][high]
    return low[valid], high[valid]


# -- pose identity helpers ------------------------------------------------


def _identities(chunk) -> np.ndarray:
    """(N, 5) int64 rows of (conformer, rotamer, tx, ty, tz)."""
    return np.column_stack(
        (
            chunk.conformers.astype(np.int64),
            chunk.rotamers.astype(np.int64),
            chunk.translations_grid.astype(np.int64),
        )
    )


def _encode(rows: np.ndarray) -> np.ndarray:
    """One opaque key per identity row, comparable with the same byte order
    np.unique(axis=0) uses (so argsort here matches np.unique's ordering)."""
    a = np.ascontiguousarray(rows.astype(np.int64))
    return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()


# -- counting -------------------------------------------------------------


@dataclass
class CountResult:
    order: list[str]  # pool names in fragment order
    fragments: list[int]
    kept: list[int]
    unique: list[int]
    total_chains: int


def _count_unique(graph: ChainGraph, pool: str, kept_idx: np.ndarray) -> int:
    if kept_idx.size == 0:
        return 0
    rows = _identities(select_pose_indices(graph._result_dir(pool), kept_idx))
    return int(len(np.unique(rows, axis=0)))


def _count_paths(layers: Layers) -> int:
    if len(layers.order) == 1:
        return int(layers.kept_mask[0].sum())
    ways = np.where(layers.kept_mask[0], 1, 0).astype(np.int64)
    for j in range(len(layers.order) - 1):
        lo, hi = _kept_edges(layers, j)
        nxt = np.zeros(layers.nposes[j + 1], dtype=np.int64)
        np.add.at(nxt, hi, ways[lo])
        ways = nxt
    return int(ways.sum())


def count_chains(graph: ChainGraph, selected: list[str]) -> CountResult:
    layers = _connectivity(graph, selected)
    return CountResult(
        order=layers.order,
        fragments=layers.fragments,
        kept=[int(m.sum()) for m in layers.kept_mask],
        unique=[
            _count_unique(graph, pool, np.flatnonzero(mask).astype(np.int64))
            for pool, mask in zip(layers.order, layers.kept_mask)
        ],
        total_chains=_count_paths(layers),
    )


# -- building -------------------------------------------------------------


def _write_unique_pose_dir(
    graph: ChainGraph, pool: str, kept_idx: np.ndarray, out_dir: Path
) -> np.ndarray:
    """Write the unique poses among ``kept_idx`` to ``out_dir`` and return, for
    each kept pose (in ``kept_idx`` order), its 1-based index in that pose dir."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    if kept_idx.size == 0:
        return np.empty(0, dtype=np.int64)

    kept_ids = _identities(select_pose_indices(graph._result_dir(pool), kept_idx))
    unique_rows, inverse = np.unique(kept_ids, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()

    packed = pack_pool(
        unique_rows[:, 0].astype(np.uint16),
        unique_rows[:, 1].astype(np.uint16),
        unique_rows[:, 2:5].astype(np.int32),
        bucket_size=DEFAULT_BUCKET_SIZE,
    )
    for i, (M, O, C, P) in enumerate(packed, start=1):
        write_arc_file(out_dir / f"poses-{i}.arc", M, O, C, P, bucket_size=DEFAULT_BUCKET_SIZE)

    # map np.unique order -> written global order, so the returned indices point
    # at the poses as they are laid out in the pose dir.
    written = _identities(select_pose_indices(out_dir, np.arange(len(unique_rows), dtype=np.int64)))
    written_pos = np.argsort(_encode(written), kind="stable")  # written[written_pos] == unique_rows
    return (written_pos[inverse] + 1).astype(np.int64)


def _enumerate_paths(layers: Layers) -> np.ndarray:
    """All complete chains as an (n_chains, n_fragments) array of rep pose ids."""
    kept0 = np.flatnonzero(layers.kept_mask[0]).astype(np.int64)
    paths = kept0[:, None]
    for j in range(len(layers.order) - 1):
        if len(paths) == 0:
            break
        lo, hi = _kept_edges(layers, j)
        highs, pidx = _join(lo, hi, paths[:, -1], np.arange(len(paths), dtype=np.int64))
        paths = np.column_stack((paths[pidx], highs))
    if len(paths) == 0:
        return np.empty((0, len(layers.order)), dtype=np.int64)
    return paths


def build_chains(
    graph: ChainGraph, selected: list[str], output_dir: Path
) -> tuple[Layers, np.ndarray]:
    """Write one pose dir of unique poses per fragment under ``output_dir`` and
    return the chains as 1-based indices into those pose dirs."""
    layers = _connectivity(graph, selected)
    output_dir.mkdir(parents=True, exist_ok=True)

    rep_to_pool: list[np.ndarray] = []
    for pool, mask, n in zip(layers.order, layers.kept_mask, layers.nposes):
        kept_idx = np.flatnonzero(mask).astype(np.int64)
        one_based = _write_unique_pose_dir(graph, pool, kept_idx, output_dir / pool)
        mapping = np.zeros(n, dtype=np.int64)
        mapping[kept_idx] = one_based
        rep_to_pool.append(mapping)

    paths = _enumerate_paths(layers)
    table = np.empty_like(paths)
    for j in range(len(layers.order)):
        table[:, j] = rep_to_pool[j][paths[:, j]]
    return layers, table


# -- selection ------------------------------------------------------------


def resolve_selection(
    graph: ChainGraph,
    target: list[str] | None,
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
    if target:
        check(target, "--target")
        chosen = set(target)
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
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Build mode: dir to write one pose dir of unique poses per fragment.",
    )
    parser.add_argument("--target", nargs="+", metavar="POOL", help="Use only these representatives.")
    parser.add_argument("--exclude", nargs="+", metavar="POOL", help="Drop these representatives.")
    parser.add_argument("--project-root", help="Override the project root for result dirs.")
    args = parser.parse_args(argv)

    if not args.count and not args.output_dir:
        parser.error("build mode requires --output-dir (or use --count)")

    data = json.loads(Path(args.graph_json).read_text())
    project_root = Path(args.project_root) if args.project_root else Path(data["project"])
    graph = ChainGraph(data, project_root)

    try:
        selected = resolve_selection(graph, args.target, args.exclude)
        if args.count:
            print(_format_count(count_chains(graph, selected)))
        else:
            layers, table = build_chains(graph, selected, Path(args.output_dir))
            sys.stdout.write("\t".join(layers.order) + "\n")
            np.savetxt(sys.stdout, table, fmt="%d", delimiter="\t")
    except ChainError as exc:
        parser.exit(2, f"error: {exc}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
