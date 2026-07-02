"""Build a connectivity/provenance graph of pose pools for chain building.

A *pose pool* is a logical set of poses carrying a particular set of propagated
constraints; it is *almost* a pose dir, but not exactly: filtered/merged pools are
indexed views over a concrete pool, several action dirs with one sigil share a
single result dir, and `score`/`mask`/`rmsd` dirs are constraints on a pool rather
than pools themselves. This module projects the middle-level action graph
(:mod:`alaric.middle.graph`) onto pose pools only and records, for each pool, the
provenance arrays that connect it to its parent pool(s):

- ``grow``   -> ``provenance.npy``  (row = this pose, value = source pose id; cross-fragment)
- ``filter`` -> ``provenance.npy``  (row = this pose, value = input pose id;  same fragment)
- ``identity`` (merge/dedup) -> ``map-1.npy`` / ``map-2.npy``
                                (rows are ``(parent_id, this_id)`` pairs; same fragment)

For chain building it then picks, per fragment, the **representative** pool: the
unique last descendant (most-filtered sink) whose lineage is wired into a
cross-fragment grow. Fragments with no such lineage are skipped; fragments with
more than one such lineage are an error. Finally it emits, per grow, the ordered
list of arrays a chain builder must compose to connect two representatives.

The output is a small skeleton (pool-level, with references to the on-disk arrays),
not the gigapose-scale per-pose chains themselves.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import PoolGraphError
from .graph import ActionGraph
from .project import Project
from .resolve import ResolvedAction, final_fragment
from .schema import OUTPUT_KIND, POSE


PROVENANCE_FILE = "provenance.npy"
MAP1_FILE = "map-1.npy"
MAP2_FILE = "map-2.npy"

# Orientation of a provenance array, so a chain builder knows how to read it.
ORIENT_DENSE = "row=this,value=parent"  # provenance.npy: one parent id per this-pose
ORIENT_PAIRS = "cols=(parent,this)"  # map-*.npy: sparse (parent_id, this_id) edge list


def _orientation(array: str) -> str:
    return ORIENT_PAIRS if array.startswith("map") else ORIENT_DENSE


@dataclass(frozen=True)
class ProvEdge:
    """A derivation edge from a pool to one parent pool.

    ``array`` is the file (relative to the pool's result dir) realizing the
    per-pose link, or ``None`` for non-provenance dependencies (``restrict``).
    """

    parent: str
    kind: str  # grow | filter | merge | dedup | restrict
    array: str | None
    cross_fragment: bool


@dataclass
class PoolNode:
    name: str
    fragment: int
    kind: str  # anchor | grow | filter | merge | dedup
    parents: list[ProvEdge] = field(default_factory=list)
    direction: str | None = None  # grow only: forward | backward
    sigil: str | None = None
    result_dir: str | None = None  # relative to project root, if a sigil exists
    materialized: bool = False
    constraint: dict[str, Any] | None = None  # filter: threshold + score_input


def _pool_kind(action: ResolvedAction) -> str:
    if action.action in {"anchor", "anchor-test"}:
        return "anchor"
    if action.action in {"grow", "grow-test"}:
        return "grow"
    if action.action == "filter":
        return "filter"
    if action.action == "identity":
        in1 = action.params.get("input1")
        in2 = action.params.get("input2")
        if isinstance(in1, ResolvedAction) and isinstance(in2, ResolvedAction) and in1.name == in2.name:
            return "dedup"
        return "merge"
    raise PoolGraphError(f"{action.name}: not a pose-pool action ({action.action})")


def _provenance_edges(action: ResolvedAction, fragment: int, frag_of: dict[str, int]) -> list[ProvEdge]:
    kind = _pool_kind(action)
    edges: list[ProvEdge] = []

    def dep(name_field: str) -> ResolvedAction | None:
        value = action.params.get(name_field)
        return value if isinstance(value, ResolvedAction) else None

    if action.action in {"grow", "grow-test"}:
        src = dep("input")
        if src is not None:
            edges.append(
                ProvEdge(src.name, "grow", PROVENANCE_FILE, frag_of[src.name] != fragment)
            )
        restrict = dep("restrict_input")
        if restrict is not None:
            # restrict is a "--to" pool at the grow target fragment; it sub-selects
            # output poses but is not where poses come from, so it carries no array
            # and is excluded from the lineage walk.
            edges.append(
                ProvEdge(restrict.name, "restrict", None, frag_of[restrict.name] != fragment)
            )
    elif action.action == "filter":
        src = dep("input")
        if src is not None:
            edges.append(
                ProvEdge(src.name, "filter", PROVENANCE_FILE, frag_of[src.name] != fragment)
            )
    elif action.action == "identity":
        in1 = dep("input1")
        in2 = dep("input2")
        if in1 is not None:
            edges.append(ProvEdge(in1.name, kind, MAP1_FILE, frag_of[in1.name] != fragment))
        if in2 is not None:
            edges.append(ProvEdge(in2.name, kind, MAP2_FILE, frag_of[in2.name] != fragment))
    return edges


def _read_sigil(action: ResolvedAction) -> str | None:
    sigil_path = action.path / "sigil.txt"
    if sigil_path.is_file():
        text = sigil_path.read_text().strip()
        return text or None
    return None


class PoolGraph:
    def __init__(self, project: Project, targets: list[str] | None = None):
        self.project = project
        graph = ActionGraph(project)
        resolved = graph.build(targets)
        self.pools: dict[str, PoolNode] = {}

        pose_actions = {
            name: action
            for name, action in resolved.items()
            if OUTPUT_KIND[action.action] == POSE
        }
        frag_of = {name: final_fragment(action) for name, action in pose_actions.items()}

        for name, action in pose_actions.items():
            fragment = frag_of[name]
            node = PoolNode(
                name=name,
                fragment=fragment,
                kind=_pool_kind(action),
                parents=_provenance_edges(action, fragment, frag_of),
            )
            if action.action in {"grow", "grow-test"}:
                node.direction = str(action.params.get("direction"))
            if action.action == "filter":
                score_input = action.params.get("score_input")
                node.constraint = {
                    "threshold": action.params.get("threshold"),
                    "score_input": score_input.name if isinstance(score_input, ResolvedAction) else None,
                }
            sigil = _read_sigil(action)
            if sigil:
                node.sigil = sigil
                result_dir = self.project.results_dir / sigil
                node.result_dir = str(result_dir.relative_to(self.project.root))
                node.materialized = result_dir.is_dir()
            self.pools[name] = node

        self.pools_by_fragment: dict[int, list[str]] = {}
        for name, node in self.pools.items():
            self.pools_by_fragment.setdefault(node.fragment, []).append(name)
        for names in self.pools_by_fragment.values():
            names.sort()

        self.representatives: dict[int, str] = {}
        self.skipped: dict[int, str] = {}
        self._select_representatives()

    # -- within-fragment topology helpers ---------------------------------

    def _within_parents(self, name: str) -> list[ProvEdge]:
        """File-bearing parent edges that stay inside the pool's fragment.

        Parallel edges to the same parent (a dedup's redundant ``map-1``/``map-2``)
        are collapsed to one, so they are not mistaken for distinct lineages.
        """
        node = self.pools[name]
        seen: set[str] = set()
        out: list[ProvEdge] = []
        for e in node.parents:
            if e.array is None or e.cross_fragment or e.parent in seen:
                continue
            seen.add(e.parent)
            out.append(e)
        return out

    def _chain_anchored(self) -> set[str]:
        """Pools wired into a cross-fragment grow: grown pools and grow sources."""
        anchored: set[str] = set()
        for name, node in self.pools.items():
            if node.kind == "grow":
                anchored.add(name)  # grown from a neighbour fragment
                for e in node.parents:
                    if e.kind == "grow":
                        anchored.add(e.parent)  # the source pool, used to grow
        return anchored

    def _components(self, fragment: int) -> list[set[str]]:
        """Connected components of fragment pools under within-fragment edges."""
        names = self.pools_by_fragment.get(fragment, [])
        parent: dict[str, str] = {n: n for n in names}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        nameset = set(names)
        for n in names:
            for e in self._within_parents(n):
                if e.parent in nameset:
                    union(n, e.parent)
        groups: dict[str, set[str]] = {}
        for n in names:
            groups.setdefault(find(n), set()).add(n)
        return list(groups.values())

    def _sinks(self, fragment: int) -> set[str]:
        """Fragment pools that no other fragment pool derives from."""
        names = set(self.pools_by_fragment.get(fragment, []))
        is_parent: set[str] = set()
        for n in names:
            for e in self._within_parents(n):
                if e.parent in names:
                    is_parent.add(e.parent)
        return {n for n in names if n not in is_parent}

    def _select_representatives(self) -> None:
        anchored = self._chain_anchored()
        for fragment in sorted(self.pools_by_fragment):
            components = self._components(fragment)
            relevant = [c for c in components if c & anchored]
            sinks = self._sinks(fragment)
            relevant_sinks = sorted(
                n for n in sinks if any(n in c for c in relevant)
            )
            if not relevant_sinks:
                self.skipped[fragment] = (
                    "no chain-connected lineage (not wired into a grow)"
                )
                continue
            if len(relevant_sinks) > 1:
                raise PoolGraphError(
                    f"fragment {fragment} has multiple chain lineages "
                    f"({', '.join(relevant_sinks)}); the representative is ambiguous"
                )
            self.representatives[fragment] = relevant_sinks[0]

    # -- path composition --------------------------------------------------

    def _path_up(self, start: str, target: str) -> list[dict[str, Any]]:
        """Ordered array steps mapping ``start`` poses up onto ``target`` poses.

        Both pools must be in the same fragment. Each step maps a pool's poses to
        one parent's poses. Raises if there is no path or more than one.
        """
        if start == target:
            return []

        found: list[list[dict[str, Any]]] = []

        def walk(node: str, trail: list[dict[str, Any]], seen: frozenset[str]) -> None:
            for e in self._within_parents(node):
                step = {
                    "pool": node,
                    "array": e.array,
                    "orientation": _orientation(e.array),
                    "to": e.parent,
                }
                if e.parent == target:
                    found.append(trail + [step])
                elif e.parent not in seen:
                    walk(e.parent, trail + [step], seen | {e.parent})

        walk(start, [], frozenset({start}))
        if not found:
            raise PoolGraphError(
                f"{start} does not descend from {target}: cannot compose chain provenance"
            )
        if len(found) > 1:
            raise PoolGraphError(
                f"multiple provenance lineages from {start} to {target}; chain link is ambiguous"
            )
        return found[0]

    def links(self) -> list[dict[str, Any]]:
        """One chain link per grow connecting two represented fragments.

        Each link gives the arrays mapping both representatives up into the grow
        source's pose-id space (the ``join_pool``), where a chain builder joins
        them by source pose id.
        """
        result: list[dict[str, Any]] = []
        for name, node in sorted(self.pools.items()):
            if node.kind != "grow":
                continue
            grow_edge = next((e for e in node.parents if e.kind == "grow"), None)
            if grow_edge is None:
                continue
            source = grow_edge.parent
            source_fragment = self.pools[source].fragment
            target_fragment = node.fragment
            if target_fragment not in self.representatives:
                continue
            if source_fragment not in self.representatives:
                continue
            target_rep = self.representatives[target_fragment]
            source_rep = self.representatives[source_fragment]

            target_to_join = self._path_up(target_rep, name)
            target_to_join.append(
                {
                    "pool": name,
                    "array": grow_edge.array,
                    "orientation": _orientation(grow_edge.array),
                    "to": source,
                }
            )
            source_to_join = self._path_up(source_rep, source)

            result.append(
                {
                    "grow_pool": name,
                    "direction": node.direction,
                    "source_fragment": source_fragment,
                    "source_rep": source_rep,
                    "target_fragment": target_fragment,
                    "target_rep": target_rep,
                    "join_pool": source,
                    "target_to_join": target_to_join,
                    "source_to_join": source_to_join,
                }
            )
        return result

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        pools: dict[str, Any] = {}
        for name in sorted(self.pools):
            node = self.pools[name]
            entry: dict[str, Any] = {
                "fragment": node.fragment,
                "kind": node.kind,
                "parents": [
                    {
                        "pool": e.parent,
                        "edge": e.kind,
                        "array": e.array,
                        "orientation": _orientation(e.array) if e.array else None,
                        "cross_fragment": e.cross_fragment,
                    }
                    for e in node.parents
                ],
                "sigil": node.sigil,
                "result_dir": node.result_dir,
                "materialized": node.materialized,
            }
            if node.direction is not None:
                entry["direction"] = node.direction
            if node.constraint is not None:
                entry["constraint"] = node.constraint
            pools[name] = entry
        return {
            "project": str(self.project.root),
            "pools": pools,
            "representatives": {str(k): v for k, v in sorted(self.representatives.items())},
            "skipped_fragments": {str(k): v for k, v in sorted(self.skipped.items())},
            "links": self.links(),
        }

    def to_dot(self) -> str:
        reps = set(self.representatives.values())
        lines = ["digraph pose_pools {", "  rankdir=LR;", "  node [shape=box];"]
        for name in sorted(self.pools):
            node = self.pools[name]
            attrs = [f'label="{name}\\nfrag{node.fragment} {node.kind}"']
            if name in reps:
                attrs.append("style=filled")
                attrs.append("fillcolor=lightblue")
            lines.append(f'  "{name}" [{", ".join(attrs)}];')
        # provenance edges (this -> parent)
        for name in sorted(self.pools):
            for e in self.pools[name].parents:
                style = "dashed" if e.kind == "restrict" else "solid"
                label = e.kind if e.array is None else f"{e.kind}:{e.array}"
                lines.append(
                    f'  "{name}" -> "{e.parent}" [label="{label}", style={style}];'
                )
        # chain links between representatives (bold red), via the grow
        for link in self.links():
            lines.append(
                f'  "{link["target_rep"]}" -> "{link["source_rep"]}" '
                f'[label="chain {link["direction"]}", color=red, penwidth=2, '
                f'constraint=false];'
            )
        lines.append("}")
        return "\n".join(lines)


def build_pool_graph(start: str | Path = ".", targets: list[str] | None = None) -> PoolGraph:
    project = Project.discover(start)
    return PoolGraph(project, targets)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="alaric-pool-graph",
        description="Build the pose-pool connectivity/provenance graph for chain building.",
    )
    parser.add_argument("project_root", nargs="?", default=".")
    parser.add_argument(
        "-f",
        "--format",
        choices=("json", "dot"),
        default="json",
        help="Output format (default: json).",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Write to this file instead of stdout.",
    )
    parser.add_argument(
        "--target",
        action="append",
        dest="targets",
        help="Restrict to this action and its dependencies (repeatable).",
    )
    args = parser.parse_args(argv)

    graph = build_pool_graph(args.project_root, args.targets)
    if args.format == "dot":
        text = graph.to_dot() + "\n"
    else:
        text = json.dumps(graph.to_dict(), indent=2, sort_keys=False) + "\n"

    if args.output:
        Path(args.output).write_text(text)
    else:
        import sys

        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
