from __future__ import annotations

from dataclasses import replace
from typing import Any

from .errors import GraphError, ResolveError
from .project import Project
from .resolve import ResolvedAction, final_fragment, final_sequence, resolve_action
from .schema import DEPENDENCY_FIELDS, OUTPUT_KIND, ActionSpec, load_action_spec


class ActionGraph:
    def __init__(self, project: Project):
        self.project = project
        self.specs: dict[str, ActionSpec] = {
            action_dir.name: load_action_spec(action_dir.name, action_dir.yaml_path)
            for action_dir in project.action_dirs()
        }
        self.resolved: dict[str, ResolvedAction] = {}
        self._visiting: set[str] = set()

    def topo(self, targets: list[str] | None = None) -> list[str]:
        ordered: list[str] = []
        permanent: set[str] = set()

        def visit(name: str) -> None:
            if name in permanent:
                return
            if name in self._visiting:
                raise GraphError(f"cycle detected at {name}")
            spec = self.specs.get(name)
            if spec is None:
                raise GraphError(f"unknown action dependency: {name}")
            self._visiting.add(name)
            for dep_name in self.dependency_names(spec).values():
                visit(dep_name)
            self._visiting.remove(name)
            permanent.add(name)
            ordered.append(name)

        names = sorted(self.specs) if targets is None else targets
        for name in names:
            visit(name)
        return ordered

    @staticmethod
    def dependency_names(spec: ActionSpec) -> dict[str, str]:
        result: dict[str, str] = {}
        for field in DEPENDENCY_FIELDS[spec.action]:
            if field in spec.raw:
                result[field] = str(spec.raw[field])
        return result

    def build(self, targets: list[str] | None = None) -> dict[str, ResolvedAction]:
        self.resolved = {}
        for name in self.topo(targets):
            spec = self.specs[name]
            deps = self.dependency_names(spec)
            for field, dep_name in deps.items():
                if dep_name not in self.resolved:
                    raise GraphError(f"{name}: missing dependency {dep_name}")
                expected = DEPENDENCY_FIELDS[spec.action][field]
                actual = OUTPUT_KIND[self.resolved[dep_name].action]
                if actual != expected:
                    raise GraphError(
                        f"{name}.{field}: expected {expected} output from {dep_name}, got {actual}"
                    )
            resolved = resolve_action(self.project, spec, self.resolved)
            params = dict(resolved.params)
            for field, dep_name in deps.items():
                params[field] = self.resolved[dep_name]
            resolved = replace(resolved, params=params)
            self._cross_validate(resolved)
            if resolved.action == "mask" and resolved.params.get("threshold") == "auto":
                params = dict(resolved.params)
                params["threshold"] = self._auto_threshold(resolved)
                resolved = replace(resolved, params=params)
            if resolved.action == "filter" and resolved.params.get("threshold") == "auto":
                params = dict(resolved.params)
                params["threshold"] = self._auto_threshold(resolved)
                resolved = replace(resolved, params=params)
            self.resolved[name] = resolved
        return self.resolved

    def _auto_threshold(self, resolved: ResolvedAction) -> float:
        score = resolved.params["score_input"]
        if not isinstance(score, ResolvedAction):
            raise GraphError(f"{resolved.name}: unresolved score_input")
        if score.action == "score_add":
            raise ResolveError(f"{resolved.name}: threshold:auto is not defined for score_add")
        if score.action == "rmsd":
            raise ResolveError(f"{resolved.name}: threshold:auto is not defined for rmsd")
        if score.action != "score":
            raise ResolveError(f"{resolved.name}: threshold:auto requires score action")
        scored_input = score.params["input"]
        if not isinstance(scored_input, ResolvedAction):
            raise GraphError(f"{score.name}: unresolved score input")
        frag_key = f"frag{final_fragment(scored_input)}"
        protein = str(score.params["protein"])
        constraints = self.project.data_dir / "constraints.json"
        from .jsonutil import read_json

        data = read_json(constraints)
        try:
            return float(data[frag_key]["scores"][protein])
        except KeyError as exc:
            raise ResolveError(
                f"{resolved.name}: no auto threshold for {frag_key} and protein {protein}"
            ) from exc

    def _cross_validate(self, resolved: ResolvedAction) -> None:
        p = resolved.params
        if resolved.action == "score_add":
            a = scored_pose_input(p["score_input1"])
            b = scored_pose_input(p["score_input2"])
            if not isinstance(a, ResolvedAction) or not isinstance(b, ResolvedAction) or a.name != b.name:
                raise GraphError(f"{resolved.name}: score_add inputs must score the same pose input")
        elif resolved.action == "mask":
            score_input = p["score_input"]
            if not isinstance(score_input, ResolvedAction):
                raise GraphError(f"{resolved.name}: unresolved score_input")
            if scored_pose_input(score_input).name != p["input"].name:
                raise GraphError(f"{resolved.name}: mask.score_input.input must equal mask.input")
        elif resolved.action == "filter":
            if "score_input" in p:
                score_input = p["score_input"]
                if scored_pose_input(score_input).name != p["input"].name:
                    raise GraphError(f"{resolved.name}: filter score route input mismatch")
            if "mask_input" in p:
                mask_input = p["mask_input"]
                if mask_input.params.get("input").name != p["input"].name:
                    raise GraphError(f"{resolved.name}: filter mask route input mismatch")
        elif resolved.action == "identity":
            if final_fragment(p["input1"]) != final_fragment(p["input2"]):
                raise GraphError(f"{resolved.name}: identity inputs must resolve to the same fragment")
        elif resolved.action == "grow":
            restrict = p.get("restrict_input")
            if isinstance(restrict, ResolvedAction):
                if final_fragment(restrict) != final_fragment(resolved):
                    raise GraphError(
                        f"{resolved.name}: restrict_input must resolve to the grow target fragment"
                    )
                if final_sequence(restrict) != final_sequence(resolved):
                    raise GraphError(
                        f"{resolved.name}: restrict_input must resolve to the grow target sequence"
                    )

    def action(self, name: str) -> ResolvedAction:
        if not self.resolved:
            self.build()
        try:
            return self.resolved[name]
        except KeyError:
            raise GraphError(f"unknown action: {name}") from None


def scored_pose_input(score: ResolvedAction) -> ResolvedAction:
    if score.action in {"score", "rmsd"}:
        dep = score.params.get("input")
        if isinstance(dep, ResolvedAction):
            return dep
    if score.action == "score_add":
        return scored_pose_input(score.params["score_input1"])
    raise GraphError(f"{score.name}: cannot resolve underlying scored pose input")
