from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from .checksum import byte_checksum, json_checksum
from .errors import MiddleError
from .graph import ActionGraph
from .jsonutil import write_canonical_json
from .project import Project
from .resolve import ResolvedAction
from .schema import DEPENDENCY_FIELDS


NON_LOAD_BEARING: dict[str, set[str]] = {
    "bridge": {
        "memory",
        "nprocs",
        "max-intermediate-poses",
        "max-final-poses",
        "rotamer-chunks",
        "estimator-seed",
        "estimator-sample-size",
    }
}


def _plain_value(value: Any) -> Any:
    if isinstance(value, ResolvedAction):
        raise TypeError("dependency value must be stripped before serialization")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_plain_value(v) for v in value]
    if isinstance(value, list):
        return [_plain_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain_value(v) for k, v in sorted(value.items())}
    return value


def parameter_record(action: ResolvedAction, dep_sigils: dict[str, str]) -> dict[str, Any]:
    dep_fields = set(DEPENDENCY_FIELDS[action.action])
    runtime_fields = NON_LOAD_BEARING.get(action.action, set())
    params: dict[str, Any] = {}
    for key, value in action.params.items():
        if key in dep_fields:
            continue
        if key in runtime_fields:
            continue
        if isinstance(value, ResolvedAction):
            continue
        params[key] = _plain_value(value)
    for field, path in sorted(action.file_fields.items()):
        params[field] = byte_checksum(path)
        params[f"__{field}"] = path.name
    if dep_sigils:
        params["dependencies"] = dep_sigils
    return params


def sigil_parts(action: ResolvedAction, dep_sigils: dict[str, str]) -> tuple[str, str, dict[str, Any]]:
    record = parameter_record(action, dep_sigils)
    part1_input = {
        k: v
        for k, v in record.items()
        if not k.startswith("__") and k != "dependencies"
    }
    part2_input = dep_sigils or None
    part1 = json_checksum(part1_input)[:10]
    part2 = "null" if part2_input is None else json_checksum(part2_input)[:10]
    return part1, part2, record


def action_sigil(action: ResolvedAction, dep_sigils: dict[str, str]) -> tuple[str, dict[str, Any]]:
    part1, part2, record = sigil_parts(action, dep_sigils)
    return f"{part1}-{part2}", record


def compute_project_sigils(project: Project, *, force: bool = False, delete: bool = False) -> dict[str, str]:
    project.ensure_cache()
    graph = ActionGraph(project)
    resolved = graph.build()
    sigils: dict[str, str] = {}
    for name in graph.topo():
        action = resolved[name]
        dep_sigils: dict[str, str] = {}
        for field in DEPENDENCY_FIELDS[action.action]:
            dep = action.params.get(field)
            if isinstance(dep, ResolvedAction):
                dep_sigils[field] = sigils[dep.name]
        sigil, record = action_sigil(action, dep_sigils)
        sigil_path = action.path / "sigil.txt"
        result_path = action.path / "result.txt"
        if sigil_path.is_file():
            old = sigil_path.read_text().strip()
            if old and old != sigil:
                if result_path.exists() and not delete:
                    raise MiddleError(f"{action.name}: stale sigil and result.txt exists; use --delete")
                if not force and not delete:
                    raise MiddleError(f"{action.name}: stale sigil {old} != {sigil}; use --force or --delete")
                if delete:
                    result_path.unlink(missing_ok=True)
                    checksum = project.checksum_dir / old
                    checksum.unlink(missing_ok=True)
                    old_results = project.results_dir / old
                    if old_results.exists():
                        shutil.rmtree(old_results)
        sigil_path.write_text(sigil + "\n")
        write_canonical_json(project.parameters_dir / sigil, record)
        sigils[name] = sigil
    return sigils


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alaric-sigil")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--force", action="store_true")
    group.add_argument("--delete", action="store_true")
    parser.add_argument("project_root", nargs="?", default=".")
    args = parser.parse_args(argv)
    project = Project.discover(args.project_root)
    sigils = compute_project_sigils(project, force=args.force, delete=args.delete)
    for name in sorted(sigils):
        print(f"{name}\t{sigils[name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
