from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .errors import SchemaError


POSE = "pose"
CONNECTED_POSE = "connected_pose"
SCORE = "score"
MASK = "mask"


ACTION_SCHEMAS: dict[str, set[str]] = {
    "anchor": {"action", "type", "fragment", "sequence", "exclude", "protein", "resid", "nucleotide", "dihedral", "angle", "margin"},
    "anchor-test": {"action", "type", "fragment", "sequence", "exclude", "protein", "resid", "nucleotide", "dihedral", "angle", "margin", "nconformers"},
    "grow": {"action", "type", "input", "fragment", "sequence", "exclude", "direction", "crmsd", "ovrmsd"},
    "score": {"action", "type", "input", "sequence", "exclude", "protein", "nb_kernel"},
    "rmsd": {"action", "type", "input", "fragment", "exclude", "reference"},
    "score_add": {"action", "type", "score_input1", "score_input2"},
    "mask": {"action", "type", "input", "score_input", "threshold"},
    "filter": {"action", "type", "input", "score_input", "threshold", "mask_input"},
    "identity": {"action", "type", "input1", "input2"},
    "bridge": {"action", "type", "input1", "input2", "lower-crmsd", "lower-ovrmsd", "upper-crmsd", "upper-ovrmsd", "memory", "max-intermediate-poses", "max-final-poses", "nprocs", "rotamer-chunks", "estimator-seed", "estimator-sample-size"},
}

REQUIRED: dict[str, set[str]] = {
    "anchor": {"fragment", "sequence", "exclude", "protein", "resid", "nucleotide"},
    "anchor-test": {"fragment", "sequence", "exclude", "protein", "resid", "nucleotide", "nconformers"},
    "grow": {"input", "fragment", "sequence", "exclude", "direction", "crmsd", "ovrmsd"},
    "score": {"input", "sequence", "exclude", "protein"},
    "rmsd": {"input", "fragment", "exclude"},
    "score_add": {"score_input1", "score_input2"},
    "mask": {"input", "score_input", "threshold"},
    "filter": {"input"},
    "identity": {"input1", "input2"},
    "bridge": {"input1", "input2"},
}

DEPENDENCY_FIELDS: dict[str, dict[str, str]] = {
    "anchor": {},
    "anchor-test": {},
    "grow": {"input": POSE},
    "score": {"input": POSE},
    "rmsd": {"input": POSE},
    "score_add": {"score_input1": SCORE, "score_input2": SCORE},
    "mask": {"input": POSE, "score_input": SCORE},
    "filter": {"input": POSE, "score_input": SCORE, "mask_input": MASK},
    "identity": {"input1": POSE, "input2": POSE},
    "bridge": {"input1": POSE, "input2": POSE},
}

OUTPUT_KIND = {
    "anchor": POSE,
    "anchor-test": POSE,
    "grow": POSE,
    "score": SCORE,
    "rmsd": SCORE,
    "score_add": SCORE,
    "mask": MASK,
    "filter": POSE,
    "identity": POSE,
    "bridge": CONNECTED_POSE,
}

AUTO_FIELDS = {
    "anchor": {"fragment", "sequence", "exclude", "protein", "dihedral", "angle"},
    "anchor-test": {"fragment", "sequence", "exclude", "protein", "dihedral", "angle"},
    "grow": {"fragment", "sequence", "exclude", "direction", "crmsd", "ovrmsd"},
    "bridge": {"lower-crmsd", "lower-ovrmsd", "upper-crmsd", "upper-ovrmsd"},
    "score": {"sequence", "exclude", "protein"},
    "rmsd": {"fragment", "exclude"},
    "mask": {"threshold"},
    "filter": {"threshold"},
}


@dataclass(frozen=True)
class ActionSpec:
    name: str
    path: Path
    action: str
    raw: dict[str, Any]


def load_action_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise SchemaError(f"{path}: expected mapping")
    return data


def normalize_action(name: str, path: Path, data: dict[str, Any]) -> ActionSpec:
    data = dict(data)
    if "action" not in data:
        if "type" not in data:
            raise SchemaError(f"{path}: missing action")
        data["action"] = data["type"]
    action = data["action"]
    if action not in ACTION_SCHEMAS:
        raise SchemaError(f"{path}: unsupported action {action!r}")
    unknown = set(data) - ACTION_SCHEMAS[action]
    if unknown:
        raise SchemaError(f"{path}: unknown keys for {action}: {', '.join(sorted(unknown))}")
    missing = REQUIRED[action] - set(data)
    if missing:
        raise SchemaError(f"{path}: missing keys for {action}: {', '.join(sorted(missing))}")

    if action in {"anchor", "anchor-test"}:
        nuc = str(data.get("nucleotide")).lower()
        if nuc not in {"first", "second"}:
            raise SchemaError(f"{path}: nucleotide must be first or second")
        data["nucleotide"] = nuc
        data.setdefault("margin", 0.5)
        if action == "anchor-test":
            nconformers = int(data["nconformers"])
            if nconformers <= 0:
                raise SchemaError(f"{path}: nconformers must be positive")
            data["nconformers"] = nconformers
    if action == "grow" and data.get("direction") != "auto":
        direction = str(data["direction"]).lower()
        if direction in {"fwd", "forward"}:
            data["direction"] = "forward"
        elif direction in {"bwd", "backward"}:
            data["direction"] = "backward"
        else:
            raise SchemaError(f"{path}: direction must be fwd/forward/bwd/backward/auto")
    if action == "score":
        data.setdefault("nb_kernel", "compiled")
        if data["nb_kernel"] not in {"compiled", "jax"}:
            raise SchemaError(f"{path}: nb_kernel must be compiled or jax")
    if action == "rmsd":
        data.setdefault("reference", "reference.pdb")
    if action == "filter":
        score_route = {"score_input", "threshold"} <= set(data)
        mask_route = "mask_input" in data
        if score_route == mask_route:
            raise SchemaError(f"{path}: filter requires either score_input+threshold or mask_input")
    for key, value in data.items():
        if value == "auto" and key not in AUTO_FIELDS.get(action, set()):
            raise SchemaError(f"{path}: {key}: auto is not supported for {action}")
    return ActionSpec(name=name, path=path, action=action, raw=data)


def load_action_spec(name: str, yaml_path: Path) -> ActionSpec:
    return normalize_action(name, yaml_path.parent, load_action_yaml(yaml_path))
