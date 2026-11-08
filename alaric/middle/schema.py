from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .errors import SchemaError


def _gaussian_filter_spec() -> dict[str, Any]:
    """Load the authoritative runtime name table from A7."""
    try:
        from ..gaussian_filter import load_spec

        spec = load_spec(Path(__file__).resolve().parents[1] / "gaussian_stacking_filter.json")
    except Exception as exc:
        raise SchemaError(f"cannot load the Gaussian filter specification: {exc}") from exc
    thresholds = spec.get("thresholds")
    if not isinstance(thresholds, dict) or len(thresholds) != 12:
        raise SchemaError("Gaussian filter specification must define all twelve names")
    return spec


POSE = "pose"
SCORE = "score"
MASK = "mask"


ACTION_SCHEMAS: dict[str, set[str]] = {
    "anchor": {"action", "type", "fragment", "sequence", "exclude", "protein", "resid", "nucleotide", "precomputed_filter_name", "precomputed_filter_threshold"},
    "anchor-test": {"action", "type", "fragment", "sequence", "exclude", "protein", "resid", "nucleotide", "dihedral", "angle", "margin", "nconformers", "conformer", "precomputed_filter_name", "precomputed_filter_threshold"},
    "anchor-refe": {"action", "type", "fragment", "sequence", "exclude", "nucleotide", "ovrmsd", "reference"},
    "grow": {"action", "type", "input", "restrict_input", "fragment", "sequence", "exclude", "direction", "crmsd", "ovrmsd"},
    "grow-test": {"action", "type", "input", "restrict_input", "fragment", "sequence", "exclude", "direction", "crmsd", "ovrmsd", "conformer"},
    "score": {"action", "type", "input", "sequence", "exclude", "protein", "nb_kernel"},
    "energy-filter": {"action", "type", "input", "fragment", "sequence", "exclude", "protein", "threshold", "epsilon", "cascade", "min_cascade"},
    "rmsd": {"action", "type", "input", "fragment", "exclude", "reference"},
    "mask": {"action", "type", "input", "score_input", "threshold"},
    "identity": {"action", "type", "input1", "input2"},
    "mask-common-conformer": {"action", "type", "input1", "input2"},
    "apply-mask": {"action", "type", "input", "mask_input", "mask"},
}

REQUIRED: dict[str, set[str]] = {
    "anchor": {"fragment", "sequence", "exclude", "protein", "resid", "nucleotide", "precomputed_filter_name", "precomputed_filter_threshold"},
    "anchor-test": {"fragment", "sequence", "exclude", "protein", "resid", "nucleotide", "precomputed_filter_name", "precomputed_filter_threshold"},
    "anchor-refe": {"fragment", "sequence", "exclude", "nucleotide", "ovrmsd"},
    "grow": {"input", "fragment", "sequence", "exclude", "direction", "crmsd", "ovrmsd"},
    "grow-test": {"input", "fragment", "sequence", "exclude", "direction", "crmsd", "ovrmsd", "conformer"},
    "score": {"input", "sequence", "exclude", "protein"},
    "energy-filter": {"input", "fragment", "sequence", "exclude", "protein", "threshold", "epsilon", "cascade", "min_cascade"},
    "rmsd": {"input", "fragment", "exclude"},
    "mask": {"input", "score_input", "threshold"},
    "identity": {"input1"},
    "mask-common-conformer": {"input1", "input2"},
    "apply-mask": {"input", "mask_input", "mask"},
}

DEPENDENCY_FIELDS: dict[str, dict[str, str]] = {
    "anchor": {},
    "anchor-test": {},
    "anchor-refe": {},
    "grow": {"input": POSE, "restrict_input": POSE},
    "grow-test": {"input": POSE, "restrict_input": POSE},
    "score": {"input": POSE},
    "energy-filter": {"input": POSE},
    "rmsd": {"input": POSE},
    "mask": {"input": POSE, "score_input": SCORE},
    "identity": {"input1": POSE, "input2": POSE},
    "mask-common-conformer": {"input1": POSE, "input2": POSE},
    "apply-mask": {"input": POSE, "mask_input": MASK},
}

OUTPUT_KIND = {
    "anchor": POSE,
    "anchor-test": POSE,
    "anchor-refe": POSE,
    "grow": POSE,
    "grow-test": POSE,
    "score": SCORE,
    "energy-filter": POSE,
    "rmsd": SCORE,
    "mask": MASK,
    "identity": POSE,
    # A paired conformer mask contains one selector for each input pool. apply-mask
    # consumes the selector matching the source pool and preserves its pose order.
    "mask-common-conformer": MASK,
    "apply-mask": POSE,
}

AUTO_FIELDS = {
    "anchor": {"fragment", "sequence", "exclude", "protein"},
    "anchor-test": {"fragment", "sequence", "exclude", "protein"},
    "anchor-refe": {"fragment", "sequence", "exclude", "ovrmsd"},
    "grow": {"fragment", "sequence", "exclude", "direction", "crmsd", "ovrmsd"},
    "grow-test": {"fragment", "sequence", "exclude", "direction", "crmsd", "ovrmsd"},
    "score": {"sequence", "exclude", "protein"},
    "energy-filter": {"fragment", "sequence", "exclude", "threshold", "cascade", "min_cascade"},
    "rmsd": {"fragment", "exclude"},
    "mask": {"threshold"},
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

    if action in {"anchor", "anchor-test", "anchor-refe"}:
        nuc = str(data.get("nucleotide")).lower()
        if nuc not in {"first", "second"}:
            raise SchemaError(f"{path}: nucleotide must be first or second")
        data["nucleotide"] = nuc
        if action in {"anchor", "anchor-test"}:
            filter_namespace = str(data["precomputed_filter_name"])
            filter_name = str(data["precomputed_filter_threshold"])
            spec = _gaussian_filter_spec()
            if filter_namespace != "gaussian-rrm":
                raise SchemaError(
                    f"{path}: precomputed_filter_name must be 'gaussian-rrm'"
                )
            if filter_name not in spec["thresholds"]:
                raise SchemaError(
                    f"{path}: unsupported Gaussian filter {filter_name!r}; "
                    "read the twelve legal names from A7"
                )
        if action == "anchor-test":
            has_nconformers = "nconformers" in data
            has_conformer = "conformer" in data
            if has_nconformers == has_conformer:
                raise SchemaError(f"{path}: anchor-test requires exactly one of nconformers or conformer")
            if has_nconformers:
                nconformers = int(data["nconformers"])
                if nconformers <= 0:
                    raise SchemaError(f"{path}: nconformers must be positive")
                data["nconformers"] = nconformers
            if has_conformer:
                conformer = int(data["conformer"])
                if conformer <= 0:
                    raise SchemaError(f"{path}: conformer must be positive")
                data["conformer"] = conformer
    if action in {"grow", "grow-test"} and data.get("direction") != "auto":
        direction = str(data["direction"]).lower()
        if direction in {"fwd", "forward"}:
            data["direction"] = "forward"
        elif direction in {"bwd", "backward"}:
            data["direction"] = "backward"
        else:
            raise SchemaError(f"{path}: direction must be fwd/forward/bwd/backward/auto")
    if action == "grow-test":
        conformer = int(data["conformer"])
        if conformer <= 0:
            raise SchemaError(f"{path}: conformer must be positive")
        data["conformer"] = conformer
    if action == "score":
        data.setdefault("nb_kernel", "compiled")
        if not isinstance(data["nb_kernel"], str) or data["nb_kernel"] not in {"compiled", "jax"}:
            raise SchemaError(f"{path}: nb_kernel must be compiled or jax")
    if action == "apply-mask":
        selector = data["mask"]
        if type(selector) is not int or selector not in {1, 2}:
            raise SchemaError(f"{path}: mask must be the integer 1 or 2")
    if action == "energy-filter":
        protein = data.get("protein")
        if not isinstance(protein, str) or re.fullmatch(r"range-\d+", protein) is None:
            raise SchemaError(f"{path}: energy-filter protein must name exactly one range-K")
        threshold = data.get("threshold")
        if threshold != "auto":
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
                raise SchemaError(f"{path}: energy-filter threshold must be numeric or auto")
            if not math.isfinite(float(threshold)):
                raise SchemaError(f"{path}: energy-filter threshold must be finite")
            data["threshold"] = float(threshold)
        epsilon = data.get("epsilon")
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or float(epsilon) != 15.0:
            raise SchemaError(f"{path}: energy-filter epsilon must be 15")
        data["epsilon"] = 15.0
        from ..energy_filter import _validate_minimum, _validate_steps

        if data["cascade"] != "auto":
            try:
                data["cascade"] = _validate_steps(data["cascade"])
            except ValueError as exc:
                raise SchemaError(f"{path}: {exc}") from exc
        if data["min_cascade"] != "auto":
            try:
                data["min_cascade"] = _validate_minimum(data["min_cascade"])
            except ValueError as exc:
                raise SchemaError(f"{path}: {exc}") from exc
    if action == "anchor-refe" and data.get("ovrmsd") != "auto":
        ovrmsd = float(data["ovrmsd"])
        if not math.isfinite(ovrmsd) or (ovrmsd <= 0 and ovrmsd != -1):
            raise SchemaError(f"{path}: ovrmsd must be positive")
        data["ovrmsd"] = ovrmsd
    if action in {"rmsd", "anchor-refe"}:
        data.setdefault("reference", "reference.pdb")
    for key, value in data.items():
        if value == "auto" and key not in AUTO_FIELDS.get(action, set()):
            raise SchemaError(f"{path}: {key}: auto is not supported for {action}")
    return ActionSpec(name=name, path=path, action=action, raw=data)


def load_action_spec(name: str, yaml_path: Path) -> ActionSpec:
    return normalize_action(name, yaml_path.parent, load_action_yaml(yaml_path))
