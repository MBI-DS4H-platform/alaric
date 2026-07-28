from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .errors import ResolveError
from .jsonutil import read_json
from .project import Project
from .schema import ActionSpec


FRAG_RE = re.compile(r"(?:^|-)frag(\d+)(?:-|$)")


@dataclass(frozen=True)
class ResolvedAction:
    name: str
    path: Path
    action: str
    params: dict[str, Any]
    file_fields: dict[str, Path]


def _fragment_from_name(name: str) -> int:
    match = FRAG_RE.search(name)
    if not match:
        raise ResolveError(f"{name}: cannot resolve fragment:auto from action dir name")
    return int(match.group(1))


def _constraints(project: Project) -> dict[str, Any]:
    path = project.data_dir / "constraints.json"
    return read_json(path) if path.is_file() else {}


def _sequence_from_reference(project: Project, fragment: int) -> str | None:
    path = project.data_dir / "reference.pdb"
    if not path.is_file():
        return None
    residues: list[tuple[str, int]] = []
    seen = set()
    mapping = {"A": "A", "C": "C", "G": "G", "U": "U", "RA": "A", "RC": "C", "RG": "G", "RU": "U"}
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        resname = line[17:20].strip().upper()
        resid_text = line[22:26].strip()
        try:
            resid = int(resid_text)
        except ValueError:
            continue
        key = (resid, resname)
        if key in seen:
            continue
        seen.add(key)
        if resname in mapping:
            residues.append((mapping[resname], resid))
    residues = sorted(residues, key=lambda item: item[1])
    idx = fragment - 1
    if idx < 0 or idx + 1 >= len(residues):
        return None
    return residues[idx][0] + residues[idx + 1][0]


def resolve_fragment(value: Any, name: str) -> int:
    return _fragment_from_name(name) if value == "auto" else int(value)


def resolve_sequence(project: Project, fragment: int) -> str:
    constraints = _constraints(project)
    frag_key = f"frag{fragment}"
    try:
        return str(constraints[frag_key]["sequence"]).upper()
    except KeyError:
        pass
    seq_file = project.data_dir / "sequence.txt"
    if seq_file.is_file():
        text = "".join(seq_file.read_text().split()).upper()
        idx = fragment - 1
        if idx >= 0 and idx + 1 < len(text):
            return text[idx : idx + 2]
    from_ref = _sequence_from_reference(project, fragment)
    if from_ref:
        return from_ref.upper()
    raise ResolveError(f"frag{fragment}: cannot resolve sequence:auto")


def resolve_exclude(project: Project, value: Any) -> list[str]:
    if value != "auto":
        if isinstance(value, (list, tuple)):
            return [str(v).lower() for v in value]
        return [str(value).lower()]
    pdbcode = project.data_dir / "pdbcode.txt"
    if pdbcode.is_file():
        return [pdbcode.read_text().strip().lower()]
    constraints = _constraints(project)
    if "pdb_code" in constraints:
        return [str(constraints["pdb_code"]).lower()]
    raise ResolveError("cannot resolve exclude:auto")


def resolve_protein(project: Project, value: Any, *, all_atom: bool) -> tuple[str, Path]:
    """Resolve a protein name to its DATA file.

    ``all_atom=True`` selects ``<name>-aa.pdb`` (all heavy atoms, for ATTRACT scoring);
    ``all_atom=False`` selects the plain ``<name>.pdb`` (original residue numbering, which
    anchor's ``--resid`` refers to). The required file must exist — there is deliberately no
    fallback to the other variant, since they carry different residue numbering.
    """
    name = "protein" if value == "auto" else str(value)
    suffix = "-aa.pdb" if all_atom else ".pdb"
    path = project.data_dir / f"{name}{suffix}"
    if not path.is_file():
        raise ResolveError(f"protein file not found: {path}")
    return name, path


def _read_anchor_yaml(project: Project, key: str) -> Any:
    path = project.data_dir / "anchor.yaml"
    if not path.is_file():
        raise ResolveError(f"cannot resolve {key}:auto: {path} missing")
    data = yaml.safe_load(path.read_text()) or {}
    if key not in data:
        raise ResolveError(f"cannot resolve {key}:auto: missing {key} in {path}")
    return data[key]


def resolve_direction(value: Any, name: str) -> str:
    if value != "auto":
        return str(value)
    if name.endswith("-fwd"):
        return "forward"
    if name.endswith("-bwd"):
        return "backward"
    raise ResolveError(f"{name}: cannot resolve direction:auto")


def _pair_thresholds(project: Project, source_fragment: int, target_fragment: int) -> tuple[float, float]:
    if abs(source_fragment - target_fragment) != 1:
        raise ResolveError("grow crmsd:auto/ovrmsd:auto requires adjacent fragments")
    down, up = sorted((source_fragment, target_fragment))
    constraints = _constraints(project)
    pairs = constraints.get("pairs", [])
    def frag_num(value: Any) -> int:
        text = str(value)
        if text.startswith("frag"):
            text = text[4:]
        return int(text)

    for pair in pairs:
        if frag_num(pair.get("down", -1)) == down and frag_num(pair.get("up", -1)) == up:
            try:
                crmsd = pair.get("crmsd", pair.get("cRMSD"))
                ovrmsd = pair.get("ovrmsd", pair.get("ovRMSD"))
                if crmsd is None or ovrmsd is None:
                    raise KeyError("crmsd/ovrmsd")
                return float(crmsd), float(ovrmsd)
            except KeyError as exc:
                raise ResolveError(f"pair {down}->{up} missing {exc.args[0]}") from exc
    raise ResolveError(f"constraints.json has no pair thresholds for {down}->{up}")


def _resolve_precomputed_filter_directory(project: Project, filter_name: str, threshold: str) -> Path:
    """Resolve precomputed filter name and threshold to the actual directory path."""
    # Assume precomputed-filters is a sibling of the project or in the alaric repo
    candidate_paths = [
        project.root / "precomputed-filters" / filter_name / threshold,
        Path(__file__).resolve().parents[2] / "precomputed-filters" / filter_name / threshold,
    ]
    for path in candidate_paths:
        if path.is_dir():
            return path
    raise ResolveError(
        f"cannot locate precomputed filter directory for {filter_name}/{threshold}; "
        f"tried: {', '.join(str(p) for p in candidate_paths)}"
    )


def _resolve_ring_reference_file(alaric_root: Path, ring_type: str, ring_code: str) -> Path:
    """Resolve ring reference file (.npy) for a given ring type and code."""
    candidate_paths = [
        alaric_root / "ring-refe" / f"refe-{ring_code}.npy",
        Path(__file__).resolve().parents[2] / "ring-refe" / f"refe-{ring_code}.npy",
    ]
    for path in candidate_paths:
        if path.is_file():
            return path
    raise ResolveError(
        f"cannot locate {ring_type} ring reference file for {ring_code}; "
        f"tried: {', '.join(str(p) for p in candidate_paths)}"
    )


def resolve_action(
    project: Project,
    spec: ActionSpec,
    resolved_by_name: dict[str, ResolvedAction] | None = None,
) -> ResolvedAction:
    resolved_by_name = resolved_by_name or {}
    p = dict(spec.raw)
    files: dict[str, Path] = {}
    action = spec.action
    if "fragment" in p:
        p["fragment"] = resolve_fragment(p["fragment"], spec.name)
    if action in {"anchor", "anchor-test", "anchor-refe", "grow", "grow-test"} and p.get("sequence") == "auto":
        p["sequence"] = resolve_sequence(project, int(p["fragment"]))
    if action == "score" and p.get("sequence") == "auto":
        dep = resolved_by_name.get(str(p["input"]))
        if dep is None:
            raise ResolveError(f"{spec.name}: score sequence:auto needs resolved input")
        p["sequence"] = final_sequence(dep)
    if p.get("exclude") == "auto" or "exclude" in p:
        p["exclude"] = resolve_exclude(project, p["exclude"])
    if "protein" in p:
        # anchor stacks on a residue selected by --resid in the ORIGINAL numbering, so it
        # needs the plain <name>.pdb; scoring needs the all-atom <name>-aa.pdb. No fallback.
        all_atom = action not in {"anchor", "anchor-test"}
        protein, path = resolve_protein(project, p["protein"], all_atom=all_atom)
        p["protein"] = protein
        p["__protein"] = path.name
        files["protein"] = path
    if action in {"anchor", "anchor-test"}:
        # Handle precomputed filter parameters: both must be present or both must be absent
        has_filter_name = "precomputed_filter_name" in p
        has_filter_threshold = "precomputed_filter_threshold" in p
        if has_filter_name != has_filter_threshold:
            raise ResolveError(
                "precomputed_filter_name and precomputed_filter_threshold must both be present or both absent"
            )
        if has_filter_name and has_filter_threshold:
            filter_name = str(p["precomputed_filter_name"])
            threshold = str(p["precomputed_filter_threshold"])
            filter_dir = _resolve_precomputed_filter_directory(project, filter_name, threshold)
            p["__precomputed_filter_dir"] = str(filter_dir)
            # Reference files for the nucleotide (always A for dinucleotide base pair)
            nuc_ref = _resolve_ring_reference_file(project.root, "nucleotide", "A")
            p["__anchor_reference_ring"] = nuc_ref.name
            files["anchor_reference_ring"] = nuc_ref
        else:
            # Without precomputed filters, angle and dihedral are required
            if p.get("dihedral") == "auto" or "dihedral" not in p:
                p["dihedral"] = _read_anchor_yaml(project, "dihedral")
            if p.get("angle") == "auto" or "angle" not in p:
                p["angle"] = _read_anchor_yaml(project, "angle")
    if action in {"grow", "grow-test"}:
        p["direction"] = resolve_direction(p["direction"], spec.name)
        dep = resolved_by_name.get(str(p["input"]))
        if dep is None:
            raise ResolveError(f"{spec.name}: grow auto thresholds need resolved input")
        source_fragment = final_fragment(dep)
        if p.get("crmsd") == "auto" or p.get("ovrmsd") == "auto":
            crmsd, ovrmsd = _pair_thresholds(project, source_fragment, int(p["fragment"]))
            if p.get("crmsd") == "auto":
                p["crmsd"] = crmsd
            if p.get("ovrmsd") == "auto":
                p["ovrmsd"] = ovrmsd
        if p.get("sequence") != "auto":
            p["source_sequence"] = final_sequence(dep)
    if action == "anchor-refe" and p.get("ovrmsd") == "auto":
        p["ovrmsd"] = float(_read_anchor_yaml(project, "ovrmsd"))
    if action in {"rmsd", "anchor-refe"}:
        reference = str(p.get("reference", "reference.pdb"))
        ref_path = Path(reference)
        if not ref_path.is_absolute():
            ref_path = project.data_dir / reference
        if not ref_path.is_file():
            raise ResolveError(f"reference file not found: {ref_path}")
        p["reference"] = ref_path.name
        p["__reference"] = ref_path.name
        files["reference"] = ref_path
    return ResolvedAction(spec.name, spec.path, action, p, files)


def final_fragment(action: ResolvedAction) -> int:
    if "fragment" in action.params:
        return int(action.params["fragment"])
    for key in ("input", "input1", "input2"):
        if key in action.params and isinstance(action.params[key], ResolvedAction):
            return final_fragment(action.params[key])
    raise ResolveError(f"{action.name}: cannot infer final fragment")


def final_sequence(action: ResolvedAction) -> str:
    if "sequence" in action.params:
        return str(action.params["sequence"]).upper()
    for key in ("input", "input1", "input2"):
        dep = action.params.get(key)
        if isinstance(dep, ResolvedAction):
            return final_sequence(dep)
    raise ResolveError(f"{action.name}: cannot infer final sequence")
