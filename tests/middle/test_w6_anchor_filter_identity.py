"""W6 runtime Gaussian names and semantic cache identity in anchor sigils."""
from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alaric.gaussian_filter import all_names, load_spec  # noqa: E402
from alaric.middle import resolve as resolve_module  # noqa: E402
from alaric.middle import schema as schema_module  # noqa: E402
from alaric.middle.errors import SchemaError  # noqa: E402
from alaric.middle.project import Project  # noqa: E402
from alaric.middle.schema import normalize_action  # noqa: E402
from alaric.middle.sigil import compute_project_sigils  # noqa: E402


def _anchor_data(filter_name: str) -> dict:
    return {
        "action": "anchor",
        "fragment": 4,
        "sequence": "GU",
        "exclude": "auto",
        "protein": "dom",
        "resid": 256,
        "nucleotide": "first",
        "precomputed_filter_name": "gaussian",
        "precomputed_filter_threshold": filter_name,
    }


@pytest.mark.parametrize("name", all_names())
def test_anchor_schema_accepts_each_a7_name(name: str, tmp_path: Path) -> None:
    spec = normalize_action("frag4-anchor", tmp_path / "alaric.yaml", _anchor_data(name))
    assert spec.raw["precomputed_filter_threshold"] == name


def test_anchor_schema_rejects_names_outside_a7(tmp_path: Path) -> None:
    with pytest.raises(SchemaError, match="unsupported Gaussian filter"):
        normalize_action("frag4-anchor", tmp_path / "alaric.yaml", _anchor_data("rrm999"))


def _make_anchor_project(root: Path) -> Project:
    data = root / "DATA"
    data.mkdir(parents=True)
    (data / "pdbcode.txt").write_text("1b7f\n")
    # The original residue numbering is preserved; residue 256 is a PHE ring.
    (data / "dom.pdb").write_bytes((ROOT / "tests/1b7f/pdbs/1b7f_dom2.pdb").read_bytes())
    action = root / "frag4-anchor"
    action.mkdir()
    (action / "alaric.yaml").write_text(yaml.safe_dump(_anchor_data("rrm50"), sort_keys=False))
    return Project.discover(root)


def test_anchor_sigil_uses_semantic_array_identity_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _make_anchor_project(tmp_path)
    canonical = load_spec(ROOT / "alaric" / "gaussian_stacking_filter.json")

    def sigil_for(spec: dict) -> str:
        monkeypatch.setattr(schema_module, "_gaussian_filter_spec", lambda: spec)
        monkeypatch.setattr(resolve_module, "_gaussian_filter_spec", lambda: spec)
        return compute_project_sigils(project, force=True)["frag4-anchor"]

    original = copy.deepcopy(canonical)
    original_sigil = sigil_for(original)

    recompressed = copy.deepcopy(canonical)
    for file_entry in recompressed["thresholds"]["rrm50"]["precalc"]["files"].values():
        file_entry["sha256"] = "0" * 64
    assert sigil_for(recompressed) == original_sigil

    changed_array = copy.deepcopy(canonical)
    first_file = next(iter(changed_array["thresholds"]["rrm50"]["precalc"]["files"]))
    changed_array["thresholds"]["rrm50"]["precalc"]["files"][first_file]["npy_sha256"] = "f" * 64
    assert sigil_for(changed_array) != original_sigil

    changed_spec = copy.deepcopy(canonical)
    changed_spec["spec_digest"] = "f" * 64
    assert sigil_for(changed_spec) != original_sigil
