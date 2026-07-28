from __future__ import annotations

from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from alaric.middle.deploy import deploy, generate_run_sh
from alaric.middle.errors import ResolveError, SchemaError
from alaric.middle.graph import ActionGraph
from alaric.middle.pool_graph import build_pool_graph
from alaric.middle.project import Project
from alaric.middle.schema import normalize_action
from alaric.middle.sigil import compute_project_sigils

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_middle_core import _write_project  # noqa: E402


def _write_anchor_refe_project(root: Path, **overrides) -> Path:
    _write_project(root)
    data = root / "DATA"
    (data / "reference.pdb").write_text("HEADER test reference\n")
    (data / "anchor.yaml").write_text("angle: 30\ndihedral: 45 -45\novrmsd: 0.75\n")
    spec = {
        "action": "anchor-refe",
        "fragment": "auto",
        "sequence": "auto",
        "exclude": "auto",
        "nucleotide": "second",
        "ovrmsd": 0.6,
    }
    spec.update(overrides)
    action_dir = root / "frag5-anchor-refe"
    action_dir.mkdir()
    (action_dir / "alaric.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    return action_dir


def test_anchor_refe_resolves_auto_fields(tmp_path: Path) -> None:
    _write_anchor_refe_project(tmp_path, ovrmsd="auto")
    project = Project.discover(tmp_path)
    action = ActionGraph(project).build()["frag5-anchor-refe"]

    assert action.params["fragment"] == 5
    assert action.params["sequence"] == "UU"
    assert action.params["exclude"] == ["1abc"]
    assert action.params["ovrmsd"] == 0.75  # from DATA/anchor.yaml
    assert action.params["reference"] == "reference.pdb"
    assert action.file_fields["reference"] == tmp_path / "DATA" / "reference.pdb"


def test_anchor_refe_is_a_source_pool_with_a_sigil(tmp_path: Path) -> None:
    _write_anchor_refe_project(tmp_path)
    project = Project.discover(tmp_path)
    sigils = compute_project_sigils(project)

    assert sigils["frag5-anchor-refe"].endswith("-null")  # no dependencies
    assert (tmp_path / "CACHE" / "parameters" / sigils["frag5-anchor-refe"]).is_file()

    pools = build_pool_graph(tmp_path).pools
    assert pools["frag5-anchor-refe"].kind == "anchor"
    assert pools["frag5-anchor-refe"].parents == []


def test_anchor_refe_local_deploy_renders_invocation(tmp_path: Path) -> None:
    _write_anchor_refe_project(tmp_path)
    project = Project.discover(tmp_path)
    compute_project_sigils(project)

    body = generate_run_sh(
        project, ActionGraph(project).build()["frag5-anchor-refe"], "local"
    )

    assert "anchor_refe.py" in body
    assert "--reference ../DATA/reference.pdb" in body
    assert "--fragment 5" in body
    assert "--sequence UU" in body
    assert "--second" in body
    assert "--ov-rmsd 0.6" in body
    assert "--pdb-exclude 1abc" in body
    assert "organize.py" in body


def test_anchor_refe_chunk_deploy_splits_over_conformers(tmp_path: Path) -> None:
    action_dir = _write_anchor_refe_project(tmp_path)
    deploy("local-chunk", action_dir, nchunks=3)

    assert (action_dir / "chunk3.sh").is_file()
    assert (action_dir / "organize.sh").is_file()
    body = (action_dir / "chunk3.sh").read_text()
    assert "NCHUNKS=3" in body
    assert "IDX=3" in body
    assert '--conformer-range "$FIRST" "$LAST"' in body
    assert "anchor_refe.py" in body
    assert "organize.py" not in body  # organize lives in organize.sh
    assert "organize.py" in (action_dir / "organize.sh").read_text()


def test_anchor_refe_rejects_bad_nucleotide_and_ovrmsd(tmp_path: Path) -> None:
    base = {
        "action": "anchor-refe",
        "fragment": 5,
        "sequence": "UU",
        "exclude": "1abc",
        "nucleotide": "third",
        "ovrmsd": 0.6,
    }
    with pytest.raises(SchemaError, match="nucleotide must be first or second"):
        normalize_action("frag5-anchor-refe", tmp_path, base)

    with pytest.raises(SchemaError, match="ovrmsd must be positive"):
        normalize_action(
            "frag5-anchor-refe", tmp_path, {**base, "nucleotide": "first", "ovrmsd": 0}
        )

    with pytest.raises(SchemaError, match="unknown keys"):
        normalize_action(
            "frag5-anchor-refe",
            tmp_path,
            {**base, "nucleotide": "first", "crmsd": 0.25},
        )


def test_anchor_refe_requires_the_reference_file(tmp_path: Path) -> None:
    _write_anchor_refe_project(tmp_path)
    (tmp_path / "DATA" / "reference.pdb").unlink()
    project = Project.discover(tmp_path)

    with pytest.raises(ResolveError, match="reference file not found"):
        ActionGraph(project).build()
