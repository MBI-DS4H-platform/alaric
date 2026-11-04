from __future__ import annotations

import os
from pathlib import Path
import subprocess
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


@pytest.mark.parametrize('threshold', [0, -0.01, -2])
def test_disabled_control_exception_admits_no_other_nonpositive_threshold(tmp_path, threshold):
    with pytest.raises(SchemaError):
        normalize_action('frag5-anchor-refe', tmp_path, {
            'action': 'anchor-refe', 'fragment': 5, 'sequence': 'UU',
            'exclude': '1abc', 'nucleotide': 'first', 'ovrmsd': threshold,
        })


def test_disabled_positive_control_resolves_and_deploys_without_blocking_siblings(tmp_path):
    sentinel_dir = _write_anchor_refe_project(tmp_path, ovrmsd=-1)
    (tmp_path / "DATA" / "reference.pdb").write_bytes(
        (ROOT / "tests" / "1b7f" / "pdbs" / "rna.pdb").read_bytes()
    )
    healthy_dir = tmp_path / "frag5-anchor-refe-healthy"
    healthy_dir.mkdir()
    (healthy_dir / "alaric.yaml").write_text(yaml.safe_dump({
        "action": "anchor-refe",
        "fragment": "auto",
        "sequence": "auto",
        "exclude": "auto",
        "nucleotide": "second",
        "ovrmsd": 0.6,
    }, sort_keys=False))
    project = Project.discover(tmp_path)
    actions = ActionGraph(project).build()
    assert actions['frag5-anchor-refe'].params['ovrmsd'] == -1
    assert actions['frag5-anchor-refe-healthy'].params['ovrmsd'] == 0.6
    compute_project_sigils(project)
    for action in actions.values():
        deploy('local', action.path)
    assert '--ov-rmsd -1' in (sentinel_dir / 'run.sh').read_text()
    assert (tmp_path / 'frag4-anchor' / 'run.sh').exists()
    assert (tmp_path / 'frag4-filter' / 'run.sh').exists()
    assert (tmp_path / 'frag4-rmsd' / 'run.sh').exists()
    assert (tmp_path / 'frag5-fwd' / 'run.sh').exists()
    from alaric.anchor_constants import validate_ovrmsd
    with pytest.raises(ValueError, match='ovRMSD threshold must be finite and positive'):
        validate_ovrmsd(actions['frag5-anchor-refe'].params['ovrmsd'])
    assert validate_ovrmsd(actions['frag4-anchor'].params['ovrmsd']) == 0.75

    env = os.environ.copy()
    env['ALARIC_DIR'] = str(ROOT / 'alaric')
    env['ALARIC_ANCHOR_REFE_EXTRA_ARGS'] = '--conformer 1'
    disabled = subprocess.run(
        ['bash', str(sentinel_dir / 'run.sh')], cwd=sentinel_dir, env=env,
        capture_output=True, text=True,
    )
    assert disabled.returncode != 0
    assert 'ovRMSD threshold must be finite and positive' in disabled.stderr

    healthy = subprocess.run(
        ['bash', str(healthy_dir / 'run.sh')], cwd=healthy_dir, env=env,
        capture_output=True, text=True,
    )
    assert healthy.returncode == 0, healthy.stderr
    assert 'total solutions' in healthy.stderr


def test_disabled_anchor_refe_fails_at_runtime_threshold_validation(tmp_path):
    import subprocess
    reference = tmp_path / 'reference.pdb'
    reference.write_text('HEADER threshold must be checked before coordinates\n')
    result = subprocess.run([sys.executable, str(ROOT / 'alaric' / 'anchor_refe.py'),
        '--reference', str(reference), '--fragment', '1', '--first',
        '--ov-rmsd', '-1', '--output', str(tmp_path / 'poses')], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'ovRMSD threshold must be finite and positive' in result.stderr
