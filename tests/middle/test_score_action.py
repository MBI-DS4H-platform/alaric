from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import alaric.middle.deploy as deploy_module
from alaric.middle.deploy import deploy
from alaric.middle.errors import SchemaError
from alaric.middle.graph import ActionGraph, scored_pose_input
from alaric.middle.project import Project
from alaric.middle.resolve import ResolvedAction, resolve_action
from alaric.middle.schema import OUTPUT_KIND, normalize_action
from alaric.middle.sigil import compute_project_sigils, parameter_record
from alaric.score_concat import concatenate


def _write_action(root: Path, name: str, spec: dict) -> Path:
    path = root / name
    path.mkdir(parents=True)
    (path / "alaric.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    return path


def _write_project(root: Path) -> Path:
    data = root / "DATA"
    data.mkdir()
    (data / "constraints.json").write_text(
        '{"schema":1,"frag4":{"sequence":"GU"},"pairs":[]}\n'
    )
    (data / "pdbcode.txt").write_text("1ABC\n")
    (data / "reference.pdb").write_text("HEADER reference\n")
    (data / "dom-aa.pdb").write_text("HEADER all atom dom\n")
    (data / "protein-aa.pdb").write_text("HEADER all atom default\n")
    _write_action(
        root,
        "frag4-pose",
        {
            "action": "anchor-refe",
            "fragment": "auto",
            "sequence": "auto",
            "exclude": "auto",
            "nucleotide": "first",
            "ovrmsd": 0.75,
        },
    )
    _write_action(
        root,
        "frag4-score",
        {
            "action": "score",
            "input": "frag4-pose",
            "sequence": "auto",
            "exclude": "auto",
            "protein": "dom",
        },
    )
    _write_action(
        root,
        "frag4-mask",
        {
            "action": "mask",
            "input": "frag4-pose",
            "score_input": "frag4-score",
            "threshold": 0.0,
        },
    )
    return root / "frag4-score"


def test_score_schema_defaults_to_compiled_and_accepts_jax(tmp_path: Path) -> None:
    fields = {"action": "score", "input": "pose", "sequence": "auto", "exclude": "auto", "protein": "auto"}
    assert normalize_action("frag4-score", tmp_path, fields).raw["nb_kernel"] == "compiled"
    assert normalize_action("frag4-score", tmp_path, {**fields, "nb_kernel": "jax"}).raw["nb_kernel"] == "jax"
    with pytest.raises(SchemaError, match="nb_kernel"):
        normalize_action("frag4-score", tmp_path, {**fields, "nb_kernel": "numpy"})
    with pytest.raises(SchemaError, match="nb_kernel"):
        normalize_action("frag4-score", tmp_path, {**fields, "nb_kernel": ["jax"]})
    assert OUTPUT_KIND["score"] == "score"


def test_score_resolves_sequence_from_input_and_named_or_auto_protein(tmp_path: Path) -> None:
    _write_project(tmp_path)
    project = Project.discover(tmp_path)
    pose = ResolvedAction(
        "frag4-pose", tmp_path / "frag4-pose", "anchor-refe",
        {"fragment": 4, "sequence": "GU"}, {},
    )
    spec = normalize_action(
        "frag4-score", tmp_path / "frag4-score", {
            "action": "score", "input": "frag4-pose", "sequence": "auto",
            "exclude": "auto", "protein": "dom",
        }
    )
    resolved = resolve_action(project, spec, {pose.name: pose})
    assert resolved.params["sequence"] == "GU"
    assert resolved.params["exclude"] == ["1abc"]
    assert resolved.params["protein"] == "dom"
    assert resolved.params["__protein"] == "dom-aa.pdb"
    assert resolved.file_fields["protein"] == tmp_path / "DATA/dom-aa.pdb"

    auto_spec = normalize_action(
        "frag4-score", tmp_path / "frag4-score", {
            "action": "score", "input": "frag4-pose", "sequence": "auto",
            "exclude": "auto", "protein": "auto",
        }
    )
    auto = resolve_action(project, auto_spec, {pose.name: pose})
    assert auto.params["protein"] == "protein"
    assert auto.params["__protein"] == "protein-aa.pdb"


def test_score_is_a_mask_score_for_its_underlying_pose_and_versions_its_cache(tmp_path: Path) -> None:
    score_dir = _write_project(tmp_path)
    project = Project.discover(tmp_path)
    graph = ActionGraph(project)
    resolved = graph.build()
    score = resolved["frag4-score"]
    assert scored_pose_input(score).name == "frag4-pose"
    assert resolved["frag4-mask"].params["score_input"].action == "score"
    assert resolved["frag4-mask"].params["threshold"] == 0.0
    assert parameter_record(score, {"input": "pose-sigil"})["score_runtime_version"] == 1


@pytest.mark.parametrize("deployer", ["local", "local-chunk", "remote", "remote-chunk"])
def test_score_deployment_renders_each_supported_deployer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deployer: str
) -> None:
    score_dir = _write_project(tmp_path)
    remote = deployer.startswith("remote")
    if remote:
        monkeypatch.setenv("ALARIC_REMOTE_HOST", "cluster")
        monkeypatch.setenv("ALARIC_REMOTE_DEPLOYMENT_DIR", "/remote/deploy")
        monkeypatch.setenv("ALARIC_REMOTE_ALARIC_DIR", "/remote/alaric")
        monkeypatch.setenv("ALARIC_REMOTE_RESULT_DIR", "/remote/results")
        monkeypatch.setenv("ALARIC_REMOTE_SCRATCH_DIR", "/scratch/alaric")
        calls: list[list[str]] = []

        def fake_run(args, check=False, **kwargs):
            calls.append(list(args))
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(deploy_module.subprocess, "run", fake_run)

    deploy(deployer, score_dir, nchunks=3 if deployer.endswith("-chunk") else None)
    for script in score_dir.glob("*.sh"):
        subprocess.run(["bash", "-n", str(script)], check=True)
    run = (score_dir / "run.sh").read_text()
    if deployer.endswith("-chunk"):
        rendered_action = (score_dir / "chunk1.sh").read_text()
        organize = (score_dir / "organize.sh").read_text()
        assert "score.sh" in rendered_action
        assert "-x 1abc" in rendered_action
        assert "dom-aa.pdb" in rendered_action
        assert " compiled " in rendered_action
        assert "score_concat.py" in organize
        assert '--nposes "$TOTAL" --nchunks 3' in organize
        assert "np.empty((0,), dtype=np.float32)" in rendered_action
        assert "result_sidecar.py array" not in organize
        if remote:
            assert "mv " in organize
        else:
            assert "score.npy.CHECKSUM ../CACHE/checksum/" in organize
        if remote:
            assert "ALARIC_SCORE_CHUNKS_DIR" in run
            assert "/scratch/alaric" in run
    else:
        assert "score.sh" in run
        assert "-x 1abc" in run
        assert "dom-aa.pdb" in run
        assert " compiled " in run
        assert "score_concat.py" not in run
    if remote:
        assert any(call[0] == "scp" for call in calls)
        assert any("/remote/deploy/SIGIL/" in " ".join(call) for call in calls)


def test_score_concat_preserves_chunk_order_float32_and_validates_pool(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks"
    (chunks / "chunk-1").mkdir(parents=True)
    (chunks / "chunk-2").mkdir()
    np.save(chunks / "chunk-2" / "score.npy", np.array([3.5, 4.5], dtype=np.float32))
    np.save(chunks / "chunk-1" / "score.npy", np.array([1.5, 2.5], dtype=np.float32))
    output = tmp_path / "result" / "score.npy"

    checksum = concatenate(chunks, output, nchunks=2, nposes=4, block=1)
    result = np.load(output)
    assert result.tolist() == [1.5, 2.5, 3.5, 4.5]
    assert result.dtype == np.float32
    assert checksum == (output.with_name("score.npy.CHECKSUM")).read_text().strip()
    assert checksum == hashlib.sha256(output.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="cover 4 poses, expected 5"):
        concatenate(chunks, tmp_path / "wrong-count.npy", nchunks=2, nposes=5)

    np.save(chunks / "chunk-2" / "score.npy", np.array([3.5, 4.5], dtype=np.float64))
    with pytest.raises(ValueError, match="inconsistent score dtype"):
        concatenate(chunks, tmp_path / "wrong-dtype.npy", nchunks=2)


def test_score_shell_forwards_exclusions_to_pose_conversion(tmp_path: Path) -> None:
    poses = tmp_path / "poses"
    poses.mkdir()
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("HEADER\n")
    wrapper = tmp_path / "python-wrapper"
    log = tmp_path / "calls.jsonl"
    wrapper.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "import numpy as np\n"
        "args = sys.argv[1:]\n"
        "with open(os.environ['ALARIC_SCORE_TEST_LOG'], 'a') as handle:\n"
        "    handle.write(json.dumps(args) + '\\n')\n"
        "script = next((arg for arg in args if arg.endswith('.py')), '')\n"
        "if script.endswith('/minfor.py'):\n"
        "    output = args[args.index('--output-npy') + 1]\n"
        "    np.save(output, np.array([1.25], dtype=np.float32))\n"
    )
    wrapper.chmod(0o755)
    output = tmp_path / "score.npy"
    subprocess.run(
        [
            "bash", str(ROOT / "alaric/score.sh"), "-x", "1b7f", "-x", "2abc",
            str(poses), "GU", str(receptor), "jax", str(output),
        ],
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHON": str(wrapper), "ALARIC_SCORE_TEST_LOG": str(log)},
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    convert = next(args for args in calls if any(arg.endswith("convert_poses.py") for arg in args))
    assert convert.count("--pdb-exclude") == 2
    assert [convert[convert.index("--pdb-exclude") + 1], convert[convert.index("--pdb-exclude") + 3]] == ["1b7f", "2abc"]
    assert np.load(output).dtype == np.float32
