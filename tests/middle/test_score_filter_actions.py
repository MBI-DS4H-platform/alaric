from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "alaric"))

from alaric.middle.deploy import deploy, render_action_template
from alaric.middle.errors import GraphError
from alaric.middle.graph import ActionGraph, scored_pose_input
from alaric.middle.project import Project
from alaric.middle.sigil import compute_project_sigils
from alaric.npy_io import load_npy
from alaric.organize import organize_pose_dir
from alaric.poses import PoseReader, PoseWriter, select_pose_indices
from alaric.score_add import add_score_arrays


AL_ENV_BIN = Path(sys.executable).parent


def _write_action(root: Path, name: str, spec: dict) -> Path:
    path = root / name
    path.mkdir(parents=True)
    (path / "alaric.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    return path


def _write_project(root: Path) -> Project:
    data = root / "DATA"
    data.mkdir()
    (data / "constraints.json").write_text(
        json.dumps({"schema": 1, "frag4": {"sequence": "GU"}, "pairs": []})
    )
    (data / "pdbcode.txt").write_text("1ABC\n")
    (data / "reference.pdb").write_text("HEADER reference\n")
    for name in ("protein", "dom", "dom2"):
        (data / f"{name}-aa.pdb").write_text(f"HEADER {name}\n")

    pose_spec = {
        "action": "anchor-refe", "fragment": 4, "sequence": "GU",
        "exclude": [], "nucleotide": "first", "ovrmsd": 0.75,
    }
    _write_action(root, "frag4-pose", pose_spec)
    _write_action(root, "frag4-pose2", pose_spec)
    score_spec = {
        "action": "score", "input": "frag4-pose", "sequence": "auto",
        "exclude": "auto", "protein": "dom",
    }
    _write_action(root, "frag4-score1", score_spec)
    _write_action(root, "frag4-score2", {**score_spec, "protein": "dom2"})
    _write_action(root, "frag4-score-add", {
        "action": "score_add", "score_input1": "frag4-score1",
        "score_input2": "frag4-score2",
    })
    _write_action(root, "frag4-mask", {
        "action": "mask", "input": "frag4-pose",
        "score_input": "frag4-score1", "threshold": 0.0,
    })
    _write_action(root, "frag4-pair-mask", {
        "action": "mask-common-conformer", "input1": "frag4-pose",
        "input2": "frag4-pose2",
    })
    _write_action(root, "frag4-filter-score", {
        "action": "filter", "input": "frag4-pose",
        "score_input": "frag4-score-add", "threshold": 0.0,
    })
    _write_action(root, "frag4-filter-mask", {
        "action": "filter", "input": "frag4-pose", "mask_input": "frag4-mask",
    })
    _write_action(root, "frag4-filter-pair", {
        "action": "filter", "input": "frag4-pose2",
        "mask_input": "frag4-pair-mask", "mask": 2,
    })
    return Project.discover(root)


def _write_pose_pool(path: Path) -> None:
    writer = PoseWriter(path, bucket_size=16, cache_poses=16)
    writer.add_chunk(
        np.array([0, 0, 1, 1], dtype=np.uint16),
        np.array([0, 1, 0, 1], dtype=np.uint16),
        np.array([[0, 0, 0], [1, 0, 0], [17, 0, 0], [33, 0, 0]], dtype=np.int32),
    )
    writer.finish()
    organize_pose_dir(path, nprocs=1)


def _materialize(project: Project, action, *, score_values=None, pose=False) -> Path:
    sigil = (action.path / "sigil.txt").read_text().strip()
    result = project.results_dir / sigil
    result.mkdir(parents=True, exist_ok=True)
    if pose:
        _write_pose_pool(result)
    if score_values is not None:
        np.save(result / "score.npy", np.asarray(score_values, dtype=np.float32))
    action.path.joinpath("result.txt").write_text("materialized\n")
    return result


def _shell_check(tmp_path: Path, text: str) -> None:
    path = tmp_path / "rendered.sh"
    path.write_text(text + "\n")
    subprocess.run(["bash", "-n", str(path)], check=True)


def test_score_add_streams_compressed_inputs_and_checks_shape(tmp_path: Path, monkeypatch) -> None:
    from alaric import score_add
    from alaric.npy_io import save_npy

    left = np.arange(13, dtype=np.float32)
    right = np.arange(13, dtype=np.float32)[::-1]
    save_npy(tmp_path / "left.npy", left)
    np.save(tmp_path / "right.npy", right)
    monkeypatch.setattr(score_add, "COPY_CHUNK", 3)

    output = add_score_arrays(tmp_path / "left.npy", tmp_path / "right.npy", tmp_path / "out/score.npy")

    assert np.load(output).tolist() == (left + right).tolist()
    assert np.load(output).dtype == np.float32
    np.save(tmp_path / "short.npy", np.zeros(12, dtype=np.float32))
    with pytest.raises(ValueError, match="shape mismatch"):
        add_score_arrays(tmp_path / "left.npy", tmp_path / "short.npy", tmp_path / "bad.npy")


def test_action_graph_accepts_score_add_and_both_filter_routes(tmp_path: Path) -> None:
    project = _write_project(tmp_path)
    resolved = ActionGraph(project).build()

    assert scored_pose_input(resolved["frag4-score-add"]).name == "frag4-pose"
    assert resolved["frag4-filter-score"].params["score_input"].action == "score_add"
    assert resolved["frag4-filter-mask"].params["mask_input"].action == "mask"
    assert resolved["frag4-filter-pair"].params["mask_input"].action == "mask-common-conformer"


def test_score_add_rejects_scores_from_different_pose_inputs(tmp_path: Path) -> None:
    project = _write_project(tmp_path)
    _write_action(tmp_path, "frag4-score-wrong-pose", {
        "action": "score", "input": "frag4-pose2", "sequence": "auto",
        "exclude": "auto", "protein": "dom2",
    })
    (tmp_path / "frag4-score-add" / "alaric.yaml").write_text(yaml.safe_dump({
        "action": "score_add", "score_input1": "frag4-score1",
        "score_input2": "frag4-score-wrong-pose",
    }))

    with pytest.raises(GraphError, match="same pose input"):
        ActionGraph(project).build()


@pytest.mark.parametrize("deployer", ["local", "local-chunk", "remote", "remote-chunk"])
@pytest.mark.parametrize("action_name", [
    "frag4-score-add", "frag4-filter-score", "frag4-filter-mask", "frag4-filter-pair",
])
def test_restored_action_templates_render_and_pass_bash_syntax(tmp_path: Path, monkeypatch, deployer: str, action_name: str) -> None:
    project = _write_project(tmp_path)
    resolved = ActionGraph(project).build()
    compute_project_sigils(project)
    action = resolved[action_name]
    location = "remote" if deployer.startswith("remote") else "local"

    lines = render_action_template(
        project, action, deployer=deployer,
        alaric_dir="/remote/alaric" if location == "remote" else str(ROOT / "alaric"),
        output_dir="../CACHE/results/output",
        location=location,
    )
    body = "\n".join(lines)
    assert "{{ " not in body
    _shell_check(tmp_path, body)
    if action_name == "frag4-score-add":
        assert "score_add.py" in body
        assert "score.npy" in body
    elif "filter-score" in action_name:
        assert "filter-poses.py" in body
        assert " 0.0 " in body
    else:
        assert "select-poses.py" in body
        assert "--provenance" in body
    if location == "remote":
        assert "ALARIC_REMOTE_RESULT_DIR" in body or action_name == "frag4-score-add"


def test_local_deploy_executes_score_add_then_score_filter(tmp_path: Path, monkeypatch) -> None:
    project = _write_project(tmp_path)
    pose_dir = tmp_path / "frag4-pose"
    resolved = ActionGraph(project).build()
    compute_project_sigils(project)

    values1 = np.array([1, -2, -3, 5], dtype=np.float32)
    values2 = np.array([-2, 4, 1, 1], dtype=np.float32)
    _materialize(project, resolved["frag4-pose"], pose=True)
    _materialize(project, resolved["frag4-score1"], score_values=values1)
    _materialize(project, resolved["frag4-score2"], score_values=values2)

    env = {
        **os.environ,
        "PATH": f"{AL_ENV_BIN}:{os.environ.get('PATH', '')}",
        "PYTHON": sys.executable,
        "ALARIC_DIR": str(ROOT / "alaric"),
    }
    deploy("local", resolved["frag4-score-add"].path)
    subprocess.run(["bash", "run.sh"], cwd=resolved["frag4-score-add"].path, env=env, check=True, capture_output=True, text=True)
    sum_sig = (resolved["frag4-score-add"].path / "sigil.txt").read_text().strip()
    summed = np.load(project.results_dir / sum_sig / "score.npy")
    np.testing.assert_array_equal(summed, values1 + values2)

    deploy("local", resolved["frag4-filter-score"].path)
    subprocess.run(["bash", "run.sh"], cwd=resolved["frag4-filter-score"].path, env=env, check=True, capture_output=True, text=True)
    filter_sig = (resolved["frag4-filter-score"].path / "sigil.txt").read_text().strip()
    output = project.results_dir / filter_sig
    provenance = load_npy(output / "provenance.npy.zst")
    assert provenance.tolist() == [0, 2]
    assert PoseReader.get_nposes(output) == 2
