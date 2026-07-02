from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import yaml
import zstandard as zstd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from alaric.mask import main as mask_main
import alaric.middle.deploy as deploy_module
from alaric.middle.checksum import byte_checksum, write_array_sidecar
from alaric.middle.deploy import deploy, generate_check_sh, generate_chunk_files, generate_run_sh
from alaric.middle.errors import SchemaError
from alaric.middle.graph import ActionGraph
from alaric.middle.project import Project
from alaric.middle.schema import normalize_action
from alaric.middle.sigil import compute_project_sigils
from alaric.score_add import main as score_add_main
from alaric.score_concat import main as score_concat_main


def _write_project(root: Path) -> None:
    data = root / "DATA"
    data.mkdir()
    (data / "constraints.json").write_text(
        """
{
  "pdb_code": "1abc",
  "frag4": {"sequence": "GU", "scores": {"dom": -1.5}},
  "frag5": {"sequence": "UU", "scores": {"dom": -2.5}},
  "pairs": [{"down": "frag4", "up": "frag5", "cRMSD": 0.25, "ovRMSD": 0.75}]
}
""".strip()
        + "\n"
    )
    (data / "anchor.yaml").write_text("angle: 30\ndihedral: 45 -45\n")
    (data / "pdbcode.txt").write_text("1abc\n")
    (data / "dom-aa.pdb").write_text("HEADER test\n")
    (data / "dom.pdb").write_text("HEADER test plain\n")

    def action(name: str, spec: dict) -> None:
        path = root / name
        path.mkdir()
        (path / "alaric.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))

    action(
        "frag4-anchor",
        {
            "action": "anchor",
            "fragment": "auto",
            "sequence": "auto",
            "exclude": "auto",
            "protein": "dom",
            "resid": 1,
            "nucleotide": "first",
        },
    )
    action(
        "frag4-score",
        {
            "action": "score",
            "input": "frag4-anchor",
            "sequence": "auto",
            "exclude": "auto",
            "protein": "dom",
        },
    )
    action(
        "frag4-filter",
        {
            "action": "filter",
            "input": "frag4-anchor",
            "score_input": "frag4-score",
            "threshold": "auto",
        },
    )
    action(
        "frag5-fwd",
        {
            "action": "grow",
            "input": "frag4-filter",
            "fragment": "auto",
            "sequence": "auto",
            "exclude": "auto",
            "direction": "auto",
            "crmsd": "auto",
            "ovrmsd": "auto",
        },
    )


def test_sigil_is_deterministic_and_writes_parameters(tmp_path: Path) -> None:
    _write_project(tmp_path)
    first = compute_project_sigils(Project.discover(tmp_path))
    second = compute_project_sigils(
        Project.discover(tmp_path),
        force=True,
    )
    assert first == second
    assert first["frag4-anchor"].endswith("-null")
    assert (tmp_path / "CACHE" / "parameters" / first["frag4-score"]).is_file()


def test_deploy_auto_sigils_only_target_dependency_closure(tmp_path: Path) -> None:
    _write_project(tmp_path)
    unrelated = tmp_path / "frag4-unrelated"
    unrelated.mkdir()
    (unrelated / "alaric.yaml").write_text(
        yaml.safe_dump(
            {
                "action": "anchor",
                "fragment": 4,
                "sequence": "auto",
                "exclude": "auto",
                "protein": "missing",
                "resid": 1,
                "nucleotide": "first",
            },
            sort_keys=False,
        )
    )

    deploy("local", tmp_path / "frag4-filter")

    assert (tmp_path / "frag4-anchor" / "sigil.txt").is_file()
    assert (tmp_path / "frag4-score" / "sigil.txt").is_file()
    assert (tmp_path / "frag4-filter" / "sigil.txt").is_file()
    assert not (tmp_path / "frag5-fwd" / "sigil.txt").exists()
    assert not (unrelated / "sigil.txt").exists()
    assert (tmp_path / "frag4-filter" / "run.sh").is_file()


def test_check_sh_reports_status_message(tmp_path: Path) -> None:
    _write_project(tmp_path)
    project = Project.discover(tmp_path)
    compute_project_sigils(project)
    action = ActionGraph(project).build()["frag4-anchor"]

    body = generate_check_sh(project, action, location="local")

    assert 'echo "check.sh: OK"' in body
    assert 'echo "check.sh: not OK"' in body
    assert 'exit "$status"' in body


def test_deploy_score_chunk_emits_independent_chunk_and_organize_scripts(tmp_path: Path) -> None:
    _write_project(tmp_path)
    project = Project.discover(tmp_path)
    compute_project_sigils(project)
    d = tmp_path / "frag4-score"
    deploy("local-chunk", d, nchunks=3)

    # Each chunk is an independent, parallel-runnable script that discovers its own range.
    for i in (1, 2, 3):
        body = (d / f"chunk{i}.sh").read_text()
        assert f"IDX={i}" in body
        assert "PoseReader.get_nposes" in body
        assert '"$FIRST"' in body and '"$LAST"' in body
        assert "np.concatenate" not in body

    # The organize step concatenates the chunk scores via the memory-safe helper.
    organize = (d / "organize.sh").read_text()
    assert "score_concat.py" in organize
    assert "np.concatenate" not in organize

    # run.sh is only a convenience wrapper: chunks one-by-one, then organize.
    run_sh = (d / "run.sh").read_text()
    assert "./check.sh" in run_sh
    for i in (1, 2, 3):
        assert f"bash ./chunk{i}.sh" in run_sh
    assert "bash ./organize.sh" in run_sh
    # The per-chunk body must NOT be inlined into run.sh.
    assert "PoseReader.get_nposes" not in run_sh


def test_score_deploy_defaults_to_compiled_kernel(tmp_path: Path) -> None:
    _write_project(tmp_path)
    project = Project.discover(tmp_path)
    sigils = compute_project_sigils(project)

    d = tmp_path / "frag4-score"
    deploy("local", d)
    body = (d / "run.sh").read_text()
    assert "\n  compiled \\\n" in body
    assert "\n  jax \\\n" not in body
    assert f"ln -s ../CACHE/results/{sigils['frag4-score']} results" in body
    assert f"cp ../CACHE/checksum/{sigils['frag4-score']} result.txt" in body


def test_anchor_test_requires_nconformers_or_conformer(tmp_path: Path) -> None:
    spec = {
        "action": "anchor-test",
        "fragment": 5,
        "sequence": "UU",
        "exclude": "1abc",
        "protein": "dom",
        "resid": 1,
        "nucleotide": "first",
    }

    with pytest.raises(SchemaError, match="exactly one of nconformers or conformer"):
        normalize_action("frag5-restrict", tmp_path, spec)

    spec["nconformers"] = 10
    spec["conformer"] = 3
    with pytest.raises(SchemaError, match="exactly one of nconformers or conformer"):
        normalize_action("frag5-restrict", tmp_path, spec)


def test_anchor_test_deploy_renders_single_conformer(tmp_path: Path) -> None:
    _write_project(tmp_path)
    d = tmp_path / "frag5-restrict"
    d.mkdir()
    (d / "alaric.yaml").write_text(
        yaml.safe_dump(
            {
                "action": "anchor-test",
                "fragment": 5,
                "sequence": "auto",
                "exclude": "auto",
                "protein": "dom",
                "resid": 1,
                "nucleotide": "first",
                "conformer": 7,
            },
            sort_keys=False,
        )
    )
    project = Project.discover(tmp_path)
    compute_project_sigils(project)

    body = generate_run_sh(project, ActionGraph(project).build()["frag5-restrict"], "local")

    assert "--conformer-range 7 7" in body
    assert "--conformer-range 1 7" not in body


def test_anchor_test_single_conformer_chunk_deploys_once(tmp_path: Path) -> None:
    _write_project(tmp_path)
    d = tmp_path / "frag5-restrict"
    d.mkdir()
    (d / "alaric.yaml").write_text(
        yaml.safe_dump(
            {
                "action": "anchor-test",
                "fragment": 5,
                "sequence": "auto",
                "exclude": "auto",
                "protein": "dom",
                "resid": 1,
                "nucleotide": "first",
                "conformer": 7,
            },
            sort_keys=False,
        )
    )
    deploy("local-chunk", d, nchunks=3)

    assert (d / "chunk1.sh").is_file()
    assert not (d / "chunk2.sh").exists()
    body = (d / "chunk1.sh").read_text()
    assert "SINGLE_CONFORMER=7" in body
    assert 'FIRST="$SINGLE_CONFORMER"' in body


def test_grow_deploy_renders_restrict_input(tmp_path: Path) -> None:
    _write_project(tmp_path)
    restrict_dir = tmp_path / "frag5-restrict"
    restrict_dir.mkdir()
    (restrict_dir / "alaric.yaml").write_text(
        yaml.safe_dump(
            {
                "action": "anchor-test",
                "fragment": 5,
                "sequence": "auto",
                "exclude": "auto",
                "protein": "dom",
                "resid": 1,
                "nucleotide": "first",
                "nconformers": 1,
            },
            sort_keys=False,
        )
    )
    (tmp_path / "frag5-fwd" / "alaric.yaml").write_text(
        yaml.safe_dump(
            {
                "action": "grow",
                "input": "frag4-filter",
                "restrict_input": "frag5-restrict",
                "fragment": "auto",
                "sequence": "auto",
                "exclude": "auto",
                "direction": "auto",
                "crmsd": "auto",
                "ovrmsd": "auto",
            },
            sort_keys=False,
        )
    )
    project = Project.discover(tmp_path)
    sigils = compute_project_sigils(project)
    action = ActionGraph(project).build()["frag5-fwd"]

    body = generate_run_sh(project, action, "local")
    assert "--restrict-poses" in body
    assert sigils["frag5-restrict"] in body

    files = generate_chunk_files(project, action, "local-chunk", nchunks=2)
    assert "--restrict-poses" in files["chunk1.sh"]
    assert sigils["frag5-restrict"] in files["chunk1.sh"]

    params = (
        tmp_path
        / "CACHE"
        / "parameters"
        / sigils["frag5-fwd"]
    ).read_text()
    assert "restrict_input" in params


def test_grow_test_requires_conformer(tmp_path: Path) -> None:
    spec = {
        "action": "grow-test",
        "input": "frag4-filter",
        "fragment": 5,
        "sequence": "UU",
        "exclude": "1abc",
        "direction": "forward",
        "crmsd": 0.25,
        "ovrmsd": 0.75,
    }

    with pytest.raises(SchemaError, match="missing keys.*conformer"):
        normalize_action("frag5-test", tmp_path, spec)

    spec["conformer"] = 0
    with pytest.raises(SchemaError, match="conformer must be positive"):
        normalize_action("frag5-test", tmp_path, spec)


def test_grow_test_deploy_renders_single_conformer(tmp_path: Path) -> None:
    _write_project(tmp_path)
    (tmp_path / "frag5-fwd" / "alaric.yaml").write_text(
        yaml.safe_dump(
            {
                "action": "grow-test",
                "input": "frag4-filter",
                "fragment": "auto",
                "sequence": "auto",
                "exclude": "auto",
                "direction": "auto",
                "crmsd": "auto",
                "ovrmsd": "auto",
                "conformer": 7,
            },
            sort_keys=False,
        )
    )
    project = Project.discover(tmp_path)
    compute_project_sigils(project)
    action = ActionGraph(project).build()["frag5-fwd"]

    body = generate_run_sh(project, action, "local")
    assert "--conformer 7" in body

    files = generate_chunk_files(project, action, "local-chunk", nchunks=2)
    assert "--conformer 7" in files["chunk1.sh"]
    assert "--pose-range \"$FIRST\" \"$LAST\"" in files["chunk1.sh"]
    sigil = (tmp_path / "frag5-fwd" / "sigil.txt").read_text().strip()
    assert (tmp_path / "CACHE" / "parameters" / sigil).is_file()


def test_grow_deploy_propagates_auto_pdb_exclude(tmp_path: Path) -> None:
    _write_project(tmp_path)
    project = Project.discover(tmp_path)
    compute_project_sigils(project)
    action = ActionGraph(project).build()["frag5-fwd"]

    body = generate_run_sh(project, action, "local")
    assert "--pdb-exclude 1abc" in body

    files = generate_chunk_files(project, action, "local-chunk", nchunks=2)
    assert "--pdb-exclude 1abc" in files["chunk1.sh"]


def test_score_chunk_deploy_defaults_to_compiled_kernel(tmp_path: Path) -> None:
    _write_project(tmp_path)
    project = Project.discover(tmp_path)
    sigils = compute_project_sigils(project)

    d = tmp_path / "frag4-score"
    deploy("local-chunk", d, nchunks=2)
    body = (d / "chunk1.sh").read_text()
    assert "\n  compiled \\\n" in body
    assert "\n  jax \\\n" not in body
    organize = (d / "organize.sh").read_text()
    assert f"ln -s ../CACHE/results/{sigils['frag4-score']} results" in organize
    assert f"cp ../CACHE/checksum/{sigils['frag4-score']} result.txt" in organize


def test_remote_score_deploy_defaults_to_compiled_kernel(tmp_path: Path) -> None:
    _write_project(tmp_path)
    project = Project.discover(tmp_path)
    sigils = compute_project_sigils(project)
    action = ActionGraph(project).build()["frag4-score"]

    body = generate_run_sh(project, action, "remote")
    assert f'cd "${{ALARIC_REMOTE_DEPLOYMENT_DIR:?}}/SIGIL/{sigils["frag4-score"]}"' in body
    assert "\n  compiled \\\n" in body
    assert "\n  jax \\\n" not in body

    files = generate_chunk_files(project, action, "remote-chunk", nchunks=2)
    chunk = files["chunk1.sh"]
    assert "\n  compiled \\\n" in chunk
    assert "\n  jax \\\n" not in chunk
    chunk_root = f"${{ALARIC_REMOTE_RESULT_DIR:?}}/{sigils['frag4-score']}-CHUNKS"
    assert f"CHUNK_DIR={chunk_root}/chunk-${{IDX}}" in chunk
    assert '"$CHUNK_DIR/score.npy"' in chunk
    assert f"score_concat.py {chunk_root} " in files["organize.sh"]
    assert "--nchunks 2" in files["organize.sh"]


def test_remote_deploy_uses_sigil_dir_and_project_alias(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_project(tmp_path)
    project = Project.discover(tmp_path)
    sigils = compute_project_sigils(project)
    action_dir = tmp_path / "frag4-filter"
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: object) -> None:
        calls.append(args)

    monkeypatch.setenv("ALARIC_REMOTE_HOST", "cluster")
    monkeypatch.setenv("ALARIC_REMOTE_DEPLOYMENT_DIR", "/remote/deploy")
    monkeypatch.setenv("ALARIC_REMOTE_ALARIC_DIR", "/remote/alaric")
    monkeypatch.setenv("ALARIC_REMOTE_RESULT_DIR", "/remote/results")
    monkeypatch.delenv("ALARIC_PROJECT", raising=False)
    monkeypatch.setattr(deploy_module.subprocess, "run", fake_run)

    deploy("remote", action_dir)

    sigil = sigils["frag4-filter"]
    assert ["ssh", "cluster", "mkdir", "-p", f"/remote/deploy/SIGIL/{sigil}", "/remote/deploy/DATA", "/remote/deploy/PROJECT"] in calls
    assert ["ssh", "cluster", "ln", "-sfn", f"../SIGIL/{sigil}", "/remote/deploy/PROJECT/frag4-filter"] in calls
    assert ["ssh", "cluster", "ln", "-sfn", f"/remote/results/{sigil}", f"/remote/deploy/SIGIL/{sigil}/results"] in calls
    assert any(call[0] == "scp" and call[-1] == f"cluster:/remote/deploy/SIGIL/{sigil}/" for call in calls)


def test_remote_chunk_python_paths_are_expandvars_compatible(tmp_path: Path) -> None:
    _write_project(tmp_path)
    project = Project.discover(tmp_path)
    compute_project_sigils(project)
    action = ActionGraph(project).build()["frag4-score"]

    files = generate_chunk_files(project, action, "remote-chunk", nchunks=3)
    body = "\n".join(files.values())

    assert "os.path.expandvars('${ALARIC_REMOTE_RESULT_DIR}/" in body
    assert "os.path.expandvars('${ALARIC_REMOTE_RESULT_DIR:?}" not in body


def test_score_add_and_mask_validate_shapes(tmp_path: Path) -> None:
    a = tmp_path / "a.npy"
    b = tmp_path / "b.npy"
    out = tmp_path / "out.npy"
    np.save(a, np.array([1.0, 2.0, 3.0]))
    np.save(b, np.array([4.0, 5.0, 6.0]))
    assert score_add_main([str(a), str(b), str(out)]) == 0
    np.testing.assert_array_equal(np.load(out), np.array([5.0, 7.0, 9.0]))

    mask_out = tmp_path / "mask.npy"
    assert mask_main([str(out), "8.0", str(mask_out)]) == 0
    mask = np.load(mask_out)
    assert mask.ndim == 1

    np.save(b, np.array([1.0]))
    with pytest.raises(ValueError):
        score_add_main([str(a), str(b), str(out)])


def test_score_concat_requires_all_expected_chunks(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks"
    (chunks / "chunk-1").mkdir(parents=True)
    np.save(chunks / "chunk-1" / "score.npy", np.array([1.0], dtype=np.float32))

    with pytest.raises(FileNotFoundError):
        score_concat_main([str(chunks), str(tmp_path / "score.npy"), "--nchunks", "2"])


def test_checksum_is_zstd_transparent(tmp_path: Path) -> None:
    payload = b"payload" * 100
    raw = tmp_path / "score.npy"
    compressed = tmp_path / "score.npy.zst"
    raw.write_bytes(payload)
    compressed.write_bytes(zstd.ZstdCompressor().compress(payload))
    assert byte_checksum(raw) == byte_checksum(compressed)
    checksum = write_array_sidecar(compressed)
    assert (tmp_path / "score.npy.CHECKSUM").read_text().strip() == checksum
