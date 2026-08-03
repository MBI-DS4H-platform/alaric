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
from alaric.mask_common_conformer import main as mask_common_conformer_main
import alaric.middle.deploy as deploy_module
from alaric.middle.checksum import byte_checksum, write_array_sidecar, write_pose_sidecars
from alaric.middle.deploy import deploy, generate_check_sh, generate_chunk_files, generate_run_sh
from alaric.middle.errors import GraphError, MiddleError, SchemaError
from alaric.middle.graph import ActionGraph
from alaric.middle.project import Project
from alaric.middle.result_sidecar import main as result_sidecar_main
from alaric.middle.schema import normalize_action
from alaric.middle.sigil import compute_project_sigils
from alaric.npy_io import load_npy, save_npy
from alaric.poses import write_arc_file
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


def test_run_sh_guards_against_stale_sigil_and_existing_result(tmp_path: Path) -> None:
    _write_project(tmp_path)
    project = Project.discover(tmp_path)
    sigils = compute_project_sigils(project)
    action = ActionGraph(project).build()["frag4-score"]
    sigil = sigils["frag4-score"]

    body = generate_run_sh(project, action, "local")
    assert f'SIGIL="{sigil}"' in body
    assert "if [ -f result.txt ]; then" in body
    assert 'echo "result.txt already exists; refusing to run" >&2' in body
    assert "alaric-sigil --pool ." in body
    assert 'echo "stale sigil, re-run alaric-deploy" >&2' in body
    # The result.txt guard and the sigil guard must both run before check.sh / setup.
    assert body.index("if [ -f result.txt ]") < body.index("./check.sh")
    assert body.index("alaric-sigil --pool .") < body.index("./check.sh")
    assert body.index("if [ -f result.txt ]") < body.index("alaric-sigil --pool .")

    # Remote scripts have no local alaric.yaml/sigil.txt to compare against.
    remote_body = generate_run_sh(project, action, "remote")
    assert "alaric-sigil" not in remote_body
    assert "result.txt already exists" not in remote_body


def test_chunk_scripts_guard_but_organize_and_wrapper_run_sh_dont(tmp_path: Path) -> None:
    _write_project(tmp_path)
    project = Project.discover(tmp_path)
    sigils = compute_project_sigils(project)
    action = ActionGraph(project).build()["frag4-score"]
    sigil = sigils["frag4-score"]

    files = generate_chunk_files(project, action, "local-chunk", nchunks=2)
    for name in ("chunk1.sh", "chunk2.sh"):
        body = files[name]
        assert f'SIGIL="{sigil}"' in body
        assert "alaric-sigil --pool ." in body
        assert body.index("alaric-sigil --pool .") < body.index("./check.sh")
    # organize.sh and the convenience run.sh wrapper are reached only through chunkN.sh,
    # which already guards; they don't need their own copy of the check.
    assert "alaric-sigil --pool ." not in files["organize.sh"]
    assert "alaric-sigil --pool ." not in files["run.sh"]


def test_sigil_pool_limits_to_action_and_upstream_dependencies(tmp_path: Path) -> None:
    from alaric.middle.sigil import main as sigil_main

    _write_project(tmp_path)
    action_dir = tmp_path / "frag4-filter"

    assert sigil_main(["--pool", str(action_dir)]) == 0

    assert (tmp_path / "frag4-anchor" / "sigil.txt").is_file()
    assert (tmp_path / "frag4-score" / "sigil.txt").is_file()
    assert (tmp_path / "frag4-filter" / "sigil.txt").is_file()
    assert not (tmp_path / "frag5-fwd" / "sigil.txt").exists()


def test_deploy_recomputes_stale_sigil_on_redeploy(tmp_path: Path) -> None:
    _write_project(tmp_path)
    project = Project.discover(tmp_path)
    compute_project_sigils(project)
    action_dir = tmp_path / "frag4-filter"

    deploy("local", action_dir)
    old_sigil = (action_dir / "sigil.txt").read_text().strip()

    anchor_dir = tmp_path / "frag4-anchor"
    spec = yaml.safe_load((anchor_dir / "alaric.yaml").read_text())
    spec["resid"] = 2
    anchor_dir.joinpath("alaric.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))

    # Before the fix, deploy() only recomputed sigils when sigil.txt was entirely missing,
    # so this second deploy would silently keep baking the stale sigil into run.sh.
    deploy("local", action_dir)
    new_sigil = (action_dir / "sigil.txt").read_text().strip()

    assert new_sigil != old_sigil
    assert new_sigil in (action_dir / "run.sh").read_text()


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


def test_filter_deploy_compresses_both_routes(tmp_path: Path) -> None:
    """Every pose-producing action writes compressed poses; filter is no exception."""
    _write_project(tmp_path)
    project = Project.discover(tmp_path)
    compute_project_sigils(project)
    action = ActionGraph(project).build()["frag4-filter"]

    for deployer in ("local", "remote"):
        body = generate_run_sh(project, action, deployer)
        # both routes are rendered into the script: select-poses.py for a mask input,
        # filter-poses.py for a score threshold, and each one asks for compression
        assert "select-poses.py" in body and "filter-poses.py" in body, deployer
        assert body.count("--compress") == 2, deployer


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
    # The per-chunk dir is bound through SCORE_CHUNKS, defaulting to the shared FS so an
    # independently submitted chunk still writes where the organize job can see it.
    assert f'SCORE_CHUNKS="${{ALARIC_SCORE_CHUNKS_DIR:-{chunk_root}}}"' in chunk
    assert "CHUNK_DIR=${SCORE_CHUNKS}/chunk-${IDX}" in chunk
    assert '"$CHUNK_DIR/score.npy"' in chunk
    assert "score_concat.py ${SCORE_CHUNKS} " in files["organize.sh"]
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


def test_remote_anchor_keeps_unorganized_pool_off_the_shared_fs(tmp_path: Path) -> None:
    _write_project(tmp_path)
    project = Project.discover(tmp_path)
    sigils = compute_project_sigils(project)
    action = ActionGraph(project).build()["frag4-anchor"]
    sigil = sigils["frag4-anchor"]
    partial = f"${{ALARIC_REMOTE_RESULT_DIR:?}}/{sigil}.partial"

    body = generate_run_sh(project, action, "remote")

    # One process owns the whole pool, so the shards go to node-local scratch and only the
    # organized result crosses onto the shared filesystem.
    assert 'POSE_POOL="$(mktemp -d -p "${ALARIC_REMOTE_SCRATCH_DIR:-${TMPDIR:-/tmp}}"' in body
    assert "trap 'rm -rf" in body and "' EXIT" in body
    assert "--output ${POSE_POOL}" in body
    assert "organize.py ${POSE_POOL}" in body
    # The sidecar is computed on the pool, so hashing reads local disk, not the shared FS.
    assert "result_sidecar.py pose ${POSE_POOL}" in body
    assert f"mv ${{POSE_POOL}} {partial}" in body
    # This script always makes its own pool; there is no override to honour.
    assert "ALARIC_UNORGANIZED_DIR" not in body
    # .partial must not be pre-created: the move across creates it, and mv into an
    # existing directory would nest the pool inside it instead.
    assert f"mkdir -p {partial}" not in body
    # Staging the shards through scratch is pointless when they are already there.
    assert "\n  --local-tempdir" not in body
    assert "\n  --local-stagedir" not in body


def test_remote_chunk_anchor_pool_defaults_to_shared_fs_and_run_sh_overrides(tmp_path: Path) -> None:
    _write_project(tmp_path)
    project = Project.discover(tmp_path)
    sigils = compute_project_sigils(project)
    action = ActionGraph(project).build()["frag4-anchor"]
    sigil = sigils["frag4-anchor"]
    partial = f"${{ALARIC_REMOTE_RESULT_DIR:?}}/{sigil}.partial"

    files = generate_chunk_files(project, action, "remote-chunk", nchunks=2)

    # Chunks are submitted independently and organize.sh runs on another node, so their
    # shared pool must default to the shared filesystem.
    for name in ("chunk1.sh", "chunk2.sh", "organize.sh"):
        assert f'POSE_POOL="${{ALARIC_UNORGANIZED_DIR:-{partial}}}"' in files[name]
    assert "--output ${POSE_POOL}" in files["chunk1.sh"]
    assert "mktemp" not in files["chunk1.sh"]

    # organize.sh serves both cases, so it decides at runtime.
    organize = files["organize.sh"]
    assert "organize.py ${POSE_POOL}" in organize
    assert 'if [ -z "${ALARIC_UNORGANIZED_DIR:-}" ]; then' in organize
    assert "organize_opts+=(--local-tempdir --local-stagedir)" in organize
    assert 'if [ -n "${ALARIC_UNORGANIZED_DIR:-}" ]; then' in organize
    assert f"  mv ${{POSE_POOL}} {partial}" in organize

    # run.sh executes every chunk itself, so it can keep the pool node-local.
    run_sh = files["run.sh"]
    assert 'ALARIC_UNORGANIZED_DIR="$(mktemp -d -p ' in run_sh
    assert "export ALARIC_UNORGANIZED_DIR" in run_sh
    assert "' EXIT" in run_sh
    assert f"mkdir -p {partial}" not in run_sh


def test_remote_chunk_score_run_sh_redirects_chunk_dir(tmp_path: Path) -> None:
    _write_project(tmp_path)
    project = Project.discover(tmp_path)
    compute_project_sigils(project)
    action = ActionGraph(project).build()["frag4-score"]

    files = generate_chunk_files(project, action, "remote-chunk", nchunks=2)

    run_sh = files["run.sh"]
    assert 'ALARIC_SCORE_CHUNKS_DIR="$(mktemp -d -p ' in run_sh
    assert "export ALARIC_SCORE_CHUNKS_DIR" in run_sh
    # score has no unorganized pool, so there are no pose sidecars to clean up.
    assert 'trap \'rm -rf "${ALARIC_SCORE_CHUNKS_DIR}"\' EXIT' in run_sh
    assert ".INDEX" not in run_sh


def test_remote_grow_still_pools_on_the_shared_fs(tmp_path: Path) -> None:
    """grow emits provenance sidecars, so its pool redirection is deferred."""
    _write_project(tmp_path)
    (tmp_path / "frag5-fwd" / "alaric.yaml").write_text(
        yaml.safe_dump(
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
            sort_keys=False,
        )
    )
    project = Project.discover(tmp_path)
    sigils = compute_project_sigils(project)
    action = ActionGraph(project).build()["frag5-fwd"]
    partial = f"${{ALARIC_REMOTE_RESULT_DIR:?}}/{sigils['frag5-fwd']}.partial"

    body = generate_run_sh(project, action, "remote")
    assert f"--output {partial}" in body
    assert "mktemp" not in body
    assert "ALARIC_UNORGANIZED_DIR" not in body


def test_remote_chunk_deploy_warns_about_shared_fs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_project(tmp_path)
    project = Project.discover(tmp_path)
    compute_project_sigils(project)

    deploy("local-chunk", tmp_path / "frag4-score", nchunks=2)
    assert capsys.readouterr().err == ""

    with pytest.raises(MiddleError):
        # Missing remote environment; the warning is emitted before that is checked.
        deploy("remote-chunk", tmp_path / "frag4-anchor", nchunks=2)
    assert (
        "Warning: parallel execution of chunks may degrade network file system performance"
        in capsys.readouterr().err
    )


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


def test_mask_action_compresses_but_keeps_the_logical_result_name(tmp_path: Path) -> None:
    scores = tmp_path / "score.npy"
    np.save(scores, np.array([-3.0, 1.0, -2.0, 4.0], dtype=np.float32))
    plain = tmp_path / "plain.npy"
    logical = tmp_path / "mask.npy"

    assert mask_main([str(scores), "0.0", str(plain)]) == 0
    assert mask_main([str(scores), "0.0", str(logical), "--compress"]) == 0

    assert not logical.exists()
    np.testing.assert_array_equal(load_npy(tmp_path / "mask.npy.zst"), np.load(plain))
    # ... and the compressed mask is the same result: same checksum, same sidecar name,
    # written by a sidecar step that was handed the logical (uncompressed) name.
    assert byte_checksum(tmp_path / "mask.npy.zst") == byte_checksum(plain)
    assert result_sidecar_main(["array", str(logical)]) == 0
    assert (tmp_path / "mask.npy.CHECKSUM").read_text().strip() == byte_checksum(plain)


def test_mask_deploy_asks_for_compression(tmp_path: Path) -> None:
    _write_project(tmp_path)
    (tmp_path / "frag4-mask").mkdir()
    (tmp_path / "frag4-mask" / "alaric.yaml").write_text(
        yaml.safe_dump(
            {
                "action": "mask",
                "input": "frag4-anchor",
                "score_input": "frag4-score",
                "threshold": -1.0,
            },
            sort_keys=False,
        )
    )
    project = Project.discover(tmp_path)
    compute_project_sigils(project)
    action = ActionGraph(project).build()["frag4-mask"]

    for deployer in ("local", "remote"):
        body = generate_run_sh(project, action, deployer)
        assert "mask.py" in body and "--compress" in body, deployer
        # the sidecar step is still handed the logical name
        assert "result_sidecar.py array" in body
        assert "mask.npy.zst" not in body


def _write_conformer_pool(path: Path, conformers: list[int]) -> None:
    path.mkdir()
    count = len(conformers)
    write_arc_file(
        path / "poses-1.arc",
        np.zeros(3, dtype=np.int16),
        np.zeros((1, 3), dtype=np.int16),
        np.array([count], dtype=np.uint32),
        np.column_stack(
            (
                np.asarray(conformers, dtype=np.uint16),
                np.zeros(count, dtype=np.uint16),
                np.zeros(count, dtype=np.uint16),
            )
        ),
        bucket_size=16,
    )


def test_mask_common_conformer_writes_per_pool_compressed_masks(tmp_path: Path) -> None:
    pool1 = tmp_path / "pool1"
    pool2 = tmp_path / "pool2"
    output = tmp_path / "output"
    _write_conformer_pool(pool1, [1, 2, 1, 3])
    _write_conformer_pool(pool2, [3, 4, 1])

    assert mask_common_conformer_main([str(pool1), str(pool2), str(output)]) == 0
    assert not (output / "mask1.npy").exists()
    assert not (output / "mask2.npy").exists()
    np.testing.assert_array_equal(load_npy(output / "mask1.npy.zst"), [True, False, True, True])
    np.testing.assert_array_equal(load_npy(output / "mask2.npy.zst"), [True, False, True])


def test_mask_common_conformer_action_renders_two_pool_backend(tmp_path: Path) -> None:
    _write_project(tmp_path)
    for name in ("frag5-anchor-bwd", "frag5-anchor-fwd"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "alaric.yaml").write_text(
            yaml.safe_dump(
                {
                    "action": "anchor",
                    "fragment": 5,
                    "sequence": "UU",
                    "exclude": "auto",
                    "protein": "dom",
                    "resid": 1,
                    "nucleotide": "first",
                },
                sort_keys=False,
            )
        )
    action_dir = tmp_path / "frag5-common"
    action_dir.mkdir()
    (action_dir / "alaric.yaml").write_text(
        "action: mask-common-conformer\ninput1: frag5-anchor-bwd\ninput2: frag5-anchor-fwd\n"
    )
    project = Project.discover(tmp_path)
    compute_project_sigils(project)
    action = ActionGraph(project).build()["frag5-common"]

    body = generate_run_sh(project, action, "local")
    assert "mask_common_conformer.py" in body
    assert "mask1.npy" not in body and "mask2.npy" not in body
    assert "result_sidecar.py pose" in body


def test_filter_selects_matching_mask_common_conformer_output(tmp_path: Path) -> None:
    _write_project(tmp_path)
    for name in ("frag5-anchor-bwd", "frag5-anchor-fwd"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "alaric.yaml").write_text(
            yaml.safe_dump(
                {
                    "action": "anchor",
                    "fragment": 5,
                    "sequence": "UU",
                    "exclude": "auto",
                    "protein": "dom",
                    "resid": 1,
                    "nucleotide": "first",
                },
                sort_keys=False,
            )
        )
    (tmp_path / "frag5-common-conformer").mkdir()
    (tmp_path / "frag5-common-conformer" / "alaric.yaml").write_text(
        "action: mask-common-conformer\ninput1: frag5-anchor-bwd\ninput2: frag5-anchor-fwd\n"
    )
    (tmp_path / "frag5-anchor-bwd-filtered").mkdir()
    (tmp_path / "frag5-anchor-bwd-filtered" / "alaric.yaml").write_text(
        "action: filter\ninput: frag5-anchor-bwd\nmask_input: frag5-common-conformer\nmask: 1\n"
    )
    project = Project.discover(tmp_path)
    compute_project_sigils(project)
    action = ActionGraph(project).build()["frag5-anchor-bwd-filtered"]

    body = generate_run_sh(project, action, "local")
    assert "/mask1.npy" in body

    (tmp_path / "frag5-anchor-bwd-filtered" / "alaric.yaml").write_text(
        "action: filter\ninput: frag5-anchor-bwd\nmask_input: frag5-common-conformer\nmask: 2\n"
    )
    with pytest.raises(GraphError, match="must select the mask"):
        ActionGraph(Project.discover(tmp_path)).build()


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


def test_compressing_a_pose_dir_index_array_keeps_the_result_checksum(tmp_path: Path) -> None:
    """Why provenance/map arrays can be compressed at all: results stay addressable.

    The deepfolder index is keyed on the logical name and hashes the *uncompressed*
    bytes, so a cached result produced before compression still matches one produced
    after it.
    """
    provenance = np.arange(1000, dtype=np.uint32)
    checksums = []
    for name, compress in (("plain", False), ("compressed", True)):
        pose_dir = tmp_path / name
        pose_dir.mkdir()
        (pose_dir / "poses-1.arc").write_bytes(b"arc payload")
        save_npy(pose_dir / "provenance.npy", provenance, compress=compress)
        checksums.append(write_pose_sidecars(pose_dir))

    assert (tmp_path / "compressed" / "provenance.npy.zst").is_file()
    assert not (tmp_path / "compressed" / "provenance.npy").exists()
    assert checksums[0] == checksums[1]
    assert (tmp_path / "plain.INDEX").read_bytes() == (tmp_path / "compressed.INDEX").read_bytes()
