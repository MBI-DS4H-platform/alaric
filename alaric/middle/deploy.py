from __future__ import annotations

import argparse
import os
import posixpath
import shlex
import shutil
import subprocess
from pathlib import Path

from .backend import render_template, template_context
from .checksum import byte_checksum
from .errors import MiddleError
from .graph import ActionGraph
from .project import Project
from .resolve import ResolvedAction
from .schema import DEPENDENCY_FIELDS, OUTPUT_KIND
from .sigil import compute_project_sigils

# Actions that support chunked deployment, and their chunking axis:
#   anchor / anchor-test -> conformers (--conformer-range)
#   anchor-refe          -> conformers (--conformer-range)
#   grow / grow-test    -> source poses (--pose-range)
#   score                -> source poses (POSE_START/POSE_END)
CHUNKABLE = {"anchor", "anchor-test", "anchor-refe", "grow", "grow-test", "score"}
# Separates the per-chunk body from the organize/finalize body in chunk templates.
ORGANIZE_DELIM = "### ORGANIZE ###"
# Remote env vars that are defined in the *local* deployer environment and must be
# baked into the generated remote scripts (they are not defined on the remote host).
_REMOTE_ENV_VARS = (
    "ALARIC_REMOTE_ALARIC_DIR",
    "ALARIC_REMOTE_DEPLOYMENT_DIR",
    "ALARIC_REMOTE_RESULT_DIR",
    "ALARIC_REMOTE_SCRATCH_DIR",
)


def _repo_alaric_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _template_path(project: Project, deployer: str, action_name: str) -> Path:
    candidates = [
        project.root / "templates" / deployer / f"{action_name}.sh",
        Path(__file__).resolve().parent / "templates" / deployer / f"{action_name}.sh",
    ]
    if action_name == "grow-test":
        candidates.extend(
            [
                project.root / "templates" / deployer / "grow.sh",
                Path(__file__).resolve().parent / "templates" / deployer / "grow.sh",
            ]
        )
    for path in candidates:
        if path.is_file():
            return path
    raise MiddleError(f"missing template for {deployer}/{action_name}.sh")


def render_action_template(
    project: Project,
    action: ResolvedAction,
    *,
    deployer: str,
    alaric_dir: str,
    output_dir: str,
    location: str,
    extra_context: dict[str, object] | None = None,
) -> list[str]:
    template = _template_path(project, deployer, action.action)
    context = template_context(action, alaric_dir=alaric_dir, output_dir=output_dir, location=location)
    if extra_context:
        context.update(extra_context)
    text = render_template(
        template.read_text(),
        context,
    ).rstrip()
    return text.splitlines()


def _read_sigil(action_dir: Path) -> str:
    path = action_dir / "sigil.txt"
    if not path.is_file():
        raise MiddleError(f"{action_dir}: missing sigil.txt; run alaric-sigil first")
    return path.read_text().strip()


def _result_output_kind(action: ResolvedAction) -> str:
    return OUTPUT_KIND[action.action]


def _alaric_dir_expr(local: bool) -> str:
    if local:
        return '"${ALARIC_DIR:-' + str(_repo_alaric_dir()) + '}"'
    return '"${ALARIC_REMOTE_ALARIC_DIR:?}"'


def _pythonpath_line(alaric_dir: str) -> str:
    return "export PYTHONPATH=" + alaric_dir + "${PYTHONPATH:+:${PYTHONPATH}}"


def _remote_env_header(local: bool) -> list[str]:
    """Bake the ALARIC_REMOTE_* vars (set locally) into the generated remote scripts.

    These variables are configured in the local deployer environment, not on the remote
    host where the scripts run, so each generated remote script must define them at the top.
    Only variables present in the local environment at deploy time are emitted; this keeps
    unit tests that render without a configured remote environment working (the body keeps
    its ``${VAR:?}`` references in that case).
    """
    if local:
        return []
    lines: list[str] = []
    defined: list[str] = []
    for var in _REMOTE_ENV_VARS:
        value = os.environ.get(var)
        if value is None:
            continue
        lines.append(f"{var}={shlex.quote(value)}")
        defined.append(var)
    if defined:
        lines.append("export " + " ".join(defined))
        lines.append("")
    return lines


def _prologue(local: bool, sigil: str) -> list[str]:
    """Shebang + strict mode + env baking + a SLURM-safe cd to the action dir.

    ``cd "$(dirname "$0")"`` is unreliable under SLURM (sbatch copies the script to a spool
    dir, so ``$0`` no longer points at the deployment dir). On remote we therefore cd to the
    absolute deployment dir baked from the local ``ALARIC_REMOTE_DEPLOYMENT_DIR``.
    """
    lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    lines.extend(_remote_env_header(local))
    if local:
        lines.append('cd "$(dirname "$0")"')
    else:
        lines.append(f'cd "${{ALARIC_REMOTE_DEPLOYMENT_DIR:?}}/SIGIL/{sigil}"')
        # Honor a SLURM core reservation (e.g. `sbatch -c N`). `--nprocs` defaults to
        # os.cpu_count(), which reports the whole node; Python >=3.13 lets PYTHON_CPU_COUNT
        # override os.cpu_count(), so the backends pick up the allocation, not the node.
        lines.extend(
            [
                'if [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then',
                '  export PYTHON_CPU_COUNT="${SLURM_CPUS_PER_TASK}"',
                'elif [ -n "${SLURM_CPUS_ON_NODE:-}" ]; then',
                '  export PYTHON_CPU_COUNT="${SLURM_CPUS_ON_NODE}"',
                "fi",
            ]
        )
    return lines


def _compute_dirs(action: ResolvedAction, sigil: str, local: bool) -> tuple[str, str, list[str]]:
    """Return (output_dir, final_dir, setup_lines).

    Remote: the (unorganized) output pool lives on the **shared** result filesystem at
    ``$ALARIC_REMOTE_RESULT_DIR/<sigil>.partial`` — NOT on ``$ALARIC_REMOTE_SCRATCH_DIR``,
    which is node-local and therefore not visible to the other SLURM nodes that run the
    sibling chunk jobs and the organize job. The organize step stages *its own* reads/writes
    to node-local scratch via ``--local-tempdir`` / ``--local-stagedir`` (commented knobs).
    The completed pool is atomically renamed ``.partial`` -> final.
    """
    if local:
        output_dir = f"../CACHE/results/{sigil}"
        setup = [f"rm -rf {output_dir}", f"mkdir -p {output_dir} ../CACHE/checksum"]
        return output_dir, output_dir, setup
    final_dir = f"${{ALARIC_REMOTE_RESULT_DIR:?}}/{sigil}"
    output_dir = f"{final_dir}.partial"
    setup = [f"rm -rf {output_dir} {final_dir}", f"mkdir -p {output_dir} ${{ALARIC_REMOTE_RESULT_DIR:?}}"]
    return output_dir, final_dir, setup


def _sidecar_command(kind: str, result_dir: str, sigil: str, *, local: bool, alaric_dir: str) -> str:
    sidecar = f"{alaric_dir}/middle/result_sidecar.py"
    if kind == "pose":
        cmd = f'"${{PYTHON:-python}}" {sidecar} pose {result_dir}'
        if local:
            cmd += f"\ncp {result_dir}.CHECKSUM ../CACHE/checksum/{sigil}"
        return cmd
    name = "mask.npy" if kind == "mask" else "score.npy"
    cmd = f'"${{PYTHON:-python}}" {sidecar} array {result_dir}/{name}'
    if local:
        cmd += f"\ncp {result_dir}/{name}.CHECKSUM ../CACHE/checksum/{sigil}"
    return cmd


def _finalize_lines(
    action: ResolvedAction,
    output_dir: str,
    final_dir: str,
    sigil: str,
    result_kind: str,
    local: bool,
    alaric_dir: str,
) -> list[str]:
    lines = [_sidecar_command(result_kind, output_dir, sigil, local=local, alaric_dir=alaric_dir)]
    if local:
        return lines
    # output_dir is "<final>.partial"; atomically promote it (and its pose sidecars) to final.
    if result_kind == "pose":
        lines.extend(
            [
                f"mv {output_dir}.INDEX {final_dir}.INDEX",
                f"mv {output_dir}.CHECKSUM {final_dir}.CHECKSUM",
            ]
        )
    lines.append(f"mv {output_dir} {final_dir}")
    return lines


def _local_success_lines(sigil: str) -> list[str]:
    return [
        "rm -f results",
        f"ln -s ../CACHE/results/{sigil} results",
        f"cp ../CACHE/checksum/{sigil} result.txt",
    ]


def generate_check_sh(project: Project, action: ResolvedAction, *, location: str) -> str:
    """Verify each input dependency is materialized at the execution location.

    Local: also provenance-guards against the sibling action dirs (sigil + result.txt).
    Remote: only the materialization check is possible/meaningful — the sibling action dirs
    are not shipped to the remote, and provenance is already pinned by the dep sigil baked
    into the result path. For a root action (no inputs) this is intentionally a no-op.
    """
    local = location != "remote"
    sigil = _read_sigil(action.path)
    lines = _prologue(local, sigil)
    lines.extend(
        [
            'check_status() {',
            '  status=$?',
            '  if [ "$status" -eq 0 ]; then',
            '    echo "check.sh: OK"',
            "  else",
            '    echo "check.sh: not OK"',
            "  fi",
            '  exit "$status"',
            "}",
            "trap check_status EXIT",
        ]
    )
    lines.append("")
    for field in DEPENDENCY_FIELDS[action.action]:
        dep = action.params.get(field)
        if not isinstance(dep, ResolvedAction):
            continue
        dep_sigil = _read_sigil(dep.path)
        if location == "remote":
            lines.append(f"test -e ${{ALARIC_REMOTE_RESULT_DIR:?}}/{dep_sigil}")
        else:
            rel = f"../{dep.name}"
            lines.extend(
                [
                    f'test "$(cat {rel}/sigil.txt)" = "{dep_sigil}"',
                    f"test -s {rel}/result.txt",
                    f"test -e ../CACHE/results/{dep_sigil}",
                ]
            )
    lines.append("")
    return "\n".join(lines)


def generate_run_sh(project: Project, action: ResolvedAction, deployer: str, nchunks: int | None = None) -> str:
    """Single-script (non-chunk) run.sh."""
    sigil = _read_sigil(action.path)
    local = deployer.startswith("local")
    location = "local" if local else "remote"
    alaric_dir = _alaric_dir_expr(local)
    output_dir, final_dir, setup = _compute_dirs(action, sigil, local)
    lines = _prologue(local, sigil)
    lines.append("./check.sh")
    lines.append("")
    lines.append(_pythonpath_line(alaric_dir))
    lines.append("")
    lines.extend(setup)
    lines.extend(
        render_action_template(
            project, action, deployer=deployer, alaric_dir=alaric_dir, output_dir=output_dir, location=location
        )
    )
    result_kind = _result_output_kind(action)
    lines.extend(_finalize_lines(action, output_dir, final_dir, sigil, result_kind, local, alaric_dir))
    if local:
        lines.extend(_local_success_lines(sigil))
    lines.append("")
    return "\n".join(lines)


def generate_chunk_files(
    project: Project, action: ResolvedAction, deployer: str, nchunks: int | None
) -> dict[str, str]:
    """Emit independent chunkN.sh scripts + organize.sh + a convenience run.sh.

    Each chunkN.sh is self-contained and can be run in parallel (it discovers the chunking
    total at runtime, computes its own inclusive 1-based range, and runs a single backend
    invocation). organize.sh runs the post-step (organize for anchor/grow, concat for score)
    and writes the result checksum. run.sh is for testing convenience: it runs the chunks
    one-by-one and then organize.sh.
    """
    sigil = _read_sigil(action.path)
    local = deployer.startswith("local")
    location = "local" if local else "remote"
    alaric_dir = _alaric_dir_expr(local)
    output_dir, final_dir, setup = _compute_dirs(action, sigil, local)
    n = max(1, int(nchunks or 1))
    if action.action == "anchor-test" and "conformer" in action.params:
        n = 1

    template_text = _template_path(project, deployer, action.action).read_text()
    if ORGANIZE_DELIM in template_text:
        chunk_tpl, organize_tpl = template_text.split(ORGANIZE_DELIM, 1)
    else:
        chunk_tpl, organize_tpl = template_text, ""

    base_ctx = template_context(action, alaric_dir=alaric_dir, output_dir=output_dir, location=location)
    pp = _pythonpath_line(alaric_dir)
    result_kind = _result_output_kind(action)

    files: dict[str, str] = {}

    for idx in range(1, n + 1):
        ctx = dict(base_ctx)
        ctx["nchunks"] = n
        ctx["chunk_index"] = idx
        body = render_template(chunk_tpl, ctx).strip()
        lines = _prologue(local, sigil)
        # Each chunk is an independent entry point (e.g. submitted to SLURM on its own),
        # so it must verify input materialization itself.
        lines.append("./check.sh")
        lines.append("")
        lines.append(pp)
        lines.append("")
        lines.append(f"mkdir -p {output_dir}")
        lines.append("")
        lines.append(body)
        lines.append("")
        files[f"chunk{idx}.sh"] = "\n".join(lines)

    org_ctx = dict(base_ctx)
    org_ctx["nchunks"] = n
    if action.action == "score":
        org_ctx["score_concat_command"] = str(org_ctx["score_concat_command"]).replace("{{ nchunks }}", str(n))
    org_body = render_template(organize_tpl, org_ctx).strip()
    org_lines = _prologue(local, sigil)
    org_lines.append("")
    org_lines.append(pp)
    org_lines.append("")
    if local:
        org_lines.append("mkdir -p ../CACHE/checksum")
    if org_body:
        org_lines.append(org_body)
    org_lines.extend(_finalize_lines(action, output_dir, final_dir, sigil, result_kind, local, alaric_dir))
    if local:
        org_lines.extend(_local_success_lines(sigil))
    org_lines.append("")
    files["organize.sh"] = "\n".join(org_lines)

    run_lines = _prologue(local, sigil)
    run_lines.append("./check.sh")
    run_lines.append("")
    run_lines.append(pp)
    run_lines.append("")
    run_lines.extend(setup)
    if action.action == "score":
        run_lines.append(f"rm -rf {base_ctx['score_chunks_path']}")
    run_lines.append("")
    for idx in range(1, n + 1):
        run_lines.append(f"bash ./chunk{idx}.sh")
    run_lines.append("bash ./organize.sh")
    run_lines.append("")
    files["run.sh"] = "\n".join(run_lines)
    return files


def _clear_stale_scripts(action_dir: Path) -> None:
    for old in list(action_dir.glob("chunk*.sh")) + [action_dir / "organize.sh"]:
        if old.exists():
            old.unlink()


def _remote_push(*, host: str, action, sigil: str, files: dict[str, str], file_fields: dict[str, Path]) -> None:
    """Ship the generated scripts and DATA file params to the remote.

    DATA is a **global, content-addressed store** shared by every project/deployment:
    ``$ALARIC_REMOTE_DEPLOYMENT_DIR/DATA/<checksum>``. Each file param is uploaded there only
    if its checksum blob is not already present (dedup), uploaded atomically (temp + rename),
    and then **hardlinked** into this action's ``SIGIL/<sigil>`` deployment dir under its
    filename — which the generated scripts reference as ``./<filename>``.
    """
    deploy_root = os.environ["ALARIC_REMOTE_DEPLOYMENT_DIR"].rstrip("/")
    result_root = os.environ["ALARIC_REMOTE_RESULT_DIR"].rstrip("/")
    project_value = os.environ.get("ALARIC_PROJECT", "PROJECT").rstrip("/")
    if not project_value:
        raise MiddleError("ALARIC_PROJECT must not be empty")
    dest = f"{deploy_root}/SIGIL/{sigil}"
    project_dir = project_value if project_value.startswith("/") else f"{deploy_root}/{project_value}"
    project_link_target = posixpath.relpath(dest, project_dir)
    global_data = f"{deploy_root}/DATA"

    subprocess.run(["ssh", host, "mkdir", "-p", dest, global_data, project_dir], check=True)
    subprocess.run(["ssh", host, "ln", "-sfn", project_link_target, f"{project_dir}/{action.name}"], check=True)
    subprocess.run(["ssh", host, "ln", "-sfn", f"{result_root}/{sigil}", f"{dest}/results"], check=True)
    payload = [str(action.path / name) for name in files]
    subprocess.run(["scp", *payload, f"{host}:{dest}/"], check=True)

    for field, src_path in sorted(file_fields.items()):
        checksum = byte_checksum(src_path)
        blob = f"{global_data}/{checksum}"
        present = subprocess.run(["ssh", host, "test", "-e", blob]).returncode == 0
        if not present:
            tmp = f"{blob}.{os.getpid()}.partial"
            subprocess.run(["scp", str(src_path), f"{host}:{tmp}"], check=True)
            subprocess.run(["ssh", host, "mv", "-f", tmp, blob], check=True)
        # Hardlink the content-addressed blob into the deployment dir under its filename.
        subprocess.run(["ssh", host, "ln", "-f", blob, f"{dest}/{src_path.name}"], check=True)


def deploy(deployer: str, action_dir: str | Path = ".", nchunks: int | None = None) -> None:
    if deployer not in {"local", "local-chunk", "remote", "remote-chunk"}:
        raise MiddleError(f"unsupported deployer: {deployer}")
    project = Project.discover(action_dir)
    action = project.get_action_dir(action_dir)
    if not (action.path / "sigil.txt").is_file():
        compute_project_sigils(project, targets=[action.name])
    graph = ActionGraph(project)
    resolved = graph.build([action.name])
    resolved_action = resolved[action.name]
    sigil = _read_sigil(action.path)
    if (action.path / "result.txt").exists():
        raise MiddleError(f"{action.name}: result.txt exists; refusing to deploy")
    cached = project.checksum_dir / sigil
    if cached.is_file():
        shutil.copyfile(cached, action.path / "result.txt")
        return

    is_remote = deployer.startswith("remote")
    if is_remote:
        for var in (
            "ALARIC_REMOTE_HOST",
            "ALARIC_REMOTE_DEPLOYMENT_DIR",
            "ALARIC_REMOTE_ALARIC_DIR",
            "ALARIC_REMOTE_RESULT_DIR",
        ):
            if var not in os.environ:
                raise MiddleError(f"{var} is required for remote deploy")

    is_chunk = deployer.endswith("-chunk")
    location = "remote" if is_remote else "local"
    check = generate_check_sh(project, resolved_action, location=location)

    _clear_stale_scripts(action.path)
    if is_chunk and resolved_action.action in CHUNKABLE:
        files = generate_chunk_files(project, resolved_action, deployer, nchunks)
    else:
        # Non-chunkable actions reuse the non-chunk equivalent deployer/template.
        base_deployer = deployer[: -len("-chunk")] if is_chunk else deployer
        files = {"run.sh": generate_run_sh(project, resolved_action, base_deployer)}
    files["check.sh"] = check

    for name, content in files.items():
        path = action.path / name
        path.write_text(content)
        os.chmod(path, 0o755)

    if is_remote:
        _remote_push(host=os.environ["ALARIC_REMOTE_HOST"], action=action, sigil=sigil, files=files, file_fields=resolved_action.file_fields)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alaric-deploy")
    parser.add_argument("deployer", choices=("local", "local-chunk", "remote", "remote-chunk"))
    parser.add_argument("arg2", nargs="?")
    parser.add_argument("arg3", nargs="?")
    args = parser.parse_args(argv)
    if args.deployer.endswith("-chunk"):
        if args.arg2 is None:
            raise MiddleError("chunk deployer requires nchunks")
        deploy(args.deployer, args.arg3 or ".", int(args.arg2))
    else:
        deploy(args.deployer, args.arg2 or ".")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
