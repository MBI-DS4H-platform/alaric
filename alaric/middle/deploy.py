from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from .backend import render_template, template_context
from .errors import MiddleError
from .graph import ActionGraph
from .project import Project
from .resolve import ResolvedAction
from .schema import DEPENDENCY_FIELDS, OUTPUT_KIND


def _repo_alaric_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _template_path(project: Project, deployer: str, action_name: str) -> Path:
    candidates = [
        project.root / "templates" / deployer / f"{action_name}.sh",
        Path(__file__).resolve().parent / "templates" / deployer / f"{action_name}.sh",
    ]
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


def generate_check_sh(project: Project, action: ResolvedAction, *, location: str) -> str:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for field in DEPENDENCY_FIELDS[action.action]:
        dep = action.params.get(field)
        if not isinstance(dep, ResolvedAction):
            continue
        dep_sigil = _read_sigil(dep.path)
        rel = f"../{dep.name}"
        result_path = (
            f"${{ALARIC_REMOTE_RESULT_DIR}}/{dep_sigil}"
            if location == "remote"
            else f"../CACHE/results/{dep_sigil}"
        )
        lines.extend(
            [
                f'test "$(cat {rel}/sigil.txt)" = "{dep_sigil}"',
                f"test -s {rel}/result.txt",
                f"test -e {result_path}",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _sidecar_command(kind: str, result_dir: str, sigil: str, *, local: bool) -> str:
    if kind == "pose":
        cmd = f'"${{PYTHON:-python}}" "${{ALARIC_DIR:-{_repo_alaric_dir()}}}/middle/result_sidecar.py" pose {result_dir}'
        if local:
            cmd += f"\ncp {result_dir}.CHECKSUM ../CACHE/checksum/{sigil}"
        return cmd
    name = "mask.npy" if kind == "mask" else "score.npy"
    cmd = f'"${{PYTHON:-python}}" "${{ALARIC_DIR:-{_repo_alaric_dir()}}}/middle/result_sidecar.py" array {result_dir}/{name}'
    if local:
        cmd += f"\ncp {result_dir}/{name}.CHECKSUM ../CACHE/checksum/{sigil}"
    return cmd


def generate_run_sh(project: Project, action: ResolvedAction, deployer: str, nchunks: int | None = None) -> str:
    sigil = _read_sigil(action.path)
    local = deployer.startswith("local")
    location = "local" if local else "remote"
    alaric_dir = '"${ALARIC_DIR:-' + str(_repo_alaric_dir()) + '}"' if local else '"${ALARIC_REMOTE_ALARIC_DIR:?}"'
    output_dir = f"../CACHE/results/{sigil}" if local else f"${{ALARIC_REMOTE_RESULT_DIR:?}}/{sigil}.partial"
    final_dir = f"../CACHE/results/{sigil}" if local else f"${{ALARIC_REMOTE_RESULT_DIR:?}}/{sigil}"
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", 'cd "$(dirname "$0")"', "./check.sh", ""]
    lines.append(f"export PYTHONPATH={alaric_dir}${{PYTHONPATH:+:${{PYTHONPATH}}}}")
    lines.append("")
    if local:
        lines.extend([f"rm -rf {output_dir}", f"mkdir -p {output_dir} ../CACHE/checksum"])
    else:
        scratch = f"${{ALARIC_REMOTE_SCRATCH_DIR:-${{TMPDIR:-/tmp}}}}/{sigil}"
        if action.action in {"anchor", "anchor-test", "grow"}:
            output_dir = scratch
            lines.extend([f"rm -rf {scratch} {final_dir}.partial", f"mkdir -p {scratch} ${{ALARIC_REMOTE_RESULT_DIR:?}}"])
        else:
            lines.extend([f"rm -rf {output_dir} {final_dir}", f"mkdir -p {output_dir} ${{ALARIC_REMOTE_RESULT_DIR:?}}"])
    if deployer.endswith("-chunk"):
        lines.extend(
            render_action_template(
                project,
                action,
                deployer=deployer,
                alaric_dir=alaric_dir,
                output_dir=output_dir,
                location=location,
                extra_context={"nchunks": int(nchunks or 1)},
            )
        )
    else:
        lines.extend(
            render_action_template(
                project,
                action,
                deployer=deployer,
                alaric_dir=alaric_dir,
                output_dir=output_dir,
                location=location,
            )
        )
    result_kind = _result_output_kind(action)
    lines.append(_sidecar_command(result_kind, output_dir, sigil, local=local))
    if not local:
        if action.action in {"anchor", "anchor-test", "grow"}:
            lines.extend(
                [
                    f"mv {output_dir}.INDEX {final_dir}.partial.INDEX",
                    f"mv {output_dir}.CHECKSUM {final_dir}.partial.CHECKSUM",
                    f"mv {output_dir} {final_dir}.partial",
                    f"mv {final_dir}.partial {final_dir}",
                    f"mv {final_dir}.partial.INDEX {final_dir}.INDEX",
                    f"mv {final_dir}.partial.CHECKSUM {final_dir}.CHECKSUM",
                ]
            )
        else:
            if result_kind == "pose":
                lines.extend(
                    [
                        f"mv {output_dir}.INDEX {final_dir}.INDEX",
                        f"mv {output_dir}.CHECKSUM {final_dir}.CHECKSUM",
                    ]
                )
            lines.extend([f"mv {output_dir} {final_dir}"])
    lines.append("")
    return "\n".join(lines)


def deploy(deployer: str, action_dir: str | Path = ".", nchunks: int | None = None) -> None:
    if deployer not in {"local", "local-chunk", "remote", "remote-chunk"}:
        raise MiddleError(f"unsupported deployer: {deployer}")
    project = Project.discover(action_dir)
    graph = ActionGraph(project)
    resolved = graph.build()
    action = project.get_action_dir(action_dir)
    resolved_action = resolved[action.name]
    sigil = _read_sigil(action.path)
    if (action.path / "result.txt").exists():
        raise MiddleError(f"{action.name}: result.txt exists; refusing to deploy")
    cached = project.checksum_dir / sigil
    if cached.is_file():
        shutil.copyfile(cached, action.path / "result.txt")
        return
    check = generate_check_sh(project, resolved_action, location="local" if deployer.startswith("local") else "remote")
    run = generate_run_sh(project, resolved_action, deployer, nchunks)
    (action.path / "check.sh").write_text(check)
    (action.path / "run.sh").write_text(run)
    os.chmod(action.path / "check.sh", 0o755)
    os.chmod(action.path / "run.sh", 0o755)
    if deployer.startswith("remote"):
        for var in ("ALARIC_REMOTE_HOST", "ALARIC_REMOTE_DEPLOYMENT_DIR", "ALARIC_REMOTE_ALARIC_DIR", "ALARIC_REMOTE_RESULT_DIR"):
            if var not in os.environ:
                raise MiddleError(f"{var} is required for remote deploy")
        host = os.environ["ALARIC_REMOTE_HOST"]
        dest = os.environ["ALARIC_REMOTE_DEPLOYMENT_DIR"].rstrip("/") + f"/{sigil}"
        subprocess.run(["ssh", host, "mkdir", "-p", dest], check=True)
        subprocess.run(["scp", str(action.path / "run.sh"), str(action.path / "check.sh"), f"{host}:{dest}/"], check=True)


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
