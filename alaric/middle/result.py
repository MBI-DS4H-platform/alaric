from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from .errors import ResultError
from .project import Project


def _sigil(action_dir: Path) -> str:
    path = action_dir / "sigil.txt"
    if not path.is_file():
        raise ResultError(f"{action_dir}: missing sigil.txt")
    return path.read_text().strip()


def _checksum(project: Project, sigil: str) -> Path:
    return project.checksum_dir / sigil


def _local_result(project: Project, sigil: str) -> Path:
    return project.results_dir / sigil


def _remote_base(sigil: str) -> str:
    result_dir = os.environ.get("ALARIC_REMOTE_RESULT_DIR")
    if not result_dir:
        raise ResultError("ALARIC_REMOTE_RESULT_DIR is required")
    return result_dir.rstrip("/") + "/" + sigil


def clean(action_dir: str | Path = ".", *, all_state: bool = False) -> None:
    project = Project.discover(action_dir)
    action = project.get_action_dir(action_dir)
    sigil = _sigil(action.path)
    result = _local_result(project, sigil)
    if result.exists():
        shutil.rmtree(result)
    if all_state:
        (action.path / "result.txt").unlink(missing_ok=True)
        _checksum(project, sigil).unlink(missing_ok=True)


def check(action_dir: str | Path = ".") -> None:
    project = Project.discover(action_dir)
    action = project.get_action_dir(action_dir)
    sigil = _sigil(action.path)
    if (action.path / "result.txt").exists():
        raise ResultError(f"{action.name}: result.txt already exists")
    local_checksum = _checksum(project, sigil)
    if local_checksum.is_file():
        shutil.copyfile(local_checksum, action.path / "result.txt")
        return
    host = os.environ.get("ALARIC_REMOTE_HOST")
    if not host:
        raise ResultError("no local checksum and ALARIC_REMOTE_HOST is unset")
    remote = _remote_base(sigil)
    proc = subprocess.run(
        ["ssh", host, f"cat {remote}.CHECKSUM 2>/dev/null || cat {remote}/score.npy.CHECKSUM 2>/dev/null || cat {remote}/mask.npy.CHECKSUM"],
        check=True,
        text=True,
        capture_output=True,
    )
    checksum = proc.stdout.strip()
    if not checksum:
        raise ResultError(f"remote result checksum not found for {sigil}")
    project.ensure_cache()
    local_checksum.write_text(checksum + "\n")
    (action.path / "result.txt").write_text(checksum + "\n")


def download(action_dir: str | Path = ".") -> None:
    project = Project.discover(action_dir)
    action = project.get_action_dir(action_dir)
    sigil = _sigil(action.path)
    result = _local_result(project, sigil)
    if result.exists():
        raise ResultError(f"local result already exists: {result}")
    host = os.environ.get("ALARIC_REMOTE_HOST")
    if not host:
        raise ResultError("ALARIC_REMOTE_HOST is required")
    tmp = result.with_name(result.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    project.ensure_cache()
    subprocess.run(["rsync", "-a", "--partial", f"{host}:{_remote_base(sigil)}/", str(tmp) + "/"], check=True)
    tmp.rename(result)
    check(action_dir)
    link = action.path / "results"
    link.unlink(missing_ok=True)
    link.symlink_to(os.path.relpath(result, action.path))


def upload(action_dir: str | Path = ".") -> None:
    project = Project.discover(action_dir)
    action = project.get_action_dir(action_dir)
    sigil = _sigil(action.path)
    result = _local_result(project, sigil)
    checksum = _checksum(project, sigil)
    if not (action.path / "result.txt").is_file() or not checksum.is_file() or not result.exists():
        raise ResultError("result.txt, CACHE/checksum/<SIGIL>, and CACHE/results/<SIGIL> are required")
    host = os.environ.get("ALARIC_REMOTE_HOST")
    if not host:
        raise ResultError("ALARIC_REMOTE_HOST is required")
    remote = _remote_base(sigil)
    partial = remote + ".partial"
    subprocess.run(["ssh", host, f"rm -rf {partial} && mkdir -p {partial}"], check=True)
    subprocess.run(["rsync", "-a", "--partial", str(result) + "/", f"{host}:{partial}/"], check=True)
    subprocess.run(["scp", str(checksum), f"{host}:{partial}.CHECKSUM"], check=True)
    subprocess.run(["ssh", host, f"rm -rf {remote} && mv {partial} {remote} && mv {partial}.CHECKSUM {remote}.CHECKSUM"], check=True)


def clean_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alaric-result-clean")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("action_dir", nargs="?", default=".")
    args = parser.parse_args(argv)
    clean(args.action_dir, all_state=args.all)
    return 0


def check_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alaric-result-check")
    parser.add_argument("action_dir", nargs="?", default=".")
    args = parser.parse_args(argv)
    check(args.action_dir)
    return 0


def download_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alaric-result-download")
    parser.add_argument("action_dir", nargs="?", default=".")
    args = parser.parse_args(argv)
    download(args.action_dir)
    return 0


def upload_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alaric-result-upload")
    parser.add_argument("action_dir", nargs="?", default=".")
    args = parser.parse_args(argv)
    upload(args.action_dir)
    return 0
