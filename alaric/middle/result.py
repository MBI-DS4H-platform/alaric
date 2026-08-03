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
    result_txt = action.path / "result.txt"
    local_checksum = _checksum(project, sigil)
    if local_checksum.is_file():
        if result_txt.exists():
            if result_txt.read_text().strip() != local_checksum.read_text().strip():
                raise ResultError(
                    f"{action.name}: result.txt does not match the cached checksum"
                )
        else:
            shutil.copyfile(local_checksum, result_txt)
        return
    host = os.environ.get("ALARIC_REMOTE_HOST")
    if not host:
        raise ResultError("no local checksum and ALARIC_REMOTE_HOST is unset")
    remote = _remote_base(sigil)
    proc = subprocess.run(
        [
            "ssh",
            host,
            f"cat {remote}.CHECKSUM 2>/dev/null || cat {remote}/score.npy.CHECKSUM 2>/dev/null || cat {remote}/mask.npy.CHECKSUM",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    checksum = proc.stdout.strip()
    if not checksum:
        raise ResultError(f"remote result checksum not found for {sigil}")
    if result_txt.exists():
        if result_txt.read_text().strip() != checksum:
            raise ResultError(
                f"{action.name}: result.txt does not match the remote checksum"
            )
    else:
        result_txt.write_text(checksum + "\n")
    project.ensure_cache()
    local_checksum.write_text(checksum + "\n")


def download(action_dir: str | Path = ".") -> None:
    _download(action_dir)


def _download(action_dir: str | Path, *, provenance_only: bool = False) -> None:
    project = Project.discover(action_dir)
    action = project.get_action_dir(action_dir)
    sigil = _sigil(action.path)
    result = _local_result(project, sigil)
    merge_into_existing = provenance_only and result.is_dir()
    if result.exists() and not merge_into_existing:
        raise ResultError(f"local result already exists: {result}")
    host = os.environ.get("ALARIC_REMOTE_HOST")
    if not host:
        raise ResultError("ALARIC_REMOTE_HOST is required")
    tmp = result.with_name(result.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    project.ensure_cache()
    command = ["rsync", "-a", "--partial"]
    if provenance_only:
        # Directories are retained so this remains correct if a future result layout
        # puts an index array below a subdirectory. Everything else, notably poses,
        # is excluded.
        command.extend(
            [
                "--include=*/",
                "--include=provenance.npy",
                "--include=provenance.npy.zst",
                "--include=prop-provenance.npy",
                "--include=prop-provenance.npy.zst",
                "--include=prop-pair.npy",
                "--include=prop-pair.npy.zst",
                "--include=map-*.npy",
                "--include=map-*.npy.zst",
                "--exclude=*",
            ]
        )
    command.extend([f"{host}:{_remote_base(sigil)}/", str(tmp) + "/"])
    subprocess.run(command, check=True)
    if merge_into_existing:
        # A representative's poses can already be materialized locally while its
        # upstream lineage exists only remotely.  The selective staging directory
        # contains no poses, so merging it can only add/refresh provenance arrays.
        subprocess.run(
            ["rsync", "-a", "--partial", str(tmp) + "/", str(result) + "/"],
            check=True,
        )
        shutil.rmtree(tmp)
    else:
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
    if (
        not (action.path / "result.txt").is_file()
        or not checksum.is_file()
        or not result.exists()
    ):
        raise ResultError(
            "result.txt, CACHE/checksum/<SIGIL>, and CACHE/results/<SIGIL> are required"
        )
    host = os.environ.get("ALARIC_REMOTE_HOST")
    if not host:
        raise ResultError("ALARIC_REMOTE_HOST is required")
    remote = _remote_base(sigil)
    partial = remote + ".partial"
    subprocess.run(["ssh", host, f"rm -rf {partial} && mkdir -p {partial}"], check=True)
    subprocess.run(
        ["rsync", "-a", "--partial", str(result) + "/", f"{host}:{partial}/"],
        check=True,
    )
    subprocess.run(["scp", str(checksum), f"{host}:{partial}.CHECKSUM"], check=True)
    subprocess.run(
        [
            "ssh",
            host,
            f"rm -rf {remote} && mv {partial} {remote} && mv {partial}.CHECKSUM {remote}.CHECKSUM",
        ],
        check=True,
    )


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


def provenance_download_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alaric-provenance-download")
    parser.add_argument("action_dir", nargs="?", default=".")
    args = parser.parse_args(argv)
    _download(args.action_dir, provenance_only=True)
    return 0


def upload_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alaric-result-upload")
    parser.add_argument("action_dir", nargs="?", default=".")
    args = parser.parse_args(argv)
    upload(args.action_dir)
    return 0
