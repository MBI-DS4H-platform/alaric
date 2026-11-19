from __future__ import annotations

from pathlib import Path
import subprocess

SCRIPT = Path(__file__).resolve().parents[2] / "util" / "alaric-copy"


def _action(parent: Path, name: str, *, used: bool = False) -> Path:
    path = parent / name
    path.mkdir(parents=True)
    (path / "alaric.yaml").write_text(f"action: identity\n# {name}\n")
    if used:
        # The state of an action dir in a run that has already been deployed and run.
        (path / "sigil.txt").write_text("0123456789-null\n")
        (path / "result.txt").write_text("abc\n")
        (path / "run.sh").write_text("#!/bin/sh\n")
        (path / "results").symlink_to("../CACHE/results/0123456789-null")
    return path


def _copy(cwd: Path, *dirs: Path | str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), *map(str, dirs)], cwd=cwd, capture_output=True, text=True
    )


def test_copies_several_dirs_yaml_only(tmp_path: Path) -> None:
    gen = tmp_path / "gen-frag2-3"
    sources = [_action(gen, "frag2-anchor-raw", used=True), _action(gen, "frag2")]
    run = tmp_path / "run"
    run.mkdir()
    proc = _copy(run, *sorted(gen.glob("*")))
    assert proc.returncode == 0, proc.stderr
    for src in sources:
        dest = run / src.name
        assert sorted(p.name for p in dest.iterdir()) == ["alaric.yaml"]
        assert (dest / "alaric.yaml").read_text() == (src / "alaric.yaml").read_text()


def test_trailing_slash(tmp_path: Path) -> None:
    src = _action(tmp_path / "gen", "frag4")
    run = tmp_path / "run"
    run.mkdir()
    assert _copy(run, f"{src}/").returncode == 0
    assert (run / "frag4" / "alaric.yaml").is_file()


def test_existing_destination_copies_nothing(tmp_path: Path) -> None:
    gen = tmp_path / "gen"
    a, b = _action(gen, "frag2-anchor-raw"), _action(gen, "frag3-anchor-raw")
    run = tmp_path / "run"
    (run / b.name).mkdir(parents=True)
    proc = _copy(run, a, b)
    assert proc.returncode == 1
    assert "destination already exists: frag3-anchor-raw" in proc.stderr
    assert not (run / a.name).exists()


def test_missing_yaml_copies_nothing(tmp_path: Path) -> None:
    a = _action(tmp_path / "gen", "frag2")
    bad = tmp_path / "gen" / "notes"
    bad.mkdir()
    run = tmp_path / "run"
    run.mkdir()
    proc = _copy(run, a, bad)
    assert proc.returncode == 1
    assert "source file does not exist" in proc.stderr
    assert list(run.iterdir()) == []


def test_same_name_twice_copies_nothing(tmp_path: Path) -> None:
    a = _action(tmp_path / "gen-a", "frag3")
    b = _action(tmp_path / "gen-b", "frag3")
    run = tmp_path / "run"
    run.mkdir()
    proc = _copy(run, a, b)
    assert proc.returncode == 1
    assert "two sources have the same name: frag3" in proc.stderr
    assert list(run.iterdir()) == []


def test_no_arguments(tmp_path: Path) -> None:
    proc = _copy(tmp_path)
    assert proc.returncode == 2
    assert "Usage" in proc.stderr
