from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / ".alaric"))
sys.path.insert(0, str(HERE / ".util"))

from poses import pack_pool, write_arc_file  # noqa: E402


def _load_wrapper():
    path = HERE / ".util" / "alaric-rmsd"
    loader = importlib.machinery.SourceFileLoader("alaric_rmsd_script", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


wrapper = _load_wrapper()

# refe-best-fit.tsv, fragment 1 (GU): the pose and its known RMSD to rna.pdb
EXPECTED = 1.050039


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "DATA").mkdir(parents=True)
    shutil.copy(HERE / "pdbs" / "rna.pdb", root / "DATA" / "reference.pdb")
    (root / "DATA" / "pdbcode.txt").write_text("1b7f\n")
    pose_dir = root / "frag1-test" / "results"
    packed = pack_pool(
        np.array([5378 - 1], dtype=np.uint16),
        np.array([5232 - 1], dtype=np.uint16),
        np.array([[73, 97, 193]], dtype=np.int16),
        bucket_size=1,
    )
    M, O, C, P = packed[0]
    write_arc_file(pose_dir / "poses-1.arc", M, O, C, P, bucket_size=1, zstd=False)
    return pose_dir


def test_small_pool_writes_text(tmp_path: Path, capsys) -> None:
    pose_dir = _project(tmp_path)
    assert wrapper.main([str(pose_dir)]) == 0
    output = pose_dir / "rmsd.txt"
    assert capsys.readouterr().out.strip().endswith(f"wrote {output}")
    assert not (pose_dir / "rmsd.npy").exists()
    assert float(output.read_text()) == pytest.approx(EXPECTED, abs=1e-3)


def test_large_pool_writes_npy_and_removes_stale_txt(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    pose_dir = _project(tmp_path)
    (pose_dir / "rmsd.txt").write_text("stale\n")
    monkeypatch.setattr(wrapper, "NPY_THRESHOLD", 0)
    assert wrapper.main([str(pose_dir)]) == 0
    output = pose_dir / "rmsd.npy"
    assert capsys.readouterr().out.strip().endswith(f"wrote {output}")
    assert not (pose_dir / "rmsd.txt").exists()
    assert np.load(output)[0] == pytest.approx(EXPECTED, abs=1e-3)


def test_threshold_is_100k() -> None:
    assert wrapper.NPY_THRESHOLD == 100_000


def test_fragment_from_symlinked_results(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "CACHE" / "results" / "SIG").mkdir(parents=True)
    (root / "frag4-anchor-fwd").mkdir()
    link = root / "frag4-anchor-fwd" / "results"
    link.symlink_to("../CACHE/results/SIG")
    assert wrapper._fragment_from_path(link) == 4
    assert wrapper._fragment_from_path(root / "CACHE" / "results" / "SIG") is None
    assert wrapper._fragment_from_path(root / "xfrag4-a") is None


def test_missing_fragment_is_an_error(tmp_path: Path, capsys) -> None:
    pose_dir = _project(tmp_path)
    renamed = pose_dir.parent.rename(pose_dir.parent.with_name("anchors"))
    assert wrapper.main([str(renamed / "results")]) == 2
    assert "--fragment" in capsys.readouterr().err
    # an explicit fragment makes it work
    assert wrapper.main([str(renamed / "results"), "--fragment", "1"]) == 0


def test_action_dir_writes_into_results(tmp_path: Path, capsys) -> None:
    pose_dir = _project(tmp_path)
    action_dir = pose_dir.parent  # frag1-test/, holds results/
    assert wrapper.main([str(action_dir)]) == 0
    assert capsys.readouterr().out.strip().endswith(f"wrote {action_dir / 'results' / 'rmsd.txt'}")
    assert float((pose_dir / "rmsd.txt").read_text()) == pytest.approx(EXPECTED, abs=1e-3)
    assert not (action_dir / "rmsd.txt").exists()


def test_action_dir_with_symlinked_results(tmp_path: Path) -> None:
    pose_dir = _project(tmp_path)
    root = pose_dir.parent.parent
    cache = root / "CACHE" / "results" / "SIG"
    cache.parent.mkdir(parents=True)
    pose_dir.rename(cache)
    pose_dir.symlink_to("../CACHE/results/SIG")
    assert wrapper.main([str(pose_dir.parent)]) == 0
    assert (cache / "rmsd.txt").is_file()


def test_action_dir_without_usable_results(tmp_path: Path, capsys) -> None:
    pose_dir = _project(tmp_path)
    root = pose_dir.parent.parent
    (root / "frag2-new").mkdir()
    assert wrapper.main([str(root / "frag2-new")]) == 2
    assert "not run yet" in capsys.readouterr().err
    (root / "frag2-new" / "results").symlink_to("../CACHE/results/GONE")
    assert wrapper.main([str(root / "frag2-new")]) == 2
    assert "dangling" in capsys.readouterr().err
