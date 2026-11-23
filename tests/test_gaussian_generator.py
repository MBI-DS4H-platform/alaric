import json
import runpy

import numpy as np
import pytest

from gaussian_common import ROOT
from alaric import gaussian_cache as gc
from alaric import gaussian_filter as gf
from alaric import gaussian_precalc as gp
from alaric import npy_io


def test_rrm50_twice_identical_bytes(tmp_path):
    b1, b2 = gp.build_filter("rrm50"), gp.build_filter("rrm50")
    assert np.array_equal(b1.cells, b2.cells)
    assert np.array_equal(gp.concat_all(b1), gp.concat_all(b2))
    for i, b in enumerate((b1, b2)):
        gp.write_cache(tmp_path / str(i), b)
    for fname in gp.FILE_NAMES:
        a = (tmp_path / "0" / f"{fname}.npy.zst").read_bytes()
        c = (tmp_path / "1" / f"{fname}.npy.zst").read_bytes()
        assert a == c
    assert b1.audit["min_rel_margin"] > 1e-10 and b1.audit["min_tilt_gap"] > 1e-10
    assert (len(b1.sizes), len(b1.translations), b1.pairs) == (7920, 728, 4111232)


def test_canonical_layout_roundtrip(tmp_path):
    b = gp.build_filter("rrm50")
    gp.write_cache(tmp_path / "c", b)
    m = gp.write_manifest(tmp_path / "c", "rrm50", b, "3.30384")
    assert m["counts"]["pairs"] == b.pairs
    sizes = npy_io.load_npy(tmp_path / "c" / "filtered-sizes.npy.zst")
    concat = npy_io.load_npy(tmp_path / "c" / "filtered-concat.npy.zst")
    assert sizes.dtype == np.int32 and concat.dtype == np.int32
    assert int(sizes.sum()) == len(concat) and sizes.min() > 0


def test_audit_failure_is_raised():
    with pytest.raises(gp.DeterminismAuditError):
        gp.accept_pairs(3.30384, audit_min=1.0)


def test_build_twice_identical_npy_sha256(tmp_path):
    from gaussian_common import load_spec

    spec = load_spec()
    m1 = gp.generate("rrm50", tmp_path / "a")
    m2 = gp.generate("rrm50", tmp_path / "b")
    assert {k: v["npy_sha256"] for k, v in m1["files"].items()} == {
        k: v["npy_sha256"] for k, v in m2["files"].items()
    }
    for k, e in spec["thresholds"]["rrm50"]["precalc"]["files"].items():
        assert m1["files"][k]["npy_sha256"] == e["npy_sha256"]


def test_all_generator_verifies_and_rehashes(tmp_path, monkeypatch):
    script = runpy.run_path(str(ROOT / "util" / "gaussian-precalc-all.py"))
    names = gf.all_names()[:2]
    spec = gf.load_spec()
    # Keep validation of the model against the full generator specification.
    full_core = gf.build_spec_core()
    monkeypatch.setattr(gf, "all_names", lambda: names)
    monkeypatch.setattr(gf, "build_spec_core", lambda: full_core)

    first = spec["thresholds"][names[0]]["precalc"]
    first["files"]["rotations"].update(sha256="0" * 64, npy_sha256="0" * 64)
    spec_path = tmp_path / "spec.json"
    original = json.dumps(spec)
    spec_path.write_text(original)
    args = ["--root", str(tmp_path), "--spec", str(spec_path)]

    with pytest.raises(
        gc.GaussianCacheInstallationError, match="compressed file has the wrong content"
    ) as cli:
        script["main"](args)
    with pytest.raises(gc.GaussianCacheInstallationError) as runtime:
        gc.verify_filter_cache(spec, names[0], tmp_path)
    assert str(cli.value) == str(runtime.value)
    assert spec_path.read_text() == original

    directories = [tmp_path / spec["thresholds"][name]["precalc"]["dir"] for name in names]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "manifest.json").write_text("stale manifest")

    build_filter = gp.build_filter
    write_manifest = gp.write_manifest

    def build(name):
        assert all(not (directory / "manifest.json").exists() for directory in directories)
        return build_filter(name)

    def manifest(*args):
        assert all((directory / "rotations.npy.zst").is_file() for directory in directories)
        assert spec_path.read_text() == original
        return write_manifest(*args)

    monkeypatch.setattr(gp, "build_filter", build)
    monkeypatch.setattr(gp, "write_manifest", manifest)
    assert script["main"]([*args, "--rehash"]) == 0
    refreshed = gf.load_spec(spec_path)
    assert refreshed["spec_digest"] == spec["spec_digest"]
    for name, directory in zip(names, directories):
        assert len(gc.verify_filter_cache(refreshed, name, tmp_path)) == 4
        manifest_files = json.loads((directory / "manifest.json").read_text())["files"]
        assert refreshed["thresholds"][name]["precalc"]["files"] == manifest_files
