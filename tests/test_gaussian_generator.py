import numpy as np
import pytest

from gaussian_common import ROOT  # noqa: F401
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
