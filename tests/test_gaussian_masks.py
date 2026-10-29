import math
from fractions import Fraction

import numpy as np
import pytest

from gaussian_common import load_spec
from alaric import gaussian_filter as gf


def test_lattice_order_and_origin():
    ijk, t = gf.lattice()
    assert len(ijk) == 41 ** 3 - 1
    assert not np.any(np.all(ijk == 0, axis=1))
    assert tuple(ijk[0]) == (-20, -20, -20) and tuple(ijk[1]) == (-20, -20, -19)
    assert np.array_equal(t, ijk * 0.25)
    keys = ijk[:, 0] * 41 * 41 + ijk[:, 1] * 41 + ijk[:, 2]
    assert np.all(np.diff(keys) > 0)


def test_hopf_grid():
    R, absr = gf.hopf_rotations()
    assert R.shape == (98304, 3, 3)
    assert np.allclose(np.swapaxes(R, 1, 2) @ R, np.eye(3), atol=1e-12)
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-12)
    assert np.array_equal(absr, np.abs(R[:, 2, 2]))
    assert len(np.unique(np.round(R.reshape(-1, 9), 9), axis=0)) == 98304


@pytest.mark.parametrize("disp45", [False, True])
def test_exhaustive_integer_masks(disp45):
    ijk, _ = gf.lattice()
    got = gf.mask_translation(ijk, disp45=disp45)
    tan60sq = 3
    for (i, j, k), g in zip(ijk.tolist(), got.tolist()):
        x, y, z = Fraction(i, 4), Fraction(j, 4), Fraction(k, 4)
        want = x * x + y * y + z * z < 25 and tan60sq * z * z > x * x + y * y
        if disp45:
            want = want and z * z > x * x + y * y
        assert want == g


def test_lattice_ties():
    def ok(x, y, z, disp45=False):
        ijk = [round(v * 4) for v in (x, y, z)]
        return bool(gf.mask_translation(ijk, disp45=disp45))

    assert not ok(0.75, 1.0, 1.25, disp45=True)  # phi = 45 exactly
    assert ok(0.75, 1.0, 1.25, disp45=False)
    assert not ok(0, 0, 5) and not ok(0, 3, 4)  # d = 5
    assert not ok(3, 4, 0) and not ok(5, 0, 0)  # phi >= 60
    assert ok(0, 0, 4.75)
    # phi = 60 ties are impossible: i^2+j^2 = 3k^2 has no nonzero solution
    ijk, _ = gf.lattice()
    s = ijk[:, 0] ** 2 + ijk[:, 1] ** 2
    assert not np.any((s == 3 * ijk[:, 2] ** 2))


def test_tilt_predicate_on_stored_r33():
    R, absr = gf.hopf_rotations()
    m = gf.mask_rotation(R[:, 2, 2], True)
    assert np.array_equal(m, 4.0 * R[:, 2, 2] ** 2 > 3.0)
    assert np.array_equal(m, np.arccos(absr) < math.radians(30))  # no near ties
    assert np.abs(4 * absr ** 2 - 3).min() > 1e-4
    assert gf.mask_rotation(R[:, 2, 2], False).all()


def test_descriptor_predicate_boundaries_from_json():
    spec = load_spec()
    for e in spec["golden"]["decisions"]:
        want = gf.accept_descriptor(e["name"], e["d"], e["phi_deg"] * gf.DEG, e["theta_deg"] * gf.DEG)
        assert want == e["accept"], e
    dec = spec["golden"]["decisions"]
    fam = lambda e: e["name"].startswith(("rrm90", "rrm95"))  # score >= 0.27592 there
    exact = [e for e in dec if e["phi_deg"] == 45.0 and fam(e)]
    assert exact
    for e in exact:
        assert e["accept"] == ("disp45" not in e["name"])
    inside = [e for e in dec if e["phi_deg"] == 44.99999 and fam(e)]
    assert inside and all(e["accept"] for e in inside)
    t30 = [e for e in dec if e["theta_deg"] == 30.0 and fam(e)]
    assert t30 and all(e["accept"] == ("tilt30" not in e["name"]) for e in t30)
    t29 = [e for e in dec if e["theta_deg"] == 29.99999 and fam(e)]
    assert t29 and all(e["accept"] for e in t29)
    lat = {(e["name"], tuple(e["xyz"])): e["mask_accept"] for e in spec["golden"]["lattice_decisions"]}
    assert lat[("rrm95-disp45", (0.75, 1.0, 1.25))] is False
    assert lat[("rrm95", (0.75, 1.0, 1.25))] is True
    assert lat[("rrm95", (0.0, 0.0, 5.0))] is False
    assert lat[("rrm95", (0.0, 3.0, 4.0))] is False
    assert lat[("rrm95", (3.0, 4.0, 0.0))] is False
    assert lat[("rrm95", (5.0, 0.0, 0.0))] is False
    assert lat[("rrm95", (0.0, 0.0, 4.75))] is True


@pytest.mark.parametrize("base", [50, 60, 70, 80, 90, 95])
def test_cutoff_straddle(base):
    """At the mean phi/theta, c(1+1e-9) and c(1-1e-9) on the d > mu_d ray flip accept."""
    import math

    name = f"rrm{base}"
    c = gf.threshold_spec(name)["cutoff"]
    phi, theta = gf.MEAN[1], gf.MEAN[2]
    pts = {}
    for sign in (+1, -1):
        target = c * (1 + sign * 1e-9)
        lo, hi = gf.MEAN[0], 5.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if gf.score(mid, phi, theta) >= target:
                lo = mid
            else:
                hi = mid
        pts[sign] = lo if sign > 0 else hi
    for sign, d in pts.items():
        s = float(gf.score(d, phi, theta))
        assert (s >= c) == (sign > 0), (name, sign, s, c)
        assert gf.accept_descriptor(name, d, phi, theta) == (sign > 0)
    assert pts[+1] < pts[-1]
    spec = load_spec()
    golden = {(e["name"], e["d"]): e["accept"] for e in spec["golden"]["decisions"]}
    assert golden[(name, pts[+1])] is True and golden[(name, pts[-1])] is False
