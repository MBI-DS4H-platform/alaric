import numpy as np
import pytest

from gaussian_common import ROOT, load_spec, require_caches
from alaric import gaussian_cache as gc
from alaric import gaussian_filter as gf
from alaric import npy_io
import sys

sys.path.insert(0, str(ROOT / "alaric"))
import anchor  # noqa: E402

pytestmark = pytest.mark.gaussian_caches
NAMES = gf.all_names()
_LOADED = {}
_LOADED_MAX = 3  # keep memory bounded (rrm95 concat is ~700 MB)


def _paths(name):
    require_caches([name])
    return gc.verify_filter_cache(load_spec(), name, ROOT)


def _load(name):
    if name not in _LOADED:
        while len(_LOADED) >= _LOADED_MAX:
            _LOADED.pop(next(iter(_LOADED)))
        p = _paths(name)
        _LOADED[name] = {k: npy_io.load_npy(v) for k, v in p.items()}
    return _LOADED[name]


@pytest.mark.parametrize("name", NAMES)
def test_checksums_match_json(name):
    spec = load_spec()
    paths = _paths(name)  # raises on mismatch
    files = spec["thresholds"][name]["precalc"]["files"]
    for k, p in paths.items():
        shape, dtype = npy_io.read_npy_header(p)
        assert list(shape) == files[k]["shape"] and dtype.str == files[k]["dtype"]


def _lattice_flat(t):
    i = np.rint(t / 0.25).astype(np.int64) + 20
    return (i[:, 0] * 41 + i[:, 1]) * 41 + i[:, 2]


@pytest.mark.parametrize("name", NAMES)
def test_csr_invariants_and_runtime_loader(name):
    a = _load(name)
    rot, tr, sizes, concat = a["rotations"], a["translations"], a["filtered-sizes"], a["filtered-concat"]
    _, tilt, disp = gf.parse_name(name)
    assert rot.dtype == tr.dtype == np.float64
    assert sizes.dtype == concat.dtype == np.int32
    assert rot.ndim == 3 and rot.flags.c_contiguous and tr.flags.c_contiguous
    assert len(sizes) == len(rot) and int(sizes.sum(dtype=np.int64)) == len(concat)
    assert sizes.min() > 0
    assert 0 <= concat.min() and concat.max() < len(tr)
    off = np.concatenate([[0], np.cumsum(sizes, dtype=np.int64)])
    inrow = np.ones(len(concat), dtype=bool)
    inrow[off[:-1]] = False  # first element of each row has no predecessor in-row
    assert np.all(np.diff(concat.astype(np.int64), prepend=-1)[inrow] > 0)
    assert np.all(np.bincount(concat, minlength=len(tr)) > 0)  # no unused translation
    cells = anchor._hopf_cell_indices(rot)
    assert np.all(np.diff(cells) > 0)
    R_all, _ = gf.hopf_rotations()
    assert np.array_equal(rot, R_all[cells])  # bitwise
    ijk = tr / 0.25
    assert np.array_equal(ijk, np.rint(ijk))
    assert gf.mask_translation(np.rint(ijk).astype(np.int64), disp45=disp).all()
    assert np.all(np.diff(_lattice_flat(tr)) > 0)  # strictly increasing: C order
    if tilt:
        assert (4.0 * rot[:, 2, 2] ** 2 > 3.0).all()
    # runtime compatibility
    anchor._validate_precalculated_arrays(rot, tr, sizes, concat)
    anchor._build_cell_lookup(rot)
    anchor._lattice_geometry(tr)


def _members(name, cells_rows, tidx):
    a = _load(name)
    off = np.concatenate([[0], np.cumsum(a["filtered-sizes"], dtype=np.int64)])
    out = np.zeros(len(cells_rows), dtype=bool)
    for n, (r, t) in enumerate(zip(cells_rows, tidx)):
        row = a["filtered-concat"][off[r] : off[r + 1]]
        k = np.searchsorted(row, t)
        out[n] = k < len(row) and row[k] == t
    return out


@pytest.mark.parametrize("name", ["rrm60", "rrm90-tilt30", "rrm95-disp45"])
def test_random_membership_matches_model(name):
    a = _load(name)
    spec = gf.threshold_spec(name)
    _, tilt, disp = gf.parse_name(name)
    R_all, _ = gf.hopf_rotations()
    ijk_all, t_all = gf.lattice()
    rng = np.random.default_rng(7)
    n = 100_000
    # 1) uniform over (cell, lattice point)
    cells = rng.integers(0, gf.HOPF_CELLS, n)
    pts = rng.integers(0, len(t_all), n)
    # 2) drawn from members
    off = np.concatenate([[0], np.cumsum(a["filtered-sizes"], dtype=np.int64)])
    rows = rng.integers(0, len(a["rotations"]), n)
    pick = off[rows] + (rng.random(n) * a["filtered-sizes"][rows]).astype(np.int64)
    cell_of_row = anchor._hopf_cell_indices(a["rotations"])
    tr_member = a["translations"][a["filtered-concat"][pick]]
    # translation index in full lattice from ijk
    def lat_index(t):
        i = np.rint(t / 0.25).astype(np.int64) + 20
        flat = (i[:, 0] * 41 + i[:, 1]) * 41 + i[:, 2]
        origin = (20 * 41 + 20) * 41 + 20
        return flat - (flat > origin)
    cells = np.concatenate([cells, cell_of_row[rows]])
    pts = np.concatenate([pts, lat_index(tr_member)])
    # 3) near the cutoff
    cand_c = rng.integers(0, gf.HOPF_CELLS, 2_000_000)
    cand_p = rng.integers(0, len(t_all), 2_000_000)
    s = gf.score(*gf.descriptors(R_all[cand_c], t_all[cand_p]))
    sel = (s > spec["cutoff"] / 2) & (s < spec["cutoff"] * 2) & ~np.isnan(s)
    cells = np.concatenate([cells, cand_c[sel][:n]])
    pts = np.concatenate([pts, cand_p[sel][:n]])
    assert sel.sum() > 1000

    R = R_all[cells]
    t = t_all[pts]
    d, phi, theta = gf.descriptors(R, t)
    want = gf.score(d, phi, theta) >= spec["cutoff"]
    want &= gf.mask_translation(ijk_all[pts], disp45=disp)
    want &= gf.mask_rotation(R[:, 2, 2], tilt30=tilt)
    # CSR membership
    row_of_cell = {int(c): i for i, c in enumerate(anchor._hopf_cell_indices(a["rotations"]))}
    tr_index = {tuple(x): i for i, x in enumerate(a["translations"].tolist())}
    got = np.zeros(len(cells), dtype=bool)
    r_idx = np.array([row_of_cell.get(int(c), -1) for c in cells])
    t_idx = np.array([tr_index.get(tuple(x), -1) for x in t.tolist()])
    valid = (r_idx >= 0) & (t_idx >= 0)
    got[valid] = _members(name, r_idx[valid], t_idx[valid])
    assert np.array_equal(got, want)
    assert want.any() and (~want).any()


def _pair_keys(c, ncells_stride):
    """Ascending int64 keys row_id * stride + translation index for every pair."""
    return np.repeat(np.arange(len(c["filtered-sizes"]), dtype=np.int64), c["filtered-sizes"]) * ncells_stride


def test_containment_on_caches():
    spec = load_spec()
    require_caches(NAMES)
    edges = gf.containment_edges(spec["thresholds"])
    direct = sorted(
        (a, b) for a, b in edges
        if not any((a, c) in edges and (c, b) in edges for c in NAMES)
    )
    assert direct
    counts = {n: spec["thresholds"][n]["precalc"]["counts"] for n in NAMES}
    for a_name, b_name in sorted(edges):  # counts along every (also transitive) edge
        for k in ("rotations", "translations", "pairs"):
            assert counts[a_name][k] <= counts[b_name][k], (a_name, b_name, k)
    flat_all = {}
    for a_name, b_name in direct:
        _LOADED.clear()
        A, B = _load(a_name), _load(b_name)
        ca = anchor._hopf_cell_indices(A["rotations"])
        cb = anchor._hopf_cell_indices(B["rotations"])
        rows_in_b = np.searchsorted(cb, ca)
        assert np.array_equal(cb[rows_in_b], ca), (a_name, b_name, "cells")
        fb = _lattice_flat(B["translations"])
        fa = _lattice_flat(A["translations"])
        tmap = np.searchsorted(fb, fa)
        assert np.array_equal(fb[tmap], fa), (a_name, b_name, "translations")
        stride = len(fb)
        sizes_a, sizes_b = A["filtered-sizes"], B["filtered-sizes"]
        row_b = np.repeat(rows_in_b, sizes_a)  # B row of every A pair
        keys_a = row_b * stride + tmap[A["filtered-concat"]]
        del row_b
        keys_b = _pair_keys(B, stride) + B["filtered-concat"]
        assert np.all(np.diff(keys_b) > 0)
        # np.isin(keys_a, keys_b).all(), in chunks and via the sorted keys_b
        step = 8_000_000
        for lo in range(0, len(keys_a), step):
            chunk = keys_a[lo : lo + step]
            pos = np.searchsorted(keys_b, chunk)
            assert np.array_equal(keys_b[pos], chunk), (a_name, b_name, "pairs")
    _LOADED.clear()


@pytest.mark.parametrize("name", NAMES)
def test_manifest_audit_thresholds(name):
    require_caches([name])
    audit = load_spec()["thresholds"][name]["precalc"]["audit"]
    assert audit["min_rel_margin"] >= 1e-10 and audit["min_tilt_gap"] >= 1e-10
    counts = load_spec()["thresholds"][name]["precalc"]["counts"]
    sizes_path = _paths(name)["filtered-sizes"]
    assert npy_io.read_npy_header(sizes_path)[0] == (counts["rotations"],)
