import copy
import hashlib
import json
import re
import subprocess
import sys

import pytest

from gaussian_common import REPORT, ROOT, SPEC_PATH, load_spec, skip_or_fail
from alaric import gaussian_filter as gf

# filter_set_version -> spec_digest: a semantic edit without a version bump fails.
DIGESTS = {"2026.1": "bf1e90c08d7ef65eb78ef13b424162059389d92411c9406d0e2376a1bf358a7a"}

NAMES = [
    "rrm50", "rrm60", "rrm70", "rrm80",
    "rrm90", "rrm90-tilt30", "rrm90-disp45", "rrm90-tilt30-disp45",
    "rrm95", "rrm95-tilt30", "rrm95-disp45", "rrm95-tilt30-disp45",
]


def _nums(cell):
    return [int(x.replace(",", "")) for x in re.findall(r"[\d,]+(?=/)|(?<=/)[\d,]+", cell)]


def test_names_cutoffs_schema():
    spec = load_spec()
    assert spec["schema"] == 1
    assert list(spec["thresholds"]) == NAMES == gf.all_names()
    for name, e in spec["thresholds"].items():
        n, tilt, disp = gf.parse_name(name)
        parent = spec["thresholds"][f"rrm{n}"]
        assert e["cutoff"] == parent["cutoff"] and e["cutoff_decimal"] == parent["cutoff_decimal"]
        assert e["raw_cutoff"] == parent["raw_cutoff"]
        assert e["cutoff"] <= e["raw_cutoff"]
        assert e["filters"] == (["tilt30"] if tilt else []) + (["disp45"] if disp else [])
    assert {n: spec["thresholds"][n]["cutoff_decimal"] for n in NAMES[::4][:1] + ["rrm90", "rrm95"]} == {
        "rrm50": "3.30384", "rrm90": "0.27592", "rrm95": "0.06141"}


@pytest.mark.parametrize("bad", ["rrm55", "rrm50-tilt30", "rrm80-disp45", "rrm90-disp45-tilt30", "foo", "rrm95-tilt45"])
def test_grammar_rejects(bad):
    with pytest.raises(ValueError):
        gf.parse_name(bad)


def test_containment_edges():
    spec = load_spec()
    got = gf.containment_edges(spec["thresholds"])
    base = ["rrm50", "rrm60", "rrm70", "rrm80", "rrm90", "rrm95"]
    want = set()
    for a, b in zip(base, base[1:]):
        want.add((a, b))
    for n in (90, 95):
        r = f"rrm{n}"
        want |= {(f"{r}-tilt30-disp45", f"{r}-tilt30"), (f"{r}-tilt30", r),
                 (f"{r}-tilt30-disp45", f"{r}-disp45"), (f"{r}-disp45", r)}
    for suffix in ("", "-tilt30", "-disp45", "-tilt30-disp45"):
        want.add((f"rrm90{suffix}", f"rrm95{suffix}"))
    changed = True
    while changed:  # transitive closure
        new = {(a, d) for a, b in want for c, d in want if b == c} - want
        changed = bool(new)
        want |= new
    assert got == want


def test_regeneration_byte_identical():
    r = subprocess.run(
        [sys.executable, str(ROOT / "util" / "gaussian-filter-json.py"), "--check"],
        capture_output=True, text=True,
    )
    if "manifest.json" in r.stderr and "FileNotFoundError" in r.stderr:
        skip_or_fail("cache manifests missing (run util/gaussian-precalc-all.py)")
    assert r.returncode == 0, r.stdout + r.stderr


def test_spec_digest_table():
    spec = load_spec()
    assert gf.spec_digest(spec) == spec["spec_digest"]
    assert DIGESTS[spec["filter_set_version"]] == spec["spec_digest"]
    core = gf.build_spec_core()
    for key in ("model", "mask", "filters", "grid"):
        assert spec[key] == core[key]
    assert "envelope" not in spec
    assert spec["mask"]["distance_lt_A"] == 5


def test_counts_equal_report_rows():
    lines = REPORT.read_text().splitlines()
    spec = load_spec()
    rows = {}
    for ln in (42, 43, 44, 45):
        c = [x.strip() for x in lines[ln - 1].strip("|").split("|")]
        rows[f"rrm{c[0][:2]}"] = c
    for ln, n in ((59, 90), (63, 95)):
        for off, suffix in enumerate(("", "-tilt30", "-disp45", "-tilt30-disp45")):
            c = [x.strip() for x in lines[ln - 1 + off].strip("|").split("|")]
            rows[f"rrm{n}{suffix}"] = c
    for name in NAMES:
        a = spec["thresholds"][name]["achieved"]
        c = rows.get(name)
        if name in ("rrm90", "rrm95"):
            # negatives/test come from the cutoff row, positives from the variants table
            cut = gf.CUTOFFS[name][1]
            cc = [x.strip() for x in lines[cut - 1].strip("|").split("|")]
            neg, test = _nums(cc[3]), _nums(cc[4])
            assert a["test_positives"] == {"count": test[0], "of": test[1]}
            assert [a["negatives"]["count"], a["negatives"]["of"]] == neg
            assert a["rrm_positives"]["count"] == _nums(c[5])[0]
        elif name in ("rrm50", "rrm60", "rrm70", "rrm80"):
            neg, test = _nums(c[3]), _nums(c[4])
            assert [a["negatives"]["count"], a["negatives"]["of"]] == neg
            assert a["test_positives"] == {"count": test[0], "of": test[1]}
            required = -(-int(name[3:]) * 460 // 100)
            assert a["rrm_positives"]["count"] == required
        else:
            assert [a["negatives"]["count"], a["negatives"]["of"]] == _nums(c[3])
            assert a["rrm_positives"]["count"] == _nums(c[5])[0]
            assert a["test_positives"] is None
        assert a["rrm_positives"]["of"] == 460
        assert a["achieved_recall"] == a["rrm_positives"]["count"] / 460


def test_consumer_refuses_other_schema():
    spec = load_spec()
    assert gf.parse_spec(spec) is spec
    bad = copy.deepcopy(spec)
    bad["schema"] = 2
    with pytest.raises(ValueError, match="schema"):
        gf.parse_spec(bad)
    bad = copy.deepcopy(spec)
    bad["thresholds"]["rrm95"]["cutoff"] = 0.0614
    with pytest.raises(ValueError, match="digest"):
        gf.parse_spec(bad)


def test_json_is_canonical_text():
    text = SPEC_PATH.read_text()
    assert text == json.dumps(json.loads(text), indent=1, ensure_ascii=True) + "\n"


def test_forbidden_cutoffs_absent_from_sources():
    forbidden = ["0." + "2288", "0." + "2700", "0." + "1644", "0." + "0293"]
    files = (
        list((ROOT / "alaric").glob("gaussian_*.py"))
        + list((ROOT / "util").glob("gaussian-*.py"))
        + list((ROOT / "tests").glob("test_gaussian_*.py"))
    )
    assert len(files) >= 10
    for f in files:
        text = f.read_text()
        for bad in forbidden:
            assert bad not in text, (f, bad)
