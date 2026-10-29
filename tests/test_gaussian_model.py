import math
import random

import numpy as np
import pytest

from gaussian_common import REPORT, load_spec
from alaric import gaussian_filter as gf


def test_precision_and_normalizer_literals():
    C = np.array(gf.COVARIANCE)
    P = np.linalg.inv(C)
    assert np.allclose(np.array(gf.PRECISION), P, rtol=1e-14, atol=0)
    norm = (2 * math.pi) ** 1.5 * math.sqrt(np.linalg.det(C))
    assert abs(gf.NORMALIZER / norm - 1) < 1e-14


def test_golden_scores_from_json_parameters():
    spec = load_spec()
    model = gf.Model(
        spec["model"]["mean"],
        spec["model"]["covariance"],
        spec["model"]["precision"],
        spec["model"]["normalizer"],
    )
    tol = spec["golden"]["tolerance_rel"]
    assert spec["golden"]["scores"]
    for g in spec["golden"]["scores"]:
        s = gf.score(g["d"], g["phi_deg"] * (math.pi / 180), g["theta_deg"] * (math.pi / 180), model)
        assert abs(s / g["score"] - 1) < tol
        if "reference_score" in g:
            assert abs(s / g["reference_score"] - 1) < 1e-10
    mode = spec["golden"]["scores"][0]
    assert abs(mode["score"] * gf.NORMALIZER - 1) < 1e-12


@pytest.mark.parametrize(
    "raw,expected",
    [
        (3.3038440180955031, "3.30384"),
        (2.9043182397831022, "2.90431"),
        (2.0419804149333398, "2.04198"),
        (1.1974880057674258, "1.19748"),
        (0.27592087772968793, "0.27592"),
        (0.061416348507101752, "0.06141"),
        (0.25, "0.25000"),
        (0.29, "0.29000"),
    ],
)
def test_floor_cutoff(raw, expected):
    assert gf.floor_cutoff(raw) == expected


def test_floor_cutoff_random_never_above():
    rng = random.Random(1)
    for _ in range(100_000):
        raw = rng.uniform(0, 6)
        s = gf.floor_cutoff(raw)
        assert float(s) <= raw
        assert raw - float(s) < 1.0001e-5


def test_report_cutoffs_and_json():
    lines = REPORT.read_text().splitlines()
    import hashlib

    assert hashlib.sha256(REPORT.read_bytes()).hexdigest() == gf.REPORT_SOURCE["sha256"]
    spec = load_spec()
    for target, (raw, cut_line, _count_line) in zip((50, 60, 70, 80, 90, 95), gf.CUTOFFS.values()):
        cells = [c.strip() for c in lines[cut_line - 1].strip("|").split("|")]
        assert cells[0] == f"{target}%" and cells[1] == "Gaussian only"
        assert float(cells[2]) == raw
        assert spec["thresholds"][f"rrm{target}"]["raw_cutoff"] == raw
        assert spec["thresholds"][f"rrm{target}"]["cutoff_decimal"] == gf.floor_cutoff(float(cells[2]))
    text = (REPORT.parent.parent.parent / "alaric" / "gaussian_stacking_filter.json").read_text()
    for forbidden in ("0." + "2288", "0." + "2700", "0." + "1644", "0." + "0293"):
        assert forbidden not in text
