"""Gaussian stacking filter: model, masks, grids and cutoff table.

Pure model/specification module (numpy, math, decimal, json only; flat imports so
it can be attached from other environments).  The bulk precalculation lives in
``gaussian_precalc``; the installation check in ``gaussian_cache``.

Everything that defines the filter identity is a literal in this file, so the
cached CSR files do not depend on LAPACK or on the committed JSON.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import namedtuple
from decimal import ROUND_FLOOR, Decimal

import numpy as np

SCHEMA = 1
FILTER_SET_VERSION = "2026.1"

Model = namedtuple("Model", "mean covariance precision normalizer")

# Descriptor order: (d in Angstrom, phi in rad, theta in rad).
MEAN = (3.778988652846167, 0.3359170155526752, 0.21710309041293038)
COVARIANCE = (
    (0.13402643421739668, 0.048747464673082525, 0.03661672105978793),
    (0.048747464673082525, 0.05051245758442225, 0.018930688724029816),
    (0.03661672105978793, 0.018930688724029816, 0.039514509672839315),
)
# Hard-coded literals (np.linalg.inv of COVARIANCE, symmetrised), computed once.
PRECISION = (
    (13.055035569772702, -9.829936278368704, -7.388302029768115),
    (-9.829936278368704, 31.531026898594526, -5.996886272605505),
    (-7.388302029768115, -5.996886272605505, 35.02664194314763),
)
NORMALIZER = 0.17639457343866755  # (2 pi)^{3/2} sqrt(det COVARIANCE)
MODEL = Model(MEAN, COVARIANCE, PRECISION, NORMALIZER)

MODEL_SOURCE = {
    "path": "/srv/shared/stacking-prediction-plots/recover-gaussian/gaussian_model.json",
    "sha256": "0bfb8acf592dc45a52819a41618583e7a5624b7f3cdcaa71d70d910f47cc7db8",
}
REPORT_SOURCE = {
    "path": "/srv/shared/stacking-prediction-plots/envelope-analysis-5-report.md",
    "sha256": "11f854aa8e4e8a3ade18dcde84394f770bed08dea175796ae20e3f8953722edc",
}

DEG = math.pi / 180.0
MASK_D_A = 5
MASK_PHI_DEG = 60
TILT30_DEG = 30
DISP45_DEG = 45

# name -> (raw cutoff, report cutoff line, counts line)
CUTOFFS = {
    "rrm50": (3.3038440180955031, 42, 42),
    "rrm60": (2.9043182397831022, 43, 43),
    "rrm70": (2.0419804149333398, 44, 44),
    "rrm80": (1.1974880057674258, 45, 45),
    "rrm90": (0.27592087772968793, 46, 59),
    "rrm95": (0.061416348507101752, 51, 63),
}

BASE_TARGETS = (50, 60, 70, 80, 90, 95)
VARIANT_TARGETS = (90, 95)
_NAME_RE = re.compile(r"^rrm(\d+)(-tilt30)?(-disp45)?$")

# Counts copied from the report (never recomputed).  positives are of 460.
# name -> (positives, negatives, negatives_of, negatives_set, test_positives)
_ENV = "envelope synthetic"
_RES = "resampled envelope synthetic (seed 20261026)"
REPORT_COUNTS = {
    "rrm50": (230, 183, 100000, _RES, 27),
    "rrm60": (276, 246, 100000, _RES, 34),
    "rrm70": (322, 493, 100000, _RES, 46),
    "rrm80": (368, 182, 18592, _ENV, 52),
    "rrm90": (414, 578, 18592, _ENV, 57),
    "rrm90-tilt30": (411, 414, 18592, _ENV, None),
    "rrm90-disp45": (413, 533, 18592, _ENV, None),
    "rrm90-tilt30-disp45": (410, 375, 18592, _ENV, None),
    "rrm95": (437, 1305, 18592, _ENV, 58),
    "rrm95-tilt30": (425, 718, 18592, _ENV, None),
    "rrm95-disp45": (432, 972, 18592, _ENV, None),
    "rrm95-tilt30-disp45": (423, 534, 18592, _ENV, None),
}
_VARIANT_LINE_OFFSET = {"": 0, "-tilt30": 1, "-disp45": 2, "-tilt30-disp45": 3}


def parse_name(name: str) -> tuple[int, bool, bool]:
    m = _NAME_RE.match(name)
    if not m:
        raise ValueError(f"invalid Gaussian filter name {name!r}")
    n = int(m.group(1))
    tilt, disp = bool(m.group(2)), bool(m.group(3))
    if n not in BASE_TARGETS:
        raise ValueError(f"invalid Gaussian filter name {name!r}: unknown target {n}")
    if (tilt or disp) and n not in VARIANT_TARGETS:
        raise ValueError(f"invalid Gaussian filter name {name!r}: suffixes only for 90/95")
    return n, tilt, disp


def all_names() -> list[str]:
    names = [f"rrm{n}" for n in (50, 60, 70, 80)]
    for n in VARIANT_TARGETS:
        for suffix in ("", "-tilt30", "-disp45", "-tilt30-disp45"):
            names.append(f"rrm{n}{suffix}")
    return names


def floor_cutoff(raw: float, decimals: int = 5) -> str:
    """Round ``raw`` down to ``decimals`` places, from its shortest repr."""
    quantum = Decimal(1).scaleb(-decimals)
    s = str(Decimal(repr(float(raw))).quantize(quantum, rounding=ROUND_FLOOR))
    assert float(s) <= raw, (s, raw)
    return s


def threshold_spec(name: str) -> dict:
    n, tilt, disp = parse_name(name)
    raw, cut_line, counts_line = CUTOFFS[f"rrm{n}"]
    dec = floor_cutoff(raw)
    suffix = ("-tilt30" if tilt else "") + ("-disp45" if disp else "")
    filters = (["tilt30"] if tilt else []) + (["disp45"] if disp else [])
    if suffix:
        counts_line += _VARIANT_LINE_OFFSET[suffix]
        table = "Added filters at fixed Gaussian-only cutoffs"
    elif n in VARIANT_TARGETS:
        table = "Added filters at fixed Gaussian-only cutoffs"
    else:
        table = "RRM-derived cutoffs evaluated on the test set"
    pos, neg, neg_of, neg_set, test = REPORT_COUNTS[name]
    return {
        "rrm_target": n,
        "parent": f"rrm{n}" if suffix else None,
        "filters": filters,
        "accept": ">=",
        "cutoff": float(dec),
        "cutoff_decimal": dec,
        "raw_cutoff": raw,
        "cutoff_rule": "parent Gaussian-only RRM-N cutoff of the report, floored to 1e-5"
        + ("; prefilters on top" if suffix else ""),
        "source": {
            "report_line": counts_line,
            "cutoff_report_line": cut_line,
            "table": table,
        },
        "achieved": {
            "rrm_positives": {"count": pos, "of": 460},
            "achieved_recall": pos / 460,
            "negatives": {"count": neg, "of": neg_of, "set": neg_set},
            "test_positives": None if test is None else {"count": test, "of": 58},
        },
    }


def containment_edges(thresholds: dict) -> set[tuple[str, str]]:
    """All (A, B) with A != B and filter A contained in filter B.

    ``thresholds`` maps name -> entry with ``filters`` and ``cutoff``.
    """
    edges = set()
    for a, ea in thresholds.items():
        for b, eb in thresholds.items():
            if a != b and set(eb["filters"]) <= set(ea["filters"]) and ea["cutoff"] >= eb["cutoff"]:
                edges.add((a, b))
    return edges


# ---------------------------------------------------------------- score

def score(d, phi_rad, theta_rad, model: Model = MODEL):
    """exp(-q/2)/normalizer for descriptors (d, phi, theta); broadcasts."""
    P = model.precision
    d0 = np.asarray(d) - model.mean[0]
    d1 = np.asarray(phi_rad) - model.mean[1]
    d2 = np.asarray(theta_rad) - model.mean[2]
    q = (
        P[0][0] * d0 * d0
        + P[1][1] * d1 * d1
        + P[2][2] * d2 * d2
        + 2 * (P[0][1] * d0 * d1 + P[0][2] * d0 * d2 + P[1][2] * d1 * d2)
    )
    return np.exp(-0.5 * q) / model.normalizer


def descriptors(R, t):
    """(d, phi, theta) of rotation(s) R [...,3,3] and translation(s) t [...,3]."""
    t = np.asarray(t, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    d = np.sqrt(t[..., 0] * t[..., 0] + t[..., 1] * t[..., 1] + t[..., 2] * t[..., 2])
    with np.errstate(invalid="ignore", divide="ignore"):
        phi = np.arccos(np.abs(t[..., 2]) / d)
    theta = np.arccos(np.minimum(1.0, np.abs(R[..., 2, 2])))
    return d, phi, theta


# ---------------------------------------------------------------- grids

HOPF_FACES = 6
HOPF_N = 16
HOPF_NPSI = 64
HOPF_CELLS = HOPF_FACES * HOPF_N * HOPF_N * HOPF_NPSI
LATTICE_HALF = 20  # indices in [-20, 20], step 0.25 A
LATTICE_STEP = 0.25

_HOPF_CACHE = None


def hopf_rotations():
    """(R [98304,3,3], |R33| [98304]) in cell order ((face*16+p)*16+q)*64+m."""
    global _HOPF_CACHE
    if _HOPF_CACHE is not None:
        return _HOPF_CACHE
    cells = np.arange(HOPF_CELLS)
    face, rest = np.divmod(cells, HOPF_N * HOPF_N * HOPF_NPSI)
    p, rest = np.divmod(rest, HOPF_N * HOPF_NPSI)
    qq, m = np.divmod(rest, HOPF_NPSI)
    a = np.tan((math.pi / 4) * (2 * (p + 0.5) / HOPF_N - 1))
    b = np.tan((math.pi / 4) * (2 * (qq + 0.5) / HOPF_N - 1))
    sign = np.where(face % 2 == 0, 1.0, -1.0)
    ones = np.ones_like(a)
    s = np.empty((HOPF_CELLS, 3))
    axis = face // 2
    for value, (major, first, second) in enumerate(((0, 1, 2), (1, 0, 2), (2, 0, 1))):
        sel = axis == value
        s[sel, major] = (sign * ones)[sel]
        s[sel, first] = a[sel]
        s[sel, second] = b[sel]
    s = s / np.sqrt(s[:, 0] ** 2 + s[:, 1] ** 2 + s[:, 2] ** 2)[:, None]
    theta_s = np.arccos(np.clip(s[:, 2], -1.0, 1.0))
    phi_s = np.arctan2(s[:, 1], s[:, 0])
    psi = 2 * math.pi * (m + 0.5) / HOPF_NPSI
    qw = np.cos(theta_s / 2) * np.cos(psi / 2)
    qx = np.cos(theta_s / 2) * np.sin(psi / 2)
    qy = np.sin(theta_s / 2) * np.cos(phi_s + psi / 2)
    qz = np.sin(theta_s / 2) * np.sin(phi_s + psi / 2)
    R = np.empty((HOPF_CELLS, 3, 3))
    R[:, 0, 0] = 1 - 2 * (qy * qy + qz * qz)
    R[:, 0, 1] = 2 * (qx * qy - qw * qz)
    R[:, 0, 2] = 2 * (qx * qz + qw * qy)
    R[:, 1, 0] = 2 * (qx * qy + qw * qz)
    R[:, 1, 1] = 1 - 2 * (qx * qx + qz * qz)
    R[:, 1, 2] = 2 * (qy * qz - qw * qx)
    R[:, 2, 0] = 2 * (qx * qz - qw * qy)
    R[:, 2, 1] = 2 * (qy * qz + qw * qx)
    R[:, 2, 2] = 1 - 2 * (qx * qx + qy * qy)
    _HOPF_CACHE = (R, np.abs(R[:, 2, 2]))
    return _HOPF_CACHE


def lattice():
    """(ijk int64 [68920,3], t float64 [68920,3]), C order, origin excluded."""
    r = np.arange(-LATTICE_HALF, LATTICE_HALF + 1)
    ijk = np.stack(np.meshgrid(r, r, r, indexing="ij"), axis=-1).reshape(-1, 3)
    ijk = ijk[np.any(ijk != 0, axis=1)]
    return ijk.astype(np.int64), ijk * LATTICE_STEP


# ---------------------------------------------------------------- masks

def mask_translation(ijk, disp45: bool = False):
    """Exact integer tests on lattice indices (never via arccos)."""
    ijk = np.asarray(ijk, dtype=np.int64)
    i, j, k = ijk[..., 0], ijk[..., 1], ijk[..., 2]
    s = i * i + j * j
    ok = np.ones(s.shape, dtype=bool)
    ok &= (s + k * k) < 400
    ok &= 3 * k * k > s
    if disp45:
        ok &= k * k > s
    return ok


def mask_rotation(R33, tilt30: bool = False):
    R33 = np.asarray(R33, dtype=np.float64)
    if not tilt30:
        return np.ones(R33.shape, dtype=bool)
    return 4.0 * R33 * R33 > 3.0


def accept_descriptor(name: str, d, phi_rad, theta_rad, model: Model = MODEL) -> bool:
    """Reference predicate on descriptors (for goldens; degrees via deg*(pi/180))."""
    n, tilt, disp = parse_name(name)
    cutoff = threshold_spec(name)["cutoff"]
    if not (d < MASK_D_A and phi_rad < MASK_PHI_DEG * DEG):
        return False
    if tilt and not theta_rad < TILT30_DEG * DEG:
        return False
    if disp and not phi_rad < DISP45_DEG * DEG:
        return False
    return bool(score(d, phi_rad, theta_rad, model) >= cutoff)


# ---------------------------------------------------------------- spec

DEFINITIONS = {
    "v": "RNA ring centroid minus protein ring centroid, in the protein ring frame (z is the protein normal)",
    "d": "|v|",
    "phi": "arccos(|v_z|/d), the angle from the protein normal; stage 3's slip angle is the same phi",
    "theta": "arccos(min(1, |R33|)), the angle between the two ring normals",
    "normals": "Kabsch fit of each ring onto the reference rings (G->A, U->C, TYR->PHE)",
}


def build_spec_core() -> dict:
    """The identity-defining part of the JSON, from module constants."""
    thresholds = {
        name: {
            "rrm_target": threshold_spec(name)["rrm_target"],
            "filters": threshold_spec(name)["filters"],
            "cutoff": threshold_spec(name)["cutoff"],
        }
        for name in all_names()
    }
    return {
        "model": {
            "descriptor_order": ["distance_A", "displacement_rad", "tilt_rad"],
            "definitions": dict(DEFINITIONS),
            "mean": list(MEAN),
            "covariance": [list(r) for r in COVARIANCE],
            "precision": [list(r) for r in PRECISION],
            "normalizer": NORMALIZER,
            "formula": "score = exp(-q/2)/normalizer",
            "quadratic_form": "q = P00*D0^2 + P11*D1^2 + P22*D2^2 + 2*(P01*D0*D1 + P02*D0*D2 + P12*D1*D2), D = x - mean",
            "degrees_to_radians": "deg*(pi/180)",
            "source": dict(MODEL_SOURCE),
        },
        "mask": {
            "applies_to": "caches, pose_pass, browser",
            "distance_lt_A": MASK_D_A,
            "displacement_lt_deg": MASK_PHI_DEG,
            "exact": {"distance": "x^2+y^2+z^2 < 25", "displacement": "3z^2 > x^2+y^2"},
        },
        "filters": {
            "tilt30": {"descriptor": "tilt", "lt_deg": TILT30_DEG, "exact": "4*R33^2 > 3"},
            "disp45": {"descriptor": "displacement", "lt_deg": DISP45_DEG, "exact": "z^2 > x^2+y^2"},
        },
        "grid": {
            "hopf": {
                "faces": HOPF_FACES,
                "n": HOPF_N,
                "npsi": HOPF_NPSI,
                "cells": HOPF_CELLS,
                "index": "((face*16+p)*16+q)*64+m",
            },
            "lattice": {
                "min": -5,
                "max": 5,
                "step": LATTICE_STEP,
                "order": "C (z fastest)",
                "origin": "excluded",
            },
            "layout": "canonical: rows/translations with >=1 accepted pair; int32 sizes/concat",
        },
        "thresholds": thresholds,
    }


def spec_digest(spec: dict) -> str:
    core = {k: spec[k] for k in ("model", "mask", "filters", "grid")}
    core["thresholds"] = {
        name: {
            "rrm_target": e["rrm_target"],
            "filters": e["filters"],
            "cutoff": e["cutoff"],
        }
        for name, e in spec["thresholds"].items()
    }
    text = json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode()).hexdigest()


def parse_spec(spec: dict) -> dict:
    """Consumer-side validation of a loaded JSON spec."""
    if spec.get("schema") != SCHEMA:
        raise ValueError(f"unsupported gaussian filter spec schema {spec.get('schema')!r}")
    if spec_digest(spec) != spec.get("spec_digest"):
        raise ValueError("gaussian filter spec_digest mismatch")
    return spec


def load_spec(path=None) -> dict:
    from pathlib import Path

    path = Path(path) if path else Path(__file__).with_name("gaussian_stacking_filter.json")
    return parse_spec(json.loads(path.read_text()))
