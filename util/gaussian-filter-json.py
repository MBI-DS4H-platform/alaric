#!/usr/bin/env python3
"""Assemble alaric/gaussian_stacking_filter.json from module constants, the
report table, computed goldens and the 12 cache manifests.

    python util/gaussian-filter-json.py --root . --output alaric/gaussian_stacking_filter.json [--check]

--check regenerates and requires the output to be byte-identical to the file.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alaric import gaussian_cache as gc  # noqa: E402
from alaric import gaussian_filter as gf  # noqa: E402

DESCRIPTORS_TEST = "/srv/shared/stacking-prediction-plots/descriptors_test.csv"
REL = "precomputed-filters/gaussian"
DEG = gf.DEG


def acc(name, d, phi_deg, theta_deg):
    return gf.accept_descriptor(name, d, phi_deg * DEG, theta_deg * DEG)


def golden(descriptors_test):
    mu_d, mu_phi, mu_theta = gf.MEAN
    phi_deg0, theta_deg0 = math.degrees(mu_phi), math.degrees(mu_theta)

    def sc(d, p, t):
        return float(gf.score(d, p * DEG, t * DEG))

    points = [("mode", gf.MEAN[0], phi_deg0, theta_deg0)]
    points += [("manual", 3.5, 0.0, 0.0), ("manual", 4.0, 30.0, 20.0),
               ("manual", 4.5, 45.0, 30.0), ("manual", 5.5, 59.0, 60.0)]
    scores = [{"d": d, "phi_deg": p, "theta_deg": t, "score": sc(d, p, t)}
              for _, d, p, t in points]
    with open(descriptors_test, newline="") as f:
        for row in list(csv.DictReader(f))[:3]:
            d, p, t = (float(row["distance_A"]), float(row["displacement_angle_deg"]),
                       float(row["planar_angle_deg"]))
            scores.append({"d": d, "phi_deg": p, "theta_deg": t, "score": sc(d, p, t),
                           "reference_score": float(row["gaussian_score"])})
    names = gf.all_names()
    decisions = []

    def add(d, p, t):
        for n in names:
            decisions.append({"name": n, "d": d, "phi_deg": p, "theta_deg": t,
                              "accept": acc(n, d, p, t)})

    for base in (50, 60, 70, 80, 90, 95):
        c = gf.threshold_spec(f"rrm{base}")["cutoff"]
        for factor in (1 + 1e-9, 1 - 1e-9):
            target = c * factor
            lo, hi = mu_d, 5.0  # score decreases on the ray d > mu_d
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if sc(mid, phi_deg0, theta_deg0) >= target:
                    lo = mid
                else:
                    hi = mid
            d = lo if factor > 1 else hi
            add(d, phi_deg0, theta_deg0)
    # boundary points sit at the conditional mean of the other descriptors, so only
    # the prefilter decides
    import numpy as np
    C = np.array(gf.COVARIANCE)
    mu = np.array(gf.MEAN)

    def cond(fixed, value_rad):
        other = [i for i in range(3) if i != fixed]
        m = mu[other] + C[other, fixed] / C[fixed, fixed] * (value_rad - mu[fixed])
        return dict(zip(other, (float(x) for x in m)))

    boundary = []
    for v in (45.0, 44.99999):
        c = cond(1, v * DEG)
        boundary.append((c[0], v, math.degrees(c[2])))
    for v in (30.0, 29.99999):
        c = cond(2, v * DEG)
        boundary.append((c[0], math.degrees(c[1]), v))
    for d, p, t in boundary:
        assert sc(d, p, t) >= 0.27592, (d, p, t)
        add(d, p, t)
    lattice = []
    for xyz in ((0.75, 1.0, 1.25), (0.0, 0.0, 5.0), (0.0, 3.0, 4.0), (3.0, 4.0, 0.0),
                (5.0, 0.0, 0.0), (0.0, 0.0, 4.75)):
        ijk = [round(v * 4) for v in xyz]
        for n in names:
            _, tilt, disp = gf.parse_name(n)
            lattice.append({"name": n, "xyz": list(xyz),
                            "mask_accept": bool(gf.mask_translation(ijk, disp45=disp))})
    notes = {
        "decisions": "accept = score >= cutoff and the applied mask and the name's prefilters",
        "boundary_points": "phi/theta boundary goldens (phi 45, 44.99999; theta 30, 29.99999) sit at the "
        "conditional mean of the other two descriptors given the fixed one, so the score is far above "
        "the cutoff and only the prefilter decides",
        "lattice_decisions": "mask_accept covers the applied mask and the name's prefilters only; no score",
    }
    return {"notes": notes, "tolerance_rel": 1e-12, "scores": scores, "decisions": decisions,
            "lattice_decisions": lattice}


def build(root, descriptors_test):
    root = Path(root)
    core = gf.build_spec_core()
    thresholds = {}
    env = None
    for name in gf.all_names():
        t = gf.threshold_spec(name)
        manifest = json.loads((root / REL / name / "manifest.json").read_text())
        assert manifest["cutoff"] == t["cutoff_decimal"], name
        env = env or manifest["environment"]
        assert manifest["environment"] == env, "mixed build environments"
        files = {k: {"sha256": v["sha256"], "npy_sha256": v["npy_sha256"],
                     "shape": v["shape"], "dtype": v["dtype"]}
                 for k, v in manifest["files"].items()}
        # digests are re-read from the published files
        for k in files:
            d = gc.describe_file(root / REL / name / f"{k}.npy")
            assert d["npy_sha256"] == files[k]["npy_sha256"], (name, k)
        t["precalc"] = {"dir": f"{REL}/{name}", "files": files,
                        "counts": manifest["counts"], "audit": manifest["audit"]}
        thresholds[name] = t
    spec = {
        "schema": gf.SCHEMA,
        "filter_set_version": gf.FILTER_SET_VERSION,
        "spec_digest": None,
        "model": core["model"],
        "mask": core["mask"],
        "filters": core["filters"],
        "grid": core["grid"],
        "thresholds": thresholds,
        "golden": golden(descriptors_test),
        "provenance": {
            "report": {**gf.REPORT_SOURCE,
                       "note": "uncommitted working-tree version; copy vendored in tests"},
            "cutoff_envelope": {"distance_lt_A": 5.7, "displacement_lt_deg": 60,
                                "positives": 460, "negatives": 18592},
            "cutoff_rounding": "floor to 1e-5 of repr(raw)",
            "generator": {"commands": [
                "python util/gaussian-precalc-all.py",
                "python util/gaussian-filter-json.py --root . --output alaric/gaussian_stacking_filter.json"]},
            "precalc_environment": env,
        },
    }
    spec["spec_digest"] = gf.spec_digest(spec)
    return json.dumps(spec, indent=1, ensure_ascii=True) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--output", default=str(ROOT / "alaric" / "gaussian_stacking_filter.json"))
    ap.add_argument("--descriptors-test", default=DESCRIPTORS_TEST)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    text = build(args.root, args.descriptors_test)
    out = Path(args.output)
    if args.check:
        same = out.read_text() == text
        print("byte-identical" if same else "DIFFERENT")
        return 0 if same else 1
    out.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
