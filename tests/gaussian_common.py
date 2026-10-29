"""Shared helpers for the Gaussian filter tests."""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPEC_PATH = ROOT / "alaric" / "gaussian_stacking_filter.json"
REPORT = Path(__file__).parent / "gaussian-data" / "envelope-analysis-5-report.md"


def load_spec():
    return json.loads(SPEC_PATH.read_text())


def skip_or_fail(msg):
    """Skip, unless ALARIC_GAUSSIAN_CACHES_REQUIRED=1, in which case fail."""
    if os.environ.get("ALARIC_GAUSSIAN_CACHES_REQUIRED") == "1":
        pytest.fail(msg)
    pytest.skip(msg)


def require_caches(names):
    """Skip (or fail with ALARIC_GAUSSIAN_CACHES_REQUIRED=1) when caches are absent."""
    from alaric import gaussian_cache as gc

    spec = load_spec()
    missing = []
    for name in names:
        d = ROOT / spec["thresholds"][name]["precalc"]["dir"]
        if not all((d / f"{f}.npy").is_file() or (d / f"{f}.npy.zst").is_file() for f in gc.FILE_NAMES):
            missing.append(name)
    if missing:
        skip_or_fail(
            f"Gaussian caches missing: {missing}. Install with: "
            f"python {ROOT}/util/gaussian-precalc-all.py"
        )
