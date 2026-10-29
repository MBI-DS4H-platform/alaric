#!/usr/bin/env python3
"""Generate one Gaussian filter cache: python util/gaussian-precalc.py rrm95 --output precomputed-filters/gaussian/rrm95

With --check, build into a temporary directory and compare against
alaric/gaussian_stacking_filter.json (npy_sha256 mismatch is a hard failure).
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alaric import gaussian_cache as gc  # noqa: E402
from alaric import gaussian_filter as gf  # noqa: E402
from alaric import gaussian_precalc as gp  # noqa: E402


def build(name, directory, decompress=False):
    manifest = gp.generate(name, directory)
    if decompress:
        gc.decompress_dir(directory)
    return manifest


def check(name, spec_path):
    spec = json.loads(Path(spec_path).read_text())
    entry = spec["thresholds"][name]["precalc"]["files"]
    with tempfile.TemporaryDirectory() as tmp:
        manifest = build(name, Path(tmp) / name)
        ok, notes = gc.compare_manifest(manifest, entry)
    for note in notes:
        print(f"{name}: {note}")
    print(f"{name}: {'OK' if ok else 'FAILED'}")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("name")
    ap.add_argument("--output", help="output directory")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--decompress", action="store_true", help="install .npy instead of .npy.zst")
    ap.add_argument("--spec", default=str(ROOT / "alaric" / "gaussian_stacking_filter.json"))
    args = ap.parse_args(argv)
    gf.parse_name(args.name)
    if args.check:
        return 0 if check(args.name, args.spec) else 1
    if not args.output:
        ap.error("--output is required")
    manifest = build(args.name, Path(args.output), args.decompress)
    print(json.dumps(manifest["counts"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
