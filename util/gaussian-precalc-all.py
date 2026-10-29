#!/usr/bin/env python3
"""Build all 12 Gaussian filter caches under <root>/precomputed-filters/gaussian/.

--check builds into a temporary directory and compares with the committed JSON:
an npy_sha256 mismatch is a hard failure (exit 1); a sha256-only mismatch is
reported as "recompressed, arrays identical" (exit 0).
--decompress installs .npy instead of .npy.zst.
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

REL = "precomputed-filters/gaussian"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--decompress", action="store_true")
    ap.add_argument("--spec", default=str(ROOT / "alaric" / "gaussian_stacking_filter.json"))
    args = ap.parse_args(argv)
    names = gf.all_names()
    if args.check:
        spec = json.loads(Path(args.spec).read_text())
        failed = False
        with tempfile.TemporaryDirectory() as tmp:
            for name in names:
                manifest = gp.generate(name, Path(tmp) / name)
                ok, notes = gc.compare_manifest(
                    manifest, spec["thresholds"][name]["precalc"]["files"]
                )
                for note in notes:
                    print(f"{name}: {note}")
                print(f"{name}: {'OK' if ok else 'FAILED'}", flush=True)
                failed |= not ok
                gp.shutil.rmtree(Path(tmp) / name)
        return 1 if failed else 0
    for name in names:
        out = Path(args.root) / REL / name
        manifest = gp.generate(name, out)
        if args.decompress:
            gc.decompress_dir(out)
        print(name, json.dumps(manifest["counts"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
