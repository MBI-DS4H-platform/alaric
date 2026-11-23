#!/usr/bin/env python3
"""Build all 12 Gaussian filter caches under <root>/precomputed-filters/gaussian/.

By default, verify the generated caches against the filter spec.
--check builds into a temporary directory and compares with the committed JSON:
an npy_sha256 mismatch is a hard failure (exit 1); a sha256-only mismatch is
reported as "recompressed, arrays identical" (exit 0).
--rehash removes the old manifests, rebuilds the caches, and updates the spec.
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


def rehash(root, spec_path, spec, names, decompress):
    if spec["spec_digest"] != gf.spec_digest(gf.build_spec_core()):
        raise ValueError("Cannot rehash a Gaussian spec that differs from the generator")

    directories = {
        name: root / spec["thresholds"][name]["precalc"]["dir"] for name in names
    }
    for directory in directories.values():
        (directory / "manifest.json").unlink(missing_ok=True)

    builds = {}
    for name, directory in directories.items():
        builds[name] = gp.build_filter(name)
        gp.write_cache(directory, builds[name])

    for name, directory in directories.items():
        threshold = spec["thresholds"][name]
        manifest = gp.write_manifest(
            directory, name, builds.pop(name), threshold["cutoff_decimal"]
        )
        for key in ("files", "counts", "audit"):
            threshold["precalc"][key] = manifest[key]
        if decompress:
            gc.decompress_dir(directory)
        gc.verify_filter_cache(spec, name, root)
        print(name, json.dumps(manifest["counts"]), flush=True)

    spec["provenance"]["precalc_environment"] = gp.environment()
    # Publish the new runtime hashes together after all caches have verified.
    with tempfile.TemporaryDirectory(dir=spec_path.parent) as tmp:
        updated = Path(tmp) / spec_path.name
        updated.write_text(json.dumps(spec, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
        updated.chmod(spec_path.stat().st_mode & 0o777)
        updated.replace(spec_path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(ROOT))
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--check", action="store_true", help="compare a temporary rebuild to the spec"
    )
    mode.add_argument(
        "--rehash", action="store_true", help="rebuild and replace the spec's cache hashes"
    )
    ap.add_argument("--decompress", action="store_true")
    ap.add_argument("--spec", default=str(ROOT / "alaric" / "gaussian_stacking_filter.json"))
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    spec_path = Path(args.spec).resolve()
    spec = gf.load_spec(spec_path)
    names = gf.all_names()
    if args.check:
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
    if args.rehash:
        rehash(root, spec_path, spec, names, args.decompress)
        return 0
    for name in names:
        out = root / spec["thresholds"][name]["precalc"]["dir"]
        manifest = gp.generate(name, out)
        gc.verify_filter_cache(spec, name, root)
        if args.decompress:
            gc.decompress_dir(out)
        print(name, json.dumps(manifest["counts"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
