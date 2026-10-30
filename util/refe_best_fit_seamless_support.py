"""Input declarations and materialization setup for the best-fit wrappers."""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import re
import runpy
import sys


def library_module(source: Path):
    sys.path.insert(0, str(source))
    import library
    return library


def identities(fraglib: Path) -> dict:
    """Read precomputed library.py sidecars, without opening library payloads."""
    files = {}
    for directory in ("conformers", "rotamers"):
        sidecars = sorted((fraglib / directory).glob("*.CHECKSUM"))
        if not sidecars:
            raise FileNotFoundError(f"No library checksum sidecars in {fraglib / directory}")
        for path in sidecars:
            checksum = path.read_text().split()[0]
            if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
                raise ValueError(f"Invalid SHA-256 checksum in {path}")
            files[str(path.relative_to(fraglib).with_suffix(""))] = checksum
    return {"schema": 1, "fraglen": 2, "sha256": files}


def _validate_identity(value: object) -> dict:
    """Validate the typed identity envelope before using it as a pin value."""
    if type(value) is not dict:
        raise ValueError("fraglib_identities must be a schema-1 object")
    if type(value.get("schema")) is not int or value["schema"] != 1:
        raise ValueError("fraglib_identities must have schema: 1")
    if type(value.get("fraglen")) is not int or value["fraglen"] != 2:
        raise ValueError("fraglib_identities must have fraglen: 2")
    checksums = value.get("sha256")
    if type(checksums) is not dict or not checksums:
        raise ValueError("fraglib_identities must include a non-empty sha256 object")
    for name, digest in checksums.items():
        if not isinstance(name, str) or not name:
            raise ValueError("fraglib_identities sha256 keys must be non-empty strings")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"Invalid SHA-256 digest for library identity {name!r}")
    return value


def input_paths() -> list[Path]:
    """Return the exact fixed file inputs declared in the T5 DEPS list."""
    util_dir = Path(__file__).resolve().parent
    deps = util_dir / "refe-best-fit.DEPS"
    entries = deps.read_text(encoding="utf-8").splitlines()
    if not entries or any(not entry or entry.startswith("#") for entry in entries):
        raise ValueError("refe-best-fit.DEPS must contain paths only, one per line")
    return [deps, *((util_dir / entry).resolve() for entry in entries)]


def run(arguments: list[str]) -> None:
    """Use materialization hints and run with source files supplied by Seamless."""
    try:
        expected = _validate_identity(json.loads(Path("fraglib_identities").read_text()))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("fraglib_identities is not valid JSON") from exc

    # Seamless can symlink each file to a content-addressed buffer. Resolve the
    # workspace location, not the file symlink's unrelated cache directory.
    workspace = Path(__file__).absolute().parent
    library = library_module(workspace)
    library._FRAGLIB_YAML_PATH = Path(os.environ["fraglib_config"]).absolute()
    config = library._fraglib_config()
    if config["fraglen"] != expected["fraglen"]:
        raise ValueError("fraglib configuration does not match fraglib_identities")
    # The library is immutable and identified by the ordinary checksum manifest.
    # Its installation is trusted to materialize those identities, as in the
    # existing fitter's config(verify_checksums=False). No data is rehashed here.
    config["templates_dir"] = "."
    rna_path = Path("rna.pdb")
    rna_path.write_bytes(Path("rna_pdb").read_bytes())
    exclude_code = Path("exclude_pdb_code").read_text(encoding="utf-8").strip()
    if not exclude_code:
        raise ValueError("exclude_pdb_code is required for every library fit")
    arguments = ["--exclude-pdb-code", exclude_code.lower(), *arguments]
    output = Path("best-fit-output")
    output.mkdir()
    sys.argv = [str(workspace / "refe-best-fit.py"), str(rna_path), *arguments,
                "--pdb-output", str(output / "pdbs")]
    with (output / "best-fits.tsv").open("w") as handle, contextlib.redirect_stdout(handle):
        runpy.run_path(str(workspace / "refe-best-fit.py"), run_name="__main__")


if __name__ == "__main__":
    if sys.argv[1:] == ["inputs"]:
        print("\n".join(str(path) for path in input_paths()))
    elif sys.argv[1:] == ["identities"]:
        root = Path(__file__).resolve().parent.parent
        print(json.dumps(identities(root / "fraglib"), sort_keys=True, separators=(",", ":")))
    elif len(sys.argv) >= 2 and sys.argv[1] == "run":
        run(sys.argv[2:])
    else:
        raise SystemExit("expected inputs, identities, or run PDB [FITTER_ARGS...]")
