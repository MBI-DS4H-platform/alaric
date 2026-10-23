"""Input declarations and materialization setup for the best-fit wrappers."""
from __future__ import annotations

import contextlib
import itertools
import json
import os
from pathlib import Path
import runpy
import sys


MODULES = (
    "library.py", "parse_pdb.py", "reference.py", "superimpose.py",
    "write_pdb.py", "rna_pdb.py", "mutate.py",
)


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


def input_paths() -> list[Path]:
    """Return original source and template paths; Seamless handles all staging."""
    root = Path(__file__).resolve().parent.parent
    templates = root / "fraglib" / "templates"
    inputs = [root / "alaric" / module for module in MODULES]
    inputs.append(root / "util" / "refe-best-fit.py")
    for sequence in [*"ACGU", *("".join(x) for x in itertools.product("ACGU", repeat=2))]:
        inputs.append((templates / f"{sequence}-ppdb.npy").resolve())
    return inputs


def run(pdb: str, arguments: list[str]) -> None:
    """Use materialization hints and run with source files supplied by Seamless."""
    # Seamless can symlink each file to a content-addressed buffer. Resolve the
    # workspace location, not the file symlink's unrelated cache directory.
    workspace = Path(__file__).absolute().parent
    library = library_module(workspace)
    library._FRAGLIB_YAML_PATH = Path(os.environ["fraglib_config"]).absolute()
    config = library._fraglib_config()
    # Bash transformers materialize value pins as files. Large strings are not
    # also exported as environment variables, so read the Seamless-made pin.
    expected = json.loads(Path("fraglib_identities").read_text())
    if config["fraglen"] != expected["fraglen"]:
        raise ValueError("refe-best-fit requires fraglen: 2")
    # The library is immutable and identified by the ordinary checksum manifest.
    # Its installation is trusted to materialize those identities, as in the
    # existing fitter's config(verify_checksums=False). No data is rehashed here.
    config["templates_dir"] = "."
    output = Path("best-fit-output")
    output.mkdir()
    sys.argv = [str(workspace / "refe-best-fit.py"), pdb, *arguments,
                "--pdb-output", str(output / "pdbs")]
    with (output / "best-fits.tsv").open("w") as handle, contextlib.redirect_stdout(handle):
        runpy.run_path(str(workspace / "refe-best-fit.py"), run_name="__main__")


if __name__ == "__main__":
    if sys.argv[1:] == ["inputs"]:
        print("\n".join(str(path) for path in input_paths()))
    elif sys.argv[1:] == ["identities"]:
        root = Path(__file__).resolve().parent.parent
        print(json.dumps(identities(root / "fraglib"), sort_keys=True, separators=(",", ":")))
    elif len(sys.argv) >= 3 and sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3:])
    else:
        raise SystemExit("expected inputs, identities, or run PDB [FITTER_ARGS...]")
