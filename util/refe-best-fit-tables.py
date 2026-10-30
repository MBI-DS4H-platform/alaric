#!/usr/bin/env python3
"""Produce A3 pair and base RMSD tables from one best-fit TSV."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import re
import runpy
import sys


ROOT = Path(__file__).resolve().parent.parent
ALARIC_DIR = ROOT / "alaric"
if str(ALARIC_DIR) not in sys.path:
    sys.path.insert(0, str(ALARIC_DIR))


def _load_library_identity(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read library identity pin {path}: {exc}") from exc
    if type(value) is not dict:
        raise ValueError("fraglib_identities must be a schema-1 fraglen-2 identity")
    if type(value.get("schema")) is not int or value["schema"] != 1:
        raise ValueError("fraglib_identities must have schema: 1")
    if type(value.get("fraglen")) is not int or value["fraglen"] != 2:
        raise ValueError("fraglib_identities must have fraglen: 2")
    checksums = value.get("sha256")
    if type(checksums) is not dict or not checksums:
        raise ValueError("fraglib_identities must include a non-empty sha256 object")
    if any(
        not isinstance(name, str)
        or not name
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        for name, digest in checksums.items()
    ):
        raise ValueError("fraglib_identities contains an invalid SHA-256 entry")
    return value


def _run_table_script(script: Path, arguments: list[str]) -> str:
    """Run an established table script and capture its exact TSV text."""
    old_argv = sys.argv
    output = io.StringIO()
    try:
        sys.argv = [str(script), *arguments]
        with redirect_stdout(output):
            try:
                runpy.run_path(str(script), run_name="__main__")
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    raise
    finally:
        sys.argv = old_argv
    return output.getvalue()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("best_fits", type=Path)
    parser.add_argument("rna_pdb", type=Path)
    parser.add_argument("--exclude-pdb-code", required=True)
    parser.add_argument("--fraglib-identities", type=Path, required=True)
    parser.add_argument("--fraglib-config", type=Path, required=True)
    parser.add_argument("--templates-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    exclude_code = args.exclude_pdb_code.strip().lower()
    if (
        len(exclude_code) != 4
        or not exclude_code[0].isdigit()
        or not exclude_code[1:].isalnum()
    ):
        raise ValueError("exclude-pdb-code must be a four-character PDB code")
    _load_library_identity(args.fraglib_identities)

    import library

    library._FRAGLIB_YAML_PATH = args.fraglib_config.expanduser().resolve()
    library_config = library._fraglib_config()
    if library_config.get("fraglen") != 2:
        raise ValueError("refe-best-fit-tables requires fraglen: 2")
    if "crmsds" not in library_config:
        raise ValueError("fraglib.yaml must include the cRMSD configuration key")
    # The template files are declared and staged inputs. Keep the library
    # locator independent of their location on the submitting host.
    library_config["templates_dir"] = str(args.templates_dir.expanduser().resolve())

    util_dir = Path(__file__).resolve().parent
    common = [
        str(args.best_fits.resolve()),
        str(args.rna_pdb.resolve()),
        "--exclude-pdb-code",
        exclude_code,
    ]
    pairs = _run_table_script(
        util_dir / "refe-best-fit-pairs.py", [common[0], *common[2:]]
    )
    bases = _run_table_script(
        util_dir / "refe-best-fit-bases.py",
        [
            common[0],
            common[1],
            *common[2:],
            "--templates-dir",
            str(args.templates_dir.resolve()),
        ],
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "pairs.tsv").write_text(pairs, encoding="utf-8")
    (args.output / "bases.tsv").write_text(bases, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
