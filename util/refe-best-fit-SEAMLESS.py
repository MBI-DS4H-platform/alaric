#!/usr/bin/env python3
"""Run the reference best-fit fitter as one cached Seamless bash transform."""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

from seamlessify_refe_best_fit import seamlessify_refe_best_fit


USAGE = """Usage: refe-best-fit-SEAMLESS.py [WRAPPER_OPTIONS] RNA.pdb [FITTER_OPTIONS]

Wrapper options (before RNA.pdb):
  --output DIR           Write TSV and fitted PDBs here (default: best-fits)
  --fraglib-config FILE  Location hints (default: ~/.alaric/fraglib.yaml)
  --conda ENV            Run the transformation in this conda environment
  -h, --help             Show this help

Required fitter input: --exclude-pdb-code CODE. The only optional fitter
argument is --rotamer-precision 0.5.
Grid spacing is fixed at sqrt(3)/3. --no-header and ignore flags are forbidden;
--pdb-output is managed by the wrapper.

Requires Python 3.12+, Seamless, numpy<2, scipy, ruamel.yaml, biopython and the
fragment library. See documents/refe-best-fit-seamless.md.
"""

def _parse_arguments(argv: list[str]):
    output = Path("best-fits")
    config = Path.home() / ".alaric" / "fraglib.yaml"
    conda_environment = None
    index = 0

    while index < len(argv):
        argument = argv[index]
        if argument in ("-h", "--help"):
            print(USAGE, end="")
            raise SystemExit(0)
        if argument == "--":
            index += 1
            break
        if argument in ("--output", "--fraglib-config", "--conda"):
            if index + 1 >= len(argv):
                raise SystemExit(f"Missing value for {argument}\n{USAGE}")
            value = argv[index + 1]
            if argument == "--output":
                output = Path(value)
            elif argument == "--fraglib-config":
                config = Path(value)
            else:
                conda_environment = value
            index += 2
            continue
        if argument.startswith("-"):
            raise SystemExit(f"Unknown wrapper option: {argument}\n{USAGE}")
        break

    if index >= len(argv):
        raise SystemExit(f"An RNA PDB file is required\n{USAGE}")

    pdb = Path(argv[index]).expanduser().resolve()
    fitter_arguments = argv[index + 1 :]
    for argument in fitter_arguments:
        if argument == "--pdb-output" or argument.startswith("--pdb-output="):
            raise SystemExit(
                "Use --output before RNA.pdb; --pdb-output is managed by the wrapper"
            )

    return (
        output.expanduser().resolve(),
        config.expanduser().resolve(),
        pdb,
        fitter_arguments,
        conda_environment,
    )


def _split_exclude_code(arguments: list[str]) -> tuple[list[str], str]:
    """Move the project-dependent exclude value to its connectable pin."""
    remaining = []
    exclude_code = ""
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument in ("--exclude", "--exclude-pdb-code"):
            if index + 1 >= len(arguments):
                raise SystemExit(f"Missing value for {argument}\n{USAGE}")
            if exclude_code:
                raise SystemExit("Specify the exclude code only once")
            exclude_code = arguments[index + 1].strip().lower()
            index += 2
            continue
        option = next(
            (name for name in ("--exclude-pdb-code=", "--exclude=") if argument.startswith(name)),
            None,
        )
        if option is not None:
            if exclude_code:
                raise SystemExit("Specify the exclude code only once")
            exclude_code = argument[len(option):].strip().lower()
            index += 1
            continue
        remaining.append(argument)
        index += 1
    if not exclude_code:
        raise SystemExit("A library fit requires --exclude-pdb-code CODE")
    if (
        len(exclude_code) != 4
        or not exclude_code[0].isdigit()
        or not re.fullmatch(r"[0-9a-z]{3}", exclude_code[1:])
    ):
        raise SystemExit("PDB codes must be 4 chars: one digit + 3 alphanumeric characters")
    return remaining, exclude_code


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    output, config, pdb, fitter_arguments, conda_environment = _parse_arguments(argv)
    fitter_arguments, exclude_code = _split_exclude_code(fitter_arguments)

    import seamless_config

    seamless_config.init()
    from seamless.util.mount_directory import write_to_directory
    if __package__:
        from . import refe_best_fit_seamless_support as support
    else:
        import refe_best_fit_seamless_support as support

    transformer = seamlessify_refe_best_fit(
        fitter_arguments,
        fraglib_config=config,
        conda_environment=conda_environment,
    )
    identities = json.dumps(
        support.identities(Path(__file__).resolve().parent.parent / "fraglib"),
        sort_keys=True,
        separators=(",", ":"),
    )
    transformer.args.rna_pdb.value = pdb.read_bytes()
    transformer.args.exclude_pdb_code.value = exclude_code
    transformer.args.fraglib_identities.value = identities

    result = transformer.run()
    write_to_directory(
        str(output), result, cleanup=False, deep=True, text_only=False
    )

    expected = output / "best-fits.tsv"
    if not expected.is_file():
        raise RuntimeError(f"Seamless transformation did not produce {expected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
