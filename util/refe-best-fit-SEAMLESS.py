#!/usr/bin/env python3
"""Run the reference best-fit fitter as one cached Seamless bash transform."""
from __future__ import annotations

from pathlib import Path
import sys

from seamlessify_refe_best_fit import seamlessify_refe_best_fit


USAGE = """Usage: refe-best-fit-SEAMLESS.py [WRAPPER_OPTIONS] RNA.pdb [FITTER_OPTIONS]

Wrapper options (before RNA.pdb):
  --output DIR           Write TSV and fitted PDBs here (default: best-fits)
  --fraglib-config FILE  Location hints (default: ~/.alaric/fraglib.yaml)
  --conda ENV            Run the transformation in this conda environment
  -h, --help             Show this help

Fitter options include --exclude-pdb-code CODE, --grid-spacing FLOAT,
--rotamer-precision FLOAT, --ignore-unknown, --ignore-missing,
--ignore-reordered and --no-header. --pdb-output is managed by the wrapper.

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


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    output, config, pdb, fitter_arguments, conda_environment = _parse_arguments(argv)

    import seamless_config

    seamless_config.init()
    from seamless.util.mount_directory import write_to_directory
    transformer = seamlessify_refe_best_fit(
        pdb,
        fitter_arguments,
        fraglib_config=config,
        conda_environment=conda_environment,
    )

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
