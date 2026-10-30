#!/usr/bin/env python3
"""Convert A3 TSV tables to schema-1 computed RMSD thresholds."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

try:
    # In a flat Seamless sandbox the declared dependency is staged beside this
    # script.  A direct checkout invocation keeps it under ../alaric instead.
    from threshold_rounding import strictly_above_cent
except ModuleNotFoundError as exc:
    if exc.name != "threshold_rounding":
        raise
    checkout_alaric = Path(__file__).resolve().parent.parent / "alaric"
    sys.path.insert(0, str(checkout_alaric))
    from threshold_rounding import strictly_above_cent


PAIR_COLUMNS = ("down", "up", "sequence", "ovRMSD", "cRMSD")
BASE_COLUMNS = (
    "fragment", "sequence", "nucleotide1_ovRMSD", "nucleotide2_ovRMSD"
)


def _read_table(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != columns:
            raise ValueError(
                f"{path} columns must be {' '.join(columns)}, got {reader.fieldnames}"
            )
        rows = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed tab-separated row in {path}")
            rows.append(row)
        return rows


def _rmsd(raw_text: str, path: Path) -> tuple[str, float]:
    threshold = strictly_above_cent(raw_text)
    raw_number = float(raw_text)
    if not math.isfinite(raw_number):
        raise ValueError(f"Non-finite RMSD in {path}: {raw_text!r}")
    return threshold, raw_number


def build_thresholds(pairs_path: Path, bases_path: Path) -> dict:
    pair_rows = []
    for row in _read_table(pairs_path, PAIR_COLUMNS):
        ovrmsd, ovrmsd_raw = _rmsd(row["ovRMSD"], pairs_path)
        crmsd, crmsd_raw = _rmsd(row["cRMSD"], pairs_path)
        pair_rows.append(
            {
                "down": int(row["down"]),
                "up": int(row["up"]),
                "sequence": row["sequence"],
                "ovrmsd": ovrmsd,
                "crmsd": crmsd,
                "ovrmsd_raw": ovrmsd_raw,
                "crmsd_raw": crmsd_raw,
            }
        )

    base_rows = []
    for row in _read_table(bases_path, BASE_COLUMNS):
        first, first_raw = _rmsd(row["nucleotide1_ovRMSD"], bases_path)
        second, second_raw = _rmsd(row["nucleotide2_ovRMSD"], bases_path)
        base_rows.append(
            {
                "fragment": int(row["fragment"]),
                "sequence": row["sequence"],
                "nuc1_ovrmsd": first,
                "nuc2_ovrmsd": second,
                "nuc1_ovrmsd_raw": first_raw,
                "nuc2_ovrmsd_raw": second_raw,
            }
        )
    return {"schema": 1, "pairs": pair_rows, "bases": base_rows}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", type=Path)
    parser.add_argument("bases", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_thresholds(args.pairs, args.bases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
