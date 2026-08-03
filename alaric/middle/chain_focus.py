"""Focus an ``alaric-chain`` build on selected pose-pool columns.

The focused directory reuses the selected pose directories as symlinks.  Its
``chains.txt`` contains the selected columns with duplicate rows removed, while
``chain-provenance.txt`` maps every input-chain row to its 1-based row number in
that deduplicated table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .chain import CHAINS_FILE, CHAINS_METADATA_FILE
from .errors import MiddleError


CHAIN_PROVENANCE_FILE = "chain-provenance.txt"


class ChainFocusError(MiddleError):
    """Invalid chain-focus request or input directory."""


def _direct_entry(name: object, what: str) -> str:
    """Return a safe, single-directory-entry name."""
    if not isinstance(name, str) or not name or name in {".", ".."}:
        raise ChainFocusError(f"invalid {what}: {name!r}")
    path = Path(name)
    if path.is_absolute() or len(path.parts) != 1:
        raise ChainFocusError(f"invalid {what}: {name!r} is not a directory entry")
    return name


def _load_input(chain_dir: Path) -> tuple[dict, np.ndarray]:
    metadata_path = chain_dir / CHAINS_METADATA_FILE
    if not metadata_path.is_file():
        raise ChainFocusError(
            f"{metadata_path} not found: {chain_dir} is not an alaric-chain output dir"
        )
    try:
        metadata = json.loads(metadata_path.read_text())
    except json.JSONDecodeError as exc:
        raise ChainFocusError(f"{metadata_path}: invalid JSON: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ChainFocusError(f"{metadata_path}: metadata must be an object")

    columns = metadata.get("columns")
    if not isinstance(columns, list) or not columns:
        raise ChainFocusError(f"{metadata_path}: no columns")
    pools: list[str] = []
    for column in columns:
        if not isinstance(column, dict):
            raise ChainFocusError(f"{metadata_path}: invalid column")
        pool = _direct_entry(column.get("pool"), "pool name")
        pose_dir = _direct_entry(column.get("pose_dir", pool), f"pose dir for {pool!r}")
        if pose_dir != pool:
            raise ChainFocusError(
                f"{metadata_path}: {pool!r} pose_dir is {pose_dir!r}; "
                "expected a directory named after its pool"
            )
        if not (chain_dir / pool).is_dir():
            raise ChainFocusError(f"{pool!r}: pose dir not found at {chain_dir / pool}")
        pools.append(pool)
    if len(set(pools)) != len(pools):
        raise ChainFocusError(f"{metadata_path}: duplicate pool columns are unsupported")

    chains_path = chain_dir / metadata.get("chains_file", CHAINS_FILE)
    if not chains_path.is_file():
        raise ChainFocusError(f"{chains_path} not found")
    lines = chains_path.read_text().splitlines()
    if not lines:
        raise ChainFocusError(f"{chains_path}: missing header")
    header = lines[0].split("\t")
    if header != pools:
        raise ChainFocusError(
            f"{chains_path} header {header} does not match {CHAINS_METADATA_FILE} pools {pools}"
        )

    rows: list[list[int]] = []
    for line_no, line in enumerate(lines[1:], start=2):
        fields = line.split("\t")
        if len(fields) != len(pools):
            raise ChainFocusError(
                f"{chains_path}:{line_no}: expected {len(pools)} columns, got {len(fields)}"
            )
        try:
            row = [int(value) for value in fields]
        except ValueError as exc:
            raise ChainFocusError(f"{chains_path}:{line_no}: invalid pose index") from exc
        if any(value < 1 for value in row):
            raise ChainFocusError(f"{chains_path}:{line_no}: pose indices must be positive")
        rows.append(row)

    try:
        nchains = int(metadata["nchains"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ChainFocusError(f"{metadata_path}: invalid nchains") from exc
    if nchains != len(rows):
        raise ChainFocusError(
            f"{metadata_path}: nchains is {nchains}, but {chains_path} has {len(rows)} rows"
        )
    return metadata, np.asarray(rows, dtype=np.int64).reshape(len(rows), len(pools))


def focus_chains(chain_dir: Path, pools: list[str], output_dir: Path) -> dict:
    """Write a focused, deduplicated chain directory and return its metadata."""
    chain_dir = Path(chain_dir).resolve()
    output_dir = Path(output_dir)
    if not chain_dir.is_dir():
        raise ChainFocusError(f"input chains dir not found: {chain_dir}")
    if output_dir.resolve() == chain_dir:
        raise ChainFocusError("output chains dir must differ from input chains dir")
    if output_dir.exists() and (output_dir.is_symlink() or not output_dir.is_dir()):
        raise ChainFocusError(f"output path is not a directory: {output_dir}")
    if output_dir.is_dir() and any(output_dir.iterdir()):
        raise ChainFocusError(f"output chains dir is not empty: {output_dir}")

    metadata, table = _load_input(chain_dir)
    columns = metadata["columns"]
    requested = [_direct_entry(pool, "requested pool") for pool in pools]
    if len(set(requested)) != len(requested):
        raise ChainFocusError("requested pools must not contain duplicates")
    known = {column["pool"] for column in columns}
    unknown = [pool for pool in requested if pool not in known]
    if unknown:
        raise ChainFocusError(f"pool(s) not in {CHAINS_METADATA_FILE}: {', '.join(unknown)}")

    # The build's fragment order remains authoritative, even if the command-line
    # pool names were supplied in another order.
    selected_indices = [i for i, column in enumerate(columns) if column["pool"] in requested]
    selected_columns = [columns[i].copy() for i in selected_indices]
    selected_table = table[:, selected_indices]

    unique_rows: list[np.ndarray] = []
    provenance: list[int] = []
    row_numbers: dict[tuple[int, ...], int] = {}
    for row in selected_table:
        key = tuple(int(value) for value in row)
        focused_number = row_numbers.get(key)
        if focused_number is None:
            focused_number = len(unique_rows) + 1
            row_numbers[key] = focused_number
            unique_rows.append(row)
        provenance.append(focused_number)
    focused_table = np.asarray(unique_rows, dtype=np.int64).reshape(
        len(unique_rows), len(selected_indices)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for column in selected_columns:
        pool = column["pool"]
        (output_dir / pool).symlink_to(chain_dir / pool, target_is_directory=True)

    with (output_dir / CHAINS_FILE).open("w") as handle:
        handle.write("\t".join(column["pool"] for column in selected_columns) + "\n")
        np.savetxt(handle, focused_table, fmt="%d", delimiter="\t")
    with (output_dir / CHAIN_PROVENANCE_FILE).open("w") as handle:
        np.savetxt(handle, np.asarray(provenance, dtype=np.int64), fmt="%d")

    focused_metadata = metadata.copy()
    focused_metadata.update(
        {
            "chains_file": CHAINS_FILE,
            "chain_provenance_file": CHAIN_PROVENANCE_FILE,
            "nchains": len(focused_table),
            "columns": selected_columns,
        }
    )
    (output_dir / CHAINS_METADATA_FILE).write_text(
        json.dumps(focused_metadata, indent=2) + "\n"
    )
    return focused_metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="alaric-chain-focus",
        description="Focus an alaric-chain output on selected pool columns.",
    )
    parser.add_argument("chain_dir", help="Build-mode output dir of alaric-chain.")
    parser.add_argument("pools", nargs="+", metavar="POOL", help="Pool columns to keep.")
    parser.add_argument("-o", "--output-dir", required=True, help="Focused output chains dir.")
    args = parser.parse_args(argv)
    try:
        metadata = focus_chains(Path(args.chain_dir), args.pools, Path(args.output_dir))
    except ChainFocusError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"total chains: {metadata['nchains']}")
    print(f"wrote {Path(args.output_dir) / CHAINS_FILE}")
    print(f"wrote {Path(args.output_dir) / CHAIN_PROVENANCE_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
