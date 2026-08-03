"""Focus an ``alaric-chain`` build on selected pose-pool columns.

The focused directory reuses the selected pose directories as symlinks.  Its
``chains.txt`` contains the selected columns with duplicate rows removed, while
``chain-provenance.txt`` maps every input-chain row to its 1-based row number in
that deduplicated table.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import sys
from pathlib import Path

import numpy as np

from .chain import CHAINS_FILE, CHAINS_METADATA_FILE
from .chain_coordinates import ChainCoordinatesError, iter_chain_table_chunks
from .errors import MiddleError


CHAIN_PROVENANCE_FILE = "chain-provenance.txt"
DEFAULT_CHAIN_CHUNK_SIZE = 10_000_000


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


def _load_input(chain_dir: Path) -> dict:
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

    try:
        nchains = int(metadata["nchains"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ChainFocusError(f"{metadata_path}: invalid nchains") from exc
    if nchains < 0:
        raise ChainFocusError(f"{metadata_path}: nchains must be non-negative")
    return metadata


def focus_chains(
    chain_dir: Path,
    pools: list[str],
    output_dir: Path,
    *,
    chunk_size: int = DEFAULT_CHAIN_CHUNK_SIZE,
    write_provenance: bool = True,
) -> dict:
    """Write a focused, deduplicated chain directory and return its metadata."""
    chain_dir = Path(chain_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not chain_dir.is_dir():
        raise ChainFocusError(f"input chains dir not found: {chain_dir}")
    if output_dir.resolve() == chain_dir:
        raise ChainFocusError("output chains dir must differ from input chains dir")
    if output_dir.exists() and (output_dir.is_symlink() or not output_dir.is_dir()):
        raise ChainFocusError(f"output path is not a directory: {output_dir}")
    if output_dir.is_dir() and any(output_dir.iterdir()):
        raise ChainFocusError(f"output chains dir is not empty: {output_dir}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    metadata = _load_input(chain_dir)
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
    output_dir.mkdir(parents=True, exist_ok=True)
    for column in selected_columns:
        pool = column["pool"]
        link = output_dir / pool
        target = chain_dir / pool
        try:
            target = Path(os.path.relpath(target, start=link.parent))
        except ValueError:
            # Different drives cannot be expressed as a relative path.
            pass
        link.symlink_to(target, target_is_directory=True)

    # Focused rows are guaranteed to be much fewer than input rows. Keep just
    # this first-seen map in memory; input rows themselves remain chunked.
    row_numbers: dict[tuple[int, ...], int] = {}
    try:
        with (
            (output_dir / CHAINS_FILE).open("w") as chains_handle,
            (
                (output_dir / CHAIN_PROVENANCE_FILE).open("w")
                if write_provenance
                else nullcontext()
            ) as provenance_handle,
        ):
            chains_handle.write(
                "\t".join(column["pool"] for column in selected_columns) + "\n"
            )
            try:
                chunks = iter_chain_table_chunks(
                    chain_dir, metadata, rows_per_chunk=chunk_size
                )
                for rows in chunks:
                    if np.any(rows < 1):
                        raise ChainFocusError("chain pose indices must be positive")
                    selected_rows = rows[:, selected_indices]
                    unique, first, inverse = np.unique(
                        selected_rows,
                        axis=0,
                        return_index=True,
                        return_inverse=True,
                    )
                    numbers = np.empty(len(unique), dtype=np.int64)
                    is_new = np.zeros(len(unique), dtype=bool)
                    in_source_order = np.argsort(first, kind="stable")
                    for unique_index in in_source_order:
                        key = tuple(int(value) for value in unique[unique_index])
                        number = row_numbers.get(key)
                        if number is None:
                            number = len(row_numbers) + 1
                            row_numbers[key] = number
                            is_new[unique_index] = True
                        numbers[unique_index] = number
                    new_in_source_order = in_source_order[is_new[in_source_order]]
                    if len(new_in_source_order):
                        np.savetxt(
                            chains_handle,
                            unique[new_in_source_order],
                            fmt="%d",
                            delimiter="\t",
                        )
                    if provenance_handle is not None:
                        np.savetxt(provenance_handle, numbers[inverse], fmt="%d")
            except ChainCoordinatesError as exc:
                raise ChainFocusError(str(exc)) from None
    except OSError as exc:
        raise ChainFocusError(str(exc)) from exc

    focused_metadata = metadata.copy()
    focused_metadata.pop("chain_provenance_file", None)
    focused_metadata.update(
        {
            "chains_file": CHAINS_FILE,
            "nchains": len(row_numbers),
            "columns": selected_columns,
        }
    )
    if write_provenance:
        focused_metadata["chain_provenance_file"] = CHAIN_PROVENANCE_FILE
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
    parser.add_argument(
        "--no-provenance",
        action="store_true",
        help=f"Do not write {CHAIN_PROVENANCE_FILE}.",
    )
    args = parser.parse_args(argv)
    try:
        metadata = focus_chains(
            Path(args.chain_dir),
            args.pools,
            Path(args.output_dir),
            write_provenance=not args.no_provenance,
        )
    except ChainFocusError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"total chains: {metadata['nchains']}")
    print(f"wrote {Path(args.output_dir) / CHAINS_FILE}")
    if not args.no_provenance:
        print(f"wrote {Path(args.output_dir) / CHAIN_PROVENANCE_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
