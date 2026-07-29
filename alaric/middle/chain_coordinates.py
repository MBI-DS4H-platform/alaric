"""Materialize chains built by ``alaric-chain`` as coordinates.

Chains consist of overlapping dinucleotides: column *j* (fragment *f*) covers
nucleotides *f* and *f+1*, so consecutive fragments share one nucleotide -- fragments
2-3-4 are nucleotides 2+3, 3+4, 4+5. Every internal nucleotide is therefore described
twice, once by each of the two fragments covering it, and the two descriptions are
averaged here. The chain ends are described once.

Input is a build-mode output dir of ``alaric-chain``: one pose dir per fragment, the
chains as 1-based pose indices (``chains.txt``) and their metadata (``chains.json``).

Output is a single ``.ppdb.npy``: an ``(nchains, natoms)`` parsed-PDB array spanning
nucleotides *f_first* .. *f_last+1*, with ``resid`` set to the nucleotide's fragment
position. Row order is chain order; that is the layout ``write_multi_pdb`` turns into a
multi-model PDB (it numbers models by row, not from the ``model`` field).
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

_ALARIC_DIR = Path(__file__).resolve().parents[1]
if str(_ALARIC_DIR) not in sys.path:
    sys.path.insert(0, str(_ALARIC_DIR))

from parse_pdb import atomic_dtype  # noqa: E402
from poses import PoseChunk, select_pose_indices  # noqa: E402
from rmsd import GRID_SPACING, _rotamers_to_matrices  # noqa: E402

from .chain import CHAINS_FILE, CHAINS_METADATA_FILE  # noqa: E402
from .errors import MiddleError  # noqa: E402
from .project import Project  # noqa: E402
from .resolve import resolve_exclude, resolve_sequence  # noqa: E402


class ChainCoordinatesError(MiddleError):
    """Invalid chain-to-coordinates request or data."""


COORDINATES_FILE = "chains.ppdb.npy"
BASES = ("A", "C", "G", "U")


@dataclass
class Column:
    """One fragment of the chain: its pose dir, and how to interpret its poses."""

    pool: str
    fragment: int
    pose_dir: Path
    nposes: int
    sequence: str


# -- metadata / table reading ---------------------------------------------


def load_metadata(chain_dir: Path) -> dict:
    path = chain_dir / CHAINS_METADATA_FILE
    if not path.is_file():
        raise ChainCoordinatesError(
            f"{path} not found: {chain_dir} is not an alaric-chain output dir "
            f"(build one with: alaric-chain GRAPH_JSON -o {chain_dir})"
        )
    metadata = json.loads(path.read_text())
    columns = metadata.get("columns")
    if not columns:
        raise ChainCoordinatesError(f"{path}: no columns")
    fragments = [int(c["fragment"]) for c in columns]
    expected = list(range(fragments[0], fragments[0] + len(fragments)))
    if fragments != expected:
        raise ChainCoordinatesError(
            f"{path}: fragments {fragments} are not consecutive and ascending, so the "
            f"chain does not describe one continuous stretch of nucleotides"
        )
    return metadata


def _chain_range(values: list[int] | None, nchains: int) -> tuple[int, int]:
    """Resolve ``--chain-range`` to a 0-based half-open ``(start, stop)``."""
    if values is None:
        start, stop = 0, nchains
    else:
        first, last = values
        if first <= 0 or last <= 0:
            raise ChainCoordinatesError("--chain-range values must be positive")
        if first > last:
            raise ChainCoordinatesError("--chain-range START must be <= END")
        if first > nchains:
            raise ChainCoordinatesError(
                f"--chain-range starts after the last chain ({nchains})"
            )
        if last > nchains:
            raise ChainCoordinatesError(
                f"--chain-range END={last} exceeds the number of chains ({nchains})"
            )
        start, stop = first - 1, last
    if stop == start:
        raise ChainCoordinatesError("no chains selected")
    return start, stop


def read_chain_table(
    chain_dir: Path, metadata: dict, chain_range: list[int] | None
) -> tuple[np.ndarray, int]:
    """Read the selected chain rows as 1-based pose indices, and where they start."""
    path = chain_dir / metadata.get("chains_file", CHAINS_FILE)
    if not path.is_file():
        raise ChainCoordinatesError(f"{path} not found")
    pools = [c["pool"] for c in metadata["columns"]]
    with path.open() as handle:
        header = handle.readline().rstrip("\n").split("\t")
    if header != pools:
        raise ChainCoordinatesError(
            f"{path} header {header} does not match {CHAINS_METADATA_FILE} pools "
            f"{pools}: the two files are out of sync"
        )

    start, stop = _chain_range(chain_range, int(metadata["nchains"]))
    table = np.loadtxt(
        path,
        dtype=np.int64,
        delimiter="\t",
        skiprows=1 + start,
        max_rows=stop - start,
        ndmin=2,
    )
    if table.shape != (stop - start, len(pools)):
        raise ChainCoordinatesError(
            f"{path}: expected {stop - start} rows of {len(pools)} columns, "
            f"got {table.shape[0]} rows of {table.shape[1]}"
        )
    return table, start


# -- sequence / exclude resolution ----------------------------------------


def _project(metadata: dict, project_root: str | None) -> Project | None:
    root = Path(project_root) if project_root else Path(str(metadata.get("project", "")))
    return Project(root) if root.is_dir() else None


def resolve_sequences(
    metadata: dict, *, sequence: str | None, project: Project | None
) -> list[str]:
    """One dinucleotide sequence per column, in fragment order."""
    columns = metadata["columns"]
    if sequence is not None:
        chain_seq = sequence.strip().upper()
        if len(chain_seq) != len(columns) + 1:
            raise ChainCoordinatesError(
                f"--sequence must be the whole chain sequence: {len(columns) + 1} "
                f"nucleotides for {len(columns)} fragments, got {len(chain_seq)}"
            )
        bad = sorted(set(chain_seq) - set(BASES))
        if bad:
            raise ChainCoordinatesError(
                f"--sequence: not a nucleotide base: {', '.join(bad)}"
            )
        sequences = [chain_seq[i : i + 2] for i in range(len(columns))]
    else:
        sequences = []
        for column in columns:
            seq = column.get("sequence")
            if seq is None and project is not None:
                try:
                    seq = resolve_sequence(project, int(column["fragment"]))
                except MiddleError:
                    seq = None
            if seq is None:
                raise ChainCoordinatesError(
                    f"frag{column['fragment']}: no sequence in "
                    f"{CHAINS_METADATA_FILE} and none resolvable from the project; "
                    f"pass --sequence (the whole chain sequence) or --project-root"
                )
            sequences.append(str(seq).upper())

    for column, seq in zip(columns, sequences):
        if len(seq) != 2:
            raise ChainCoordinatesError(
                f"frag{column['fragment']}: sequence {seq!r} is not a dinucleotide"
            )
    # consecutive fragments share a nucleotide, so they must agree on its base
    for lower, upper, seq_lower, seq_upper in zip(
        columns, columns[1:], sequences, sequences[1:]
    ):
        if seq_lower[1] != seq_upper[0]:
            raise ChainCoordinatesError(
                f"frag{lower['fragment']} ({seq_lower}) and frag{upper['fragment']} "
                f"({seq_upper}) disagree on shared nucleotide "
                f"{upper['fragment']}: {seq_lower[1]} vs {seq_upper[0]}"
            )
    return sequences


def resolve_excluded(
    metadata: dict,
    *,
    exclude: list[str] | None,
    no_exclude: bool,
    project: Project | None,
) -> list[str] | None:
    """PDB codes whose conformers are replaced, as when the poses were generated."""
    if no_exclude:
        return None
    if exclude:
        return [code.strip().lower() for code in exclude]
    from_metadata = metadata.get("exclude")
    if from_metadata:
        return [str(code).lower() for code in from_metadata]
    if project is not None:
        try:
            return resolve_exclude(project, "auto")
        except MiddleError:
            pass
    raise ChainCoordinatesError(
        f"no excluded PDB code in {CHAINS_METADATA_FILE} and none resolvable from the "
        f"project; the library conformers depend on it, so pass --exclude PDB (or "
        f"--no-exclude if the poses were built without exclusion)"
    )


# -- pose coordinates -----------------------------------------------------


def pose_coordinates(
    chunk: PoseChunk, *, coordinates: np.ndarray, library
) -> np.ndarray:
    """Transform a chunk of poses into ``(nposes, natoms, 3)`` world coordinates.

    Poses are grouped by conformer, since rotamers are indexed per conformer. Same math
    as ``write-poses-pdb.py``, over a selected chunk rather than a streamed pose dir.
    """
    conf_ids = chunk.conformers
    rot_ids = chunk.rotamers
    translations = chunk.translations_grid.astype(np.float32, copy=False) * GRID_SPACING
    result = np.empty((len(conf_ids), coordinates.shape[1], 3), dtype=np.float32)

    conf_order = np.argsort(conf_ids, kind="stable")
    sorted_conf = conf_ids[conf_order]
    boundaries = np.flatnonzero(np.diff(sorted_conf)) + 1
    starts = np.concatenate((np.array([0], dtype=np.intp), boundaries))
    stops = np.concatenate((boundaries, np.array([len(conf_order)], dtype=np.intp)))

    for start, stop in zip(starts, stops):
        rows = conf_order[start:stop]
        conf = int(sorted_conf[start])
        if conf >= len(coordinates):
            raise ChainCoordinatesError(f"conformer index {conf} out of range")

        rotamers = library.get_rotamers(conf)
        batch_rot = rot_ids[rows].astype(np.int64, copy=False)
        if batch_rot.size and int(batch_rot.max()) >= len(rotamers):
            raise ChainCoordinatesError(
                f"rotamer index {int(batch_rot.max())} out of range for conformer "
                f"{conf} (n={len(rotamers)})"
            )

        rotations = _rotamers_to_matrices(rotamers[batch_rot])
        transformed = np.einsum("aj,njk->nak", coordinates[conf], rotations)
        transformed += translations[rows, None, :]
        result[rows] = transformed

    return result


# -- chain building -------------------------------------------------------


@contextmanager
def _open_library(libraries: dict, sequence: str, exclude: list[str] | None):
    """Yield the dinucleotide library for ``sequence``, with rotamers attached."""
    try:
        factory = libraries[sequence]
    except KeyError:
        raise ChainCoordinatesError(
            f"sequence {sequence} not available in the fragment library"
        ) from None
    factory.load_rotaconformers()
    try:
        yield factory.create(exclude, with_rotaconformers=True)
    finally:
        factory.unload_rotaconformers()


def _nucleotide_masks(template: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Atom masks for the first and second nucleotide of a dinucleotide template."""
    resids = np.unique(template["resid"])
    if len(resids) != 2:
        raise ChainCoordinatesError(
            f"dinucleotide template has {len(resids)} residues, expected 2"
        )
    return template["resid"] == resids[0], template["resid"] == resids[1]


def build_chain_coordinates(
    columns: list[Column],
    table: np.ndarray,
    *,
    open_library,
    templates: dict[str, np.ndarray],
) -> np.ndarray:
    """Average the overlaps and return one parsed-PDB row per chain.

    Chains share poses, so each pose is transformed once, per column, and expanded
    back over the chains that use it. Columns are grouped by sequence, because
    attaching a library's rotamers is what dominates the cost.
    """
    contributions: dict[int, list[np.ndarray]] = {}
    bases: dict[int, str] = {}

    by_sequence: dict[str, list[int]] = {}
    for j, column in enumerate(columns):
        by_sequence.setdefault(column.sequence, []).append(j)

    for sequence, indices in by_sequence.items():
        with open_library(sequence) as library:
            coordinates = library.coordinates.astype(np.float32, copy=False)
            first_mask, second_mask = _nucleotide_masks(library.template)
            for j in indices:
                column = columns[j]
                ids, inverse = np.unique(table[:, j] - 1, return_inverse=True)
                if int(ids.min()) < 0 or (
                    column.nposes and int(ids.max()) >= column.nposes
                ):
                    raise ChainCoordinatesError(
                        f"{column.pool}: chain pose index out of range 1..."
                        f"{column.nposes} (got {int(ids.min()) + 1}..{int(ids.max()) + 1})"
                    )
                chunk = select_pose_indices(column.pose_dir, ids)
                coords = pose_coordinates(
                    chunk, coordinates=coordinates, library=library
                )
                inverse = np.asarray(inverse).ravel()
                for mask, nucleotide, base in (
                    (first_mask, column.fragment, sequence[0]),
                    (second_mask, column.fragment + 1, sequence[1]),
                ):
                    contributions.setdefault(nucleotide, []).append(
                        coords[:, mask][inverse]
                    )
                    bases[nucleotide] = base

    template_parts = []
    coordinate_parts = []
    for nucleotide in sorted(contributions):
        parts = contributions[nucleotide]
        # the shared nucleotide of two consecutive fragments: average the two copies
        averaged = parts[0] if len(parts) == 1 else np.mean(np.stack(parts), axis=0)
        base = bases[nucleotide]
        try:
            nucleotide_template = templates[base].copy()
        except KeyError:
            raise ChainCoordinatesError(
                f"no mononucleotide template for base {base}"
            ) from None
        if len(nucleotide_template) != averaged.shape[1]:
            raise ChainCoordinatesError(
                f"nucleotide {nucleotide} ({base}): mononucleotide template has "
                f"{len(nucleotide_template)} atoms, but the fragment library "
                f"describes it with {averaged.shape[1]}"
            )
        nucleotide_template["resid"] = nucleotide
        template_parts.append(nucleotide_template)
        coordinate_parts.append(averaged)

    template = np.concatenate(template_parts).astype(atomic_dtype)
    chain_coordinates = np.concatenate(coordinate_parts, axis=1)
    nchains, natoms = chain_coordinates.shape[:2]

    atoms = np.tile(template, nchains).reshape(nchains, natoms)
    atoms["index"] = np.arange(1, natoms + 1, dtype=atomic_dtype["index"])
    atoms["x"] = chain_coordinates[:, :, 0]
    atoms["y"] = chain_coordinates[:, :, 1]
    atoms["z"] = chain_coordinates[:, :, 2]
    return atoms


def chain_coordinates(
    chain_dir: Path,
    *,
    chain_range: list[int] | None = None,
    sequence: str | None = None,
    exclude: list[str] | None = None,
    no_exclude: bool = False,
    project_root: str | None = None,
    verify_checksums: bool = False,
) -> tuple[np.ndarray, int]:
    """Build the parsed-PDB array for a chain dir, and the 0-based first chain in it."""
    chain_dir = Path(chain_dir)
    metadata = load_metadata(chain_dir)
    table, start = read_chain_table(chain_dir, metadata, chain_range)
    project = _project(metadata, project_root)
    sequences = resolve_sequences(metadata, sequence=sequence, project=project)
    excluded = resolve_excluded(
        metadata, exclude=exclude, no_exclude=no_exclude, project=project
    )

    columns = [
        Column(
            pool=entry["pool"],
            fragment=int(entry["fragment"]),
            pose_dir=chain_dir / entry.get("pose_dir", entry["pool"]),
            nposes=int(entry.get("nposes") or 0),
            sequence=seq,
        )
        for entry, seq in zip(metadata["columns"], sequences)
    ]
    for column in columns:
        if not column.pose_dir.is_dir():
            raise ChainCoordinatesError(
                f"{column.pool}: pose dir not found at {column.pose_dir}"
            )

    from library import config, load_mononucleotide_templates

    libraries, _ = config(verify_checksums=verify_checksums)
    templates = load_mononucleotide_templates()

    def open_library(seq: str):
        return _open_library(libraries, seq, excluded)

    atoms = build_chain_coordinates(
        columns, table, open_library=open_library, templates=templates
    )
    return atoms, start


# -- CLI ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alaric-chain-coordinates",
        description=(
            "Materialize chains built by alaric-chain as a parsed-PDB coordinate "
            "array, averaging the nucleotide shared by consecutive fragments."
        ),
    )
    parser.add_argument("chain_dir", help="Build-mode output dir of alaric-chain.")
    parser.add_argument(
        "-o",
        "--output",
        help=f"Parsed-PDB array to write (default: CHAIN_DIR/{COORDINATES_FILE}).",
    )
    parser.add_argument(
        "--chain-range",
        nargs=2,
        metavar=("START", "END"),
        type=int,
        help="1-based, end-inclusive range of chains to write.",
    )
    parser.add_argument(
        "--sequence",
        help="Whole chain sequence (one nucleotide more than the number of "
        "fragments), overriding the sequences in chains.json.",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        metavar="PDB",
        help="PDB code(s) excluded from the fragment library, overriding chains.json. "
        "Must match what the poses were built with.",
    )
    parser.add_argument(
        "--no-exclude",
        action="store_true",
        help="Use the library unmodified (the poses were built without exclusion).",
    )
    parser.add_argument(
        "--project-root",
        help="Project root to resolve sequence/exclude from, if chains.json lacks them.",
    )
    parser.add_argument(
        "--verify-checksums",
        action="store_true",
        help="Enable fraglib checksum verification when loading the library config.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.exclude and args.no_exclude:
        parser.error("--exclude and --no-exclude are mutually exclusive")

    chain_dir = Path(args.chain_dir)
    output = Path(args.output) if args.output else chain_dir / COORDINATES_FILE
    try:
        atoms, start = chain_coordinates(
            chain_dir,
            chain_range=args.chain_range,
            sequence=args.sequence,
            exclude=args.exclude,
            no_exclude=args.no_exclude,
            project_root=args.project_root,
            verify_checksums=args.verify_checksums,
        )
    except ChainCoordinatesError as exc:
        parser.exit(2, f"error: {exc}\n")

    np.save(output, atoms)
    print(
        f"wrote chains {start + 1}..{start + len(atoms)} "
        f"({len(atoms)} models x {atoms.shape[1]} atoms) to {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
