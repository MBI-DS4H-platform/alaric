from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Sequence

# Same rationale as grow.py: the trace kernel is built from many tiny k=9 matrix
# products, where multi-threaded BLAS costs more than it gains. Use --nprocs for
# parallelism unless the caller explicitly overrides these.
for _thread_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import numpy as np

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from anchor import _select_conformers
from grow import (
    SourceConformerCache,
    _precompute_translation_sets,
    run_pooled_trace_grow,
)
from library import config
from parse_pdb import parse_pdb
from poses import discover_organized
from reference import Reference


def _existing_file(path: str) -> Path:
    p = Path(path)
    if not p.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    return p


def _dinucleotide_sequence(seq: str) -> str:
    seq = seq.strip().upper()
    if len(seq) != 2:
        raise argparse.ArgumentTypeError("--sequence must be a dinucleotide (length 2)")
    if any(ch not in set("ACGU") for ch in seq):
        raise argparse.ArgumentTypeError("--sequence must contain only A/C/G/U")
    return seq


def _pdb_code(code: str) -> str:
    code = code.strip()
    if len(code) != 4 or not code[0].isdigit() or not code[1:].isalnum():
        raise argparse.ArgumentTypeError(
            "PDB codes must be 4 chars: one digit + 3 alphanumeric characters (e.g. 1B7F)"
        )
    return code.upper()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anchor-refe",
        description=(
            "Create initial fragment poses from a reference structure: enumerate all "
            "library poses whose nucleobase is within an ovRMSD threshold of the "
            "corresponding nucleobase in the reference."
        ),
    )
    parser.add_argument(
        "--debug", action="store_true", help="Show a full traceback on errors."
    )
    parser.add_argument(
        "--reference",
        required=True,
        type=_existing_file,
        help="Reference PDB file containing the RNA chain.",
    )
    parser.add_argument(
        "--fragment",
        required=True,
        type=int,
        metavar="N",
        help=(
            "1-based fragment position in the reference: residues N and N+1, counted "
            "from the first residue of the reference."
        ),
    )
    parser.add_argument(
        "--sequence",
        type=_dinucleotide_sequence,
        help="Expected dinucleotide sequence. If given, it must match the reference.",
    )
    nuc_group = parser.add_mutually_exclusive_group(required=True)
    nuc_group.add_argument(
        "--first",
        action="store_true",
        help="Fit the base of the first nucleotide (exactly one of --first/--second).",
    )
    nuc_group.add_argument(
        "--second",
        action="store_true",
        help="Fit the base of the second nucleotide (exactly one of --first/--second).",
    )
    parser.add_argument(
        "--ov-rmsd",
        required=True,
        type=float,
        help="Nucleobase RMSD threshold towards the reference (Angstrom).",
    )
    parser.add_argument(
        "--output",
        metavar="POSE_DIR",
        required=True,
        help="Output directory where unorganized .arc.zst pose shards are written.",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=50_000_000,
        metavar="N",
        help=(
            "Approximate weighted pose cache before flushing to disk; unsorted poses "
            "count as 7 and sorted poses count as 1 (default: 50000000)."
        ),
    )
    parser.add_argument(
        "--bucket-size",
        type=int,
        default=16,
        metavar="N",
        help="Grid bucket edge length written into each .arc header (default: 16).",
    )
    parser.add_argument(
        "--unorganized-subdirs",
        action="store_true",
        help=(
            "Write unorganized pose shards as unorganized-PID/RANDOM.arc.zst instead "
            "of flat unorganized-PID-RANDOM.arc.zst files."
        ),
    )
    parser.add_argument(
        "--nprocs",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of parallel worker processes, each handling a slice of conformers "
            "and writing to the same output directory (default: 1)."
        ),
    )
    conformer_group = parser.add_mutually_exclusive_group()
    conformer_group.add_argument(
        "--conformer-range",
        nargs=2,
        type=int,
        default=None,
        metavar=("FIRST", "LAST"),
        help="Restrict the search to conformers with 1-based indices FIRST..LAST inclusive.",
    )
    conformer_group.add_argument(
        "--conformer",
        type=int,
        default=None,
        metavar="INDEX",
        help="Restrict the search to a single 1-based conformer index.",
    )
    parser.add_argument(
        "--pdb-exclude",
        nargs="+",
        default=[],
        type=_pdb_code,
        metavar="PDB",
        help="One or more PDB codes to exclude from the fragment library.",
    )
    return parser


def load_reference_fragment(
    reference_path: str | Path,
    fragment: int,
    sequence: str | None = None,
) -> tuple[str, np.ndarray]:
    """Return the sequence and template-ordered coordinates of a reference fragment.

    Atoms are matched to the mononucleotide templates by name, so a reference whose
    atoms are ordered differently is accepted (and reordered), as in
    util/refe-best-fit.py.
    """
    if fragment < 1:
        raise ValueError("--fragment must be positive")
    ppdb = parse_pdb(Path(reference_path).read_text())
    if len(ppdb) == 0:
        raise ValueError("Reference PDB contains no atoms")
    models = np.unique(ppdb["model"])
    if len(models) > 1:
        ppdb = ppdb[ppdb["model"] == models[0]]
    reference = Reference(
        ppdb,
        rna=True,
        ignore_unknown=False,
        ignore_missing=False,
        ignore_reordered=False,
    )
    inferred_sequence = reference.get_sequence(fragment, fraglen=2)
    if sequence is not None and sequence != inferred_sequence:
        raise ValueError(
            f"--sequence {sequence} does not match reference sequence {inferred_sequence} "
            f"at fragment {fragment}"
        )
    coordinates = reference.get_coordinates(fragment, fraglen=2)
    return inferred_sequence, coordinates.astype(np.float32, copy=False)


def load_library(
    sequence: str,
    *,
    first: bool,
    excluded_pdb_codes: set[str],
):
    """Load the base-only library for the selected nucleotide of the fragment."""
    dinucleotide_libraries, _ = config(verify_checksums=False)
    if sequence not in dinucleotide_libraries:
        raise ValueError(f"Unsupported sequence: {sequence}")
    factory = dinucleotide_libraries[sequence]
    factory.load_rotaconformers()
    pdb_code_filter: str | list[str] | None = None
    if excluded_pdb_codes:
        pdb_code_filter = sorted(excluded_pdb_codes)
    library = factory.create(
        pdb_code=pdb_code_filter,
        nucleotide_mask=np.array([first, not first], dtype=bool),
        only_base=True,
        with_rotaconformers=True,
    )
    if library.rotaconformers is None or library.rotaconformers_index is None:
        raise RuntimeError("Rotaconformers were not loaded")
    if library.atom_mask is None:
        raise RuntimeError("Library was created without an atom mask")
    return library, factory


def reference_source_cache(base_coordinates: np.ndarray) -> SourceConformerCache:
    """Wrap fixed reference coordinates as a single-pose grow source.

    The reference is already in world coordinates, so its rotation is the identity
    and its grid translation is zero; the grow kernel then reduces to fitting every
    library rotaconformer against those coordinates.
    """
    coords = np.ascontiguousarray(base_coordinates, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("reference base coordinates must have shape [N,3]")
    mean0 = coords.mean(axis=0).astype(np.float32, copy=False)
    centered = (coords - mean0).astype(np.float32, copy=False)
    identity = np.eye(3, dtype=np.float32)[None]
    return SourceConformerCache(
        conformer=0,
        coords=coords,
        centered=centered,
        mean0=mean0,
        pose_trace=float(np.einsum("ij,ij->", centered, centered)),
        rotamer_indices=np.zeros(1, dtype=np.int64),
        rotamer_matrices=identity,
        rotamer_flat=np.ascontiguousarray(identity.reshape(1, 9)),
        mean_rotated=np.ascontiguousarray(mean0[None]),
        instance_translations=np.zeros((1, 3), dtype=np.int16),
        instance_source_ids=np.zeros(1, dtype=np.uint32),
        instance_starts=np.zeros(1, dtype=np.int64),
        instance_counts=np.ones(1, dtype=np.int64),
    )


def _run(args: argparse.Namespace) -> int:
    if args.ov_rmsd <= 0.0:
        raise ValueError("--ov-rmsd must be positive")
    if args.cache_size <= 0:
        raise ValueError("--cache-size must be positive")
    if args.bucket_size < 1 or args.bucket_size > 65535:
        raise ValueError("--bucket-size must be in 1..65535")
    if args.nprocs <= 0:
        raise ValueError("--nprocs must be positive")

    output_dir = Path(args.output)
    if discover_organized(output_dir):
        raise ValueError(
            f"--output directory already contains organized poses-*.arc: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    print("[1/5] Loading reference structure...", file=sys.stderr)
    sequence, reference_coordinates = load_reference_fragment(
        args.reference,
        args.fragment,
        args.sequence,
    )
    nucleotide = "first" if args.first else "second"
    print(
        f"      fragment {args.fragment} ({sequence}), {nucleotide} nucleotide",
        file=sys.stderr,
    )

    print("[2/5] Loading fragment library...", file=sys.stderr)
    library, factory = load_library(
        sequence,
        first=bool(args.first),
        excluded_pdb_codes=set(args.pdb_exclude),
    )
    try:
        base_coordinates = reference_coordinates[library.atom_mask]
        if base_coordinates.shape != library.coordinates.shape[1:]:
            raise ValueError(
                "reference nucleobase and library nucleobase have different shapes: "
                f"{base_coordinates.shape} versus {library.coordinates.shape[1:]}"
            )

        conformer_mask = np.ones(len(library.coordinates), dtype=bool)
        if library.conformer_mask is not None:
            conformer_mask &= library.conformer_mask
        selected_conformers = _select_conformers(
            conformer_mask,
            args.conformer_range,
            args.conformer,
        )
        if len(selected_conformers) and int(selected_conformers.max()) > np.iinfo(
            np.uint16
        ).max:
            raise ValueError("Conformer index exceeds uint16 range")

        print(
            f"[3/5] Fitting {len(base_coordinates)} nucleobase atoms against "
            f"{len(selected_conformers)} conformers...",
            file=sys.stderr,
        )
        source_caches = {0: reference_source_cache(base_coordinates)}
        target_to_sources = {
            int(conformer): np.zeros(1, dtype=np.int64)
            for conformer in selected_conformers.tolist()
        }
        target_conformers = np.asarray(selected_conformers, dtype=np.int64)

        print("[4/5] Running pooled trace fit...", file=sys.stderr)
        total_poses = run_pooled_trace_grow(
            source_caches,
            library,
            target_to_sources,
            target_conformers,
            ov_rmsd=args.ov_rmsd,
            output_dir=output_dir,
            cache_size=args.cache_size,
            bucket_size=args.bucket_size,
            translation_sets=_precompute_translation_sets(),
            nprocs=args.nprocs,
            # anchor-refe is a source pool: its poses have no originating pose.
            emit_provenance=False,
            progress_desc="Reference trace fit",
        )
    finally:
        factory.unload_rotaconformers()

    print("[5/5] Finalizing...", file=sys.stderr)
    elapsed = time.perf_counter() - t0
    print(
        (
            f"sequence={sequence} nucleotide={nucleotide} "
            f"conformers={len(target_conformers)} ov_rmsd={args.ov_rmsd} "
            f"accepted_poses={total_poses} elapsed={elapsed:.2f}s"
        ),
        file=sys.stderr,
    )
    print(f"{total_poses} total solutions", file=sys.stderr)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return int(_run(args))
    except SystemExit:
        raise
    except Exception as exc:
        if getattr(args, "debug", False):
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
