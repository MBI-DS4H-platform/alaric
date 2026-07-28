"""Numeric checks for the anchor-refe action (requires the fragment library)."""

from __future__ import annotations

from math import sqrt
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / ".alaric"))

from anchor_refe import load_library, load_reference_fragment, main  # noqa: E402
from organize import main as organize_main  # noqa: E402
from poses import PoseReader, discover_organized  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402


GRID_SPACING = sqrt(3) / 3
REFERENCE = HERE / "pdbs" / "rna.pdb"
EXCLUDE = "1b7f"
# Brute force scans this many grid steps around the optimal translation of each
# rotamer; the threshold used in the tests keeps every accepted pose well inside.
BRUTE_FORCE_GRID_RADIUS = 2


def _library(fragment: int, first: bool):
    sequence, reference_coordinates = load_reference_fragment(REFERENCE, fragment)
    library, factory = load_library(
        sequence,
        first=first,
        excluded_pdb_codes={EXCLUDE.upper()},
    )
    base = reference_coordinates[library.atom_mask].astype(np.float64)
    return sequence, base, library, factory


def _rotated_base(library, conformer: int) -> np.ndarray:
    coords = library.coordinates[conformer].astype(np.float64)
    rotamers = library.get_rotamers(conformer)
    if rotamers.ndim == 2:
        rotamers = Rotation.from_rotvec(rotamers).as_matrix()
    return np.einsum("aj,njk->nak", coords, rotamers)


def _brute_force_poses(
    base: np.ndarray,
    library,
    conformers: list[int],
    ov_rmsd: float,
) -> set[tuple[int, int, int, int, int]]:
    natoms = len(base)
    radius = BRUTE_FORCE_GRID_RADIUS
    axis = np.arange(-radius, radius + 1)
    offsets = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    accepted: set[tuple[int, int, int, int, int]] = set()
    for conformer in conformers:
        rotated = _rotated_base(library, conformer)
        continuous = base.mean(axis=0)[None] - rotated.mean(axis=1)
        best_grid = np.rint(continuous / GRID_SPACING).astype(np.int64)
        for offset in offsets:
            grid = best_grid + offset
            difference = rotated + (grid * GRID_SPACING)[:, None, :] - base[None]
            rmsd = np.sqrt(np.einsum("nij,nij->n", difference, difference) / natoms)
            for rotamer in np.flatnonzero(rmsd < ov_rmsd):
                accepted.add((conformer, int(rotamer), *grid[rotamer].tolist()))
    return accepted


def _read_poses(pose_dir: Path) -> set[tuple[int, int, int, int, int]]:
    poses: set[tuple[int, int, int, int, int]] = set()
    if not discover_organized(pose_dir):  # an empty pool writes no .arc files
        return poses
    for chunk in PoseReader(pose_dir).iter_chunks():
        for conformer, rotamer, translation in zip(
            chunk.conformers, chunk.rotamers, chunk.translations_grid
        ):
            poses.add(
                (int(conformer), int(rotamer), *[int(x) for x in translation])
            )
    return poses


def _run_anchor_refe(
    output_dir: Path,
    *,
    fragment: int,
    nucleotide: str,
    ov_rmsd: float,
    extra: list[str],
) -> set[tuple[int, int, int, int, int]]:
    status = main(
        [
            "--debug",
            "--reference",
            str(REFERENCE),
            "--fragment",
            str(fragment),
            f"--{nucleotide}",
            "--ov-rmsd",
            str(ov_rmsd),
            "--output",
            str(output_dir),
            "--pdb-exclude",
            EXCLUDE,
            *extra,
        ]
    )
    assert status == 0
    assert organize_main([str(output_dir), "--compress"]) == 0
    return _read_poses(output_dir)


@pytest.mark.parametrize(
    "fragment,nucleotide,conformer,ov_rmsd",
    [
        (5, "second", 3995, 0.6),  # pyrimidine base, tight threshold
        (5, "first", 3995, 1.0),  # same conformer, wide threshold
        (1, "first", 5378, 0.7),  # purine base
    ],
)
def test_anchor_refe_reproduces_brute_force_single_conformer(
    tmp_path: Path,
    fragment: int,
    nucleotide: str,
    conformer: int,
    ov_rmsd: float,
) -> None:
    poses = _run_anchor_refe(
        tmp_path / "poses",
        fragment=fragment,
        nucleotide=nucleotide,
        ov_rmsd=ov_rmsd,
        extra=["--conformer", str(conformer)],
    )

    _sequence, base, library, factory = _library(fragment, nucleotide == "first")
    try:
        expected = _brute_force_poses(base, library, [conformer - 1], ov_rmsd)
    finally:
        factory.unload_rotaconformers()

    assert poses == expected
    assert poses


def test_anchor_refe_matches_brute_force_and_is_parallel_deterministic(
    tmp_path: Path,
) -> None:
    fragment, ov_rmsd = 5, 0.7
    first_conformer, last_conformer = 3990, 4010
    serial = _run_anchor_refe(
        tmp_path / "serial",
        fragment=fragment,
        nucleotide="second",
        ov_rmsd=ov_rmsd,
        extra=["--conformer-range", str(first_conformer), str(last_conformer)],
    )
    parallel = _run_anchor_refe(
        tmp_path / "parallel",
        fragment=fragment,
        nucleotide="second",
        ov_rmsd=ov_rmsd,
        extra=[
            "--conformer-range",
            str(first_conformer),
            str(last_conformer),
            "--nprocs",
            "4",
            "--unorganized-subdirs",
        ],
    )
    # Two chunks covering the same conformer range, as a chunk deployment would do.
    chunked = _run_anchor_refe(
        tmp_path / "chunk-a",
        fragment=fragment,
        nucleotide="second",
        ov_rmsd=ov_rmsd,
        extra=["--conformer-range", str(first_conformer), "4000"],
    ) | _run_anchor_refe(
        tmp_path / "chunk-b",
        fragment=fragment,
        nucleotide="second",
        ov_rmsd=ov_rmsd,
        extra=["--conformer-range", "4001", str(last_conformer)],
    )

    _sequence, base, library, factory = _library(fragment, first=False)
    try:
        expected = _brute_force_poses(
            base,
            library,
            list(range(first_conformer - 1, last_conformer)),
            ov_rmsd,
        )
        assert serial == expected
        assert parallel == expected
        assert chunked == expected

        # Every emitted pose really is within the threshold.
        rmsds = []
        for conformer in sorted({pose[0] for pose in serial}):
            rotated = _rotated_base(library, conformer)
            rows = [pose for pose in serial if pose[0] == conformer]
            grid = np.array([pose[2:] for pose in rows], dtype=np.float64)
            rotamers = np.array([pose[1] for pose in rows], dtype=np.int64)
            difference = (
                rotated[rotamers] + (grid * GRID_SPACING)[:, None, :] - base[None]
            )
            rmsds.append(
                np.sqrt(np.einsum("nij,nij->n", difference, difference) / len(base))
            )
        assert np.concatenate(rmsds).max() < ov_rmsd
    finally:
        factory.unload_rotaconformers()


def test_anchor_refe_brackets_the_best_fitting_reference_pose(tmp_path: Path) -> None:
    """The best-fit fixture pose is found above its base RMSD and missed below it."""
    fixture = np.genfromtxt(
        HERE / "data" / "refe-best-fit.tsv", names=True, dtype=None, encoding=None
    )
    row = fixture[fixture["fragment"] == 5][0]
    conformer = int(row["conformer"]) - 1
    rotamer = int(row["rotamer"]) - 1
    grid = (int(row["grid_x"]), int(row["grid_y"]), int(row["grid_z"]))

    _sequence, base, library, factory = _library(5, first=True)
    try:
        rotated = _rotated_base(library, conformer)[rotamer]
        difference = rotated + np.array(grid) * GRID_SPACING - base
        base_rmsd = float(np.sqrt((difference * difference).sum() / len(base)))
    finally:
        factory.unload_rotaconformers()

    pose = (conformer, rotamer, *grid)
    above = _run_anchor_refe(
        tmp_path / "above",
        fragment=5,
        nucleotide="first",
        ov_rmsd=base_rmsd + 0.05,
        extra=["--conformer", str(conformer + 1)],
    )
    below = _run_anchor_refe(
        tmp_path / "below",
        fragment=5,
        nucleotide="first",
        ov_rmsd=base_rmsd - 0.05,
        extra=["--conformer", str(conformer + 1)],
    )
    assert pose in above
    assert pose not in below


def test_anchor_refe_rejects_mismatching_sequence() -> None:
    with pytest.raises(ValueError, match="does not match reference sequence"):
        load_reference_fragment(REFERENCE, 5, "AC")
