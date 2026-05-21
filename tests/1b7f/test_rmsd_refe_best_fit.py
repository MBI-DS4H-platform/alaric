from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "code"))

from library import config  # noqa: E402
from parse_pdb import parse_pdb  # noqa: E402
from poses import PoseReader, pack_pool, write_arc_file  # noqa: E402
from rmsd import iter_rmsd_chunks_with_library, write_npy_output_chunks  # noqa: E402
from write_pdb import write_pdb  # noqa: E402


def test_rmsd_reproduces_refe_best_fit_fixture(tmp_path: Path) -> None:
    fixture = np.genfromtxt(
        HERE / "data" / "refe-best-fit.tsv",
        names=True,
        dtype=None,
        encoding=None,
    )
    if fixture.ndim == 0:
        fixture = fixture.reshape(1)

    rows_by_sequence = {}
    for row in fixture:
        rows_by_sequence.setdefault(str(row["sequence"]), []).append(row)

    libraries, _ = config(verify_checksums=False)
    reference = parse_pdb((HERE / "pdbs" / "rna.pdb").read_text())
    checked_two_residue_reference = False
    for sequence, rows in rows_by_sequence.items():
        factory = libraries[sequence]
        factory.load_rotaconformers()
        try:
            library = factory.create("1b7f", with_rotaconformers=True)
            for row in rows:
                fragment = int(row["fragment"])
                conformer = int(row["conformer"]) - 1
                rotamer = int(row["rotamer"]) - 1
                grid = np.array(
                    [[int(row["grid_x"]), int(row["grid_y"]), int(row["grid_z"])]],
                    dtype=np.int16,
                )

                packed = pack_pool(
                    np.array([conformer], dtype=np.uint16),
                    np.array([rotamer], dtype=np.uint16),
                    grid,
                    bucket_size=1,
                )
                assert len(packed) == 1

                pose_dir = tmp_path / f"frag-{fragment}"
                arc_path = pose_dir / "poses-1.arc"
                M, O, C, P = packed[0]
                write_arc_file(arc_path, M, O, C, P, bucket_size=1, zstd=False)
                assert PoseReader.get_nposes(pose_dir) == 1

                chunks = list(
                    iter_rmsd_chunks_with_library(
                        PoseReader(pose_dir),
                        HERE / "pdbs" / "rna.pdb",
                        fragment=fragment,
                        sequence=sequence,
                        library=library,
                    )
                )
                assert len(chunks) == 1
                actual = chunks[0][0]
                expected = float(row["rmsd"])
                assert np.isclose(actual, expected, atol=5e-6, rtol=0), (
                    f"Fragment {fragment}: expected RMSD {expected:.6f}, got {actual:.6f}"
                )

                if not checked_two_residue_reference:
                    ref_fragment = reference[
                        np.isin(reference["resid"], [fragment, fragment + 1])
                    ]
                    ref_path = tmp_path / f"frag-{fragment}-reference.pdb"
                    ref_path.write_text(write_pdb(ref_fragment))
                    chunks_no_fragment = list(
                        iter_rmsd_chunks_with_library(
                            PoseReader(pose_dir),
                            ref_path,
                            fragment=None,
                            sequence=sequence,
                            library=library,
                            nprocs=2,
                        )
                    )
                    assert len(chunks_no_fragment) == 1
                    actual_no_fragment = chunks_no_fragment[0][0]
                    assert np.isclose(actual_no_fragment, expected, atol=5e-6, rtol=0)
                    checked_two_residue_reference = True
        finally:
            factory.unload_rotaconformers()


def test_npy_output_accepts_unordered_indexed_chunks(tmp_path: Path) -> None:
    output = tmp_path / "rmsd.npy"
    chunks = iter(
        [
            (2, np.array([3.0, 4.0], dtype=np.float32)),
            (0, np.array([1.0, 2.0], dtype=np.float32)),
        ]
    )
    write_npy_output_chunks(output, chunks, total_poses=4)
    assert np.array_equal(
        np.load(output),
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
    )
