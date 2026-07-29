from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "alaric"))

from parse_pdb import atomic_dtype  # noqa: E402
from poses import pack_pool, write_arc_file  # noqa: E402
from rmsd import GRID_SPACING  # noqa: E402

from alaric.middle.chain import CHAINS_FILE, CHAINS_METADATA_FILE  # noqa: E402
from alaric.middle.chain_coordinates import (  # noqa: E402
    ChainCoordinatesError,
    Column,
    build_chain_coordinates,
    load_metadata,
    read_chain_table,
    resolve_excluded,
    resolve_sequences,
)
from alaric.middle.errors import MiddleError  # noqa: E402


# -- fake fragment library ------------------------------------------------

NUC_ATOMS = 2
# conformer 0: nucleotide 1 at x=0,1 and nucleotide 2 at x=10,11
FAKE_COORDINATES = np.array(
    [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0], [11.0, 0.0, 0.0]]],
    dtype=np.float32,
)


def _mono_template(base: str) -> np.ndarray:
    template = np.zeros(NUC_ATOMS, dtype=atomic_dtype)
    template["name"] = [b"P", b"C1'"]
    template["resname"] = base.encode()
    template["chain"] = b"B"
    template["altloc"] = b" "
    template["icode"] = b" "
    template["index"] = np.arange(1, NUC_ATOMS + 1)
    template["resid"] = 1
    template["occupancy"] = 1.0
    template["element"] = [b"P", b"C"]
    return template


class FakeLibrary:
    """One-conformer dinucleotide library, two atoms per nucleotide, no rotation."""

    def __init__(self, sequence: str) -> None:
        self.coordinates = FAKE_COORDINATES
        first = _mono_template(sequence[0])
        second = _mono_template(sequence[1])
        second["resid"] = 2
        self.template = np.concatenate((first, second))
        self._rotamers = np.eye(3, dtype=np.float32)[None, :, :]

    def get_rotamers(self, conformer: int) -> np.ndarray:
        assert conformer == 0
        return self._rotamers


@contextmanager
def _fake_library(sequence: str):
    yield FakeLibrary(sequence)


def _templates() -> dict[str, np.ndarray]:
    return {base: _mono_template(base) for base in "ACGU"}


def _expected_nucleotide(nucleotide_index: int, translation) -> np.ndarray:
    """Where the fake library puts nucleotide 1 or 2 of a pose, independently."""
    atoms = FAKE_COORDINATES[0][
        NUC_ATOMS * (nucleotide_index - 1) : NUC_ATOMS * nucleotide_index
    ]
    return atoms + np.asarray(translation, dtype=np.float64) * GRID_SPACING


# -- chain dir fixtures ---------------------------------------------------


def _write_pose_dir(path: Path, poses: list[tuple[int, int, tuple[int, int, int]]]) -> None:
    """Write one organized poses-1.arc; global pose index == list order."""
    path.mkdir(parents=True, exist_ok=True)
    packed = pack_pool(
        np.array([p[0] for p in poses], dtype=np.uint16),
        np.array([p[1] for p in poses], dtype=np.uint16),
        np.array([p[2] for p in poses], dtype=np.int32),
        bucket_size=16,
        sort_offsets=False,
    )
    assert len(packed) == 1, "test poses must fit one bucket"
    M, O, C, P = packed[0]
    write_arc_file(path / "poses-1.arc", M, O, C, P, bucket_size=16)


def _write_chain_dir(
    chain_dir: Path,
    columns: list[tuple[str, int, str | None, list]],
    table: list[list[int]],
    *,
    exclude: list[str] | None = ["1b7f"],
    project: str = "/nonexistent-project",
) -> Path:
    chain_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for pool, fragment, sequence, poses in columns:
        _write_pose_dir(chain_dir / pool, poses)
        entries.append(
            {
                "pool": pool,
                "fragment": fragment,
                "pose_dir": pool,
                "nposes": len(poses),
                "sequence": sequence,
            }
        )
    with (chain_dir / CHAINS_FILE).open("w") as handle:
        handle.write("\t".join(entry["pool"] for entry in entries) + "\n")
        np.savetxt(handle, np.asarray(table, dtype=np.int64), fmt="%d", delimiter="\t")
    (chain_dir / CHAINS_METADATA_FILE).write_text(
        json.dumps(
            {
                "project": project,
                "graph": None,
                "chains_file": CHAINS_FILE,
                "nchains": len(table),
                "exclude": list(exclude) if exclude else None,
                "columns": entries,
            }
        )
    )
    return chain_dir


def _columns(chain_dir: Path, metadata: dict, sequences: list[str]) -> list[Column]:
    return [
        Column(
            pool=entry["pool"],
            fragment=entry["fragment"],
            pose_dir=chain_dir / entry["pose_dir"],
            nposes=entry["nposes"],
            sequence=sequence,
        )
        for entry, sequence in zip(metadata["columns"], sequences)
    ]


# -- averaging the shared nucleotide --------------------------------------


def test_shared_nucleotide_is_averaged(tmp_path):
    # frag1 has two poses, frag2 one; chains: (frag1 pose1, frag2 pose1) and
    # (frag1 pose2, frag2 pose1)
    chain_dir = _write_chain_dir(
        tmp_path / "chains",
        [
            ("frag1-pool", 1, "AA", [(0, 0, (0, 0, 0)), (0, 0, (3, 0, 0))]),
            ("frag2-pool", 2, "AA", [(0, 0, (0, 6, 0))]),
        ],
        [[1, 1], [2, 1]],
    )
    metadata = load_metadata(chain_dir)
    table, start = read_chain_table(chain_dir, metadata, None)
    assert start == 0

    atoms = build_chain_coordinates(
        _columns(chain_dir, metadata, ["AA", "AA"]),
        table,
        open_library=_fake_library,
        templates=_templates(),
    )

    # two dinucleotides -> three nucleotides, numbered by fragment position
    assert atoms.dtype == atomic_dtype
    assert atoms.shape == (2, 3 * NUC_ATOMS)
    assert atoms["resid"][0].tolist() == [1, 1, 2, 2, 3, 3]
    assert atoms["index"][0].tolist() == list(range(1, 3 * NUC_ATOMS + 1))

    # one row per chain, in chain order
    for chain, frag1_translation in enumerate([(0, 0, 0), (3, 0, 0)]):
        frag2_translation = (0, 6, 0)
        expected = np.concatenate(
            (
                _expected_nucleotide(1, frag1_translation),
                # nucleotide 2 is described by both fragments: the mean of the two
                0.5
                * (
                    _expected_nucleotide(2, frag1_translation)
                    + _expected_nucleotide(1, frag2_translation)
                ),
                _expected_nucleotide(2, frag2_translation),
            )
        )
        found = np.stack(
            (atoms["x"][chain], atoms["y"][chain], atoms["z"][chain]), axis=-1
        )
        np.testing.assert_allclose(found, expected, atol=1e-5)


def test_shared_poses_are_transformed_once(tmp_path):
    # both chains use frag5 pose 1: the two rows must still agree on nucleotide 6
    chain_dir = _write_chain_dir(
        tmp_path / "chains",
        [
            ("frag4-pool", 4, "GU", [(0, 0, (0, 0, 0)), (0, 0, (3, 0, 0))]),
            ("frag5-pool", 5, "UU", [(0, 0, (0, 6, 0))]),
        ],
        [[1, 1], [2, 1]],
    )
    metadata = load_metadata(chain_dir)
    table, _ = read_chain_table(chain_dir, metadata, None)
    atoms = build_chain_coordinates(
        _columns(chain_dir, metadata, ["GU", "UU"]),
        table,
        open_library=_fake_library,
        templates=_templates(),
    )
    assert atoms["resid"][0].tolist() == [4, 4, 5, 5, 6, 6]
    assert [bytes(name) for name in atoms["resname"][0]] == [
        b"G",
        b"G",
        b"U",
        b"U",
        b"U",
        b"U",
    ]
    last = slice(2 * NUC_ATOMS, 3 * NUC_ATOMS)
    np.testing.assert_allclose(atoms["x"][0][last], atoms["x"][1][last], atol=1e-6)


def test_pose_index_out_of_range(tmp_path):
    chain_dir = _write_chain_dir(
        tmp_path / "chains",
        [
            ("frag1-pool", 1, "AA", [(0, 0, (0, 0, 0))]),
            ("frag2-pool", 2, "AA", [(0, 0, (0, 6, 0))]),
        ],
        [[1, 2]],
    )
    metadata = load_metadata(chain_dir)
    table, _ = read_chain_table(chain_dir, metadata, None)
    with pytest.raises(ChainCoordinatesError, match="out of range"):
        build_chain_coordinates(
            _columns(chain_dir, metadata, ["AA", "AA"]),
            table,
            open_library=_fake_library,
            templates=_templates(),
        )


# -- metadata / table reading --------------------------------------------


def test_non_contiguous_fragments_rejected(tmp_path):
    chain_dir = _write_chain_dir(
        tmp_path / "chains",
        [
            ("frag1-pool", 1, "AA", [(0, 0, (0, 0, 0))]),
            ("frag3-pool", 3, "AA", [(0, 0, (0, 6, 0))]),
        ],
        [[1, 1]],
    )
    with pytest.raises(ChainCoordinatesError, match="not consecutive"):
        load_metadata(chain_dir)


def test_missing_metadata_names_the_builder(tmp_path):
    (tmp_path / "chains").mkdir()
    with pytest.raises(ChainCoordinatesError, match="alaric-chain"):
        load_metadata(tmp_path / "chains")


def _two_column_dir(tmp_path, nchains: int) -> Path:
    """A chain dir whose second column is 1..nchains, so row order is observable."""
    return _write_chain_dir(
        tmp_path / "chains",
        [
            ("frag1-pool", 1, "AA", [(0, 0, (0, 0, 0))]),
            ("frag2-pool", 2, "AA", [(0, 0, (i, 6, 0)) for i in range(nchains)]),
        ],
        [[1, i + 1] for i in range(nchains)],
    )


def test_chain_range_selects_rows(tmp_path):
    chain_dir = _two_column_dir(tmp_path, 5)
    metadata = load_metadata(chain_dir)
    table, start = read_chain_table(chain_dir, metadata, [2, 4])
    assert start == 1
    assert table.tolist() == [[1, 2], [1, 3], [1, 4]]

    table, start = read_chain_table(chain_dir, metadata, [3, 3])
    assert (start, table.tolist()) == (2, [[1, 3]])

    table, start = read_chain_table(chain_dir, metadata, None)
    assert (start, len(table)) == (0, 5)

    with pytest.raises(ChainCoordinatesError, match="exceeds the number of chains"):
        read_chain_table(chain_dir, metadata, [4, 6])
    with pytest.raises(ChainCoordinatesError, match="START must be <= END"):
        read_chain_table(chain_dir, metadata, [4, 2])


def test_header_mismatch_rejected(tmp_path):
    chain_dir = _two_column_dir(tmp_path, 2)
    metadata = load_metadata(chain_dir)
    (chain_dir / CHAINS_FILE).write_text("other-pool\tfrag2-pool\n1\t1\n")
    with pytest.raises(ChainCoordinatesError, match="out of sync"):
        read_chain_table(chain_dir, metadata, None)


# -- sequence / exclude resolution ---------------------------------------


def _metadata(sequences: list[str | None], exclude=None) -> dict:
    return {
        "nchains": 1,
        "exclude": exclude,
        "columns": [
            {"pool": f"frag{i}-pool", "fragment": i, "sequence": seq}
            for i, seq in enumerate(sequences, start=1)
        ],
    }


def test_sequences_from_metadata_and_override():
    metadata = _metadata(["GU", "UU"])
    assert resolve_sequences(metadata, sequence=None, project=None) == ["GU", "UU"]
    # --sequence is the whole chain: one nucleotide more than the fragment count
    assert resolve_sequences(metadata, sequence="acg", project=None) == ["AC", "CG"]

    with pytest.raises(ChainCoordinatesError, match="whole chain sequence"):
        resolve_sequences(metadata, sequence="AC", project=None)
    with pytest.raises(ChainCoordinatesError, match="not a nucleotide base"):
        resolve_sequences(metadata, sequence="ACX", project=None)


def test_overlapping_sequences_must_agree():
    with pytest.raises(ChainCoordinatesError, match="shared nucleotide 2"):
        resolve_sequences(_metadata(["GU", "AU"]), sequence=None, project=None)


def test_missing_sequence_asks_for_override():
    with pytest.raises(ChainCoordinatesError, match="--sequence"):
        resolve_sequences(_metadata(["GU", None]), sequence=None, project=None)


def test_exclude_resolution():
    metadata = _metadata(["GU", "UU"], exclude=["1B7F"])
    assert resolve_excluded(
        metadata, exclude=None, no_exclude=False, project=None
    ) == ["1b7f"]
    assert resolve_excluded(
        metadata, exclude=["3SXL"], no_exclude=False, project=None
    ) == ["3sxl"]
    assert (
        resolve_excluded(metadata, exclude=None, no_exclude=True, project=None) is None
    )
    # unresolved exclusion would silently change the conformer coordinates
    with pytest.raises(ChainCoordinatesError, match="--exclude"):
        resolve_excluded(
            _metadata(["GU", "UU"]), exclude=None, no_exclude=False, project=None
        )


def test_chain_coordinates_error_is_a_middle_error():
    assert issubclass(ChainCoordinatesError, MiddleError)
