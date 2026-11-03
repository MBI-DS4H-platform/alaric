"""The standalone anchor CLI must always resolve a committed A7 filter."""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alaric import anchor  # noqa: E402


def test_anchor_parser_requires_filter_name(tmp_path: Path, capsys) -> None:
    protein = tmp_path / "protein.pdb"
    reference = tmp_path / "reference.npy"
    protein.write_text("HEADER test\n")
    reference.write_bytes(b"placeholder")

    with pytest.raises(SystemExit) as excinfo:
        anchor.build_parser().parse_args(
            [
                "--protein",
                str(protein),
                "--resid",
                "1",
                "--sequence",
                "AA",
                "--first",
                "--output",
                str(tmp_path / "out"),
                "--precalculated-anchor",
                str(tmp_path / "precalc"),
                "--anchor-reference-ring",
                str(reference),
            ]
        )
    assert excinfo.value.code == 2
    assert "--precalculated-filter-name" in capsys.readouterr().err


def test_anchor_runtime_guard_rejects_missing_filter_name(tmp_path: Path) -> None:
    args = SimpleNamespace(
        cache_size=1,
        bucket_size=16,
        nprocs=1,
        poselock=None,
        precalculated_anchor=str(tmp_path / "precalc"),
        output=str(tmp_path / "out"),
    )
    with pytest.raises(ValueError, match="precalculated-filter-name is required"):
        anchor._run(args)


def test_anchor_runtime_rejects_filter_not_in_a7_before_inputs(tmp_path: Path) -> None:
    protein = tmp_path / "protein.pdb"
    reference = tmp_path / "reference.npy"
    protein.write_text("HEADER test\n")
    reference.write_bytes(b"placeholder")
    args = anchor.build_parser().parse_args(
        [
            "--protein",
            str(protein),
            "--resid",
            "1",
            "--sequence",
            "AA",
            "--first",
            "--output",
            str(tmp_path / "out"),
            "--precalculated-anchor",
            str(tmp_path / "precalc"),
            "--precalculated-filter-name",
            "r90",
            "--anchor-reference-ring",
            str(reference),
        ]
    )
    with pytest.raises(ValueError, match="unsupported Gaussian filter name"):
        anchor._run(args)
