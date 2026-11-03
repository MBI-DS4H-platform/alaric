"""W6 A11 coverage and runtime-constant regressions."""
from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alaric import anchor_constants, anchor_refe, grow  # noqa: E402


def test_a11_constants_are_fixed_and_match_grow_recompute_tolerance() -> None:
    assert anchor_constants.L_OV == math.sqrt(3.0)
    assert anchor_constants.L_OV > grow.MAX_OVERLAP_RMSD
    assert anchor_constants.BOUNDARY_RECOMPUTE_TOLERANCE == 1e-5
    assert grow.TRACE_RMSD_BOUNDARY_TOLERANCE == anchor_constants.BOUNDARY_RECOMPUTE_TOLERANCE
    assert anchor_constants.MIN_CELL_FRACTION == 1 / 3


@pytest.mark.parametrize("override", ["0", "0.01", "0.9"])
def test_anchor_action_override_does_not_change_default_or_a9(
    override: str,
) -> None:
    env = dict(os.environ)
    env["ALARIC_ANCHOR_MIN_CELL_FRACTION"] = override
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from alaric.anchor_constants import MIN_CELL_FRACTION as default; "
            "from alaric.anchor import MIN_CELL_FRACTION as action; "
            "from alaric.gaussian_check import MIN_CELL_FRACTION as a9; "
            f"assert default == a9 == 1/3 and action == {float(override)!r}",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_anchor_action_defaults_to_shared_fraction_without_override() -> None:
    env = dict(os.environ)
    env.pop("ALARIC_ANCHOR_MIN_CELL_FRACTION", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from alaric.anchor_constants import MIN_CELL_FRACTION as default; "
            "from alaric.anchor import MIN_CELL_FRACTION as action; "
            "assert default == action == 1/3",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


class _AfterLimitValidation(Exception):
    pass


def test_grow_rejects_ovrmsd_above_sqrt3_before_loading_inputs(tmp_path: Path) -> None:
    source = tmp_path / "empty-source"
    source.mkdir()
    args = grow.build_parser().parse_args(
        [
            "--source-poses",
            str(source),
            "--source-sequence",
            "GU",
            "--target-sequence",
            "UU",
            "--direction",
            "forward",
            "--crmsd",
            "0.5",
            "--ov-rmsd",
            str(anchor_constants.L_OV + 1e-8),
            "--output",
            str(tmp_path / "grow-out"),
        ]
    )
    with pytest.raises(ValueError):
        grow._run(args)


def test_anchor_refe_rejects_ovrmsd_above_sqrt3_before_loading_inputs(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.pdb"
    reference.write_text("HEADER test\n")
    args = anchor_refe.build_parser().parse_args(
        [
            "--reference",
            str(reference),
            "--fragment",
            "1",
            "--first",
            "--ov-rmsd",
            str(anchor_constants.L_OV + 1e-8),
            "--output",
            str(tmp_path / "anchor-refe-out"),
        ]
    )
    with pytest.raises(ValueError):
        anchor_refe._run(args)


def test_both_actions_allow_the_exact_sqrt3_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Get past each upper-bound guard without starting library-backed work."""

    source = tmp_path / "empty-source"
    source.mkdir()
    grow_args = grow.build_parser().parse_args(
        [
            "--source-poses",
            str(source),
            "--source-sequence",
            "GU",
            "--target-sequence",
            "UU",
            "--direction",
            "forward",
            "--crmsd",
            "0.5",
            "--ov-rmsd",
            str(anchor_constants.L_OV),
            "--output",
            str(tmp_path / "grow-boundary"),
        ]
    )

    def stop_grow(*args, **kwargs):
        raise _AfterLimitValidation

    monkeypatch.setattr(grow, "_resolve_growth_layout", stop_grow)
    with pytest.raises(_AfterLimitValidation):
        grow._run(grow_args)

    reference = tmp_path / "reference.pdb"
    reference.write_text("HEADER test\n")
    refe_args = anchor_refe.build_parser().parse_args(
        [
            "--reference",
            str(reference),
            "--fragment",
            "1",
            "--first",
            "--ov-rmsd",
            str(anchor_constants.L_OV),
            "--output",
            str(tmp_path / "refe-boundary"),
        ]
    )

    def stop_refe(*args, **kwargs):
        raise _AfterLimitValidation

    monkeypatch.setattr(anchor_refe, "load_reference_fragment", stop_refe)
    with pytest.raises(_AfterLimitValidation):
        anchor_refe._run(refe_args)
