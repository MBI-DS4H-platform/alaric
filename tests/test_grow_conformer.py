from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alaric.grow import _select_single_target_conformer


class _TargetLibrary:
    def __init__(self, nconformers: int, conformer_mask: np.ndarray | None = None):
        self.coordinates = np.zeros((nconformers, 1, 3), dtype=np.float32)
        self.conformer_mask = conformer_mask


def test_select_single_target_conformer_is_one_based_and_keeps_all_sources() -> None:
    source_conformers = np.array([4, 9, 12], dtype=np.int64)
    targets, target_to_sources = _select_single_target_conformer(
        _TargetLibrary(5),
        source_conformers,
        None,
        3,
    )

    np.testing.assert_array_equal(targets, np.array([2], dtype=np.int64))
    np.testing.assert_array_equal(target_to_sources[2], source_conformers)


def test_select_single_target_conformer_validates_availability() -> None:
    mask = np.array([True, False, True], dtype=bool)
    source_conformers = np.array([0], dtype=np.int64)

    with pytest.raises(ValueError, match="must be positive"):
        _select_single_target_conformer(_TargetLibrary(3), source_conformers, None, 0)
    with pytest.raises(ValueError, match="out of range"):
        _select_single_target_conformer(_TargetLibrary(3), source_conformers, None, 4)
    with pytest.raises(ValueError, match="not available after exclusions"):
        _select_single_target_conformer(_TargetLibrary(3, mask), source_conformers, None, 2)
    with pytest.raises(ValueError, match="not available in --restrict-poses"):
        _select_single_target_conformer(_TargetLibrary(3), source_conformers, {0: {}}, 2)
