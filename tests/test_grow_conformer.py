from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alaric.grow import _select_single_target_conformer


def test_select_single_target_conformer_is_one_based_and_keeps_crmsd_sources() -> None:
    target_to_sources = {2: np.array([4, 9, 12], dtype=np.int64)}
    targets, target_to_sources = _select_single_target_conformer(
        np.array([2], dtype=np.int64),
        target_to_sources,
        3,
    )

    np.testing.assert_array_equal(targets, np.array([2], dtype=np.int64))
    np.testing.assert_array_equal(
        target_to_sources[2],
        np.array([4, 9, 12], dtype=np.int64),
    )


def test_select_single_target_conformer_validates_availability() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _select_single_target_conformer(
            np.array([0], dtype=np.int64),
            {0: np.array([0])},
            0,
        )
    with pytest.raises(ValueError, match="not available"):
        _select_single_target_conformer(
            np.array([0], dtype=np.int64),
            {0: np.array([0])},
            2,
        )
