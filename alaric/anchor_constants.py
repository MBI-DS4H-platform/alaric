"""Numerical limits shared by Alaric's anchor-related actions."""

from __future__ import annotations

import math


L_OV = math.sqrt(3.0)
BOUNDARY_RECOMPUTE_TOLERANCE = 1e-5
MIN_CELL_FRACTION = 1 / 3


def validate_ovrmsd(value: float) -> float:
    """Return a valid overlap threshold or reject unsupported coverage."""
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("ovRMSD threshold must be finite and positive")
    if value > L_OV:
        raise ValueError(
            f"ovRMSD threshold {value:g} Å exceeds the supported coverage limit "
            f"L_OV={L_OV:.10g} Å"
        )
    return value
