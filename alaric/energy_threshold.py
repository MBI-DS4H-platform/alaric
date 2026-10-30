"""Standard-library energy threshold conversion for preparic and Alaric."""
from __future__ import annotations

try:
    from .threshold_rounding import strictly_above_float
except ImportError:  # The module is also staged as a flat Seamless attachment.
    from threshold_rounding import strictly_above_float


def energy_threshold(raw: float) -> str:
    """Return the smallest two-decimal value strictly above a raw energy."""
    return strictly_above_float(raw)
