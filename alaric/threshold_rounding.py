"""Shared standard-library rounding for strict Alaric thresholds."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_FLOOR
import math


_CENT = Decimal("0.01")


def strictly_above_cent(raw_decimal_text: str) -> str:
    """Return the smallest two-decimal multiple strictly above ``raw``."""
    if not isinstance(raw_decimal_text, str) or not raw_decimal_text.strip():
        raise ValueError("raw threshold must be non-empty decimal text")
    try:
        raw = Decimal(raw_decimal_text.strip())
    except InvalidOperation as exc:
        raise ValueError(f"invalid decimal threshold {raw_decimal_text!r}") from exc
    if not raw.is_finite():
        raise ValueError("raw threshold must be finite")
    steps = (raw / _CENT).to_integral_value(rounding=ROUND_FLOOR) + 1
    return format(steps * _CENT, ".2f")


def strictly_above_float(raw: float) -> str:
    """Round a float using its shortest round-trip decimal representation."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise TypeError("raw threshold must be a real number")
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError("raw threshold must be finite")
    return strictly_above_cent(repr(value))
