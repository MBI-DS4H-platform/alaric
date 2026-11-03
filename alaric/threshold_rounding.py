"""Shared standard-library rounding for strict Alaric thresholds."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import math


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
    if raw.is_zero():
        return "0.01"

    # Compute floor(raw * 100) + 1 with integers. Decimal division and
    # multiplication use the active context precision, which can round a long
    # decimal that lies just below a cent boundary onto the boundary itself.
    sign, digits, exponent = raw.as_tuple()
    coefficient = int("".join(map(str, digits)))
    shift = exponent + 2
    if shift >= 0:
        numerator = coefficient * (10 ** shift)
        denominator = 1
    elif raw.adjusted() < -2:
        # Here abs(raw * 100) < 1, so its floor is 0 for positive values and
        # -1 for negative values. Avoid constructing a huge power of ten for
        # inputs with very small exponents.
        return "0.01" if sign == 0 else "0.00"
    else:
        numerator = coefficient
        denominator = 10 ** (-shift)
    if sign:
        numerator = -numerator
    cents = numerator // denominator + 1

    whole, fractional = divmod(abs(cents), 100)
    prefix = "-" if cents < 0 else ""
    return f"{prefix}{whole}.{fractional:02d}"


def strictly_above_float(raw: float) -> str:
    """Round a float using its shortest round-trip decimal representation."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise TypeError("raw threshold must be a real number")
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError("raw threshold must be finite")
    return strictly_above_cent(repr(value))
