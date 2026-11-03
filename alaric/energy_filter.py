"""Grid-assisted, gridless-correct energy filtering and threshold-file parsing.

The cascade is only an accelerator: every pose left in its uncertainty band is
scored by the exact gridless kernel before the result is returned.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import yaml


DEFAULT_SCORE_FILTER = {
    "min_cascade": 1000,
    "cascade": [
        {"plateau_distance": 5, "neighbour_distance": 7, "max_error": 1.0},
        {"plateau_distance": 10, "neighbour_distance": 12, "max_error": 0.05},
    ],
}
_STEP_KEYS = {"plateau_distance", "neighbour_distance", "max_error"}
_FRAGMENT_KEY = re.compile(r"frag-(\d+)-([ACGU]{2})\Z")
_RANGE_KEY = re.compile(r"range-\d+\Z")
_TWO_DECIMALS = re.compile(r"-?(?:0|[1-9]\d*)\.\d{2}\Z")


def _threshold_mapping(path: str | Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read energy-thresholds file {path}: {exc}") from exc
    if not isinstance(data, dict) or type(data.get("schema")) is not int or data["schema"] != 1:
        raise ValueError(f"{path}: energy-thresholds must have schema: 1")
    for fragment_key, ranges in data.items():
        if fragment_key == "schema":
            continue
        if not isinstance(fragment_key, str) or _FRAGMENT_KEY.fullmatch(fragment_key) is None:
            raise ValueError(f"{path}: invalid fragment key {fragment_key!r}")
        if not isinstance(ranges, dict) or not ranges:
            raise ValueError(f"{path}: {fragment_key} must map range-K keys to thresholds")
        for range_key, threshold in ranges.items():
            if not isinstance(range_key, str) or _RANGE_KEY.fullmatch(range_key) is None:
                raise ValueError(f"{path}: invalid protein range key {range_key!r}")
            if not isinstance(threshold, str) or _TWO_DECIMALS.fullmatch(threshold) is None:
                raise ValueError(
                    f"{path}: {fragment_key}/{range_key} must be a two-decimal string"
                )
            if not math.isfinite(float(threshold)):
                raise ValueError(f"{path}: {fragment_key}/{range_key} is not finite")
    return data


def load_threshold(
    path: str | Path, fragment: int, sequence: str, protein: str
) -> float:
    """Load one strict P5.7 energy threshold from the export file."""
    data = _threshold_mapping(path)
    fragment_key = f"frag-{int(fragment)}-{str(sequence).upper()}"
    if _FRAGMENT_KEY.fullmatch(fragment_key) is None:
        raise ValueError(f"invalid fragment/sequence pair: {fragment_key}")
    ranges = data.get(fragment_key)
    if not isinstance(ranges, dict) or protein not in ranges:
        raise KeyError(f"{path}: missing energy threshold for {fragment_key}/{protein}")
    return float(ranges[protein])


def _validate_steps(value: Any) -> list[dict[str, float | int]]:
    if not isinstance(value, list):
        raise ValueError("cascade must be a list of grid steps")
    steps: list[dict[str, float | int]] = []
    for index, step in enumerate(value):
        if not isinstance(step, dict) or set(step) != _STEP_KEYS:
            raise ValueError(
                f"cascade step {index} must have exactly plateau_distance, "
                "neighbour_distance and max_error"
            )
        normalized: dict[str, float | int] = {}
        for key in ("plateau_distance", "neighbour_distance", "max_error"):
            number = step[key]
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                raise ValueError(f"cascade step {index} {key} must be numeric")
            number = float(number)
            if not math.isfinite(number):
                raise ValueError(f"cascade step {index} {key} must be finite")
            normalized[key] = number
        if normalized["plateau_distance"] <= 0 or normalized["neighbour_distance"] <= 0:
            raise ValueError(f"cascade step {index} grid distances must be positive")
        if normalized["neighbour_distance"] < normalized["plateau_distance"]:
            raise ValueError(
                f"cascade step {index} neighbour_distance must be at least plateau_distance"
            )
        if normalized["max_error"] < 0:
            raise ValueError(f"cascade step {index} max_error must be non-negative")
        steps.append(normalized)
    return steps


def _validate_minimum(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("min_cascade must be a non-negative integer")
    return value


def _read_defaults(path: str | Path) -> tuple[list[dict[str, float | int]], int]:
    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read score-filter defaults {path}: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"cascade", "min_cascade"}:
        raise ValueError(f"{path}: expected exactly cascade and min_cascade")
    return _validate_steps(payload["cascade"]), _validate_minimum(payload["min_cascade"])


def resolve_config(
    cascade: Any,
    min_cascade: Any,
    defaults_path: str | Path,
) -> tuple[list[dict[str, float | int]], int]:
    """Resolve explicit values and ``auto`` values from score-filter.yaml."""
    if cascade == "auto" or min_cascade == "auto":
        default_steps, default_minimum = _read_defaults(defaults_path)
    else:
        default_steps, default_minimum = [], 0
    steps = default_steps if cascade == "auto" else _validate_steps(cascade)
    minimum = default_minimum if min_cascade == "auto" else _validate_minimum(min_cascade)
    return steps, minimum


def _score_vector(values: Any, expected: int, label: str) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or len(scores) != expected:
        raise ValueError(f"{label} returned {scores.shape}, expected ({expected},)")
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"{label} returned non-finite energies")
    return scores


def grid_precision_unsafe(scores: Any, threshold: float, max_error: float) -> np.ndarray:
    """Mark grid scores whose float64 spacing reaches the allowed error.

    The empirical acceptance gate is strictly below half of ``max_error``.
    If float64 spacing at either the score or decision threshold reaches that
    half-error margin, send the pose to the exact scorer instead of classifying
    it from the grid value.
    """
    values = np.asarray(scores, dtype=np.float64)
    threshold = float(threshold)
    max_error = float(max_error)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("grid scores must be a finite one-dimensional vector")
    if not math.isfinite(threshold):
        raise ValueError("energy threshold must be finite")
    if not math.isfinite(max_error) or max_error < 0:
        raise ValueError("max_error must be finite and non-negative")
    score_ulp = np.spacing(np.abs(values))
    threshold_ulp = abs(float(np.spacing(np.float64(abs(threshold)))))
    precision_limit = max_error / 2.0
    return (score_ulp >= precision_limit) | (threshold_ulp >= precision_limit)


def cascade_filter(
    pose_ids: Iterable[int] | np.ndarray,
    *,
    threshold: float,
    cascade: Any,
    min_cascade: Any,
    grid_score: Callable[[np.ndarray, dict[str, Any]], Any],
    gridless_score: Callable[[np.ndarray], Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return passing global pose IDs in their input order and route statistics.

    ``grid_score`` is called with all currently grid-eligible IDs for one
    complete action step. The callback may internally split that work into
    chunks; the minimum-cascade decision stays here, over all unresolved poses,
    including poses already routed to exact scoring for precision safety.
    """
    threshold = float(threshold)
    if not math.isfinite(threshold):
        raise ValueError("energy threshold must be finite")
    ids = np.asarray(list(pose_ids) if not isinstance(pose_ids, np.ndarray) else pose_ids)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("pose_ids must be a one-dimensional integer array")
    ids = ids.astype(np.int64, copy=False)
    steps = _validate_steps(cascade)
    minimum = _validate_minimum(min_cascade)

    undecided = np.arange(len(ids), dtype=np.int64)
    kept = np.zeros(len(ids), dtype=bool)
    stats: dict[str, Any] = {
        "grid_counts": [],
        "gridless_count": 0,
        "precision_fallback_counts": [],
        "precision_fallback_count": 0,
        "cascade_exit": None,
    }
    precision_fallback = np.zeros(len(ids), dtype=bool)
    for step_index, step in enumerate(steps):
        if len(undecided) == 0:
            stats["cascade_exit"] = {
                "before_step": step_index,
                "undecided": int(np.count_nonzero(precision_fallback)),
                "min_cascade": minimum,
            }
            break
        unresolved_count = len(undecided) + int(np.count_nonzero(precision_fallback))
        if unresolved_count < minimum:
            stats["cascade_exit"] = {
                "before_step": step_index,
                "undecided": int(len(undecided) + np.count_nonzero(precision_fallback)),
                "min_cascade": minimum,
            }
            break
        stats["grid_counts"].append(int(len(undecided)))
        scores = _score_vector(grid_score(ids[undecided], step), len(undecided), "grid scorer")
        error = float(step["max_error"])
        unsafe = grid_precision_unsafe(scores, threshold, error)
        unsafe_count = int(np.count_nonzero(unsafe))
        stats["precision_fallback_counts"].append(unsafe_count)
        stats["precision_fallback_count"] += unsafe_count
        precision_fallback[undecided[unsafe]] = True
        safe_positions = undecided[~unsafe]
        safe_scores = scores[~unsafe]
        accept = safe_scores < threshold - error
        reject = safe_scores >= threshold + error
        kept[safe_positions[accept]] = True
        undecided = safe_positions[~(accept | reject)]
        unresolved_count = len(undecided) + int(np.count_nonzero(precision_fallback))
        if unresolved_count < minimum:
            stats["cascade_exit"] = {
                "after_step": step_index,
                "undecided": int(len(undecided) + np.count_nonzero(precision_fallback)),
                "min_cascade": minimum,
            }
            break

    exact_mask = precision_fallback.copy()
    exact_mask[undecided] = True
    exact_positions = np.flatnonzero(exact_mask)
    stats["gridless_count"] = int(len(exact_positions))
    if len(exact_positions):
        exact = _score_vector(
            gridless_score(ids[exact_positions]), len(exact_positions), "gridless scorer"
        )
        kept[exact_positions] = exact < threshold
    return ids[kept], stats

