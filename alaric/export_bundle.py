"""Internal, snapshot-bound bundle for the deferred project export seam.

This module deliberately prepares values and readiness evidence only.  The
browser export page and the on-disk project layout remain deferred.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from typing import Any

try:
    from .anchor_constants import L_OV, MIN_CELL_FRACTION
    from .energy_filter import DEFAULT_SCORE_FILTER
except ImportError:  # pragma: no cover - direct flat staging
    from anchor_constants import L_OV, MIN_CELL_FRACTION
    from energy_filter import DEFAULT_SCORE_FILTER


_FRAG_KEY = re.compile(r"frag-(\d+)-([ACGU]{2})\Z")
_RANGE_KEY = re.compile(r"range-\d+\Z")
_TWO_DECIMALS = re.compile(r"-?(?:0|[1-9]\d*)\.\d{2}\Z")
_FILTER_NAMES = {
    "rrm50", "rrm60", "rrm70", "rrm80", "rrm90", "rrm95",
    "rrm90-tilt30", "rrm90-disp45", "rrm90-tilt30-disp45",
    "rrm95-tilt30", "rrm95-disp45", "rrm95-tilt30-disp45",
}


def _canonical(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"__bytes_sha256__": hashlib.sha256(value).hexdigest(), "length": len(value)}
    if isinstance(value, bytearray):
        return _canonical(bytes(value))
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _canonical(value.item())
        except (ValueError, TypeError):
            pass
    if hasattr(value, "tolist"):
        try:
            return _canonical(value.tolist())
        except (ValueError, TypeError):
            pass
    return value


def _snapshot_digest(snapshot: dict[str, Any]) -> str:
    data = json.dumps(_canonical(snapshot), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _exclude_code(value: Any, label: str) -> str:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"{label} must contain exactly one PDB code")
        value = value[0]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be one non-empty PDB code")
    return value.strip().lower()


def _kept_fragment_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = {
            str(key)
            for key, present in value.items()
            if present is not False
            and not (
                isinstance(present, dict)
                and (present.get("clipped") is True or present.get("unclipped") is False)
            )
        }
    elif isinstance(value, (list, tuple, set)):
        keys = set()
        for entry in value:
            if isinstance(entry, str):
                keys.add(entry)
            elif isinstance(entry, dict):
                if entry.get("clipped") is True or entry.get("unclipped") is False:
                    continue
                keys.add(f"frag-{int(entry['fragment'])}-{str(entry['sequence']).upper()}")
            else:
                raise ValueError("kept_fragments entries must be keys or fragment mappings")
    else:
        raise ValueError("kept_fragments must be a mapping or list")
    invalid = sorted(key for key in keys if _FRAG_KEY.fullmatch(key) is None)
    if invalid:
        raise ValueError(f"invalid kept fragment key: {invalid[0]}")
    return keys


def _decision_hard_blocks(validations: Any) -> list[str]:
    blocks: list[str] = []
    if not isinstance(validations, dict):
        return ["validation evidence is missing or invalid"]
    results = validations.get("results")
    if not isinstance(results, dict):
        results = {}
        blocks.append("validation results are missing or invalid")
    for name in ("fit", "energetics", "gaussian", "thresholds"):
        if name not in results:
            blocks.append(f"result {name} is missing")
    for name, raw_status in results.items():
        if isinstance(raw_status, dict):
            status = raw_status.get("status")
        else:
            status = raw_status
        if status != "fresh":
            blocks.append(f"result {name} is {status or 'missing'}")

    decisions = validations.get("decisions")
    if not isinstance(decisions, dict):
        decisions = {}
        blocks.append("validation decisions are missing or invalid")
    if "thresholds" not in decisions:
        blocks.append("decision thresholds are missing")
    for name, decision in decisions.items():
        valid = True
        value = decision
        status = None
        if isinstance(decision, dict):
            valid = decision.get("valid") is True
            status = decision.get("status")
            value = decision.get("value")
        elif isinstance(decision, bool):
            valid, value = decision, decision
        if value is None or not valid or status in {"undecided", "invalid"}:
            blocks.append(f"decision {name} is {status or 'undecided or invalid'}")
    thresholds = decisions.get("thresholds")
    if isinstance(thresholds, dict) and not (
        thresholds.get("valid") is True and thresholds.get("value") is not None
    ):
        # The normal decision loop records invalid decisions too; this explicit
        # condition also rejects an empty thresholds mapping.
        if "decision thresholds is undecided or invalid" not in blocks:
            blocks.append("decision thresholds is undecided or invalid")
    return blocks


def _finite_threshold(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return number


def _energy_thresholds(
    kept_fragments: set[str], matrix: Any, edits: Any
) -> dict[str, dict[str, str]]:
    if not isinstance(matrix, dict) or not isinstance(edits, dict):
        raise ValueError("energy_matrix and edited_energy_thresholds must be mappings")
    output: dict[str, dict[str, str]] = {"schema": 1}
    for fragment_key in sorted(kept_fragments):
        cells = matrix.get(fragment_key, {})
        values = edits.get(fragment_key, {})
        if not isinstance(cells, dict) or not isinstance(values, dict):
            raise ValueError(f"{fragment_key}: energy matrix rows must be mappings")
        true_cells: dict[str, str] = {}
        for protein, is_true in sorted(cells.items()):
            if _RANGE_KEY.fullmatch(str(protein)) is None or type(is_true) is not bool:
                raise ValueError(f"{fragment_key}: invalid energy matrix cell {protein!r}")
            if not is_true:
                continue
            threshold = values.get(protein)
            if not isinstance(threshold, str) or _TWO_DECIMALS.fullmatch(threshold) is None:
                raise ValueError(
                    f"{fragment_key}/{protein}: true kept cell needs a two-decimal string"
                )
            if not math.isfinite(float(threshold)):
                raise ValueError(f"{fragment_key}/{protein}: threshold must be finite")
            true_cells[str(protein)] = threshold
        if true_cells:
            output[fragment_key] = true_cells
    return output


def build_export_bundle(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Build a validated immutable-by-copy bundle from one decision snapshot.

    Required values are named in the W8 contract and are deliberately supplied
    as one mapping, so this seam can be exercised without an export page.
    """
    if not isinstance(snapshot, dict):
        raise ValueError("snapshot must be a mapping")
    frozen = copy.deepcopy(snapshot)

    exclude = _exclude_code(frozen.get("exclude_code"), "exclude_code")
    prepared_exclude = _exclude_code(frozen.get("prepared_exclude_code"), "prepared_exclude_code")
    if exclude != prepared_exclude:
        raise ValueError("X1: export exclude code must match the prepared exclude code")
    try:
        fraction = float(frozen.get("min_cell_fraction"))
        prepared_fraction = float(frozen.get("prepared_min_cell_fraction"))
    except (TypeError, ValueError) as exc:
        raise ValueError("X1: export minimum cell fraction must match the canonical default") from exc
    if (
        not math.isfinite(fraction)
        or not math.isfinite(prepared_fraction)
        or fraction != prepared_fraction
        or fraction != MIN_CELL_FRACTION
        or prepared_fraction != MIN_CELL_FRACTION
    ):
        raise ValueError("X1: export minimum cell fraction must match the canonical default")

    reference_bytes = frozen.get("prepared_reference")
    if isinstance(reference_bytes, bytearray):
        reference_bytes = bytes(reference_bytes)
    if not isinstance(reference_bytes, bytes):
        raise ValueError("X2: prepared_reference must be the exact RNA bytes")
    reference_checksum = hashlib.sha256(reference_bytes).hexdigest()
    expected_checksum = frozen.get("prepared_sha256")
    if expected_checksum is not None and expected_checksum != reference_checksum:
        raise ValueError("X2: prepared RNA checksum does not match its bytes")

    kept = _kept_fragment_keys(frozen.get("kept_fragments"))
    kept_sequences_by_fragment: dict[int, set[str]] = {}
    for key in kept:
        match = _FRAG_KEY.fullmatch(key)
        assert match is not None
        kept_sequences_by_fragment.setdefault(int(match.group(1)), set()).add(match.group(2))
    anchor_actions: list[dict[str, Any]] = []
    selected_anchors = frozen.get("selected_anchors")
    hard_blocks = _decision_hard_blocks(frozen.get("validations"))
    if not isinstance(selected_anchors, list):
        selected_anchors = []
        hard_blocks.append("selected anchor evidence is missing or invalid")
    elif not selected_anchors:
        hard_blocks.append("selected anchor evidence is empty")
    required_anchor_candidates: set[tuple[int, str, str]] = set()
    for anchor in selected_anchors:
        if not isinstance(anchor, dict):
            raise ValueError("selected anchor must be a mapping")
        fragment = int(anchor["fragment"])
        sequence = str(anchor["sequence"]).upper()
        filter_name = str(anchor.get("filter_name", anchor.get("edited_filter_name", "")))
        if filter_name not in _FILTER_NAMES:
            raise ValueError(f"unsupported Gaussian filter name {filter_name!r}")
        nucleotide = str(anchor["nucleotide"]).lower()
        if nucleotide not in {"first", "second"}:
            raise ValueError("anchor nucleotide must be first or second")
        residue_ordinal = fragment + (1 if nucleotide == "second" else 0)
        candidates = sorted(
            (candidate_fragment, candidate_nucleotide, candidate_sequence)
            for candidate_fragment, candidate_nucleotide in (
                (residue_ordinal - 1, "second"), (residue_ordinal, "first")
            )
            for candidate_sequence in kept_sequences_by_fragment.get(candidate_fragment, ())
        )
        if not candidates:
            hard_blocks.append(
                f"selected anchor at RNA residue {residue_ordinal} has no kept candidate fragment"
            )
            continue
        required_anchor_candidates.update(candidates)
        action_fragment, action_nucleotide, action_sequence = next(
            (candidate for candidate in candidates
             if candidate == (fragment, nucleotide, sequence)),
            candidates[0],
        )
        anchor_actions.append(
            {
                "action": "anchor",
                "fragment": action_fragment,
                "sequence": action_sequence,
                "exclude": "auto",
                "protein": str(anchor["protein"]),
                "resid": int(anchor["resid"]),
                "nucleotide": action_nucleotide,
                "precomputed_filter_name": "gaussian-rrm",
                "precomputed_filter_threshold": filter_name,
            }
        )

    reference_candidates = frozen.get("reference_candidates")
    if not isinstance(reference_candidates, list):
        reference_candidates = []
        hard_blocks.append("reference candidate evidence is missing or invalid")
    elif not reference_candidates:
        hard_blocks.append("reference candidate evidence is empty")
    refe_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    computed_by_key: dict[tuple[int, str], tuple[float, float, str]] = {}
    comment_by_key: dict[tuple[int, str], str] = {}
    available_anchor_candidates: set[tuple[int, str, str]] = set()
    for candidate in reference_candidates:
        if not isinstance(candidate, dict):
            raise ValueError("reference candidate must be a mapping")
        if candidate.get("clipped") is True or candidate.get("unclipped") is False:
            continue
        fragment = int(candidate["fragment"])
        nucleotide = str(candidate["nucleotide"]).lower()
        if nucleotide not in {"first", "second"}:
            raise ValueError("reference candidate nucleotide must be first or second")
        requested_sequence = str(candidate.get("sequence", "")).upper()
        matching_keys = sorted(
            key
            for key in kept
            if (match := _FRAG_KEY.fullmatch(key)) is not None
            and match.group(1) == str(fragment)
            and (not requested_sequence or match.group(2) == requested_sequence)
        )
        fragment_key = matching_keys[0] if matching_keys else None
        if kept and fragment_key is None:
            continue
        sequence = str(candidate.get("sequence") or (_FRAG_KEY.fullmatch(fragment_key).group(2) if fragment_key else "")).upper()
        if not sequence:
            raise ValueError(f"fragment {fragment} sequence is required for anchor-refe")
        threshold = candidate.get("ovrmsd")
        raw_threshold = candidate.get("ovrmsd_raw")
        if isinstance(threshold, bool) or isinstance(raw_threshold, bool):
            raise ValueError(f"fragment {fragment} {nucleotide} computed ovRMSD evidence is malformed")
        threshold_number = _finite_threshold(threshold, f"fragment {fragment} {nucleotide} ovrmsd")
        try:
            raw_threshold_number = float(raw_threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"fragment {fragment} {nucleotide} raw ovrmsd must be numeric") from exc
        if not math.isfinite(raw_threshold_number) or raw_threshold_number < 0:
            raise ValueError(f"fragment {fragment} {nucleotide} raw ovrmsd must be finite and nonnegative")
        if threshold_number <= raw_threshold_number:
            raise ValueError(f"fragment {fragment} {nucleotide} rounded ovrmsd must exceed its raw value")
        key = (fragment, nucleotide)
        evidence = (threshold_number, raw_threshold_number, sequence)
        previous_evidence = computed_by_key.get(key)
        if previous_evidence is not None and previous_evidence != evidence:
            hard_blocks.append(
                f"conflicting computed anchor-refe evidence for fragment {fragment}/{nucleotide}"
            )
        else:
            computed_by_key[key] = evidence
        exceeds_limit = threshold_number > L_OV
        emitted_threshold: str | int = -1 if exceeds_limit else (
            str(threshold) if isinstance(threshold, str) else str(threshold)
        )
        action = {
            "action": "anchor-refe",
            "fragment": fragment,
            "sequence": sequence,
            "exclude": "auto",
            "nucleotide": nucleotide,
            "ovrmsd": emitted_threshold,
            "reference": "reference.pdb",
        }
        if exceeds_limit:
            comment = (
                f"Computed base ovRMSD raw={raw_threshold}, rounded={threshold} Å; "
                f"L_ov={L_OV:.17g} Å. The required rounded positive-control threshold exceeds "
                "the ovRMSD Alaric can search without misses, so this action cannot work."
            )
            if raw_threshold_number > L_OV:
                comment += " The best-fit pose itself also lies beyond the coverage limit."
            comment += " This action is kept on purpose as a disabled positive control."
            comment_by_key[key] = comment
        available_anchor_candidates.add((fragment, nucleotide, sequence))
        previous = refe_by_key.get(key)
        if previous is not None and previous["sequence"] != action["sequence"]:
            hard_blocks.append(
                f"conflicting anchor-refe sequences for fragment {fragment}/{nucleotide}"
            )
        else:
            refe_by_key.setdefault(key, action)
    anchor_refe_actions = [refe_by_key[key] for key in sorted(refe_by_key)]
    anchor_refe_comments = [
        {"fragment": key[0], "nucleotide": key[1], "comment": comment_by_key[key]}
        for key in sorted(comment_by_key)
    ]
    for fragment, nucleotide, sequence in sorted(required_anchor_candidates):
        if (fragment, nucleotide, sequence) not in available_anchor_candidates:
            hard_blocks.append(
                f"selected anchor candidate fragment {fragment} {nucleotide} {sequence} "
                "has no unclipped reference candidate"
            )

    handoff = frozen.get("handoff")
    if (
        not isinstance(handoff, dict)
        or not isinstance(handoff.get("fragment_rows"), list)
        or not handoff["fragment_rows"]
    ):
        hard_blocks.append("handoff fragment evidence is missing or empty")
    class_table = frozen.get("class_table")
    if (
        not isinstance(class_table, dict)
        or not isinstance(class_table.get("rows"), list)
        or not class_table["rows"]
    ):
        hard_blocks.append("class evidence is missing or empty")

    energy_matrix = frozen.get("energy_matrix")
    if not isinstance(energy_matrix, dict):
        hard_blocks.append("energy matrix evidence is missing or invalid")
        energy_matrix = {}
    energy_threshold_file = _energy_thresholds(
        kept,
        energy_matrix,
        frozen.get("edited_energy_thresholds", {}),
    )
    energy_filter_actions: list[dict[str, Any]] = []
    for fragment_key in sorted(energy_threshold_file):
        if fragment_key == "schema":
            continue
        match = _FRAG_KEY.fullmatch(fragment_key)
        assert match is not None
        prior_action = f"frag-{match.group(1)}-bestfit"
        for protein in sorted(energy_threshold_file[fragment_key]):
            action_name = f"energy-filter-frag{match.group(1)}-{protein}"
            energy_filter_actions.append(
                {
                    "name": action_name,
                    "action": "energy-filter",
                    "input": prior_action,
                    "fragment": "auto",
                    "sequence": "auto",
                    "exclude": "auto",
                    "protein": protein,
                    "threshold": "auto",
                    "epsilon": 15,
                    "cascade": "auto",
                    "min_cascade": "auto",
                }
            )
            prior_action = action_name

    kept_pairs = frozen.get("kept_pairs")
    if not isinstance(kept_pairs, list):
        hard_blocks.append("kept pair evidence is missing or invalid")
        kept_pairs = []
    kept_fragment_numbers = set(kept_sequences_by_fragment)
    required_kept_pair_addresses = {
        (fragment, fragment + 1)
        for fragment in kept_fragment_numbers
        if fragment + 1 in kept_fragment_numbers
    }
    observed_kept_pair_addresses: set[tuple[int, int]] = set()
    grow_thresholds = []
    for pair in kept_pairs:
        if not isinstance(pair, dict):
            raise ValueError("kept pair must be a mapping")
        pair_kept = pair.get("kept", True)
        if type(pair_kept) is not bool:
            hard_blocks.append("kept pair evidence has an invalid kept flag")
            continue
        if not pair_kept:
            continue

        down = pair.get("down")
        up = pair.get("up")
        valid_address = (
            type(down) is int and type(up) is int and down > 0 and up == down + 1
        )
        if not valid_address:
            hard_blocks.append("kept pair has invalid adjacent fragment addressing")
        elif (down, up) in required_kept_pair_addresses:
            observed_kept_pair_addresses.add((down, up))
        down_sequences = kept_sequences_by_fragment.get(down, set()) if valid_address else set()
        up_sequences = kept_sequences_by_fragment.get(up, set()) if valid_address else set()
        valid_identity = len(down_sequences) == 1 and len(up_sequences) == 1
        if valid_address and not valid_identity:
            hard_blocks.append("kept pair fragments are missing or ambiguous in kept fragments")

        sequence = pair.get("sequence")
        valid_sequence = True
        if "sequence" in pair:
            if not isinstance(sequence, str) or re.fullmatch(r"[ACGU]{3}", sequence) is None:
                hard_blocks.append("kept pair has an invalid sequence")
                valid_sequence = False
            elif valid_address and valid_identity:
                down_sequence = next(iter(down_sequences))
                up_sequence = next(iter(up_sequences))
                if (
                    down_sequence[1] != up_sequence[0]
                    or sequence != down_sequence + up_sequence[1]
                ):
                    hard_blocks.append("kept pair sequence does not match kept fragments")
                    valid_sequence = False
            else:
                valid_sequence = False

        raw_crmsd = pair.get("crmsd", pair.get("cRMSD"))
        raw_ovrmsd = pair.get("ovrmsd", pair.get("ovRMSD"))
        valid_thresholds = True
        if isinstance(raw_crmsd, bool):
            hard_blocks.append("kept pair crmsd must be finite and positive")
            valid_thresholds = False
        else:
            try:
                crmsd = _finite_threshold(raw_crmsd, "kept pair crmsd")
            except ValueError:
                hard_blocks.append("kept pair crmsd must be finite and positive")
                valid_thresholds = False
        if isinstance(raw_ovrmsd, bool):
            hard_blocks.append("kept pair ovrmsd must be finite and positive")
            valid_thresholds = False
        else:
            try:
                threshold = _finite_threshold(raw_ovrmsd, "kept pair ovrmsd")
            except ValueError:
                hard_blocks.append("kept pair ovrmsd must be finite and positive")
                valid_thresholds = False
        if not valid_address or not valid_identity or not valid_thresholds or not valid_sequence:
            continue
        if threshold > L_OV:
            hard_blocks.append(
                f"X6: kept grow pair {down}->{up} ovrmsd exceeds L_OV"
            )
        grow_pair = copy.deepcopy(pair)
        grow_pair["down"] = down
        grow_pair["up"] = up
        if "crmsd" not in grow_pair:
            grow_pair["crmsd"] = raw_crmsd
        if "ovrmsd" not in grow_pair:
            grow_pair["ovrmsd"] = raw_ovrmsd
        grow_thresholds.append(grow_pair)

    for down, up in sorted(required_kept_pair_addresses - observed_kept_pair_addresses):
        hard_blocks.append(f"kept pair {down}->{up} is missing from pair evidence")

    proceed_condition = frozen.get("proceed_anyway_required")
    if type(proceed_condition) is not bool:
        hard_blocks.append("browser proceed-anyway condition is missing or invalid")
        proceed_required = False
    else:
        proceed_required = proceed_condition
    proceed = frozen.get("proceed_anyway") is True
    bundle = {
        "snapshot": frozen,
        "_snapshot_digest": _snapshot_digest(frozen),
        "reference_pdb": reference_bytes,
        "reference_sha256": reference_checksum,
        "exclude_code": exclude,
        "min_cell_fraction": fraction,
        "anchor_actions": anchor_actions,
        "anchor_refe_actions": anchor_refe_actions,
        "anchor_refe_comments": anchor_refe_comments,
        "energy_filter_actions": energy_filter_actions,
        "energy_thresholds": energy_threshold_file,
        "score_filter_defaults": copy.deepcopy(DEFAULT_SCORE_FILTER),
        "grow_thresholds": grow_thresholds,
        "kept_fragments": copy.deepcopy(frozen["kept_fragments"]),
        "handoff": copy.deepcopy(frozen.get("handoff")),
        "class_table": copy.deepcopy(frozen.get("class_table")),
        "validations": copy.deepcopy(frozen.get("validations")),
        "hard_blocks": sorted(set(hard_blocks)),
        "proceed_anyway_required": proceed_required,
        "proceed_anyway": proceed,
        "ready": not hard_blocks and (not proceed_required or proceed),
        # The export runtime must not set this variable; the action reads the
        # same default that validation used unless an operator explicitly chose it.
        "environment": {},
    }
    return bundle


def bundle_is_stale(bundle: dict[str, Any], snapshot: dict[str, Any]) -> bool:
    """Compare a later decision set with the bundle's immutable snapshot digest."""
    return _snapshot_digest(snapshot) != bundle.get("_snapshot_digest")

