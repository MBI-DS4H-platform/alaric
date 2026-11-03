"""Small RNA backbone continuity check shared by Alaric command-line tools."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite, sqrt


MAX_BACKBONE_BOND_DISTANCE = 2.2


def peptide_link_is_continuous(left_atoms, right_atoms) -> bool:
    """Check the O3'–P distance between adjacent RNA residues.

    Inputs may be atom-name-to-coordinate mappings or iterables of structured
    atom records with ``name``, ``x``, ``y`` and ``z`` fields. Residue-number
    continuity is the caller's responsibility.
    """

    def atom_map(atoms):
        if isinstance(atoms, Mapping):
            result = {}
            for name, coordinate in atoms.items():
                key = str(name).strip().replace("*", "'")
                result[key] = tuple(float(value) for value in coordinate)
            return result

        result = {}
        for atom in atoms:
            raw_name = atom["name"]
            name = (
                raw_name.decode(errors="replace")
                if isinstance(raw_name, bytes)
                else str(raw_name)
            )
            key = name.strip().replace("*", "'")
            result[key] = (float(atom["x"]), float(atom["y"]), float(atom["z"]))
        return result

    left = atom_map(left_atoms)
    right = atom_map(right_atoms)
    if "O3'" not in left or "P" not in right:
        return False
    try:
        distance = sqrt(
            sum(
                (a - b) ** 2
                for a, b in zip(left["O3'"], right["P"], strict=True)
            )
        )
    except (TypeError, ValueError, OverflowError):
        return False
    return isfinite(distance) and distance <= MAX_BACKBONE_BOND_DISTANCE
