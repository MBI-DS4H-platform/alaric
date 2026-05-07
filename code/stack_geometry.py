from __future__ import annotations

import numpy as np


RINGS = {
    "T": ["C6", "C5", "C7", "N3", "O2", "O4"],
    "U": ["N1", "C2", "N3", "C4", "C5", "C6"],
    "C": ["N1", "C2", "N3", "C4", "C5", "C6"],
    "G": ["N1", "C2", "N3", "C4", "C5", "C6", "N7", "C8", "N9"],
    "A": ["N1", "C2", "N3", "C4", "C5", "C6", "N7", "C8", "N9"],
    "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"],
    "ARG": ["CD", "NE", "CZ", "NH1", "NH2"],
    "TRP": ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
}
for left, right in (
    ("DT", "T"),
    ("RU", "U"),
    ("DC", "C"),
    ("RC", "C"),
    ("DG", "G"),
    ("RG", "G"),
    ("DA", "A"),
    ("RA", "A"),
):
    RINGS[left] = RINGS[right]


def as_coordinates(mol: np.ndarray) -> np.ndarray:
    return np.stack((mol["x"], mol["y"], mol["z"]), axis=-1)


def calc_plane(coordinates: np.ndarray) -> np.ndarray:
    centered = coordinates - coordinates.mean(axis=0)
    _, _, wt = np.linalg.svd(centered.T.dot(centered))
    plane_directionless = wt[2] / np.linalg.norm(wt[2])
    plane0_directed = np.cross((centered[1] - centered[0]), (centered[2] - centered[1]))
    flip = np.sign(np.dot(plane_directionless, plane0_directed))
    return plane_directionless * flip


def residue_ring_atom_mask(residue_atoms: np.ndarray) -> tuple[np.ndarray, str]:
    if len(residue_atoms) == 0:
        raise ValueError("Cannot select ring atoms from an empty residue")

    resnames = np.unique(residue_atoms["resname"])
    if len(resnames) != 1:
        raise ValueError("Residue resolves to multiple residue names")

    resname = resnames[0].decode()
    if resname not in RINGS:
        raise ValueError(
            f"Residue ({resname}) is unsupported; expected one of {sorted(RINGS)}"
        )

    ring_names = [name.encode() for name in RINGS[resname]]
    ring_mask = np.isin(residue_atoms["name"], ring_names)
    if not np.any(ring_mask):
        raise ValueError(f"Could not find ring atoms for residue ({resname})")
    if int(ring_mask.sum()) < 3:
        raise ValueError(f"Residue ({resname}) has too few ring atoms ({int(ring_mask.sum())})")
    return ring_mask, resname


def residue_ring_coordinates(
    atoms: np.ndarray,
    resid: int,
    *,
    structure_label: str,
) -> tuple[np.ndarray, str]:
    mask_resid = atoms["resid"] == resid
    if not np.any(mask_resid):
        raise ValueError(f"Residue {resid} was not found in {structure_label}")

    chains = np.unique(atoms["chain"][mask_resid])
    if len(chains) != 1:
        chains_str = ", ".join(
            ch.decode(errors="ignore").strip() or "<blank>" for ch in chains
        )
        raise ValueError(
            f"Residue ID {resid} is ambiguous across chains ({chains_str}); select a unique residue ID"
        )

    residue_atoms = atoms[mask_resid]
    ring_mask, resname = residue_ring_atom_mask(residue_atoms)
    coordinates = as_coordinates(residue_atoms[ring_mask])
    return coordinates, resname


def protein_ring_coordinates(protein_atoms: np.ndarray, resid: int) -> np.ndarray:
    coordinates, resname = residue_ring_coordinates(
        protein_atoms,
        resid,
        structure_label="protein",
    )
    if resname not in {"PHE", "TYR", "HIS", "ARG", "TRP"}:
        raise ValueError(
            f"Residue {resid} ({resname}) is not aromatic or unsupported; expected one of {sorted(RINGS)}"
        )
    return coordinates


def rna_ring_coordinates(rna_atoms: np.ndarray, resid: int) -> np.ndarray:
    coordinates, _ = residue_ring_coordinates(
        rna_atoms,
        resid,
        structure_label="RNA",
    )
    return coordinates


def stacking_angle_dihedral(
    protein_ring: np.ndarray,
    nucleotide_ring: np.ndarray,
) -> tuple[float, float]:
    protein_center = protein_ring.mean(axis=0)
    protein_plane = calc_plane(protein_ring)
    protein_x = protein_ring[0] - protein_center
    protein_x /= np.linalg.norm(protein_x)
    protein_y = np.cross(protein_plane, protein_x)
    protein_y /= np.linalg.norm(protein_y)

    base_center = nucleotide_ring.mean(axis=0)
    base_plane = calc_plane(nucleotide_ring)
    base_x = nucleotide_ring[0] - base_center

    cross_norm = np.linalg.norm(np.cross(protein_plane, base_plane))
    cross_norm = min(float(cross_norm), 1.0)
    angle = float(np.arcsin(cross_norm))

    ring_x = base_x.copy()
    ring_x -= np.dot(ring_x, protein_plane) * protein_plane
    ring_x_norm = np.linalg.norm(ring_x)
    if ring_x_norm < 0.0001:
        dihedral = float(np.pi / 2)
    else:
        ring_x /= ring_x_norm
        dihedral_x = np.dot(ring_x, protein_x)
        dihedral_y = np.dot(ring_x, protein_y)
        dihedral = float(np.angle(dihedral_x + 1.0j * dihedral_y))
    return angle, dihedral
