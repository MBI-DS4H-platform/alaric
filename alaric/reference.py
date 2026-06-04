from __future__ import annotations

import numpy as np

from library import load_mononucleotide_templates
from rna_pdb import ppdb2nucseq


class Reference:
    """Template-ordered nucleotide coordinates from a parsed PDB reference."""

    def __init__(
        self,
        ppdb: np.ndarray,
        *,
        rna: bool = True,
        ignore_unknown: bool,
        ignore_missing: bool,
        ignore_reordered: bool,
    ) -> None:
        mononucleotide_templates = load_mononucleotide_templates()
        for base, template in mononucleotide_templates.items():
            seq, _ = ppdb2nucseq(
                template,
                rna=rna,
                ignore_unknown=False,
                return_index=True,
            )
            if seq != base:
                raise ValueError(f"Invalid template for {base} ({seq})")

        monoseq, indices = ppdb2nucseq(
            ppdb, rna=rna, ignore_unknown=ignore_unknown, return_index=True
        )
        assert isinstance(indices, list)

        sequence = []
        resids = []
        coordinates = []
        for nuc_index, base in enumerate(monoseq):
            first, last = indices[nuc_index]
            first_atom = ppdb[first]
            resid0 = int(first_atom["resid"])

            if base not in mononucleotide_templates:
                if ignore_unknown:
                    continue
                raise ValueError(f"Unknown base '{base}'")

            template = mononucleotide_templates[base]
            if last - first < len(template):
                if ignore_missing:
                    continue
                raise ValueError(
                    f"Nucleotide '{base}', resid {resid0}: "
                    f"incorrect number of atoms {last - first}, "
                    f"while template has {len(template)} atoms"
                )

            atom_indices = []
            has_missing_atoms = False
            for template_atom in template:
                atom_name = template_atom["name"]
                matches = [
                    atom_index
                    for atom_index in range(first, last)
                    if ppdb[atom_index]["name"] == atom_name
                ]
                if not matches:
                    if ignore_missing:
                        has_missing_atoms = True
                        break
                    raise ValueError(
                        f"Nucleotide '{base}', resid {resid0}: missing atom {atom_name!r}"
                    )
                if len(matches) > 1:
                    raise ValueError(
                        f"Nucleotide '{base}', resid {resid0}: duplicate atom {atom_name!r}"
                    )
                atom_indices.append(matches[0])

            if has_missing_atoms:
                continue

            if ignore_reordered and atom_indices != list(range(first, last)):
                continue

            resids.append(resid0)
            ordered = ppdb[atom_indices]
            refe_coor = np.stack(
                (ordered["x"], ordered["y"], ordered["z"]), axis=-1
            ).astype(float)
            sequence.append(base)
            coordinates.append(refe_coor)

        self.ppdb = ppdb
        self.rna = rna
        self.sequence = "".join(sequence)
        self._coordinates = coordinates
        self._resids = resids

    def get_fragment_positions(self, fraglen: int) -> list[int]:
        """Return 1-based positions for all contiguous valid fragments."""
        result = []
        for pos0 in range(len(self._resids) - fraglen + 1):
            resids = self._resids[pos0 : pos0 + fraglen]
            first = resids[0]
            if resids != list(range(first, first + fraglen)):
                continue
            result.append(first - self._resids[0] + 1)
        return result

    def get_resids(self, fragment_position: int, fraglen: int) -> list[int]:
        """Return residue IDs for a valid fragment position."""
        pos = self._get_pos(fragment_position, fraglen)
        return self._resids[pos : pos + fraglen]

    def get_coordinates(self, fragment_position: int, fraglen: int) -> np.ndarray:
        """Return template-ordered coordinates for a valid fragment position."""
        pos = self._get_pos(fragment_position, fraglen)
        coordinates = self._coordinates[pos : pos + fraglen]
        if fraglen == 1:
            return coordinates[0].copy()
        return np.concatenate(coordinates)

    def get_sequence(self, fragment_position: int, fraglen: int) -> str:
        """Return the nucleotide sequence for a valid fragment position."""
        pos = self._get_pos(fragment_position, fraglen)
        return self.sequence[pos : pos + fraglen]

    def _get_pos(self, fragment_position: int, fraglen: int) -> int:
        if not self._resids:
            raise ValueError("Reference contains no valid nucleotides")
        offset = self._resids[0]
        first_resid = offset + fragment_position - 1
        try:
            pos = self._resids.index(first_resid)
        except ValueError:
            raise ValueError("Not a valid fragment") from None
        if self._resids[pos : pos + fraglen] != list(
            range(first_resid, first_resid + fraglen)
        ):
            raise ValueError("Not a valid fragment")
        return pos
