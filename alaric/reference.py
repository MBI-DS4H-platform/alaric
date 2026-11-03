from __future__ import annotations

import numpy as np

try:
    from .backbone import MAX_BACKBONE_BOND_DISTANCE, peptide_link_is_continuous
except ImportError:  # pragma: no cover - direct CLI import
    from backbone import MAX_BACKBONE_BOND_DISTANCE, peptide_link_is_continuous

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
        self._continuous_links = []
        for index in range(max(0, len(resids) - 1)):
            if resids[index + 1] != resids[index] + 1:
                self._continuous_links.append(False)
                continue
            left_template = mononucleotide_templates[sequence[index]]
            right_template = mononucleotide_templates[sequence[index + 1]]
            o3_index = _template_atom_index(left_template, {"O3'", "O3*"})
            p_index = _template_atom_index(right_template, {"P"})
            if o3_index is None or p_index is None:
                self._continuous_links.append(False)
                continue
            self._continuous_links.append(
                peptide_link_is_continuous(
                    {"O3'": coordinates[index][o3_index]},
                    {"P": coordinates[index + 1][p_index]},
                )
            )

    def get_fragment_positions(self, fraglen: int) -> list[int]:
        """Return 1-based positions for all contiguous valid fragments."""
        result = []
        for pos0 in range(len(self._resids) - fraglen + 1):
            resids = self._resids[pos0 : pos0 + fraglen]
            first = resids[0]
            if resids != list(range(first, first + fraglen)):
                continue
            if not self._has_continuous_backbone(pos0, fraglen):
                continue
            # Fragment numbers describe positions in the prepared RNA sequence,
            # not PDB residue labels.  A numbering gap leaves a hole while all
            # later fragment numbers retain their ordinal value.
            result.append(pos0 + 1)
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
        if fragment_position < 1:
            raise ValueError("Not a valid fragment") from None
        pos = fragment_position - 1
        if pos + fraglen > len(self._resids):
            raise ValueError("Not a valid fragment")
        first_resid = self._resids[pos]
        if self._resids[pos : pos + fraglen] != list(
            range(first_resid, first_resid + fraglen)
        ):
            raise ValueError("Not a valid fragment")
        if not self._has_continuous_backbone(pos, fraglen):
            raise ValueError("Not a valid fragment: RNA backbone is discontinuous")
        return pos

    def _has_continuous_backbone(self, pos: int, fraglen: int) -> bool:
        if fraglen <= 1:
            return True
        links = getattr(self, "_continuous_links", None)
        if links is None:
            # A lightweight Reference assembled from already-validated positions
            # may not carry coordinates; retain the residue-number continuity rule.
            return all(
                self._resids[index + 1] == self._resids[index] + 1
                for index in range(pos, pos + fraglen - 1)
            )
        return len(links) >= pos + fraglen - 1 and all(
            links[index] for index in range(pos, pos + fraglen - 1)
        )


def _template_atom_index(template: np.ndarray, names: set[str]) -> int | None:
    for index, raw_name in enumerate(template["name"]):
        atom_name = bytes(raw_name).decode(errors="replace").strip().replace("*", "'")
        if atom_name in names:
            return index
    return None
