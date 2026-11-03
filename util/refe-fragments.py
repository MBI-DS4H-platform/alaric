#!/usr/bin/env python3
"""Cut an RNA PDB into template-ordered reference dinucleotide PDBs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
ALARIC_DIR = ROOT / "alaric"
if str(ALARIC_DIR) not in sys.path:
    sys.path.insert(0, str(ALARIC_DIR))

from parse_pdb import parse_pdb  # noqa: E402
from rna_pdb import ppdb2nucseq  # noqa: E402
from backbone import peptide_link_is_continuous  # noqa: E402
from write_pdb import write_pdb  # noqa: E402


def _text(value) -> str:
    return value.decode("ascii").strip() if isinstance(value, bytes) else str(value).strip()


def write_reference_fragments(
    rna_pdb: str,
    templates_dir: str | Path,
    output_dir: str | Path,
) -> list[str]:
    """Write one `frag-N-SEQ-refe.pdb` for every valid RNA dinucleotide."""
    templates_dir = Path(templates_dir)
    output_dir = Path(output_dir)
    mononucleotide_templates = {
        base: np.load(templates_dir / f"{base}-ppdb.npy", allow_pickle=False)
        for base in "ACGU"
    }
    ppdb = parse_pdb(rna_pdb)
    sequence, indices = ppdb2nucseq(
        ppdb, rna=True, ignore_unknown=False, return_index=True
    )
    residues = []
    atoms_by_residue = []
    for nucleotide_index, (first, last) in enumerate(indices):
        base = sequence[nucleotide_index]
        observed = ppdb[first:last]
        template = mononucleotide_templates[base]
        expected_names = [_text(atom["name"]) for atom in template]
        observed_names = [_text(atom["name"]) for atom in observed]
        if observed_names != expected_names:
            resid = int(ppdb[first]["resid"])
            raise ValueError(
                f"Nucleotide {base} at residue {resid} is missing, reordered, "
                "or has duplicate template atoms"
            )
        residues.append(int(ppdb[first]["resid"]))
        atoms_by_residue.append(
            {name: atom for name, atom in zip(observed_names, observed)}
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for residue_index in range(len(sequence) - 1):
        if residues[residue_index + 1] != residues[residue_index] + 1:
            continue
        if not peptide_link_is_continuous(
            atoms_by_residue[residue_index].values(),
            atoms_by_residue[residue_index + 1].values(),
        ):
            continue
        fragment_sequence = sequence[residue_index : residue_index + 2]
        fragment_number = residue_index + 1
        template = np.load(
            templates_dir / f"{fragment_sequence}-ppdb.npy", allow_pickle=False
        ).copy()
        for atom_index, template_atom in enumerate(template):
            template_residue = int(template_atom["resid"])
            if template_residue not in (1, 2):
                raise ValueError(
                    f"Template {fragment_sequence} contains residue {template_residue}; expected 1 or 2"
                )
            atom_name = _text(template_atom["name"])
            source = atoms_by_residue[residue_index + template_residue - 1].get(atom_name)
            if source is None:
                raise ValueError(
                    f"Template {fragment_sequence} atom {atom_name} has no reference coordinate"
                )
            template[atom_index]["x"] = source["x"]
            template[atom_index]["y"] = source["y"]
            template[atom_index]["z"] = source["z"]
        path = output_dir / f"frag-{fragment_number}-{fragment_sequence}-refe.pdb"
        path.write_text(write_pdb(template))
        written.append(path.name)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rna_pdb", type=Path)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    write_reference_fragments(
        args.rna_pdb.read_text(encoding="utf-8"), args.templates, args.output
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
