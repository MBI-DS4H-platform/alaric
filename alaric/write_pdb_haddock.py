"""Write a HADDOCK3-compatible PDB ensemble of a protein with RNA chains.

Author: Sjoerd de Vries

Input is a ppdb array file as created by ``alaric-chain-coordinates``: an
``(nchains, natoms)`` parsed-PDB array of RNA atoms, one row (model) per chain.
The protein comes from a separate PDB file, given on the command line.

HADDOCK has the following requirements:

- Create a single PDB multimodel ensemble file (prot + rna)
- To each model, the ATOMs from a protein PDB file (command line argument) must be
  added (prepended) as a chain
- Both RNA and protein chain must have chain ID (hard-coded here as chain A for the
  protein and chain B for the RNA)
- Change two-column RNA resnames into one-column, e.g. change " RU " into "  U "
- remove the 1st phosphate group (P, O1P, O2P) from the RNA coordinates
"""

import argparse
import sys

import numpy as np

from parse_pdb import atomic_dtype, parse_pdb
from rna_pdb import map_resname
from write_pdb import _decode, write_pdb_atom

PROTEIN_CHAIN = b"A"
RNA_CHAIN = b"B"

# the 5'-terminal phosphate, which HADDOCK does not want on the first nucleotide
TERMINAL_PHOSPHATE = ("P", "O1P", "O2P")


def _normalize(atoms: np.ndarray, chain: bytes) -> None:
    """Give every atom the chain ID (and matching segid) and clean occupancy/bfactor."""
    atoms["chain"] = chain
    atoms["segid"] = chain.ljust(4)  # segids are left-justified in columns 73-76
    atoms["occupancy"] = 1.0
    atoms["bfactor"] = 0.0


def load_protein_atoms(pdbfile: str) -> np.ndarray:
    """Read the ATOMs of a protein PDB file as chain A of a parsed-PDB array."""
    with open(pdbfile) as f:
        atoms = parse_pdb(f.read())

    models = np.unique(atoms["model"])
    if len(models) > 1:
        print(
            f"Warning: {pdbfile} has {len(models)} models, using the first one",
            file=sys.stderr,
        )
        atoms = atoms[atoms["model"] == models[0]]

    # HETATM records (waters, ligands, ...) are not part of the protein
    atoms = atoms[np.char.strip(atoms["hetero"]) == b""]  # masking copies
    if not len(atoms):
        raise ValueError(f"{pdbfile} contains no protein ATOMs")

    atoms["model"] = 1
    _normalize(atoms, PROTEIN_CHAIN)
    return atoms


def _squeeze_resnames(atoms: np.ndarray) -> None:
    """Change two-column RNA resnames into one-column, e.g. " RU " into "  U "."""
    for resname in np.unique(atoms["resname"]):
        try:
            base = map_resname(resname)
        except KeyError:
            raise ValueError(
                f"resname {_decode(resname).strip()!r} is not an RNA residue name"
            ) from None
        atoms["resname"][atoms["resname"] == resname] = base.encode()


def _terminal_phosphate_mask(atoms: np.ndarray) -> np.ndarray:
    """Mask of the atoms of the 1st phosphate group, over the atoms of one model."""
    first_resid = atoms["resid"][0]
    names = np.char.strip(atoms["name"])
    phosphate = np.isin(names, [name.encode() for name in TERMINAL_PHOSPHATE])
    return phosphate & (atoms["resid"] == first_resid)


def prepare_rna_atoms(struc: np.ndarray) -> np.ndarray:
    """Make an RNA chain ensemble HADDOCK-compatible, as chain B.

    Sets the chain ID, squeezes the resnames and strips the 5'-terminal phosphate.
    All models describe the same nucleotides, so the same atoms are stripped from each.
    """
    if struc.dtype != atomic_dtype:
        raise TypeError("Input is not a parsed-PDB array")
    if struc.ndim != 2:
        raise ValueError(
            "Input must be an (nchains, natoms) parsed-PDB array as written by "
            f"alaric-chain-coordinates, got {struc.ndim} dimension(s)"
        )
    if not struc.size:
        raise ValueError("Input contains no atoms")

    for field in ("name", "resid"):
        if not (struc[field] == struc[field][0]).all():
            raise ValueError(
                f"the models of the input do not all have the same {field}s, so they "
                f"do not describe the same nucleotides"
            )

    atoms = struc[:, ~_terminal_phosphate_mask(struc[0])]  # masking copies
    _normalize(atoms, RNA_CHAIN)
    _squeeze_resnames(atoms)
    return atoms


def write_haddock_model(protein: np.ndarray, rna: np.ndarray) -> str:
    """Write one model: the protein chain, then the RNA chain, each closed by TER."""
    model = np.concatenate((protein, rna))
    model["index"] = np.arange(1, len(model) + 1, dtype=atomic_dtype["index"])
    pdb = ""
    for atom in model[: len(protein)]:
        pdb += write_pdb_atom(atom)
    pdb += "TER\n"
    for atom in model[len(protein) :]:
        pdb += write_pdb_atom(atom)
    pdb += "TER\n"
    return pdb


def write_haddock_pdb(struc: np.ndarray, protein: np.ndarray) -> str:
    """Write the RNA chains and the protein as a single multimodel ensemble."""
    rna = prepare_rna_atoms(struc)
    pdb = ""
    for n, model in enumerate(rna):
        pdb += "MODEL     %4d\n" % (n + 1)
        pdb += write_haddock_model(protein, model)
        pdb += "ENDMDL\n"
    pdb += "END\n"
    return pdb


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="write_pdb_haddock",
        description=(
            "Write a HADDOCK3-compatible multimodel PDB ensemble from the RNA chains "
            "built by alaric-chain-coordinates and a protein PDB file."
        ),
    )
    parser.add_argument(
        "npy_file", help="Parsed-PDB array of RNA chains (chains.ppdb.npy)."
    )
    parser.add_argument("protein_pdb", help="PDB file of the protein.")
    parser.add_argument("outfile", help="PDB file to write.")
    args = parser.parse_args(argv)

    struc = np.load(args.npy_file)
    protein = load_protein_atoms(args.protein_pdb)
    pdb_data = write_haddock_pdb(struc, protein)
    with open(args.outfile, "w") as f:
        f.write(pdb_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
