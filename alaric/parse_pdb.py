"""Utilities to parse PDB-format text into a structured numpy array representation.

This module provides:
- parse_pdb: Parse a PDB file content (as a string) using BioPython into a
  structured numpy.ndarray containing per-atom fields (coordinates, residue
  identifiers, chain, element, occupancy, etc.). The array dtype is defined by
  `atomic_dtype` and is suitable for downstream numeric processing and saving.
- get_coor: Extract an (N, 3) float array of XYZ coordinates from a parsed PDB
  structured array.

Notes:
- Bio.PDB is used internally; PDBConstructionWarning messages are suppressed.
- The returned array uses the aligned dtype `atomic_dtype`. Fields that are
  text in Bio.PDB (bytes/str) are placed into fixed-length byte/string fields
  as defined in `atomic_dtype`.
"""

import warnings
from io import StringIO
from typing import TypeAlias, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    # Type-only imports so static checkers (Pylance) know methods/attributes.
    # These are not imported at runtime to avoid importing Bio.PDB unless used.
    from Bio.PDB.PDBParser import PDBParser as _PDBParser  # type: ignore
    from Bio.PDB.Structure import Structure as _Structure  # type: ignore
    from Bio.PDB.Model import Model as _Model  # type: ignore
    from Bio.PDB.Residue import Residue as _Residue  # type: ignore
    from Bio.PDB.Atom import Atom as _Atom  # type: ignore

atomic_dtype0: list[tuple[str, str]] = [
    ("model", "uint16"),
    ("hetero", "S1"),
    ("name", "S4"),
    ("altloc", "S1"),
    ("resname", "S3"),
    ("chain", "S4"),
    ("index", "uint32"),
    ("icode", "S1"),
    ("resid", "uint16"),
    ("x", "float32"),
    ("y", "float32"),
    ("z", "float32"),
    ("occupancy", "float32"),
    ("bfactor", "float32"),
    ("segid", "S4"),
    ("element", "S2"),
]

atomic_dtype_unaligned: np.dtype = np.dtype(atomic_dtype0, align=False)
atomic_dtype: np.dtype = np.dtype(atomic_dtype0, align=True)

ParsedPDB: TypeAlias = np.ndarray


def parse_pdb(pdbdata: str) -> ParsedPDB:
    """Parse PDB text and return a structured numpy array of atoms.

    This function reads PDB-format text using Bio.PDB.PDBParser and constructs a
    numpy structured array with one row per atom. The dtype and field names are
    defined by `atomic_dtype`. Fields populated include model index, residue
    identifiers (resname, resid, icode), chain id, atom name and serial index,
    coordinates (x, y, z), occupancy, bfactor, segid and element symbol.

    Args:
        pdbdata: The raw contents of a PDB file as a single string.

    Returns:
        ParsedPDB: A numpy structured array with dtype `atomic_dtype`. The number
        of rows equals the number of ATOM/HETATM records in the provided PDB
        content.

    Raises:
        Any exceptions raised by Bio.PDB.PDBParser for malformed input will
        propagate (but PDBConstructionWarning is suppressed).
    """

    # Import public-facing locations to satisfy static checkers (Pylance).
    from Bio.PDB.PDBParser import PDBParser
    from Bio.PDB.PDBExceptions import PDBConstructionWarning

    warnings.simplefilter("ignore", PDBConstructionWarning)

    pdb_obj = StringIO(pdbdata)

    p = PDBParser()
    struc = p.get_structure("PDB", pdb_obj)
    # Narrow the returned type for static checkers: get_structure may be
    # annotated to possibly return None/Unknown; assert it is present.
    assert struc is not None
    natoms = len(list(struc.get_atoms()))
    atomstate = np.zeros(natoms, dtype=atomic_dtype)

    a = atomstate
    count = 0
    for modelnr, model in enumerate(struc.get_models()):
        atomlist = list(model.get_atoms())
        atomlist.sort(key=lambda atom: atom.serial_number)
        for atom in atomlist:
            residue = atom.get_parent()
            # Narrow potential Optional return for static checkers.
            assert residue is not None
            hetero, resid, icode = residue.get_id()
            segid = residue.segid
            resname = residue.resname
            chain_parent = residue.get_parent()
            assert chain_parent is not None
            chainid = chain_parent.id
            aa = a[count]
            aa["model"] = modelnr + 1
            aa["hetero"] = hetero
            aa["name"] = atom.name
            aa["altloc"] = atom.altloc
            aa["resname"] = resname
            aa["chain"] = chainid
            aa["index"] = atom.serial_number
            aa["icode"] = icode
            aa["resid"] = resid
            aa["x"] = atom.coord[0]
            aa["y"] = atom.coord[1]
            aa["z"] = atom.coord[2]
            occ = atom.occupancy
            if occ is None or occ < 0:
                occ = 0
            aa["occupancy"] = occ
            aa["segid"] = segid
            aa["element"] = atom.element
            count += 1
    return atomstate


def get_coor(ppdb: ParsedPDB) -> np.ndarray:
    """Return an (N, 3) float32 array of XYZ coordinates from a parsed PDB array.

    Args:
        ppdb: A ParsedPDB structured numpy array (as produced by parse_pdb).

    Returns:
        numpy.ndarray: Shape (N, 3) with columns (x, y, z) and dtype float32.

    Raises:
        TypeError: If ppdb does not support the required fields (will naturally
        raise when indexing fields if incorrect).
    """
    return np.stack((ppdb["x"], ppdb["y"], ppdb["z"]), axis=-1)


if __name__ == "__main__":
    import sys

    pdbfile: str = sys.argv[1]
    outfile: str = sys.argv[2]
    data: ParsedPDB = parse_pdb(open(pdbfile).read())
    np.save(outfile, data, allow_pickle=False)
