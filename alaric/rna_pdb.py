"""Convert a ParsedPDB atomic array (from parse_pdb) into nucleotide sequences.

This module provides utilities to map residue names in a parsed PDB array to
single-letter nucleotide codes and to extract contiguous nucleotide sequences
(with optional atom-index ranges). It supports both RNA and DNA modes and can
be extended with custom mappings for modified residues.

Key functions:
- map_resname: Map a single residue name (bytes or str) to a single-letter base.
- ppdb2nucseq: Convert a ParsedPDB to a nucleotide sequence string; optionally
  return per-nucleotide atom index ranges.

Behavior details:
- The code expects a structured numpy array with fields matching `atomic_dtype`
  (from parse_pdb). If an unaligned dtype is provided, it will be converted.
- The module enforces a single chain; multi-chain inputs raise ValueError.
- Unknown/modified residues can either raise a KeyError or be skipped when
  ignore_unknown=True.
"""

import numpy as np
from typing import Tuple, Any
from parse_pdb import ParsedPDB, atomic_dtype, atomic_dtype_unaligned

_basic_mapping_rna: dict[str, str] = {k: k for k in ("A", "C", "G", "U")}
_basic_mapping_dna: dict[str, str] = {k: k for k in ("A", "C", "G", "T")}
_basic_mapping_rna.update({"R" + k: k for k in ("A", "C", "G", "U")})
_basic_mapping_dna.update({"D" + k: k for k in ("A", "C", "G", "T")})


def map_resname(
    resname: Any,
    *,
    rna: bool = True,
    extra_mapping: dict[str, str] | None = None,
) -> str:
    """Map a single residue name (from a parsed PDB) to a single-letter base.

    This function accepts resname as either bytes (common in numpy structured
    arrays) or str. It consults a built-in canonical mapping for RNA (A,C,G,U)
    or DNA (A,C,G,T). Optionally an `extra_mapping` dict can be provided to
    translate modified or non-standard residue names (keys and values are str).

    Args:
        resname: Residue name (bytes or str) as present in the parsed PDB array.
        rna: If True interpret mapping as RNA; if False interpret as DNA.
        extra_mapping: Optional dict of additional mappings (resname -> base).

    Returns:
        Single-letter base as a str ("A","C","G","U"/"T").

    Raises:
        KeyError: If resname is not found in the selected mapping. If the
        user accidentally selected the wrong mode (rna vs DNA) but the residue
        exists in the opposite mapping, a helpful KeyError is raised.
    """

    # Normalize a variety of bytes-like / numpy scalar inputs into a str.
    if isinstance(resname, (bytes, bytearray, memoryview, np.bytes_)):
        try:
            resname = bytes(resname).decode()
        except Exception:
            resname = str(resname)
    elif not isinstance(resname, str):
        # numpy scalar (e.g. numpy.bytes_) or 0-d ndarray: try .item()
        if hasattr(resname, "item"):
            val = resname.item()
            if isinstance(val, (bytes, bytearray, memoryview, np.bytes_)):
                try:
                    resname = bytes(val).decode()
                except Exception:
                    resname = str(val)
            else:
                resname = str(val)
        else:
            resname = str(resname)

    mapping = _basic_mapping_rna if rna else _basic_mapping_dna
    if extra_mapping is not None:
        mapping = mapping.copy()
        mapping.update(extra_mapping)
    try:
        return mapping[resname]
    except KeyError as exc:
        if not rna and resname in _basic_mapping_rna:
            raise KeyError("Mode is DNA (rna=False) but resname is RNA") from None
        elif rna and resname in _basic_mapping_dna:
            raise KeyError("Mode is RNA but resname is DNA") from None
        else:
            raise exc from None


def ppdb2nucseq(
    ppdb: ParsedPDB,
    *,
    rna: bool,
    ignore_unknown: bool = False,
    return_index: bool = False,
) -> str | Tuple[str, list[tuple[int, int]]]:
    """Convert a parsed PDB structured array to a mono-nucleotide sequence.

    The function groups atoms by residue id (resid) and extracts the residue
    name of the first atom of each residue to determine the nucleotide. It
    validates that the input structured array has the expected fields and
    attempts to canonicalize unaligned or slightly different dtypes.

    Args:
        ppdb: ParsedPDB structured numpy array with fields matching atomic_dtype.
        rna: Interpret residues as RNA (True) or DNA (False).
        ignore_unknown: If True, residues that lack a mapping are skipped;
            otherwise a KeyError is raised on the first unknown residue.
        return_index: If True, return a tuple (sequence, indices) where indices
            is a list of (first_atom_index, last_atom_index) for each nucleotide.
            The index ranges are half-open: [first, last).

    Returns:
        If return_index is False: a string with the nucleotide sequence.
        If return_index is True: (sequence: str, indices: list[tuple[int,int]]).

    Raises:
        TypeError: If ppdb is not a structured array with the expected fields.
        ValueError: If the parsed PDB contains more than one chain.
        KeyError: If an unknown residue name is encountered and ignore_unknown
            is False.
    """
    if ppdb.dtype.names is None:
        raise TypeError("ppdb must be a structured array")

    required_fields = atomic_dtype.names
    if ppdb.dtype.names != required_fields:
        raise TypeError(f"ppdb has unexpected fields: {ppdb.dtype.names}")

    def _canonicalize(ppdb_arr: ParsedPDB) -> ParsedPDB:
        """Ensure the parsed array has the aligned `atomic_dtype`.

        This helper will:
        - Return the array unchanged if it already has the aligned dtype.
        - Cast from the unaligned dtype variant if that matches.
        - Attempt to copy fields into a newly allocated aligned array if the
          field names match but some field sizes differ (common when chain is
          a one-byte S1 field in some inputs).

        Raises TypeError if the array cannot be canonicalized.
        """
        if np.dtype(ppdb_arr.dtype) == atomic_dtype:
            return ppdb_arr
        if np.dtype(ppdb_arr.dtype) == atomic_dtype_unaligned:
            return ppdb_arr.astype(atomic_dtype)

        if ppdb_arr.dtype.names == atomic_dtype.names:
            fields = ppdb_arr.dtype.fields
            if fields is None:
                raise TypeError("dtype.fields is None; cannot canonicalize")
            chain_dtype = fields["chain"][0]
            if chain_dtype.kind == "S" and chain_dtype.itemsize == 1:
                corrected = np.empty(ppdb_arr.shape, dtype=atomic_dtype)
                for name in atomic_dtype.names:  # type: ignore
                    corrected[name] = ppdb_arr[name]
                return corrected
        raise TypeError(np.dtype(ppdb_arr.dtype), atomic_dtype)

    ppdb = _canonicalize(ppdb)
    if len(np.unique(ppdb["chain"])) > 1:
        raise ValueError("Multiple chains in parsed PDB")

    indices0 = np.nonzero(np.diff(ppdb["resid"], prepend=-1, append=-1))[0]
    indices = []
    nucs = []
    for n in range(len(indices0) - 1):
        ind1, ind2 = indices0[n], indices0[n + 1]
        # Extract a Python scalar from possible numpy scalar/0-d array before mapping
        raw_res = ppdb[ind1]["resname"]
        if hasattr(raw_res, "item"):
            raw_res = raw_res.item()
        try:
            nuc = map_resname(raw_res, rna=rna)
        except KeyError as exc:
            if ignore_unknown:
                continue
            else:
                raise exc from None
        nucs.append(nuc)
        indices.append((ind1, ind2))

    seq = "".join(nucs)
    if return_index:
        return seq, indices
    else:
        return seq
