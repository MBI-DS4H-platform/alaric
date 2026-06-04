#!/usr/bin/env python3
# Copyright Sjoerd J. De Vries (INSERM)
"""Reduce a parsed-PDB npy file to ATTRACT bead coordinates and atom types.

Reads a ppdb.npy produced by alaric/parse_pdb.py, applies the ATTRACT reduction
defined in alaric/reduce.dat, and writes:
  <output-prefix>-coor.npy       -- (nrbeads, 3) float32 bead coordinates
  <output-prefix>-atomtypes.npy  -- (nrbeads,)   int64   ATTRACT atom type

Usage:
    python3 reduce-npy.py <ppdb.npy> <output-prefix>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_CODE = Path(__file__).with_name("alaric")
_REDUCE_DAT = _CODE / "reduce.dat"


def read_forcefield(forcefieldfile: str | Path) -> dict:
    ff: dict = {}
    aa = None
    for line in open(forcefieldfile):
        pound = line.find("#")
        if pound > -1:
            line = line[:pound]
        line = line.strip()
        if not line:
            continue
        ll = line.split()
        if len(ll) == 1:
            aa = ll[0]
            assert len(aa) <= 3, line
            ff[aa] = []
        else:
            assert aa is not None
            try:
                atomtype = int(ll[0])
            except ValueError:
                raise ValueError(line)
            atoms = ll[2:]
            charge = 0.0
            try:
                charge = float(atoms[-1])
                atoms = atoms[:-1]
            except ValueError:
                pass
            ff[aa].append((int(ll[0]), ll[1], set(atoms), charge))
    return ff


def reduce_ppdb(ppdb: np.ndarray, ff: dict) -> tuple[np.ndarray, np.ndarray]:
    """Reduce all-atom ppdb to ATTRACT bead coordinates and atom types.

    Returns
    -------
    coor : (nrbeads, 3) float32
    atomtypes : (nrbeads,) int64
    """
    # Build topology: list of (resname, {atom_name: atom_index_in_ppdb})
    topology: list[tuple[str, dict[str, int]]] = []
    prev_res_key = None
    atoms: dict[str, int] | None = None
    resname: str = ""

    for i, atom in enumerate(ppdb):
        res_key = (bytes(atom["chain"]).strip(), int(atom["resid"]), bytes(atom["icode"]).strip())
        if res_key != prev_res_key:
            if atoms is not None:
                topology.append((resname, atoms))
            prev_res_key = res_key
            resname = bytes(atom["resname"]).decode().strip()
            assert resname in ff, (
                f"Residue '{resname}' at chain {res_key[0]} resid {res_key[1]} "
                f"not found in reduce.dat"
            )
            atoms = {}
        atom_name = bytes(atom["name"]).decode().strip()
        atoms[atom_name] = i  # type: ignore[index]

    if atoms:
        topology.append((resname, atoms))

    # Count output beads
    nrbeads = sum(len(ff[rname]) for rname, _ in topology)

    xyz = np.stack([ppdb["x"], ppdb["y"], ppdb["z"]], axis=-1)  # (natoms, 3)

    coor_out = np.zeros((nrbeads, 3), dtype=np.float32)
    atomtypes_out = np.zeros(nrbeads, dtype=np.int64)

    beadpos = 0
    for resnr, (rname, atom_map) in enumerate(topology):
        for beadindex, beadname, beadatoms, charge in ff[rname]:
            bead_xyz = np.zeros(3, dtype=np.float64)
            for beadatom in beadatoms:
                assert beadatom in atom_map, (resnr + 1, rname, beadatom)
                bead_xyz += xyz[atom_map[beadatom]]
            coor_out[beadpos] = bead_xyz / len(beadatoms)
            atomtypes_out[beadpos] = beadindex
            beadpos += 1

    return coor_out, atomtypes_out


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <ppdb.npy> <output-prefix>", file=sys.stderr)
        sys.exit(1)

    ppdb_path = Path(sys.argv[1])
    out_prefix = sys.argv[2]

    ppdb = np.load(ppdb_path, allow_pickle=False)
    if ppdb.ndim != 1 or "x" not in ppdb.dtype.names:
        raise ValueError(
            f"{ppdb_path}: expected 1-D structured ppdb array with x/y/z fields"
        )

    ff = read_forcefield(_REDUCE_DAT)
    coor, atomtypes = reduce_ppdb(ppdb, ff)

    coor_path = out_prefix + "-coor.npy"
    atomtypes_path = out_prefix + "-atomtypes.npy"
    np.save(coor_path, coor)
    np.save(atomtypes_path, atomtypes)
    print(
        f"Reduced {len(ppdb)} atoms → {len(coor)} beads; "
        f"saved {coor_path} and {atomtypes_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
