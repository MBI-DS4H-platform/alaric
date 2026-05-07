import gc
import hashlib
import itertools
import os
import re
from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, TypeAlias, cast
import numpy as np
from ruamel.yaml import YAML

from rna_pdb import ppdb2nucseq
from mutate import mutate

Coordinates: TypeAlias = np.ndarray
ReductionMap: TypeAlias = dict[str, list[tuple[str, ...]]]


@lru_cache(maxsize=None)
def _load_reduction_map(reduce_file: str) -> ReductionMap:
    mapping: ReductionMap = {}
    current_residue: Optional[str] = None
    with open(reduce_file) as handle:
        lines = handle.readlines()

    for lineno, raw in enumerate(lines, start=1):
        line = raw.split("#", maxsplit=1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) == 1:
            current_residue = fields[0]
            continue
        if current_residue is None:
            raise ValueError(f"Malformed reduce file {reduce_file}:{lineno}: {raw.strip()}")
        if not current_residue.startswith("R"):
            continue
        try:
            atom_type = int(fields[0])
        except ValueError as exc:
            raise ValueError(
                f"Malformed atom type in {reduce_file}:{lineno}: {fields[0]}"
            ) from exc
        if atom_type == 99:
            continue
        if len(fields) < 3:
            raise ValueError(f"Malformed reduce file {reduce_file}:{lineno}: {raw.strip()}")
        atom_names = list(fields[2:])
        try:
            float(atom_names[-1])
        except ValueError:
            pass
        else:
            atom_names = atom_names[:-1]
        if not atom_names:
            raise ValueError(
                f"Reduction bead without atoms in {reduce_file}:{lineno}: {raw.strip()}"
            )
        residue_name = current_residue[1:]
        mapping.setdefault(residue_name, []).append(tuple(atom_names))
    return mapping


def _build_reduction_plan(
    template: np.ndarray, reduction_map: ReductionMap
) -> list[np.ndarray]:
    if template.dtype.names is None:
        raise TypeError("Template must be a structured array")
    for field in ("resid", "resname", "name"):
        if field not in template.dtype.names:
            raise TypeError(f"Template is missing required field '{field}'")

    if len(template) == 0:
        return []

    residue_ids = template["resid"]
    residue_names = np.char.strip(template["resname"].astype("U"))
    atom_names = np.char.strip(template["name"].astype("U"))

    boundaries = np.nonzero(residue_ids[1:] != residue_ids[:-1])[0] + 1
    starts = np.concatenate((np.array([0], dtype=int), boundaries))
    ends = np.concatenate((boundaries, np.array([len(template)], dtype=int)))

    plan: list[np.ndarray] = []
    for start, end in zip(starts, ends):
        residue_name = residue_names[start]
        residue_id = int(residue_ids[start])
        groups = reduction_map.get(residue_name)
        if groups is None:
            raise ValueError(
                f"No reduction definition for residue '{residue_name}' (resid {residue_id})"
            )

        atom_lookup: dict[str, int] = {}
        for atom_index in range(start, end):
            atom_name = atom_names[atom_index]
            if atom_name in atom_lookup:
                raise ValueError(
                    f"Duplicate atom name '{atom_name}' in residue '{residue_name}' (resid {residue_id})"
                )
            atom_lookup[atom_name] = atom_index

        for atom_group in groups:
            missing = [name for name in atom_group if name not in atom_lookup]
            if missing:
                missing_list = ", ".join(missing)
                raise ValueError(
                    f"Missing atom(s) {missing_list} for residue '{residue_name}' (resid {residue_id})"
                )
            plan.append(np.array([atom_lookup[name] for name in atom_group], dtype=np.intp))
    return plan


@dataclass(frozen=True)
class Library:
    """
    Nucleotide library for a fragment sequence (potentially filtered and concatenated)

    Attributes:

    coordinates: np.ndarray
        (X, Y, 3) coordinate array.
        Contains at least the primary array, potentially replaced using the replacement array.
        If both a primary and extension coordinate array were present, "coordinates" is their concatenation.
        The (extension part of the) array may have been filtered based on conformer origin.


    nprimary: int
        The first nprimary conformers of "coordinates" are primary conformers.

    sequence: str
        Nucleotide sequence

    template: np.ndarray
        Parsed PDB template (same atom order/selection as coordinates axis 1)

    atom_mask: Optional[np.ndarray]
        Optional mask that indicates which atoms were selected by residue selection

    conformer_mask: Optional[np.ndarray]
        Optional mask that indicates which conformers are valid.
            Where False, the (extension) conformer must not be used, because of its PDB origin.
        Mutually exclusive with conformer_mapping

    conformer_mapping: Optional[np.ndarray]
        Optional, present only if the conformer array were pruned based on PDB origin.
        Maps the filtered-out conformer indices to the original ones.
        Mutually exclusive with conformer_mask and rotaconformers.

    rotaconformers: Optional[np.ndarray]
        Optional: discrete rotamers (3x3 rotation matrices), concatenated for all conformers.
        Use get_rotamers() to get the rotamers for a specific conformer.

    rotaconformers_index: Optional[np.ndarray]
        Optional: for each conformer, the index of the first rotamer in the rotaconformer array
        Use get_rotamers() to get the rotamers for a specific conformer.

    """

    coordinates: np.ndarray
    nprimary: int
    sequence: str
    template: np.ndarray
    atom_mask: Optional[np.ndarray]
    conformer_mask: Optional[np.ndarray]
    conformer_mapping: Optional[np.ndarray]
    rotaconformers: Optional[np.ndarray]
    rotaconformers_index: Optional[np.ndarray]

    def __post_init__(self) -> None:
        self._validate()

    def get_rotamers(self, conformer: int) -> np.ndarray:
        """Get all discrete rotamers for a conformer"""
        if conformer < 0 or conformer >= len(self.coordinates):
            raise ValueError
        if self.rotaconformers is None:
            raise AttributeError
        assert self.rotaconformers_index is not None
        last = self.rotaconformers_index[conformer]
        if conformer == 0:
            first = 0
        else:
            first = self.rotaconformers_index[conformer - 1]
        return self.rotaconformers[first:last]

    def reduce(self, reduce_file: str | Path | None = None) -> np.ndarray:
        """Reduce atomic coordinates to bead coordinates defined in reduce.dat."""
        if self.coordinates.ndim != 3 or self.coordinates.shape[2] != 3:
            raise ValueError("coordinates must have shape (X, Y, 3)")
        if len(self.template) != self.coordinates.shape[1]:
            raise ValueError("Template and coordinates atom count mismatch")
        if reduce_file is None:
            reduce_path = Path(__file__).with_name("reduce.dat")
        else:
            reduce_path = Path(reduce_file)
        reduction_map = _load_reduction_map(str(reduce_path.resolve()))
        plan = _build_reduction_plan(self.template, reduction_map)

        reduced = np.empty(
            (self.coordinates.shape[0], len(plan), 3), dtype=self.coordinates.dtype
        )
        for bead_index, atom_indices in enumerate(plan):
            reduced[:, bead_index] = self.coordinates[:, atom_indices].mean(axis=1)
        return reduced

    def _validate(self) -> None:
        if self.conformer_mask is not None:
            if self.conformer_mapping is not None:
                raise ValueError(
                    "conformer_mask and conformer_mapping are mutually exclusive"
                )
        if self.rotaconformers is not None:
            if self.rotaconformers_index is None:
                raise ValueError(
                    "rotaconformers_index must be set when rotaconformers is set"
                )
            if len(self.rotaconformers_index) != len(self.coordinates):
                raise ValueError("rotaconformers_index length must match coordinates")
            if self.rotaconformers_index[-1] != len(self.rotaconformers):
                raise ValueError(
                    "rotaconformers_index[-1] must equal number of rotaconformers"
                )
        if self.rotaconformers_index is not None:
            if self.rotaconformers is None:
                raise ValueError(
                    "rotaconformers must be set when rotaconformers_index is set"
                )
        if self.conformer_mapping is not None:
            if self.conformer_mask is not None:
                raise ValueError(
                    "conformer_mapping and conformer_mask are mutually exclusive"
                )
            if self.rotaconformers is not None:
                raise ValueError(
                    "conformer_mapping is incompatible with rotaconformers"
                )
            if self.rotaconformers_index is not None:
                raise ValueError(
                    "conformer_mapping is incompatible with rotaconformers_index"
                )

        # Invariants for type checkers (should be guaranteed by the checks above).
        assert (self.conformer_mask is None) or (self.conformer_mapping is None)
        assert (self.rotaconformers is None) == (self.rotaconformers_index is None)
        if self.rotaconformers is not None:
            assert self.rotaconformers_index is not None
            assert len(self.rotaconformers_index) == len(self.coordinates)
            assert self.rotaconformers_index[-1] == len(self.rotaconformers)
        if self.conformer_mapping is not None:
            assert self.conformer_mask is None
            assert self.rotaconformers is None
            assert self.rotaconformers_index is None


class LibraryFactory:
    def __init__(
        self,
        sequence: str,
        primary_coordinates: Coordinates,
        *,
        template: np.ndarray,
        rna: bool,
        replacement_origins: Optional[list[str]] = None,
        replacement_coordinates: Optional[Coordinates] = None,
        extension_coordinates: Optional[Coordinates] = None,
        extension_origins: Optional[list[str]] = None,
        rotaconformers_file: Optional[str] = None,
        rotaconformers_index_file: Optional[str] = None,
        rotaconformers_extension_file: Optional[str] = None,
        rotaconformers_extension_index_file: Optional[str] = None,
    ) -> None:
        """Factory class for nucleotide libraries.
        Instantiated by LibraryDirectory.load.
        Use self.create() to create actual libraries
        """
        if (rotaconformers_file is None) != (rotaconformers_index_file is None):
            raise ValueError(
                "rotaconformers_file and rotaconformers_index_file must be set together"
            )
        if (rotaconformers_extension_file is None) != (
            rotaconformers_extension_index_file is None
        ):
            raise ValueError(
                "rotaconformers_extension_file and rotaconformers_extension_index_file must be set together"
            )
        if rotaconformers_file and extension_coordinates is not None:
            if rotaconformers_extension_file is None:
                raise ValueError(
                    "rotaconformers_extension_file must be set when extension_coordinates is set"
                )
            if rotaconformers_extension_index_file is None:
                raise ValueError(
                    "rotaconformers_extension_index_file must be set when extension_coordinates is set"
                )
        # Invariants for type checkers (should be guaranteed by the checks above).
        assert (rotaconformers_file is None) == (rotaconformers_index_file is None)
        assert (rotaconformers_extension_file is None) == (
            rotaconformers_extension_index_file is None
        )
        self.sequence = sequence
        self.template = template
        self.rna = rna
        self.primary_coordinates = primary_coordinates
        self.nprimary = len(primary_coordinates)
        self.replacement_coordinates = replacement_coordinates
        self.replacement_origins = replacement_origins
        self.extension_coordinates = extension_coordinates
        self.extension_origins = extension_origins
        self.rotaconformers = None
        self.rotaconformers_index = None
        self.rotaconformers_file = rotaconformers_file
        self.rotaconformers_index_file = rotaconformers_index_file
        self.rotaconformers_extension_file = rotaconformers_extension_file
        self.rotaconformers_extension_index_file = rotaconformers_extension_index_file

    def load_rotaconformers(self) -> None:
        """Load the rotamers from file into memory.
        Convert them into rotation matrix form.
        This is expensive, both in terms of disk I/O and in terms of memory,

        If the rotaconformers have been loaded already, do nothing."""
        if self.rotaconformers is not None:
            return
        rotaconformers = None
        rotaconformers_index = None
        if self.rotaconformers_file:
            assert self.rotaconformers_index_file is not None
            rotaconformers = np.load(self.rotaconformers_file)
            rotaconformers_index = np.load(self.rotaconformers_index_file)
            assert len(rotaconformers_index) == len(self.primary_coordinates)
            if self.extension_coordinates is not None:
                assert self.rotaconformers_extension_file is not None
                assert self.rotaconformers_extension_index_file is not None
                rotaconformers_extension0 = np.load(self.rotaconformers_extension_file)
                rotaconformers_extension_index = np.load(
                    self.rotaconformers_extension_index_file
                )
                assert len(rotaconformers_extension_index) == len(
                    self.extension_coordinates
                )
                offset = len(rotaconformers)
                rotaconformers = np.concatenate(
                    (rotaconformers, rotaconformers_extension0)
                )
                del rotaconformers_extension0
                rotaconformers_index = np.concatenate(
                    (rotaconformers_index, rotaconformers_extension_index + offset)
                )

        self.rotaconformers = rotaconformers
        self.rotaconformers_index = rotaconformers_index

    def unload_rotaconformers(self) -> None:
        """Detach the rotaconformers from the factory.
        Note that they remain attached to any created library objects.
        Memory will only be freed if all created libraries have been destroyed."""
        self.rotaconformers = None
        self.rotaconformers_index = None
        gc.collect()

    def rotamer_index_uint16(self) -> bool:
        """Returns if all rotamer indices for all conformers can fit in an uint16"""
        if self.rotaconformers_file is None:
            raise RuntimeError("No rotaconformers")
        if self.rotaconformers is None:
            raise RuntimeError("Rotaconformers must be loaded first")
        assert self.rotaconformers_index is not None

        return np.diff(self.rotaconformers_index, prepend=0).max() < 2**16

    def create(
        self,
        pdb_code: Optional[str | list[str] | tuple[str, ...] | set[str]],
        *,
        prune_conformers: bool = False,
        nucleotide_mask: Optional[np.ndarray] = None,
        only_base: bool = False,
        with_rotaconformers: bool = False,
    ) -> Library:
        """Creates a Library, filtered by origin and/or nucleotide selection.

        - Filtering by origin. Provide a pdb_code (string) or a list/tuple/set of pdb_code strings.
            All conformers in the primary list with one of those PDB codes as origin will be replaced.
            All conformers in the extension list with one of those PDB codes will be invalidated.
                If prune_conformers, the conformers will be physically removed from the array,
                 and a mapping from the old array to the new array is added to the returned library
                This is incompatible with with_rotaconformers.

        - Filtering by nucleotide mask. Provide a boolean mask with the same length as the sequence.
            Only the atoms of the nucleotides where the mask is True are selected in the conformer coordinates
            Rotaconformers are unaffected.

        If only_base = True, only the coordinates of the base are returned.

        nucleotide mask and only_base are reflected in the atom mask of the library (Library.atom_mask).

        If with_rotaconformers is True, the factory's rotaconformers are attached to the library object.
        This is cheap in terms of memory. It requires that the rotaconformers have been loaded with .load_rotaconformers

        """
        if nucleotide_mask is not None:
            if len(nucleotide_mask) != len(self.sequence):
                raise ValueError(
                    "Nucleotide mask must have the same length as the sequence"
                )
        assert not (prune_conformers and with_rotaconformers)

        if (
            with_rotaconformers
            and self.rotaconformers_file is not None
            and self.rotaconformers is None
        ):
            raise RuntimeError("Rotaconformers must be loaded first")

        pdb_codes = _normalise_pdb_codes(pdb_code)

        primary_coordinates = self.primary_coordinates
        if pdb_codes is not None and self.replacement_origins is not None:
            to_replace = _origin_indices_for_pdb_codes(
                self.replacement_origins, pdb_codes
            )
            if to_replace:
                assert self.replacement_coordinates is not None
                primary_coordinates = primary_coordinates.copy()
                primary_coordinates[to_replace] = self.replacement_coordinates[
                    to_replace
                ]
        conformer_mask = None
        conformer_mapping = None
        if self.extension_coordinates is not None:
            extension_coordinates = self.extension_coordinates
            if pdb_codes is not None and self.extension_origins is not None:
                to_replace = _origin_indices_for_pdb_codes(
                    self.extension_origins, pdb_codes
                )
                if to_replace:
                    offset = len(primary_coordinates)
                    mask = np.zeros(len(extension_coordinates), bool)
                    mask[to_replace] = True
                    if prune_conformers:
                        mapping = np.arange(len(extension_coordinates), dtype=int)[
                            ~mask
                        ]
                        extension_coordinates = extension_coordinates[~mask]
                        conformer_mapping = np.concatenate(
                            (
                                np.arange(len(primary_coordinates), dtype=int),
                                mapping + offset,
                            )
                        )
                    else:
                        conformer_mask = np.concatenate(
                            (np.ones((len(primary_coordinates)), bool), ~mask)
                        )

            coordinates = np.concatenate((primary_coordinates, extension_coordinates))
            del primary_coordinates, extension_coordinates
        else:
            coordinates = primary_coordinates

        template = self.template
        atom_mask = None
        if only_base or nucleotide_mask is not None:
            atom_mask = np.ones(len(self.template), bool)
            if nucleotide_mask is not None:
                nucseq, nuc_indices = ppdb2nucseq(
                    self.template, rna=self.rna, return_index=True
                )
                assert nucseq == self.sequence
                for n in range(len(self.sequence)):
                    if not nucleotide_mask[n]:
                        first, last = nuc_indices[n]
                        atom_mask[first:last] = 0
            if only_base:
                base = np.array(
                    [len(name) == 2 for name in self.template["name"]], bool
                )
                atom_mask &= base
            coordinates = coordinates[:, atom_mask]
            template = template[atom_mask]

        rotaconformers = None
        rotaconformers_index = None
        if with_rotaconformers:
            rotaconformers = self.rotaconformers
            rotaconformers_index = self.rotaconformers_index

        result = Library(
            sequence=self.sequence,
            coordinates=coordinates,
            nprimary=self.nprimary,
            template=template,
            atom_mask=atom_mask,
            conformer_mask=conformer_mask,
            conformer_mapping=conformer_mapping,
            rotaconformers=rotaconformers,
            rotaconformers_index=rotaconformers_index,
        )
        result._validate()
        return result


class LibraryDirectory:
    def __init__(
        self,
        fraglen: int,
        filepattern: str,
        *,
        rna: bool = True,
        replacement_filepattern: Optional[str] = None,
        replacement_origin_filepattern: Optional[str] = None,
        extension_filepattern: Optional[str] = None,
        extension_origin_filepattern: Optional[str] = None,
        rotaconformers_filepattern: Optional[str] = None,
        rotaconformers_index_filepattern: Optional[str] = None,
        rotaconformers_extension_filepattern: Optional[str] = None,
        rotaconformers_extension_index_filepattern: Optional[str] = None,
    ) -> None:
        """
        Nucleotide fragment library directory. Must contain A/C conformer arrays.

        fraglen: length of nucleotide fragment.

        filepattern: file pattern for primary conformer array.

        When loading, every file pattern has (X * fraglen) replaced by the A/C sequence

        Optional:

        - rna: True if RNA (default), else DNA

        - replacement_filepattern: file pattern for the replacement conformer array.
        These contain intra-cluster (i.e. within the precision of the library)
        replacements for conformers that come from the same PDB that is being modeled.

        - replacement_origin_filepattern: file pattern that contains the PDB origin of
        each conformer, and the replacement index that is to be used.

        - extension_filepattern: file pattern for the extension conformer library.
        This library contains singletons (i.e. not replaceable) that improve the fit.

        - rotaconformers_filepattern: file pattern for the array of discrete rotamers
        Rotamers are scaled axis-angle 3-vectors, converted to 3x3 rotation matrices upon loading.
        The array is concatenated for all conformers.

        - rotaconformers_index_filepattern: file pattern for the rotaconformer indices.
        These are, for each conformer, the index of the first rotamer in the rotaconformer array,

        - rotaconformers_extension_filepattern: file pattern for array of discrete rotamers for the extension library

        - rotaconformers_extension_index_filepattern: file pattern for rotamers for the extension library
        """
        self.fraglen = fraglen
        self.filepattern = filepattern
        self.rna = rna
        self.replacement_filepattern = replacement_filepattern
        self.replacement_origin_filepattern = replacement_origin_filepattern
        self.extension_filepattern = extension_filepattern
        self.extension_origin_filepattern = extension_origin_filepattern
        self.rotaconformers_filepattern = rotaconformers_filepattern
        self.rotaconformers_index_filepattern = rotaconformers_index_filepattern
        self.rotaconformers_extension_filepattern = rotaconformers_extension_filepattern
        self.rotaconformers_extension_index_filepattern = (
            rotaconformers_extension_index_filepattern
        )

    def load(
        self,
        sequence: str,
        template: np.ndarray,
        *,
        with_extension: bool = False,
        with_replacement: bool = False,
        with_rotaconformers: bool = False,
        loaded_libraries: Optional[dict[str, "LibraryFactory"]] = None,
    ) -> LibraryFactory:
        """Load a library, returning it as a LibraryFactory.

        Optionally, a cache of previously loaded LibraryFactory instances
        may be provided, to prevent redundant loading from disk"""
        assert len(sequence) == self.fraglen

        if with_extension:
            assert self.extension_filepattern is not None
        if with_replacement:
            assert self.replacement_filepattern is not None
            assert self.replacement_origin_filepattern is not None
            if with_extension:
                assert self.extension_origin_filepattern is not None
        if with_rotaconformers:
            assert self.rotaconformers_filepattern is not None
            assert self.rotaconformers_index_filepattern is not None
            if with_extension:
                assert self.rotaconformers_extension_filepattern is not None
                assert self.rotaconformers_extension_index_filepattern is not None
        template_sequence = ppdb2nucseq(template, rna=self.rna)
        if template_sequence != sequence:
            raise ValueError(
                f"Template contains sequence {template_sequence}, whereas {sequence} is specified"
            )
        seq0 = _representative_sequence(sequence, rna=self.rna)
        if loaded_libraries is not None and seq0 in loaded_libraries:
            primary_coordinates0 = loaded_libraries[seq0].primary_coordinates
        else:
            primary_coordinates0 = np.load(_sequence_path(self.filepattern, seq0))
        primary_coordinates = mutate(primary_coordinates0, seq0, sequence)
        extension_coordinates = None
        if with_extension:
            if loaded_libraries is not None and seq0 in loaded_libraries:
                extension_coordinates0 = loaded_libraries[seq0].extension_coordinates
            else:
                assert self.extension_filepattern is not None
                extension_coordinates0 = np.load(
                    _sequence_path(self.extension_filepattern, seq0)
                )
            assert extension_coordinates0 is not None
            extension_coordinates = mutate(extension_coordinates0, seq0, sequence)
        replacement_coordinates = None
        replacement_origins = None
        extension_origins = None
        if with_replacement:
            assert self.replacement_filepattern is not None
            replacement_coordinates0 = np.load(
                _sequence_path(self.replacement_filepattern, seq0)
            )
            if len(replacement_coordinates0) != len(primary_coordinates):
                raise ValueError(
                    "Shape mismatch between primary coordinates and replacement coordinates"
                )
            replacement_coordinates = mutate(replacement_coordinates0, seq0, sequence)
            assert self.replacement_origin_filepattern is not None
            with open(_sequence_path(self.replacement_origin_filepattern, seq0)) as f:
                replacement_origins0 = f.read()
            replacement_origins = [
                l.split()[0] for l in replacement_origins0.splitlines()[1:]
            ]
            if len(replacement_origins) != len(primary_coordinates):
                raise ValueError(
                    "Length mismatch between primary coordinates and origins"
                )
            if with_extension:
                assert self.extension_origin_filepattern is not None
                with open(_sequence_path(self.extension_origin_filepattern, seq0)) as f:
                    extension_origins0 = f.read()
                extension_origins = [l.strip() for l in extension_origins0.splitlines()]
            if extension_origins is not None and extension_coordinates is not None:
                if len(extension_origins) != len(extension_coordinates):
                    raise ValueError(
                        "Length mismatch between extension coordinates and extension origins"
                    )
            elif extension_origins is not None:
                raise ValueError(
                    "Length mismatch between extension coordinates and extension origins"
                )

        rotaconformers_file = None
        rotaconformers_index_file = None
        rotaconformers_extension_file = None
        rotaconformers_extension_index_file = None
        if with_rotaconformers:
            assert self.rotaconformers_filepattern is not None
            rotaconformers_file = _sequence_path(self.rotaconformers_filepattern, seq0)
            assert os.path.exists(rotaconformers_file)
            assert self.rotaconformers_index_filepattern is not None
            rotaconformers_index_file = _sequence_path(
                self.rotaconformers_index_filepattern, seq0
            )
            assert os.path.exists(rotaconformers_index_file)
            if with_extension:
                assert self.rotaconformers_extension_filepattern is not None
                rotaconformers_extension_file = (
                    _sequence_path(self.rotaconformers_extension_filepattern, seq0)
                )
                assert os.path.exists(rotaconformers_extension_file)
                assert self.rotaconformers_extension_index_filepattern is not None
                rotaconformers_extension_index_file = (
                    _sequence_path(
                        self.rotaconformers_extension_index_filepattern, seq0
                    )
                )
                assert os.path.exists(rotaconformers_extension_index_file)

        return LibraryFactory(
            sequence,
            primary_coordinates,
            template=template,
            rna=self.rna,
            replacement_coordinates=replacement_coordinates,
            replacement_origins=replacement_origins,
            extension_coordinates=extension_coordinates,
            extension_origins=extension_origins,
            rotaconformers_file=rotaconformers_file,
            rotaconformers_extension_file=rotaconformers_extension_file,
            rotaconformers_index_file=rotaconformers_index_file,
            rotaconformers_extension_index_file=rotaconformers_extension_index_file,
        )


_FRAGLIB_YAML_PATH = Path("~/.alaric/fraglib.yaml").expanduser()
_VAR_PATTERN = re.compile(r"\$(\w+)|\${([^}]+)}")
_FRAGLIB_REQUIRED_KEYS = [
    "fraglen",
    "conformers",
    "conformer_replacements",
    "conformer_replacement_origins",
    "conformer_extensions",
    "conformer_extension_origins",
    "rotamers",
    "rotamers_indices",
    "rotamer_extensions",
    "rotamer_extension_indices",
    "crmsds",
]


def _load_fraglib_yaml(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"fraglib.yaml not found: {path}")
    yaml = YAML(typ="safe")
    with path.open("r") as handle:
        data = yaml.load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("fraglib.yaml must contain a mapping at the top level")
    return data


def _substitute_vars(value: str, mapping: dict[str, object]) -> str:
    def replacer(match: re.Match) -> str:
        var = match.group(1) or match.group(2)
        if var in mapping:
            return str(mapping[var])
        if var in os.environ:
            return os.environ[var]
        raise KeyError(var)

    return _VAR_PATTERN.sub(replacer, value)


def _expand_fraglib_config(raw: dict[str, object]) -> dict[str, object]:
    resolved = dict(raw)
    for _ in range(len(resolved) + 1):
        changed = False
        for key, value in list(resolved.items()):
            if not isinstance(value, str):
                continue
            expanded = os.path.expanduser(value)
            try:
                expanded = _substitute_vars(expanded, resolved)
            except KeyError:
                continue
            if expanded != value:
                resolved[key] = expanded
                changed = True
        if not changed:
            break

    unresolved = {
        key: value
        for key, value in resolved.items()
        if isinstance(value, str) and _VAR_PATTERN.search(value)
    }
    if unresolved:
        details = ", ".join(
            f"{key}={value}" for key, value in sorted(unresolved.items())
        )
        raise ValueError(f"Unresolved variables in fraglib.yaml: {details}")

    return resolved


@lru_cache(maxsize=None)
def _load_fraglib_config(path: str) -> dict[str, object]:
    raw_config = _load_fraglib_yaml(Path(path))
    config_data = _expand_fraglib_config(raw_config)
    _require_config_keys(config_data, _FRAGLIB_REQUIRED_KEYS)
    return config_data


def _fraglib_config() -> dict[str, object]:
    return _load_fraglib_config(str(_FRAGLIB_YAML_PATH))


def _require_config_keys(config: dict[str, object], keys: list[str]) -> None:
    missing = [key for key in keys if key not in config or config[key] is None]
    if missing:
        missing_str = ", ".join(missing)
        raise KeyError(f"Missing required fraglib.yaml keys: {missing_str}")


def _require_str(config: dict[str, object], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Expected string for key '{key}'")
    return value


def _require_int(config: dict[str, object], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Expected int for key '{key}'")
    return value


def _representative_sequence(sequence: str, *, rna: bool = True) -> str:
    sequence = sequence.upper()
    base_ut = "U" if rna else "T"
    invalid = sorted(set(sequence) - {"A", "C", "G", base_ut})
    if invalid:
        invalid_str = ", ".join(invalid)
        raise ValueError(f"Invalid nucleotide base(s): {invalid_str}")
    return sequence.replace("G", "A").replace(base_ut, "C")


def _normalise_pdb_codes(
    pdb_code: Optional[str | list[str] | tuple[str, ...] | set[str]],
) -> Optional[set[str]]:
    if pdb_code is None:
        return None
    if isinstance(pdb_code, str):
        pdb_codes = {pdb_code.lower()}
    elif isinstance(pdb_code, (list, tuple, set)):
        pdb_codes = {str(code).lower() for code in pdb_code}
    else:
        raise TypeError("pdb_code must be a string or a list/tuple/set of strings")
    return pdb_codes or None


def _origin_indices_for_pdb_codes(origins: list[str], pdb_codes: set[str]) -> list[int]:
    return [
        origin_index
        for origin_index, origin in enumerate(origins)
        if origin.lower() in pdb_codes
    ]


def _sequence_path(filepattern: str, sequence: str) -> str:
    placeholder = "X" * len(sequence)
    if placeholder not in filepattern:
        raise ValueError(
            f"File pattern {filepattern!r} does not contain placeholder {placeholder!r}"
        )
    return filepattern.replace(placeholder, sequence)


def _extension_origin_indices(
    config_data: dict[str, object],
    sequence: str,
    pdb_codes: set[str],
) -> np.ndarray:
    primary_path = Path(
        _sequence_path(_require_str(config_data, "conformers"), sequence)
    )
    primary_len = int(np.load(primary_path, mmap_mode="r").shape[0])

    origins_path = Path(
        _sequence_path(_require_str(config_data, "conformer_extension_origins"), sequence)
    )
    origins = [line.strip() for line in origins_path.read_text().splitlines()]
    matching = _origin_indices_for_pdb_codes(origins, pdb_codes)
    return primary_len + np.array(matching, dtype=np.intp)


def load_crmsds(
    ab: str,
    bc: str,
    pdb_code: Optional[str | list[str] | tuple[str, ...] | set[str]] = None,
) -> np.ndarray:
    """Load the cRMSD matrix for overlapping dinucleotide libraries AB and BC.

    The matrix is stored under the representative A/C trinucleotide sequence
    ABC, following the same G->A and U->C convention as the dinucleotide
    library loader. If pdb_code is provided, extension conformers with that
    origin are invalidated by setting their rows or columns to infinity.
    Primary conformer replacement-table substitution is intentionally ignored.
    """
    ab = ab.upper()
    bc = bc.upper()
    if len(ab) != 2 or len(bc) != 2:
        raise ValueError("ab and bc must both be dinucleotide sequences")
    if ab[1] != bc[0]:
        raise ValueError(f"Dinucleotides must overlap as AB/BC, got {ab!r} and {bc!r}")

    pair_sequence = ab + bc[1]
    crmsd_sequence = _representative_sequence(pair_sequence)
    ab_sequence = crmsd_sequence[:2]
    bc_sequence = crmsd_sequence[1:]

    config_data = _fraglib_config()

    crmsd_path = Path(
        _sequence_path(_require_str(config_data, "crmsds"), crmsd_sequence)
    )
    crmsds = np.load(crmsd_path)

    pdb_codes = _normalise_pdb_codes(pdb_code)
    if pdb_codes is None:
        return crmsds

    _require_config_keys(
        config_data,
        ["conformers", "conformer_extension_origins"],
    )
    rows = _extension_origin_indices(config_data, ab_sequence, pdb_codes)
    cols = _extension_origin_indices(config_data, bc_sequence, pdb_codes)
    if rows.size and int(rows.max()) >= crmsds.shape[0]:
        raise ValueError(
            f"Extension origin rows for {ab_sequence} exceed cRMSD matrix shape {crmsds.shape}"
        )
    if cols.size and int(cols.max()) >= crmsds.shape[1]:
        raise ValueError(
            f"Extension origin columns for {bc_sequence} exceed cRMSD matrix shape {crmsds.shape}"
        )
    if (rows.size or cols.size) and not np.issubdtype(crmsds.dtype, np.floating):
        crmsds = crmsds.astype(float)
    if rows.size:
        crmsds[rows, :] = np.inf
    if cols.size:
        crmsds[:, cols] = np.inf
    return crmsds


def _checksum_target_path(
    checksum_path: Path,
    fraglib_root: Path,
    conformer_dir: Path,
    rotamer_dir: Path,
    crmsd_dir: Path,
) -> Path:
    relative = checksum_path.relative_to(fraglib_root)
    if relative.parts and relative.parts[0] == "conformers":
        subpath = Path(*relative.parts[1:]).with_suffix("")
        return conformer_dir / subpath
    if relative.parts and relative.parts[0] == "rotamers":
        subpath = Path(*relative.parts[1:]).with_suffix("")
        return rotamer_dir / subpath
    if relative.parts and relative.parts[0] == "crmsds":
        subpath = Path(*relative.parts[1:]).with_suffix("")
        return crmsd_dir / subpath
    return fraglib_root / relative.with_suffix("")


def _sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checksums(config: dict[str, object], fraglib_root: Path) -> None:
    checksum_files = sorted(fraglib_root.rglob("*.CHECKSUM"))
    if not checksum_files:
        raise FileNotFoundError(f"No .CHECKSUM files found in {fraglib_root}")

    conformer_dir = Path(os.path.dirname(_require_str(config, "conformers")))
    rotamer_dir = Path(os.path.dirname(_require_str(config, "rotamers")))
    crmsd_dir = Path(os.path.dirname(_require_str(config, "crmsds")))

    errors = []
    for checksum_path in checksum_files:
        expected_raw = checksum_path.read_text().strip()
        expected = expected_raw.split()[0] if expected_raw else ""
        target_path = _checksum_target_path(
            checksum_path, fraglib_root, conformer_dir, rotamer_dir, crmsd_dir
        )
        if not target_path.exists():
            errors.append(f"Missing file for checksum: {target_path}")
            continue
        actual = _sha256sum(target_path)
        if expected.lower() != actual.lower():
            errors.append(
                f"Checksum mismatch for {target_path}: expected {expected}, got {actual}"
            )

    if errors:
        details = "\n".join(errors)
        raise RuntimeError(f"Checksum verification failed:\n{details}")


def _resolve_templates_dir(config: dict[str, object], fraglib_root: Path) -> Path:
    for key in ("templates_dir", "template_dir", "templates"):
        if key in config:
            value = config[key]
            if isinstance(value, str):
                return Path(value)
    fraglib_dir = config.get("FRAGLIB_DIR") or config.get("fraglib_dir")
    if fraglib_dir:
        candidate = Path(cast(str, fraglib_dir)) / "templates"
        if candidate.is_dir():
            return candidate
    return fraglib_root / "templates"


def config(
    verify_checksums: bool = True,
) -> tuple[dict[str, "LibraryFactory"], dict[str, np.ndarray]]:
    config_data = _fraglib_config()

    fraglib_root = Path(__file__).resolve().parent.parent / "fraglib"
    if verify_checksums:
        _verify_checksums(config_data, fraglib_root)

    lib_dinuc_directory = LibraryDirectory(
        fraglen=_require_int(config_data, "fraglen"),
        filepattern=_require_str(config_data, "conformers"),
        replacement_filepattern=_require_str(config_data, "conformer_replacements"),
        replacement_origin_filepattern=_require_str(
            config_data, "conformer_replacement_origins"
        ),
        extension_filepattern=_require_str(config_data, "conformer_extensions"),
        extension_origin_filepattern=_require_str(
            config_data, "conformer_extension_origins"
        ),
        rotaconformers_filepattern=_require_str(config_data, "rotamers"),
        rotaconformers_index_filepattern=_require_str(config_data, "rotamers_indices"),
        rotaconformers_extension_filepattern=_require_str(
            config_data, "rotamer_extensions"
        ),
        rotaconformers_extension_index_filepattern=_require_str(
            config_data, "rotamer_extension_indices"
        ),
    )

    _bases = ["A", "C", "G", "U"]
    mononucleotide_templates = {}
    templates_dir = _resolve_templates_dir(config_data, fraglib_root)
    for base in _bases:
        ppdb = np.load(str(templates_dir / f"{base}-ppdb.npy"))
        mononucleotide_templates[base] = ppdb

    dinucleotide_templates = {}
    dinucleotide_libraries = {}
    for seq0 in itertools.product(_bases, repeat=2):
        seq = "".join(seq0)
        ppdb = np.load(str(templates_dir / f"{seq}-ppdb.npy"))
        dinucleotide_templates[seq] = ppdb
        libf = lib_dinuc_directory.load(
            sequence=seq,
            template=ppdb,
            with_extension=True,
            with_replacement=True,
            with_rotaconformers=True,
            loaded_libraries=dinucleotide_libraries,
        )
        dinucleotide_libraries[seq] = libf
    return dinucleotide_libraries, dinucleotide_templates
