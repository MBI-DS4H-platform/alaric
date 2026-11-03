"""All-filter Gaussian validation over a T20 job list."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

# Alaric modules use flat imports in the action entry points. Add only this
# module directory so package imports and the flattened Seamless sandbox share
# the same declared dependency set.
_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

from anchor_constants import MIN_CELL_FRACTION
from anchor_precalc import (
    GRID_SPACING,
    _load_reference_coordinates,
    _rotamers_to_matrices,
    build_precalculated_anchor,
    cell_fill,
    prepare_precalculated_arrays,
)
from gaussian_cache import FILE_NAMES, verify_filter_cache
from gaussian_filter import Model, containment_edges, load_spec, score as gaussian_score
import library
from npy_io import load_npy
from parse_pdb import parse_pdb
from rna_pdb import ppdb2nucseq
from stack_geometry import residue_ring_atom_mask, residue_ring_coordinates


@dataclass(frozen=True)
class _PreparedJob:
    key: Any
    fragment: int
    nucleotide: str
    sequence: str
    protein_range: int
    protein_resid: int
    grid: np.ndarray
    ring_coordinates: np.ndarray
    conformer_to_cache_reference: np.ndarray
    rotamer_matrix: np.ndarray
    protein_path: Path
    protein_ring: np.ndarray
    protein_center: np.ndarray
    nucleotide_cache_reference: np.ndarray
    protein_cache_reference: np.ndarray
    nucleotide_score_reference: np.ndarray
    protein_score_reference: np.ndarray
    score: float
    distance: float
    phi: float
    theta: float
    geometry_key: tuple[str, int, str]


class _SegmentedRotamers:
    """Slice primary and extension rotamers from read-only memory maps."""

    def __init__(self, primary: np.ndarray, extension: np.ndarray | None = None):
        self.primary = primary
        self.extension = extension
        self.offset = len(primary)

    def __len__(self) -> int:
        return self.offset + (0 if self.extension is None else len(self.extension))

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            if step != 1:
                return np.asarray([self[index] for index in range(start, stop, step)])
            if stop <= self.offset or self.extension is None:
                return self.primary[start:stop]
            if start >= self.offset:
                return self.extension[start - self.offset : stop - self.offset]
            return np.concatenate(
                (
                    self.primary[start : self.offset],
                    self.extension[: stop - self.offset],
                )
            )
        index = int(key)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if index < self.offset or self.extension is None:
            return self.primary[index]
        return self.extension[index - self.offset]


def _attach_lazy_rotamers(factory, pose_library) -> None:
    if factory.rotaconformers_file is None or factory.rotaconformers_index_file is None:
        raise ValueError(f"rotamer files are missing for {factory.sequence}")
    primary = np.load(factory.rotaconformers_file, mmap_mode="r", allow_pickle=False)
    primary_index = np.load(
        factory.rotaconformers_index_file, mmap_mode="r", allow_pickle=False
    )
    extension = None
    if factory.rotaconformers_extension_file is not None:
        extension = np.load(
            factory.rotaconformers_extension_file,
            mmap_mode="r",
            allow_pickle=False,
        )
        if factory.rotaconformers_extension_index_file is None:
            raise ValueError(f"extension rotamer index is missing for {factory.sequence}")
        extension_index = np.load(
            factory.rotaconformers_extension_index_file,
            mmap_mode="r",
            allow_pickle=False,
        )
        indices = np.concatenate((primary_index, extension_index + len(primary)))
    else:
        indices = primary_index
    accessor = _SegmentedRotamers(primary, extension)
    object.__setattr__(pose_library, "rotaconformers", accessor)
    object.__setattr__(pose_library, "rotaconformers_index", indices)
    pose_library._validate()


def semantic_cache_identity(spec: dict[str, Any]) -> dict[str, Any]:
    """Return computation identity, excluding compression-only SHA-256 values."""
    arrays: dict[str, dict[str, str]] = {}
    for name, entry in sorted(spec.get("thresholds", {}).items()):
        files = entry.get("precalc", {}).get("files", {})
        arrays[name] = {
            filename: str(files[filename]["npy_sha256"])
            for filename in FILE_NAMES
        }
    return {"spec_digest": str(spec["spec_digest"]), "arrays": arrays}


def _parse_semantic_identity(value: str | dict[str, Any]) -> dict[str, Any]:
    """Parse the explicit A9 identity input supplied to a Seamless transform."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("semantic_identity must be a JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError("semantic_identity must be a JSON object")
    return value


def _validate_semantic_identity(
    supplied: str | dict[str, Any] | None, spec: dict[str, Any]
) -> dict[str, Any]:
    """Check the semantic transform input against the authoritative A7 JSON.

    The JSON itself is read at runtime so its compressed SHA-256 values remain
    available to the cache verifier. It is deliberately not a declared code
    dependency: Seamless keys this computation by the separate semantic identity
    input, declared code, and ordinary runtime inputs.
    """
    expected = semantic_cache_identity(spec)
    if supplied is None:
        return expected
    actual = _parse_semantic_identity(supplied)
    if actual != expected:
        raise ValueError(
            "A9 semantic_identity does not match the authoritative Gaussian "
            "specification (spec_digest or npy_sha256 changed)"
        )
    return actual


def _repo_root(value: str | Path | None) -> Path:
    root = Path(value).resolve() if value is not None else Path(__file__).resolve().parents[1]
    if not (root / "alaric" / "gaussian_stacking_filter.json").is_file():
        raise FileNotFoundError(
            f"Gaussian filter specification not found under repository root {root}"
        )
    return root


def _spec_path(root: Path) -> Path:
    staged = _MODULE_DIR / "gaussian_stacking_filter.json"
    if staged.is_file():
        return staged
    return root / "alaric" / "gaussian_stacking_filter.json"


def _filter_order(
    spec: dict[str, Any],
) -> tuple[list[str], set[tuple[str, str]], dict[str, set[str]]]:
    thresholds = spec["thresholds"]
    edges = containment_edges(thresholds)
    containers = {
        name: {broader for contained, broader in edges if contained == name}
        for name in thresholds
    }
    # The count of containing filters strictly increases toward narrower
    # filters, yielding a topological broad-to-strict walk from the JSON graph.
    order = sorted(
        thresholds,
        key=lambda name: (
            len(containers[name]),
            len(thresholds[name].get("filters", [])),
            -float(thresholds[name]["cutoff"]),
            name,
        ),
    )
    return order, edges, containers


def _reference_ring(path: Path) -> tuple[np.ndarray, str]:
    atoms = parse_pdb(path.read_text(encoding="utf-8"))
    if len(atoms) == 0:
        raise ValueError(f"empty reference ring PDB: {path}")
    resid = int(atoms[0]["resid"])
    return residue_ring_coordinates(atoms, resid, structure_label=str(path))


def _ring_reference_paths(root: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read A10 PDB geometry and anchor's exact cache-reference arrays."""
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for code in ("A", "C", "PHE", "HIS", "ARG", "TRP"):
        candidates = (_MODULE_DIR, _MODULE_DIR / "ring-refe", root / "ring-refe")
        pdb_path = next(
            (directory / f"refe-{code}.pdb" for directory in candidates
             if (directory / f"refe-{code}.pdb").is_file()),
            root / "ring-refe" / f"refe-{code}.pdb",
        )
        npy_path = next(
            (directory / f"refe-{code}.npy" for directory in candidates
             if (directory / f"refe-{code}.npy").is_file()),
            root / "ring-refe" / f"refe-{code}.npy",
        )
        score_reference, actual_code = _reference_ring(pdb_path)
        if actual_code != code:
            raise ValueError(
                f"{pdb_path}: expected reference residue {code}, got {actual_code}"
            )
        cache_reference = _load_reference_coordinates(npy_path)
        if score_reference.shape != cache_reference.shape:
            raise ValueError(f"ring reference PDB/NPY shape mismatch for {code}")
        result[code] = score_reference, cache_reference
    return result


def _nucleotide_reference_code(base: str) -> str:
    if base in {"A", "G"}:
        return "A"
    if base in {"C", "U"}:
        return "C"
    raise ValueError(f"unsupported nucleotide base {base!r}")


def _protein_reference_code(resname: str) -> str:
    if resname == "TYR":
        return "PHE"
    if resname not in {"PHE", "HIS", "ARG", "TRP"}:
        raise ValueError(f"unsupported protein ring {resname!r}")
    return resname


def _range_number(value: Any) -> int:
    if isinstance(value, str) and value.startswith("range-"):
        value = value.removeprefix("range-")
    number = int(value)
    if number < 2:
        raise ValueError("Gaussian jobs must refer to protein range 2 or later")
    return number


def _protein_path(proteins: Path, protein_range: int) -> Path:
    candidates = (
        proteins / f"range-{protein_range}-aa.pdb",
        proteins / f"range-{protein_range}.pdb",
        proteins / f"range-{protein_range}",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"protein PDB for range-{protein_range} not found under {proteins}"
    )


def _bestfit_path(best_fit: Path, fragment: int, sequence: str) -> Path:
    name = f"frag-{fragment}-{sequence}-bestfit.pdb"
    for candidate in (best_fit / "pdbs" / name, best_fit / name):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"best-fit PDB not found: {name} under {best_fit}")


def _read_exclude_codes(value: str | list[str] | tuple[str, ...]) -> list[str]:
    values = [value] if isinstance(value, str) else list(value)
    result = sorted({str(code).strip().lower() for code in values if str(code).strip()})
    if not result:
        raise ValueError("exclude_pdb_code must identify at least one library exclusion")
    return result


def _validate_library_identity(value: str | dict[str, Any]) -> None:
    identity = json.loads(value) if isinstance(value, str) else value
    if (
        not isinstance(identity, dict)
        or identity.get("schema") != 1
        or identity.get("fraglen") != 2
    ):
        raise ValueError("fraglib_identities must be a schema-1 dinucleotide identity")
    checksums = identity.get("sha256")
    if not isinstance(checksums, dict) or not checksums:
        raise ValueError("fraglib_identities must include library checksums")


def _job_indices(job: dict[str, Any]) -> tuple[int, int, np.ndarray]:
    conformer = int(job["conformer"])
    rotamer = int(job["rotamer"])
    if conformer <= 0 or rotamer <= 0:
        raise ValueError("conformer and rotamer indices are one-based positive integers")
    grid = np.asarray(job["grid"], dtype=np.int64)
    if grid.shape != (3,):
        raise ValueError("grid must contain exactly three integer coordinates")
    return conformer - 1, rotamer - 1, grid


def _rotation_to_reference(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    from anchor_precalc import _superimpose_rotation

    return _superimpose_rotation(source, target)


def _descriptor_values(
    rotation: np.ndarray, translation: np.ndarray
) -> tuple[float, float, float]:
    distance = float(np.linalg.norm(translation))
    if distance == 0.0:
        return distance, float("nan"), float("nan")
    phi = float(np.arccos(np.clip(abs(float(translation[2])) / distance, 0.0, 1.0)))
    theta = float(np.arccos(np.clip(abs(float(rotation[2, 2])), 0.0, 1.0)))
    return distance, phi, theta


def _prepare_jobs(
    jobs: list[dict[str, Any]],
    best_fit: Path,
    proteins: Path,
    exclude_codes: list[str],
    ring_references: dict[str, tuple[np.ndarray, np.ndarray]],
    model: Model,
) -> list[dict[str, Any]]:
    fraglib_config = os.environ.get("fraglib_config")
    if not fraglib_config and Path("fraglib.yaml").is_file():
        fraglib_config = "fraglib.yaml"
    if fraglib_config:
        library._FRAGLIB_YAML_PATH = Path(fraglib_config).expanduser().resolve()
    template_names = [
        *(f"{base}-ppdb.npy" for base in "ACGU"),
        *(f"{first}{second}-ppdb.npy" for first in "ACGU" for second in "ACGU"),
    ]
    for template_dir in (
        _MODULE_DIR,
        Path.cwd(),
        Path(__file__).resolve().parent.parent / "fraglib" / "templates",
    ):
        if all((template_dir / name).is_file() for name in template_names):
            library._fraglib_config()["templates_dir"] = str(template_dir.resolve())
            break
    prepared: list[dict[str, Any]] = []
    by_sequence: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, job in enumerate(jobs):
        prepared.append({"job": job})
        if not isinstance(job, dict):
            prepared[index] = {"job": job, "error": "ValueError: job must be an object"}
            continue
        try:
            sequence = str(job["sequence"]).upper()
            if len(sequence) != 2 or set(sequence) - set("ACGU"):
                raise ValueError(f"invalid dinucleotide sequence {sequence!r}")
        except Exception as exc:
            prepared[index] = {
                "job": job,
                "error": f"{type(exc).__name__}: {exc}",
            }
            continue
        by_sequence.setdefault(sequence, []).append((index, job))

    factories, templates = library.config(
        verify_checksums=False,
        sequences=tuple(by_sequence),
    )
    protein_cache: dict[Path, np.ndarray] = {}
    protein_geometry_cache: dict[
        tuple[Path, int],
        tuple[np.ndarray, str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ] = {}
    for sequence, sequence_jobs in by_sequence.items():
        factory = factories[sequence]
        pose_library = factory.create(
            pdb_code=exclude_codes,
            with_rotaconformers=False,
        )
        _attach_lazy_rotamers(factory, pose_library)
        template = templates[sequence]
        sequence_from_template, nucleotide_indices = ppdb2nucseq(
            template, rna=True, return_index=True
        )
        if sequence_from_template != sequence:
            raise ValueError(
                f"fragment library template mismatch: expected {sequence}, got {sequence_from_template}"
            )

        try:
            for index, job in sequence_jobs:
                try:
                    key = job.get("key")
                    fragment = int(job["fragment"])
                    nucleotide = str(job["nucleotide"]).lower()
                    if nucleotide not in {"first", "second"}:
                        raise ValueError(f"invalid nucleotide selector {nucleotide!r}")
                    protein_range = _range_number(
                        job.get("range", job.get("protein_range"))
                    )
                    protein_resid = int(job["protein_resid"])
                    conformer_index, rotamer_index, grid = _job_indices(job)
                    if conformer_index >= len(pose_library.coordinates):
                        raise ValueError(
                            f"conformer {conformer_index + 1} is outside the library"
                        )
                    if (
                        pose_library.conformer_mask is not None
                        and not pose_library.conformer_mask[conformer_index]
                    ):
                        raise ValueError(
                            f"conformer {conformer_index + 1} is excluded by the pinned PDB code"
                        )
                    rotamers = pose_library.get_rotamers(conformer_index)
                    if rotamer_index >= len(rotamers):
                        raise ValueError(
                            f"rotamer {rotamer_index + 1} is outside the conformer library"
                        )
                    rotamer_matrix = _rotamers_to_matrices(
                        rotamers[rotamer_index : rotamer_index + 1]
                    )[0]

                    nucleotide_index = 0 if nucleotide == "first" else 1
                    atom_start, atom_end = nucleotide_indices[nucleotide_index]
                    local_ring_mask, _ = residue_ring_atom_mask(
                        template[atom_start:atom_end]
                    )
                    full_ring_mask = np.zeros(len(template), dtype=bool)
                    full_ring_mask[atom_start:atom_end] = local_ring_mask
                    ring_coordinates = pose_library.coordinates[
                        conformer_index, full_ring_mask
                    ]

                    bestfit_file = _bestfit_path(best_fit, fragment, sequence)
                    bestfit_atoms = parse_pdb(bestfit_file.read_text(encoding="utf-8"))
                    if np.any(bestfit_atoms["model"] == 1):
                        bestfit_atoms = bestfit_atoms[bestfit_atoms["model"] == 1]
                    rebuilt = (
                        pose_library.coordinates[conformer_index] @ rotamer_matrix
                        + grid[None, :] * GRID_SPACING
                    )
                    if len(bestfit_atoms) != len(rebuilt):
                        raise ValueError(
                            f"best-fit PDB atom count {len(bestfit_atoms)} does not match rebuilt pose {len(rebuilt)}"
                        )
                    for field in ("name", "resname", "resid"):
                        if not np.array_equal(
                            bestfit_atoms[field], pose_library.template[field]
                        ):
                            raise ValueError(
                                f"best-fit PDB atom identity differs at {field}"
                            )
                    bestfit_coordinates = np.column_stack(
                        (
                            bestfit_atoms["x"],
                            bestfit_atoms["y"],
                            bestfit_atoms["z"],
                        )
                    ).astype(np.float64)
                    rebuild_rmsd = float(
                        np.sqrt(
                            np.mean(
                                np.sum((rebuilt - bestfit_coordinates) ** 2, axis=1)
                            )
                        )
                    )
                    if rebuild_rmsd > 0.002:
                        raise ValueError(
                            f"reconstructed pose RMSD {rebuild_rmsd:.6f} Å exceeds 0.002 Å"
                        )

                    # T28 scores the published PDB, so A9 uses the same rounded
                    # geometry for score parity. The cell lookup below uses library
                    # coordinates and the exact reference NPYs used by anchor.py.
                    rna_resid = 1 if nucleotide == "first" else 2
                    rna_ring, _ = residue_ring_coordinates(
                        bestfit_atoms,
                        rna_resid,
                        structure_label=str(bestfit_file),
                    )
                    nuc_code = _nucleotide_reference_code(
                        sequence[nucleotide_index]
                    )
                    nuc_score_reference, nuc_cache_reference = ring_references[nuc_code]
                    conformer_to_cache_reference = _rotation_to_reference(
                        ring_coordinates, nuc_cache_reference
                    )
                    nuc_to_reference = _rotation_to_reference(
                        rna_ring, nuc_score_reference
                    )

                    protein_path = _protein_path(proteins, protein_range)
                    protein_path_key = protein_path.resolve()
                    protein_atoms = protein_cache.get(protein_path_key)
                    if protein_atoms is None:
                        protein_atoms = parse_pdb(
                            protein_path.read_text(encoding="utf-8")
                        )
                        if np.any(protein_atoms["model"] == 1):
                            protein_atoms = protein_atoms[protein_atoms["model"] == 1]
                        protein_cache[protein_path_key] = protein_atoms
                    protein_geometry_key = (protein_path_key, protein_resid)
                    protein_geometry = protein_geometry_cache.get(
                        protein_geometry_key
                    )
                    if protein_geometry is None:
                        protein_ring, protein_resname = residue_ring_coordinates(
                            protein_atoms,
                            protein_resid,
                            structure_label=str(protein_path),
                        )
                        protein_code = _protein_reference_code(protein_resname)
                        protein_score_reference, protein_cache_reference = (
                            ring_references[protein_code]
                        )
                        protein_to_world = _rotation_to_reference(
                            protein_score_reference, protein_ring
                        )
                        protein_center = protein_ring.mean(axis=0)
                        protein_geometry = (
                            protein_ring,
                            protein_resname,
                            protein_code,
                            protein_score_reference,
                            protein_cache_reference,
                            protein_to_world,
                            protein_center,
                        )
                        protein_geometry_cache[protein_geometry_key] = protein_geometry
                    (
                        protein_ring,
                        protein_resname,
                        protein_code,
                        protein_score_reference,
                        protein_cache_reference,
                        protein_to_world,
                        protein_center,
                    ) = protein_geometry
                    actual_rotation = nuc_to_reference.T @ protein_to_world.T
                    actual_translation = (
                        rna_ring.mean(axis=0) - protein_center
                    ) @ protein_to_world.T
                    distance, phi, theta = _descriptor_values(
                        actual_rotation, actual_translation
                    )
                    value = float(
                        gaussian_score(distance, phi, theta, model=model)
                    )
                    prepared[index] = {
                        "prepared": _PreparedJob(
                            key=key,
                            fragment=fragment,
                            nucleotide=nucleotide,
                            sequence=sequence,
                            protein_range=protein_range,
                            protein_resid=protein_resid,
                            grid=grid,
                            ring_coordinates=ring_coordinates,
                            conformer_to_cache_reference=conformer_to_cache_reference,
                            rotamer_matrix=rotamer_matrix,
                            protein_path=protein_path,
                            protein_ring=protein_ring,
                            protein_center=protein_center,
                            nucleotide_cache_reference=nuc_cache_reference,
                            protein_cache_reference=protein_cache_reference,
                            nucleotide_score_reference=nuc_score_reference,
                            protein_score_reference=protein_score_reference,
                            score=value,
                            distance=distance,
                            phi=phi,
                            theta=theta,
                            geometry_key=(
                                str(protein_path_key),
                                protein_resid,
                                protein_code,
                            ),
                        )
                    }
                except Exception as exc:
                    prepared[index] = {"job": job, "error": f"{type(exc).__name__}: {exc}"}

        finally:
            pose_library = None

    return prepared


def _pose_pass(
    job: _PreparedJob,
    name: str,
    entry: dict[str, Any],
    spec: dict[str, Any],
) -> bool:
    mask = spec["mask"]
    if not job.distance < float(mask["distance_lt_A"]):
        return False
    if not job.phi < math.radians(float(mask["displacement_lt_deg"])):
        return False
    for filter_name in entry.get("filters", []):
        cutoff_degrees = float(spec["filters"][filter_name]["lt_deg"])
        value = job.theta if filter_name == "tilt30" else job.phi
        if not value < math.radians(cutoff_degrees):
            return False
    return job.score >= float(entry["cutoff"])


def _cache_geometry_key(
    job: _PreparedJob, lattice_step: float
) -> tuple[Any, ...]:
    return (*job.geometry_key, float(lattice_step))


def _run_gaussian_check(
    jobs: list[dict[str, Any]],
    best_fit: str | Path,
    proteins: str | Path,
    exclude_pdb_code: str | list[str] | tuple[str, ...],
    fraglib_identities: str | dict[str, Any],
    repo_root: str | Path | None = None,
    *,
    semantic_identity: str | dict[str, Any] | None = None,
    verification_executor: ThreadPoolExecutor | None = None,
) -> dict[str, Any]:
    """Run A9 once across every job and every filter in the installed A7 spec."""
    _validate_library_identity(fraglib_identities)
    root = _repo_root(repo_root)
    best_fit = Path(best_fit).resolve()
    proteins = Path(proteins).resolve()
    if not best_fit.is_dir():
        raise FileNotFoundError(f"best-fit folder does not exist: {best_fit}")
    if not proteins.is_dir():
        raise FileNotFoundError(f"protein range folder does not exist: {proteins}")

    spec = load_spec(_spec_path(root))
    _validate_semantic_identity(semantic_identity, spec)
    thresholds = spec.get("thresholds")
    if not isinstance(thresholds, dict) or len(thresholds) != 12:
        raise ValueError("Gaussian specification must contain all twelve filter names")
    model_data = spec["model"]
    model = Model(
        mean=tuple(float(x) for x in model_data["mean"]),
        covariance=tuple(
            tuple(float(x) for x in row) for row in model_data["covariance"]
        ),
        precision=tuple(
            tuple(float(x) for x in row) for row in model_data["precision"]
        ),
        normalizer=float(model_data["normalizer"]),
    )
    ring_references = _ring_reference_paths(root)
    exclude_codes = _read_exclude_codes(exclude_pdb_code)
    order, containment, containers = _filter_order(spec)

    if not jobs:
        return {
            "schema": 1,
            "filter_set_version": spec["filter_set_version"],
            "min_cell_fraction": MIN_CELL_FRACTION,
            "jobs": [],
        }

    prepared = _prepare_jobs(
        jobs,
        best_fit,
        proteins,
        exclude_codes,
        ring_references,
        model,
    )
    output_jobs: list[dict[str, Any]] = []
    for job, entry in zip(jobs, prepared):
        if isinstance(job, dict):
            try:
                fragment = int(job.get("fragment"))
            except (TypeError, ValueError):
                fragment = None
            result = {
                "key": job.get("key"),
                "fragment": fragment,
                "nucleotide": str(job.get("nucleotide", "")).lower(),
                "filters": {},
            }
        else:
            result = {"key": None, "fragment": None, "nucleotide": "", "filters": {}}
        if "error" in entry:
            result["error"] = entry["error"]
        output_jobs.append(result)

    geometry_cache: dict[
        tuple[Any, ...], tuple[np.ndarray, np.ndarray | None, float]
    ] = {}
    phase_capacity_caches: dict[
        tuple[Any, ...], dict[tuple[int, int, int], int]
    ] = {}
    for name in order:
        eligible_indices = [
            index
            for index, entry in enumerate(prepared)
            if "prepared" in entry
            and not any(
                container in output_jobs[index]["filters"]
                and not output_jobs[index]["filters"][container]["emitted"]
                for container in containers[name]
            )
        ]
        if not eligible_indices:
            continue

        # Verification decides which exact paths may be opened. In particular,
        # a corrupt preferred .npy is a hard failure in verify_filter_cache.
        verified_paths = verify_filter_cache(
            spec, name, root, executor=verification_executor
        )
        arrays = {
            filename: load_npy(path)
            for filename, path in verified_paths.items()
        }
        validated_arrays = prepare_precalculated_arrays(arrays)
        for index in eligible_indices:
            prepared_job: _PreparedJob = prepared[index]["prepared"]
            cache_key = _cache_geometry_key(
                prepared_job, validated_arrays.lattice_step
            )
            shared_geometry = geometry_cache.get(cache_key)
            precalc = build_precalculated_anchor(
                validated_arrays,
                prepared_job.nucleotide_cache_reference,
                prepared_job.protein_cache_reference,
                prepared_job.protein_ring,
                shared_geometry=shared_geometry,
                build_capacity=False,
            )
            if shared_geometry is None:
                geometry_cache[cache_key] = (
                    precalc.protein_reference_to_world,
                    precalc.capacity_table,
                    validated_arrays.lattice_step,
                )
            phase_capacity_cache = phase_capacity_caches.setdefault(cache_key, {})
            count, capacity = cell_fill(
                precalc,
                prepared_job.ring_coordinates,
                prepared_job.rotamer_matrix,
                prepared_job.protein_center,
                prepared_job.grid,
                capacity_cache=phase_capacity_cache,
                conformer_to_reference=prepared_job.conformer_to_cache_reference,
            )
            if capacity <= 0:
                raise ValueError(f"{name}: pose cell has zero lattice capacity")
            fill = count / capacity
            emitted = fill >= MIN_CELL_FRACTION
            output_jobs[index]["filters"][name] = {
                "count": int(count),
                "capacity": int(capacity),
                "fill": float(fill),
                "emitted": bool(emitted),
                "pose_pass": bool(
                    _pose_pass(prepared_job, name, thresholds[name], spec)
                ),
                "score": float(prepared_job.score),
            }
        if eligible_indices:
            del precalc, prepared_job
        del arrays, validated_arrays

    # Branch-implied rejections contain no fabricated measurements.
    for output_job in output_jobs:
        filters = output_job["filters"]
        for name in order:
            if name in filters:
                continue
            failed = [
                broader
                for broader, result in filters.items()
                if (name, broader) in containment and not result["emitted"]
            ]
            if not failed:
                continue
            implied_by = max(
                failed,
                key=lambda broader: (len(containers[broader]), broader),
            )
            filters[name] = {"emitted": False, "implied_by": implied_by}

    for output_job, entry in zip(output_jobs, prepared):
        if "error" in entry:
            continue
        if len(output_job["filters"]) != len(thresholds):
            missing = sorted(set(thresholds) - set(output_job["filters"]))
            raise RuntimeError(
                "A9 left filters unevaluated without a containing rejection: "
                f"{missing}"
            )

    return {
        "schema": 1,
        "filter_set_version": spec["filter_set_version"],
        "min_cell_fraction": MIN_CELL_FRACTION,
        "jobs": output_jobs,
    }


def run_gaussian_check(
    jobs: list[dict[str, Any]],
    best_fit: str | Path,
    proteins: str | Path,
    exclude_pdb_code: str | list[str] | tuple[str, ...],
    fraglib_identities: str | dict[str, Any],
    repo_root: str | Path | None = None,
    *,
    semantic_identity: str | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run A9 with one reusable pool for independent cache verification.

    Seamless callers pass ``semantic_identity`` as an explicit input so its
    transformation checksum depends on ``spec_digest`` and decompressed array
    identities, not the JSON's compression-only SHA-256 fields. The complete
    JSON is still loaded here and checked before any result is produced.
    """
    with ThreadPoolExecutor(max_workers=4) as executor:
        return _run_gaussian_check(
            jobs,
            best_fit,
            proteins,
            exclude_pdb_code,
            fraglib_identities,
            repo_root,
            semantic_identity=semantic_identity,
            verification_executor=executor,
        )


def _identity_argument(value: str) -> str:
    """Read identities from a JSON file or retain their literal JSON value."""
    path = Path(value)
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8")
    except OSError:
        pass
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gaussian-check",
        description="Check one T20 job list against all installed Gaussian filters.",
    )
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--best-fit", type=Path, required=True)
    parser.add_argument("--proteins", type=Path, required=True)
    parser.add_argument("--exclude-code", required=True)
    parser.add_argument("--fraglib-identities", required=True)
    parser.add_argument(
        "--semantic-identity",
        required=True,
        help=(
            "Canonical A9 identity JSON (spec_digest and npy_sha256 values only); "
            "pass it as a separate transform input, not through the A7 JSON pin."
        ),
    )
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    jobs = json.loads(args.jobs.read_text(encoding="utf-8"))
    if not isinstance(jobs, list):
        raise ValueError("--jobs JSON must contain an array")
    result = run_gaussian_check(
        jobs,
        args.best_fit,
        args.proteins,
        args.exclude_code,
        _identity_argument(args.fraglib_identities),
        args.repo_root or os.environ.get("repo_root"),
        semantic_identity=_identity_argument(args.semantic_identity),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
