"""Shared Gaussian-anchor precalculation and pose-cell lookup."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from npy_io import find_npy, load_npy
from stack_geometry import as_coordinates

GRID_SPACING = sqrt(3) / 3

HOPF_N = 16

HOPF_NPSI = 64

HOPF_CELLS = 6 * HOPF_N * HOPF_N * HOPF_NPSI


@dataclass(frozen=True)
class _PrecalculatedAnchor:
    rotations: np.ndarray
    translations: np.ndarray
    filtered_sizes: np.ndarray
    filtered_offsets: np.ndarray
    filtered_concat: np.ndarray
    # HOPF_CELLS entries: cache row for each Hopf cell, -1 where the cell was
    # not retained (nothing passes the filter anywhere in that cell).
    cell_to_row: np.ndarray
    nucleotide_reference: np.ndarray
    protein_reference_to_world: np.ndarray
    translation_vectors_world: np.ndarray
    lattice_origin: np.ndarray
    lattice_step: float
    # Lattice points per pose cell, indexed by the cell's phase; see
    # _build_capacity_table.  Turns the keep rule into a fraction of capacity.
    capacity_table: np.ndarray | None


@dataclass(frozen=True)
class _ValidatedPrecalculatedArrays:
    rotations: np.ndarray
    translations: np.ndarray
    filtered_sizes: np.ndarray
    filtered_offsets: np.ndarray
    filtered_concat: np.ndarray
    cell_to_row: np.ndarray
    lattice_origin: np.ndarray
    lattice_step: float


def _find_precalculated_array(directory: Path, name: str) -> Path:
    path = find_npy(directory / f"{name}.npy")
    if path is None:
        raise ValueError(
            f"missing {name}.npy or {name}.npy.zst in precalculation directory {directory}"
        )
    return path

def _resolve_precalculated_directory(directory: str | Path) -> Path:
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"precalculation directory does not exist: {directory}")
    return directory

def _load_reference_coordinates(path: str | Path) -> np.ndarray:
    path = Path(path)
    coordinates = load_npy(path)
    if coordinates.dtype.names is not None:
        required = {"x", "y", "z"}
        if not required.issubset(coordinates.dtype.names):
            raise ValueError(f"structured reference array lacks x/y/z fields: {path}")
        coordinates = as_coordinates(coordinates)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(f"reference ring must have shape [N,3]: {path}")
    if len(coordinates) < 3 or not np.all(np.isfinite(coordinates)):
        raise ValueError(f"reference ring contains invalid coordinates: {path}")
    return coordinates

def _superimpose_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the row-vector rotation for centered source onto centered target."""
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(
            f"rings to superimpose must have matching [N,3] shapes, got {source.shape} and {target.shape}"
        )
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    if np.linalg.matrix_rank(source_centered, tol=1e-7) < 2:
        raise ValueError("source ring coordinates are degenerate")
    if np.linalg.matrix_rank(target_centered, tol=1e-7) < 2:
        raise ValueError("target ring coordinates are degenerate")
    v, _, wt = np.linalg.svd(source_centered.T @ target_centered)
    if np.linalg.det(v) * np.linalg.det(wt) < 0:
        v[:, -1] *= -1
    return v @ wt

def _validate_precalculated_arrays(
    rotations: np.ndarray,
    translations: np.ndarray,
    filtered_sizes: np.ndarray,
    filtered_concat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rotations = np.asarray(rotations)
    rotations = _rotamers_to_matrices(rotations).astype(np.float64, copy=False)
    if len(rotations) == 0:
        raise ValueError("precalculated rotations must not be empty")
    if not np.all(np.isfinite(rotations)):
        raise ValueError("precalculated rotations contain non-finite values")
    identity = np.eye(3)
    if not np.allclose(np.swapaxes(rotations, 1, 2) @ rotations, identity, atol=1e-5):
        raise ValueError("precalculated rotations are not orthonormal")
    if not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-5):
        raise ValueError("precalculated rotations must have determinant +1")

    translations = np.asarray(translations, dtype=np.float64)
    if translations.ndim != 2 or translations.shape[1] != 3:
        raise ValueError("precalculated translations must have shape [M,3]")
    if len(translations) == 0:
        raise ValueError("precalculated translations must not be empty")
    if not np.all(np.isfinite(translations)):
        raise ValueError("precalculated translations contain non-finite values")

    filtered_sizes = np.asarray(filtered_sizes)
    if filtered_sizes.ndim != 1 or len(filtered_sizes) != len(rotations):
        raise ValueError("filtered-sizes must be a 1D array with one count per rotation")
    if not np.issubdtype(filtered_sizes.dtype, np.integer):
        raise ValueError("filtered-sizes must contain integers")
    if filtered_sizes.size and filtered_sizes.max() > np.iinfo(np.int32).max:
        raise ValueError("filtered-sizes contains a count outside int32 range")
    filtered_sizes = filtered_sizes.astype(np.int32, copy=False)
    if filtered_sizes.size and filtered_sizes.min() < 0:
        raise ValueError("filtered-sizes contains a negative count")

    filtered_concat = np.asarray(filtered_concat)
    if filtered_concat.ndim != 1 or not np.issubdtype(
        filtered_concat.dtype, np.integer
    ):
        raise ValueError("filtered-concat must be a 1D integer array")
    if filtered_concat.size and filtered_concat.max() > np.iinfo(np.int32).max:
        raise ValueError("filtered-concat contains an index outside int32 range")
    filtered_concat = filtered_concat.astype(np.int32, copy=False)
    expected = int(filtered_sizes.sum(dtype=np.int64))
    if expected != len(filtered_concat):
        raise ValueError(
            f"sum(filtered-sizes)={expected} does not equal len(filtered-concat)={len(filtered_concat)}"
        )
    if filtered_concat.size and (
        filtered_concat.min() < 0 or filtered_concat.max() >= len(translations)
    ):
        raise ValueError("filtered-concat contains an out-of-range translation index")
    offsets = np.empty(len(filtered_sizes) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(filtered_sizes, out=offsets[1:])
    return rotations, translations, filtered_sizes, offsets, filtered_concat

def _hopf_cell_axes(face: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Cube-face direction for Hopf base cells (face, p, q)."""
    a = np.tan((np.pi / 4) * (2 * (p + 0.5) / HOPF_N - 1))
    b = np.tan((np.pi / 4) * (2 * (q + 0.5) / HOPF_N - 1))
    ones = np.ones_like(a)
    axis = face // 2
    sign = np.where(face % 2 == 0, 1.0, -1.0)
    directions = np.empty(a.shape + (3,), dtype=np.float64)
    for value, (major, first, second) in enumerate(((0, 1, 2), (1, 0, 2), (2, 0, 1))):
        selected = axis == value
        directions[selected, major] = (sign * ones)[selected]
        directions[selected, first] = a[selected]
        directions[selected, second] = b[selected]
    return directions / np.linalg.norm(directions, axis=1)[:, None]

def _hopf_cell_matrices(cells: np.ndarray) -> np.ndarray:
    """Rotation matrix at the centre of each Hopf cell index."""
    cells = np.asarray(cells, dtype=np.int64)
    face, rest = np.divmod(cells, HOPF_N * HOPF_N * HOPF_NPSI)
    p, rest = np.divmod(rest, HOPF_N * HOPF_NPSI)
    q, m = np.divmod(rest, HOPF_NPSI)
    axes = _hopf_cell_axes(face, p, q)
    theta_s = np.arccos(np.clip(axes[:, 2], -1.0, 1.0))
    phi_s = np.arctan2(axes[:, 1], axes[:, 0])
    psi = 2 * np.pi * (m + 0.5) / HOPF_NPSI
    # scipy quaternion order is (x, y, z, w); the generator lists (w, x, y, z).
    quaternions = np.column_stack(
        (
            np.cos(theta_s / 2) * np.sin(psi / 2),
            np.sin(theta_s / 2) * np.cos(phi_s + psi / 2),
            np.sin(theta_s / 2) * np.sin(phi_s + psi / 2),
            np.cos(theta_s / 2) * np.cos(psi / 2),
        )
    )
    return Rotation.from_quat(quaternions).as_matrix()

def _hopf_cell_indices(rotations: np.ndarray) -> np.ndarray:
    """Hopf cell index of each rotation matrix; inverse of _hopf_cell_matrices."""
    quaternions = Rotation.from_matrix(rotations).as_quat()
    x, y, z, w = (quaternions[:, i] for i in range(4))
    # q and -q denote the same rotation and shift both half-angles by pi, so the
    # fibre coordinate psi (mod 2pi) and the base coordinate phi_s are unchanged.
    psi_half = np.arctan2(x, w)
    psi = np.mod(2 * psi_half, 2 * np.pi)
    phi_s = np.arctan2(z, y) - psi_half
    theta_s = 2 * np.arctan2(np.hypot(y, z), np.hypot(w, x))
    sin_theta = np.sin(theta_s)
    axes = np.column_stack(
        (sin_theta * np.cos(phi_s), sin_theta * np.sin(phi_s), np.cos(theta_s))
    )
    major = np.argmax(np.abs(axes), axis=1)
    rows = np.arange(len(axes))
    denominator = np.abs(axes[rows, major])
    face = 2 * major + (axes[rows, major] < 0)
    minor = np.array([[1, 2], [0, 2], [0, 1]])[major]
    a = axes[rows, minor[:, 0]] / denominator
    b = axes[rows, minor[:, 1]] / denominator
    p = np.floor(HOPF_N * (1 + (4 / np.pi) * np.arctan(a)) / 2).astype(np.int64)
    q = np.floor(HOPF_N * (1 + (4 / np.pi) * np.arctan(b)) / 2).astype(np.int64)
    m = np.floor(HOPF_NPSI * psi / (2 * np.pi)).astype(np.int64)
    np.clip(p, 0, HOPF_N - 1, out=p)
    np.clip(q, 0, HOPF_N - 1, out=q)
    np.clip(m, 0, HOPF_NPSI - 1, out=m)
    return ((face * HOPF_N + p) * HOPF_N + q) * HOPF_NPSI + m

def _build_cell_lookup(rotations: np.ndarray) -> np.ndarray:
    """Map every Hopf cell to its cache row, or -1 when the cell was dropped."""
    cells = _hopf_cell_indices(rotations)
    if len(np.unique(cells)) != len(cells):
        raise ValueError(
            "precalculated rotations do not sit on distinct Hopf cells; the cache "
            f"was not built on the assumed {HOPF_CELLS}-cell grid"
        )
    # float32 round-tripping in _rotamers_to_matrices costs ~1e-7; landing on the
    # wrong cell would cost ~1e-1, so any tolerance between the two separates them.
    residual = np.abs(_hopf_cell_matrices(cells) - rotations).max()
    if residual > 1e-5:
        raise ValueError(
            "precalculated rotations are not Hopf cell centres of the assumed "
            f"{HOPF_CELLS}-cell grid (max deviation {residual:.3e})"
        )
    cell_to_row = np.full(HOPF_CELLS, -1, dtype=np.int32)
    cell_to_row[cells] = np.arange(len(cells), dtype=np.int32)
    return cell_to_row

def _lattice_geometry(translations: np.ndarray) -> tuple[np.ndarray, float]:
    """Infer the generator's translation lattice (origin, step) from its samples."""
    steps = []
    for axis in range(3):
        values = np.unique(np.round(translations[:, axis], 9))
        if len(values) < 2:
            raise ValueError("precalculated translations do not span a 3D lattice")
        steps.append(np.diff(values).min())
    step = float(min(steps))
    if step <= 0:
        raise ValueError("precalculated translations have a non-positive lattice step")
    origin = translations.min(axis=0)
    residual = np.abs(
        (translations - origin) / step - np.rint((translations - origin) / step)
    ).max()
    if residual > 1e-6:
        raise ValueError(
            f"precalculated translations are not on a regular {step} lattice "
            f"(max deviation {residual:.3e} of a step)"
        )
    return origin, step

def _build_capacity_table(
    step: float,
    rotation: np.ndarray,
    resolution: int = 64,
) -> np.ndarray:
    """Lattice points per pose cell, as a function of the cell's lattice phase.

    A pose cell is an axis-aligned cube of side GRID_SPACING in world space, so in
    lattice units it is a cube of side GRID_SPACING/step rotated by `rotation`.
    How many lattice points it holds depends only on where its centre sits inside
    a lattice cell, because the lattice is periodic and the rotation is fixed for
    a whole run.  Tabulating that over one lattice cell makes the count O(1).
    """
    half = GRID_SPACING / (2 * step)
    reach = int(np.ceil(half * np.sqrt(3))) + 1
    axis = np.arange(-reach, reach + 1)
    neighbourhood = np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    grid = (np.arange(resolution) + 0.5) / resolution
    phases = np.stack(
        np.meshgrid(grid, grid, grid, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    table = np.empty(len(phases), dtype=np.int16)
    chunk = max(1, 2_000_000 // len(neighbourhood))
    for start in range(0, len(phases), chunk):
        block = phases[start : start + chunk]
        delta = neighbourhood[None, :, :] - block[:, None, :]
        table[start : start + chunk] = np.count_nonzero(
            np.all(np.abs(delta @ rotation) <= half, axis=2), axis=1
        )
    return table.reshape(resolution, resolution, resolution)

def _capacity_for_phase_bin(
    phase_bin: tuple[int, int, int],
    precalculated: _PrecalculatedAnchor,
    resolution: int = 64,
) -> int:
    """Count lattice points for one quantized phase bin."""
    half = GRID_SPACING / (2 * precalculated.lattice_step)
    reach = int(np.ceil(half * np.sqrt(3))) + 1
    axis = np.arange(-reach, reach + 1)
    neighbourhood = np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    centre_phase = (np.asarray(phase_bin, dtype=np.float64) + 0.5) / resolution
    delta = neighbourhood - centre_phase
    return int(
        np.count_nonzero(
            np.all(
                np.abs(delta @ precalculated.protein_reference_to_world) <= half,
                axis=1,
            )
        )
    )


def _pose_cell_capacity(
    grid: np.ndarray,
    offsets: np.ndarray,
    precalculated: _PrecalculatedAnchor,
    *,
    phase_capacity_cache: dict[tuple[int, int, int], int] | None = None,
) -> np.ndarray:
    """Lattice points that can land in each pose cell (its capacity)."""
    centres = (
        (grid * GRID_SPACING - offsets) @ precalculated.protein_reference_to_world.T
    )
    phase = np.mod(
        (centres - precalculated.lattice_origin) / precalculated.lattice_step, 1.0
    )
    if precalculated.capacity_table is None:
        resolution = 64
        index = np.minimum((phase * resolution).astype(np.int64), resolution - 1)
        if phase_capacity_cache is not None:
            result = np.empty(len(index), dtype=np.int16)
            for row, values in enumerate(index):
                key = tuple(int(value) for value in values)
                capacity = phase_capacity_cache.get(key)
                if capacity is None:
                    capacity = _capacity_for_phase_bin(key, precalculated, resolution)
                    phase_capacity_cache[key] = capacity
                result[row] = capacity
            return result
        unique, inverse = np.unique(index, axis=0, return_inverse=True)
        half = GRID_SPACING / (2 * precalculated.lattice_step)
        reach = int(np.ceil(half * np.sqrt(3))) + 1
        axis = np.arange(-reach, reach + 1)
        neighbourhood = np.stack(
            np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
        ).reshape(-1, 3)
        phases = (unique + 0.5) / resolution
        result = np.empty(len(unique), dtype=np.int16)
        for i, centre_phase in enumerate(phases):
            delta = neighbourhood - centre_phase
            result[i] = np.count_nonzero(
                np.all(
                    np.abs(delta @ precalculated.protein_reference_to_world) <= half,
                    axis=1,
                )
            )
        return result[inverse]
    resolution = precalculated.capacity_table.shape[0]
    index = np.minimum((phase * resolution).astype(np.int64), resolution - 1)
    return precalculated.capacity_table[index[:, 0], index[:, 1], index[:, 2]]

def _load_precalculated_anchor(
    directory: str | Path,
    nucleotide_reference_path: str | Path,
    protein_reference_path: str | Path,
    protein_ring: np.ndarray,
    *,
    arrays: dict[str, np.ndarray] | None = None,
) -> _PrecalculatedAnchor:
    if arrays is None:
        directory = _resolve_precalculated_directory(directory)
        arrays = {
            name: load_npy(_find_precalculated_array(directory, name))
            for name in (
                "rotations",
                "translations",
                "filtered-sizes",
                "filtered-concat",
            )
        }
    else:
        missing = {
            "rotations", "translations", "filtered-sizes", "filtered-concat"
        } - set(arrays)
        if missing:
            raise ValueError(f"precalculated arrays missing: {', '.join(sorted(missing))}")
    validated = prepare_precalculated_arrays(arrays)
    nucleotide_reference = _load_reference_coordinates(nucleotide_reference_path)
    protein_reference = _load_reference_coordinates(protein_reference_path)
    return build_precalculated_anchor(
        validated, nucleotide_reference, protein_reference, protein_ring
    )


def prepare_precalculated_arrays(
    arrays: dict[str, np.ndarray],
) -> _ValidatedPrecalculatedArrays:
    """Validate the four verified cache arrays once for reuse across jobs."""
    rotations, translations, sizes, offsets, concat = _validate_precalculated_arrays(
        arrays["rotations"],
        arrays["translations"],
        arrays["filtered-sizes"],
        arrays["filtered-concat"],
    )
    lattice_origin, lattice_step = _lattice_geometry(translations)
    return _ValidatedPrecalculatedArrays(
        rotations=rotations,
        translations=translations,
        filtered_sizes=sizes,
        filtered_offsets=offsets,
        filtered_concat=concat,
        cell_to_row=_build_cell_lookup(rotations),
        lattice_origin=lattice_origin,
        lattice_step=lattice_step,
    )


def build_precalculated_anchor(
    arrays: _ValidatedPrecalculatedArrays,
    nucleotide_reference: np.ndarray,
    protein_reference: np.ndarray,
    protein_ring: np.ndarray,
    *,
    shared_geometry: tuple[np.ndarray, np.ndarray | None, float] | None = None,
    build_capacity: bool = True,
) -> _PrecalculatedAnchor:
    """Bind validated cache arrays to ring geometry, reusing capacity work."""
    nucleotide_reference = np.asarray(nucleotide_reference, dtype=np.float64)
    protein_reference = np.asarray(protein_reference, dtype=np.float64)
    if nucleotide_reference.ndim != 2 or nucleotide_reference.shape[1] != 3:
        raise ValueError("nucleotide reference ring must have shape [N,3]")
    if protein_reference.ndim != 2 or protein_reference.shape[1] != 3:
        raise ValueError("protein reference ring must have shape [N,3]")
    if shared_geometry is None:
        protein_reference_to_world = _superimpose_rotation(
            protein_reference, protein_ring
        )
        capacity_table = (
            _build_capacity_table(arrays.lattice_step, protein_reference_to_world)
            if build_capacity
            else None
        )
    else:
        protein_reference_to_world, capacity_table, capacity_step = shared_geometry
        if not np.isclose(capacity_step, arrays.lattice_step, atol=1e-12):
            raise ValueError("shared capacity geometry uses a different lattice step")
        # Check compatibility instead of silently reusing a capacity table built
        # for a different lattice or protein-ring orientation.
        expected_rotation = _superimpose_rotation(protein_reference, protein_ring)
        if not np.allclose(protein_reference_to_world, expected_rotation, atol=1e-12):
            raise ValueError("shared capacity geometry does not match protein ring")
        if capacity_table is None and build_capacity:
            capacity_table = _build_capacity_table(
                arrays.lattice_step, protein_reference_to_world
            )
    return _PrecalculatedAnchor(
        rotations=arrays.rotations,
        translations=arrays.translations,
        filtered_sizes=arrays.filtered_sizes,
        filtered_offsets=arrays.filtered_offsets,
        filtered_concat=arrays.filtered_concat,
        cell_to_row=arrays.cell_to_row,
        nucleotide_reference=nucleotide_reference,
        protein_reference_to_world=protein_reference_to_world,
        translation_vectors_world=arrays.translations @ protein_reference_to_world,
        lattice_origin=arrays.lattice_origin,
        lattice_step=arrays.lattice_step,
        capacity_table=capacity_table,
    )

def _rotamers_to_matrices(rotamers: np.ndarray) -> np.ndarray:
    if rotamers.ndim == 3 and rotamers.shape[1:] == (3, 3):
        return rotamers.astype(np.float32, copy=False)
    if rotamers.ndim == 2 and rotamers.shape[1] == 3:
        return Rotation.from_rotvec(rotamers).as_matrix().astype(np.float32, copy=False)
    raise ValueError(
        "Unsupported rotamer representation; expected shape [N,3] rotvec or [N,3,3] matrix"
    )

def _snap_cache_rotations(
    rotations: np.ndarray,
    precalculated: _PrecalculatedAnchor,
) -> np.ndarray:
    """Cache row per rotamer, or -1 when its Hopf cell holds no candidates.

    The cache was built by evaluating the filter on the universal Hopf division
    of SO(3), so each rotamer belongs to exactly one cell.  A cell the cache does
    not list is a cell in which nothing passes: the rotamer is out of the region.
    """
    cells = _hopf_cell_indices(rotations)
    return precalculated.cell_to_row[cells].astype(np.int64, copy=False)[:, None]

def _translation_segments(
    cache_rotation_indices: np.ndarray,
    precalculated: _PrecalculatedAnchor,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand cache segments once per unique lookup rotation in this batch."""
    cache_rotation_indices = np.asarray(cache_rotation_indices)
    if cache_rotation_indices.ndim == 1:
        cache_rotation_indices = cache_rotation_indices[:, None]

    segment_by_rotation: dict[int, np.ndarray] = {}
    for cache_index in np.unique(cache_rotation_indices):
        if cache_index < 0:
            continue
        start = int(precalculated.filtered_offsets[cache_index])
        end = int(precalculated.filtered_offsets[cache_index + 1])
        segment_by_rotation[int(cache_index)] = precalculated.filtered_concat[
            start:end
        ]

    segment_by_stencil: dict[tuple[int, ...], np.ndarray] = {}
    segments: list[np.ndarray] = []
    for stencil in cache_rotation_indices:
        key = tuple(int(x) for x in np.unique(stencil) if x >= 0)
        segment = segment_by_stencil.get(key)
        if segment is None:
            parts = [segment_by_rotation[index] for index in key]
            if not parts:
                segment = np.empty(0, dtype=np.int64)
            elif len(parts) == 1:
                segment = parts[0]
            else:
                segment = np.unique(np.concatenate(parts))
            segment_by_stencil[key] = segment
        segments.append(segment)

    sizes = np.fromiter(
        (len(segment) for segment in segments),
        dtype=np.int64,
        count=len(segments),
    )
    if not np.any(sizes):
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int64)
    owners = np.repeat(np.arange(len(segments), dtype=np.int32), sizes)
    translation_indices = np.concatenate(
        [segment for segment in segments if len(segment)]
    )
    return owners, translation_indices

def _deduplicate_owner_grid(
    owners: np.ndarray,
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse candidates onto pose grid points, counting the hits on each.

    Several cache translations round onto the same pose grid point.  The number
    that landed there, over the cell's capacity, is the fraction of the cell that
    lies inside the accepted region -- which is what decides whether to emit the
    pose, without ever consulting the underlying filter.
    """
    if len(owners) == 0:
        return owners, grid, np.empty(0, dtype=np.int64)
    if owners.min() < 0 or owners.max() > np.iinfo(np.uint16).max:
        raise ValueError("candidate owner index exceeds uint16 range")
    if grid.min() < np.iinfo(np.int16).min or grid.max() > np.iinfo(
        np.int16
    ).max:
        raise ValueError("precalculated translations exceed the int16 pose grid")

    shifted = grid.astype(np.int64, copy=False) - np.iinfo(np.int16).min
    keys = (
        (owners.astype(np.uint64) << np.uint64(48))
        | (shifted[:, 0].astype(np.uint64) << np.uint64(32))
        | (shifted[:, 1].astype(np.uint64) << np.uint64(16))
        | shifted[:, 2].astype(np.uint64)
    )
    _, first, counts = np.unique(keys, return_index=True, return_counts=True)
    return owners[first].astype(np.int32, copy=False), grid[first], counts


def candidate_cells(
    precalc: _PrecalculatedAnchor,
    ring_coords: np.ndarray,
    rotamer_matrices: np.ndarray,
    protein_center: np.ndarray,
    *,
    _include_capacity: bool = True,
    conformer_to_reference: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return accepted pose-grid cells for one or more rotamers.

    The return arrays are ``owners``, ``grid``, ``counts`` and ``capacity``.
    Owners index the supplied rotamers; each grid row is one deduplicated pose
    cell, and its fill is ``counts / capacity``.
    """
    ring_coords = np.asarray(ring_coords)
    if ring_coords.ndim != 2 or ring_coords.shape[1] != 3:
        raise ValueError("ring coordinates must have shape [N,3]")
    if ring_coords.shape != precalc.nucleotide_reference.shape:
        raise ValueError(
            "selected nucleotide ring and reference ring have different shapes: "
            f"{ring_coords.shape} versus {precalc.nucleotide_reference.shape}"
        )
    rotamer_matrices = _rotamers_to_matrices(np.asarray(rotamer_matrices))
    if rotamer_matrices.ndim == 2:
        rotamer_matrices = rotamer_matrices[None, :, :]
    if rotamer_matrices.ndim != 3 or rotamer_matrices.shape[1:] != (3, 3):
        raise ValueError("rotamer matrices must have shape [N,3,3]")
    protein_center = np.asarray(protein_center)
    if protein_center.shape != (3,):
        raise ValueError("protein center must have shape [3]")

    if conformer_to_reference is None:
        conformer_to_reference = _superimpose_rotation(
            ring_coords, precalc.nucleotide_reference
        )
    else:
        conformer_to_reference = np.asarray(conformer_to_reference, dtype=np.float64)
        if conformer_to_reference.shape != (3, 3):
            raise ValueError("conformer-to-reference rotation must have shape [3,3]")
    world_to_protein_reference = precalc.protein_reference_to_world.T
    cache_rotations = np.einsum(
        "ij,njk,kl->nil",
        conformer_to_reference.T,
        rotamer_matrices,
        world_to_protein_reference,
    )
    snapped_cache_indices = _snap_cache_rotations(cache_rotations, precalc)
    ring_centers = (ring_coords.mean(axis=0) @ rotamer_matrices).astype(
        np.float32, copy=False
    )
    owners, translation_indices = _translation_segments(
        snapped_cache_indices, precalc
    )
    if not len(owners):
        return (
            np.empty(0, dtype=np.int32),
            np.empty((0, 3), dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int16),
        )
    pose_translations = (
        protein_center[None, :]
        + precalc.translation_vectors_world[translation_indices]
        - ring_centers[owners]
    )
    grid = np.round(pose_translations / GRID_SPACING).astype(np.int64)
    owners, grid, counts = _deduplicate_owner_grid(owners, grid)
    if not len(owners):
        return owners, grid, counts, np.empty(0, dtype=np.int16)
    offsets = protein_center[None, :] - ring_centers[owners]
    capacity = (
        _pose_cell_capacity(grid, offsets, precalc)
        if _include_capacity
        else np.empty(0, dtype=np.int16)
    )
    return owners, grid, counts, capacity


def cell_fill(
    precalc: _PrecalculatedAnchor,
    ring_coords: np.ndarray,
    rotamer_matrix: np.ndarray,
    protein_center: np.ndarray,
    grid_point: np.ndarray,
    *,
    capacity_cache: dict[tuple[int, int, int], int] | None = None,
    conformer_to_reference: np.ndarray | None = None,
) -> tuple[int, int]:
    """Return (accepted translations, cell capacity) for one pose cell."""
    rotamer_matrix = np.asarray(rotamer_matrix)
    if rotamer_matrix.shape != (3, 3):
        rotamer_matrix = _rotamers_to_matrices(rotamer_matrix[None, ...])[0]
    grid_point = np.asarray(grid_point, dtype=np.int64)
    if grid_point.shape != (3,):
        raise ValueError("grid point must have shape [3]")
    owners, grid, counts, capacity = candidate_cells(
        precalc,
        ring_coords,
        rotamer_matrix[None, :, :],
        protein_center,
        _include_capacity=False,
        conformer_to_reference=conformer_to_reference,
    )
    matches = np.all(grid == grid_point, axis=1)
    if np.any(matches):
        index = int(np.flatnonzero(matches)[0])
        count = int(counts[index])
    else:
        count = 0

    ring_center = (
        np.asarray(ring_coords).mean(axis=0) @ rotamer_matrix
    ).astype(np.float32, copy=False)
    offset = np.asarray(protein_center)[None, :] - ring_center[None, :]
    cell_capacity = _pose_cell_capacity(
        grid_point[None, :], offset, precalc,
        phase_capacity_cache=capacity_cache,
    )
    return count, int(cell_capacity[0])
