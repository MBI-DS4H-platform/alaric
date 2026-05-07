"""
Precompute discrete offsets within spheres of radius 1..19 in a [-19, 19]^3 box.

Each grid point has a uint16 P index and a reverse map to its int8 xyz coords.
`get_discrete_offsets` rounds continuous offsets to int16, computes the modulo
vectors, and bins them by a boundary radius of `ceil(||mod|| + radius)` (triangle
inequality) to choose the smallest precomputed sphere that can contain all valid
points. For each bin, it filters sphere points with
`||(coord - mod)||^2 < radius^2` and returns:

- D: displacement vector indices (sorted),
- P: P indices into the precomputed box,
- R: rounded displacement vectors,
- M: reverse map from P index to xyz coords.

Use `expand_discrete_offsets` to materialize coordinates.
"""

from __future__ import annotations

import numpy as np

_BOX_MIN = -19
_BOX_MAX = 19

_BOX_RANGE = np.arange(_BOX_MIN, _BOX_MAX + 1, dtype=np.int8)
_bx, _by, _bz = np.meshgrid(_BOX_RANGE, _BOX_RANGE, _BOX_RANGE, indexing="ij")
_BOX_COORDS = np.column_stack((_bx.ravel(), _by.ravel(), _bz.ravel())).astype(np.int8)
_P_INDICES = np.arange(_BOX_COORDS.shape[0], dtype=np.uint16)
_P_REVERSE = {int(p): _BOX_COORDS[i].copy() for i, p in enumerate(_P_INDICES)}

_BOX_RSQ = (_BOX_COORDS.astype(np.int16) ** 2).sum(axis=1)

_SPHERE_COORDS = [None] * (_BOX_MAX + 1)
_SPHERE_P = [None] * (_BOX_MAX + 1)
for _r in range(_BOX_MAX, 0, -1):
    _mask = _BOX_RSQ <= _r * _r
    _SPHERE_COORDS[_r] = _BOX_COORDS[_mask]
    _SPHERE_P[_r] = _P_INDICES[_mask]


def get_discrete_offsets(
    continuous_offsets: np.ndarray, radii: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Obtain all discrete offset vectors within a certain radius from each continuous offset.
    continuous_offsets is a [N,3] array of continuous offset vectors.

    Both continuous_offsets and radii are in grid units;
    they must have been pre-divided by the grid spacing.

    Returns:
    - D: a [X] int32 array of displacement vector indices.
        These indices are grouped, i.e. all entries where index=X are together.
        However, the array is not sorted by index:
          the first entries generally have the lowest number of entries.
    - P: a [X] uint16 array of P indices into the precomputed box.
    - R: a [N, 3] int16 array of rounded displacement vectors.
    - M: a dict mapping P index to an xyz np.int8 array.

    The sphere boundary used per offset is ceil(||mod|| + radius).
    """

    continuous_offsets = np.asarray(continuous_offsets, dtype=np.float32)
    if continuous_offsets.ndim != 2 or continuous_offsets.shape[1] != 3:
        raise ValueError("continuous_offsets must be a [N,3] array")

    radii = np.asarray(radii, dtype=np.float32)
    if radii.ndim == 0:
        radii = np.full((continuous_offsets.shape[0],), radii, dtype=np.float32)
    if radii.shape[0] != continuous_offsets.shape[0]:
        raise ValueError("radii must be a scalar or have the same length as offsets")

    rounded = np.round(continuous_offsets).astype(np.int16)
    if rounded.size:
        min_val = int(rounded.min())
        max_val = int(rounded.max())
        if (
            min_val - 128 < np.iinfo(np.int16).min
            or max_val + 128 > np.iinfo(np.int16).max
        ):
            raise ValueError("rounded offsets exceed int16 range with +/-128 margin")

    modulo = continuous_offsets - rounded

    size_sq = (modulo * modulo).sum(axis=1)
    boundary = np.ceil(np.sqrt(size_sq) + radii).astype(np.int16)

    if boundary.size:
        max_boundary = int(boundary.max())
        if max_boundary > _BOX_MAX:
            raise ValueError("boundary exceeds precomputed sphere size")

    unique_bounds, first_idx, inverse = np.unique(
        boundary, return_index=True, return_inverse=True
    )

    disp_indices_list = []
    p_indices_list = []
    for b in unique_bounds:
        if b < 1:
            continue
        sphere_coords = _SPHERE_COORDS[int(b)]
        sphere_p = _SPHERE_P[int(b)]
        if sphere_coords is None:
            continue

        bin_mask = boundary == b
        if not np.any(bin_mask):
            continue

        idx = np.nonzero(bin_mask)[0]
        mod_bin = modulo[idx].astype(np.float32)
        radius_sq = (radii[idx] * radii[idx]).astype(np.float32)

        diff = sphere_coords.astype(np.float32)[None, :, :] - mod_bin[:, None, :]
        dist_sq = (diff * diff).sum(axis=2)
        valid = dist_sq < radius_sq[:, None]
        i_rel, j = np.nonzero(valid)
        if i_rel.size == 0:
            continue

        disp_indices_list.append(idx[i_rel].astype(np.int32))
        p_indices_list.append(sphere_p[j])

    if disp_indices_list:
        disp_indices = np.concatenate(disp_indices_list)
        p_indices = np.concatenate(p_indices_list)
    else:
        disp_indices = np.empty((0,), dtype=np.int32)
        p_indices = np.empty((0,), dtype=np.uint16)

    return disp_indices, p_indices, rounded, _P_REVERSE


def expand_discrete_offsets(disp_indices, p_indices, rounded, reverse_map=None):
    """
    Expand the results of get_discrete_offsets into a list of numpy arrays.

    :param disp_indices: displacement indices (D) obtained by get_discrete_offsets
    :param p_indices: P indices (P) obtained by get_discrete_offsets
    :param rounded: rounded displacement vectors (R) obtained by get_discrete_offsets
    :param reverse_map: optional map from P index to xyz np.int8 array

    Returns a list of numpy arrays indexed by displacement index.
    Missing displacement indices are represented as None entries.
    """
    if reverse_map is None:
        reverse_map = _P_REVERSE

    if disp_indices.size == 0:
        return []

    buckets = {}
    for i, p in zip(disp_indices, p_indices):
        buckets.setdefault(int(i), []).append(int(p))

    max_idx = max(buckets.keys())
    results = [None] * (max_idx + 1)
    for idx in sorted(buckets.keys()):
        p_seg = np.array(buckets[idx], dtype=np.uint16)
        coords = np.stack([reverse_map[int(p)] for p in p_seg], axis=0).astype(np.int16)
        results[idx] = coords + rounded[idx].astype(np.int16)
    return results


def gather_discrete_offsets(list_of_lists: list[list[np.ndarray | None]]) -> np.ndarray:

    data = np.concatenate([l for l in list_of_lists if l is not None])
    assert data.max() < 2**15 and data.min() > -(2**15)
    data = data.astype(np.int16)
    repeats = [len(l) if l is not None else 0 for l in list_of_lists]
    indices = np.repeat(np.arange(len(list_of_lists), dtype=np.uint32), repeats=repeats)
    assert len(data) == len(indices)
    return indices, data
