from __future__ import annotations

import io
from collections import deque
from pathlib import Path
import tempfile

import numpy as np

from offsets import expand_discrete_offsets, gather_discrete_offsets


_POSE_ZSTD_SUFFIX = ".npy.zst"


def canonical_center_coord(v: np.ndarray | int) -> np.ndarray | int:
    if isinstance(v, np.ndarray):
        out = ((v.astype(np.int32) + 128) // 256) * 256
        if out.size and (
            out.min() < np.iinfo(np.int16).min or out.max() > np.iinfo(np.int16).max
        ):
            raise ValueError("canonical center out of int16 range")
        return out.astype(np.int16, copy=False)
    out = ((int(v) + 128) // 256) * 256
    if out < np.iinfo(np.int16).min or out > np.iinfo(np.int16).max:
        raise ValueError("canonical center out of int16 range")
    return out


def canonical_center_tuple(xyz: np.ndarray) -> tuple[int, int, int]:
    x = int(canonical_center_coord(int(xyz[0])))
    y = int(canonical_center_coord(int(xyz[1])))
    z = int(canonical_center_coord(int(xyz[2])))
    return x, y, z


def canonical_center_codes(translations: np.ndarray) -> np.ndarray:
    if len(translations) == 0:
        return np.empty((0,), dtype=np.uint32)
    translations = np.asarray(translations, dtype=np.int16)
    if translations.ndim != 2 or translations.shape[1] != 3:
        raise ValueError("translations must be a [N,3] array")
    cc = np.empty_like(translations, dtype=np.int16)
    cc[:, 0] = canonical_center_coord(translations[:, 0])
    cc[:, 1] = canonical_center_coord(translations[:, 1])
    cc[:, 2] = canonical_center_coord(translations[:, 2])
    cx = ((cc[:, 0].astype(np.int32) + 32768) >> 8).astype(np.uint32)
    cy = ((cc[:, 1].astype(np.int32) + 32768) >> 8).astype(np.uint32)
    cz = ((cc[:, 2].astype(np.int32) + 32768) >> 8).astype(np.uint32)
    return (cx << np.uint32(16)) | (cy << np.uint32(8)) | cz


def _as_uint16_array(name: str, arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if arr.size == 0:
        return arr.astype(np.uint16, copy=False)
    if arr.min() < 0 or arr.max() > np.iinfo(np.uint16).max:
        raise ValueError(f"{name} values must fit in uint16")
    return arr.astype(np.uint16, copy=False)


def _finalize_chunk(
    results: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    conf_list: list[int],
    rot_list: list[int],
    off_idx_list: list[int],
    offsets_list: list[np.ndarray],
) -> None:
    if not conf_list:
        return

    offsets_arr = np.array(offsets_list, dtype=np.int16)
    max_xyz = offsets_arr.max(axis=0).astype(np.int32)
    mean_offset = np.clip(max_xyz - 127, -32768, 32767).astype(np.int16)
    offsets_int8 = (offsets_arr - mean_offset).astype(np.int8)
    if offsets_int8.max() > 127 or offsets_int8.min() < -128:
        raise ValueError("Offsets exceed int8 range after mean subtraction")

    offsets_uint8 = offsets_int8.view(np.uint8)

    poses = np.column_stack(
        (
            np.array(conf_list, dtype=np.uint16),
            np.array(rot_list, dtype=np.uint16),
            np.array(off_idx_list, dtype=np.uint16),
        )
    )
    results.append((poses, mean_offset, offsets_uint8))


def _pack_expanded_poses_slow(
    conf: np.ndarray,
    rot: np.ndarray,
    translations: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    results: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    conf_list: list[int] = []
    rot_list: list[int] = []
    off_idx_list: list[int] = []
    offsets_list: list[np.ndarray] = []
    offset_map: dict[tuple[int, int, int], int] = {}
    min_xyz: np.ndarray | None = None
    max_xyz: np.ndarray | None = None
    pose_count = 0

    for i in range(len(translations)):
        trans = translations[i].astype(np.int16, copy=False)
        key = (int(trans[0]), int(trans[1]), int(trans[2]))
        while True:
            new_offset = key not in offset_map
            if new_offset:
                new_min = trans.astype(np.int32)
                new_max = trans.astype(np.int32)
                if min_xyz is not None:
                    new_min = np.minimum(min_xyz, new_min)
                    new_max = np.maximum(max_xyz, new_max)
                spread = new_max - new_min
                too_wide = np.any(spread > 255)
                too_many_offsets = len(offsets_list) + 1 > 2**16
            else:
                too_wide = False
                too_many_offsets = False

            too_many_poses = pose_count + 1 > 2**32
            need_new_chunk = too_wide or too_many_offsets or too_many_poses

            if need_new_chunk and pose_count > 0:
                _finalize_chunk(
                    results, conf_list, rot_list, off_idx_list, offsets_list
                )
                conf_list = []
                rot_list = []
                off_idx_list = []
                offsets_list = []
                offset_map = {}
                min_xyz = None
                max_xyz = None
                pose_count = 0
                continue

            if need_new_chunk and pose_count == 0:
                raise ValueError("Single offset violates packing constraints")

            break

        if new_offset:
            offset_map[key] = len(offsets_list)
            offsets_list.append(trans.copy())
            if min_xyz is None:
                min_xyz = trans.astype(np.int32)
                max_xyz = trans.astype(np.int32)
            else:
                min_xyz = np.minimum(min_xyz, trans.astype(np.int32))
                max_xyz = np.maximum(max_xyz, trans.astype(np.int32))

        conf_list.append(int(conf[i]))
        rot_list.append(int(rot[i]))
        off_idx_list.append(offset_map[key])
        pose_count += 1

    _finalize_chunk(results, conf_list, rot_list, off_idx_list, offsets_list)
    return results


def pack_expanded_poses(
    conformer_indices: np.ndarray,
    rotamer_indices: np.ndarray,
    translations: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Pack already-expanded poses and translations into the on-disk pose format.
    """
    conf = _as_uint16_array("conformer_indices", conformer_indices)
    rot = _as_uint16_array("rotamer_indices", rotamer_indices)
    if conf.shape != rot.shape:
        raise ValueError("conformer_indices and rotamer_indices must have same shape")

    translations = np.asarray(translations, dtype=np.int16)
    if translations.ndim != 2 or translations.shape[1] != 3:
        raise ValueError("translations must be a [N,3] array")
    if len(translations) != len(conf):
        raise ValueError("translations length must match conformer/rotamer indices")
    if len(translations) == 0:
        return []

    if len(translations) <= 2**32:
        min_xyz = translations.min(axis=0).astype(np.int32)
        max_xyz = translations.max(axis=0).astype(np.int32)
        spread = max_xyz - min_xyz
        if np.all(spread <= 255):
            translations_c = np.ascontiguousarray(translations, dtype=np.int16)
            packed_dtype = np.dtype((np.void, translations_c.dtype.itemsize * 3))
            packed_offsets = translations_c.view(packed_dtype).ravel()
            unique_packed, inverse = np.unique(packed_offsets, return_inverse=True)
            unique_offsets = unique_packed.view(np.int16).reshape(-1, 3)
            if len(unique_offsets) <= 2**16:
                max_xyz = unique_offsets.max(axis=0).astype(np.int32)
                mean_offset = np.clip(max_xyz - 127, -32768, 32767).astype(np.int16)
                offsets_int16 = unique_offsets.astype(np.int16)
                offsets_int8 = (offsets_int16 - mean_offset).astype(np.int8)
                if offsets_int8.max() <= 127 and offsets_int8.min() >= -128:
                    poses = np.column_stack(
                        (
                            conf,
                            rot,
                            inverse.astype(np.uint16, copy=False),
                        )
                    )
                    offsets_uint8 = offsets_int8.view(np.uint8)
                    return [(poses, mean_offset, offsets_uint8)]

    return _pack_expanded_poses_slow(conf, rot, translations)


def pack_poses_from_discrete_offsets(
    conformer_indices: np.ndarray,
    rotamer_indices: np.ndarray,
    offsets_tuple: tuple[np.ndarray, np.ndarray, np.ndarray, dict],
    *,
    reverse_table: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Pack poses directly from flattened discrete-offset indices.

    `conformer_indices` and `rotamer_indices` are indexed by displacement index.
    `offsets_tuple` is `(disp_indices, p_indices, rounded, reverse_map)` where
    `disp_indices` and `p_indices` are already flattened candidate assignments.
    """
    conf_by_disp = _as_uint16_array("conformer_indices", conformer_indices)
    rot_by_disp = _as_uint16_array("rotamer_indices", rotamer_indices)
    if conf_by_disp.shape != rot_by_disp.shape:
        raise ValueError("conformer_indices and rotamer_indices must have same shape")

    disp_indices, p_indices, rounded, reverse_map = offsets_tuple
    disp_indices = np.asarray(disp_indices)
    p_indices = np.asarray(p_indices)
    rounded = np.asarray(rounded, dtype=np.int16)

    if disp_indices.ndim != 1 or p_indices.ndim != 1:
        raise ValueError("disp_indices and p_indices must be 1D arrays")
    if len(disp_indices) != len(p_indices):
        raise ValueError("disp_indices and p_indices must have the same length")
    if rounded.ndim != 2 or rounded.shape[1] != 3:
        raise ValueError("rounded must be a [N,3] array")
    if len(disp_indices) == 0:
        return []

    if disp_indices.min() < 0 or disp_indices.max() >= len(conf_by_disp):
        raise ValueError("disp_indices contains out-of-range displacement index")

    if reverse_table is None:
        max_p_index = max(reverse_map.keys()) if reverse_map else -1
        reverse_table = np.empty((max_p_index + 1, 3), dtype=np.int16)
        for p_index, xyz in reverse_map.items():
            reverse_table[p_index] = xyz
    else:
        reverse_table = np.asarray(reverse_table, dtype=np.int16)
        if reverse_table.ndim != 2 or reverse_table.shape[1] != 3:
            raise ValueError("reverse_table must be a [P,3] array")

    if p_indices.min() < 0 or p_indices.max() >= len(reverse_table):
        raise ValueError("p_indices contains out-of-range P index")

    disp_indices = disp_indices.astype(np.int64, copy=False)
    p_indices = p_indices.astype(np.int64, copy=False)
    expanded_conf = conf_by_disp[disp_indices]
    expanded_rot = rot_by_disp[disp_indices]
    expanded_offsets = rounded[disp_indices] + reverse_table[p_indices]
    return pack_expanded_poses(expanded_conf, expanded_rot, expanded_offsets)


class PoseStreamAccumulator:
    def __init__(
        self,
        outdir: str | Path,
        *,
        max_poses_per_chunk: int = 2**32,
        canonical_centers: bool = True,
        zstd: bool = False,
    ) -> None:
        self._outdir = Path(outdir)
        self._max_poses_per_chunk = int(max_poses_per_chunk)
        self._zstd = bool(zstd)
        if self._max_poses_per_chunk <= 0:
            raise ValueError("max_poses_per_chunk must be positive")
        if self._max_poses_per_chunk > 2**32:
            raise ValueError("max_poses_per_chunk must be <= 2**32")
        self._canonical_centers = bool(canonical_centers)
        self._chunk_index = 1
        self._written_chunks: list[tuple[Path, Path]] = []
        self._pose_count = 0
        self.total_poses = 0
        self._new_tempfile()
        self._offset_map: dict[tuple[int, int, int], int] = {}
        self._offsets: list[np.ndarray] = []
        self._min_xyz: np.ndarray | None = None
        self._max_xyz: np.ndarray | None = None
        self._chunk_center: tuple[int, int, int] | None = None

    def _new_tempfile(self) -> None:
        self._tmp = tempfile.NamedTemporaryFile(
            prefix="poses_stream_", suffix=".bin", delete=False
        )
        self._tmp_path = Path(self._tmp.name)
        self._tmp.close()

    def _current_paths(self) -> tuple[Path, Path]:
        poses_suffix = _POSE_ZSTD_SUFFIX if self._zstd else ".npy"
        return (
            self._outdir / f"poses-{self._chunk_index}{poses_suffix}",
            self._outdir / f"offsets-{self._chunk_index}.dat",
        )

    def _register_offset(self, xyz: np.ndarray) -> int:
        key = (int(xyz[0]), int(xyz[1]), int(xyz[2]))
        existing = self._offset_map.get(key)
        if existing is not None:
            return existing

        index = len(self._offsets)
        if index >= 2**16:
            raise ValueError("Too many unique offsets (>65536) for single chunk output")

        xyz32 = xyz.astype(np.int32)
        if self._min_xyz is None:
            new_min = xyz32
            new_max = xyz32
        else:
            new_min = np.minimum(self._min_xyz, xyz32)
            new_max = np.maximum(self._max_xyz, xyz32)

        if np.any((new_max - new_min) > 255):
            raise ValueError("Offset spread exceeds 255 for single chunk output")

        self._offset_map[key] = index
        self._offsets.append(xyz.astype(np.int16, copy=True))
        self._min_xyz = new_min
        self._max_xyz = new_max
        return index

    def _try_add_batch(
        self,
        conformer_indices: np.ndarray,
        rotamer_indices: np.ndarray,
        translations: np.ndarray,
        center: tuple[int, int, int],
    ) -> bool:
        conf = _as_uint16_array("conformer_indices", conformer_indices)
        rot = _as_uint16_array("rotamer_indices", rotamer_indices)
        if conf.shape != rot.shape:
            raise ValueError(
                "conformer_indices and rotamer_indices must have same shape"
            )

        translations = np.asarray(translations, dtype=np.int16)
        if translations.ndim != 2 or translations.shape[1] != 3:
            raise ValueError("translations must be a [N,3] array")
        if len(translations) != len(conf):
            raise ValueError("translations length must match conformer/rotamer indices")
        if len(translations) == 0:
            return True

        if self._pose_count + len(translations) > self._max_poses_per_chunk:
            return False

        if self._canonical_centers:
            if self._chunk_center is None:
                self._chunk_center = center
            elif center != self._chunk_center:
                return False

        translations_c = np.ascontiguousarray(translations, dtype=np.int16)
        packed_dtype = np.dtype((np.void, translations_c.dtype.itemsize * 3))
        packed_offsets = translations_c.view(packed_dtype).ravel()
        unique_packed, inverse = np.unique(packed_offsets, return_inverse=True)
        unique_offsets = unique_packed.view(np.int16).reshape(-1, 3)

        new_offsets: list[np.ndarray] = []
        new_min = None
        new_max = None
        unique_to_global = np.empty(len(unique_offsets), dtype=np.uint16)
        for idx, xyz in enumerate(unique_offsets):
            key = (int(xyz[0]), int(xyz[1]), int(xyz[2]))
            existing = self._offset_map.get(key)
            if existing is not None:
                unique_to_global[idx] = existing
                continue
            new_offsets.append(xyz)
            unique_to_global[idx] = len(self._offsets) + len(new_offsets) - 1
            xyz32 = xyz.astype(np.int32)
            if new_min is None:
                new_min = xyz32
                new_max = xyz32
            else:
                new_min = np.minimum(new_min, xyz32)
                new_max = np.maximum(new_max, xyz32)

        if len(self._offsets) + len(new_offsets) > 2**16:
            return False

        if new_min is not None:
            if self._min_xyz is None:
                combined_min = new_min
                combined_max = new_max
            else:
                combined_min = np.minimum(self._min_xyz, new_min)
                combined_max = np.maximum(self._max_xyz, new_max)
            if np.any((combined_max - combined_min) > 255):
                return False

        for xyz in new_offsets:
            key = (int(xyz[0]), int(xyz[1]), int(xyz[2]))
            self._offset_map[key] = len(self._offsets)
            self._offsets.append(xyz.astype(np.int16, copy=True))
            if self._min_xyz is None:
                self._min_xyz = xyz.astype(np.int32)
                self._max_xyz = xyz.astype(np.int32)
            else:
                self._min_xyz = np.minimum(self._min_xyz, xyz.astype(np.int32))
                self._max_xyz = np.maximum(self._max_xyz, xyz.astype(np.int32))

        offset_indices = unique_to_global[inverse]
        poses_chunk = np.column_stack((conf, rot, offset_indices))
        with self._tmp_path.open("ab") as handle:
            poses_chunk.tofile(handle)
        self._pose_count += len(translations)
        self.total_poses += len(translations)
        return True

    def add_chunk(
        self,
        conformer_indices: np.ndarray,
        rotamer_indices: np.ndarray,
        translations: np.ndarray,
    ) -> None:
        queue = deque([(conformer_indices, rotamer_indices, translations, None)])
        while queue:
            conf, rot, trans, center = queue.popleft()
            if len(trans) == 0:
                continue
            conf = _as_uint16_array("conformer_indices", conf)
            rot = _as_uint16_array("rotamer_indices", rot)
            if conf.shape != rot.shape:
                raise ValueError(
                    "conformer_indices and rotamer_indices must have same shape"
                )
            trans = np.asarray(trans, dtype=np.int16)
            if trans.ndim != 2 or trans.shape[1] != 3:
                raise ValueError("translations must be a [N,3] array")
            if len(trans) != len(conf):
                raise ValueError(
                    "translations length must match conformer/rotamer indices"
                )

            if self._canonical_centers and center is None:
                codes = canonical_center_codes(trans)
                if len(codes) > 1 and np.any(codes[1:] != codes[0]):
                    order = np.argsort(codes, kind="stable")
                    sorted_codes = codes[order]
                    boundaries = (
                        np.flatnonzero(sorted_codes[1:] != sorted_codes[:-1]) + 1
                    )
                    starts = np.concatenate(([0], boundaries))
                    stops = np.concatenate((boundaries, [len(order)]))
                    subbatches = []
                    for start, stop in zip(starts, stops):
                        idx = order[start:stop]
                        center_i = canonical_center_tuple(trans[idx[0]])
                        subbatches.append((conf[idx], rot[idx], trans[idx], center_i))
                    queue.extendleft(reversed(subbatches))
                    continue
                center = canonical_center_tuple(trans[0])

            if center is None:
                center = canonical_center_tuple(trans[0])

            if self._try_add_batch(conf, rot, trans, center):
                continue
            if self._pose_count > 0:
                self._flush_current()
                queue.appendleft((conf, rot, trans, center))
            else:
                if len(trans) == 1:
                    raise ValueError("Single pose violates packing constraints")
                mid = len(trans) // 2
                queue.appendleft((conf[mid:], rot[mid:], trans[mid:], center))
                queue.appendleft((conf[:mid], rot[:mid], trans[:mid], center))

    def _flush_current(self) -> None:
        if self._pose_count == 0:
            self.cleanup()
            self._new_tempfile()
            return

        poses_path, offsets_path = self._current_paths()
        self._outdir.mkdir(parents=True, exist_ok=True)
        materialized_poses_path = poses_path
        tmp_npy_path: Path | None = None
        if self._zstd:
            with tempfile.NamedTemporaryFile(
                prefix="poses_materialized_", suffix=".npy", delete=False
            ) as handle:
                tmp_npy_path = Path(handle.name)
            materialized_poses_path = tmp_npy_path

        try:
            poses_mmap = np.lib.format.open_memmap(
                materialized_poses_path,
                mode="w+",
                dtype=np.uint16,
                shape=(self._pose_count, 3),
            )
            chunk_rows = 5_000_000
            with self._tmp_path.open("rb") as handle:
                row = 0
                while row < self._pose_count:
                    nrows = min(chunk_rows, self._pose_count - row)
                    buf = np.fromfile(handle, dtype=np.uint16, count=nrows * 3)
                    if len(buf) != nrows * 3:
                        raise ValueError(
                            f"Unexpected EOF while materializing {poses_path.name}"
                        )
                    poses_mmap[row : row + nrows] = buf.reshape((nrows, 3))
                    row += nrows
            del poses_mmap

            if self._zstd:
                _compress_pose_npy_file(materialized_poses_path, poses_path)
        finally:
            if tmp_npy_path is not None and tmp_npy_path.exists():
                tmp_npy_path.unlink()

        offsets_arr = np.array(self._offsets, dtype=np.int16)
        if self._canonical_centers:
            if self._chunk_center is None:
                raise ValueError("Missing canonical chunk center while flushing poses")
            mean_offset = np.array(self._chunk_center, dtype=np.int16)
            centered = offsets_arr.astype(np.int32) - mean_offset.astype(np.int32)
            if centered.size and (
                centered.min() < np.iinfo(np.int8).min
                or centered.max() > np.iinfo(np.int8).max
            ):
                raise ValueError("Offsets exceed int8 range for canonical center")
            offsets_int8 = centered.astype(np.int8, copy=False)
        else:
            max_xyz = offsets_arr.max(axis=0).astype(np.int32)
            mean_offset = np.clip(max_xyz - 127, -32768, 32767).astype(np.int16)
            offsets_int8 = (offsets_arr - mean_offset).astype(np.int8)
            if offsets_int8.max() > 127 or offsets_int8.min() < -128:
                raise ValueError("Offsets exceed int8 range after mean subtraction")
        _write_offsets_file(offsets_path, mean_offset, offsets_int8.view(np.uint8))

        self._written_chunks.append((poses_path, offsets_path))
        self._chunk_index += 1
        self._pose_count = 0
        self._offset_map = {}
        self._offsets = []
        self._min_xyz = None
        self._max_xyz = None
        self._chunk_center = None
        self.cleanup()
        self._new_tempfile()

    def finish(self) -> list[tuple[Path, Path]]:
        if self.total_poses == 0:
            poses_path, offsets_path = self._current_paths()
            self._outdir.mkdir(parents=True, exist_ok=True)
            _write_pose_array_file(
                poses_path,
                np.empty((0, 3), dtype=np.uint16),
                zstd=self._zstd,
            )
            _write_offsets_file(
                offsets_path,
                np.zeros((3,), dtype=np.int16),
                np.empty((0, 3), dtype=np.uint8),
            )
            self.cleanup()
            return [(poses_path, offsets_path)]

        self._flush_current()
        self.cleanup()
        return self._written_chunks

    def cleanup(self) -> None:
        if self._tmp_path.exists():
            self._tmp_path.unlink()


def pack_all_poses(
    conformer_indices: np.ndarray,
    rotamer_indices: np.ndarray,
    offsets_tuple: tuple[np.ndarray, np.ndarray, np.ndarray, dict],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Pack poses into the on-disk representation (poses.npy + offsets.dat).

    Inputs are expanded per translation. Offsets are deduplicated and stored
    in a table, with poses referencing them by index.
    """
    conf = _as_uint16_array("conformer_indices", conformer_indices)
    rot = _as_uint16_array("rotamer_indices", rotamer_indices)
    if conf.shape != rot.shape:
        raise ValueError("conformer_indices and rotamer_indices must have same shape")

    disp_indices, p_indices, rounded, reverse_map = offsets_tuple
    translation_lists = expand_discrete_offsets(
        disp_indices, p_indices, rounded, reverse_map
    )
    if not translation_lists:
        return []

    _, tdata = gather_discrete_offsets(translation_lists)
    if len(conf) != len(tdata):
        raise ValueError("Input indices length must match gathered translation length")
    return pack_expanded_poses(conf, rot, tdata)


def unpack_poses(
    poses: np.ndarray, mean_offset: np.ndarray, offsets: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Unpack a pose tuple into indices and a deduplicated offset table.
    """
    poses = np.asarray(poses)
    if poses.ndim != 2 or poses.shape[1] != 3:
        raise ValueError("poses must be a [N,3] array")

    mean_offset = np.asarray(mean_offset, dtype=np.int16)
    if mean_offset.shape != (3,):
        raise ValueError("mean_offset must have shape (3,)")

    offsets_uint8 = np.asarray(offsets, dtype=np.uint8)
    if offsets_uint8.ndim != 2 or offsets_uint8.shape[1] != 3:
        raise ValueError("offsets must be a [K,3] array")

    offsets_int8 = offsets_uint8.view(np.int8)
    offset_table = offsets_int8.astype(np.int16) + mean_offset.astype(np.int16)

    conformer_indices = poses[:, 0].astype(np.uint16, copy=False)
    rotamer_indices = poses[:, 1].astype(np.uint16, copy=False)
    offset_indices = poses[:, 2].astype(np.uint16, copy=False)

    return conformer_indices, rotamer_indices, offset_indices, offset_table


def _write_offsets_file(
    path: Path, mean_offset: np.ndarray, offsets: np.ndarray
) -> None:
    mean_offset = np.asarray(mean_offset, dtype=np.int16)
    if mean_offset.shape != (3,):
        raise ValueError("mean_offset must have shape (3,)")

    offsets_uint8 = np.asarray(offsets, dtype=np.uint8)
    if offsets_uint8.ndim != 2 or offsets_uint8.shape[1] != 3:
        raise ValueError("offsets must be a [K,3] array")

    with path.open("wb") as handle:
        handle.write(mean_offset.tobytes(order="C"))
        handle.write(offsets_uint8.tobytes(order="C"))


def _read_offsets_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = path.read_bytes()
    if len(raw) < 6:
        raise ValueError("offsets.dat is too small")
    mean_offset = np.frombuffer(raw[:6], dtype=np.int16).copy()
    remainder = raw[6:]
    if len(remainder) % 3 != 0:
        raise ValueError("offsets.dat remainder length must be divisible by 3")
    offsets_uint8 = np.frombuffer(remainder, dtype=np.uint8).copy()
    offsets_uint8 = offsets_uint8.reshape((-1, 3))
    return mean_offset, offsets_uint8


def write_pose_files(
    outdir: str | Path, packed: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> list[tuple[Path, Path]]:
    """
    Write poses-*.npy and offsets-*.dat files to disk (1-based indices).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    written: list[tuple[Path, Path]] = []
    for i, (poses, mean_offset, offsets) in enumerate(packed, start=1):
        poses_path = outdir / f"poses-{i}.npy"
        offsets_path = outdir / f"offsets-{i}.dat"

        np.save(poses_path, np.asarray(poses, dtype=np.uint16))
        _write_offsets_file(offsets_path, mean_offset, offsets)
        written.append((poses_path, offsets_path))

    return written


def _is_compressed_pose_name(name: str) -> bool:
    return name.endswith(_POSE_ZSTD_SUFFIX)


def _compressed_pose_suffixes_text() -> str:
    return _POSE_ZSTD_SUFFIX


def _require_zstandard(action: str):
    try:
        import zstandard as zstd
    except ImportError as exc:
        raise ImportError(
            f"zstandard is required to {action} compressed pose files "
            f"({_compressed_pose_suffixes_text()})"
        ) from exc
    return zstd


def _compress_pose_npy_file(source_path: Path, target_path: Path) -> None:
    zstd = _require_zstandard("write")
    with tempfile.NamedTemporaryFile(
        prefix=f"{target_path.stem}.",
        suffix=f"{target_path.suffix}.tmp",
        dir=target_path.parent,
        delete=False,
    ) as handle:
        tmp_target_path = Path(handle.name)
    try:
        with source_path.open("rb") as src, tmp_target_path.open("wb") as dst:
            zstd.ZstdCompressor().copy_stream(src, dst)
        tmp_target_path.replace(target_path)
    finally:
        if tmp_target_path.exists():
            tmp_target_path.unlink()


def _write_pose_array_file(
    path: Path, poses: np.ndarray, *, zstd: bool = False
) -> None:
    poses = np.asarray(poses, dtype=np.uint16)
    if not zstd:
        np.save(path, poses)
        return

    with tempfile.NamedTemporaryFile(
        prefix="poses_write_", suffix=".npy", delete=False
    ) as handle:
        tmp_npy_path = Path(handle.name)
    try:
        np.save(tmp_npy_path, poses)
        _compress_pose_npy_file(tmp_npy_path, path)
    finally:
        if tmp_npy_path.exists():
            tmp_npy_path.unlink()


def _pose_path_priority(path: Path) -> int:
    return 1 if path.name.endswith(_POSE_ZSTD_SUFFIX) else 0


def _read_npy_header(fileobj) -> tuple[tuple[int, ...], bool, np.dtype]:
    version = np.lib.format.read_magic(fileobj)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(fileobj)
    elif version == (2, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(fileobj)
    elif version == (3, 0) and hasattr(np.lib.format, "read_array_header_3_0"):
        shape, fortran_order, dtype = np.lib.format.read_array_header_3_0(fileobj)
    else:
        raise ValueError(f"Unsupported .npy version {version}")
    return shape, fortran_order, dtype


def pose_array_shape(path: str | Path) -> tuple[int, ...]:
    path = Path(path)
    if _is_compressed_pose_name(path.name):
        zstd = _require_zstandard("read")
        with path.open("rb") as compressed:
            with zstd.ZstdDecompressor().stream_reader(compressed) as reader:
                shape, _, _ = _read_npy_header(reader)
        return shape

    with path.open("rb") as handle:
        shape, _, _ = _read_npy_header(handle)
    return shape


def pose_array_length(path: str | Path) -> int:
    shape = pose_array_shape(path)
    if len(shape) == 0:
        raise ValueError(f"Scalar array has no length: {path}")
    return int(shape[0])


def pose_index_from_name(name: str) -> int | None:
    if name.endswith(_POSE_ZSTD_SUFFIX):
        base = name[: -len(".zst")]
    elif name.endswith(".npy"):
        base = name
    else:
        return None
    if not base.startswith("poses-") or not base.endswith(".npy"):
        return None
    index_text = base[len("poses-") : -4]
    if not index_text.isdigit():
        return None
    return int(index_text)


def discover_pose_pairs(directory: str | Path) -> list[tuple[Path, Path]]:
    directory = Path(directory)
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"Pose directory not found: {directory}")

    pose_files = list(directory.glob("poses-*.npy")) + list(
        directory.glob(f"poses-*{_POSE_ZSTD_SUFFIX}")
    )
    if not pose_files:
        raise FileNotFoundError(f"No pose files found in {directory}")

    indexed: dict[int, Path] = {}
    for pose_path in pose_files:
        index = pose_index_from_name(pose_path.name)
        if index is None:
            continue
        if index not in indexed or _pose_path_priority(pose_path) > _pose_path_priority(
            indexed[index]
        ):
            indexed[index] = pose_path

    if not indexed:
        raise FileNotFoundError(
            f"No poses-*.npy or poses-*{_POSE_ZSTD_SUFFIX} files found in {directory}"
        )

    pairs: list[tuple[Path, Path]] = []
    for index in sorted(indexed):
        pose_path = indexed[index]
        offsets_path = directory / f"offsets-{index}.dat"
        if not offsets_path.exists():
            raise FileNotFoundError(f"Missing {offsets_path} for {pose_path}")
        pairs.append((pose_path, offsets_path))
    return pairs


def discover_pose_pairs_with_counts(directory: str | Path) -> list[tuple[Path, Path, int]]:
    pairs_with_counts: list[tuple[Path, Path, int]] = []
    for pose_path, offsets_path in discover_pose_pairs(directory):
        pairs_with_counts.append((pose_path, offsets_path, pose_array_length(pose_path)))
    return pairs_with_counts


def open_pose_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    if _is_compressed_pose_name(path.name):
        zstd = _require_zstandard("read")
        with path.open("rb") as compressed:
            with zstd.ZstdDecompressor().stream_reader(compressed) as reader:
                decompressed = reader.read()
        return np.load(io.BytesIO(decompressed))

    return np.load(path, mmap_mode="r")


def _load_pose_array(path: str | Path) -> np.ndarray:
    return open_pose_array(path)


def load_offset_table(path: str | Path) -> np.ndarray:
    path = Path(path)
    mean_offset, offsets_uint8 = _read_offsets_file(path)
    offsets_int8 = offsets_uint8.view(np.int8)
    return offsets_int8.astype(np.int16) + mean_offset.astype(np.int16)


def read_pose_files(
    outdir: str | Path,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Read poses-*.npy (or compressed poses-*.npy.zst) and offsets-*.dat files from
    a directory.
    Returns a list of (poses, mean_offset, offsets) tuples.
    """
    results: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for pose_path, offsets_path in discover_pose_pairs(outdir):
        poses = _load_pose_array(pose_path)
        mean_offset, offsets = _read_offsets_file(offsets_path)
        results.append((poses, mean_offset, offsets))

    return results
