"""Small real-pool execution oracles for export C1/C2 (no installed library)."""

from collections import Counter
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "alaric"))
sys.path.insert(0, str(ROOT))

import alaric.identity_filter as identity_filter
from alaric.mask_common_conformer import main as make_masks
from alaric.npy_io import load_npy, save_npy
from alaric.poses import PoseReader, discover_organized, pack_pool, write_arc_file


GOLDEN = ROOT / "tests/middle/data/export-e1-two-input.json"


def _write_pool(path, rows, *, split=None):
    path.mkdir(parents=True, exist_ok=True)
    for member in path.glob("poses-*.arc*"):
        member.unlink()
    groups = [rows] if split is None else [rows[:split], rows[split:]]
    number = 0
    for group in groups:
        if not group:
            continue
        packed = pack_pool(np.array([row[0] for row in group], dtype=np.uint16),
                           np.array([row[1] for row in group], dtype=np.uint16),
                           np.array([row[2] for row in group], dtype=np.int32), bucket_size=16)
        for m, offsets, counts, poses in packed:
            number += 1
            write_arc_file(path / f"poses-{number}.arc", m, offsets, counts, poses, bucket_size=16)
    if not rows:
        # The writer omits empty archives, but the reader accepts a zero-pose
        # organized bucket with one zero-count offset. nO must be 1: its raw
        # zero encoding denotes 65536 offsets in this file format.
        (path / "poses-1.arc").write_bytes(
            struct.pack("<7sH3hHI3hI", b"alaric1", 16, 0, 0, 0, 1, 0, 0, 0, 0, 0))


def _records(path):
    if not discover_organized(path):
        return []
    result = []
    for chunk in PoseReader(path).iter_chunks():
        result.extend((int(conf), int(rot), tuple(int(x) for x in grid))
                      for conf, rot, grid in zip(chunk.conformers, chunk.rotamers, chunk.translations_grid))
    return result


def _apply(source, masks, output, side):
    # Import inside the test so the two-input pre-change artifact fixture can
    # still be captured before the new action exists.
    from alaric.apply_mask import run_apply_mask
    return run_apply_mask(source, masks, output, side)


@pytest.fixture
def paired(tmp_path):
    one, two, masks = tmp_path / "one", tmp_path / "two", tmp_path / "masks"
    _write_pool(one, [(0, 0, (0, 0, 0)), (1, 0, (1, 0, 0)), (1, 0, (1, 0, 0)), (2, 0, (2, 0, 0))])
    _write_pool(two, [(0, 1, (0, 1, 0)), (3, 1, (1, 1, 0)), (2, 1, (2, 1, 0))])
    assert make_masks([str(one), str(two), str(masks)]) == 0
    return one, two, masks


@pytest.mark.parametrize("side", [1, 2])
def test_apply_mask_matches_direct_mask_selection_preserving_record_order(paired, tmp_path, side):
    one, two, masks = paired
    source = one if side == 1 else two
    mask = load_npy(masks / f"mask{side}.npy.zst")
    expected = [row for row, keep in zip(_records(source), mask) if keep]
    output = tmp_path / "output"
    _apply(source, masks, output, side)
    assert _records(output) == expected
    assert len(expected) == 2


@pytest.mark.parametrize("common", [False, True])
def test_apply_mask_empty_and_all_true(paired, tmp_path, common):
    one, two, _ = paired
    rows = [(0, 0, (0, 0, 0)), (1, 0, (1, 0, 0))]
    _write_pool(one, rows)
    _write_pool(two, rows if common else [(2, 0, (0, 0, 0)), (3, 0, (1, 0, 0))])
    masks = tmp_path / "new-masks"
    make_masks([str(one), str(two), str(masks)])
    for side, source in ((1, one), (2, two)):
        output = tmp_path / f"out-{side}"
        _apply(source, masks, output, side)
        assert _records(output) == (_records(source) if common else [])


@pytest.mark.parametrize("side", [True, False, 0, 3, 1.0, "1", None])
def test_apply_mask_callable_rejects_invalid_selector(paired, tmp_path, side):
    one, _, masks = paired
    with pytest.raises((ValueError, TypeError)):
        _apply(one, masks, tmp_path / "out", side)


def test_apply_mask_rejects_wrong_side_source_even_with_compatible_fragment(paired, tmp_path):
    one, two, masks = paired
    for source, side in ((one, 2), (two, 1)):
        with pytest.raises(ValueError):
            _apply(source, masks, tmp_path / f"out-{side}", side)


def test_apply_mask_rejects_changed_source_order(paired, tmp_path):
    one, _, masks = paired
    _write_pool(one, list(reversed(_records(one))))
    with pytest.raises(ValueError):
        _apply(one, masks, tmp_path / "out", 1)


@pytest.mark.parametrize("bad", [np.array([True]), np.ones((4, 1), dtype=bool),
                                 np.array([1, 0, 0, 1], dtype=np.uint8),
                                 np.array([1., 0., 0., 1.])])
def test_apply_mask_rejects_wrong_length_shape_and_dtype(paired, tmp_path, bad):
    one, _, masks = paired
    save_npy(masks / "mask1.npy", bad, compress=True)
    with pytest.raises(ValueError):
        _apply(one, masks, tmp_path / "out", 1)


def test_apply_mask_missing_selected_mask_is_error(paired, tmp_path):
    one, _, masks = paired
    (masks / "mask1.npy.zst").unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        _apply(one, masks, tmp_path / "out", 1)


def test_apply_mask_cli_uses_selected_member(paired, tmp_path):
    _, two, masks = paired
    output = tmp_path / "cli-output"
    completed = subprocess.run([sys.executable, "-m", "alaric.apply_mask", str(two), str(masks), str(output), "--mask", "2"],
                               cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert _records(output) == [row for row, keep in zip(_records(two), load_npy(masks / "mask2.npy.zst")) if keep]


def test_two_input_identity_artifacts_match_prechange_golden(tmp_path):
    one, two, output = tmp_path / "one", tmp_path / "two", tmp_path / "output"
    _write_pool(one, [(0,0,(0,0,0)), (1,1,(1,0,0)), (1,1,(1,0,0)), (2,2,(17,0,0))])
    _write_pool(two, [(0,0,(0,0,0)), (1,1,(1,0,0)), (3,3,(18,0,0))])
    identity_filter.run_identity_filter(one, two, output)
    expected = json.loads(GOLDEN.read_text())
    actual = {path.name: path.read_bytes().hex() for path in output.iterdir() if path.name != "identity-filter.json"}
    assert actual == expected["files_hex"]
    manifest = (output / "identity-filter.json").read_text().replace(str(one), "<one>").replace(str(two), "<two>")
    assert manifest == expected["manifest_text"]


@pytest.mark.parametrize("rows,split", [
    ([(0,0,(0,0,0)), (1,1,(1,0,0)), (1,1,(1,0,0)), (2,2,(17,0,0))], 2),
    ([(0,0,(0,0,0)), (1,1,(1,0,0))], None),
    ([], None),
])
def test_single_identity_equals_same_pool_intersection(rows, split, tmp_path):
    source, legacy, single = tmp_path / "source", tmp_path / "legacy", tmp_path / "single"
    _write_pool(source, rows, split=split)
    identity_filter.run_identity_filter(source, source, legacy)
    identity_filter.run_identity_filter(source, None, single)
    assert _records(single) == _records(legacy)
    assert len(_records(single)) == len(set(_records(single)))
    np.testing.assert_array_equal(load_npy(single / "map-1.npy"), load_npy(legacy / "map-1.npy"))
    assert not (single / "map-2.npy").exists()
    assert not (single / "map-2.npy.zst").exists()


@pytest.mark.parametrize("split", [None, 2, 4])
def test_single_identity_maps_every_duplicate_to_its_pose(tmp_path, split):
    # Duplicates are a pose's alternative parents: alaric-chain needs all of them, so
    # map-1 must hold one row per *input* pose, not one per unique output pose.
    source, single = tmp_path / "source", tmp_path / "single"
    # _write_pool needs rows grouped by translation; the splits put duplicates in two files.
    rows = [(0,0,(0,0,0)), (3,3,(0,0,0)), (0,0,(0,0,0)), (1,1,(1,0,0)), (1,1,(1,0,0)),
            (1,1,(1,0,0)), (2,2,(17,0,0))]
    _write_pool(source, rows, split=split)
    manifest = identity_filter.run_identity_filter(source, None, single)
    inputs, outputs = _records(source), _records(single)
    assert sorted(inputs) == sorted(rows)
    mapping = load_npy(single / "map-1.npy").astype(np.int64)
    assert len(outputs) == len(set(rows)) == 4
    assert sorted(mapping[:, 0].tolist()) == list(range(len(inputs)))
    assert all(inputs[parent] == outputs[child] for parent, child in mapping.tolist())
    assert manifest["map_1_rows"] == len(inputs)


def test_single_identity_reads_each_source_file_once(tmp_path, monkeypatch):
    source = tmp_path / "source"
    _write_pool(source, [(0,0,(0,0,0)), (1,1,(1,0,0)), (1,1,(1,0,0)), (2,2,(17,0,0))], split=2)
    reads = Counter()
    original = identity_filter._read_loaded

    def counted(entry):
        reads[entry.path] += 1
        return original(entry)

    monkeypatch.setattr(identity_filter, "_read_loaded", counted)
    identity_filter.run_identity_filter(source, None, tmp_path / "single")
    assert reads == Counter({path: 1 for path in discover_organized(source)})


def test_single_identity_rejects_directory_without_organized_pose_files(tmp_path):
    source = tmp_path / "empty-directory"
    source.mkdir()
    with pytest.raises(ValueError, match="no organized pose files found"):
        identity_filter.run_identity_filter(source, None, tmp_path / "output")


def test_identity_cli_has_explicit_single_mode_and_preserves_three_positionals(tmp_path):
    source, single, two = tmp_path / "source", tmp_path / "single", tmp_path / "two"
    _write_pool(source, [(0,0,(0,0,0)), (1,1,(1,0,0)), (1,1,(1,0,0))])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "alaric"), str(ROOT), env.get("PYTHONPATH", "")))
    for args in (("--single-input", str(source), str(single)), (str(source), str(source), str(two))):
        completed = subprocess.run([sys.executable, str(ROOT / "alaric/identity_filter.py"), *args],
                                   cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
        assert completed.returncode == 0, completed.stdout + completed.stderr
    assert _records(single) == _records(two)
    assert (two / "map-1.npy").exists() and (two / "map-2.npy").exists()
