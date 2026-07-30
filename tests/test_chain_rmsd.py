from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "util" / "chain_rmsd.py"
    spec = importlib.util.spec_from_file_location("chain_rmsd", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def chain_rmsd():
    return _load_module()


def _write_chain_dir(root: Path, *, table, fragments=(2, 3, 4), header=None) -> Path:
    chain_dir = root / "chains"
    chain_dir.mkdir()
    pools = [f"pool{fragment}" for fragment in fragments]
    for pool in pools:
        (chain_dir / pool).mkdir()
    (chain_dir / "chains.txt").write_text(
        "\t".join(header or pools)
        + "\n"
        + "\n".join("\t".join(str(value) for value in row) for row in table)
        + "\n"
    )
    (chain_dir / "chains.json").write_text(
        json.dumps(
            {
                "nchains": len(table),
                "chains_file": "chains.txt",
                "columns": [
                    {
                        "pool": pool,
                        "fragment": fragment,
                        "pose_dir": pool,
                        "nposes": 2,
                        "sequence": "AA",
                    }
                    for pool, fragment in zip(pools, fragments)
                ],
            }
        )
    )
    data = root / "DATA"
    data.mkdir()
    (data / "reference.pdb").write_text("HEADER fixture\n")
    (data / "pdbcode.txt").write_text("1b7f\n")
    return chain_dir


def _fake_rmsd_vectors(pose_dir, _reference, _library, *, chunksize):
    assert chunksize == 17
    vectors = {
        "pool2": ([10, 20], [1, 2], [2, 4]),
        "pool3": ([30, 31], [6, 7], [8, 9]),
        "pool4": ([40, 42], [9, 10], [11, 12]),
    }
    return tuple(np.asarray(value, dtype=np.float32) for value in vectors[pose_dir.name])


def _patch_calculation_dependencies(monkeypatch, module):
    monkeypatch.setattr(module.PoseReader, "get_nposes", lambda _path: 2)
    monkeypatch.setattr(module, "_reference_sequence", lambda _path, _fragment: "AA")
    library = type("Library", (), {"template": np.empty(4, dtype=[] )})()
    monkeypatch.setattr(
        module,
        "_load_library",
        lambda *_args, **_kwargs: (library, type("Factory", (), {"unload_rotaconformers": lambda self: None})()),
    )
    monkeypatch.setattr(
        module,
        "_load_reference_coordinates",
        lambda *_args, **_kwargs: ("AA", np.zeros((4, 3), dtype=np.float32)),
    )
    monkeypatch.setattr(module, "_rmsd_vectors", _fake_rmsd_vectors)


def test_chain_table_uses_pose_order_and_meaned_shared_nucleotides(
    tmp_path, monkeypatch, chain_rmsd
):
    chain_dir = _write_chain_dir(tmp_path, table=[[2, 1, 2], [1, 1, 1]])
    _patch_calculation_dependencies(monkeypatch, chain_rmsd)

    headers, values = chain_rmsd.calculate_chain_rmsds(chain_dir, chunksize=17)

    assert headers == ["chain_rmsd", "pool2", "pool3", "pool4"]
    expected_chain = [np.sqrt(63.5), 7.25]
    np.testing.assert_allclose(values[:, 0], expected_chain, rtol=0, atol=1e-6)
    np.testing.assert_allclose(values[:, 1:], [[20, 30, 42], [10, 30, 40]])

    output = tmp_path / "table.txt"
    chain_rmsd.write_table(output, headers, values)
    assert output.read_text() == (
        "chain_rmsd\tpool2\tpool3\tpool4\n"
        "7.969\t20.000\t30.000\t42.000\n"
        "7.250\t10.000\t30.000\t40.000\n"
    )


def test_sibling_data_inputs_are_required(tmp_path, chain_rmsd):
    chain_dir = tmp_path / "chains"
    chain_dir.mkdir()
    with pytest.raises(chain_rmsd.ChainRmsdError, match="sibling DATA"):
        chain_rmsd._data_inputs(chain_dir)

    data = tmp_path / "DATA"
    data.mkdir()
    with pytest.raises(chain_rmsd.ChainRmsdError, match="reference PDB"):
        chain_rmsd._data_inputs(chain_dir)
    (data / "reference.pdb").write_text("HEADER fixture\n")
    with pytest.raises(chain_rmsd.ChainRmsdError, match="excluded PDB code"):
        chain_rmsd._data_inputs(chain_dir)


def test_out_of_range_chain_pose_is_rejected(tmp_path, monkeypatch, chain_rmsd):
    chain_dir = _write_chain_dir(tmp_path, table=[[3, 1, 1]])
    monkeypatch.setattr(chain_rmsd.PoseReader, "get_nposes", lambda _path: 2)
    with pytest.raises(chain_rmsd.ChainRmsdError, match="out of range"):
        chain_rmsd.calculate_chain_rmsds(chain_dir)


def test_bad_chain_header_is_rejected(tmp_path, chain_rmsd):
    chain_dir = _write_chain_dir(
        tmp_path, table=[[1, 1, 1]], header=["wrong", "pool3", "pool4"]
    )
    with pytest.raises(chain_rmsd.ChainRmsdError, match="out of sync"):
        chain_rmsd.calculate_chain_rmsds(chain_dir)


def test_nonconsecutive_fragments_are_rejected(tmp_path, chain_rmsd):
    chain_dir = _write_chain_dir(tmp_path, table=[[1, 1]], fragments=(2, 4))
    with pytest.raises(chain_rmsd.ChainRmsdError, match="not consecutive"):
        chain_rmsd.calculate_chain_rmsds(chain_dir)


def test_missing_pose_dir_is_rejected(tmp_path, chain_rmsd):
    chain_dir = _write_chain_dir(tmp_path, table=[[1, 1, 1]])
    (chain_dir / "pool2").rmdir()
    with pytest.raises(chain_rmsd.ChainRmsdError, match="pose dir not found"):
        chain_rmsd.calculate_chain_rmsds(chain_dir)
