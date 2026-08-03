from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from alaric.middle.chain_focus import (  # noqa: E402
    CHAIN_PROVENANCE_FILE,
    ChainFocusError,
    focus_chains,
)


def _write_chains(path: Path) -> None:
    path.mkdir()
    for pool in ("r1", "r2", "r3"):
        (path / pool).mkdir()
    (path / "chains.txt").write_text(
        "r1\tr2\tr3\n"
        "1\t5\t2\n"
        "1\t5\t3\n"
        "2\t5\t2\n"
        "1\t5\t2\n"
    )
    (path / "chains.json").write_text(
        json.dumps(
            {
                "project": "/project",
                "chains_file": "chains.txt",
                "nchains": 4,
                "columns": [
                    {"pool": "r1", "fragment": 1, "pose_dir": "r1", "nposes": 2},
                    {"pool": "r2", "fragment": 2, "pose_dir": "r2", "nposes": 5},
                    {"pool": "r3", "fragment": 3, "pose_dir": "r3", "nposes": 3},
                ],
            }
        )
    )


def test_focus_links_selected_pools_and_deduplicates_rows(tmp_path):
    source = tmp_path / "chains"
    _write_chains(source)
    output = tmp_path / "focused"

    metadata = focus_chains(source, ["r3", "r1"], output)

    # Columns retain their source fragment order, and r2 is deliberately ignored.
    assert (output / "r1").is_symlink()
    assert (output / "r1").resolve() == source / "r1"
    assert os.readlink(output / "r1") == "../chains/r1"
    assert (output / "r3").is_symlink()
    assert not (output / "r2").exists()
    assert (output / "chains.txt").read_text() == "r1\tr3\n1\t2\n1\t3\n2\t2\n"
    assert (output / CHAIN_PROVENANCE_FILE).read_text() == "1\n2\n3\n1\n"
    assert metadata["nchains"] == 3
    assert [column["pool"] for column in metadata["columns"]] == ["r1", "r3"]
    assert metadata["chain_provenance_file"] == CHAIN_PROVENANCE_FILE
    assert json.loads((output / "chains.json").read_text()) == metadata


def test_focus_can_skip_chain_provenance(tmp_path):
    source = tmp_path / "chains"
    _write_chains(source)
    output = tmp_path / "focused"

    metadata = focus_chains(source, ["r1"], output, write_provenance=False)

    assert not (output / CHAIN_PROVENANCE_FILE).exists()
    assert "chain_provenance_file" not in metadata
    assert "chain_provenance_file" not in json.loads((output / "chains.json").read_text())


def test_focus_chunked_zstd_input_matches_existing_output(tmp_path):
    source = tmp_path / "chains"
    _write_chains(source)
    plain = source / "chains.txt"
    import zstandard as zstd

    plain.with_name("chains.txt.zst").write_bytes(
        zstd.ZstdCompressor().compress(plain.read_bytes())
    )
    plain.unlink()
    output = tmp_path / "focused"

    metadata = focus_chains(source, ["r3", "r1"], output, chunk_size=2)

    # The duplicate is in a different source chunk from its first occurrence.
    assert (output / "chains.txt").read_text() == "r1\tr3\n1\t2\n1\t3\n2\t2\n"
    assert (output / CHAIN_PROVENANCE_FILE).read_text() == "1\n2\n3\n1\n"
    assert metadata["nchains"] == 3


def test_focus_rejects_nonempty_output_dir(tmp_path):
    source = tmp_path / "chains"
    _write_chains(source)
    output = tmp_path / "focused"
    output.mkdir()
    (output / "keep").write_text("do not overwrite")

    try:
        focus_chains(source, ["r1"], output)
    except ChainFocusError as exc:
        assert "not empty" in str(exc)
    else:
        raise AssertionError("expected a nonempty output dir to be rejected")
