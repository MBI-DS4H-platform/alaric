from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import yaml
import zstandard as zstd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from alaric.mask import main as mask_main
from alaric.middle.checksum import byte_checksum, write_array_sidecar
from alaric.middle.deploy import deploy, generate_chunk_files
from alaric.middle.graph import ActionGraph
from alaric.middle.project import Project
from alaric.middle.sigil import compute_project_sigils
from alaric.score_add import main as score_add_main


def _write_project(root: Path) -> None:
    data = root / "DATA"
    data.mkdir()
    (data / "constraints.json").write_text(
        """
{
  "pdb_code": "1abc",
  "frag4": {"sequence": "GU", "scores": {"dom": -1.5}},
  "frag5": {"sequence": "UU", "scores": {"dom": -2.5}},
  "pairs": [{"down": "frag4", "up": "frag5", "cRMSD": 0.25, "ovRMSD": 0.75}]
}
""".strip()
        + "\n"
    )
    (data / "anchor.yaml").write_text("angle: 30\ndihedral: 45 -45\n")
    (data / "pdbcode.txt").write_text("1abc\n")
    (data / "dom-aa.pdb").write_text("HEADER test\n")
    (data / "dom.pdb").write_text("HEADER test plain\n")

    def action(name: str, spec: dict) -> None:
        path = root / name
        path.mkdir()
        (path / "alaric.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))

    action(
        "frag4-anchor",
        {
            "action": "anchor",
            "fragment": "auto",
            "sequence": "auto",
            "exclude": "auto",
            "protein": "dom",
            "resid": 1,
            "nucleotide": "first",
        },
    )
    action(
        "frag4-score",
        {
            "action": "score",
            "input": "frag4-anchor",
            "sequence": "auto",
            "exclude": "auto",
            "protein": "dom",
        },
    )
    action(
        "frag4-filter",
        {
            "action": "filter",
            "input": "frag4-anchor",
            "score_input": "frag4-score",
            "threshold": "auto",
        },
    )
    action(
        "frag5-fwd",
        {
            "action": "grow",
            "input": "frag4-filter",
            "fragment": "auto",
            "sequence": "auto",
            "exclude": "auto",
            "direction": "auto",
            "crmsd": "auto",
            "ovrmsd": "auto",
        },
    )


def test_sigil_is_deterministic_and_writes_parameters(tmp_path: Path) -> None:
    _write_project(tmp_path)
    first = compute_project_sigils(Project.discover(tmp_path))
    second = compute_project_sigils(
        Project.discover(tmp_path),
        force=True,
    )
    assert first == second
    assert first["frag4-anchor"].endswith("-null")
    assert (tmp_path / "CACHE" / "parameters" / first["frag4-score"]).is_file()


def test_deploy_score_chunk_emits_independent_chunk_and_organize_scripts(tmp_path: Path) -> None:
    _write_project(tmp_path)
    project = Project.discover(tmp_path)
    compute_project_sigils(project)
    d = tmp_path / "frag4-score"
    deploy("local-chunk", d, nchunks=3)

    # Each chunk is an independent, parallel-runnable script that discovers its own range.
    for i in (1, 2, 3):
        body = (d / f"chunk{i}.sh").read_text()
        assert f"IDX={i}" in body
        assert "PoseReader.get_nposes" in body
        assert '"$FIRST"' in body and '"$LAST"' in body
        assert "np.concatenate" not in body

    # The organize step (concatenation) lives in its own script.
    organize = (d / "organize.sh").read_text()
    assert "np.concatenate" in organize

    # run.sh is only a convenience wrapper: chunks one-by-one, then organize.
    run_sh = (d / "run.sh").read_text()
    assert "./check.sh" in run_sh
    for i in (1, 2, 3):
        assert f"bash ./chunk{i}.sh" in run_sh
    assert "bash ./organize.sh" in run_sh
    # The per-chunk body must NOT be inlined into run.sh.
    assert "PoseReader.get_nposes" not in run_sh


def test_remote_chunk_python_paths_are_expandvars_compatible(tmp_path: Path) -> None:
    _write_project(tmp_path)
    project = Project.discover(tmp_path)
    compute_project_sigils(project)
    action = ActionGraph(project).build()["frag4-score"]

    files = generate_chunk_files(project, action, "remote-chunk", nchunks=3)
    body = "\n".join(files.values())

    assert "os.path.expandvars('${ALARIC_REMOTE_RESULT_DIR}/" in body
    assert "os.path.expandvars('${ALARIC_REMOTE_RESULT_DIR:?}" not in body


def test_score_add_and_mask_validate_shapes(tmp_path: Path) -> None:
    a = tmp_path / "a.npy"
    b = tmp_path / "b.npy"
    out = tmp_path / "out.npy"
    np.save(a, np.array([1.0, 2.0, 3.0]))
    np.save(b, np.array([4.0, 5.0, 6.0]))
    assert score_add_main([str(a), str(b), str(out)]) == 0
    np.testing.assert_array_equal(np.load(out), np.array([5.0, 7.0, 9.0]))

    mask_out = tmp_path / "mask.npy"
    assert mask_main([str(out), "8.0", str(mask_out)]) == 0
    mask = np.load(mask_out)
    assert mask.ndim == 1

    np.save(b, np.array([1.0]))
    with pytest.raises(ValueError):
        score_add_main([str(a), str(b), str(out)])


def test_checksum_is_zstd_transparent(tmp_path: Path) -> None:
    payload = b"payload" * 100
    raw = tmp_path / "score.npy"
    compressed = tmp_path / "score.npy.zst"
    raw.write_bytes(payload)
    compressed.write_bytes(zstd.ZstdCompressor().compress(payload))
    assert byte_checksum(raw) == byte_checksum(compressed)
    checksum = write_array_sidecar(compressed)
    assert (tmp_path / "score.npy.CHECKSUM").read_text().strip() == checksum
