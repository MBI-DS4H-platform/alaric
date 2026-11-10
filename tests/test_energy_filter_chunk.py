"""Chunked energy-filter replay equals the whole-action cascade (synthetic minfor oracle).

The real-minfor equivalence is exercised by deploying an energy-filter action with a chunk
deployer; this checks the speculation/replay bookkeeping over many chunkings and exits.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "alaric")]

import energy_filter_action as efa  # noqa: E402
import energy_filter_chunk as efc  # noqa: E402

N = 400
THRESHOLD = 0.0
STEPS = [
    {"plateau_distance": 5.0, "neighbour_distance": 7.0, "max_error": 1.0},
    {"plateau_distance": 10.0, "neighbour_distance": 12.0, "max_error": 0.05},
]
_rng = np.random.default_rng(7)
EXACT = _rng.normal(0.0, 2.0, N)
GRID = [EXACT + _rng.uniform(-0.45, 0.45, N), EXACT + _rng.uniform(-0.02, 0.02, N)]
UNSAFE = [np.arange(N) % 11 == 0, np.arange(N) % 13 == 0]


class FakeInvoker:
    """Classify prepared poses as minfor --filter would, from fixed synthetic energies."""

    def __init__(self, work, nposes, **kwargs):
        self.ids = np.load(Path(work) / "ids.npy")
        assert len(self.ids) == nposes
        self.count = 0

    def __call__(self, indices, step):
        self.count += 1
        local = np.arange(len(self.ids), dtype=np.uint64) if indices is None else np.asarray(indices, dtype=np.uint64)
        ids = self.ids[local.astype(np.intp)]
        if step is None:
            white = EXACT[ids] < THRESHOLD
            return local[white], np.empty(0, dtype=np.uint64), np.zeros(0, dtype=bool)
        index = next(i for i, s in enumerate(STEPS) if s == step)
        grid, margin = GRID[index][ids], step["max_error"]
        white = grid < THRESHOLD - margin
        grey = ~white & (grid < THRESHOLD + margin)
        return local[white], local[grey], UNSAFE[index][ids[grey]]


def _fake_prepare(pose_dir, protein, sequence, exclude, work, *, pose_range=None, pose_indices=None):
    if pose_range is not None:
        ids = np.arange(pose_range[0] - 1, pose_range[1])
    elif pose_indices is not None:
        ids = np.load(pose_indices).astype(np.int64)
    else:
        ids = np.arange(N)
    np.save(Path(work) / "ids.npy", ids)
    np.save(Path(work) / "poses.rotvec.npy", np.zeros((len(ids), 6)))


def _fake_publish(pose_dir, protein, output_dir, kept_ids, stats, work, *, invocations, **kwargs):
    return {"kept": kept_ids, "stats": stats, "invocations": invocations}


@pytest.fixture
def fake_minfor(monkeypatch):
    monkeypatch.setattr(efa, "_prepare_minfor_inputs", _fake_prepare)
    monkeypatch.setattr(efa, "_MinforInvoker", FakeInvoker)
    monkeypatch.setattr(efa, "_publish", _fake_publish)
    monkeypatch.setattr(efc, "_total_poses", lambda pose_dir: N)


def _whole(minimum):
    work_ids = {"ids": np.arange(N)}

    class Whole(FakeInvoker):
        def __init__(self):
            self.ids, self.count = work_ids["ids"], 0

    invoke = Whole()
    kept, stats = efa._cascade_minfor(N, STEPS, minimum, invoke)
    return kept, stats, invoke.count


def _args(tmp_path, mode, minimum, nchunks, **extra):
    protein = tmp_path / "protein-aa.pdb"
    protein.write_text("HEADER\n")
    argv = [mode, "--pose-dir", str(tmp_path), "--protein", str(protein), "--sequence", "GU",
            "--fragment", "3", "--protein-range", "range-2", "--threshold", str(THRESHOLD),
            "--epsilon", "15", "--cascade", json.dumps(STEPS), "--min-cascade", str(minimum),
            "--pdb-exclude", "1b7f", "--nchunks", str(nchunks), "--chunks-dir", str(tmp_path / "chunks")]
    for key, value in extra.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    return efc.build_parser().parse_args(argv)


@pytest.mark.parametrize("minimum", [0, 60, 200, 1000])
@pytest.mark.parametrize("nchunks", [1, 3, 7, 450])
def test_chunked_replay_matches_whole_action(tmp_path, fake_minfor, minimum, nchunks):
    kept, stats, invocations = _whole(minimum)
    for index in range(1, nchunks + 1):
        efc.run_chunk(_args(tmp_path, "chunk", minimum, nchunks, chunk_index=index))
    result = efc.run_assemble(_args(tmp_path, "assemble", minimum, nchunks, output=tmp_path / "out"))
    np.testing.assert_array_equal(result["kept"], kept)
    np.testing.assert_array_equal(kept, np.flatnonzero(EXACT < THRESHOLD))
    assert result["stats"] == stats
    assert result["invocations"] == invocations


def test_assemble_rejects_missing_and_stale_chunks(tmp_path, fake_minfor):
    for index in (1, 2):
        efc.run_chunk(_args(tmp_path, "chunk", 60, 3, chunk_index=index))
    with pytest.raises(FileNotFoundError, match="chunk-3"):
        efc.run_assemble(_args(tmp_path, "assemble", 60, 3, output=tmp_path / "out"))
    # A record from a deploy at another nchunks covers a different pose range.
    with pytest.raises(ValueError, match="stale chunk record.*last, nchunks"):
        efc.run_assemble(_args(tmp_path, "assemble", 60, 2, output=tmp_path / "out"))
    with pytest.raises(ValueError, match="stale chunk record.*settings"):
        efc.run_assemble(_args(tmp_path, "assemble", 61, 3, output=tmp_path / "out"))


def test_chunk_retry_skips_completed_output(tmp_path, fake_minfor, monkeypatch, capsys):
    args = _args(tmp_path, "chunk", 60, 3, chunk_index=1)
    record = efc.run_chunk(args)
    final = args.chunks_dir / "chunk-1"
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in final.iterdir()}

    def unexpected_prepare(*args, **kwargs):
        pytest.fail("completed chunk was recomputed")

    monkeypatch.setattr(efa, "_prepare_minfor_inputs", unexpected_prepare)
    assert efc.run_chunk(args) == record
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in final.iterdir()} == before
    assert "already complete; skipping" in capsys.readouterr().err


@pytest.mark.parametrize("stage", ["prepare", "publish"])
def test_chunk_retry_recomputes_unpublished_output(tmp_path, fake_minfor, monkeypatch, stage):
    args = _args(tmp_path, "chunk", 60, 3, chunk_index=1)

    def terminate(*args, **kwargs):
        raise SystemExit("simulated termination")

    with monkeypatch.context() as patch:
        if stage == "prepare":
            patch.setattr(efa, "_prepare_minfor_inputs", terminate)
        else:
            patch.setattr(Path, "rename", terminate)
        with pytest.raises(SystemExit, match="simulated termination"):
            efc.run_chunk(args)
    partial = args.chunks_dir / "chunk-1.partial"
    assert partial.is_dir()
    assert (partial / "record.json").exists() == (stage == "publish")
    assert not (args.chunks_dir / "chunk-1").exists()
    (partial / "leftover").write_text("interrupted output")
    efc.run_chunk(args)
    assert not partial.exists()
    final = args.chunks_dir / "chunk-1"
    assert (final / "record.json").is_file()
    assert not (final / "leftover").exists()


@pytest.mark.parametrize("change", ["nchunks", "min_cascade", "protein"])
def test_chunk_retry_recomputes_incompatible_output(tmp_path, fake_minfor, change):
    args = _args(tmp_path, "chunk", 60, 3, chunk_index=1)
    original = efc.run_chunk(args)
    if change == "protein":
        args.protein.write_text("HEADER changed protein\n")
    else:
        setattr(args, change, getattr(args, change) + 1)
    updated = efc.run_chunk(args)
    assert updated != original
    assert json.loads((args.chunks_dir / "chunk-1" / "record.json").read_text()) == updated


def test_chunk_ranges_tile_the_pool():
    for total in (0, 1, 5, 400):
        for nchunks in (1, 3, 7):
            ranges = [efc.chunk_range(total, i, nchunks) for i in range(1, nchunks + 1)]
            ids = [x for first, last in ranges for x in range(first, last + 1)]
            assert ids == list(range(1, total + 1))


def _write_energy_project(root: Path) -> Path:
    import yaml

    data = root / "DATA"
    data.mkdir()
    (data / "constraints.json").write_text('{"schema":1,"frag4":{"sequence":"GU"},"pairs":[]}\n')
    (data / "pdbcode.txt").write_text("1ABC\n")
    (data / "range-2-aa.pdb").write_text("HEADER all atom range 2\n")
    for name, spec in {
        "frag4-pose": {"action": "anchor-refe", "fragment": "auto", "sequence": "auto",
                       "exclude": "auto", "nucleotide": "first", "ovrmsd": 0.75},
        "frag4-energy": {"action": "energy-filter", "input": "frag4-pose", "fragment": "auto",
                         "sequence": "auto", "exclude": "auto", "protein": "range-2",
                         "threshold": -2.5, "epsilon": 15, "cascade": STEPS, "min_cascade": 1000},
    }.items():
        (root / name).mkdir()
        (root / name / "alaric.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    (data / "reference.pdb").write_text("HEADER reference\n")
    return root / "frag4-energy"


@pytest.mark.parametrize("deployer", ["local", "local-chunk", "remote", "remote-chunk"])
def test_energy_filter_deploys_with_every_deployer(tmp_path, monkeypatch, deployer):
    import subprocess
    from types import SimpleNamespace

    sys.path.insert(0, str(ROOT))
    import alaric.middle.deploy as deploy_module

    action_dir = _write_energy_project(tmp_path)
    remote = deployer.startswith("remote")
    chunked = deployer.endswith("-chunk")
    if remote:
        for var, value in {"ALARIC_REMOTE_HOST": "cluster", "ALARIC_REMOTE_DEPLOYMENT_DIR": "/remote/deploy",
                           "ALARIC_REMOTE_ALARIC_DIR": "/remote/alaric", "ALARIC_REMOTE_RESULT_DIR": "/remote/results",
                           "ALARIC_REMOTE_SCRATCH_DIR": "/scratch/alaric"}.items():
            monkeypatch.setenv(var, value)
        monkeypatch.setattr(deploy_module.subprocess, "run",
                            lambda args, check=False, **kwargs: SimpleNamespace(returncode=0))
    deploy_module.deploy(deployer, action_dir, nchunks=3 if chunked else None)
    for script in action_dir.glob("*.sh"):
        subprocess.run(["bash", "-n", str(script)], check=True)
    run = (action_dir / "run.sh").read_text()
    if not chunked:
        assert not (action_dir / "chunk1.sh").exists()
        assert "energy_filter_action.py" in run
        assert ("ALARIC_REMOTE_SCRATCH_DIR" in run) == remote
        return
    chunks = [(action_dir / f"chunk{i}.sh").read_text() for i in (1, 2, 3)]
    organize = (action_dir / "organize.sh").read_text()
    for index, body in enumerate(chunks, start=1):
        assert "energy_filter_chunk.py chunk" in body
        assert f"--chunk-index {index} " in body
        assert "--nchunks 3 " in body
        assert "assemble" not in body
    assert "energy_filter_chunk.py assemble" in organize
    assert "--nchunks 3 " in organize
    assert "result_sidecar.py pose" in organize
    assert [f"bash ./chunk{i}.sh" in run for i in (1, 2, 3)] == [True] * 3
    if remote:
        # Independently submitted chunks share records on the result FS; run.sh, which
        # runs every chunk itself, keeps them on node-local scratch.
        assert "--chunks-dir ${ENERGY_FILTER_CHUNKS}" in chunks[0]
        assert 'ENERGY_FILTER_CHUNKS="${ALARIC_ENERGY_FILTER_CHUNKS_DIR:-${ALARIC_REMOTE_RESULT_DIR:?}/' in organize
        assert "-CHUNKS}" in organize
        assert "ALARIC_ENERGY_FILTER_CHUNKS_DIR" in run and "mktemp -d" in run
        assert "mv ${ALARIC_REMOTE_RESULT_DIR:?}/" in organize
    else:
        sigil = (action_dir / "sigil.txt").read_text().strip()
        assert f"--chunks-dir ../CACHE/results/{sigil}-CHUNKS" in chunks[0]
        assert f"rm -rf ../CACHE/results/{sigil}-CHUNKS" in run
