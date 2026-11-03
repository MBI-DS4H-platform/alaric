"""Focused W6 regressions without generating or modifying installed caches."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alaric import gaussian_cache as gc
from alaric import gaussian_filter as gf


@pytest.fixture
def tiny_install(tmp_path):
    import zstandard

    directory = tmp_path / "precomputed-filters/gaussian/rrm95"
    directory.mkdir(parents=True)
    files = {}
    for name in gc.FILE_NAMES:
        stream = io.BytesIO()
        np.save(stream, np.arange(3, dtype=np.int32))
        raw = stream.getvalue()
        compressed = zstandard.ZstdCompressor(level=1).compress(raw)
        (directory / f"{name}.npy.zst").write_bytes(compressed)
        files[name] = {
            "npy_sha256": hashlib.sha256(raw).hexdigest(),
            "sha256": hashlib.sha256(compressed).hexdigest(),
        }
    spec = {"thresholds": {"rrm95": {"precalc": {
        "dir": str(directory.relative_to(tmp_path)), "files": files,
    }}}}
    return tmp_path, directory, spec, raw


@pytest.mark.parametrize("threaded", [False, True])
def test_verifier_resolves_mixed_forms_and_propagates_bad_preferred_npy(tiny_install, threaded):
    import zstandard

    root, directory, spec, raw = tiny_install
    (directory / "rotations.npy").write_bytes(raw)
    # Force the decompressed identity path independently of zstd version/level.
    files = spec["thresholds"]["rrm95"]["precalc"]["files"]
    files["translations"]["sha256"] = "0" * 64
    (directory / "translations.npy.zst").write_bytes(
        zstandard.ZstdCompressor(level=19).compress(raw)
    )
    pool = ThreadPoolExecutor(max_workers=4) if threaded else None
    try:
        resolved = gc.verify_filter_cache(spec, "rrm95", root, executor=pool)
        assert resolved == {
            name: directory / (f"{name}.npy" if name == "rotations" else f"{name}.npy.zst")
            for name in gc.FILE_NAMES
        }
        _, status = gc.verify_file(directory / "translations.npy", "0" * 64,
                                   files["translations"]["npy_sha256"])
        assert status == "recompressed"
        # A good .zst beside the preferred bad .npy must never hide corruption.
        (directory / "rotations.npy").write_bytes(b"corrupt")
        with pytest.raises(gc.GaussianCacheInstallationError) as error:
            gc.verify_filter_cache(spec, "rrm95", root, executor=pool)
        assert gc.generator_command("rrm95", root) in str(error.value)
        assert "wrong checksum" in str(error.value)
        (directory / "rotations.npy").unlink()
        (directory / "rotations.npy.zst").unlink()
        with pytest.raises(gc.GaussianCacheInstallationError) as error:
            gc.verify_filter_cache(spec, "rrm95", root, executor=pool)
        assert gc.generator_command("rrm95", root) in str(error.value)
        assert "missing" in str(error.value)
        assert not (directory / "rotations.npy.zst").exists()
    finally:
        if pool is not None:
            pool.shutdown(wait=True)


@pytest.mark.parametrize("sequences", [None, (), ("GU",), ("GU", "AC", "GU")])
def test_library_config_loads_only_requested_sequences(monkeypatch, tmp_path, sequences):
    from alaric import library

    loaded = []
    templates = []
    sentinel = np.zeros((1, 3))

    def load_template(path):
        templates.append(Path(path).name.removesuffix("-ppdb.npy"))
        return sentinel

    def load_library(**kwargs):
        assert kwargs["template"] is sentinel
        loaded.append(kwargs["sequence"])
        return object()

    monkeypatch.setattr(library, "LibraryDirectory", lambda **kwargs: SimpleNamespace(load=load_library))
    monkeypatch.setattr(library, "_resolve_templates_dir", lambda *args: tmp_path)
    monkeypatch.setattr(library, "load_mononucleotide_templates", lambda: {})
    monkeypatch.setattr(library.np, "load", load_template)
    libraries, actual_templates = library.config(verify_checksums=False, sequences=sequences)
    expected = {a + b for a in "ACGU" for b in "ACGU"} if sequences is None else set(sequences)
    assert set(libraries) == set(actual_templates) == expected
    assert len(loaded) == len(templates) == len(expected)
    assert set(loaded) == set(templates) == expected


def test_a9_pose_predicate_obeys_json_goldens_and_inclusive_cutoff():
    from alaric.gaussian_check import _pose_pass

    spec = gf.load_spec(ROOT / "alaric/gaussian_stacking_filter.json")
    for golden in spec["golden"]["decisions"]:
        # The committed descriptor goldens cover the strict masks/prefilters.
        job = SimpleNamespace(distance=golden["d"], phi=math.radians(golden["phi_deg"]),
                              theta=math.radians(golden["theta_deg"]),
                              score=float(gf.score(golden["d"], math.radians(golden["phi_deg"]),
                                                   math.radians(golden["theta_deg"]))))
        name = golden["name"]
        assert _pose_pass(job, name, spec["thresholds"][name], spec) is golden["accept"]
    for name, entry in spec["thresholds"].items():
        job = SimpleNamespace(distance=3.0, phi=0.0, theta=0.0, score=float(entry["cutoff"]))
        assert _pose_pass(job, name, entry, spec)
        job.score = np.nextafter(job.score, -math.inf)
        assert not _pose_pass(job, name, entry, spec)
        job.score = 1e20
        job.distance = float(spec["mask"]["distance_lt_A"])
        assert not _pose_pass(job, name, entry, spec)
        job.distance = 3.0
        job.phi = math.radians(float(spec["mask"]["displacement_lt_deg"]))
        assert not _pose_pass(job, name, entry, spec)
