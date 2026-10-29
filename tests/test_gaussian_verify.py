import shutil

import pytest

from gaussian_common import ROOT
from alaric import gaussian_cache as gc
from alaric import gaussian_precalc as gp


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("rrm50") / "rrm50"
    b = gp.build_filter("rrm50")
    gp.write_cache(d, b)
    return gp.write_manifest(d, "rrm50", b, "3.30384"), d


@pytest.fixture
def work(built, tmp_path):
    manifest, src = built
    dst = tmp_path / "rrm50"
    shutil.copytree(src, dst)
    return manifest["files"], dst


CMD = "gaussian-precalc"


def _verify(files, d, name="filtered-concat"):
    e = files[name]
    return gc.verify_file(d / f"{name}.npy", e["sha256"], e["npy_sha256"], command=CMD)


def test_zst_form(work):
    files, d = work
    path, status = _verify(files, d)
    assert status == "zst" and path.name == "filtered-concat.npy.zst"


def test_npy_form(work):
    files, d = work
    gc.decompress_dir(d)
    path, status = _verify(files, d)
    assert status == "npy" and path.name == "filtered-concat.npy"


def test_recompressed_level19(work):
    import zstandard

    files, d = work
    p = d / "filtered-concat.npy.zst"
    raw = zstandard.ZstdDecompressor().stream_reader(p.open("rb")).read()
    p.write_bytes(zstandard.ZstdCompressor(level=19).compress(raw))
    assert p.read_bytes() != b""
    path, status = _verify(files, d)
    assert status == "recompressed"


def test_flipped_byte_truncated_missing(work):
    files, d = work
    p = d / "rotations.npy.zst"
    data = bytearray(p.read_bytes())
    data[len(data) // 2] ^= 0xFF
    p.write_bytes(bytes(data))
    with pytest.raises(gc.GaussianCacheInstallationError, match=CMD):
        _verify(files, d, "rotations")
    p = d / "translations.npy.zst"
    p.write_bytes(p.read_bytes()[:-20])
    with pytest.raises(gc.GaussianCacheInstallationError, match=CMD):
        _verify(files, d, "translations")
    (d / "filtered-sizes.npy.zst").unlink()
    with pytest.raises(gc.GaussianCacheInstallationError, match=CMD) as ei:
        _verify(files, d, "filtered-sizes")
    assert files["filtered-sizes"]["npy_sha256"] in str(ei.value)


def test_npy_checked_when_both_exist(work):
    files, d = work
    e = files["filtered-sizes"]
    shutil.copy(d / "filtered-sizes.npy.zst", d / "keep.zst")
    gc.decompress_dir(d)
    shutil.copy(d / "keep.zst", d / "filtered-sizes.npy.zst")  # valid zst alongside
    (d / "filtered-sizes.npy").write_bytes(b"garbage")
    with pytest.raises(gc.GaussianCacheInstallationError):
        gc.verify_file(d / "filtered-sizes.npy", e["sha256"], e["npy_sha256"], command=CMD)


def test_generator_command_names_script():
    cmd = gc.generator_command("rrm95", ROOT)
    assert "util/gaussian-precalc.py rrm95 --output" in cmd and "gaussian-precalc-all.py" in cmd


def test_compare_manifest():
    entry = {"npy_sha256": "a" * 64, "sha256": "b" * 64}
    ok, notes = gc.compare_manifest({"files": {"rotations": dict(entry)}}, {"rotations": entry})
    assert ok and notes == []
    ok, notes = gc.compare_manifest(
        {"files": {"rotations": {"npy_sha256": "a" * 64, "sha256": "c" * 64}}}, {"rotations": entry}
    )
    assert ok and len(notes) == 1 and "recompressed" in notes[0]
    ok, notes = gc.compare_manifest(
        {"files": {"rotations": {"npy_sha256": "d" * 64, "sha256": "b" * 64}}}, {"rotations": entry}
    )
    assert not ok and "npy_sha256" in notes[0]
