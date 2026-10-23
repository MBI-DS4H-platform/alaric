"""Library identities come from precomputed sidecars, never payload reads."""
from pathlib import Path
import runpy


SUPPORT = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "util" / "refe_best_fit_seamless_support.py")
)


def test_library_identity_uses_sidecars_and_survives_relocation(tmp_path):
    roots = []
    for location in ("original", "relocated"):
        root = tmp_path / location
        for directory in ("conformers", "rotamers"):
            (root / directory).mkdir(parents=True)
            (root / directory / "library.npy.CHECKSUM").write_text("a" * 64 + "\n")
        roots.append(root)

    original = SUPPORT["identities"](roots[0])
    assert SUPPORT["identities"](roots[1]) == original
    (roots[1] / "conformers" / "library.npy.CHECKSUM").write_text("b" * 64 + "\n")
    assert SUPPORT["identities"](roots[1]) != original
    assert str(tmp_path) not in str(original)


def test_input_declarations_are_original_files_without_preparation_writes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    inputs = SUPPORT["input_paths"]()
    root = Path(__file__).resolve().parents[1]
    manifest = SUPPORT["identities"](root / "fraglib")
    assert len(manifest["sha256"]) == 36
    assert len(inputs) == 28
    assert all(path.is_file() for path in inputs)
    assert any(path.name == "library.py" and path.parent.name == "alaric" for path in inputs)
    assert not list(tmp_path.iterdir())
