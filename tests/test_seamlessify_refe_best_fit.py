"""Contract tests for the importable best-fit Transformer factory."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PDB = ROOT / "tests" / "1b7f" / "pdbs" / "rna.pdb"
META_PIN = "META__FILE__fraglib.yaml"


@pytest.mark.parametrize("import_style", ["namespace", "top_level"])
def test_factory_import_styles_do_not_initialize_seamless(import_style, tmp_path):
    util_dir = ROOT / "util"
    if import_style == "namespace":
        cwd = ROOT
        import_statement = (
            "from util.seamlessify_refe_best_fit import seamlessify_refe_best_fit"
        )
    else:
        cwd = tmp_path
        import_statement = (
            f"import sys; sys.path.insert(0, {str(util_dir)!r}); "
            "from seamlessify_refe_best_fit import seamlessify_refe_best_fit"
        )
    script = f"""
import seamless_config
def fail(*args, **kwargs):
    raise AssertionError('factory import initialized seamless_config')
seamless_config.init = fail
{import_statement}
assert callable(seamlessify_refe_best_fit)
"""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_factory_returns_unexecuted_transformer_with_pins_and_context_binding(
    tmp_path, monkeypatch
):
    import seamless_config

    def fail_init(*args, **kwargs):
        raise AssertionError("factory or Context assembly initialized seamless_config")

    monkeypatch.setattr(seamless_config, "init", fail_init)

    from seamless_transformer.transformer_class import BashTransformer
    from seamless import Checksum
    from seamless_workflow import Context
    import util.seamlessify_refe_best_fit as factory

    config = tmp_path / "fraglib.yaml"
    config.write_text("conformers: /library/conformers\nfraglen: 2\n")
    calls = []

    def fake_files_to_checksums(filelist, **kwargs):
        calls.append((list(filelist), kwargs))
        checksums = {
            path: ("b" if Path(path) == config else "a") * 64 for path in filelist
        }
        return checksums, {}

    monkeypatch.setattr(factory, "files_to_checksums", fake_files_to_checksums)
    monkeypatch.setattr(
        BashTransformer,
        "run",
        lambda *args, **kwargs: pytest.fail("factory executed its Transformer"),
    )
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.iterdir())
    fitter_arguments = ("--exclude-pdb-code", "1b7f", "--grid-spacing", "1.25")

    transformer = factory.seamlessify_refe_best_fit(
        PDB,
        fitter_arguments,
        fraglib_config=config,
        conda_environment="best-fit-worker",
    )

    assert isinstance(transformer, BashTransformer)
    assert transformer.language == "bash"
    code = transformer.code.content.decode()
    assert "--exclude-pdb-code 1b7f --grid-spacing 1.25" in code
    assert transformer.environment.get_conda_env() == "best-fit-worker"
    assert transformer.celltypes[META_PIN] == "bytes"
    assert transformer.args[META_PIN].checksum == Checksum("b" * 64)
    assert transformer.celltypes["refe-best-fit.py"] == "bytes"
    assert transformer.args["refe-best-fit.py"].checksum == Checksum("a" * 64)
    assert transformer.celltypes["AA-ppdb.npy"] == "bytes"
    assert transformer.args["AA-ppdb.npy"].checksum == Checksum("a" * 64)
    assert transformer.celltypes.fraglib_identities == "str"
    identity_pin = transformer.args.fraglib_identities.value
    assert isinstance(identity_pin, str)
    identities = json.loads(identity_pin)
    assert identities["fraglen"] == 2
    assert len(identities["sha256"]) == 36

    filelist, options = calls[0]
    assert str(config) in filelist
    assert str(PDB) in filelist
    assert str(ROOT / "util" / "refe-best-fit.py") in filelist
    assert str(ROOT / "util" / "refe_best_fit_seamless_support.py") in filelist
    assert options == {
        "directories": {},
        "max_upload_files": None,
        "max_upload_size": None,
        "auto_confirm": "yes",
    }
    assert set(tmp_path.iterdir()) == before

    with Context() as context:
        context.step = transformer
        assert context.step.language == "bash"
        assert context.step.celltypes[META_PIN] == "bytes"
        assert context.step.args[META_PIN].checksum == Checksum("b" * 64)
        assert context.step.environment["conda_environment"] == "best-fit-worker"
        assert context.step.celltypes.fraglib_identities == "str"


def test_factory_rejects_managed_pdb_output_argument(tmp_path, monkeypatch):
    import util.seamlessify_refe_best_fit as factory

    monkeypatch.setattr(
        factory,
        "files_to_checksums",
        lambda filelist, **kwargs: ({path: "a" * 64 for path in filelist}, {}),
    )
    config = tmp_path / "fraglib.yaml"
    config.write_text("fraglen: 2\n")
    with pytest.raises(ValueError, match="--pdb-output is managed internally"):
        factory.seamlessify_refe_best_fit(
            PDB,
            ("--pdb-output=/tmp/managed",),
            fraglib_config=config,
        )
