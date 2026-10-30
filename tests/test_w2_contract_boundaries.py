"""Supplemental W2 oracles for G6/G8 and A1's fixed scientific output.

These boundary tests do not execute the fitter or load scientific libraries.
The numerical and sandbox goldens remain in the original W2 acceptance set.
"""
from __future__ import annotations

import importlib.util
import json
from math import sqrt
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
VALID_IDENTITY = {
    "schema": 1,
    "fraglen": 2,
    "sha256": {"conformers/AA.npy": "a" * 64},
}


def _invalid_identities():
    return [
        pytest.param({key: value for key, value in VALID_IDENTITY.items()
                      if key != "schema"}, id="missing-schema"),
        pytest.param({**VALID_IDENTITY, "schema": 2}, id="unsupported-schema"),
        pytest.param({**VALID_IDENTITY, "schema": True}, id="boolean-schema"),
        pytest.param({**VALID_IDENTITY, "schema": 1.0}, id="float-schema"),
        pytest.param({**VALID_IDENTITY, "fraglen": True}, id="boolean-fraglen"),
        pytest.param({**VALID_IDENTITY, "fraglen": 2.0}, id="float-fraglen"),
        pytest.param({**VALID_IDENTITY, "fraglen": 3}, id="wrong-fraglen"),
        pytest.param({**VALID_IDENTITY, "sha256": {}}, id="empty-identities"),
        pytest.param({**VALID_IDENTITY, "sha256": {"conformers/AA.npy": "a" * 63}},
                     id="short-digest"),
        pytest.param({**VALID_IDENTITY, "sha256": {"conformers/AA.npy": "g" * 64}},
                     id="nonhex-digest"),
    ]


@pytest.mark.parametrize("identity", _invalid_identities())
def test_a1_rejects_invalid_identity_before_fitter_execution(tmp_path, monkeypatch, identity):
    import util.refe_best_fit_seamless_support as support

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("fraglib_config", str(tmp_path / "fraglib.yaml"))
    library = SimpleNamespace(_fraglib_config=lambda: {"fraglen": 2})
    monkeypatch.setattr(support, "library_module", lambda workspace: library)
    (tmp_path / "fraglib_identities").write_text(json.dumps(identity))
    (tmp_path / "rna_pdb").write_bytes(b"test RNA bytes")
    (tmp_path / "exclude_pdb_code").write_text("1b7f")
    monkeypatch.setattr(support.sys, "argv", ["support.py"])

    def must_not_fit(*args, **kwargs):
        pytest.fail("invalid identity reached the scientific fitter")

    monkeypatch.setattr(support.runpy, "run_path", must_not_fit)
    with pytest.raises(ValueError, match=r"(?i)schema|identity|identities|fraglen|sha256|manifest"):
        support.run([])


@pytest.fixture
def table_module():
    spec = importlib.util.spec_from_file_location(
        "w2_table_identity_boundary", ROOT / "util" / "refe-best-fit-tables.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("identity", _invalid_identities())
def test_a3_rejects_invalid_identity_before_table_computation(tmp_path, table_module, identity):
    path = tmp_path / "identity.json"
    path.write_text(json.dumps(identity))
    with pytest.raises(ValueError, match=r"(?i)schema|identity|identities|fraglen|sha256|manifest"):
        table_module._load_library_identity(path)


def test_a3_accepts_schema_one_identity_without_rehashing_payloads(tmp_path, table_module):
    path = tmp_path / "identity.json"
    path.write_text(json.dumps(VALID_IDENTITY))
    assert table_module._load_library_identity(path) == VALID_IDENTITY


FORBIDDEN_FITTER_ARGUMENTS = [
    pytest.param(("--no-header",), id="no-header"),
    pytest.param(("--ignore-unknown",), id="ignore-unknown"),
    pytest.param(("--ignore-missing",), id="ignore-missing"),
    pytest.param(("--ignore-reordered",), id="ignore-reordered"),
    pytest.param(("--grid-spacing", "1"), id="changed-grid"),
    pytest.param(("--grid-spacing=1",), id="changed-grid-equals"),
    pytest.param(("--grid-spacing", "0"), id="zero-grid"),
    pytest.param(("--grid-spacing", "nan"), id="nan-grid"),
    pytest.param(("--grid-spacing", "inf"), id="infinite-grid"),
    pytest.param(("--no-head",), id="abbreviated-no-header"),
    pytest.param(("--ignore-unk",), id="abbreviated-ignore-unknown"),
    pytest.param(("--grid-s=1",), id="abbreviated-grid-equals"),
    pytest.param(("--rotamer-p=2",), id="abbreviated-rotamer-equals"),
    pytest.param(("--exclude-p=1abc",), id="abbreviated-exclude-equals"),
]


@pytest.mark.parametrize("arguments", FORBIDDEN_FITTER_ARGUMENTS)
def test_a1_factory_rejects_output_changing_arguments_before_staging(monkeypatch, arguments):
    import util.seamlessify_refe_best_fit as factory

    def must_not_stage(*args, **kwargs):
        pytest.fail("forbidden fitter arguments reached factory file staging")

    monkeypatch.setattr(factory, "files_to_checksums", must_not_stage)
    with pytest.raises(ValueError):
        factory.seamlessify_refe_best_fit(arguments)


@pytest.mark.parametrize("arguments", [(), ("--grid-spacing", str(sqrt(3) / 3)),
                                        ("--grid-spacing=" + str(sqrt(3) / 3),),
                                        ("--rotamer-precision", "0.5")])
def test_a1_factory_accepts_contract_defaults(tmp_path, monkeypatch, arguments):
    import util.seamlessify_refe_best_fit as factory

    def staged_checksums(paths, **kwargs):
        return {str(path): "a" * 64 for path in paths}, {}

    monkeypatch.setattr(factory, "files_to_checksums", staged_checksums)
    config = tmp_path / "fraglib.yaml"
    config.write_text("fraglen: 2\n")
    transformer = factory.seamlessify_refe_best_fit(arguments, fraglib_config=config)
    assert transformer.celltypes.result == "folder"
    assert transformer.args.rna_pdb.checksum is None
    assert transformer.args.exclude_pdb_code.checksum is None
