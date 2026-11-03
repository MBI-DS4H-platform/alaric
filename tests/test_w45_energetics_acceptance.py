"""Additional black-box A5 cases; deliberately share the W4 sandbox harness."""
from __future__ import annotations

import json
import shutil

from test_energetics_deps import (
    LIGAND, _assert_result, _reference_energy, _run_flat, _stage, _unknown_residue,
)


def test_receptor_failure_removes_previous_success_artifact(tmp_path):
    sandbox, ligands, receptor = _stage(tmp_path)
    first = _run_flat(sandbox, ligands, receptor)
    assert first.returncode == 0, first.stdout + first.stderr
    output = sandbox / "energies.json"
    assert output.is_file()
    receptor.write_text(_unknown_residue(receptor.read_text()))
    second = _run_flat(sandbox, ligands, receptor)
    assert second.returncode != 0, second.stdout + second.stderr
    assert not output.exists(), "a failed invocation retained the preceding success"


def test_empty_ligand_is_row_local_and_non_pdb_files_are_not_scored(tmp_path):
    reference, counts = _reference_energy(tmp_path)
    sandbox, ligands, receptor = _stage(tmp_path)
    (ligands / "unknown.pdb").write_text("")
    (ligands / "metadata.json").write_text('{"schema":1}')
    (ligands / "nested.pdb").mkdir()
    completed = _run_flat(sandbox, ligands, receptor)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads((sandbox / "energies.json").read_text())
    _assert_result(result, reference, counts)
    assert "unknown.pdb" not in result["ligands"], "empty ligand invented a bead count"


def test_reference_and_bestfit_names_remain_separate_with_equal_bytes(tmp_path):
    """T12 may contain both poses; A5 must not collapse by fragment number."""
    reference, counts = _reference_energy(tmp_path)
    sandbox, ligands, receptor = _stage(tmp_path)
    (ligands / "unknown.pdb").unlink()
    reference_name = LIGAND.name.replace("-bestfit.pdb", "-refe.pdb")
    shutil.copyfile(LIGAND, ligands / reference_name)
    completed = _run_flat(sandbox, ligands, receptor)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads((sandbox / "energies.json").read_text())
    assert set(result["ligands"]) == {reference_name, LIGAND.name}
    rows = {row["ligand"]: row for row in result["pairs"]}
    assert len(rows) == len(result["pairs"]) == 2
    for name in rows:
        assert result["ligands"][name] == {"beads": counts["ligand"]}
        assert rows[name]["error"] is None
        assert abs(rows[name]["energy"] - reference) <= 1e-6
    assert rows[reference_name]["energy"] == rows[LIGAND.name]["energy"]


def test_ligand_folder_relocation_preserves_scientific_output(tmp_path):
    """This proves output parity; graph identity is a separate acceptance check."""
    sandbox, ligands, receptor = _stage(tmp_path)
    first = _run_flat(sandbox, ligands, receptor)
    assert first.returncode == 0, first.stdout + first.stderr
    original = (sandbox / "energies.json").read_bytes()
    relocated = sandbox / "renamed-inputs"
    shutil.copytree(ligands, relocated)
    second = _run_flat(sandbox, relocated, receptor)
    assert second.returncode == 0, second.stdout + second.stderr
    assert (sandbox / "energies.json").read_bytes() == original
