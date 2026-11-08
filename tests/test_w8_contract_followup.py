"""W8 follow-up checks for action deployment, dependency closure and RNA breaks.

The retired synthetic export-bundle checks moved to real input-pin and rendered-tree
acceptance tests in Preparic. This file keeps the independent Alaric action checks.
"""
from itertools import product
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import re
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "alaric"), str(ROOT / "tests")]

@pytest.mark.parametrize("deployer", ["local", "local-chunk", "remote", "remote-chunk"])
def test_energy_filter_backend_context_renders_real_deployed_template(tmp_path, deployer):
    from alaric.energy_filter_action import build_parser
    from alaric.middle.backend import template_context
    from alaric.middle.deploy import ORGANIZE_DELIM, render_action_template
    from alaric.middle.project import Project
    from alaric.middle.resolve import ResolvedAction

    source_dir = tmp_path / "frag3-bestfit"
    source_dir.mkdir()
    (source_dir / "sigil.txt").write_text("source-pose-sigil\n")
    source = ResolvedAction("frag3-bestfit", source_dir, "anchor-refe",
                            {"fragment": 3, "sequence": "GU"}, {})
    action = ResolvedAction("frag3-energy-range-2", tmp_path / "filter", "energy-filter", {
        "input": source, "__protein": "range-2-aa.pdb", "protein": "range-2",
        "fragment": 3, "sequence": "GU", "exclude": ["1b7f"],
        "threshold": -2.0, "epsilon": 15, "cascade": [], "min_cascade": 0,
        "nprocs": 2, "chunk_poses": 999,
    }, {})
    arguments = dict(alaric_dir=str(ROOT / "alaric"), output_dir="../CACHE/results/filtered",
                     location="local")
    context = template_context(action, **arguments)
    assert shlex.split(context["protein_range"]) == ["range-2"]
    # Use deployment's real template discovery and renderer, with no injected
    # context that could accidentally conceal a missing backend field.
    chunked = deployer.endswith("-chunk")
    extra = {"nchunks": 3, "chunk_index": 2} if chunked else None
    lines = render_action_template(Project(tmp_path), action, deployer=deployer,
                                   extra_context=extra, **arguments)
    script = "\n".join(lines)
    assert re.search(r"\{\{\s*[a-z_][a-z_0-9]*\s*\}\}", script) is None

    def command_of(text):
        # Drop the remote templates' TMPDIR export; the action command is what remains.
        text = "\n".join(line for line in text.splitlines() if not line.startswith("export "))
        return shlex.split(text.replace("\\\n", " "), comments=True)

    if not chunked:
        command = command_of(script)
        assert Path(command[1]).name == "energy_filter_action.py"
        parsed = [build_parser().parse_args(command[2:])]
    else:
        from alaric.energy_filter_chunk import build_parser as chunk_parser

        assert script.count(ORGANIZE_DELIM) == 1
        chunk_text, organize_text = script.split(ORGANIZE_DELIM)
        parsed = []
        for text, mode in ((chunk_text, "chunk"), (organize_text, "assemble")):
            command = command_of(text)
            assert Path(command[1]).name == "energy_filter_chunk.py"
            assert command[2] == mode
            args = chunk_parser().parse_args(command[2:])
            assert args.nchunks == 3
            assert args.chunks_dir.as_posix() == "../CACHE/results/filtered-CHUNKS"
            parsed.append(args)
        assert parsed[0].chunk_index == 2
        assert parsed[1].output.as_posix() == "../CACHE/results/filtered"
    for args in parsed:
        assert args.protein_range == "range-2"
        assert args.protein.name == "range-2-aa.pdb"
        assert args.pose_dir.as_posix() == "../CACHE/results/source-pose-sigil"
        assert args.pdb_exclude == ["1b7f"]
        assert args.chunk_poses == 999
    assert parsed[-1].nprocs == 2
    if deployer.startswith("remote"):
        assert script.count('export TMPDIR="${ALARIC_REMOTE_SCRATCH_DIR:-') == (2 if chunked else 1)


def test_reference_fragment_deps_pin_only_minimal_backbone_helper():
    entries = (ROOT / "util/refe-fragments.DEPS").read_text().splitlines()
    templates = {f"../fraglib/templates/{''.join(seq)}-ppdb.npy"
                 for length in (1, 2) for seq in product("ACGU", repeat=length)}
    code = {"refe-fragments.py", "../alaric/parse_pdb.py", "../alaric/rna_pdb.py",
            "../alaric/write_pdb.py", "../alaric/backbone.py"}
    assert set(entries) == code | templates
    assert len(entries) == len(set(entries)), "dependency lines must be unique"
    assert len({Path(entry).name for entry in entries}) == len(entries)
    assert "../alaric/reference.py" not in entries


@pytest.mark.parametrize("manifest,script_entries,module_names", [
    ("refe-best-fit.DEPS", {"refe-best-fit.py", "refe_best_fit_seamless_support.py"},
     {"library", "parse_pdb", "reference", "superimpose", "write_pdb", "rna_pdb", "mutate", "backbone"}),
    ("refe-best-fit-tables.DEPS", {"refe-best-fit-tables.py", "refe-best-fit-pairs.py", "refe-best-fit-bases.py"},
     {"library", "parse_pdb", "reference", "rna_pdb", "superimpose", "mutate", "backbone"}),
], ids=["best-fit", "best-fit-tables"])
def test_reference_consumers_declare_exact_backbone_dependency_closure(manifest, script_entries, module_names):
    entries = (ROOT / "util" / manifest).read_text().splitlines()
    templates = {f"../fraglib/templates/{''.join(seq)}-ppdb.npy"
                 for length in (1, 2) for seq in product("ACGU", repeat=length)}
    expected = script_entries | {f"../alaric/{name}.py" for name in module_names} | templates
    assert "../alaric/backbone.py" in entries
    assert set(entries) == expected
    assert len(entries) == len(expected)
    assert len({Path(entry).name for entry in entries}) == len(entries)


@pytest.fixture
def broken_backbone():
    from alaric.parse_pdb import parse_pdb
    from alaric.reference import Reference

    atoms = parse_pdb((ROOT / "tests/1b7f/pdbs/rna.pdb").read_text())
    resids = list(dict.fromkeys(int(resid) for resid in atoms["resid"]))[:4]
    atoms = atoms[np.isin(atoms["resid"], resids)].copy()
    # Renumber to a non-one start with consecutive labels. Moving the complete
    # downstream block breaks only link 2→3, preserving links 1→2 and 3→4.
    for index, resid in enumerate(resids):
        atoms["resid"][atoms["resid"] == resid] = 101 + index
    kwargs = dict(rna=True, ignore_unknown=False, ignore_missing=False, ignore_reordered=False)
    intact = Reference(atoms, **kwargs)
    assert intact.get_fragment_positions(2) == [1, 2, 3]
    broken = atoms.copy()
    broken["x"][broken["resid"] >= 103] += 100.0
    reference = Reference(broken, **kwargs)
    assert reference.get_fragment_positions(2) == [1, 3]
    return broken, reference


def test_rmsd_rejects_consecutive_label_chemical_break_consistently(broken_backbone):
    from alaric.rmsd import _infer_fragment

    broken, reference = broken_backbone
    with pytest.raises(ValueError):
        _infer_fragment(broken, 2)
    # The two-residue implicit selector must enforce the same chemical rule.
    with pytest.raises(ValueError):
        _infer_fragment(broken[np.isin(broken["resid"], [102, 103])], None)
    for position in reference.get_fragment_positions(2):
        sequence, selected = _infer_fragment(broken, position)
        assert sequence == reference.get_sequence(position, 2)
        np.testing.assert_array_equal(np.unique(selected["resid"]), [1, 2])
        coords = np.column_stack((selected["x"], selected["y"], selected["z"]))
        np.testing.assert_allclose(coords, reference.get_coordinates(position, 2), rtol=0, atol=0)


def test_reference_fragment_cli_omits_only_chemical_break_window(broken_backbone, tmp_path):
    from alaric.parse_pdb import parse_pdb
    from alaric.write_pdb import write_pdb

    broken, reference = broken_backbone
    source = tmp_path / "broken-reference.pdb"
    source.write_text(write_pdb(broken))
    output = tmp_path / "fragments"
    result = subprocess.run([
        sys.executable, str(ROOT / "util/refe-fragments.py"), str(source),
        "--templates", str(ROOT / "fraglib/templates"), "--output", str(output),
    ], check=False, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    expected = {f"frag-{pos}-{reference.get_sequence(pos, 2)}-refe.pdb"
                for pos in reference.get_fragment_positions(2)}
    assert {path.name for path in output.iterdir()} == expected
    for position in reference.get_fragment_positions(2):
        path = output / f"frag-{position}-{reference.get_sequence(position, 2)}-refe.pdb"
        emitted = parse_pdb(path.read_text())
        coords = np.column_stack((emitted["x"], emitted["y"], emitted["z"]))
        np.testing.assert_allclose(coords, reference.get_coordinates(position, 2), rtol=0, atol=0.001)
