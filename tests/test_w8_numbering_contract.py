"""O3/A12 ordinal numbers retain holes at gap-spanning windows."""
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'alaric')]
from alaric.reference import Reference


@pytest.mark.parametrize('resids,positions', [([101,102,103,104],[1,2,3]),([101,102,104,105],[1,3])])
def test_reference_positions_sequences_and_atom_coordinates(resids,positions):
    # Isolate addressing from PDB parsing/template installation. Each nucleotide
    # has distinctive coordinates so a wrong but valid residue cannot pass.
    ref = Reference.__new__(Reference)
    ref._resids = resids
    ref.sequence = 'ACGU'
    ref._coordinates = [np.full((2,3), i) for i in range(4)]
    assert ref.get_fragment_positions(2) == positions
    for pos in positions:
        assert ref.get_resids(pos,2) == resids[pos-1:pos+1]
        assert ref.get_sequence(pos,2) == 'ACGU'[pos-1:pos+1]
        np.testing.assert_array_equal(ref.get_coordinates(pos,2),np.concatenate(ref._coordinates[pos-1:pos+1]))
    for pos in set(range(1,5))-set(positions):
        with pytest.raises(ValueError):
            ref.get_coordinates(pos,2)


@pytest.mark.parametrize('value', [1.73, 3**.5])
def test_both_actions_allow_supported_ovrmsd(value):
    from alaric.anchor_constants import validate_ovrmsd
    assert validate_ovrmsd(value) == value


def test_both_actions_raise_at_174():
    from alaric.anchor_constants import validate_ovrmsd
    with pytest.raises(ValueError):
        validate_ovrmsd(1.74)


@pytest.mark.parametrize('action',['grow','anchor_refe'])
@pytest.mark.parametrize('limit',[1.73,1.74])
def test_actual_action_entry_validates_ovrmsd(tmp_path,monkeypatch,action,limit):
    from alaric import grow,anchor_refe
    module = grow if action == 'grow' else anchor_refe
    class AfterLimit(Exception):
        pass
    def stop(*args,**kwargs):
        raise AfterLimit
    if action == 'grow':
        source=tmp_path/'source'
        source.mkdir()
        argv=['--source-poses',str(source),'--source-sequence','GU','--target-sequence','UU',
              '--direction','forward','--crmsd','0.5']
        monkeypatch.setattr(module,'_resolve_growth_layout',stop)
    else:
        reference=tmp_path/'reference.pdb'
        reference.write_text('HEADER\n')
        argv=['--reference',str(reference),'--fragment','1','--first']
        monkeypatch.setattr(module,'load_reference_fragment',stop)
    args=module.build_parser().parse_args(argv+['--ov-rmsd',str(limit),'--output',str(tmp_path/'output')])
    with pytest.raises(AfterLimit if limit == 1.73 else ValueError):
        module._run(args)


@pytest.mark.parametrize('resids,positions', [([101,102,103,104],[1,2,3]),([101,102,104,105],[1,3])])
def test_rmsd_fragment_selection_is_positional(resids,positions):
    from alaric.rmsd import _infer_fragment
    from alaric.parse_pdb import atomic_dtype
    parsed=np.zeros(8,dtype=atomic_dtype)
    parsed['resid']=np.repeat(resids,2)
    parsed['resname']=np.repeat([b'A',b'C',b'G',b'U'],2)
    parsed['name']=np.tile([b"O3'",b'P'],4)
    parsed['x']=np.repeat(np.arange(4)*3.0,2)+np.tile([1.0,0.0],4)
    for position in positions:
        sequence,atoms=_infer_fragment(parsed,position)
        assert sequence == 'ACGU'[position-1:position+1]
        np.testing.assert_array_equal(
            atoms['x'],
            [3*(position-1)+1, 3*(position-1), 3*position+1, 3*position],
        )
        np.testing.assert_array_equal(atoms['resid'],[1,1,2,2])
    if positions == [1,3]:
        with pytest.raises(ValueError):
            _infer_fragment(parsed,2)


@pytest.mark.parametrize('gap',[False,True])
def test_renumbered_1b7f_reference_and_anchor_refe_agree(tmp_path,gap):
    from alaric.parse_pdb import parse_pdb
    from alaric.anchor_refe import load_reference_fragment
    text=(ROOT/'tests/1b7f/pdbs/rna.pdb').read_text()
    ids=[]
    for line in text.splitlines():
        if line.startswith(('ATOM  ','HETATM')):
            resid=int(line[22:26])
            if resid not in ids:
                ids.append(resid)
    mapping={resid:101+i+(1 if gap and i>=2 else 0) for i,resid in enumerate(ids)}
    lines=[]
    for line in text.splitlines(keepends=True):
        if line.startswith(('ATOM  ','HETATM')):
            line=line[:22]+f'{mapping[int(line[22:26])]:4d}'+line[26:]
        lines.append(line)
    target=tmp_path/'reference.pdb'
    target.write_text(''.join(lines))
    kwargs=dict(rna=True,ignore_unknown=False,ignore_missing=False,ignore_reordered=False)
    original=Reference(parse_pdb(text),**kwargs)
    renumbered=Reference(parse_pdb(target.read_text()),**kwargs)
    positions=[n for n in original.get_fragment_positions(2) if not(gap and n==2)]
    assert renumbered.get_fragment_positions(2)==positions
    for position in positions:
        sequence,coordinates=load_reference_fragment(target,position)
        assert sequence == original.get_sequence(position,2)
        np.testing.assert_array_equal(coordinates,original.get_coordinates(position,2).astype(np.float32))


def test_alaric_reference_omits_consecutive_label_chemical_chain_break():
    """D15 Alaric check; this does not test Preparic's stage-2 refusal."""
    from alaric.parse_pdb import parse_pdb
    parsed=parse_pdb((ROOT/'tests/1b7f/pdbs/rna.pdb').read_text())
    kwargs=dict(rna=True,ignore_unknown=False,ignore_missing=False,ignore_reordered=False)
    original=Reference(parsed,**kwargs)
    ids=list(dict.fromkeys(int(r) for r in parsed['resid']))
    assert ids[2]==ids[1]+1
    # Move the entire downstream block. Every internal bond is preserved;
    # only the O3'–P connection between residue positions 2 and 3 is broken.
    broken=parsed.copy()
    broken['x'][np.isin(broken['resid'],ids[2:])]+=100.
    ref=Reference(broken,**kwargs)
    assert ref.get_fragment_positions(2)==[p for p in original.get_fragment_positions(2) if p!=2]
    with pytest.raises(ValueError):
        ref.get_coordinates(2,2)
