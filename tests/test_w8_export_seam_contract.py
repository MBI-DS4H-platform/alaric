"""Fixture-based X1–X7 oracles for the internal bundle, not an export UI.

X5's numerical classification belongs to the browser: these tests consume its
explicit proceed-anyway condition and check the bundle's admission semantics.
"""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
FIXTURE=json.loads((ROOT/'tests/fixtures/w8/export-seam-cases.json').read_text())


def api():
    return importlib.import_module('alaric.export_bundle')


@pytest.fixture
def snapshot():
    candidates=deepcopy(FIXTURE['reference_candidates'])
    for row in candidates:
        row['sequence']='AC' if row['fragment']==1 else 'GU'
    # Anchor 1/second addresses RNA residue 2, shared by fragment 2/first;
    # anchor 3/first addresses residue 3, shared by fragment 2/second.
    candidates.extend([
        dict(fragment=2,sequence='CG',nucleotide='first',ovrmsd='0.80',ovrmsd_raw=0.79),
        dict(fragment=2,sequence='CG',nucleotide='second',ovrmsd='0.90',ovrmsd_raw=0.89),
    ])
    reference=b'HEADER prepared RNA\r\nATOM preserved bytes\r\nEND\r\n'
    return dict(
        prepared_reference=reference,prepared_sha256=hashlib.sha256(reference).hexdigest(),
        exclude_code='1b7f',prepared_exclude_code='1b7f',
        min_cell_fraction=1/3,prepared_min_cell_fraction=1/3,
        kept_fragments={'frag-1-AC':{'fragment':1,'sequence':'AC'},'frag-2-CG':{'fragment':2,'sequence':'CG'},'frag-3-GU':{'fragment':3,'sequence':'GU'}},
        kept_pairs=[dict(down=1,up=2,sequence='ACG',crmsd='0.42',ovrmsd='1.73'),
                    dict(down=2,up=3,sequence='CGU',crmsd='0.42',ovrmsd='1.73')],
        selected_anchors=[dict(fragment=1,sequence='AC',protein='range-2',resid=42,nucleotide='second',filter_name='rrm95'),
                          dict(fragment=3,sequence='GU',protein='range-2',resid=43,nucleotide='first',filter_name='rrm90-disp45')],
        reference_candidates=candidates,
        energy_matrix={'frag-1-AC':{'range-2':True,'range-3':False},'frag-3-GU':{'range-2':True},'frag-9-UU':{'range-2':True}},
        edited_energy_thresholds={'frag-1-AC':{'range-2':'-16.59','range-3':'999.00'},'frag-3-GU':{'range-2':'-2.00'},'frag-9-UU':{'range-2':'100.00'}},
        bestfit_energies={'frag-1-AC':{'range-2':-16.6},'frag-3-GU':{'range-2':-2.01}},
        class_table={'schema':1,'rows':[{'key':'anchor-a','class':95}]},
        handoff={'schema':1,'fragment_rows':['frag-1-AC','frag-2-CG','frag-3-GU']},
        validations={'results':{'fit':'fresh','energetics':'fresh','gaussian':'fresh','thresholds':'fresh'},
                     'decisions':{'thresholds':{'valid':True,'value':{'schema':1}}}},
        proceed_anyway_required=False,proceed_anyway=False)


def test_x1_x2_preserves_exact_reference_bytes_checksum_and_common_parameters(snapshot):
    bundle=api().build_export_bundle(snapshot)
    assert bundle['reference_pdb']==snapshot['prepared_reference']
    assert bundle['reference_sha256']==hashlib.sha256(snapshot['prepared_reference']).hexdigest()
    assert bundle['snapshot']['exclude_code']=='1b7f'
    assert bundle['snapshot']['min_cell_fraction']==1/3
    assert 'ALARIC_ANCHOR_MIN_CELL_FRACTION' not in repr(bundle)
    assert not bundle['hard_blocks']
    assert bundle['ready']


@pytest.mark.parametrize('field,value',[
    ('prepared_sha256','0'*64),('exclude_code','3sxl'),
    ('min_cell_fraction',.5),('prepared_min_cell_fraction',.5)])
def test_x1_x2_inconsistent_preparation_identity_refused(snapshot,field,value):
    snapshot[field]=value
    # Invalid data can be represented as an error or a non-overridable block.
    try:
        bundle=api().build_export_bundle(snapshot)
    except ValueError:
        return
    assert bundle['hard_blocks']
    assert not bundle['ready']


def test_x2_computes_missing_checksum_without_normalizing_bytes(snapshot):
    del snapshot['prepared_sha256']
    bundle=api().build_export_bundle(snapshot)
    assert bundle['reference_pdb'].endswith(b'\r\n')
    assert bundle['reference_sha256']==hashlib.sha256(snapshot['prepared_reference']).hexdigest()


def test_x3_both_modes_explicit_computed_thresholds_and_candidate_deduplication(snapshot):
    bundle=api().build_export_bundle(snapshot)
    anchors=bundle['anchor_actions']
    assert len(anchors)==2
    assert [row['precomputed_filter_name'] for row in anchors]==['gaussian-rrm']*2
    assert [row['precomputed_filter_threshold'] for row in anchors]==['rrm95','rrm90-disp45']
    refs=bundle['anchor_refe_actions']
    assert len(refs)==4
    assert {(r['fragment'],r['nucleotide'],float(r['ovrmsd'])) for r in refs}=={
        (1,'second',.70),(2,'first',.80),(2,'second',.90),(3,'first',1.73)}
    assert all(r['action']=='anchor-refe' and r['ovrmsd']!='auto' for r in refs)


def test_x4_only_true_kept_energy_cells_and_exact_default_parameters(snapshot):
    bundle=api().build_export_bundle(snapshot)
    assert bundle['energy_thresholds']==FIXTURE['energy_thresholds']
    assert bundle['score_filter_defaults']==FIXTURE['default_score_filter']
    actions=bundle['energy_filter_actions']
    assert len(actions)==2
    assert {r['name'] for r in actions}=={'energy-filter-frag1-range-2','energy-filter-frag3-range-2'}
    for row in actions:
        assert row['action']=='energy-filter'
        assert row['protein']=='range-2'
        assert row['epsilon']==15
        assert row['threshold']==row['cascade']==row['min_cascade']=='auto'
        assert row['fragment']==row['sequence']==row['exclude']=='auto'
    assert bundle['grow_thresholds']==snapshot['kept_pairs']
    for key in ('kept_fragments','handoff','class_table','validations'):
        assert bundle[key]==snapshot[key]


@pytest.mark.parametrize('case',[c for c in FIXTURE['readiness_cases'] if c['clause']=='X5'],ids=lambda c:c['change'])
def test_x5_browser_soft_failures_need_explicit_acknowledgment(snapshot,case):
    snapshot['proceed_anyway_required']=case['proceed_anyway']
    # Numerical red/amber/energy tests are browser-owned; this is the resulting
    # condition carried into one coherent snapshot.
    first=api().build_export_bundle(snapshot)
    assert not first['hard_blocks']
    assert first['proceed_anyway_required']
    assert not first['ready']
    snapshot['proceed_anyway']=True
    second=api().build_export_bundle(snapshot)
    assert not second['hard_blocks']
    assert second['ready']


@pytest.mark.parametrize('case',[c for c in FIXTURE['readiness_cases'] if c['clause']=='X6'],ids=lambda c:c['change'])
def test_x6_hard_blocks_cannot_be_overridden(snapshot,case):
    change=case['change']
    if 'anchor-refe' in change:
        snapshot['reference_candidates'][0]['ovrmsd']='1.74'
    elif 'kept grow' in change:
        snapshot['kept_pairs'][0]['ovrmsd']='1.74'
    elif 'unkept pair' in change:
        # All-pair source data can contain unsupported values, but only
        # kept_pairs are actually handed off for grow actions.
        snapshot['all_pairs']=[dict(down=9,up=10,crmsd='0.42',ovrmsd='1.74')]
    elif 'required result' in change:
        snapshot['validations']['results']['fit']=change.split()[-1]
    elif 'decision null' in change:
        snapshot['validations']['decisions']['thresholds']=None
    elif 'decision invalid' in change:
        snapshot['validations']['decisions']['thresholds']['valid']=False
    else:
        pytest.fail('Unhandled contract fixture: '+change)
    snapshot['proceed_anyway_required']=True
    snapshot['proceed_anyway']=True
    bundle=api().build_export_bundle(snapshot)
    assert bool(bundle['hard_blocks'])==case['hard_block']
    assert bundle['ready']==(not case['hard_block'])


@pytest.mark.parametrize('field', ['prepared_reference','exclude_code','kept_fragments','kept_pairs',
    'selected_anchors','reference_candidates','edited_energy_thresholds','class_table','handoff','validations','proceed_anyway'])
def test_x7_any_decision_edit_stales_existing_bundle(snapshot,field):
    bundle=api().build_export_bundle(snapshot)
    assert not api().bundle_is_stale(bundle,deepcopy(snapshot))
    changed=deepcopy(snapshot)
    if field=='prepared_reference':
        changed[field]+=b'REMARK edit\n'
    elif isinstance(changed[field],dict):
        changed[field]['edited']=True
    elif isinstance(changed[field],list):
        changed[field].append({'edited':True})
    elif isinstance(changed[field],bool):
        changed[field]=not changed[field]
    else:
        changed[field]='edited'
    assert api().bundle_is_stale(bundle,changed)


def test_x7_bundle_is_immutable_snapshot_of_nested_inputs(snapshot):
    original=deepcopy(snapshot)
    bundle=api().build_export_bundle(snapshot)
    frozen=deepcopy(bundle)
    snapshot['reference_candidates'][0]['ovrmsd']='1.74'
    snapshot['edited_energy_thresholds']['frag-1-AC']['range-2']='-99.00'
    snapshot['handoff']['fragment_rows'].append('frag-9-UU')
    assert bundle==frozen
    assert bundle['snapshot']==original
    assert api().bundle_is_stale(bundle,snapshot)


@pytest.mark.parametrize("removed", [(1,), (3,), (1, 3)], ids=["clip-5prime", "clip-3prime", "keep-middle-only"])
def test_w8a_export_preserves_selected_stackings_and_both_modes_from_kept_candidates(snapshot, removed):
    for fragment in removed:
        key = next(key for key, row in snapshot["kept_fragments"].items() if row["fragment"] == fragment)
        del snapshot["kept_fragments"][key]
    kept = {row["fragment"] for row in snapshot["kept_fragments"].values()}
    snapshot["kept_pairs"] = [row for row in snapshot["kept_pairs"] if row["down"] in kept and row["up"] in kept]
    snapshot["handoff"]["fragment_rows"] = list(snapshot["kept_fragments"])
    for candidate in snapshot["reference_candidates"]:
        if candidate["fragment"] in removed:
            candidate["ovrmsd"] = "1.74"  # clipped failure cannot block export
    classes_before = deepcopy(snapshot["class_table"])
    bundle = api().build_export_bundle(snapshot)
    assert not bundle["hard_blocks"]
    assert bundle["ready"]
    assert len(bundle["anchor_actions"]) == 2
    assert {row["fragment"] for row in bundle["anchor_actions"]} <= kept
    assert [row["precomputed_filter_threshold"] for row in bundle["anchor_actions"]] == ["rrm95", "rrm90-disp45"]
    expected = {(fragment, nucleotide) for fragment, nucleotide in [(1, "second"), (2, "first"), (2, "second"), (3, "first")] if fragment in kept}
    assert {(row["fragment"], row["nucleotide"]) for row in bundle["anchor_refe_actions"]} == expected
    assert all(row["ovrmsd"] != "auto" for row in bundle["anchor_refe_actions"])
    assert snapshot["class_table"] == classes_before


@pytest.mark.parametrize('rounded,raw,sentinel', [('1.73', 1.72, False), ('1.74', 1.731, True), ('2.25', 2.24, True)])
def test_w8b_computed_threshold_is_copied_or_disabled_with_associated_comment(snapshot, rounded, raw, sentinel):
    row = snapshot['reference_candidates'][0]
    row.update(ovrmsd=rounded, ovrmsd_raw=raw)
    bundle = api().build_export_bundle(snapshot)
    action = next(a for a in bundle['anchor_refe_actions'] if (a['fragment'], a['nucleotide']) == (row['fragment'], row['nucleotide']))
    assert not bundle['hard_blocks']
    assert bundle['ready']
    assert bundle['proceed_anyway_required'] is False
    assert float(action['ovrmsd']) == (-1 if sentinel else float(rounded))
    if sentinel:
        assert isinstance(action['ovrmsd'], (int, float)) and not isinstance(action['ovrmsd'], bool)
    assert 'comment' not in action
    comments = [c for c in bundle['anchor_refe_comments'] if (c['fragment'], c['nucleotide']) == (row['fragment'], row['nucleotide'])]
    assert len(comments) == int(sentinel)
    if sentinel:
        comment = comments[0]['comment']
        assert str(raw) in comment and rounded in comment
        assert 'L_ov' in comment or 'L_OV' in comment
        assert 'disabled positive control' in comment.lower()
        assert 'without misses' in comment.lower()
        if raw < 3 ** .5:
            assert 'best-fit pose itself lies beyond' not in comment.lower()
    frozen = deepcopy(bundle['anchor_refe_comments'])
    row['ovrmsd_raw'] = 99
    assert bundle['anchor_refe_comments'] == frozen


@pytest.mark.parametrize('field,value', [
    ('ovrmsd', None), ('ovrmsd', 'auto'), ('ovrmsd', True), ('ovrmsd', 'NaN'),
    ('ovrmsd_raw', None), ('ovrmsd_raw', 'auto'), ('ovrmsd_raw', True), ('ovrmsd_raw', float('inf')),
])
def test_w8b_missing_or_malformed_computed_candidate_evidence_is_refused(snapshot, field, value):
    snapshot['reference_candidates'][0][field] = value
    try:
        bundle = api().build_export_bundle(snapshot)
    except ValueError:
        return
    assert bundle['hard_blocks'] and not bundle['ready']


@pytest.mark.parametrize('field', ['ovrmsd', 'ovrmsd_raw'])
def test_w8b_missing_computed_candidate_value_is_refused(snapshot, field):
    del snapshot['reference_candidates'][0][field]
    try:
        bundle = api().build_export_bundle(snapshot)
    except ValueError:
        return
    assert bundle['hard_blocks'] and not bundle['ready']


@pytest.mark.parametrize('field,second', [('ovrmsd', '2.26'), ('ovrmsd_raw', 2.23)])
def test_w8b_dedup_checks_original_values_before_sentinel_substitution(snapshot, field, second):
    target = snapshot['reference_candidates'][0]
    target.update(ovrmsd='2.25', ovrmsd_raw=2.24)
    duplicate = {**target, field: second}
    snapshot['reference_candidates'].append(duplicate)
    try:
        bundle = api().build_export_bundle(snapshot)
    except ValueError:
        return
    assert bundle['hard_blocks'] and not bundle['ready']


@pytest.mark.parametrize('status', [None, 'stale', 'running', 'failed'])
def test_w8b_s9_freshness_is_required_at_export_even_after_continue(snapshot, status):
    snapshot['validations']['results']['thresholds'] = status
    snapshot['proceed_anyway'] = True
    try:
        bundle = api().build_export_bundle(snapshot)
    except ValueError:
        return
    assert bundle['hard_blocks'] and not bundle['ready']


def test_w8b_clipping_disabled_candidate_removes_its_action_and_comment(snapshot):
    snapshot['reference_candidates'][0].update(ovrmsd='2.25', ovrmsd_raw=2.24)
    del snapshot['kept_fragments']['frag-1-AC']
    snapshot['kept_pairs'] = snapshot['kept_pairs'][1:]
    snapshot['handoff']['fragment_rows'] = list(snapshot['kept_fragments'])
    bundle = api().build_export_bundle(snapshot)
    assert bundle['ready'] and not bundle['hard_blocks']
    assert not any(row['fragment'] == 1 for row in bundle['anchor_refe_actions'])
    assert not any(row['fragment'] == 1 for row in bundle['anchor_refe_comments'])
