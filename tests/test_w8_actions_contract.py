"""W8 contract oracles for A12; no generated numerical expectations."""
from pathlib import Path
import importlib
import json
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def runtime():
    return importlib.import_module('alaric.energy_filter')


@pytest.mark.parametrize('energies,expected', [([-1.01, -1., -.99], [0]), ([], [])])
def test_gridless_strict_threshold(energies, expected):
    energies = np.asarray(energies)
    kept, stats = runtime().cascade_filter(np.arange(len(energies)), threshold=-1., cascade=[], min_cascade=0,
        grid_score=lambda *_: pytest.fail('gridless path used a grid'), gridless_score=lambda ids: energies[ids])
    assert list(kept) == expected
    assert stats['gridless_count'] == len(energies)


def test_injected_error_edges_and_input_order():
    # At lower edge still undecided; at upper edge rejected. Accepted poses
    # straddle undecided ones, so concatenating decision batches is wrong.
    g = np.array([-2.01, -2., -.5, 0., -3.])
    exact = np.array([-2., -1.5, -.1, .2, -2.5])
    seen = []
    def direct(ids):
        seen.extend(ids)
        return exact[ids]
    kept, stats = runtime().cascade_filter(np.arange(5), threshold=-1.,
        cascade=[dict(plateau_distance=5, neighbour_distance=7, max_error=1.)], min_cascade=0,
        grid_score=lambda ids, step: g[ids], gridless_score=direct)
    assert list(kept) == [0, 1, 4]
    assert seen == [1, 2]
    assert stats['grid_counts'] == [5]
    assert stats['gridless_count'] == 2


@pytest.mark.parametrize('minimum,expected_counts,expected_direct', [(6, [], 5), (3, [5], 2), (2, [5, 2], 2)])
def test_min_cascade_checked_before_every_step(minimum, expected_counts, expected_direct):
    g = np.array([-3., 1., -1.1, -.9, -4.])
    steps = [dict(plateau_distance=p, neighbour_distance=n, max_error=e)
             for p,n,e in [(5,7,1.), (10,12,.05)]]
    def grid(ids, step):
        return g[ids] if step['plateau_distance'] == 5 else np.full(len(ids), -1.)
    kept, stats = runtime().cascade_filter(np.arange(5), threshold=-1., cascade=steps,
        min_cascade=minimum, grid_score=grid, gridless_score=lambda ids: g[ids])
    assert list(kept) == [0, 2, 4]
    assert stats['grid_counts'] == expected_counts
    assert stats['gridless_count'] == expected_direct


def test_float64_ulp_routes_grid_scores_to_exact():
    module = runtime()
    unsafe = module.grid_precision_unsafe(
        np.array([float(2**53), -float(2**53), 0.]), threshold=0., max_error=1.
    )
    np.testing.assert_array_equal(unsafe, [True, True, False])
    # A coarse representable threshold makes every grid decision unsafe too.
    np.testing.assert_array_equal(
        module.grid_precision_unsafe(np.array([0., 1.]), float(2**53), 1.),
        [True, True],
    )

    ids = np.array([41, 7, 99])
    exact_values = {41: -2., 7: 2., 99: -.25}
    exact_seen = []

    def direct(batch):
        exact_seen.extend(batch.tolist())
        return np.array([exact_values[int(pose_id)] for pose_id in batch])

    kept, stats = module.cascade_filter(
        ids, threshold=0.,
        cascade=[dict(plateau_distance=5, neighbour_distance=7, max_error=1.)],
        min_cascade=0,
        grid_score=lambda batch, step: np.array([float(2**53), -float(2**53), 0.]),
        gridless_score=direct,
    )
    assert kept.tolist() == [41, 99]
    assert exact_seen == [41, 7, 99]
    assert stats['precision_fallback_counts'] == [2]
    assert stats['precision_fallback_count'] == 2
    assert stats['gridless_count'] == 3


def test_min_cascade_counts_precision_fallback_as_unresolved():
    ids = np.array([10, 20, 30, 40])
    grid_ids = []
    exact_ids = []
    step_values = {
        10: float(2**53), 20: -1., 30: -.9, 40: 0.,
    }

    def grid(batch, step):
        grid_ids.append(batch.tolist())
        if step['plateau_distance'] == 5:
            return np.array([step_values[int(pose_id)] for pose_id in batch])
        second = {20: -1., 30: -.9, 40: .1}
        return np.array([second[int(pose_id)] for pose_id in batch])

    def direct(batch):
        exact_ids.extend(batch.tolist())
        return np.array([-2.] * len(batch))

    kept, stats = runtime().cascade_filter(
        ids, threshold=0., min_cascade=4,
        cascade=[
            dict(plateau_distance=5, neighbour_distance=7, max_error=1.),
            dict(plateau_distance=10, neighbour_distance=12, max_error=.05),
        ],
        grid_score=grid, gridless_score=direct,
    )
    assert grid_ids == [[10, 20, 30, 40], [20, 30, 40]]
    assert exact_ids == [10]
    assert kept.tolist() == [10, 20, 30]
    assert stats['grid_counts'] == [4, 3]
    assert stats['precision_fallback_counts'] == [1, 0]
    assert stats['precision_fallback_count'] == 1
    assert stats['gridless_count'] == 1


def test_ranges_are_successive_and_never_summed():
    # Summing would incorrectly retain pose 0 (-100 + 2 < 0).
    ids = np.arange(3)
    for energies in [np.array([-100., -2., -3.]), np.array([2., -2., 0.])]:
        ids, _ = runtime().cascade_filter(ids, threshold=0., cascade=[], min_cascade=0,
            grid_score=lambda *_: pytest.fail('unexpected grid'), gridless_score=lambda i: energies[i])
    assert list(ids) == [1]


def test_threshold_file_exact_mapping_and_missing_cell(tmp_path):
    path = tmp_path / 'energy-thresholds.json'
    path.write_text(json.dumps({'schema':1,'frag-3-AC':{'range-2':'-16.59'}}))
    assert runtime().load_threshold(path, 3, 'AC', 'range-2') == -16.59
    with pytest.raises((ValueError, KeyError)):
        runtime().load_threshold(path, 3, 'AC', 'range-1')


@pytest.mark.parametrize('payload', [
    {'schema':2,'frag-3-AC':{'range-2':'-16.59'}},
    {'schema':1,'source':'bestfit','frag-3-AC':{'range-2':'-16.59'}},
    {'schema':1,'frag-3-AC':{'range-2':-16.59}},
    {'schema':1,'frag-3-AC':{'range-2':'-16.590'}},
    {'schema':1,'frag-3-AC':{'range-2':['-16.59']}},
    {'schema':1,'frag-3-AC':{'dom':'-16.59'}},
])
def test_threshold_file_rejects_bad_schema_metadata_or_shape(tmp_path, payload):
    path = tmp_path / 'energy-thresholds.json'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        runtime().load_threshold(path, 3, 'AC', 'range-2')


def test_auto_resolution_and_explicit_gridless_override(tmp_path):
    path = tmp_path / 'score-filter.yaml'
    path.write_text('min_cascade: 1000\ncascade:\n  - {plateau_distance: 5, neighbour_distance: 7, max_error: 1.0}\n  - {plateau_distance: 10, neighbour_distance: 12, max_error: 0.05}\n')
    steps, minimum = runtime().resolve_config('auto', 'auto', path)
    assert minimum == 1000
    assert steps == [dict(plateau_distance=5,neighbour_distance=7,max_error=1.),dict(plateau_distance=10,neighbour_distance=12,max_error=.05)]
    assert runtime().resolve_config([], 0, tmp_path / 'absent') == ([], 0)
    with pytest.raises((ValueError, FileNotFoundError)):
        runtime().resolve_config('auto', 'auto', tmp_path / 'absent')


@pytest.mark.parametrize('text', ['[]', 'cascade: []', 'min_cascade: nope\ncascade: []', 'min_cascade: 1\ncascade: [{plateau_distance: 5}]'])
def test_auto_malformed_defaults_are_errors(tmp_path, text):
    path = tmp_path / 'score-filter.yaml'
    path.write_text(text)
    with pytest.raises(ValueError):
        runtime().resolve_config('auto', 'auto', path)


@pytest.mark.parametrize('action', ['score','filter','score_add'])
def test_removed_actions_rejected(action,tmp_path):
    from alaric.middle.schema import normalize_action
    from alaric.middle.errors import SchemaError
    with pytest.raises(SchemaError):
        normalize_action('frag1-old', tmp_path, {'action':action})


@pytest.mark.parametrize('other', [
    {'frag-4-UU': {'range-2':'oops'}},
    {'bad-fragment': {'range-2':'-2.00'}},
    {'frag-4-UU': []},
])
def test_bad_unrequested_threshold_cells_are_not_partially_consumed(tmp_path,other):
    path=tmp_path/'energy-thresholds.json'
    path.write_text(json.dumps({'schema':1,'frag-3-AC':{'range-2':'-16.59'},**other}))
    with pytest.raises(ValueError):
        runtime().load_threshold(path,3,'AC','range-2')


@pytest.mark.parametrize('steps,minimum', [
    ([dict(plateau_distance=5,neighbour_distance=7,max_error=-1)],0),
    ([dict(plateau_distance=5,neighbour_distance=7,max_error=float('nan'))],0),
    ([],1.5),
    ([],True),
])
def test_invalid_cascade_parameters_are_rejected(steps,minimum,tmp_path):
    with pytest.raises(ValueError):
        runtime().resolve_config(steps,minimum,tmp_path/'unused')


def test_pose_identity_is_preserved_when_callbacks_rechunk_work():
    # Whole-action invocation stays unchanged while scorer batching changes.
    ids=np.array([41,7,99,3,18,52,6,84])
    values={i:e for i,e in zip(ids,[-3.,-1.1,1.,-4.,-.9,-2.5,-1.,.5])}
    def score(batch):
        return np.array([values[i] for i in batch])
    results=[]
    for chunk in [1,3,100]:
        def batched(batch,step=None):
            return np.concatenate([score(batch[k:k+chunk]) for k in range(0,len(batch),chunk)]) if len(batch) else np.array([])
        kept,stats=runtime().cascade_filter(ids,threshold=-1.,
            cascade=[dict(plateau_distance=5,neighbour_distance=7,max_error=1.)],min_cascade=5,
            grid_score=batched,gridless_score=batched)
        assert list(kept)==[41,7,3,52]
        assert stats['grid_counts']==[8]
        assert stats['gridless_count']==3
        results.append(np.asarray(kept).tobytes())
    assert len(set(results))==1


def test_energy_filter_action_accepts_full_export_parameters(tmp_path):
    from alaric.middle.schema import normalize_action
    spec=dict(action='energy-filter',input='frag3-anchor',protein='range-2',
        threshold='auto',epsilon=15,cascade='auto',min_cascade='auto',
        fragment='auto',sequence='auto',exclude='auto')
    normalized=normalize_action('frag3-energy-range-2',tmp_path,spec)
    assert normalized is not None


@pytest.mark.parametrize('protein',[['range-2','range-3'],'range-2 range-3',None])
def test_energy_filter_cannot_sum_or_group_ranges(tmp_path,protein):
    from alaric.middle.schema import normalize_action
    from alaric.middle.errors import SchemaError
    spec=dict(action='energy-filter',input='frag3-anchor',protein=protein,
        threshold=-1.,epsilon=15,cascade=[],min_cascade=0,
        fragment=3,sequence='AC',exclude='1b7f')
    with pytest.raises(SchemaError):
        normalize_action('frag3-energy-range-2',tmp_path,spec)
