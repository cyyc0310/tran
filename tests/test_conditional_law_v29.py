"""Pure synthetic closure, conditional moments, geographic OOF and path tests."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from scripts import conditional_law_v29 as m
from scripts import coarse_supervision_v28 as old
from scripts import group_regression_v27 as prior
from transcif.research.fuel_transport_v7 import recompose_paths


def rows():
    rng = np.random.default_rng(29011); result = []
    for region in m.SOURCES:
        for start in pd.date_range('2022-01-01', '2024-12-01', freq='MS', tz='UTC'):
            end = start+pd.offsets.MonthBegin(1); mix = rng.dirichlet(np.ones(8))
            if ((region == 'CISO' and start == pd.Timestamp('2022-01-01', tz='UTC'))
                    or (region == 'NYIS' and start.year == 2022 and start.month in [2, 3])):
                continue
            result.append(dict(region=region, period_start=start.isoformat(), period_end=end.isoformat(),
                available_at=(end+pd.Timedelta(days=30)).isoformat(), **dict(zip(old.SHARES, map(str, mix)))))
    return result


def bank_stream(*, fallback=False, zero_query=False):
    rng = np.random.default_rng(29012); mixes, qs, regions, days = [], [], [], []
    for region in m.SOURCES[1:]:
        for month in [1, 2]:
            mix = rng.dirichlet(np.ones(8))
            if fallback and region == 'CISO' and month == 1:
                mix[m.POS] = 0; mix[m.ZERO] = .25
            q = rng.uniform(.1, 2, (24, 8)); q /= q.mean(0)
            q[:, mix == 0] = 1
            mixes.append(mix); qs.append(q); regions.append(region)
            days.append(pd.Timestamp(year=2024, month=month, day=25 if month == 1 else 1, tz='UTC').value)
    mix = np.stack(mixes); q = np.ascontiguousarray(np.stack(qs).transpose(0, 2, 1)).transpose(0, 2, 1)
    bank = dict(q=q, donor_mix=mix, support=mix > 0, raw_cif=recompose_paths(mix, q),
                regions=np.asarray(regions), days=np.asarray(days, np.int64))
    dates = pd.date_range('2025-01-01', periods=2, tz='UTC'); draws = rng.dirichlet(np.ones(5), (2, 128))
    if zero_query:draws[:] = [0, .25, .25, .25, .25]
    ids = rng.integers(0, 12, (2, 128), dtype=np.int64)
    if fallback:ids[:] = 0
    borrowed = np.full_like(ids, -1); raw, fine = [], []
    for t in range(2):
        need = (bank['donor_mix'][ids[t]][:, m.POS].sum(1) == 0) & (draws[t, :, 0] > 0)
        borrowed[t, need] = 1
        used_q, _, _, ratio = prior._effective(bank, ids[t], borrowed[t], draws[t, :, 0])
        fd = np.zeros((128, 8)); fd[:, m.ZERO] = draws[t, :, 1:]; fd[:, m.POS] = draws[t, :, :1]*ratio
        fine.append(fd); raw.append(recompose_paths(fd, used_q))
    baseline = dict(raw=np.stack(raw), fine_draws=np.stack(fine), donor_ids=ids, fallback_ids=borrowed)
    return bank, baseline, draws, dates


def make_experiment(fallback=False, zero_query=False):
    bank, baseline, draws, dates = bank_stream(fallback=fallback, zero_query=zero_query); history = rows()
    context = m.fit_leaf_controls(bank, history, outer='BPAT')
    parent = prior.fit_regressions(bank, outer='BPAT').predict(baseline, draws, dates)
    background27 = {k: np.stack([parent['arms'][a][k] for a in prior.ARMS]) for k in old.FIELDS}
    background27.update(arms=np.asarray(prior.ARMS), origins=dates.asi8,
                        donor_ids=baseline['donor_ids'], fallback_ids=baseline['fallback_ids'])
    parent28 = old.fit_heads(bank, history, outer='BPAT').predict(baseline, draws, dates, background27)
    background28 = {k: np.stack([parent28['arms'][a][k] for a in old.ARMS]) for k in old.FIELDS}
    background28.update(arms=np.asarray(old.ARMS), origins=dates.asi8, L24_ET_query_raw=parent28['L24_ET_query_raw'],
                        donor_ids=baseline['donor_ids'], fallback_ids=baseline['fallback_ids'])
    return context, baseline, draws, dates, background28


@pytest.fixture(scope='module')
def experiment():
    return make_experiment()


def test_leaf_weights_normalized_weighted_and_et_mean(experiment):
    context, _, draws, _, _ = experiment
    x = draws.reshape(-1, 5); conditional = context.final.conditional(x)
    weight = conditional['weights']; np.testing.assert_allclose(weight.sum(1), 1, atol=1e-14)
    assert np.all(weight >= 0)
    np.testing.assert_allclose(weight@context.final.y, context.final.model.predict(x), atol=1e-12, rtol=0)
    # Independently walk one query's128leaves using training sample weights.
    expected = np.zeros(len(context.final.y))
    for tree in range(128):
        held = context.final.leaf_ids[:, tree] == conditional['leaf_ids'][0, tree]
        expected[held] += context.final.weights[held]/context.final.weights[held].sum()/128
    np.testing.assert_allclose(expected, weight[0], atol=1e-16, rtol=0)


def test_covariance_is_total_across_trees_not_average_within_leaf():
    class Model:
        def apply(self, x):return np.zeros((len(x), 128), np.int64)
        def predict(self, x):return np.tile([.5, .5, 0, 0], (len(x), 1))
    y = np.array([[1., 0, 0, 0], [0., 1, 0, 0]])
    pools = [{0: (np.array([i % 2]), np.ones(1))} for i in range(128)]
    law = m.LeafLaw(Model(), np.ones((2, 5))*.2, y, np.ones(2), np.zeros((2, 128), np.int64), pools, {})
    result = law.conditional(np.ones((1, 5))*.2)
    np.testing.assert_array_equal(result['weights'], [[.5, .5]])
    delta = (y-[.5, .5, 0, 0])@m.HELMERT
    np.testing.assert_allclose(result['covariance3'][0], delta.T@delta/2, atol=1e-16)
    assert np.trace(result['covariance3'][0]) == pytest.approx(.5)
    assert result['scale'][0]**2 == pytest.approx(.5+1e-12)


def test_constant_leaf_response_has_zero_scatter_and_fixed_floor():
    class Model:
        def apply(self, x):return np.zeros((len(x), 128), np.int64)
        def predict(self, x):return np.tile([.25]*4, (len(x), 1))
    y = np.full((2, 4), .25)
    pools = [{0: (np.arange(2), np.array([.25, .75]))} for _ in range(128)]
    law = m.LeafLaw(Model(), np.full((2, 5), .2), y, np.array([1., 3.]), np.zeros((2, 128), np.int64), pools, {})
    result = law.conditional(np.full((1, 5), .2))
    np.testing.assert_array_equal(result['covariance3'], 0)
    np.testing.assert_array_equal(result['scale'], 1e-6)


def test_joint_leaf_sampling_and_positive_cdf_endpoint():
    w = np.full((1, 11), .1); w[0, -1] = 0
    assert m.sample_rows(w, np.array([np.nextafter(1., 0)]))[0] == 9
    assert m.sample_rows(np.array([[0., .3, .7]]), np.array([0.]))[0] == 1
    with pytest.raises(ValueError):m.sample_rows(w, np.ones(1))


def test_gaussian_degenerate_and_full_tangent_covariance():
    mean = np.tile([.25]*4, (2, 1)); cov = np.zeros((2, 3, 3)); z = np.array([[1., 2, 3], [-1., 3, 4]])
    np.testing.assert_array_equal(m.gaussian_ratios(mean, cov, z), mean)
    np.testing.assert_allclose(m.HELMERT.T@m.HELMERT, np.eye(3), atol=4e-16)
    np.testing.assert_allclose(m.HELMERT.sum(0), 0, atol=2e-16)
    cov[0] = np.array([[.1, .03, 0], [.03, .08, .02], [0, .02, .04]])
    before = m.gaussian_ratios(mean, cov, z)
    values, vectors = np.linalg.eigh(cov[0])
    root = vectors@np.diag(np.sqrt(values))@vectors.T
    np.testing.assert_allclose(before[0], mean[0]+root@z[0]@m.HELMERT.T, atol=1e-16)
    np.testing.assert_allclose(before.sum(1), 1, atol=3e-16)
    np.testing.assert_allclose(prior.project(before).sum(1), 1, atol=2e-16)
    cov[0, 0, 0] = -1
    with pytest.raises(ValueError, match='positive semidefinite'):m.gaussian_ratios(mean, cov, z)


def test_fold_excludes_all_teacher_history_and_weighted_pools(experiment):
    context = experiment[0]
    assert context.audit['fitted_head_models'] == 7
    for teacher, fold in context.folds.items():
        meta = fold.metadata
        assert teacher not in meta['training_regions'] and 'BPAT' not in meta['training_regions']
        assert all(not key.startswith(teacher+'/') for key in meta['training_keys'])
        assert all(at <= m.CUTOFF.value for at in meta['available_at'])
        np.testing.assert_allclose(fold.weights.mean(), 1, atol=1e-15)
        by_region = {r: sum(w for key, w in zip(meta['training_keys'], fold.weights) if key.startswith(r+'/'))
                     for r in meta['training_regions']}
        np.testing.assert_allclose(list(by_region.values()), len(fold.weights)/len(by_region), atol=1e-13)
        held = context.table['regions'] == teacher
        result = fold.conditional(context.table['X'][held])
        np.testing.assert_array_equal(context.oof_mean[held], result['mean'])
        np.testing.assert_array_equal(context.oof_scale[held], result['scale'])
        positive = context.history['positive_mask']; labels = context.history['regions'][positive]
        assert not context.oof_weights[held][:, labels == teacher].any()
    np.testing.assert_allclose(context.oof_weights.sum(1), 1, atol=1e-14)
    np.testing.assert_array_equal(context.oof_residual, context.table['ratios']-context.oof_mean)
    json.dumps(context.audit, allow_nan=False)


def test_teacher_response_change_does_not_change_its_fold(experiment):
    context = experiment[0]; changed = rows()
    for row in changed:
        if row['region'] == 'CISO':
            values = [row[old.SHARES[i]] for i in m.POS[::-1]]
            for i, value in zip(m.POS, values):row[old.SHARES[i]] = value
    other = m.fit_leaf_controls(context.bank, changed, outer='BPAT')
    held = context.table['regions'] == 'CISO'
    np.testing.assert_array_equal(context.folds['CISO'].leaf_ids, other.folds['CISO'].leaf_ids)
    np.testing.assert_array_equal(context.oof_mean[held], other.oof_mean[held])
    np.testing.assert_array_equal(context.oof_scale[held], other.oof_scale[held])
    np.testing.assert_array_equal(context.oof_residual[held], other.oof_residual[held])


def test_full_grid_exact_backgrounds_law_paths_and_replay(experiment):
    context, baseline, draws, dates, background = experiment
    out = context.predict(baseline, draws, dates, background)
    assert tuple(out['arms']) == m.ARMS and len(m.ARMS) == 10
    for arm in m.ARMS[:3]:
        for field in m.FIELDS:
            np.testing.assert_array_equal(out['arms'][arm][field], background[field][old.ARMS.index(arm)])
    np.testing.assert_allclose(out['query_mean'], background['L24_ET_query_raw'], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(out['donor_ids'], baseline['donor_ids'])
    expected_scale = out['query_mean']+out['scale_ratio'][..., None]*context.oof_residual[out['effective_month_ids']]
    np.testing.assert_array_equal(out['arms']['Scale-recompose']['unprojected_ratios'], expected_scale)
    np.testing.assert_array_equal(out['arms']['Leaf-recompose']['ratios'], context.final.y[out['leaf_sample_ids']])
    for family in ['Leaf', 'Gaussian', 'Scale']:
        a, b = out['arms'][family+'-anchor'], out['arms'][family+'-recompose']
        np.testing.assert_array_equal(a['fine_draws'], b['fine_draws'])
        expected = context.bank['raw_cif'][baseline['donor_ids']]+((a['fine_draws']-context.bank['donor_mix'][baseline['donor_ids']])@m.EF)[..., None]
        np.testing.assert_array_equal(a['raw'], expected)
    for arm in out['arms'].values():
        assert arm['raw'].shape == (2, 128, 24) and np.isfinite(arm['raw']).all()
        np.testing.assert_allclose(arm['fine_draws'].sum(-1), 1, atol=1e-15)
        np.testing.assert_array_equal(arm['fine_draws'][..., m.ZERO], draws[..., 1:])
    assert out['query_weights'].shape == (2, 128, len(context.final.y))
    assert out['query_leaf_ids'].shape == (2, 128, 128) and out['query_covariance3'].shape == (2, 128, 3, 3)
    assert np.all(out['query_floor_fraction'] > 0) and np.all(out['query_floor_fraction'] <= 1)
    assert np.all(out['source_floor_fraction'] > 0) and np.all(out['source_floor_fraction'] <= 1)
    json.dumps(out['audit'], allow_nan=False)
    again = context.predict(baseline, draws, dates, background)
    for key, value in out.items():
        if isinstance(value, np.ndarray):np.testing.assert_array_equal(value, again[key])


def test_shuffle_uses_whole_effective_q_and_support_and_keeps_fine(experiment):
    context, baseline, draws, dates, background = experiment
    out = context.predict(baseline, draws, dates, background)
    a, b = out['arms']['Scale-recompose'], out['arms']['Scale-shuffle_recompose']
    np.testing.assert_array_equal(a['fine_draws'], b['fine_draws'])
    for t in range(2):
        perm = out['permutation'][t]; np.testing.assert_array_equal(np.sort(perm), np.arange(128))
        q, support, selected, _ = prior._effective(context.bank, baseline['donor_ids'][t], baseline['fallback_ids'][t], draws[t, :, 0])
        np.testing.assert_array_equal(b['raw'][t], recompose_paths(b['fine_draws'][t], q[perm]))
        np.testing.assert_array_equal(out['shuffled_original_ids'][t], baseline['donor_ids'][t, perm])
        np.testing.assert_array_equal(out['shuffled_effective_pos_ids'][t], selected[perm])
        np.testing.assert_array_equal(b['unsupported_positive_energy_fraction'][t], prior._unsupported(b['fine_draws'][t], q[perm], support[perm]))
        np.testing.assert_array_equal(q[perm][np.argsort(perm)], q)


def test_fallback_j_drives_source_scale_residual_and_pos_q():
    context, baseline, draws, dates, background = make_experiment(fallback=True)
    out = context.predict(baseline, draws, dates, background)
    assert np.all(out['effective_pos_ids'] == 1)
    np.testing.assert_array_equal(out['effective_month_ids'], context.table['bank_month_ids'][np.ones((2, 128), np.int64)])
    np.testing.assert_array_equal(out['source_scale'], context.oof_scale[out['effective_month_ids']])
    assert np.all(context.oof_residual[context.table['zero_pos_mask']] == 0)
    for t in range(2):
        q, _, _, _ = prior._effective(context.bank, baseline['donor_ids'][t], baseline['fallback_ids'][t], draws[t, :, 0])
        np.testing.assert_array_equal(out['arms']['Scale-recompose']['raw'][t], recompose_paths(out['arms']['Scale-recompose']['fine_draws'][t], q))


def test_zero_query_pos_does_not_need_fallback_or_create_positive_energy():
    context, baseline, draws, dates, background = make_experiment(fallback=True, zero_query=True)
    assert np.all(baseline['fallback_ids'] == -1)
    out = context.predict(baseline, draws, dates, background)
    np.testing.assert_array_equal(out['effective_pos_ids'], baseline['donor_ids'])
    for fields in out['arms'].values():
        assert np.isfinite(fields['raw']).all()
        np.testing.assert_array_equal(fields['fine_draws'][..., m.POS], 0)
        np.testing.assert_array_equal(fields['unsupported_positive_energy_fraction'], 0)


@pytest.mark.parametrize('field', ['raw', 'fine_draws'])
def test_one_ulp_original_stream_rejected(experiment, field):
    context, baseline, draws, dates, background = experiment
    bad = copy.deepcopy(baseline); bad[field][0, 0, 0] = np.nextafter(bad[field][0, 0, 0], np.inf)
    with pytest.raises(ValueError, match='Exact'):context.predict(bad, draws, dates, background)


def test_outer_and_unreleased_poison_filtered_before_numeric(experiment):
    context = experiment[0]; history = rows()
    for row in history:
        if row['region'] == 'BPAT':
            row['period_start'] = 'OUTER_METADATA_POISON'; row.update({k: 'OUTER_NUMERIC_POISON' for k in old.SHARES})
        elif row['period_start'] >= '2024-07-01':row.update({k: 'OUTSIDE_WINDOW_NUMERIC_POISON' for k in old.SHARES})
    changed = m.fit_leaf_controls(context.bank, history, outer='BPAT')
    np.testing.assert_array_equal(changed.final.y, context.final.y)
    np.testing.assert_array_equal(changed.final.leaf_ids, context.final.leaf_ids)


def test_cutoff_guard_runs_before_any_model_or_background_access(experiment, monkeypatch):
    context = experiment[0]
    monkeypatch.setattr(context.final, 'conditional', lambda *a: pytest.fail('Illegal origin reached model'))
    with pytest.raises(ValueError, match='deployment cutoff'):
        context.predict({}, np.empty(0), pd.DatetimeIndex(['2024-12-31'], tz='UTC'), {})


def test_no_query_hidden_features_or_rng_mutation(experiment):
    context, baseline, draws, dates, background = experiment
    state = np.random.get_state()
    a = context.predict({**baseline, 'hidden_eight_truth': np.ones((2, 8))}, draws, dates, background)
    b = context.predict({**baseline, 'hidden_eight_truth': np.zeros((2, 8))}, draws, dates, background)
    for arm in m.ARMS:
        for field in m.FIELDS:np.testing.assert_array_equal(a['arms'][arm][field], b['arms'][arm][field])
    current = np.random.get_state()
    assert current[0] == state[0] and current[2:] == state[2:]
    np.testing.assert_array_equal(current[1], state[1])


def test_streams_are_origin_stable_and_separate():
    dates = pd.date_range('2025-01-01', periods=3, tz='UTC')
    ordinary = {d.value: m._rng(d, m.STREAM_SALTS['leaf']).random(128) for d in dates}
    for d in dates[::-1]:
        np.testing.assert_array_equal(m._rng(d, m.STREAM_SALTS['leaf']).random(128), ordinary[d.value])
        assert not np.array_equal(m._rng(d, m.STREAM_SALTS['gaussian']).random(128), ordinary[d.value])
