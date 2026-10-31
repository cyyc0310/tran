"""Synthetic proper-month deduplication, source OOF and shared-stream tests."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from scripts import group_regression_v27 as m
from transcif.research.fuel_transport_v7 import recompose_paths


def make_bank(zero_month=False, unsupported_coal=False):
    rng = np.random.default_rng(27023)
    fields = {k: [] for k in ('q', 'donor_mix', 'support', 'regions', 'days')}
    for region in m.SOURCES[1:]:
        for month in (1, 2):
            generation = rng.uniform(.1, 3, (2, 24, 8))
            if zero_month and region == 'CISO' and month == 1:
                generation[:, :, m.POS] = 0
            if unsupported_coal and region == 'CISO':
                generation[:, :, 0] = 0
            mean = generation.mean((0, 1)); support = mean > 0
            q = np.divide(generation, mean[None, None], out=np.ones_like(generation), where=support[None, None])
            first = pd.Timestamp(year=2024, month=month, day=25 if month == 1 else 1, tz='UTC')
            fields['q'].extend(q)
            fields['donor_mix'].extend(np.repeat((mean/mean.sum())[None], 2, axis=0))
            fields['support'].extend(np.repeat(support[None], 2, axis=0))
            fields['regions'].extend([region]*2)
            fields['days'].extend(pd.date_range(first, periods=2, tz='UTC').asi8)
    bank = {k: np.array(v) for k, v in fields.items()}
    bank['q'] = np.ascontiguousarray(bank['q'].transpose(0, 2, 1)).transpose(0, 2, 1)
    bank['raw_cif'] = recompose_paths(bank['donor_mix'], bank['q'])
    return bank


def stream(bank, *, zero_pos=False, borrow=False):
    rng = np.random.default_rng(27031)
    dates = pd.date_range('2025-01-01', periods=2, tz='UTC')
    draws = rng.dirichlet(np.ones(5), (2, 128))
    if zero_pos:
        draws[:] = [0, .25, .25, .25, .25]
    ids = rng.integers(0, len(bank['days']), size=(2, 128), dtype=np.int64)
    if borrow:
        ids[:] = 0
    fallback = np.full((2, 128), -1, np.int64)
    raw, fine = [], []
    for t in range(2):
        selected = ids[t].copy()
        original_mass = bank['donor_mix'][ids[t]][:, m.POS].sum(1)
        need = (original_mass == 0) & (draws[t, :, 0] > 0)
        if need.any():
            supported = np.flatnonzero(bank['donor_mix'][:, m.POS].sum(1) > 0)[0]
            selected[need] = supported; fallback[t, need] = supported
        mass = bank['donor_mix'][selected][:, m.POS].sum(1)
        ratio = np.divide(bank['donor_mix'][selected][:, m.POS], mass[:, None],
                          out=np.zeros((128, 4)), where=mass[:, None] > 0)
        fd = np.zeros((128, 8)); fd[:, m.ZERO] = draws[t, :, 1:]; fd[:, m.POS] = draws[t, :, :1]*ratio
        q = bank['q'][ids[t]].copy(order='K')
        for particle in np.flatnonzero(need):
            q[particle][:, m.POS] = bank['q'][selected[particle]][:, m.POS]
        fine.append(fd); raw.append(recompose_paths(fd, q))
    return dict(raw=np.stack(raw), fine_draws=np.stack(fine), donor_ids=ids, fallback_ids=fallback), draws, dates


@pytest.fixture(scope='module')
def context():
    return m.fit_regressions(make_bank(), outer='BPAT')


def test_region_month_dedup_and_exact_mixes():
    bank = make_bank()
    table = m.region_months(bank, outer='BPAT')
    assert table['X'].shape == (12, 5) and table['ratios'].shape == (12, 4)
    np.testing.assert_array_equal(table['row_counts'], 2)
    assert len(set(table['keys'])) == 12 and table['audit']['month_mix_exact'] is True
    extra = {k: np.concatenate([v, v[:1]], axis=0) for k, v in bank.items()}
    extra['days'][-1] = pd.Timestamp('2024-01-27', tz='UTC').value
    deduplicated = m.region_months(extra, outer='BPAT')
    np.testing.assert_array_equal(deduplicated['X'], table['X'])
    np.testing.assert_array_equal(deduplicated['ratios'], table['ratios'])
    np.testing.assert_array_equal(m._weights(deduplicated['regions']), m._weights(table['regions']))
    assert deduplicated['row_counts'][0] == 3
    bad = copy.deepcopy(bank)
    bad['donor_mix'][1, 0] = np.nextafter(bad['donor_mix'][1, 0], np.inf)
    with pytest.raises(ValueError, match='must be exact'):
        m.region_months(bad, outer='BPAT')


def test_month_weights_region_balanced_and_mean_one():
    labels = np.array(['CISO', 'CISO', 'ERCO'])
    w = m._weights(labels)
    np.testing.assert_array_equal(w, [.75, .75, 1.5])
    assert w.mean() == 1 and w[labels == 'CISO'].sum() == w[labels == 'ERCO'].sum()
    with pytest.raises(ValueError):
        m._weights(np.array([]))


def test_fixed_parameters_and_all_fold_exclusions(context):
    assert context.models['Ridge'].alpha == 1
    assert context.models['Ridge'].fit_intercept is True
    et = context.models['ET']
    assert (et.n_estimators, et.min_samples_leaf, et.max_features, et.random_state, et.n_jobs) == (128, 2, 1., 17023, 1)
    for name in ('Ridge', 'ET'):
        full = context.audit['fits'][name]['final']
        assert set(full['training_regions']) == set(m.SOURCES)-{'BPAT'}
        for fold in context.audit['fits'][name]['folds']:
            assert 'BPAT' not in fold['training_regions']
            assert fold['excluded_oof_region'] not in fold['training_regions']
            assert len(fold['training_regions']) == 5
            assert not set(fold['training_keys']) & set(fold['predicted_keys'])
            assert np.mean(fold['sample_weights']) == 1
    json.dumps(context.audit, allow_nan=False)


def test_oof_held_response_perturbation_cannot_change_its_prediction(context):
    changed = copy.deepcopy(context.bank)
    held = changed['regions'] == 'CISO'
    changed['donor_mix'][held[:, None] & np.isin(np.arange(8)[None], m.POS)] = changed['donor_mix'][held][:, m.POS[::-1]].ravel()
    changed['raw_cif'] = recompose_paths(changed['donor_mix'], changed['q'])
    other = m.fit_regressions(changed, outer='BPAT')
    np.testing.assert_allclose(context.table['X'], other.table['X'], atol=1e-15)
    mask = context.table['regions'] == 'CISO'
    for name in ('Ridge', 'ET'):
        old = context.audit['fits'][name]['oof_predictions_raw']
        new = other.audit['fits'][name]['oof_predictions_raw']
        np.testing.assert_array_equal(np.array(old)[mask], np.array(new)[mask])
        assert not np.array_equal(context.residuals[name][mask], other.residuals[name][mask])


def test_zero_pos_month_excluded_and_recorded():
    ctx = m.fit_regressions(make_bank(zero_month=True), outer='BPAT')
    assert ctx.table['zero_pos_mask'].sum() == 1
    assert ctx.audit['cohort']['zero_pos_keys'] == ['CISO/2024-01']
    for name in ('Ridge', 'ET'):
        assert 'CISO/2024-01' not in ctx.audit['fits'][name]['final']['training_keys']
        np.testing.assert_array_equal(ctx.residuals[name][ctx.table['zero_pos_mask']], 0)
    baseline, draws, days = stream(ctx.bank, borrow=True)
    out = ctx.predict(baseline, draws, days)
    assert np.all(out['effective_pos_ids'] != 0)
    for name in ('Ridge', 'ET'):
        unprojected = out['arms'][name+'-residual_recompose']['unprojected_ratios']
        point = out['point_ridge_raw' if name == 'Ridge' else 'point_et_raw']
        np.testing.assert_array_equal(unprojected, point+ctx.residuals[name][out['effective_month_ids']])


def test_all_seven_exact_background_matching_pairs_and_residual_formula(context):
    baseline, draws, days = stream(context.bank)
    before = copy.deepcopy(baseline)
    out = context.predict(baseline, draws, days)
    assert tuple(out['arms']) == m.ARMS
    np.testing.assert_array_equal(out['arms'][m.ARMS[0]]['raw'], baseline['raw'])
    np.testing.assert_array_equal(out['arms'][m.ARMS[0]]['fine_draws'], baseline['fine_draws'])
    for field in baseline:
        np.testing.assert_array_equal(baseline[field], before[field])
    for name in ('Ridge', 'ET'):
        point = out['point_ridge_raw' if name == 'Ridge' else 'point_et_raw']
        a, b = [out['arms'][name+'-residual_'+end] for end in ('anchor', 'recompose')]
        np.testing.assert_array_equal(a['unprojected_ratios'], point+context.residuals[name][out['effective_month_ids']])
        np.testing.assert_array_equal(a['fine_draws'], b['fine_draws'])
        np.testing.assert_array_equal(a['ratios'], b['ratios'])
        expected = context.bank['raw_cif'][baseline['donor_ids']]+((a['fine_draws']-context.bank['donor_mix'][baseline['donor_ids']])@m.EF)[..., None]
        np.testing.assert_array_equal(a['raw'], expected)
    np.testing.assert_array_equal(out['arms']['Ridge-plugin_recompose']['unprojected_ratios'], out['point_ridge_raw'])
    for arm in out['arms'].values():
        assert arm['raw'].shape == (2, 128, 24) and np.isfinite(arm['raw']).all()
        np.testing.assert_allclose(arm['fine_draws'][..., m.POS].sum(-1), draws[..., 0], atol=1e-15)
        np.testing.assert_array_equal(arm['fine_draws'][..., m.ZERO], draws[..., 1:])
        np.testing.assert_allclose(arm['ratios'].sum(-1), 1, atol=1e-15)
        lo, hi = out['audit']['source_EF_support']
        ef = arm['ratios']@m.EF[m.POS]
        np.testing.assert_array_equal(arm['source_EF_outside'], (ef < lo-1e-10) | (ef > hi+1e-10))
    json.dumps(out['audit'], allow_nan=False)


def test_query_hidden_composition_is_not_a_feature(context):
    baseline, draws, days = stream(context.bank)
    first = {**baseline, 'query_true_eight_shares': np.ones((2, 8))/8}
    second = {**baseline, 'query_true_eight_shares': np.tile([1., 0, 0, 0, 0, 0, 0, 0], (2, 1))}
    a, b = context.predict(first, draws, days), context.predict(second, draws, days)
    for arm in m.ARMS:
        for field in a['arms'][arm]:
            np.testing.assert_array_equal(a['arms'][arm][field], b['arms'][arm][field])
    assert context.models['Ridge'].n_features_in_ == context.models['ET'].n_features_in_ == 5
    with pytest.raises(ValueError, match='five visible'):
        context.predict(baseline, np.ones((2, 128, 8))/8, days)


@pytest.mark.parametrize('field', ['raw', 'fine_draws'])
def test_baseline_one_ulp_rejected(context, field):
    baseline, draws, days = stream(context.bank)
    baseline[field][0, 0, 0] = np.nextafter(baseline[field][0, 0, 0], np.inf)
    with pytest.raises(ValueError, match='Exact'):
        context.predict(baseline, draws, days)


def test_unsupported_shape_fraction_and_zero_query_mass():
    ctx = m.fit_regressions(make_bank(unsupported_coal=True), outer='BPAT')
    baseline, draws, days = stream(ctx.bank, zero_pos=True)
    out = ctx.predict(baseline, draws, days)
    for arm in out['arms'].values():
        np.testing.assert_array_equal(arm['fine_draws'][..., m.POS], 0)
        np.testing.assert_array_equal(arm['unsupported_positive_energy_fraction'], 0)
    baseline, draws, days = stream(ctx.bank)
    out = ctx.predict(baseline, draws, days)
    for arm in out['arms'].values():
        for t in range(2):
            q = ctx.bank['q'][baseline['donor_ids'][t]]
            energy = arm['fine_draws'][t][:, None, m.POS]*q[:, :, m.POS]
            missing_coal = ~ctx.bank['support'][baseline['donor_ids'][t], 0]
            expected = energy[:, :, 0].sum(1)*missing_coal/energy.sum((1, 2))
            np.testing.assert_allclose(arm['unsupported_positive_energy_fraction'][t], expected, atol=1e-15)
    assert out['arms']['Ridge-plugin_recompose']['unsupported_positive_energy_fraction'].max() > 0


def test_visible_distance_is_row_distance_and_does_not_change_routing(context):
    baseline, draws, days = stream(context.bank)
    out = context.predict(baseline, draws, days)
    expected = np.sqrt(((draws[:, :, None]-context.table['X'][None, None])**2).sum(-1)).min(-1)
    np.testing.assert_array_equal(out['query_nearest_visible_distance'], expected)
    assert 'no routing' in out['audit']['visible_distance_definition']


def test_project_is_nonnegative_simplex_and_not_componentwise_clip():
    np.testing.assert_array_equal(m.project(np.array([2., -1, 0, 0])), [1, 0, 0, 0])
    x = np.array([.8, .6, -.2, -.2])
    y = m.project(x)
    np.testing.assert_allclose(y, [.6, .4, 0, 0], atol=1e-15)
    assert y.sum() == 1
