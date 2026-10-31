"""Synthetic source-prefix, double-exclusion and utility routing tests only."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from scripts import gain_direction_v44 as m
from scripts import monthly_transfer_v25 as original
from scripts import coarse_context_v30 as v30
from scripts import capacity_routing_v36 as v36


def coarse_rows(region='BPAT', periods=20):
    rows = []
    for i, at in enumerate(pd.date_range('2022-01-01', periods=periods, freq='MS', tz='UTC')):
        end = at+pd.offsets.MonthBegin(1)
        rows.append(dict(region=region, period_start=at.isoformat(), period_end=end.isoformat(),
            available_at=(end+pd.Timedelta(days=30)).isoformat(),
            **{v30.SHARES[j]: str(.08+.001*i) for j in v30.ZERO}))
    return rows


def test_source_law_matches_original_mathematics_with_coarse_only_input():
    rows = coarse_rows(); at = pd.Timestamp('2023-08-01', tz='UTC')
    legal = [r for r in rows if pd.Timestamp(r['available_at']) <= at]
    frame = pd.DataFrame([{**{k:r[k] for k in ('region','period_start','period_end','available_at')},
        **dict(zip(original.MG_COLUMNS, v30._visible(r)))} for r in legal])
    expected = original.coarse_law(frame, 'BPAT', pd.DatetimeIndex([at]), budget='MG')
    actual = m.source_mg(rows, ['BPAT/2023-08'])
    np.testing.assert_array_equal(actual['draws'], expected['draws'])
    np.testing.assert_array_equal(actual['points'], expected['mix'])
    assert actual['auditrows'][0]['harmonic_solve_count'] > 0


def test_prefix_before_coarse_access_donors_coldstart_and_order():
    class Guard(dict):
        def __getitem__(self, key):
            if key.startswith('fuelshare_') and (key not in [v30.SHARES[i] for i in v30.ZERO]
                    or pd.Timestamp(super().__getitem__('available_at')) > pd.Timestamp('2022-05-01',tz='UTC')):
                raise AssertionError('Unreleased/current/fine numeric accessed')
            return super().__getitem__(key)
    rows = [Guard(r) for r in coarse_rows('SWPP', 6)]
    result = m.source_mg(rows, ['SWPP/2022-05'])
    audit = result['auditrows'][0]
    assert audit['prefix_count'] == audit['residual_count'] == 3
    assert audit['residual_zero_history_count'] == 3 and audit['prior_count'] == 3
    assert audit['fallback_count'] == 1
    again = m.source_mg(rows[::-1], ['SWPP/2022-05'])
    np.testing.assert_array_equal(again['draws'], result['draws'])
    with pytest.raises(ValueError): m.source_mg(rows, ['PACE/2022-05'])
    with pytest.raises(ValueError): m.source_mg(rows, ['SWPP/2022-05']*2)


def test_empty_prefix_prior_and_origin_seed_order_invariance():
    rows = coarse_rows()
    cold = m.source_mg(rows, ['BPAT/2022-01'])
    np.testing.assert_array_equal(cold['points'][0], [.5,.125,.125,.125,.125])
    np.testing.assert_array_equal(cold['draws'][0], np.tile(cold['points'][0], (128,1)))
    assert cold['auditrows'][0]['residual_count'] == 0 and cold['auditrows'][0]['prior_count'] == 1
    keys = ['BPAT/2023-07', 'BPAT/2023-08']
    a, b = m.source_mg(rows, keys), m.source_mg(rows, keys[::-1])
    np.testing.assert_array_equal(a['draws'], b['draws'][::-1])


def inner_data():
    regions = np.repeat([s for s in m.ALL_SOURCES if s != 'SCL'], 2)
    rng = np.random.default_rng(44); n = len(regions)
    x = rng.normal(size=(n, 31)); x[:, :5] = rng.dirichlet(np.ones(5), n)
    for lag in range(3): x[:, 7+5*lag:12+5*lag] = rng.dirichlet(np.ones(5), n)
    x[:, 30] = np.log(200.)
    q = np.tile([.2, .5, .1, .2], (n, 1)); c = np.full(n, 300.)
    y = rng.dirichlet(np.ones(4), n)
    mg = rng.dirichlet(np.ones(5), (n, 128))
    return x, q, c, y, regions, mg


def test_inner_models_exclude_both_sources_and_do_not_read_held_y():
    args = list(inner_data()); outer, inner = 'BPAT', 'CISO'
    class NoFine:
        def __float__(self): raise AssertionError('held/outer fine consumed')
    y = args[3].astype(object); y[np.isin(args[4], [outer, inner])] = NoFine(); args[3] = y
    result = m.fit_inner(*args, outer=outer, inner=inner)
    held = np.flatnonzero(args[4] == inner)
    fold = dict(inner=inner, held_indices=held, **result['audit'])
    assert fold['fit_count'] == 3
    np.testing.assert_array_equal(result['held_indices'], held)
    assert result['EF_draws'].shape == result['model_EF'].shape == (3, 2, 128)
    np.testing.assert_array_equal(result['CIF_draws'], result['EF_draws']*args[5][held, :, 0][None])
    for fit in result['audit']['fits']:
        train_regions = args[4][fit['training_indices']]
        assert not set(train_regions) & {outer, inner}
        assert fit['sample_weights'].sum() == pytest.approx(154.)
        if fit['model'] == 'Ridge': assert set(train_regions) == set(m.LEGACY_SOURCES)-{outer, inner}
        else:
            assert len(set(train_regions)) == 14
            assert type(fit['parameters']['max_features']) is float
    expected_w = np.any(args[1][held][:, [0,2]] > args[1][result['audit']['fits'][0]['training_indices']][:, [0,2]].max(0)+1e-12, axis=1)
    np.testing.assert_array_equal(result['old_coil_weights'], expected_w)


def test_donor_inner_keeps_all_six_old_sources_and_fpl_mass():
    args = inner_data(); result = m.fit_inner(*args, outer='FPL', inner='SWPP')
    ridge = result['audit']['fits'][0]
    assert len(set(args[4][ridge['training_indices']])) == 6
    assert ridge['sample_weights'].sum() == pytest.approx(156.)


def test_router_binary_endpoints_and_held_current_truth_are_exactly_preserved():
    args = list(inner_data()); held = np.flatnonzero(args[4] == 'CISO')
    args[1][held[0]] = [.4,.3,.1,.2]
    result = m.fit_inner(*args, outer='BPAT', inner='CISO')
    np.testing.assert_array_equal(result['old_coil_weights'], [True,False])
    np.testing.assert_array_equal(result['EF_draws'][0,0], np.full(128,args[1][held[0]]@m.EF))
    np.testing.assert_array_equal(result['EF_draws'][0,1], result['model_EF'][0,1])
    poison = args[0].copy(); poison[held,:5] = np.nan; args[0] = poison
    again = m.fit_inner(*args, outer='BPAT', inner='CISO')
    np.testing.assert_array_equal(result['EF_draws'], again['EF_draws'])
    np.testing.assert_array_equal(result['CIF_draws'], again['CIF_draws'])


def test_meta_features_ignore_current_truth_and_use_population_std():
    x, q, c, y, regions, mg = inner_data()
    before = m.meta_features(x, mg, q, c)
    poison = x.copy(); poison[:, :5] = np.nan
    after = m.meta_features(poison, mg, q, c)
    for k in before: np.testing.assert_array_equal(before[k], after[k])
    assert before['HistoryForecast'].shape == (len(x), 28)
    assert before['HistoryForecastCap'].shape == (len(x), 33)
    np.testing.assert_allclose(before['HistoryForecast'][:, -2], mg[..., 0].mean(1))
    np.testing.assert_allclose(before['HistoryForecast'][:, -1], mg[..., 0].std(1, ddof=0))
    np.testing.assert_allclose(before['HistoryForecastCap'][:, -1], np.exp(x[:,30])*x[:,7]/c)
    particle_x = np.broadcast_to(poison[:,None], (len(x),128,31)).copy()
    particle = m.meta_features(particle_x, mg, q, c)
    for k in before: np.testing.assert_array_equal(before[k], particle[k])
    particle_x[0, 1, 8] += .1
    with pytest.raises(ValueError): m.meta_features(particle_x, mg, q, c)


def test_gain_labels_use_mean_then_absolute_error_with_correct_sign():
    ef = np.array([[[400., 1000.]], [[600., 600.]], [[800., 800.]]])
    cif = np.array([[[80., 800.]], [[120., 480.]], [[160., 640.]]])
    result = m.gain_targets(ef, cif, np.array([600.]), np.array([300.]))
    np.testing.assert_allclose(result[:,0,0], [100/600, -100/600])
    np.testing.assert_allclose(result[:,1,0], [140/1000, 40/1000])
    np.testing.assert_allclose(result[:,2], .5*(result[:,0]+result[:,1]))


def test_meta_models_weight_mass_order_tie_and_fit_counts():
    args = inner_data(); keep = args[4] != 'BPAT'
    features = m.meta_features(args[0][keep], args[5][keep], args[1][keep], args[2][keep])
    query = {k:v[:6].copy() for k,v in features.items()}
    gains = np.zeros((2,3,int(keep.sum())))
    result = m.fit_meta(features, gains, args[4][keep], query, outer='BPAT')
    assert result['gain_prediction'].shape == result['actions'].shape == (30,6)
    assert not result['actions'].any()
    assert result['audit']['sklearn_fit_count'] == 24 and result['audit']['mean_count'] == 6
    assert len(result['audit']['fits']) == 30
    for fit in result['audit']['fits']:
        assert fit['sample_weights'].sum() == pytest.approx(6.)
    assert m.ARMS == tuple(f'Gain-{c}-{t}-{model}' for c in m.CANDIDATES for t in m.TARGETS for model in m.META_MODELS)


def test_meta_actual_fit_counts_nonzero_gain_sign_and_mean_baseline(monkeypatch):
    counts = {'Ridge':0, 'ET':0}
    ridge, et = m.Ridge, m.ExtraTreesRegressor
    class Ridge(ridge):
        def fit(self, *a, **k): counts['Ridge'] += 1; return super().fit(*a, **k)
    class ET(et):
        def fit(self, *a, **k): counts['ET'] += 1; return super().fit(*a, **k)
    monkeypatch.setattr(m, 'Ridge', Ridge); monkeypatch.setattr(m, 'ExtraTreesRegressor', ET)
    args = inner_data(); keep = args[4] != 'BPAT'; n = int(keep.sum())
    features = m.meta_features(args[0][keep], args[5][keep], args[1][keep], args[2][keep])
    query = {k:v[:6] for k,v in features.items()}
    constants = np.array([[.2,-.3,0.],[-.1,.4,0.]])
    result = m.fit_meta(features, np.broadcast_to(constants[:,:,None], (2,3,n)), args[4][keep], query, outer='BPAT')
    assert counts == {'Ridge':12, 'ET':12}
    expected = np.repeat(constants.ravel(), 5)
    np.testing.assert_allclose(result['gain_prediction'], np.broadcast_to(expected[:,None], (30,6)), atol=1e-14)
    np.testing.assert_array_equal(result['actions'], np.broadcast_to((expected > 0)[:,None], (30,6)))


def test_meta_source_guard_precedes_gain_and_feature_numbers():
    class NoNumbers:
        def __array__(self, *a, **k): raise AssertionError('numeric data accessed before metadata guard')
    with pytest.raises(ValueError, match='outer-excluded'):
        m.fit_meta(NoNumbers(), NoNumbers(), np.array(['BPAT']), NoNumbers(), outer='BPAT')


@pytest.mark.parametrize('kind', ['outer', 'unknown', 'badcap', 'badq', 'badmg'])
def test_invalid_inputs_and_meta_outer_guard(kind):
    args = list(inner_data())
    if kind == 'outer':
        with pytest.raises(ValueError): m.fit_inner(*args, outer='BPAT', inner='BPAT')
        return
    if kind == 'unknown': args[4][0] = 'PACE'
    elif kind == 'badcap': args[2][0] = 0
    elif kind == 'badq': args[1][0,0] = -.1
    else: args[5][0,0,0] = np.nan
    with pytest.raises(ValueError): m.fit_inner(*args, outer='BPAT', inner='CISO')
