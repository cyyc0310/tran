"""Synthetic daily context, sealed-head replay and hourly controls only."""
import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

from scripts import daily_bridge_v47 as model
from scripts import fleet_information_v34 as v34
from scripts import forward_context_v31 as v31


class GuardRow(dict):
    def __getitem__(self, key):
        if key in {'fuelshare_coal', 'fuelshare_gas', 'fuelshare_oil', 'fuelshare_other'}:
            raise AssertionError('Hidden fine field accessed')
        return super().__getitem__(key)


class Poison:
    def __array__(self, *args, **kwargs):
        raise AssertionError('Training numerics accessed before metadata guard')


def monthly_rows():
    rows = []
    for at in pd.date_range('2024-08-01', '2025-06-01', freq='MS', tz='UTC'):
        end = at + pd.offsets.MonthBegin(1)
        rows.append(GuardRow(region='BPAT', period_start=at.isoformat(), period_end=end.isoformat(),
            available_at=(end + pd.Timedelta(days=9)).isoformat(), total_mwh='72000',
            fuelshare_nuclear='.1', fuelshare_hydro='.2', fuelshare_wind='.1', fuelshare_solar='.1'))
    # This row is never eligible in the registered daily query window.
    rows[-1]['total_mwh'] = Poison()
    rows[-1]['fuelshare_hydro'] = Poison()
    return rows


@pytest.fixture(scope='module')
def query():
    rng = np.random.default_rng(4701)
    mg = rng.dirichlet(np.ones(5), size=(181, 128))
    return model.daily_features(monthly_rows(), 'BPAT', mg, np.array([.3, .4, .1, .2]), 1000.)


def test_daily_context_release_switch_and_month_start_parity(query):
    assert query['features']['Size'].shape == (181, 128, 31)
    jan9, jan10 = 8, 9
    assert query['context_keys'][jan9, 0] == 'BPAT/2024-11'
    assert query['context_keys'][jan10, 0] == 'BPAT/2024-12'
    assert query['context_release_age_days'][jan10, 0] == 0
    assert query['context_month_age'][jan9, 0] == 2
    assert query['context_month_age'][jan10, 0] == 1
    ids = np.flatnonzero(model.DAYS.day == 1)
    dates = model.DAYS[ids]
    context = v31._context(monthly_rows(), 'BPAT', dates)
    old = v31._features(query['MG_draws'][ids], context, dates)
    for name in context:
        np.testing.assert_array_equal(query[name][ids], context[name])
    np.testing.assert_array_equal(query['features']['Size'][ids], old['Size'])
    cap = v34._features(old['Size'], np.tile([.3, .4, .1, .2], (6, 1)), np.full(6, 1000.))
    np.testing.assert_array_equal(query['features']['SizeCap'][ids], cap['SizeCap'])
    np.testing.assert_array_equal(query['features']['DirectEF'], query['features']['SizeCap'])
    assert np.all(query['context_release_ns'] <= query['origin_ns'][:, None])
    names = model.FEATURE_NAMES['Size']
    release_col = names.index('lag0_release_age_days')
    assert query['features']['Size'][jan9, 0, release_col] - query['features']['Size'][jan9-1, 0, release_col] == 1


def test_release_time_is_not_floored_and_other_source_numerics_are_unread():
    rows = monthly_rows()
    rows[4]['available_at'] = '2025-01-10T12:00:00+00:00'
    other = GuardRow(rows[0])
    other.update(region='CISO', total_mwh=Poison(), fuelshare_hydro=Poison())
    result = model.daily_features(rows+[other], 'BPAT', np.full((181, 128, 5), .2),
                                  np.array([0., 1., 0., 0.]), 1000.)
    assert result['context_keys'][9, 0] == 'BPAT/2024-11'
    assert result['context_keys'][10, 0] == 'BPAT/2024-12'
    assert result['context_release_age_days'][10, 0] == .5
    assert np.all(result['features']['SizeCap'][..., 31] == 0)


@pytest.mark.parametrize('kind', ['region', 'mg_shape', 'mg_complex', 'mg_negative', 'q_sum', 'capacity'])
def test_daily_invalid_inputs_precede_any_row_access(kind):
    class ForbiddenRows:
        def __iter__(self):
            raise AssertionError('Invalid query consumed rows')
    mg = np.full((181, 128, 5), .2)
    region, q, c = 'BPAT', np.full(4, .25), 1000.
    if kind == 'region': region = 'PACE'
    elif kind == 'mg_shape': mg = mg[:-1]
    elif kind == 'mg_complex': mg = mg.astype(complex)
    elif kind == 'mg_negative': mg[0, 0] = [-.1, .3, .3, .3, .2]
    elif kind == 'q_sum': q *= 2
    else: c = 0.
    with pytest.raises(ValueError):
        model.daily_features(ForbiddenRows(), region, mg, q, c)


def records():
    rng = np.random.default_rng(4702)
    old_sources = [s for s in model.SOURCES if s != 'BPAT']
    extra = ['SWPP', 'AVA', 'BANC', 'JEA', 'SC', 'NWMT', 'WACM', 'SEC', 'SCEG']
    old_n = len(old_sources) * 2
    sources = np.repeat(old_sources + extra, 2)
    keys = [f'{s}/2023-{m:02d}' for s in old_sources + extra for m in (5, 6)]
    starts = np.array([pd.Timestamp(k.split('/')[1]+'-01', tz='UTC').value for k in keys], np.int64)
    ends = np.array([(pd.Timestamp(int(t), tz='UTC')+pd.offsets.MonthBegin(1)).value for t in starts], np.int64)
    size = rng.normal(size=(len(keys), 31))
    size[:, :5] = rng.dirichlet(np.ones(5), size=len(keys))
    q = rng.dirichlet(np.ones(4), size=len(keys))
    # log_MW0 and lag0_POS need not have physical interpretation in this replay toy.
    x = {'Size': size, **v34._features(size, q, np.full(len(keys), 1000.))}
    y = rng.dirichlet(np.ones(4), size=len(keys))
    result = []
    for scope in ('Legacy', 'Expanded'):
        n = old_n if scope == 'Legacy' else len(keys)
        r = sources[:n]
        mass = n if scope == 'Legacy' else 154
        w = np.array([mass/(len(set(r))*np.count_nonzero(r == s)) for s in r])
        for family in ('Size', 'SizeCap', 'DirectEF'):
            for method in ('Ridge', 'ET'):
                xx = x[family][:n]
                target = (y[:n] @ model.EF[model.POS])/1000 if family == 'DirectEF' else y[:n]
                mean, scale = np.zeros(xx.shape[1]), np.ones(xx.shape[1])
                if method == 'Ridge':
                    mean = np.average(xx, axis=0, weights=w)
                    scale = np.sqrt(np.average((xx-mean)**2, axis=0, weights=w))
                    scale[scale <= 1e-12] = 1
                    fitted = Ridge(**model.PARAMETERS[method]).fit((xx-mean)/scale, target, sample_weight=w)
                    saved = dict(coefficients=fitted.coef_.copy(), intercept=np.asarray(fitted.intercept_).copy())
                else:
                    fitted = ExtraTreesRegressor(**model.PARAMETERS[method]).fit(xx, target, sample_weight=w)
                    saved = dict(feature_importances=fitted.feature_importances_.copy())
                result.append(dict(arm=family+'-'+method, family=family, model=method, excluded_outer='BPAT',
                    parameters=model.PARAMETERS[method].copy(), feature_names=model.FEATURE_NAMES[family],
                    training_keys=keys[:n], training_regions=r.copy(), training_period_start_ns=starts[:n].copy(),
                    training_period_end_ns=ends[:n].copy(), training_available_ns=ends[:n]+86400*10**9,
                    training_X=xx.copy(), training_y=target.copy(), sample_weights=w.copy(),
                    scaler_mean=mean, scaler_scale=scale,
                    projection_kind='scalar_EF' if family == 'DirectEF' else 'simplex', **saved))
    return result


@pytest.fixture(scope='module')
def frozen():
    return records()


def test_replay_exact_six_et_and_zero_ridge_fits(query, frozen, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Ridge replay attempted a fit')
    monkeypatch.setattr(Ridge, 'fit', forbidden)
    original, calls = ExtraTreesRegressor.fit, []
    def tracked(self, *args, **kwargs):
        calls.append(self.max_features)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(ExtraTreesRegressor, 'fit', tracked)
    result = model.replay_heads(frozen, query)
    assert result['EF_draws'].shape == result['projection_l1'].shape == (12, 181, 128)
    assert calls == [1.0] * 6 and all(type(x) is float for x in calls)
    assert result['audit']['fit_count'] == 6 and result['audit']['Ridge_replays'] == 6
    assert len(result['raw_predictions']) == 12
    assert np.all((result['EF_draws'] >= 400-1e-9) & (result['EF_draws'] <= 1000+1e-9))
    np.testing.assert_array_equal(result['CIF_draws'], result['EF_draws']*query['MG_draws'][None, ..., 0])
    for i in (0, 2, 4, 6, 8, 10):
        f = frozen[i]
        xx = query['features'][f['family']]
        raw = ((xx-f['scaler_mean'])/f['scaler_scale']) @ f['coefficients'].T + f['intercept']
        expected = np.clip(raw, .4, 1.)*1000 if f['family'] == 'DirectEF' else model.v30.project(raw) @ model.EF[model.POS]
        np.testing.assert_array_equal(result['EF_draws'][i], expected)
        np.testing.assert_array_equal(result['raw_predictions'][i], raw)
        if f['family'] == 'DirectEF':
            np.testing.assert_array_equal(result['projection_l1'][i], abs(raw-np.clip(raw, .4, 1.)))


@pytest.mark.parametrize('kind', ['outer', 'parameters', 'maxfeatures_int', 'feature_order', 'future', 'key', 'cohort'])
def test_all_metadata_checked_before_any_training_conversion(query, frozen, kind):
    fits = copy.deepcopy(frozen)
    fits[0]['training_X'] = Poison()
    f = fits[-1]
    if kind == 'outer': f['excluded_outer'] = 'FPL'
    elif kind == 'parameters': f['parameters']['random_state'] = 3
    elif kind == 'maxfeatures_int': f['parameters']['max_features'] = 1
    elif kind == 'feature_order': f['feature_names'] = tuple(reversed(f['feature_names']))
    elif kind == 'future': f['training_available_ns'][0] = pd.Timestamp('2025-02-01', tz='UTC').value
    elif kind == 'key': f['training_keys'][0] = 'PACE/2023-05'
    else: f['training_keys'] = f['training_keys'][::-1]
    with pytest.raises(ValueError):
        model.replay_heads(fits, query)


@pytest.mark.parametrize('kind', ['et_scaler', 'ridge_scaler', 'importance', 'weights', 'scalar_y', 'x_alignment'])
def test_training_replay_guards(query, frozen, kind):
    fits = copy.deepcopy(frozen)
    if kind == 'et_scaler': fits[1]['scaler_mean'][0] = .01
    elif kind == 'ridge_scaler': fits[0]['scaler_scale'][0] *= 2
    elif kind == 'importance': fits[1]['feature_importances'] = np.zeros(31)
    elif kind == 'weights': fits[0]['sample_weights'] *= 2
    elif kind == 'scalar_y': fits[4]['training_y'][0] = .9
    else: fits[2]['training_X'][0, 0] += .1
    with pytest.raises(ValueError):
        model.replay_heads(fits, query)


@pytest.fixture(scope='module')
def paths(query):
    rng = np.random.default_rng(4703)
    mg = query['MG_draws'].copy()
    mg[:, :2] = [0., .25, .25, .25, .25]
    fine = np.empty((181, 128, 8))
    fine[..., model.POS] = mg[..., :1] * rng.dirichlet(np.ones(4), size=(181, 128))
    fine[..., model.ZERO] = mg[..., 1:]
    mean = np.broadcast_to([.25, .25, .25, .25], (181, 128, 4)).copy()
    u = fine @ model.EF
    raw = u[..., None] + 80*np.sin(np.arange(24)*np.pi/12) + rng.normal(0, 30, (181, 128, 1))
    raw[:, :2] = 12
    s = mg[None, ..., 0] * rng.uniform(400, 1000, (15, 181, 128))
    args = (raw, fine, mean, mg, s, raw+5, raw-7)
    return args, model.build_paths(*args)


def test_path_registry_references_and_all_controls(paths):
    args, result = paths
    assert len(model.LEVEL_ARMS) == 15 and len(model.ARMS) == len(set(model.ARMS)) == 63
    assert [*model.BUDGET_BY_ARM.values()].count('B0') == 18
    assert [*model.BUDGET_BY_ARM.values()].count('B1') == 44
    for i, source in enumerate((args[0], args[5], args[6])):
        np.testing.assert_array_equal(result['raw'][i], source)
    assert result['raw'].shape == (63, 181, 128, 24)
    assert all(x.shape == (60, 181) for x in result['diagnostics'].values())
    assert max(result['audit']['invariant_max_abs'].values()) < 1e-9
    for i in range(15):
        keep, center, flat = (result['raw'][start+i] for start in (18, 33, 48))
        np.testing.assert_allclose(center.mean(1), keep.mean(1), atol=1e-10, rtol=0)
        np.testing.assert_allclose(flat.mean(-1), keep.mean(-1), atol=1e-10, rtol=0)
        assert np.max(flat.std(-1)) < 1e-10
        assert np.max(result['diagnostics']['raw_centered_shape_max_abs_change'][45+i]) > 70
    assert result['raw'].min() < 0
    assert np.all(result['diagnostics']['zero_mg_pos_draw_count'] == 2)
    assert result['diagnostics']['zero_mg_pos_draw_count'].dtype == np.int64
    assert np.all(result['diagnostics']['zero_mg_pos_nonzero_raw_hour_count'][:15] == 48)


def test_registered_month_start_paths_match_sealed_v46_formula(paths):
    args, result = paths
    take = np.flatnonzero(model.DAYS.day == 1)
    old = [a[take] if i != 4 else a[:, take] for i, a in enumerate(args)]
    old[4] = np.concatenate((old[4], np.repeat(old[4][:1], 30, axis=0)))
    reference = model.v46.build(*old)
    names = list(model.v46.ARMS)
    for i, name in enumerate(model.ARMS):
        if name in names:
            np.testing.assert_allclose(result['raw'][i, take], reference['raw'][names.index(name)],
                                       atol=1e-9, rtol=0)


@pytest.mark.parametrize('kind', ['shape', 'complex', 'fine_mg', 'level_bounds'])
def test_path_domain_guards(paths, kind):
    args = list(paths[0])
    if kind == 'shape': args[0] = args[0][:-1]
    elif kind == 'complex': args[1] = args[1].astype(complex)
    elif kind == 'fine_mg':
        args[1] = args[1].copy()
        args[1][0, 2, [3, 4]] = args[1][0, 2, [4, 3]]
    else:
        args[4] = args[4].copy()
        args[4][0, 0, 2] = 1001*args[3][0, 2, 0]
    with pytest.raises(ValueError):
        model.build_paths(*args)
