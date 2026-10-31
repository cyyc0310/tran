"""Synthetic B1 point heads; no archived files, real capacity or target reads."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from scripts import coarse_context_v30 as v30
from scripts import forward_context_v31 as v31
from scripts import fleet_information_v34 as m
from scripts import fleet_inventory_v33 as inventory


EXPECTED_ARMS = ('Size-Ridge', 'Size-ET', 'Capacity-OP', 'Capacity-Operable', 'Utilization-OP',
                 'SizeCap-Ridge', 'SizeCap-ET', 'IntensiveCap-Ridge', 'IntensiveCap-ET',
                 'DirectEF-Ridge', 'DirectEF-ET')


def capacity_rows(multiplier=1.):
    rows = []
    for year in range(2020, 2024):
        for i, region in enumerate(v31.SOURCES):
            values = np.array([80+11*i, 130-4*i, 9+i, 45, 35, 30, 20, 7+2*i], float)
            values *= multiplier*(1+.05*(year-2020))
            row = dict(year=str(year), region=region,
                       **{f'capacity_{f}_mw': str(value) for f, value in zip(v30.FUELS, values)},
                       capacity_pos_mw=str(values[v31.POS].sum()), capacity_total_mw=str(values.sum()))
            rows.append(row)
    return rows


def fixture_data():
    region = 'MISO'
    keys, regions, dates, features, labels = [], [], [], [], []
    for k, source in enumerate(s for s in v31.SOURCES if s != region):
        for j in range(k+2):
            at = pd.Timestamp('2022-01-01', tz='UTC') + pd.DateOffset(months=4*j)
            pos = .45+.01*j+.02*k
            g = np.r_[pos, np.repeat((1-pos)/4, 4)]
            lag = np.tile(np.r_[.6, np.repeat(.1, 4)], (3, 1))
            phase = 2*np.pi*(at.month-1)/12
            x = np.r_[g, np.sin(phase), np.cos(phase), lag.ravel(), [1, 2, 3],
                      [15, 45, 75], [.1+.002*j, -.02], np.log(700+70*k+20*j)]
            y = [.15+.02*k+.005*j, .55-.02*k-.005*j, .1+.001*j, .2-.001*j]
            keys.append(v30._key(source, at)); regions.append(source); dates.append(at)
            features.append(x); labels.append(y)
    x = np.asarray(features); y = np.asarray(labels); regions = np.asarray(regions)
    n = len(x)
    weights = np.array([n/(6*np.count_nonzero(regions == source)) for source in regions])
    starts = np.array([at.value for at in dates], dtype=np.int64)
    ends = np.array([(at+pd.offsets.MonthBegin(1)).value for at in dates], dtype=np.int64)
    audit = dict(region=region, outer=region, fit_count=10,
                 fits=[dict(arm=arm) for arm in v30.ARMS if arm not in ('Size-Ridge', 'Size-ET')])
    for model in ('Ridge', 'ET'):
        mean = np.average(x, axis=0, weights=weights) if model == 'Ridge' else np.zeros(31)
        scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=weights)) if model == 'Ridge' else np.ones(31)
        scale[scale <= 1e-12] = 1
        audit['fits'].append(dict(arm='Size-'+model, family='Size', model=model, excluded_outer=region,
            parameters=v31.PARAMETERS[model].copy(), feature_names=v31.FEATURE_NAMES['Size'],
            training_keys=list(keys), training_regions=regions.copy(), training_X=x.copy(), training_y=y.copy(),
            sample_weights=weights.copy(), scaler_mean=mean, scaler_scale=scale,
            training_period_start_ns=starts.copy(), training_period_end_ns=ends.copy(),
            training_available_ns=ends+5*86400*10**9))
    origins = pd.date_range('2025-01-01', periods=6, freq='MS', tz='UTC').as_unit('ns')
    pos = np.linspace(.2, .8, 128)
    draws = np.tile(np.c_[pos, np.repeat(((1-pos)/4)[:, None], 4, axis=1)][None], (6, 1, 1))
    context = dict(context_g=np.tile(np.r_[.55, np.repeat(.45/4, 4)], (6, 3, 1)),
                   context_MW=np.tile([800., 790., 780.], (6, 1)),
                   context_month_age=np.tile([1., 2., 3.], (6, 1)),
                   context_release_age_days=np.tile([15., 45., 75.], (6, 1)))
    query = v31._features(draws, context, origins)['Size']
    base_ef = np.array([500+20*i+50*pos for i in range(6)])
    ef = np.stack([base_ef+5*i for i in range(6)])
    cif = ef*draws[None, ..., 0]
    archived = dict(origins=origins.asi8.copy(), arms=np.asarray(v31.ARMS), MG_draws=draws,
                    query_feature_Size=query, EF_draws=ef, CIF_draws=cif,
                    EF_point=ef.mean(-1), CIF_point=cif.mean(-1),
                    projection_l1=np.full((6, 6, 128), .123))
    sensitive = capacity_rows(1.2)
    for row in sensitive:
        row['capacity_coal_mw'] = str(float(row['capacity_coal_mw'])+35)
        reconcile(row)
    return audit, archived, capacity_rows(), sensitive


@pytest.fixture(scope='module')
def fixture():
    args = fixture_data()
    return args, m.fit_predict(*args, region='MISO')


def invoke(args, **overrides):
    names = ('parent_audit', 'archived_query', 'capacity_op_rows', 'capacity_operable_rows')
    values = dict(zip(names, args)); values.update(overrides)
    return m.fit_predict(**values, region='MISO')


def same(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same(a, b)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    else:
        assert left == right


def test_constants_shapes_seven_fits_and_old_arms_bitwise_copy(fixture):
    args, out = fixture
    assert m.ARMS == EXPECTED_ARMS
    assert m.SOURCES == v31.SOURCES
    np.testing.assert_array_equal(m.POS, v31.POS)
    assert len(m.FEATURE_NAMES['SizeCap']) == 36
    assert len(m.FEATURE_NAMES['IntensiveCap']) == 35
    assert m.FEATURE_NAMES['DirectEF'] == m.FEATURE_NAMES['SizeCap']
    assert out['CIF_draws'].shape == out['EF_draws'].shape == out['projection_l1'].shape == (11, 6, 128)
    assert out['CIF_point'].shape == out['EF_point'].shape == (11, 6)
    for field in ('CIF_draws', 'EF_draws', 'CIF_point', 'EF_point', 'projection_l1'):
        np.testing.assert_array_equal(out[field][:2], args[1][field][[4, 5]])
        assert np.isfinite(out[field]).all()
    assert out['audit']['fit_count'] == len(out['audit']['fits']) == 7
    assert {f['arm'] for f in out['audit']['fits']} == set(m.ARMS)-set(m.ARMS[:4])
    assert out['projection_kind'].tolist() == ['archived_simplex', 'archived_simplex', 'none', 'none', 'none',
                                              'simplex', 'simplex', 'simplex', 'simplex', 'scalar_EF', 'scalar_EF']
    for flag in ('hourly_values_read', 'target_values_opened', 'unseen_values_opened', 'new_training_labels_used'):
        assert out['audit'][flag] is False


def test_cif_particle_mean_not_product_of_means_and_capacity_anchors(fixture):
    args, out = fixture
    np.testing.assert_allclose(out['CIF_draws'], out['EF_draws']*args[1]['MG_draws'][None, ..., 0])
    np.testing.assert_allclose(out['CIF_point'], out['CIF_draws'].mean(-1))
    np.testing.assert_allclose(out['EF_point'], out['EF_draws'].mean(-1))
    expected = np.array([float(next(r for r in args[2] if r['year'] == '2023' and r['region'] == 'MISO')[f'capacity_{f}_mw'])
                         for f in ('coal', 'gas', 'oil', 'other')])
    anchor_ef = (expected/expected.sum())@v31.EF[v31.POS]
    np.testing.assert_allclose(out['EF_draws'][2], anchor_ef)
    sensitive = next(r for r in args[3] if r['year'] == '2023' and r['region'] == 'MISO')
    weights = np.array([float(sensitive[f'capacity_{f}_mw']) for f in ('coal', 'gas', 'oil', 'other')])
    np.testing.assert_allclose(out['EF_draws'][3], weights/weights.sum()@v31.EF[v31.POS])
    assert not np.array_equal(out['EF_draws'][2], out['EF_draws'][3])
    assert np.max(np.abs(out['CIF_point'][5:]-out['EF_point'][5:]*args[1]['MG_draws'][..., 0].mean(-1))) > 1e-7


def test_frozen_training_identity_weights_feature_names_and_inventory_asof(fixture):
    args, out = fixture
    reference = next(f for f in args[0]['fits'] if f['arm'] == 'Size-Ridge')
    for fit in out['audit']['fits']:
        assert set(fit['training_regions']) == set(m.SOURCES)-{'MISO'}
        assert fit['excluded_outer'] == 'MISO'
        assert fit['training_keys'] == reference['training_keys']
        np.testing.assert_array_equal(fit['sample_weights'], reference['sample_weights'])
        for key in ('training_period_start_ns', 'training_period_end_ns', 'training_available_ns'):
            np.testing.assert_array_equal(fit[key], reference[key])
    assert [(r['year'], r['available_at']) for r in out['audit']['query_inventory']] == [(2023, '2024-09-23T00:00:00+00:00')]*6
    for item, at_ns in zip(out['audit']['training_inventory'], reference['training_period_start_ns']):
        at = pd.Timestamp(int(at_ns), unit='ns', tz='UTC')
        assert item['year'] == at.year-2
        assert pd.Timestamp(item['available_at']) <= at


def test_train_only_weighted_scalers_direct_target_and_fixed_fit_parameters(fixture):
    args, out = fixture
    reference_y = next(f for f in args[0]['fits'] if f['arm'] == 'Size-Ridge')['training_y']
    for fit in out['audit']['fits']:
        if fit['arm'] == 'Utilization-OP':
            assert fit['parameters']['max_nfev'] == 10000
            assert fit['parameters']['ftol'] == fit['parameters']['xtol'] == fit['parameters']['gtol'] == 1e-10
            assert fit['beta'][1] == 0
            assert len(fit['beta']) == 4 and np.max(np.abs(fit['beta'])) <= 6
            assert fit['fit_result']['success'] is True
            assert 0 <= fit['fit_result']['jacobian_rank'] <= 3
            assert len(fit['fit_result']['active_mask']) == 3
            continue
        w, x = fit['sample_weights'], fit['training_X']
        if fit['model'] == 'Ridge':
            mean = np.average(x, axis=0, weights=w)
            scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=w)); scale[scale <= 1e-12] = 1
            np.testing.assert_allclose(fit['scaler_mean'], mean, atol=1e-12, rtol=0)
            np.testing.assert_allclose(fit['scaler_scale'], scale, atol=1e-12, rtol=0)
            assert fit['parameters']['alpha'] == 1
        else:
            assert fit['parameters'] == v31.PARAMETERS['ET']
            assert type(fit['parameters']['max_features']) is float
        if fit['arm'].startswith('DirectEF'):
            np.testing.assert_allclose(fit['training_y'], reference_y@v31.EF[v31.POS]/1000)
        else:
            np.testing.assert_array_equal(fit['training_y'], reference_y)


def test_uniform_volume_capacity_rescaling_leaves_intensive_features_invariant(fixture):
    args, expected = fixture
    changed = deepcopy(args)
    for fit in changed[0]['fits']:
        if fit['arm'] not in ('Size-Ridge', 'Size-ET'):
            continue
        fit['training_X'][:, v31.FEATURE_NAMES['Size'].index('log_MW0')] += np.log(8)
        if fit['model'] == 'Ridge':
            fit['scaler_mean'] = np.average(fit['training_X'], axis=0, weights=fit['sample_weights'])
            fit['scaler_scale'] = np.sqrt(np.average((fit['training_X']-fit['scaler_mean'])**2, axis=0, weights=fit['sample_weights']))
            fit['scaler_scale'][fit['scaler_scale'] <= 1e-12] = 1
    changed[1]['query_feature_Size'][..., v31.FEATURE_NAMES['Size'].index('log_MW0')] += np.log(8)
    for table in changed[2:]:
        for row in table:
            for key in row:
                if key.startswith('capacity_'):
                    row[key] = str(float(row[key])*8)
    actual = invoke(changed)
    np.testing.assert_allclose(actual['query_feature_IntensiveCap'], expected['query_feature_IntensiveCap'], atol=1e-12, rtol=0)
    for left, right in zip(actual['audit']['fits'], expected['audit']['fits']):
        if left['arm'].startswith('IntensiveCap'):
            np.testing.assert_allclose(left['training_X'], right['training_X'], atol=1e-12, rtol=0)


def test_pressure_uses_lag0_pos_and_is_constant_across_mg_particles(fixture):
    args, expected = fixture
    changed = deepcopy(args)
    draws = changed[1]['MG_draws']
    draws[..., 0] = np.linspace(.1, .9, 128)
    draws[..., 1:] = (1-draws[..., :1])/4
    changed[1]['query_feature_Size'][..., :5] = draws
    changed[1]['CIF_draws'] = changed[1]['EF_draws']*draws[None, ..., 0]
    changed[1]['CIF_point'] = changed[1]['CIF_draws'].mean(-1)
    actual = invoke(changed)
    i = m.FEATURE_NAMES['SizeCap'].index(m.CAPACITY_FEATURE_NAMES[-1])
    np.testing.assert_array_equal(actual['query_feature_SizeCap'][..., i], expected['query_feature_SizeCap'][..., i])
    np.testing.assert_array_equal(actual['query_feature_SizeCap'][..., i], actual['query_feature_SizeCap'][:, :1, i]+np.zeros((6, 128)))
    assert 'log_MW0' not in m.FEATURE_NAMES['IntensiveCap']


def reconcile(row):
    row['capacity_pos_mw'] = str(sum(float(row[f'capacity_{v30.FUELS[i]}_mw']) for i in v31.POS))
    row['capacity_total_mw'] = str(sum(float(row[f'capacity_{f}_mw']) for f in v30.FUELS))


def test_zero_registered_fuel_stays_zero_in_capacity_and_utilization_support(fixture):
    args, _ = fixture
    changed = deepcopy(args)
    for table in changed[2:]:
        for row in table:
            row['capacity_oil_mw'] = '0'; reconcile(row)
    result = invoke(changed)
    ratio_arm = result['ratio_arms'].tolist()
    for arm in ('Capacity-OP', 'Capacity-Operable', 'Utilization-OP'):
        assert np.all(result['ratios'][ratio_arm.index(arm), ..., 2] == 0)
    fit = next(f for f in result['audit']['fits'] if f['arm'] == 'Utilization-OP')
    assert fit['fit_result']['jacobian_rank'] < 3  # Diagnostic only: no rank-based fallback.


class PoisonRows:
    def __iter__(self):
        raise AssertionError('numeric/source table accessed before metadata guard')


@pytest.mark.parametrize('origins', [np.array([], dtype=np.int64), ['2024-12-01T00:00:00Z'],
                                    ['2025-07-01T00:00:00Z'], ['2025-01-01T00:00:00'],
                                    ['2025-01-01T00:00:00Z']*6])
def test_origin_guard_precedes_parent_capacity_and_fitting(origins):
    with pytest.raises(ValueError):
        m.fit_predict({}, {'origins': origins}, PoisonRows(), PoisonRows(), region='MISO')


def test_training_metadata_guard_precedes_training_numeric_access(fixture):
    args, _ = fixture
    changed = deepcopy(args)
    head = next(f for f in changed[0]['fits'] if f['arm'] == 'Size-Ridge')
    head['training_available_ns'][0] = pd.Timestamp('2025-01-02', tz='UTC').value
    head['training_X'] = 'FORBIDDEN_TRAIN_NUMBERS'
    with pytest.raises(ValueError, match='metadata|released|cutoff'):
        invoke(changed)


@pytest.mark.parametrize('field', ['training_X', 'training_y', 'sample_weights', 'training_period_start_ns', 'training_period_end_ns', 'training_available_ns'])
def test_size_heads_must_have_identical_cohort_and_training_values(fixture, field):
    args, _ = fixture
    changed = deepcopy(args)
    et = next(f for f in changed[0]['fits'] if f['arm'] == 'Size-ET')
    et[field].flat[0] += .1 if not field.endswith('_ns') else 1
    with pytest.raises(ValueError):
        invoke(changed)


def test_outer_region_exclusion_and_feature_names_are_enforced(fixture):
    args, _ = fixture
    for mode in ('outer', 'names', 'weights'):
        changed = deepcopy(args)
        for fit in changed[0]['fits']:
            if not fit['arm'].startswith('Size-'):
                continue
            if mode == 'outer':
                fit['training_regions'][0] = 'MISO'
            elif mode == 'names':
                fit['feature_names'] = tuple(reversed(fit['feature_names']))
            else:
                fit['sample_weights'][0] *= 2
        with pytest.raises(ValueError):
            invoke(changed)


@pytest.mark.parametrize('mode', ['duplicate', 'missing', 'nan', 'negative', 'zero-pos', 'future-year', 'mismatched-pos', 'operable-smaller'])
def test_bad_capacity_rows_fail_without_fallback(fixture, mode):
    args, _ = fixture
    changed = deepcopy(args)
    row = changed[2][0]
    if mode == 'duplicate':
        changed[2].append(deepcopy(row))
    elif mode == 'missing':
        changed[2].pop(0)
    elif mode == 'nan':
        row['capacity_coal_mw'] = 'NaN'
    elif mode == 'negative':
        row['capacity_oil_mw'] = '-1'
    elif mode == 'zero-pos':
        for i in v31.POS:
            row[f'capacity_{v30.FUELS[i]}_mw'] = '0'
        reconcile(row)
    elif mode == 'future-year':
        row['year'] = '2024'
    elif mode == 'mismatched-pos':
        row['capacity_pos_mw'] = '1'
    else:
        changed[3][0]['capacity_gas_mw'] = '1'; reconcile(changed[3][0])
    with pytest.raises(ValueError):
        invoke(changed)


def test_extra_non_source_capacity_and_unused_archive_members_are_not_numeric_inputs(fixture):
    args, expected = fixture
    changed = deepcopy(args)
    for table in changed[2:]:
        table.append({'region': 'UNSEEN', 'year': 'YEAR_POISON', 'capacity_coal_mw': 'CAP_POISON'})
    changed[1]['unknown_labels'] = PoisonRows()
    for key in ('EF_draws', 'CIF_draws', 'EF_point', 'CIF_point', 'projection_l1'):
        changed[1][key] = changed[1][key].astype(object)
        changed[1][key][:4] = 'UNUSED_HEAD_POISON'
    same(expected, invoke(changed))


def test_query_schema_or_mg_correspondence_changed_is_rejected(fixture):
    args, _ = fixture
    for mode in ('width', 'nan', 'mg', 'history-particle'):
        changed = deepcopy(args)
        if mode == 'width':
            changed[1]['query_feature_Size'] = changed[1]['query_feature_Size'][..., :-1]
        elif mode == 'nan':
            changed[1]['query_feature_Size'][0, 0, -1] = np.nan
        elif mode == 'mg':
            changed[1]['query_feature_Size'][0, 0, 0] += .01
        else:
            changed[1]['query_feature_Size'][0, 0, -1] += 1
        with pytest.raises(ValueError):
            invoke(changed)


def test_inventory_release_guard_precedes_capacity_values(fixture, monkeypatch):
    args, _ = fixture
    monkeypatch.setattr(inventory, 'RELEASE_DATES', {**inventory.RELEASE_DATES, 2020: '2023-01-01'})
    with pytest.raises(ValueError, match='release'):
        m.fit_predict(args[0], args[1], PoisonRows(), PoisonRows(), region='MISO')


def test_utilization_recovers_known_relative_multiplier_and_residual_weights():
    caps = np.array([[.1+.01*i, .5-.01*i, .2, .2] for i in range(16)])
    expected = np.array([.7, 0, -.4, .2])
    y = caps*np.exp(expected); y /= y.sum(1, keepdims=True)
    w = np.arange(1, 17, dtype=float)
    beta, result = m._fit_utilization(caps, y, w)
    np.testing.assert_allclose(m._utilization_ratios(caps, beta), y, atol=1e-8, rtol=0)
    assert beta[1] == 0 and result['success']
    raw = m._utilization_ratios(caps, beta)-y
    np.testing.assert_allclose(result['residuals'], (np.sqrt(w)[:, None]*raw).ravel(), atol=1e-12)


def test_optimizer_failure_is_not_silently_replaced(fixture, monkeypatch):
    args, _ = fixture
    from types import SimpleNamespace
    monkeypatch.setattr(m, 'least_squares', lambda *a, **kw: SimpleNamespace(success=False, message='synthetic failure'))
    with pytest.raises(ValueError, match='utilization|Utilization'):
        invoke(args)


def test_exact_fit_count_has_no_refitted_old_baseline(fixture, monkeypatch):
    args, _ = fixture
    calls = []
    ridge, trees, optimizer = m.Ridge, m.ExtraTreesRegressor, m.least_squares

    def factory(cls, name):
        def construct(**kwargs):
            model = cls(**kwargs)
            original = model.fit

            def fit(x, y, sample_weight):
                calls.append((name, x.shape[1], y.ndim))
                return original(x, y, sample_weight=sample_weight)

            model.fit = fit
            return model
        return construct

    def solve(*a, **kw):
        calls.append(('least_squares', 3, 0))
        return optimizer(*a, **kw)

    monkeypatch.setattr(m, 'Ridge', factory(ridge, 'Ridge'))
    monkeypatch.setattr(m, 'ExtraTreesRegressor', factory(trees, 'ET'))
    monkeypatch.setattr(m, 'least_squares', solve)
    invoke(args)
    assert calls == [('least_squares', 3, 0), ('Ridge', 36, 2), ('ET', 36, 2),
                     ('Ridge', 35, 2), ('ET', 35, 2), ('Ridge', 36, 1), ('ET', 36, 1)]


def test_outer_capacity_changes_query_but_not_training_or_learned_beta(fixture):
    args, expected = fixture
    changed = deepcopy(args)
    for table in changed[2:]:
        for row in table:
            if row['region'] == 'MISO':
                row['capacity_coal_mw'] = str(float(row['capacity_coal_mw'])*2)
                reconcile(row)
    actual = invoke(changed)
    for left, right in zip(actual['audit']['fits'], expected['audit']['fits']):
        for key in ('training_X', 'training_y', 'sample_weights', 'scaler_mean', 'scaler_scale'):
            np.testing.assert_array_equal(left[key], right[key])
        for key in ('beta', 'coefficients', 'intercept', 'feature_importances'):
            if key in left:
                np.testing.assert_array_equal(left[key], right[key])
    assert not np.array_equal(actual['EF_draws'][2], expected['EF_draws'][2])


def test_direct_ef_scalar_clipping_and_proportion_simplex_projection(fixture, monkeypatch):
    args, _ = fixture

    class Fixed:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, x, y, sample_weight):
            self.scalar = y.ndim == 1
            self.coef_ = np.zeros(x.shape[1]) if self.scalar else np.zeros((4, x.shape[1]))
            self.intercept_ = 0. if self.scalar else np.zeros(4)
            self.feature_importances_ = np.zeros(x.shape[1])
            return self

        def predict(self, x):
            if self.scalar:
                return np.resize([.1, 1.4], len(x))
            return np.tile([-2., 1., 1., 3.], (len(x), 1))

    monkeypatch.setattr(m, 'Ridge', Fixed); monkeypatch.setattr(m, 'ExtraTreesRegressor', Fixed)
    out = invoke(args)
    np.testing.assert_array_equal(out['EF_draws'][9:].reshape(2, -1)[:, :2], [[400., 1000.], [400., 1000.]])
    np.testing.assert_allclose(out['projection_l1'][9:].reshape(2, -1)[:, :2], [[.3, .4], [.3, .4]])
    expected = v30.project(np.array([-2., 1., 1., 3.]))
    for arm in ('SizeCap-Ridge', 'SizeCap-ET', 'IntensiveCap-Ridge', 'IntensiveCap-ET'):
        idx = m.ARMS.index(arm)
        np.testing.assert_allclose(out['EF_draws'][idx], expected@v31.EF[v31.POS])
        np.testing.assert_allclose(out['projection_l1'][idx], np.abs(expected-[-2., 1., 1., 3.]).sum())


def test_determinism_row_order_and_input_immutability(fixture):
    args, expected = fixture
    changed = deepcopy(args); before = deepcopy(changed)
    changed[2].reverse(); changed[3].reverse()
    actual = invoke(changed)
    changed[2].reverse(); changed[3].reverse()
    same(changed, before)
    same(actual, expected)
