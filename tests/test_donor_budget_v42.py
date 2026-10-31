"""Synthetic-only metadata boundaries, source weights and ordinary heads."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from scripts import donor_budget_v42 as m
from scripts import coarse_context_v30 as old
from scripts import fleet_information_v34 as v34


def rows(months=6):
    result = []
    for i, region in enumerate(m.ALL_SOURCES):
        for j, start in enumerate(pd.date_range('2022-01-01', periods=months, freq='MS', tz='UTC')):
            end = start+pd.offsets.MonthBegin(1)
            result.append(dict(region=region, period_start=start.isoformat(), period_end=end.isoformat(),
                available_at=end.isoformat(), total_mwh=str((100+i+j)*((end-start)/pd.Timedelta(hours=1))),
                **{name: '.1' for name in old.SHARES}))
    return result


def same(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: same(a[key], b[key])
    elif isinstance(a, np.ndarray): np.testing.assert_array_equal(a, b)
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b): same(x, y)
    else: assert a == b


class CoarseOnly(dict):
    def __getitem__(self, key):
        if key in {old.SHARES[i] for i in old.POS}:
            raise AssertionError('Hidden fine field accessed')
        return super().__getitem__(key)


def test_legacy_cases_exact_and_expanded_has_same_legacy_rows():
    raw = rows(); before = deepcopy(raw)
    expected = old.build_cases(raw)
    same(m.build_cases(raw, sources=m.LEGACY_SOURCES), expected)
    expanded = m.build_cases(raw)
    assert len(expanded['keys']) == 17*3
    for region in m.LEGACY_SOURCES:
        a = expanded['regions'] == region; b = expected['regions'] == region
        np.testing.assert_array_equal(expanded['X_by_family']['Size'][a], expected['X_by_family']['Size'][b])
    same(raw, before)


def test_cases_never_read_hidden_or_current_total_and_unknown_metadata():
    raw = [CoarseOnly(r) for r in rows(4)]
    for r in raw:
        if r['period_start'].startswith('2022-04'): r['total_mwh'] = 'CURRENT_POISON'
    expected = m.build_cases(raw)
    assert len(expected['keys']) == 17
    class RegionOnly(dict):
        def __getitem__(self, key):
            if key != 'region': raise AssertionError('forbidden metadata access')
            return super().__getitem__(key)
    raw.extend(RegionOnly(region=r) for r in ('PACE', 'LGEE', 'UNKNOWN'))
    same(m.build_cases(raw), expected)
    assert expected['audit']['hidden_fields_read'] is expected['audit']['current_total_read'] is False


def test_release_guard_and_gapped_contexts_reuse_original_rules():
    raw = [r for r in rows(6) if not r['period_start'].startswith('2022-03')]
    result = m.build_cases(raw); i = result['keys'].index('SWPP/2022-05')
    assert result['context_keys'][i].tolist() == ['SWPP/2022-04', 'SWPP/2022-02', 'SWPP/2022-01']
    np.testing.assert_array_equal(result['context_month_age'][i], [1, 3, 4])
    for r in raw:
        if r['region'] == 'SWPP' and r['period_start'].startswith('2022-04'):
            r['available_at'] = '2022-05-01T00:00:00.000000001+00:00'
    assert 'SWPP/2022-05' not in m.build_cases(raw)['keys']


def test_future_numeric_fields_and_duplicate_metadata_are_guarded():
    raw = rows(4); expected = m.build_cases(raw)
    future = dict(raw[0], region='SWPP', period_start='2025-01-01T00:00:00Z',
        period_end='2025-02-01T00:00:00Z', available_at='2025-03-03T00:00:00Z', total_mwh='POISON')
    future.update({key: 'POISON' for key in old.SHARES})
    same(m.build_cases(raw+[CoarseOnly(future)]), expected)
    with pytest.raises(ValueError, match='Duplicate'): m.build_cases(raw+[raw[0].copy()])


@pytest.mark.parametrize('bad', [(), m.ALL_SOURCES[::-1], ('PACE',), list(m.LEGACY_SOURCES)])
def test_bad_whitelist_before_rows(bad):
    class NoRows:
        def __iter__(self): raise AssertionError('rows accessed')
    with pytest.raises(ValueError, match='sources'): m.build_cases(NoRows(), sources=bad)


def models(outer='BPAT'):
    rng = np.random.default_rng(42)
    sources = [s for s in m.LEGACY_SOURCES if s != outer]+list(m.DONOR_SOURCES[:2])
    regions = np.array([s for i, s in enumerate(sources) for _ in range(1+i%3)])
    n = len(regions); x = rng.normal(size=(n, 31)); x[:, 26] = 7.
    x[:, :5] = rng.dirichlet(np.ones(5), n)
    for lag in range(3): x[:, 7+5*lag:12+5*lag] = rng.dirichlet(np.ones(5), n)
    x[:, -1] = np.log(rng.uniform(100, 200, n))
    q = np.tile([.3, .5, 0., .2], (n, 1)); cpos = np.linspace(100., 250., n)
    y = rng.dirichlet(np.ones(4), n)  # Positive oil despite q_oil=0 must remain in the cohort.
    size = np.broadcast_to(x[0], (6, 128, 31)).copy()
    query = v34._features(size, np.tile(q[0], (6, 1)), np.full(6, 150.))
    return x, q, cpos, y, regions, {k: (size if k == 'Size' else query[k]) for k in m.FAMILIES}


@pytest.fixture(scope='module')
def fitted():
    args = models()
    return args, m.fit_heads(*args, outer='BPAT')


def test_fixed_weight_mass_actual_sources_and_reused_features(fitted):
    args, result = fitted; x, q, cpos, y, regions, query = args
    expected = np.array([154/(len(set(regions))*np.count_nonzero(regions == s)) for s in regions])
    np.testing.assert_array_equal(result['sample_weights'], expected)
    assert expected.sum() == pytest.approx(154.)
    for s in set(regions): assert expected[regions == s].sum() == pytest.approx(154/len(set(regions)))
    np.testing.assert_array_equal(result['training_features']['Size'], x)
    for name in ('SizeCap', 'DirectEF'):
        np.testing.assert_array_equal(result['training_features'][name], v34._features(x, q, cpos)[name])
    assert result['audit']['fit_count'] == 6
    assert result['EF_draws'].shape == result['EF_raw'].shape == (6, 6, 128)
    assert result['ratios'].shape == result['ratios_raw'].shape == (4, 6, 128, 4)
    np.testing.assert_allclose(result['ratios'].sum(-1), 1.)
    assert (result['EF_draws'] >= 400.-1e-10).all() and (result['EF_draws'] <= 1000.+1e-10).all()


def test_ridge_replay_scaler_labels_and_et_configuration(fitted):
    args, result = fitted
    for fit in result['audit']['fits']:
        assert fit['parameters'] == v34.PARAMETERS[fit['model']]
        x = result['training_features'][fit['family']]; w = result['sample_weights']
        target = args[3]@m.EF_POS/1000 if fit['family'] == 'DirectEF' else args[3]
        np.testing.assert_array_equal(fit['training_y'], target)
        if fit['model'] == 'Ridge':
            mean = np.average(x, axis=0, weights=w)
            scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=w)); scale[scale <= 1e-12] = 1
            np.testing.assert_array_equal(fit['scaler_mean'], mean)
            np.testing.assert_array_equal(fit['scaler_scale'], scale)
            z = (args[5][fit['family']]-mean)/scale
            replay = z@np.asarray(fit['coefficients']).T+fit['intercept']
            np.testing.assert_allclose(replay, fit['raw_query_prediction'], atol=1e-14)
        else:
            assert type(fit['parameters']['max_features']) is float
            assert fit['parameters']['max_features'] == 1.0
            np.testing.assert_array_equal(fit['scaler_mean'], 0)
            np.testing.assert_array_equal(fit['scaler_scale'], 1)


def test_fpl_weight_mass_is_156_and_inputs_immutable():
    args = models('FPL'); before = deepcopy(args)
    result = m.fit_heads(*args, outer='FPL')
    assert result['sample_weights'].sum() == pytest.approx(156.)
    same(args, before)


@pytest.mark.parametrize('bad', ['outer', 'unknown', 'PACE', 'LGEE'])
def test_region_guard_precedes_any_numbers(bad):
    args = list(models()); args[4][0] = 'BPAT' if bad == 'outer' else bad
    class NoNumbers:
        def __array__(self, *a, **k): raise AssertionError('numeric data accessed before source guard')
    args[0] = args[3] = NoNumbers()
    with pytest.raises(ValueError, match='source|outer'): m.fit_heads(*args, outer='BPAT')


@pytest.mark.parametrize('index,kind', [(0, 'shape'), (1, 'sum'), (1, 'negative'),
    (2, 'zero'), (2, 'nan'), (3, 'sum'), (3, 'complex'), (5, 'mismatch')])
def test_invalid_model_inputs_fail_without_fitting(index, kind, monkeypatch):
    args = list(models())
    if kind == 'mismatch': args[5]['SizeCap'][0, 0, 0] += .01
    elif kind == 'shape': args[index] = args[index][:, :-1]
    else:
        if kind == 'complex': args[index] = args[index].astype(complex)
        args[index].flat[0] = {'sum': .99, 'negative': -.1, 'zero': 0, 'nan': np.nan, 'complex': 1j}[kind]
    monkeypatch.setattr(m, 'Ridge', lambda **kwargs: pytest.fail('invalid inputs must fail before fit'))
    with pytest.raises(ValueError): m.fit_heads(*args, outer='BPAT')


def test_scalar_clips_ratio_simplex_and_raw_ef_are_distinguished(monkeypatch):
    class FakeModel:
        def __init__(self, **kwargs): pass
        def fit(self, x, y, sample_weight):
            self.scalar = y.ndim == 1
            self.coef_ = np.zeros(x.shape[1] if self.scalar else (4, x.shape[1]))
            self.intercept_ = np.array(.4) if self.scalar else np.zeros(4)
            self.feature_importances_ = np.full(x.shape[1], 1/x.shape[1])
            return self
        def predict(self, x):
            return np.linspace(-.2, 1.2, len(x)) if self.scalar else np.tile([-.5, 1.4, .4, -.3], (len(x), 1))
    monkeypatch.setattr(m, 'Ridge', FakeModel)
    monkeypatch.setattr(m, 'ExtraTreesRegressor', FakeModel)
    result = m.fit_heads(*models(), outer='BPAT')
    for i in range(4):
        expected = old.project(result['ratios_raw'][i])
        np.testing.assert_array_equal(result['ratios'][i], expected)
        np.testing.assert_array_equal(result['projection_l1'][i], np.abs(expected-result['ratios_raw'][i]).sum(-1))
        np.testing.assert_array_equal(result['EF_raw'][i], result['ratios_raw'][i]@m.EF_POS)
    for i in (4, 5):
        assert result['EF_raw'][i, 0, 0] == -200.
        assert result['EF_raw'][i, -1, -1] == 1200.
        np.testing.assert_array_equal(result['EF_draws'][i], np.clip(result['EF_raw'][i]/1000, .4, 1)*1000)
        assert result['projection_kind'][i] == 'scalar_EF'
        np.testing.assert_allclose(result['projection_l1'][i, 0, 0], .6)
