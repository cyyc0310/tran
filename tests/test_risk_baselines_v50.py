"""Synthetic standard risk heads, paired transport and external-only calibration."""
import copy

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from scripts import risk_baselines_v50 as model


def features(point, hour, level, shape, budget='B0'):
    n = len(point)
    base = np.zeros((n, 24, len(model.FEATURE_NAMES[budget])-9))
    extra = np.stack((point, hour, np.broadcast_to(point.mean(1)[:, None], point.shape),
        np.broadcast_to(point.min(1)[:, None], point.shape), np.broadcast_to(point.max(1)[:, None], point.shape),
        np.broadcast_to(level[:, None], point.shape), np.broadcast_to(shape[:, None], point.shape),
        np.broadcast_to(np.sin(2*np.pi*np.arange(24)/24), point.shape),
        np.broadcast_to(np.cos(2*np.pi*np.arange(24)/24), point.shape)), axis=-1)
    return np.concatenate((base, extra), axis=-1)


def training():
    sources = [s for s in model.SOURCES if s != 'BPAT']
    regions = np.repeat(sources, [1, 2, 1, 2, 1, 2])
    keys = [f'{s}/2024-02-{i+1:02d}' for s in sources for i in range(np.count_nonzero(regions == s))]
    n = len(keys)
    point = np.full((n, 24), 500.)
    e = np.arange(n)[:, None]+np.tile([-2., 2.], 12)
    hour, level, shape = np.full_like(point, 4.), np.zeros(n), np.full(n, 2.)
    return (features(point, hour, level, shape), point, point+e, hour, level, shape, regions, keys)


@pytest.fixture(scope='module')
def fitted():
    params = model.PARAMETERS | dict(max_iter=2, min_samples_leaf=2, max_leaf_nodes=3)
    with threadpool_limits(limits=1):
        return model._fit_baselines(*training(), outer='BPAT', budget='B0',
                                    feature_names=model.FEATURE_NAMES['B0'], parameters=params)


def test_registry_fixed_default_and_source_equal_weights(fitted):
    assert model.METHODS == ('GlobalABS', 'MCStd', 'Component', 'HETScale', 'HETLocationScale', 'CQRplus')
    assert len(model.FEATURE_NAMES['B0']) == 40 and len(model.FEATURE_NAMES['B1']) == 45
    assert model.PARAMETERS == dict(max_iter=128, max_leaf_nodes=15, min_samples_leaf=40,
        l2_regularization=1., learning_rate=.1, max_features=1., early_stopping=False, random_state=17023)
    assert type(model.PARAMETERS['max_features']) is float
    assert fitted.audit['fit_count'] == 11 and fitted.audit['synthetic_parameters_override'] is True
    assert fitted.audit['risk_mean_oof_is_base_teacher_third_source_exclusion'] is False
    r, w = np.asarray(fitted.audit['training_regions']), np.asarray(fitted.audit['sample_weight_days'])
    totals = [w[r == s].sum() for s in set(r)]
    np.testing.assert_allclose(totals, len(r)/6)
    assert w[r == 'CISO'][0] != w[r == 'ERCO'][0]
    assert len(fitted.audit['mean_oof']) == 6
    for fold in fitted.audit['mean_oof']:
        assert fold['held_source'] not in fold['training_sources']
        assert len(fold['training_sources']) == 5
    np.testing.assert_allclose(fitted.shape_standardized.mean(1), 0, atol=1e-12)
    e = training()[2]-training()[1]
    np.testing.assert_array_equal(fitted.audit['location_scale_log_target'],
                                  np.log1p(abs(e-fitted.audit['oof_bias_clipped'])))


def test_exact_paired_component_quantiles_and_batch_invariance():
    level = np.array([-1., 1.])
    shape = np.array([np.tile([-2., 2.], 12), np.tile([2., -2.], 12)])
    point = np.full((3, 24), 500.)
    args = (level, shape, np.array(['CISO', 'ERCO']), point, np.full(3, 2.), np.full(3, 3.))
    lo, hi = model.component_bounds(*args, batch_size=1)
    expected = point[:, None]+2*level[None, :, None]+3*shape[None]
    np.testing.assert_array_equal(lo, expected.min(1))
    np.testing.assert_array_equal(hi, expected.max(1))
    other = model.component_bounds(*args, batch_size=3)
    np.testing.assert_array_equal(other[0], lo)
    np.testing.assert_array_equal(other[1], hi)
    # Independent level/shape pairing would add unsupported extrema here.
    assert lo[0, 0] == 492 and hi[0, 0] == 508


def test_equal_source_calibration_not_pooled_hours():
    score = np.r_[np.zeros((100, 24)), np.full((1, 24), 100.)]
    regions = np.array(['SWPP']*100+['AVA'])
    assert model.calibrate(score, regions, 'equal_region')['q'] == 100
    assert np.quantile(score, .9) == 0


@pytest.mark.parametrize('k,expected_level', [(6, 1.05), (9, 1.), (10, .99)])
def test_hcp_infinity_and_finite_boundary(k, expected_level):
    regions = np.array(model.DONOR_SOURCES[:k])
    score = np.broadcast_to(np.arange(k)[:, None], (k, 24)).copy()
    result = model.calibrate(score, regions, 'HCP')
    assert result['finite_level'] == pytest.approx(expected_level)
    assert result['q'] == (np.inf if k == 6 else k-1)
    assert result['coverage_guarantee_claim'] is False


def test_scores_and_zero_infinity_final_bounds():
    point = np.full((2, 24), 500.)
    y = point+10
    np.testing.assert_array_equal(model.scores(point, point, np.full_like(point, 2.), y), 5)
    result = model.final_bounds(point-20, point+40, np.ones_like(point), 0.)
    np.testing.assert_array_equal(result['lo'], point-20)
    np.testing.assert_array_equal(result['hi'], point+40)
    result = model.final_bounds(point-20, point+40, np.ones_like(point), np.inf)
    np.testing.assert_array_equal(result['lo'], 0)
    np.testing.assert_array_equal(result['hi'], 1000)
    assert np.isfinite(result['lo']).all() and np.isfinite(result['hi']).all()


def test_prediction_cqr_crossing_and_scale_floor_without_refitting(fitted, monkeypatch):
    class Constant:
        def __init__(self, value): self.value = value
        def predict(self, x): return np.full(len(x), self.value)
    changed = copy.copy(fitted)
    changed.models = dict(fitted.models, q05=Constant(20), q95=Constant(-10), location=Constant(2000))
    monkeypatch.setattr(model.HistGradientBoostingRegressor, 'fit',
                        lambda *_args, **_kwargs: pytest.fail('Prediction/calibration refit a risk head'))
    point, hour, level, shape = np.full((2, 24), 500.), np.zeros((2, 24)), np.zeros(2), np.zeros(2)
    x = features(point, hour, level, shape)
    with threadpool_limits(limits=1):
        result = changed.predict(x, point, hour, level, shape,
                                 np.array(['SWPP', 'BPAT']), ['SWPP/2024-07-01', 'BPAT/2025-01-01'])
    assert result['lo0'].shape == result['hi0'].shape == result['scales'].shape == (6, 2, 24)
    assert np.all(result['scales'] >= 1)
    np.testing.assert_array_equal(result['lo0'][5], 490)
    np.testing.assert_array_equal(result['hi0'][5], 520)
    assert result['diagnostics']['cqr_crossing_count'] == 48
    np.testing.assert_array_equal(result['centers'][4], 1500)  # no clipping initial center/bounds
    np.testing.assert_array_equal(result['centers'][0], 500)
    assert result['audit']['new_fits'] == 0
    calibration = model.scores(result['lo0'][0, :1], result['hi0'][0, :1], result['scales'][0, :1], point[:1])
    assert model.calibrate(calibration, ['SWPP'], 'equal_region')['q'] == 0


@pytest.mark.parametrize('kind', ['outer', 'donor', 'closed', 'H2', 'keys', 'names', 'budget'])
def test_fit_metadata_guards_precede_training_numeric_access(kind):
    class Poison:
        def __array__(self, *args, **kwargs): raise AssertionError('Numeric array consumed before metadata rejection')
    values = list(training()); values[0] = Poison()
    names, budget = model.FEATURE_NAMES['B0'], 'B0'
    values[6] = values[6].copy(); values[7] = list(values[7])
    if kind in ('outer', 'donor', 'closed'):
        values[6][0] = {'outer':'BPAT', 'donor':'SWPP', 'closed':'PACE'}[kind]
    elif kind == 'H2': values[7][0] = 'CISO/2024-07-01'
    elif kind == 'keys': values[7][0] = 'ERCO/2024-02-01'
    elif kind == 'names': names = names[::-1]
    else: budget = 'B1'
    with pytest.raises(ValueError):
        model.fit_baselines(*values, outer='BPAT', budget=budget, feature_names=names)


@pytest.mark.parametrize('kind', ['complex', 'nan_truth', 'point_inf', 'shape', 'negative_std', 'extra'])
def test_training_numeric_guards_precede_any_fit(kind, monkeypatch):
    values = list(training())
    if kind == 'complex': values[0] = values[0].astype(complex)
    elif kind == 'nan_truth': values[2][0, 0] = np.nan
    elif kind == 'point_inf': values[1][0, 0] = np.inf
    elif kind == 'shape': values[0] = values[0][..., :-1]
    elif kind == 'negative_std': values[3][0, 0] = -1
    else: values[0][0, 0, -8] += 1
    monkeypatch.setattr(model.HistGradientBoostingRegressor, 'fit',
                        lambda *_args, **_kwargs: pytest.fail('Malformed numeric input reached fit'))
    with pytest.raises(ValueError):
        model.fit_baselines(*values, outer='BPAT', budget='B0', feature_names=model.FEATURE_NAMES['B0'])


@pytest.mark.parametrize('kind', ['proper_source', 'early_cal', 'late_cal', 'late_query', 'extra', 'complex', 'nan', 'negative_std', 'shape'])
def test_query_role_and_numeric_guards(fitted, kind):
    point, hour = np.full((1, 24), 500.), np.ones((1, 24))
    level, shape = np.ones(1), np.ones(1)
    region, key = 'SWPP', 'SWPP/2024-07-01'
    if kind == 'proper_source': region, key = 'CISO', 'CISO/2025-01-01'
    elif kind == 'early_cal': key = 'SWPP/2024-06-30'
    elif kind == 'late_cal': key = 'SWPP/2024-12-25'
    elif kind == 'late_query': region, key = 'BPAT', 'BPAT/2025-07-01'
    x = features(point, hour, level, shape)
    if kind == 'extra': x[0, 0, -9] += 1
    elif kind == 'complex': x = x.astype(complex)
    elif kind == 'nan': point[0, 0] = np.nan
    elif kind == 'negative_std': hour[0, 0] = -1
    elif kind == 'shape': x = x[:, :, :-1]
    with pytest.raises(ValueError):
        fitted.predict(x, point, hour, level, shape, [region], [key])


def test_calibration_rejects_target_and_invalid_scores():
    with pytest.raises(ValueError): model.calibrate(np.ones((1, 24)), ['BPAT'], 'HCP')
    with pytest.raises(ValueError): model.calibrate(np.full((1, 24), np.nan), ['SWPP'], 'HCP')
    with pytest.raises(ValueError): model.calibrate(np.ones((1, 24), complex), ['SWPP'], 'HCP')
    with pytest.raises(ValueError): model.calibrate(-np.ones((1, 24)), ['SWPP'], 'HCP')
    with pytest.raises(ValueError): model.final_bounds(np.ones((1, 24)), np.ones((1, 24)), np.zeros((1, 24)), np.inf)
