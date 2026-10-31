"""Closed-form V35 diagnostics and affine Ridge replay; zero model fitting."""
from copy import deepcopy
import json

import numpy as np
import pytest

from scripts import conditional_diagnostics_v35 as m


A = np.array([0., 1., 0., 0., 0.])
B = np.array([1., 0., 0., 0., 0.])


def test_deterministic_correct_forecast_zero_spread_ties_and_scores():
    draws = np.tile(A, (128, 1))
    result = m.forecast_case(draws, A, A)
    for key in ('pos_mean', 'pos_median', 'pos_std', 'pos_mean_bias', 'pos_mean_abs_error',
                'pos_median_bias', 'pos_median_abs_error', 'pos_crps', 'g5_energy',
                'past_pos_abs_error', 'past_g5_euclidean_error'):
        assert result[key] == 0
    for key in ('pos_q05', 'pos_q10', 'pos_q50', 'pos_q90', 'pos_q95', 'pos_min', 'pos_max'):
        assert result[key] == 0
    assert result['pos_less_count'] == 0 and result['pos_equal_count'] == 128
    assert result['pos_mid_cdf'] == .5
    assert result['pos_coverage80'] and result['pos_coverage90'] and result['pos_minmax_contains']
    json.dumps(result, allow_nan=False)


def test_deterministic_wrong_forecast_closed_form_scores_and_past_baseline():
    forecast = np.array([.2, .8, 0, 0, 0])
    truth = np.array([.7, .3, 0, 0, 0])
    past = np.array([.1, .9, 0, 0, 0])
    result = m.forecast_case(np.tile(forecast, (128, 1)), truth, past)
    assert result['pos_std'] == pytest.approx(0, abs=1e-15)
    assert result['pos_mean_bias'] == pytest.approx(-.5)
    assert result['pos_mean_abs_error'] == pytest.approx(.5)
    assert result['pos_crps'] == pytest.approx(.5)
    assert result['g5_energy'] == pytest.approx(np.sqrt(.5))
    assert result['past_pos_abs_error'] == pytest.approx(.6)
    assert result['past_g5_euclidean_error'] == pytest.approx(np.sqrt(.72))
    assert result['pos_less_count'] == 128 and result['pos_equal_count'] == 0
    assert result['pos_mid_cdf'] == 1
    assert not result['pos_coverage80'] and not result['pos_coverage90'] and not result['pos_minmax_contains']


@pytest.mark.parametrize('truth,mid_cdf,equal,less', [(A, .25, 64, 0), (B, .75, 64, 64), ((A+B)/2, .5, 0, 64)])
def test_two_atom_empirical_crps_energy_linear_quantiles_and_exact_ties(truth, mid_cdf, equal, less):
    draws = np.vstack([np.tile(A, (64, 1)), np.tile(B, (64, 1))])
    result = m.forecast_case(draws, truth, A)
    assert result['pos_mean'] == result['pos_median'] == result['pos_std'] == .5
    assert [result[k] for k in ('pos_q05', 'pos_q10', 'pos_q50', 'pos_q90', 'pos_q95')] == [0, 0, .5, 1, 1]
    assert result['pos_crps'] == pytest.approx(.25)
    assert result['g5_energy'] == pytest.approx(np.sqrt(2)/4)
    assert result['pos_mid_cdf'] == mid_cdf
    assert result['pos_equal_count'] == equal and result['pos_less_count'] == less
    assert result['pos_coverage80'] and result['pos_coverage90']


def test_quantiles_match_linear_interpolation_and_coverage_is_inclusive():
    pos = np.linspace(0, 1, 128)
    draws = np.c_[pos, 1-pos, np.zeros((128, 3))]
    truth = np.array([.1, .9, 0, 0, 0])
    result = m.forecast_case(draws, truth, truth)
    for level, name in ((.05, '05'), (.1, '10'), (.5, '50'), (.9, '90'), (.95, '95')):
        assert result['pos_q'+name] == np.quantile(pos, level, method='linear')
    assert result['pos_coverage80'] and result['pos_coverage90']
    outside = m.forecast_case(draws, A, truth)
    assert not outside['pos_coverage80'] and not outside['pos_coverage90']
    assert outside['pos_minmax_contains']


@pytest.mark.parametrize('where,bad', [('draws', 'nan'), ('draws', 'inf'), ('draws', 'negative'),
                                     ('draws', 'sum'), ('draws', 'shape'), ('truth', 'nan'),
                                     ('truth', 'shape'), ('past', 'negative'), ('past', 'sum')])
def test_forecast_rejects_bad_shape_nonfinite_and_invalid_simplex(where, bad):
    values = {'draws': np.tile((A+B)/2, (128, 1)), 'truth': (A+B)/2, 'past': (A+B)/2}
    value = values[where].copy()
    if bad == 'shape':
        value = value[:-1]
    elif bad == 'sum':
        value.flat[0] += .1
    else:
        value.flat[0] = {'nan': np.nan, 'inf': np.inf, 'negative': -.1}[bad]
    values[where] = value
    with pytest.raises(ValueError):
        m.forecast_case(values['draws'], values['truth'], values['past'])


def test_exact_ties_do_not_use_tolerance_rank_smoothing():
    draws = np.tile((A+B)/2, (128, 1))
    truth = np.array([np.nextafter(.5, 1), 1-np.nextafter(.5, 1), 0, 0, 0])
    result = m.forecast_case(draws, truth, A)
    assert result['pos_equal_count'] == 0 and result['pos_less_count'] == 128
    assert result['pos_mid_cdf'] == 1


def test_point_case_mean_median_signed_absolute_and_squared_errors():
    cif = np.r_[np.zeros(127), 128.]
    ef = cif+400
    result = m.point_case(cif, ef, 0., 400.)
    assert set(result) == {'mean', 'median'}
    assert result['mean'] == dict(cif_point=1., ef_point=401., cif_bias=1., cif_abs_error=1., cif_sq_error=1.,
                                  ef_bias=1., ef_abs_error=1., ef_sq_error=1.)
    assert result['median'] == dict(cif_point=0., ef_point=400., cif_bias=0., cif_abs_error=0., cif_sq_error=0.,
                                    ef_bias=0., ef_abs_error=0., ef_sq_error=0.)
    json.dumps(result, allow_nan=False)


def test_empirical_mean_and_median_have_different_loss_targets():
    draws = np.r_[np.zeros(127), 128.]
    zero = m.point_case(draws, draws, 0., 0.)
    high = m.point_case(draws, draws, 128., 128.)
    risk = {mode: {loss: (127*zero[mode][loss]+high[mode][loss])/128
                   for loss in ('cif_abs_error', 'cif_sq_error')} for mode in ('mean', 'median')}
    assert risk['mean']['cif_sq_error'] == 127 < risk['median']['cif_sq_error'] == 128
    assert risk['median']['cif_abs_error'] == 1 < risk['mean']['cif_abs_error']


@pytest.mark.parametrize('which,bad', [('cif', np.full(128, np.nan)), ('ef', np.full(128, np.inf)),
                                    ('cif', np.ones(127)), ('ef', np.ones((128, 1))),
                                    ('truth_cif', np.nan), ('truth_ef', [400.])])
def test_point_case_rejects_wrong_shapes_and_nonfinite(which, bad):
    args = dict(cif=np.ones(128), ef=np.ones(128)*400, truth_cif=1., truth_ef=400.)
    args[which] = bad
    with pytest.raises(ValueError):
        m.point_case(args['cif'], args['ef'], args['truth_cif'], args['truth_ef'])


def fit_record(arm):
    family = arm.removesuffix('-Ridge')
    d = {'Size': 31, 'SizeCap': 36, 'IntensiveCap': 35, 'DirectEF': 36}[family]
    mean, scale = np.zeros(d), np.ones(d)
    mean[0], scale[0] = 2., 2.
    if family == 'DirectEF':
        coef = np.zeros(d); coef[0] = 1
        intercept = .2
    else:
        coef = np.zeros((4, d)); coef[0, 0] = 1
        intercept = [0., .4, .3, .3]
    return dict(arm=arm, family=family, model='Ridge', coefficients=coef, intercept=intercept,
                scaler_mean=mean, scaler_scale=scale)


@pytest.mark.parametrize('arm', ['Size-Ridge', 'SizeCap-Ridge', 'IntensiveCap-Ridge'])
def test_reconstructs_frozen_ratio_ridge_without_fitting_and_projects(arm):
    fit = fit_record(arm)
    x = np.zeros((3, len(fit['scaler_mean']))); x[:, 0] = [2, 4, -2]
    np.testing.assert_allclose(m.ridge_ef(fit, x), [520., 870., 520.], atol=1e-12, rtol=0)
    np.testing.assert_allclose(m.ridge_ef(fit, x.reshape(1, 3, -1)), [[520., 870., 520.]])


def test_reconstructs_frozen_scalar_ridge_and_clips_normalized_ef():
    fit = fit_record('DirectEF-Ridge')
    x = np.zeros((3, 36)); x[:, 0] = [2, 4, 3]
    np.testing.assert_allclose(m.ridge_ef(fit, x), [400., 1000., 700.], atol=1e-12, rtol=0)


@pytest.mark.parametrize('key,value', [('arm', 'Size-ET'), ('family', 'DirectEF'), ('model', 'ET'),
                                    ('coefficients', np.ones((4, 30))), ('coefficients', np.full((4, 31), np.nan)),
                                    ('intercept', [1., 2.]), ('scaler_scale', np.zeros(31)),
                                    ('scaler_mean', np.full(31, np.inf))])
def test_replay_rejects_identity_shape_scaler_and_finite_errors(key, value):
    fit = fit_record('Size-Ridge'); fit[key] = value
    with pytest.raises(ValueError):
        m.ridge_ef(fit, np.zeros((128, 31)))


@pytest.mark.parametrize('features', [np.zeros((128, 30)), np.full((128, 31), np.nan), np.zeros((0, 31))])
def test_replay_rejects_bad_features(features):
    with pytest.raises(ValueError):
        m.ridge_ef(fit_record('Size-Ridge'), features)


def test_zero_fit_guards_and_input_immutability(monkeypatch):
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import ExtraTreesRegressor

    def forbidden(*args, **kwargs):
        raise AssertionError('V35 must not fit or solve new coefficients')

    monkeypatch.setattr(Ridge, 'fit', forbidden)
    monkeypatch.setattr(ExtraTreesRegressor, 'fit', forbidden)
    monkeypatch.setattr(np.linalg, 'solve', forbidden)
    monkeypatch.setattr(np.linalg, 'lstsq', forbidden)
    fit = fit_record('Size-Ridge'); x = np.zeros((128, 31)); draws = np.tile((A+B)/2, (128, 1))
    original = deepcopy((fit, x, draws))
    m.ridge_ef(fit, x); m.forecast_case(draws, A, B); m.point_case(np.ones(128), np.ones(128)*400, 1., 400.)
    np.testing.assert_array_equal(x, original[1]); np.testing.assert_array_equal(draws, original[2])
    for key in fit:
        if isinstance(fit[key], np.ndarray):
            np.testing.assert_array_equal(fit[key], original[0][key])
        else:
            assert fit[key] == original[0][key]
