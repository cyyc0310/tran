"""Synthetic coarse-only features and six-source SUM-ridge calibration."""
import numpy as np
import pytest

from scripts import coarse_residual_v40 as m


def data():
    x = np.array([[-.3, .1], [-.2, .2], [-.1, .4], [.1, .5], [.2, .7], [.3, .8]])
    bias = 1000*(.03+.2*x[:, 0]-.1*x[:, 1])
    query = np.broadcast_to(x[None], (6, 6, 2)).copy()
    base = np.full((6, 6), 600.)
    return x, bias, query, base


def test_features_use_coarse_pos_and_capacity_normalized_contributions():
    c = np.array([[20., 40., 0., 40.], [10., 20., 10., 10.]])
    u = np.array([[.5, .25, 3., .1], [2., .2, .4, .3]])
    pos = np.array([40., 0.])
    result = m.features(c, pos, u)
    q = c/c.sum(1)[:, None]; xhat = q*u; z = pos/c.sum(1)
    ef = (xhat/xhat.sum(1)[:, None])@np.array([1000, 400, 800, 400])
    np.testing.assert_allclose(result['X'], [(z-xhat.sum(1)).mean(), ((ef-400)/600).mean()])
    for name, expected in [('q', q), ('xhat', xhat), ('z', z), ('d', z-xhat.sum(1)), ('EFpred', ef)]:
        np.testing.assert_allclose(result['audit'][name], expected)
    np.testing.assert_array_equal(result['audit']['xhat'][0, 2], 0)
    scaled = m.features(c*100, pos*100, u)
    np.testing.assert_allclose(result['X'], scaled['X'])
    assert result['audit']['xhat'][1, 0] > q[1, 0]  # No utilization clipping.


def test_five_arms_exact_none_and_sum_ridge_normal_equations():
    args = data(); result = m.fit_predict(*args)
    assert m.ARMS == ('EnsembleNone', 'ResidualMean', 'ResidualD', 'ResidualS', 'ResidualDS')
    assert result['raw_component_EF'].shape == result['component_EF'].shape == (5, 6, 6)
    assert result['clipmask'].dtype == bool and result['ensemble_EF'].shape == (5, 6)
    np.testing.assert_array_equal(result['raw_component_EF'][0], args[3])
    np.testing.assert_array_equal(result['component_EF'][0], args[3])
    np.testing.assert_array_equal(result['ensemble_EF'], result['component_EF'].mean(2))
    assert len(result['audit']['fits']) == 4
    for fit in result['audit']['fits']:
        ids = fit['selected_columns']; values = args[0][:, ids]
        mean = values.mean(0); scale = values.std(0)
        np.testing.assert_array_equal(fit['scaler_mean'], mean)
        np.testing.assert_array_equal(fit['scaler_scale'], scale)
        z = np.c_[np.ones(6), (values-mean)/scale]
        y = args[1]/1000
        penalty = np.diag(np.r_[0., np.ones(len(ids))])
        expected = np.linalg.solve(z.T@z+penalty, z.T@y)
        np.testing.assert_allclose(fit['theta'], expected, atol=1e-15)
        np.testing.assert_array_equal(fit['Z'], z); np.testing.assert_array_equal(fit['y'], y)
        np.testing.assert_allclose(fit['normal_equation_residual'], 0, atol=1e-14)
        query = args[2][..., ids]
        design = np.concatenate([np.ones((6, 6, 1)), (query-mean)/scale], axis=2)
        index = m.ARMS.index(fit['arm'])
        np.testing.assert_allclose(result['raw_component_EF'][index], args[3]+1000*(design@expected))


def test_intercept_not_penalized_and_true_minus_pred_bias_is_added():
    x, _, query, base = data(); bias = np.full(6, 50.)
    result = m.fit_predict(x, bias, query, base)
    for fit in result['audit']['fits']:
        assert fit['theta'][0] == pytest.approx(.05)
        np.testing.assert_allclose(fit['theta'][1:], 0, atol=1e-15)
    np.testing.assert_allclose(result['raw_component_EF'][1:], 650.)


def test_strict_constant_columns_use_scale_one_and_zero_train_and_query():
    x, bias, query, base = data(); x[:, 0] = .2; query[..., 0] = 1e100
    result = m.fit_predict(x, bias, query, base)
    d = result['audit']['fits'][1]
    np.testing.assert_array_equal(d['constant_mask'], [True])
    np.testing.assert_array_equal(d['scaler_scale'], [1])
    np.testing.assert_array_equal(d['Z'][:, 1], 0)
    np.testing.assert_array_equal(result['raw_component_EF'][1], result['raw_component_EF'][2])
    # A genuinely varying column is retained even at a very small absolute scale.
    x, bias, query, base = data(); x[:, 0] *= 1e-15; query[..., 0] *= 1e-15
    small = m.fit_predict(x, bias, query, base)['audit']['fits'][1]
    assert not small['constant_mask'][0] and 0 < small['scaler_scale'][0] < 1e-12
    np.testing.assert_allclose(small['Z'][:, 1].std(), 1.)


def test_constant_query_columns_are_zeroed_before_unused_arithmetic():
    x, bias, query, base = data(); x[:, 0] = 1e308; query[..., 0] = -1e308
    result = m.fit_predict(x, bias, query, base)
    np.testing.assert_array_equal(result['raw_component_EF'][1], result['raw_component_EF'][2])
    assert np.isfinite(result['ensemble_EF']).all()


def test_clip_components_before_ensemble_and_preserve_raw_values():
    x, bias, query, base = data()
    bias[:] = 100.; base[:] = [400., 400., 400., 1000., 1000., 1000.]
    result = m.fit_predict(x, bias, query, base)
    np.testing.assert_allclose(result['raw_component_EF'][1, 0], [500, 500, 500, 1100, 1100, 1100])
    np.testing.assert_allclose(result['component_EF'][1, 0], [500, 500, 500, 1000, 1000, 1000])
    assert result['ensemble_EF'][1, 0] == pytest.approx(750.)
    assert np.clip(result['raw_component_EF'][1, 0].mean(), 400, 1000) == pytest.approx(800.)
    np.testing.assert_array_equal(result['clipmask'][1, 0], [False, False, False, True, True, True])
    bias[:] = -100.
    result = m.fit_predict(x, bias, query, base)
    np.testing.assert_allclose(result['component_EF'][1, 0], [400, 400, 400, 900, 900, 900])


def test_input_immutability_and_source_order_invariance():
    args = data(); copies = [a.copy() for a in args]
    original = m.fit_predict(*args)
    permuted = m.fit_predict(args[0][::-1], args[1][::-1], args[2], args[3])
    np.testing.assert_allclose(original['ensemble_EF'], permuted['ensemble_EF'], atol=1e-12)
    for a, b in zip(args, copies): np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('which,kind', [(0, 'complex'), (0, 'shape'), (1, 'nan'), (1, 'shape'),
    (2, 'inf'), (2, 'shape'), (3, 'nan'), (3, 'complex'), (3, 'shape')])
def test_calibration_invalid_inputs(which, kind):
    args = list(data())
    if kind == 'shape': args[which] = args[which][:-1]
    else:
        if kind == 'complex': args[which] = args[which].astype(complex)
        args[which].flat[0] = {'complex': 1j, 'nan': np.nan, 'inf': np.inf,
                              'negative': -1, 'above': 1001}[kind]
    with pytest.raises(ValueError): m.fit_predict(*args)


def test_none_raw_baseline_is_literal_copy_and_all_arms_share_clipping():
    x, bias, query, base = data(); base[0, :2] = [-10., 1000.+1e-10]
    result = m.fit_predict(x, bias, query, base)
    np.testing.assert_array_equal(result['raw_component_EF'][0], base)
    np.testing.assert_array_equal(result['component_EF'], np.clip(result['raw_component_EF'], 400, 1000))
    np.testing.assert_array_equal(result['clipmask'], result['raw_component_EF'] != result['component_EF'])


@pytest.mark.parametrize('which,kind', [(0, 'negative'), (0, 'zero'), (0, 'complex'),
    (0, 'shape'), (1, 'negative'), (1, 'nan'), (1, 'shape'), (2, 'negative'),
    (2, 'zero'), (2, 'shape')])
def test_feature_invalid_domains(which, kind):
    args = [np.ones((12, 4)), np.ones(12), np.ones((12, 4))]
    if kind == 'shape': args[which] = args[which][:-1]
    elif kind == 'zero': args[which][0] = 0
    else:
        if kind == 'complex': args[which] = args[which].astype(complex)
        args[which].flat[0] = {'negative': -1, 'complex': 1j, 'nan': np.nan}[kind]
    with pytest.raises(ValueError): m.features(*args)
