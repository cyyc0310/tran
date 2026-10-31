"""Synthetic inventory-profile and constrained inverse tests only."""
import numpy as np
import pytest

from scripts import coarse_inverse_v38 as m


def profiles():
    c = np.array([[10., 20., 0., 5.], [10., 0., 4., 5.], [20., 20., 4., 5.]])
    u = np.array([[.2, .4, 0., 2.], [.4, 0., .3, 1.], [.5, .2, .9, 0.]])
    return c*u, c, np.array([1, 1, 2]), np.array([1., 2., 3.]), ['a', 'b', 'c']


def test_weighted_valid_profiles_and_unit_weight_month_shrinkage():
    args = profiles(); result = m.fit_profiles(*args)
    g, c, months, w, keys = args
    expected = np.array([(.2+2*.4+3*.5)/6, (.4+3*.2)/4, (2*.3+3*.9)/5, (2.+2.)/6])
    np.testing.assert_allclose(result['global'], expected)
    assert result['valid'].dtype == bool
    np.testing.assert_array_equal(result['valid'], c > 0)
    np.testing.assert_allclose(result['month'][0], [(1.+expected[0])/4,
        (.4+expected[1])/2, (.6+expected[2])/3, (4.+expected[3])/4])
    np.testing.assert_array_equal(result['month'][2:], np.broadcast_to(result['global'], (10, 4)))
    assert result['audit']['zero_capacity'] == [dict(key='a', fuel='oil'), dict(key='b', fuel='gas')]
    np.testing.assert_array_equal(result['audit']['valid_count'], [3, 2, 2, 3])
    np.testing.assert_array_equal(result['audit']['valid_weight'], [6, 4, 5, 6])
    assert result['global'][3] > 0  # The first observed source rate is 2, without clipping.
    one = m.fit_profiles(np.full((1, 4), 2.), np.ones((1, 4)), [1], [1], ['x'])
    np.testing.assert_array_equal(one['global'], 2.)
    np.testing.assert_array_equal(m.profile_predict(result['global'], [12, 1]), np.tile(result['global'], (2, 1)))
    np.testing.assert_array_equal(m.profile_predict(result['month'], [2, 1]), result['month'][[1, 0]])


def test_input_order_and_immutability():
    args = profiles(); copies = [x.copy() for x in args]
    a = m.fit_profiles(*args)
    b = m.fit_profiles(*(x[::-1] for x in args))
    np.testing.assert_allclose(a['global'], b['global'])
    np.testing.assert_allclose(a['month'], b['month'])
    for x, copy in zip(args, copies): np.testing.assert_array_equal(x, copy)


def test_zero_capacity_positive_generation_reports_key_and_fuel():
    args = list(profiles()); args[0][0, 2] = 1
    with pytest.raises(ValueError, match='a.*oil'):
        m.fit_profiles(*args)
    args = list(profiles()); args[1][:, 0] = 0; args[0][:, 0] = 0
    with pytest.raises(ValueError, match='coal'):
        m.fit_profiles(*args)


def test_common_inverse_recovers_scale_and_preserves_carbon_composition():
    rng = np.random.default_rng(38); b = rng.uniform(.1, .4, (12, 4))
    z = 2.3*b.sum(1)
    fit = m.adapt(b, z, 'Common')
    np.testing.assert_allclose(fit['gamma'], 2.3, atol=1e-14)
    assert fit['objective'] < 1e-28 and fit['KKT']['max_violation'] <= 1e-8
    query = rng.uniform(.1, 3., (6, 4))
    base = m.decode(query, np.ones(4)); changed = m.decode(query, fit['gamma'])
    np.testing.assert_allclose(base['EF'], changed['EF'], atol=1e-12)
    np.testing.assert_allclose(base['ratios'], changed['ratios'], atol=1e-15)
    np.testing.assert_allclose(changed['mass'], 2.3*base['mass'])


def test_fuel_ridge_matches_unconstrained_closed_form_when_interior():
    rng = np.random.default_rng(3); b = rng.uniform(.1, .4, (12, 4)); target = [1.2, .7, 1.1, .8]
    z = b@target
    expected = np.linalg.solve(b.T@b/12+.01*np.eye(4), b.T@z/12+.01*np.ones(4))
    fit = m.adapt(b, z, 'FuelRidge')
    assert (expected > 0).all()
    np.testing.assert_allclose(fit['gamma'], expected, atol=1e-13)
    residual = b@expected-z
    assert fit['objective'] == pytest.approx(np.mean(residual**2)+.01*np.sum((expected-1)**2))
    np.testing.assert_allclose(fit['gradient'], 2*b.T@residual/12+.02*(expected-1), atol=1e-15)
    assert fit['KKT']['max_violation'] <= 1e-8


def test_ridge_unique_rank_deficient_solution_and_active_boundary_kkt():
    zero = np.zeros((12, 4)); z = np.ones(12)
    fit = m.adapt(zero, z, 'FuelRidge')
    np.testing.assert_allclose(fit['gamma'], 1, atol=1e-14)
    assert fit['objective'] == pytest.approx(1.)
    b = np.tile([1., 2., 3., 4.], (12, 1))
    fit = m.adapt(b, np.zeros(12), 'FuelRidge')
    assert (fit['gamma'] == 0).any()
    positive = fit['gamma'] > 0
    assert np.max(np.abs(fit['gradient'][positive])) <= 1e-8
    assert (fit['gradient'][~positive] >= -1e-8).all()
    assert np.linalg.eigvalsh(2*b.T@b/12+.02*np.eye(4)).min() > 0


def test_none_common_zero_and_decode_zero_mass_fail_explicitly():
    b = np.ones((12, 4)); z = np.zeros(12)
    none = m.adapt(b, z, 'None')
    np.testing.assert_array_equal(none['gamma'], 1)
    assert none['objective'] == 16 and not none['KKT']['applicable']
    common = m.adapt(b, z, 'Common')
    np.testing.assert_array_equal(common['gamma'], 0)
    with pytest.raises(ValueError, match='mass'):
        m.decode(b, common['gamma'])
    with pytest.raises(ValueError, match='mass'):
        m.decode(np.zeros(4), np.ones(4))


def test_common_all_zero_operator_is_unidentifiable_before_nnls(monkeypatch):
    monkeypatch.setattr(m, 'nnls', lambda *a, **k: pytest.fail('unidentifiable Common must not fit'))
    for z in (np.zeros(12), np.ones(12)):
        with pytest.raises(ValueError, match='unidentifiable'):
            m.adapt(np.zeros((12, 4)), z, 'Common')


def test_decoder_keeps_zero_support_and_known_ef():
    result = m.decode(np.array([[2., 0., 1., 0.], [0., 1., 0., 1.]]), np.ones(4))
    np.testing.assert_allclose(result['EF'], [2800/3, 400])
    np.testing.assert_array_equal(result['ratios'] == 0, [[False, True, False, True], [True, False, True, False]])


def test_operator_rank_threshold_and_singular_condition():
    b = np.vstack([np.diag([4., 3., 2., 1.]), np.zeros((8, 4))])
    d = m.operator_diagnostics(b)
    np.testing.assert_array_equal(d['singular_values'], [4, 3, 2, 1])
    assert d['rank'] == 4 and d['full_rank'] and d['condition_number'] == 4
    b[:, 3] = 0
    d = m.operator_diagnostics(b)
    assert d['rank'] == 3 and not d['full_rank'] and d['condition_number'] is None
    b[3, 3] = 1e-11
    assert m.operator_diagnostics(b)['rank'] == 3


@pytest.mark.parametrize('index,kind', [(0, 'complex'), (0, 'negative'), (0, 'nan'),
    (1, 'negative'), (1, 'inf'), (1, 'shape'), (2, 'month0'), (2, 'fraction'),
    (3, 'zero'), (3, 'nan'), (4, 'duplicate')])
def test_profile_invalid_inputs(index, kind):
    args = list(profiles())
    if kind == 'duplicate': args[4][1] = args[4][0]
    elif kind == 'shape': args[index] = args[index][:, :3]
    else:
        if kind == 'complex': args[index] = args[index].astype(complex)
        elif kind in ('month0', 'fraction'): args[index] = args[index].astype(float)
        args[index].flat[0] = {'complex': 1+1j, 'negative': -1, 'nan': np.nan, 'inf': np.inf,
                              'month0': 0, 'fraction': 1.5, 'zero': 0}[kind]
    with pytest.raises(ValueError): m.fit_profiles(*args)


@pytest.mark.parametrize('kind', ['negative', 'complex', 'nan', 'shape', 'kind'])
def test_inverse_invalid_inputs(kind):
    b = np.ones((12, 4)); z = np.ones(12); name = 'FuelRidge'
    if kind == 'negative': b[0, 0] = -1
    elif kind == 'complex': z = z.astype(complex); z[0] += 1j
    elif kind == 'nan': z[0] = np.nan
    elif kind == 'shape': b = b[:11]
    else: name = 'other'
    with pytest.raises(ValueError): m.adapt(b, z, name)


def test_nnls_configuration_and_bad_solution_fail_closed(monkeypatch):
    real = m.nnls; calls = []
    def wrapped(a, b, **kwargs):
        calls.append((a.copy(), b.copy(), kwargs)); return real(a, b, **kwargs)
    monkeypatch.setattr(m, 'nnls', wrapped)
    b = np.ones((12, 4)); z = np.ones(12)
    m.adapt(b, z, 'FuelRidge')
    np.testing.assert_allclose(calls[0][0], np.vstack([b/np.sqrt(12), .1*np.eye(4)]))
    np.testing.assert_allclose(calls[0][1], np.r_[z/np.sqrt(12), np.full(4, .1)])
    assert calls[0][2] == {'maxiter': 10000}
    monkeypatch.setattr(m, 'nnls', lambda *a, **k: (np.full(4, 10.), 0))
    with pytest.raises(ValueError, match='KKT'): m.adapt(b, z, 'FuelRidge')


@pytest.mark.parametrize('profile,months', [(np.ones(3), [1]), (np.ones((11, 4)), [1]),
    (np.full(4, np.nan), [1]), (np.ones(4, complex), [1]), (np.ones(4), [13]),
    (np.ones(4), []), (np.ones(4), [[1]])])
def test_prediction_invalid_inputs(profile, months):
    with pytest.raises(ValueError): m.profile_predict(profile, months)


@pytest.mark.parametrize('b,gamma', [(np.ones(3), np.ones(4)),
    (np.ones(4), np.ones(3)), (np.ones(4), [1, 1, -1, 1]),
    (np.ones(4, complex), np.ones(4)), (np.full(4, np.nan), np.ones(4)),
    (np.empty((0, 4)), np.ones(4))])
def test_decode_invalid_inputs(b, gamma):
    with pytest.raises(ValueError): m.decode(b, gamma)


def test_nnls_invalid_numerics_and_runtime_failure_propagate(monkeypatch):
    b = np.ones((12, 4)); z = np.ones(12)
    monkeypatch.setattr(m, 'nnls', lambda *a, **k: (np.full(4, np.nan), 0))
    with pytest.raises(ValueError): m.adapt(b, z, 'FuelRidge')
    def failed(*args, **kwargs): raise RuntimeError('synthetic NNLS failure')
    monkeypatch.setattr(m, 'nnls', failed)
    with pytest.raises(RuntimeError, match='synthetic'): m.adapt(b, z, 'FuelRidge')
