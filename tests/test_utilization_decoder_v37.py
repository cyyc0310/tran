"""Synthetic CCE, support and optimizer tests; no archived source inputs."""
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import utilization_decoder_v37 as m


def toy():
    rng = np.random.default_rng(3701)
    x = rng.normal(size=(12, 36)); x[:, 2] = 7
    q = rng.dirichlet(np.ones(4), size=12)
    x[:, 31:35] = q
    query_q = rng.dirichlet(np.ones(4), size=6)
    query = rng.normal(size=(6, 128, 36)); query[..., 2] = 7
    query[..., 31:35] = query_q[:, None]
    y = .7*q+.3*np.array([.1, .5, .2, .2])
    w = np.linspace(.4, 1.6, 12)
    return x, y, w, q, query, query_q, [f'synthetic-{i}' for i in range(12)]


@pytest.mark.parametrize('offset', [False, True])
def test_analytic_gradient_matches_central_differences(offset):
    rng = np.random.default_rng(4)
    z = np.c_[np.ones(5), rng.normal(size=(5, 2))]
    y = rng.dirichlet(np.ones(4), 5); w = np.arange(1., 6.)
    q = rng.dirichlet(np.ones(4), 5) if offset else None
    theta = rng.normal(size=9)*.2
    value, grad = m.objective(theta, z, y, w, q)
    numeric = []
    for j in range(len(theta)):
        delta = np.zeros_like(theta); delta[j] = 1e-6
        numeric.append((m.objective(theta+delta, z, y, w, q)[0]
                        -m.objective(theta-delta, z, y, w, q)[0])/2e-6)
    assert np.isfinite(value)
    np.testing.assert_allclose(grad, numeric, atol=1e-8, rtol=1e-6)


def test_initialization_and_masked_zero_support_include_absent_gas():
    z = np.ones((2, 1)); theta = np.zeros((1, 3))
    q = np.array([[0., 0., .6, .4], [.2, .8, 0., 0.]])
    np.testing.assert_allclose(m.probabilities(theta, z), .25)
    np.testing.assert_array_equal(m.probabilities(theta, z, q), q)
    value, grad = m.objective(theta.ravel(), z, q, np.ones(2), q)
    assert np.isfinite(value); np.testing.assert_allclose(grad, 0, atol=1e-15)
    bad = q.copy(); bad[0] = [.1, 0., .5, .4]
    assert np.isfinite(m.objective(theta.ravel(), z, bad, np.ones(2))[0])
    with pytest.raises(ValueError, match='coal'):
        m.objective(theta.ravel(), z, bad, np.ones(2), q)


def test_stable_extreme_logits_do_not_log_underflowed_probabilities():
    theta = np.array([[1000., -1000., 500.]])
    y = np.array([[0., 1., 0., 0.]])
    value, grad = m.objective(theta.ravel(), np.ones((1, 1)), y, np.ones(1))
    assert np.isfinite(value) and np.isfinite(grad).all()


def test_weighted_sum_and_intercept_penalty_are_exact():
    z = np.ones((2, 1)); theta = np.array([.2, -.4, .3])
    y = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])
    w = np.array([2., 3.]); penalty = .5*theta@theta
    a, ga = m.objective(theta, z, y, w)
    b, gb = m.objective(theta, z, y, 2*w)
    assert b-penalty == pytest.approx(2*(a-penalty))
    np.testing.assert_allclose(gb-theta, 2*(ga-theta))
    p = m.probabilities(theta.reshape(1, 3), z)
    expected = -np.sum(w*np.log(p[[0, 1], [0, 1]]))+penalty
    assert a == pytest.approx(expected)


def test_three_fits_weighted_scaler_support_certificate_and_immutability():
    args = toy(); originals = [a.copy() for a in args[:-1]]
    result = m.fit_predict(*args)
    assert m.ARMS == ('PlainSoftmax36', 'OffsetSoftmax36', 'OffsetInterceptOnly')
    assert result['ratios'].shape == (3, 6, 128, 4)
    assert result['EF_draws'].shape == (3, 6, 128)
    np.testing.assert_allclose(result['ratios'].sum(-1), 1, atol=1e-14)
    np.testing.assert_allclose(result['EF_draws'], result['ratios']@np.array([1000, 400, 800, 400]))
    x, y, w, q, query, query_q, keys = args
    mean = np.average(x, axis=0, weights=w)
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=w)); scale[scale <= 1e-12] = 1
    fits = result['audit']['fits']
    for i, fit in enumerate(fits):
        assert fit['arm'] == m.ARMS[i] and fit['training_keys'] == keys
        np.testing.assert_array_equal(fit['scaler_mean'], mean if i < 2 else np.zeros(0))
        np.testing.assert_array_equal(fit['scaler_scale'], scale if i < 2 else np.ones(0))
        assert fit['theta'].shape == ((37, 3) if i < 2 else (1, 3))
        assert fit['gradient_l2'] <= 1e-5 and fit['optimizer']['success']
        assert fit['objective_gap_upper_bound'] == pytest.approx(fit['gradient_l2']**2/2)
        assert fit['objective_penalized'] == pytest.approx(fit['objective_unpenalized']+fit['regularization'])
    for actual, original in zip(args[:-1], originals):
        np.testing.assert_array_equal(actual, original)
    np.testing.assert_array_equal(result['ratios'][2], np.broadcast_to(result['ratios'][2, :, :1], (6, 128, 4)))


def test_zero_support_at_prediction_is_exact_but_plain_is_unrestricted():
    args = list(toy()); args[5][:, 0] = 0; args[5] /= args[5].sum(1)[:, None]
    args[4][..., 31:35] = args[5][:, None]
    r = m.fit_predict(*args)['ratios']
    assert (r[0, ..., 0] > 0).all()
    np.testing.assert_array_equal(r[1:, ..., 0], 0)


def test_support_mismatch_reports_training_key_and_fuel_before_solver(monkeypatch):
    args = list(toy()); args[3][2, 0] = 0; args[3][2] /= args[3][2].sum()
    args[0][:, 31:35] = args[3]
    monkeypatch.setattr(m, 'minimize', lambda *a, **k: pytest.fail('must reject before fitting'))
    with pytest.raises(ValueError, match='synthetic-2.*coal'):
        m.fit_predict(*args)


@pytest.mark.parametrize('kind', ['failure', 'nan', 'large_gradient'])
def test_optimizer_failures_never_fallback(monkeypatch, kind):
    def fake(fun, x0, **kwargs):
        theta = x0.copy()
        if kind == 'nan': theta[0] = np.nan
        return SimpleNamespace(x=theta, success=kind != 'failure', status=2, message='toy',
                               fun=1., jac=np.zeros_like(theta), nit=1, nfev=2, njev=2)
    monkeypatch.setattr(m, 'minimize', fake)
    with pytest.raises(ValueError, match='optimization|gradient|[Ff]inite'):
        m.fit_predict(*toy())


def test_minimize_configuration_and_zero_initialization_are_locked(monkeypatch):
    real = m.minimize; calls = []
    def wrapped(fun, x0, **kwargs):
        calls.append((x0.copy(), kwargs.copy()))
        return real(fun, x0, **kwargs)
    monkeypatch.setattr(m, 'minimize', wrapped)
    m.fit_predict(*toy())
    assert len(calls) == 3
    for x0, kwargs in calls:
        np.testing.assert_array_equal(x0, 0)
        assert kwargs == dict(jac=True, method='L-BFGS-B', options=dict(
            maxiter=10000, maxls=50, ftol=1e-15, gtol=1e-9))


@pytest.mark.parametrize('index,kind', [(0, 'nan'), (0, 'shape'), (1, 'negative'),
    (1, 'sum'), (2, 'zero'), (2, 'nan'), (3, 'sum'), (4, 'shape'), (5, 'negative'),
    (6, 'duplicate'), (0, 'q_alignment'), (4, 'q_alignment')])
def test_invalid_inputs_fail_closed(index, kind):
    args = list(toy())
    if kind == 'duplicate': args[6][1] = args[6][0]
    elif kind == 'shape': args[index] = args[index][:-1] if index == 4 else args[index][:, :-1]
    elif kind == 'q_alignment': args[index][..., 31] += .01
    else: args[index].flat[0] = {'nan': np.nan, 'negative': -.1, 'sum': .9, 'zero': 0}[kind]
    with pytest.raises(ValueError): m.fit_predict(*args)


def test_strong_convexity_certificate_bounds_toy_objective_gap():
    z = np.ones((3, 1)); y = np.repeat([[.1, .6, .2, .1]], 3, axis=0); w = np.ones(3)
    from scipy.optimize import minimize
    best = minimize(lambda t: m.objective(t, z, y, w), np.zeros(3), jac=True, method='BFGS',
                    options={'gtol': 1e-11})
    t = np.array([.4, -.3, .2]); value, grad = m.objective(t, z, y, w)
    assert value-best.fun <= grad@grad/2+1e-12


def test_fit_with_absent_gas_and_zero_responses_preserves_exact_support():
    args = list(toy())
    for q in (args[3], args[5]):
        q[:, 1] = 0; q /= q.sum(1, keepdims=True)
    args[1] = args[3].copy()
    args[0][:, 31:35] = args[3]; args[4][..., 31:35] = args[5][:, None]
    result = m.fit_predict(*args)
    np.testing.assert_array_equal(result['ratios'][1:, ..., 1], 0)
    assert np.isfinite(result['EF_draws']).all()
    assert (result['ratios'][0, ..., 1] > 0).all()


def test_uniform_capacity_offsets_match_plain_objective_and_gradient():
    args = toy(); z = np.c_[np.ones(12), args[0]]
    theta = np.linspace(-.1, .1, 111)
    plain = m.objective(theta, z, args[1], args[2])
    offset = m.objective(theta, z, args[1], args[2], np.full((12, 4), .25))
    assert plain[0] == pytest.approx(offset[0])
    np.testing.assert_allclose(plain[1], offset[1], atol=1e-14)
