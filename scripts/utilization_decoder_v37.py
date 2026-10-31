"""Pure B1 weighted-CCE baselines; registered inventory is not physical support."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from scripts import fleet_information_v34 as v34


ARMS = ('PlainSoftmax36', 'OffsetSoftmax36', 'OffsetInterceptOnly')
FUELS = ('coal', 'gas', 'oil', 'other')
FREE_CLASSES = (0, 2, 3)
FEATURE_NAMES = v34.FEATURE_NAMES['SizeCap']
EF = np.asarray(v34.EF[v34.POS], float)
LAMBDA = 1.
GRADIENT_TOL = 1e-5
OPTIMIZER_OPTIONS = dict(maxiter=10000, maxls=50, ftol=1e-15, gtol=1e-9)


def _real_array(value: Any, name: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f'Real {name} required')
    return np.asarray(value, float)


def _finite(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = _real_array(value, name)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f'Finite {name} with shape {shape} required')
    return array


def _simplex(value: Any, n: int, name: str) -> np.ndarray:
    array = _finite(value, (n, 4), name)
    if np.any(array < 0) or not np.allclose(array.sum(1), 1., atol=1e-10, rtol=0):
        raise ValueError(f'Nonnegative simplex {name} required')
    return array


def _support(y: np.ndarray, q: np.ndarray, keys: list[str] | None = None) -> None:
    mismatches = np.argwhere((q == 0) & (y > 0))
    if len(mismatches):
        row, fuel = mismatches[0]
        key = str(row) if keys is None else keys[row]
        raise ValueError(f'Offset support mismatch: {key} fuel={FUELS[fuel]}; no epsilon or fallback')


def _log_probabilities(theta: np.ndarray, design: np.ndarray,
                       q: np.ndarray | None) -> np.ndarray:
    design = _real_array(design, 'design')
    if design.ndim != 2 or len(design) == 0:
        raise ValueError('Nonempty two-dimensional design required')
    _finite(design, design.shape, 'design')
    theta = _finite(theta, (design.shape[1], 3), 'theta')
    logits = np.zeros((len(design), 4))
    with np.errstate(over='raise', invalid='raise'):
        try:
            logits[:, FREE_CLASSES] = design@theta
            if q is not None:
                q = _simplex(q, len(design), 'capacity shares')
                offset = np.full_like(q, -np.inf)
                np.log(q, out=offset, where=q > 0)
                logits += offset
            logp = logits-logsumexp(logits, axis=1, keepdims=True)
        except FloatingPointError as exc:
            raise ValueError('Finite active log probabilities required') from exc
    active = np.ones_like(logp, dtype=bool) if q is None else q > 0
    if not np.isfinite(logp[active]).all():
        raise ValueError('Finite active log probabilities required')
    return logp


def probabilities(theta: np.ndarray, design: np.ndarray,
                  q: np.ndarray | None = None) -> np.ndarray:
    """Gas-reference softmax; q=0 stays exactly zero, with no epsilon."""
    return np.exp(_log_probabilities(theta, design, q))


def objective(flat_theta: np.ndarray, design: np.ndarray, y: np.ndarray,
              weights: np.ndarray, q: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """Weighted SUM CCE plus lambda/2 times all free parameters squared."""
    design = _real_array(design, 'design')
    if design.ndim != 2 or not len(design):
        raise ValueError('Nonempty two-dimensional design required')
    y = _simplex(y, len(design), 'responses')
    weights = _finite(weights, (len(design),), 'weights')
    if np.any(weights <= 0):
        raise ValueError('Strictly positive weights required')
    flat_theta = _finite(flat_theta, (design.shape[1]*3,), 'flat theta')
    theta = flat_theta.reshape(design.shape[1], 3)
    if q is not None:
        q = _simplex(q, len(design), 'capacity shares')
        _support(y, q)
    logp = _log_probabilities(theta, design, q)
    positive = y > 0
    terms = np.zeros_like(y)
    terms[positive] = y[positive]*logp[positive]
    value = -np.sum(weights*terms.sum(1)) + .5*LAMBDA*np.sum(theta**2)
    residual = np.exp(logp)*y.sum(1, keepdims=True)-y
    gradient = design.T@(weights[:, None]*residual[:, FREE_CLASSES])+LAMBDA*theta
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise ValueError('Finite objective and gradient required')
    return float(value), gradient.ravel()


def fit_predict(train_X: np.ndarray, y: np.ndarray, weights: np.ndarray,
                train_q: np.ndarray, query_X: np.ndarray, query_q: np.ndarray,
                training_keys: list[str]) -> dict[str, Any]:
    """Fit three fixed source-only heads; caller enforces cohort/date provenance."""
    train_X = _real_array(train_X, 'training features')
    if train_X.ndim != 2 or not len(train_X):
        raise ValueError('Nonempty (N,36) training features required')
    n = len(train_X)
    train_X = _finite(train_X, (n, 36), 'training features')
    query_X = _finite(query_X, (6, 128, 36), 'query features')
    y = _simplex(y, n, 'responses')
    train_q = _simplex(train_q, n, 'training capacity shares')
    query_q = _simplex(query_q, 6, 'query capacity shares')
    weights = _finite(weights, (n,), 'weights')
    if np.any(weights <= 0):
        raise ValueError('Strictly positive weights required')
    if (not isinstance(training_keys, list) or len(training_keys) != n
            or any(not isinstance(key, str) or not key for key in training_keys)
            or len(set(training_keys)) != n):
        raise ValueError('Unique nonempty training keys required')
    ids = [FEATURE_NAMES.index(name) for name in v34.CAPACITY_FEATURE_NAMES[:4]]
    if (not np.array_equal(train_X[:, ids], train_q)
            or not np.array_equal(query_X[..., ids], np.broadcast_to(query_q[:, None], (6, 128, 4)))):
        raise ValueError('Feature names and capacity share alignment required')
    _support(y, train_q, training_keys)  # Offset-only incompatibility, before any of the three fits.
    mean = np.average(train_X, axis=0, weights=weights)
    scale = np.sqrt(np.average((train_X-mean)**2, axis=0, weights=weights))
    scale[scale <= 1e-12] = 1.
    design = np.c_[np.ones(n), (train_X-mean)/scale]
    query_design = np.c_[np.ones(6*128), ((query_X-mean)/scale).reshape(-1, 36)]
    ratios, fits = [], []
    for index, arm in enumerate(ARMS):
        z = design if index < 2 else np.ones((n, 1))
        zquery = query_design if index < 2 else np.ones((6*128, 1))
        q = None if index == 0 else train_q
        query_offset = None if index == 0 else np.repeat(query_q, 128, axis=0)
        fun = lambda theta: objective(theta, z, y, weights, q)
        result = minimize(fun, np.zeros(z.shape[1]*3), jac=True, method='L-BFGS-B',
                          options=OPTIMIZER_OPTIONS.copy())
        if not result.success:
            raise ValueError(f'{arm} optimization failed: {result.message}; no fallback')
        theta = _finite(result.x, (z.shape[1]*3,), 'optimizer theta')
        _finite(result.jac, theta.shape, 'optimizer gradient')
        if not np.isfinite(result.fun):
            raise ValueError('Finite optimizer objective required')
        value, gradient = fun(theta)
        norm = float(np.linalg.norm(gradient))
        if norm > GRADIENT_TOL:
            raise ValueError(f'{arm} gradient_l2={norm} exceeds {GRADIENT_TOL}; no fallback')
        matrix = theta.reshape(z.shape[1], 3)
        prediction = probabilities(matrix, zquery, query_offset).reshape(6, 128, 4)
        regularization = float(.5*LAMBDA*np.sum(theta**2))
        fits.append(dict(arm=arm, training_keys=training_keys.copy(), theta=matrix.copy(),
            scaler_mean=mean.copy() if index < 2 else np.zeros(0),
            scaler_scale=scale.copy() if index < 2 else np.ones(0),
            objective_penalized=value, objective_unpenalized=value-regularization,
            regularization=regularization, gradient=gradient.copy(), gradient_l2=norm,
            gradient_inf=float(np.max(np.abs(gradient))),
            objective_gap_upper_bound=norm**2/(2*LAMBDA),
            certificate='lambda-strong-convexity: suboptimality <= gradient_l2^2/(2*lambda)',
            lambda_=LAMBDA, gas_reference=0., free_classes=FREE_CLASSES, penalize_intercept=True,
            objective_reduction='weighted_sum', feature_names=FEATURE_NAMES if index < 2 else (),
            optimizer=dict(method='L-BFGS-B', options=OPTIMIZER_OPTIONS.copy(), x0=np.zeros_like(theta),
                success=bool(result.success), status=int(result.status), message=str(result.message),
                nit=int(result.nit), nfev=int(result.nfev),
                njev=None if result.njev is None else int(result.njev),
                fun=float(result.fun), jac=np.asarray(result.jac).copy())))
        ratios.append(prediction)
    predictions = np.stack(ratios)
    return dict(ratios=predictions, EF_draws=predictions@EF,
                audit=dict(arms=ARMS, fits=fits, sample_weights=weights.copy(),
                           scope='registered inventory; output zero is not certified physical absence'))
