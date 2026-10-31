"""Pure source profiles and coarse-history NNLS; no physical capacity bounds."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import nnls


FUELS = ('coal', 'gas', 'oil', 'other')
EF = np.array([1000., 400., 800., 400.])
KINDS = ('None', 'Common', 'FuelRidge')
LAMBDA = .01
KKT_ATOL = 1e-8
RANK_RTOL = 1e-10
HISTORY_LENGTH = 12
NNLS_MAXITER = 10000


def _array(value: Any, name: str, shape: tuple[int, ...] | None = None,
           nonnegative: bool = True) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f'Real {name} required')
    try:
        array = np.asarray(raw, float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Real numeric {name} required') from exc
    if ((shape is not None and array.shape != shape) or not np.isfinite(array).all()
            or (nonnegative and np.any(array < 0))):
        raise ValueError(f'Finite nonnegative {name} with shape {shape} required')
    return array


def _months(value: Any, n: int | None = None) -> np.ndarray:
    array = _array(value, 'calendar months')
    if (array.ndim != 1 or not len(array) or (n is not None and array.shape != (n,))
            or np.any(array != np.floor(array)) or np.any((array < 1) | (array > 12))):
        raise ValueError('Integer calendar months 1..12 required')
    return array.astype(int)


def fit_profiles(generation_mw: np.ndarray, capacities: np.ndarray, months: np.ndarray,
                 weights: np.ndarray, keys: list[str]) -> dict[str, Any]:
    """Valid C>0 rates only; excluded C=G=0 cells never become rate observations."""
    g = _array(generation_mw, 'generation')
    if g.ndim != 2 or g.shape[1] != 4 or not len(g):
        raise ValueError('Nonempty (N,4) generation required')
    n = len(g)
    c = _array(capacities, 'capacities', (n, 4))
    month_ids = _months(months, n)
    w = _array(weights, 'weights', (n,))
    if np.any(w <= 0):
        raise ValueError('Strictly positive weights required')
    if (not isinstance(keys, list) or len(keys) != n
            or any(not isinstance(key, str) or not key for key in keys) or len(set(keys)) != n):
        raise ValueError('Unique nonempty source keys required')
    valid = c > 0
    contradictions = np.argwhere(~valid & (g > 0))
    if len(contradictions):
        row, fuel = contradictions[0]
        raise ValueError(f'Zero capacity positive generation: {keys[row]} fuel={FUELS[fuel]}')
    global_mean, count, weight_sum, weighted_sum = (np.zeros(4) for _ in range(4))
    month_count, month_weight, month_sum = (np.zeros((12, 4)) for _ in range(3))
    for fuel in range(4):
        mask = valid[:, fuel]
        if not mask.any():
            raise ValueError(f'No defined source rates for {FUELS[fuel]}')
        rates = _array(g[mask, fuel]/c[mask, fuel], 'defined source rates')
        selected_w = w[mask]
        count[fuel], weight_sum[fuel] = mask.sum(), selected_w.sum()
        weighted_sum[fuel] = selected_w@rates
        global_mean[fuel] = weighted_sum[fuel]/weight_sum[fuel]
        for month in range(1, 13):
            selected = month_ids[mask] == month
            month_count[month-1, fuel] = selected.sum()
            month_weight[month-1, fuel] = selected_w[selected].sum()
            month_sum[month-1, fuel] = selected_w[selected]@rates[selected]
    curves = (month_sum+global_mean)/(month_weight+1.)
    _array(global_mean, 'global profile'); _array(curves, 'month profile')
    _array(weighted_sum, 'weighted source rate sums'); _array(month_sum, 'monthly weighted source rate sums')
    _array(weight_sum, 'valid weight sums'); _array(month_weight, 'monthly valid weight sums')
    zero_capacity = [dict(key=keys[row], fuel=FUELS[fuel]) for row, fuel in np.argwhere(~valid)]
    return dict(month=curves, valid=valid, audit=dict(
        zero_capacity=zero_capacity, zero_capacity_count=(~valid).sum(0),
        valid_count=count.astype(int), valid_weight=weight_sum,
        valid_rate_weighted_sum=weighted_sum, valid_rate_mean=global_mean.copy(),
        month_valid_count=month_count.astype(int), month_valid_weight=month_weight,
        month_rate_weighted_sum=month_sum, shrinkage_weight=1.)) | {'global': global_mean}


def profile_predict(profile: np.ndarray, month_ids: np.ndarray) -> np.ndarray:
    """Select a fixed global or calendar-month source profile."""
    profile = _array(profile, 'profile')
    months = _months(month_ids)
    if profile.shape == (4,):
        return np.broadcast_to(profile, (len(months), 4)).copy()
    if profile.shape != (12, 4):
        raise ValueError('Profile shape (4,) or (12,4) required')
    return profile[months-1].copy()


def _kkt(parameters: np.ndarray, gradient: np.ndarray) -> dict[str, Any]:
    positive = parameters > 0
    active_error = float(np.max(np.abs(gradient[positive]), initial=0.))
    boundary_error = float(np.max(np.maximum(-gradient[~positive], 0), initial=0.))
    violation = max(active_error, boundary_error)
    if violation > KKT_ATOL:
        raise ValueError(f'NNLS KKT violation {violation} exceeds {KKT_ATOL}; no fallback')
    return dict(applicable=True, positive_gradient_max_abs=active_error,
                zero_negative_gradient_max=boundary_error, max_violation=violation, tolerance=KKT_ATOL)


def adapt(B: np.ndarray, z: np.ndarray, kind: str = 'None') -> dict[str, Any]:
    """Fixed None/Common/FuelRidge coarse adaptation; Common=0 has no decoder fallback."""
    B = _array(B, 'normalized operator', (12, 4))
    z = _array(z, 'normalized POS response', (12,))
    if kind not in KINDS:
        raise ValueError('Adaptation kind must be None, Common or FuelRidge')
    alpha, common_gradient, solver_residual = None, None, None
    if kind == 'None':
        gamma = np.ones(4)
    else:
        if kind == 'Common':
            if not np.any(B.sum(1) > 0):
                raise ValueError('Common alpha is unidentifiable for an all-zero operator')
            operator = B.sum(1)[:, None]/np.sqrt(12)
            response = z/np.sqrt(12)
        else:
            operator = np.vstack([B/np.sqrt(12), np.sqrt(LAMBDA)*np.eye(4)])
            response = np.r_[z/np.sqrt(12), np.full(4, np.sqrt(LAMBDA))]
        fitted, solver_residual = nnls(operator, response, maxiter=NNLS_MAXITER)
        fitted = _array(fitted, 'NNLS parameters', (1,) if kind == 'Common' else (4,))
        if not np.isfinite(solver_residual):
            raise ValueError('Finite NNLS residual required')
        alpha = float(fitted[0]) if kind == 'Common' else None
        gamma = np.repeat(alpha, 4) if kind == 'Common' else fitted
    residual = _array(B@gamma-z, 'history residual', (12,), nonnegative=False)
    gradient = 2*B.T@residual/12
    penalty = 0.
    if kind == 'FuelRidge':
        penalty = float(LAMBDA*np.sum((gamma-1)**2))
        gradient += 2*LAMBDA*(gamma-1)
    gradient = _array(gradient, 'objective gradient', (4,), nonnegative=False)
    if kind == 'Common':
        common_gradient = float(gradient.sum())
        kkt = _kkt(np.array([alpha]), np.array([common_gradient]))
    elif kind == 'FuelRidge':
        kkt = _kkt(gamma, gradient)
    else:
        kkt = dict(applicable=False)
    objective = float(np.mean(residual**2)+penalty)
    if not np.isfinite(objective):
        raise ValueError('Finite inverse objective required')
    return dict(kind=kind, gamma=gamma, alpha=alpha, residual=residual, objective=objective,
                regularization=penalty, gradient=gradient, common_gradient=common_gradient, KKT=kkt,
                solver_residual=solver_residual, lambda_=LAMBDA if kind == 'FuelRidge' else 0.,
                kkt_atol=KKT_ATOL, nnls_maxiter=NNLS_MAXITER)


def decode(b: np.ndarray, gamma: np.ndarray) -> dict[str, Any]:
    """Normalize positive forward mass; zero mass fails, even for Common=0."""
    b = _array(b, 'query forward mass')
    if b.ndim < 1 or b.shape[-1] != 4 or not b.size:
        raise ValueError('Query forward mass must have last dimension four')
    gamma = _array(gamma, 'fuel multipliers', (4,))
    mass_by_fuel = _array(b*gamma, 'adapted forward mass')
    mass = mass_by_fuel.sum(-1)
    if not np.isfinite(mass).all() or np.any(mass <= 0):
        raise ValueError('Strictly positive finite decoded mass required; no fallback')
    ratios = mass_by_fuel/mass[..., None]
    return dict(EF=ratios@EF, ratios=ratios, mass=mass)


def operator_diagnostics(B: np.ndarray) -> dict[str, Any]:
    """Numerical algebra only; rank and regularization do not certify identification."""
    B = _array(B, 'normalized operator', (12, 4))
    singular_values = _array(np.linalg.svd(B, compute_uv=False), 'singular values', (4,))
    rank = int(np.sum(singular_values > RANK_RTOL*singular_values[0]))
    condition = float(singular_values[0]/singular_values[-1]) if rank == 4 else None
    return dict(singular_values=singular_values, rank=rank, full_rank=rank == 4,
                condition_number=condition, rank_rtol=RANK_RTOL)
