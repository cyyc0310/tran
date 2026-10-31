"""Finite-support CRPS quadratics and ordinary constrained map optimizers."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, nnls

from cif_pit_recalibration import GRID, check_maps

K = len(GRID)
PACKED = 3 * K
INTERIOR = K - 2
# Nonnegative simplex gaps generate all monotone maps with fixed endpoints.
GAP_TO_VALUE = (np.arange(INTERIOR)[:, None] >= np.arange(K - 1)[None]).astype(float)


def statistics(law, actual):
    """Exact CRPS/1000 coefficients; the last entry includes support-outside loss."""
    y = np.asarray(actual, float)
    if y.ndim != 1 or not np.isfinite(y).all():
        raise ValueError('invalid quadratic outcomes')
    n = len(y)
    result = np.zeros((n, PACKED))
    result[:, -1] = abs(law.x[-1] - y) / 1000
    if len(law.x) == 1:
        return result
    width = np.diff(law.x)
    cumulative = np.clip(np.cumsum(law.w)[:-1], 0, 1)
    left = np.clip(np.searchsorted(GRID, cumulative, side='right') - 1, 0, K - 2)
    fraction = (cumulative - GRID[left]) / (GRID[left + 1] - GRID[left])
    basis = np.zeros((len(width), K))
    basis[np.arange(len(width)), left] = 1 - fraction
    basis[np.arange(len(width)), left + 1] = fraction
    diagonal = (np.bincount(left, weights=width * (1 - fraction)**2, minlength=K)
                + np.bincount(left + 1, weights=width * fraction**2, minlength=K))
    off = np.bincount(left, weights=width * fraction * (1 - fraction), minlength=K - 1)
    tail = np.vstack([np.cumsum((width[:, None] * basis)[::-1], axis=0)[::-1], np.zeros(K)])
    interval = np.searchsorted(law.x, y, side='right') - 1
    inside = (interval >= 0) & (interval < len(width))
    b = np.zeros((n, K))
    b[y < law.x[0]] = tail[0]
    selected = interval[inside]
    b[inside] = tail[selected + 1] + (law.x[selected + 1] - y[inside])[:, None] * basis[selected]
    result[:, :K] = diagonal / 1000
    result[:, K:2 * K - 1] = off / 1000
    result[:, 2 * K - 1:-1] = b / 1000
    if not np.isfinite(result).all() or (result < -1e-12).any():
        raise ValueError('invalid quadratic sufficient statistics')
    return result


def unpack(packed):
    packed = np.asarray(packed, float)
    if packed.shape[-1] != PACKED or not np.isfinite(packed).all():
        raise ValueError('invalid packed coefficients')
    H = np.zeros(packed.shape[:-1] + (K, K))
    ids = np.arange(K)
    H[..., ids, ids] = packed[..., :K]
    ids = np.arange(K - 1)
    H[..., ids, ids + 1] = packed[..., K:2 * K - 1]
    H[..., ids + 1, ids] = packed[..., K:2 * K - 1]
    return H, packed[..., 2 * K - 1:-1], packed[..., -1]


def value(packed, mapping, alpha):
    H, b, constant = unpack(packed)
    D = (1 - .95 * alpha) * GRID + .95 * alpha * np.asarray(mapping)
    return np.einsum('...i,...ij,...j->...', D, H, D) - 2 * np.einsum('...i,...i->...', D, b) + constant


def regularization(H):
    return 1e-6 * max(float(np.trace(H)) / INTERIOR, 1e-8)


def mean_system(packed, alpha):
    H, b, _ = unpack(packed)
    if H.ndim != 2 or alpha not in (.5, 1.):
        raise ValueError('mean map expects one quadratic and a fixed strength')
    lam = .95 * alpha
    fixed = (1 - lam) * GRID
    fixed[-1] += lam
    rho = regularization(H)
    A = lam**2 * H[1:-1, 1:-1] + rho * np.eye(INTERIOR)
    rhs = lam * (b[1:-1] - H[1:-1] @ fixed) + rho * GRID[1:-1]
    scale = max(float(np.trace(A)), float(max(abs(rhs))), 1e-16)
    return A / scale, rhs / scale, rho, scale


def simplex_kkt(A, rhs, gaps):
    c = GAP_TO_VALUE @ gaps
    gradient = 2 * GAP_TO_VALUE.T @ (A @ c - rhs)
    positive = gaps > 1e-10
    multiplier = float(np.mean(gradient[positive]))
    error = max(float(max(abs(gradient[positive] - multiplier))),
                float(max(np.maximum(multiplier - gradient[~positive], 0))) if (~positive).any() else 0.,
                float(max(abs(gaps * (gradient - multiplier)))))
    feasible = max(float(max(np.maximum(-gaps, 0))), abs(float(gaps.sum()) - 1))
    return error, feasible


def solve_mean(packed, alpha):
    """Convex simplex active-set solve with explicit numerical fallback."""
    A, rhs, rho, scale = mean_system(packed, alpha)
    gaps = np.diff(GRID).copy()
    active = np.arange(K - 1)
    reason, iterations = 'iteration_limit', 0
    for iterations in range(1, 101):
        last = GAP_TO_VALUE[:, active[-1]]
        U = GAP_TO_VALUE[:, active[:-1]] - last[:, None]
        candidate = np.zeros(K - 1)
        try:
            coefficients = np.linalg.solve(U.T @ A @ U, U.T @ (rhs - A @ last)) if U.shape[1] else np.empty(0)
        except np.linalg.LinAlgError:
            reason = 'linear_solve_failed'
            break
        candidate[active[:-1]] = coefficients
        candidate[active[-1]] = 1 - coefficients.sum()
        if min(candidate[active]) < -1e-12:
            direction = candidate - gaps
            falling = active[direction[active] < -1e-14]
            if not len(falling):
                reason = 'no_feasible_direction'
                break
            step = float(min(gaps[falling] / -direction[falling]))
            if not 0 <= step <= 1 + 1e-10:
                reason = 'invalid_boundary_step'
                break
            gaps += min(step, 1.) * direction
            gaps[gaps < 1e-12] = 0
            gaps /= gaps.sum()
            active = np.flatnonzero(gaps > 0)
            continue
        gaps = np.maximum(candidate, 0)
        gaps /= gaps.sum()
        error, feasible = simplex_kkt(A, rhs, gaps)
        if error <= 1e-7 and feasible <= 1e-12:
            mapping = np.r_[0., GAP_TO_VALUE @ gaps, 1.]
            mapping = check_maps(mapping)
            gain = float(value(packed, GRID, alpha) - value(packed, mapping, alpha))
            penalty = rho * float(np.sum((mapping[1:-1] - GRID[1:-1])**2))
            if gain - penalty < -1e-10:
                reason = 'objective_increased'
                break
            return mapping, {'status': 'solved', 'iterations': iterations, 'used_identity_fallback': False,
                             'kkt_error': error, 'feasibility_error': feasible,
                             'objective_gain': gain, 'regularization': rho, 'objective_scale': scale}
        gradient = 2 * GAP_TO_VALUE.T @ (A @ (GAP_TO_VALUE @ gaps) - rhs)
        active = np.flatnonzero(gaps > 0)
        inactive = np.flatnonzero(gaps == 0)
        if not len(inactive):
            reason = 'uncertified_interior'
            break
        entering = int(inactive[np.argmin(gradient[inactive])])
        active = np.sort(np.r_[active, entering])
    return GRID.copy(), {'status': reason, 'iterations': iterations, 'used_identity_fallback': True,
                         'kkt_error': None, 'feasibility_error': 0., 'objective_gain': 0.,
                         'regularization': rho, 'objective_scale': scale}


def monotone_constraints(c):
    return np.diff(np.r_[0., c, 1.])


def monotone_jacobian():
    J = np.zeros((K - 1, INTERIOR))
    J[np.arange(INTERIOR), np.arange(INTERIOR)] = 1
    J[np.arange(1, K - 1), np.arange(INTERIOR)] -= 1
    return J


def solve_group_max(packed, alpha):
    """Standard global minimax absolute source-region CRPS, not a target guarantee."""
    H, b, constant = unpack(packed)
    if H.ndim != 3 or alpha not in (.5, 1.) or len(H) < 2:
        raise ValueError('group map expects at least two regional quadratics')
    rho = regularization(H.mean(axis=0))
    lam = .95 * alpha
    Jmono = monotone_jacobian()

    def risk(c):
        D = (1 - lam) * GRID + lam * np.r_[0., c, 1.]
        return np.einsum('i,rij,j->r', D, H, D) - 2 * (b @ D) + constant

    def risk_jac(c):
        D = (1 - lam) * GRID + lam * np.r_[0., c, 1.]
        return 2 * lam * (np.einsum('rij,j->ri', H, D) - b)[:, 1:-1]

    def constraints(z):
        return np.r_[monotone_constraints(z[:-1]), z[-1] - risk(z[:-1])]

    def jacobian(z):
        return np.vstack([np.column_stack([Jmono, np.zeros(K - 1)]),
                          np.column_stack([-risk_jac(z[:-1]), np.ones(len(H))])])

    def objective(z):
        return float(z[-1] + rho * np.sum((z[:-1] - GRID[1:-1])**2))

    def gradient(z):
        return np.r_[2 * rho * (z[:-1] - GRID[1:-1]), 1.]

    initial_risk = risk(GRID[1:-1])
    initial = np.r_[GRID[1:-1], max(initial_risk)]
    result = minimize(objective, initial, jac=gradient, method='SLSQP',
                      constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
                      options={'maxiter': 300, 'ftol': 1e-12})
    reason = str(result.message)
    if result.success and np.isfinite(result.x).all():
        slack = constraints(result.x)
        feasible = float(max(np.maximum(-slack, 0)))
        active = slack <= 1e-7
        try:
            multipliers, _ = nnls(jacobian(result.x)[active].T, gradient(result.x), maxiter=1000)
            residual = gradient(result.x) - jacobian(result.x)[active].T @ multipliers
            error = max(float(max(abs(residual))), float(max(abs(multipliers * slack[active]))))
        except (RuntimeError, np.linalg.LinAlgError):
            error = float('inf')
        if feasible <= 1e-9 and error <= 1e-6 and objective(result.x) <= objective(initial) + 1e-10:
            # Clean only sub-tolerance negative gaps, then certify the returned map.
            gaps = np.maximum(np.diff(np.r_[0., result.x[:-1], 1.]), 0)
            gaps /= gaps.sum()
            mapping = np.r_[0., np.cumsum(gaps)]
            mapping[-1] = 1.
            mapping = check_maps(mapping)
            final = np.r_[mapping[1:-1], max(risk(mapping[1:-1]))]
            slack = constraints(final)
            active = slack <= 1e-7
            try:
                multipliers, _ = nnls(jacobian(final)[active].T, gradient(final), maxiter=1000)
                error = max(float(max(abs(gradient(final) - jacobian(final)[active].T @ multipliers))),
                            float(max(abs(multipliers * slack[active]))))
            except (RuntimeError, np.linalg.LinAlgError):
                error = float('inf')
            gain = float(max(initial_risk) - max(risk(mapping[1:-1])))
            final_feasibility = float(max(np.maximum(-slack, 0)))
            if (error <= 1e-6 and final_feasibility <= 1e-9 and gain >= -1e-10
                    and objective(final) <= objective(initial) + 1e-10):
                return mapping, {'status': 'solved', 'iterations': int(result.nit), 'used_identity_fallback': False,
                                 'kkt_error': error, 'feasibility_error': final_feasibility,
                                 'objective_gain': gain, 'regularization': rho, 'objective_scale': 1.}
            reason = 'returned_map_not_certified'
        else:
            reason = 'solution_not_certified'
    return GRID.copy(), {'status': reason, 'iterations': int(result.nit), 'used_identity_fallback': True,
                         'kkt_error': None, 'feasibility_error': 0., 'objective_gain': 0.,
                         'regularization': rho, 'objective_scale': 1.}
