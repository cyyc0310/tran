"""Independent CRPS identities, mean QP checks and an analytic minimax toy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize

from cif_crps_quadratic import (GRID, INTERIOR, statistics, value, solve_mean,
                                solve_group_max, mean_system, monotone_constraints, monotone_jacobian)
from cif_pit_recalibration import recalibrate
from v151_hourly_law import sha
from v154_hourly_strong_baselines import CompressedLaw

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default='r1')
    args = parser.parse_args()
    out = ROOT / f'results/v175-quadratic-preflight-{args.version}'
    if out.exists():
        raise FileExistsError(out)
    rng = np.random.default_rng(20261029)
    identity_error = transformed_error = independent_qp_error = 0.
    certificates = []
    started = time.monotonic()
    for case in range(100):
        n = 1 if case == 0 else int(rng.integers(2, 100))
        x = np.sort(rng.choice(np.arange(1001.), size=n, replace=False))
        law = CompressedLaw(x, rng.dirichlet(np.ones(n)))
        actual = np.r_[rng.uniform(-100, 1100, 20), x[0], x[-1]]
        stats = statistics(law, actual)
        identity_error = max(identity_error, float(max(abs(value(stats, GRID, 1.) * 1000 - law.crps(actual)))))
        for alpha in (.5, 1.):
            mapping = np.r_[0., np.sort(rng.uniform(size=INTERIOR)), 1.]
            updated = recalibrate(law, mapping, alpha)
            transformed_error = max(transformed_error, float(max(abs(value(stats, mapping, alpha) * 1000 - updated.crps(actual)))))
            aggregate = np.mean(stats, axis=0)
            mapping, cert = solve_mean(aggregate, alpha)
            certificates.append({'case': case, 'alpha': alpha, **cert})
            if cert['used_identity_fallback']:
                continue
            if case < 30:
                A, rhs, _, _ = mean_system(aggregate, alpha)
                optimized = minimize(lambda c: float(c @ A @ c - 2 * rhs @ c), GRID[1:-1],
                                     jac=lambda c: 2 * (A @ c - rhs), method='SLSQP',
                                     constraints={'type': 'ineq', 'fun': monotone_constraints,
                                                  'jac': lambda c: monotone_jacobian()},
                                     options={'maxiter': 500, 'ftol': 1e-12})
                if not optimized.success:
                    raise ValueError('independent toy QP failed')
                diff = abs(float(mapping[1:-1] @ A @ mapping[1:-1] - 2 * rhs @ mapping[1:-1]) - optimized.fun)
                independent_qp_error = max(independent_qp_error, diff)
    regional = np.stack([statistics(CompressedLaw([0, 1000], [.5, .5]), [0])[0],
                         statistics(CompressedLaw([0, 500], [.5, .5]), [500])[0]])
    analytic = np.sqrt(2) / (1 + np.sqrt(2))
    minimax = []
    for alpha in (.5, 1.):
        mapping, cert = solve_group_max(regional, alpha)
        if cert['used_identity_fallback']:
            raise ValueError('analytic minimax toy fell back')
        D = (1 - .95 * alpha) * .5 + .95 * alpha * mapping[np.flatnonzero(GRID == .5)[0]]
        if abs(D - analytic) > 1e-6:
            raise ValueError('analytic minimax optimum differs')
        minimax.append({'alpha': alpha, 'D_half': float(D), 'analytic': float(analytic), **cert})
    if identity_error > 1e-9 or transformed_error > 1e-9 or independent_qp_error > 1e-7:
        raise ValueError('independent numerical identity failed')
    out.mkdir(parents=True)
    (out / 'certificates.json').write_text(json.dumps(certificates, indent=2, allow_nan=False) + '\n')
    result = {'assertion_failures': 0, 'random_mean_cases': len(certificates),
              'mean_fallbacks': sum(c['used_identity_fallback'] for c in certificates),
              'max_identity_CRPS_error': identity_error, 'max_transformed_CRPS_error': transformed_error,
              'max_independent_QP_objective_error': independent_qp_error, 'analytic_minimax': minimax,
              'seconds': time.monotonic() - started, 'held_CIF_scored': False,
              'input_hashes': {str(p.relative_to(ROOT)): sha(p) for p in
                               [Path(__file__), ROOT / 'scripts/cif_crps_quadratic.py',
                                ROOT / 'docs/V175_DIRECT_CRPS_MAP_PROTOCOL.md']},
              'output_hashes': {'certificates.json': sha(out / 'certificates.json')}}
    (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('input_hashes', 'output_hashes', 'analytic_minimax')}, indent=2))


if __name__ == '__main__':
    main()
