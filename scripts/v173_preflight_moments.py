"""Synthetic certificates for moment protection; no held-out CIF scoring."""
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from moment_protection import MomentGeometry
from moment_protected_cif import MomentProtectedCIF, protected_features
from frozen_recomposition import FrozenRecomposition
from hourly_fuel_data import HourlyFuelData
from monthly_history_features import MonthlyInformation
from threadpoolctl import threadpool_limits
from scipy.optimize import minimize
from v151_hourly_law import EF, sha
from zero_odds_cif import odds_update

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/v173-moment-preflight-r4'


def check(prior, proposed, fuel):
    results = []
    for mode in ('GE', 'fuel8'):
        features = protected_features(fuel, mode)
        geometry = MomentGeometry(prior, features, float(prior @ fuel.sum(axis=1)))
        posterior, details = geometry.project(proposed)
        assert np.isfinite(posterior).all() and (posterior > 0).all() and abs(posterior.sum() - 1) < 1e-12
        assert geometry.error(posterior) <= 1e-10
        values = fuel @ EF / fuel.sum(axis=1)
        results.append({'mode': mode, **details, 'probability': posterior.tolist(),
                        'cif_mean_before': float(prior @ values), 'cif_mean_after': float(posterior @ values)})
    return results


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    fuel = np.zeros((8, 8))
    fuel[:, [0, 1, 4]] = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [2, 1, 1], [1, 2, 1]]
    prior = np.full(8, 1 / 8)
    proposed = odds_update(prior, fuel == 0, np.full(8, .2), 1.)
    cases = [{'case': 'mixed', 'results': check(prior, proposed, fuel)}]
    assert any(abs(r['cif_mean_after'] - r['cif_mean_before']) > .01 for r in cases[0]['results'])
    assert not any(r['used_prior_fallback'] for r in cases[0]['results'])
    independent_checks = []
    for record in cases[0]['results']:
        geometry = MomentGeometry(prior, protected_features(fuel, record['mode']), float(prior @ fuel.sum(axis=1)))
        matrix = np.column_stack([np.ones(len(prior)), geometry.coordinates])
        target = matrix.T @ prior
        optimum = minimize(lambda p: float(p @ (np.log(p) - np.log(proposed))), prior,
                           jac=lambda p: np.log(p) - np.log(proposed) + 1,
                           bounds=[(1e-12, 1.)] * len(prior), method='SLSQP',
                           constraints=[{'type': 'eq', 'fun': lambda p: matrix.T @ p - target,
                                         'jac': lambda p: matrix.T}],
                           options={'ftol': 1e-13, 'maxiter': 500})
        projected = np.asarray(record['probability'])
        error = abs(float(projected @ (np.log(projected) - np.log(proposed))) - float(optimum.fun))
        assert optimum.success and error < 1e-9 and np.max(abs(projected - optimum.x)) < 1e-6
        independent_checks.append({'mode': record['mode'], 'objective_error': error,
                                   'max_probability_error': float(np.max(abs(projected - optimum.x)))})
    complementary = np.zeros((2, 8)); complementary[:, :2] = np.eye(2)
    cases.append({'case': 'complementary_zero', 'results': check(np.array([.3, .7]), np.array([.8, .2]), complementary)})
    assert all(np.max(abs(np.asarray(r['probability']) - [.3, .7])) < 1e-9 for r in cases[-1]['results'])
    constant = np.zeros((3, 8)); constant[:, 1] = 1.
    cases.append({'case': 'constant', 'results': check(np.full(3, 1 / 3), np.array([.2, .3, .5]), constant)})
    collinear = np.zeros((4, 8)); collinear[:, 1] = [1, 2, 3, 4]
    cases.append({'case': 'collinear_constant_cif', 'results': check(np.full(4, .25), np.array([.1, .1, .1, .7]), collinear)})
    cases.append({'case': 'rare', 'results': check(np.array([1 - 1e-12, 1e-12]), np.array([.5, .5]), complementary)})
    assert not any(r['used_prior_fallback'] for c in cases for r in c['results'])
    with threadpool_limits(limits=2):
        data, monthly = HourlyFuelData(ROOT), MonthlyInformation(ROOT)
        started = time.monotonic()
        certificates, queries = [], []
        for fold in range(6):
            frozen = FrozenRecomposition(data, ROOT / 'results/v159-fuel-activity-source-pilot-r1', fold, 'asof_30d')
            predictor = MomentProtectedCIF(frozen, monthly, ROOT / f'results/v171-two-scale-zero-r1/fold_{fold}')
            for query in frozen.queries.profile_id.iloc[[0, -1]]:
                laws, _, details, cert = predictor.build(int(query))
                assert len(laws) == 18 and len(details) == 16 and len(cert) == 768
                certificates.extend({'fold': fold, 'profile_id': int(query)} | c for c in cert)
                queries.append(int(query))
        seconds = time.monotonic() - started
        assert max(c['moment_error'] for c in certificates) <= 1e-10
        source_checks = {'queries': queries, 'projection_calls': len(certificates),
                         'prior_fallbacks': sum(c['used_prior_fallback'] for c in certificates),
                         'max_moment_error': max(c['moment_error'] for c in certificates),
                         'max_generation_error': max(c['generation_error'] for c in certificates),
                         'max_emissions_error': max(c['emissions_error'] for c in certificates),
                         'max_successful_stationarity_error': max(c['stationarity_error'] for c in certificates if not c['used_prior_fallback']),
                         'max_successful_pythagorean_error': max(c['pythagorean_error'] for c in certificates if not c['used_prior_fallback']),
                         'law_build_seconds': seconds}
    OUT.mkdir(parents=True)
    pd.DataFrame(certificates).to_csv(OUT / 'source_certificates.csv', index=False)
    result = {'assertion_failures': 0, 'cases': cases, 'independent_primal_checks': independent_checks,
              'proper_source_checks': source_checks,
              'script_sha256': sha(Path(__file__)),
              'helper_sha256': sha(ROOT / 'scripts/moment_protection.py'),
              'prediction_helper_sha256': sha(ROOT / 'scripts/moment_protected_cif.py'),
              'output_hashes': {'source_certificates.csv': sha(OUT / 'source_certificates.csv')},
              'scope': 'Synthetic and proper-source law construction only; no held-out CIF scoring. Protected moments do not fix the nonlinear CIF time law.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
