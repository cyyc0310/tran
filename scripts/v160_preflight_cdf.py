"""Finite CDF/RFF split identities, without fitting or evaluating a forecast."""
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
from threadpoolctl import threadpool_limits

from v151_hourly_law import sha
from v160_carbon_cdf_source import split_labels

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/v160-cdf-preflight-r1'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    inputs = [Path(__file__), ROOT / 'scripts/v160_carbon_cdf_source.py',
              ROOT / 'scripts/fuel_activity_forest.py', ROOT / 'scripts/v159_fuel_activity_source_pilot.py',
              ROOT / 'scripts/v151_hourly_law.py', ROOT / 'scripts/v139_physical_reserve_calibration.py']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    rng = np.random.default_rng(160)
    n = 512
    activity = np.exp(rng.normal(size=(n, 8)))
    scalar = rng.normal(0, 150, n)
    weights = rng.uniform(.1, 4, n)
    data = SimpleNamespace(shares=rng.dirichlet(np.ones(8), 5))
    prior = np.ones(8) / 8
    with threadpool_limits(limits=2):
        labels, meta = split_labels(data, activity, scalar, weights, list(range(5)), prior, 0, 'same_month')
        scaled, _ = split_labels(data, activity * np.exp(rng.normal(size=(n, 1))), scalar,
                                 weights, list(range(5)), prior, 0, 'same_month')
        invariance = {name: float(np.max(np.abs(value - scaled[name]))) for name, value in labels.items()}
        if max(invariance.values()) > 1e-10:
            raise ValueError('positive ray invariance')
        for name in ('scalar_cdf_forest', 'carbon_cdf_forest', 'activity_mmd_forest'):
            if labels[name].shape != (n, 80):
                raise ValueError('feature dimension budget')
        mask = np.arange(n) % 2 == 0
        left = weights[mask] / weights[mask].sum()
        right = weights[~mask] / weights[~mask].sum()
        errors = {}
        for name in ('scalar_cdf_forest', 'carbon_cdf_forest', 'activity_mmd_forest'):
            phi = labels[name]
            mean = np.average(phi, axis=0, weights=weights)
            ml = left @ phi[mask]
            mr = right @ phi[~mask]
            parent = np.sum(weights[:, None] * (phi - mean)**2) / weights.sum()
            child = (np.sum(weights[mask, None] * (phi[mask] - ml)**2)
                     + np.sum(weights[~mask, None] * (phi[~mask] - mr)**2)) / weights.sum()
            coefficient = weights[mask].sum() * weights[~mask].sum() / weights.sum()**2
            gain = coefficient * np.sum((ml - mr)**2)
            if name == 'activity_mmd_forest':
                gram_ll = phi[mask] @ phi[mask].T
                gram_rr = phi[~mask] @ phi[~mask].T
                gram_lr = phi[mask] @ phi[~mask].T
                explicit = left @ gram_ll @ left + right @ gram_rr @ right - 2 * left @ gram_lr @ right
                error = max(abs(parent - child - gain), abs(explicit - np.sum((ml - mr)**2)))
            else:
                # Means of indicator labels are the exact weighted empirical CDFs at the grid.
                error = abs(parent - child - gain)
            errors[name] = float(error)
        if max(errors.values()) > 1e-10:
            raise ValueError('split distribution metric identity')
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('preflight code changed')
    OUT.mkdir(parents=True)
    result = {'assertion_failures': 0, 'toy_rows': n, 'split_identity_errors': errors,
              'positive_ray_invariance_errors': invariance, 'matched_distribution_feature_dimension': 80,
              'gaussian_bandwidth': meta['gaussian_bandwidth'], 'input_hashes': hashes,
              'scope': 'Finite-feature probability algebra only; no predictive accuracy or novelty evidence.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
