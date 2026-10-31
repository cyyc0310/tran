"""Numerical identities for the source activity pilot; no accuracy comparison."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from fuel_activity_forest import IndexedLeafMixture, activity_law, carrier, response_values, source_activity
from hourly_fuel_data import HourlyFuelData
from v151_hourly_law import EF, Law, sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/v159-preflight-activity-r1'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    dependencies = [Path(__file__), ROOT / 'scripts/fuel_activity_forest.py',
                    ROOT / 'scripts/hourly_fuel_data.py', ROOT / 'scripts/v151_hourly_law.py',
                    ROOT / 'scripts/v154_hourly_strong_baselines.py',
                    ROOT / 'results/v144-canonical-profile-cache-r1/profiles.npz',
                    ROOT / 'results/v144-canonical-profile-cache-r1/profile_index.csv',
                    ROOT / 'results/v153-calendar-times-r1/timestamps.npz',
                    ROOT / 'results/v156-anchored-hourly-sampled-r1/source_sampling.csv']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in dependencies}
    data = HourlyFuelData(ROOT)
    samples = pd.read_csv(dependencies[-1])
    samples = samples.loc[samples.target_region.eq('PACE') & samples.seed.eq(41)
                          & ~samples.region.isin(['PACE', 'LGEE', 'SC'])]
    prior = np.ones(8) / 8
    maximum_identity_error = 0.
    verified_rows = 0
    for row in samples.itertuples(index=False):
        allowed = np.asarray(json.loads(row.sampled_raw_hour_indices), int)
        source = source_activity(data, row.region, row.budget, allowed, prior)
        local = np.asarray([data.local(int(q), row.budget) for q in source['profiles']], int)
        values, positive = response_values(carrier(data.shares[local], row.budget, prior), source['u'])
        g = data.raw[allowed]
        actual = g @ EF / g.sum(axis=1)
        error = float(np.max(np.abs(values - actual)))
        maximum_identity_error = max(maximum_identity_error, error)
        if not positive.all() or error > 1e-9:
            raise ValueError('matching carrier source CIF reconstruction')
        projective, _ = response_values(carrier(data.shares[local], row.budget, prior),
                                        source['u'] / source['u'].sum(axis=1, keepdims=True))
        if not np.allclose(projective, values, rtol=0, atol=1e-9):
            raise ValueError('projective quotient invariance')
        verified_rows += len(allowed)
    # Positive generation and zero carbon are distinct events.
    B = np.array([.5, 0, 0, .5, 0, 0, 0, 0])
    u = np.zeros((3, 8))
    u[0, 0] = 1
    u[1, 3] = 1
    u[2, 1] = 1
    law, rejected = activity_law(B, u, np.array([.2, .3, .5]))
    if rejected != .5 or not np.allclose(law.mean, 400) or law.quantile(.5) != 0:
        raise ValueError('conditional mass/zero carbon support')
    try:
        activity_law(B, u[2:], np.ones(1))
    except ValueError as error:
        if str(error) != 'no positive-generation probability support':
            raise
    else:
        raise ValueError('empty positive support not rejected')
    rng = np.random.default_rng(159)
    X = rng.normal(size=(2048, 4))
    activity = np.exp(rng.normal(size=(2048, 8)))
    weights = rng.uniform(.1, 4, len(X))
    max_distribution_error = 0.
    with threadpool_limits(limits=2):
        model = IndexedLeafMixture(X, activity, weights)
        train_leaves = model.forest.apply(X)
        query_leaves = model.forest.apply(X[:11])
        for query, signature in enumerate(query_leaves):
            ids, mass = model.atoms(signature)
            if not np.allclose(np.average(activity[ids], axis=0, weights=mass),
                               model.forest.predict(X[query:query+1])[0], rtol=0, atol=1e-10):
                raise ValueError('weighted index mixture mean')
            values, positive = response_values(B, activity)
            if not positive.all():
                raise ValueError('positive toy support')
            expanded_values, expanded_weights = [], []
            for tree, leaf in enumerate(signature):
                selected = np.flatnonzero(train_leaves[:, tree] == leaf)
                expanded_values.extend(values[selected])
                expanded_weights.extend(weights[selected] / weights[selected].sum() / 32)
            expanded = Law(expanded_values, expanded_weights)
            compressed, _ = activity_law(B, activity[ids], mass)
            grid = np.linspace(0, 1000, 51)
            error = float(np.max(np.abs(expanded.crps(grid) - compressed.crps(grid))))
            max_distribution_error = max(max_distribution_error, error)
            if error > 1e-9 or any(abs(expanded.quantile(p) - compressed.quantile(p)) > 1e-10
                                   for p in (.05, .5, .95)):
                raise ValueError('weighted index mixture distribution')
    new_fuel = carrier(np.array([1., 0, 0, 0, 0, 0, 0, 0]), 'asof_30d', prior)
    if not (new_fuel > 0).all():
        raise ValueError('historical zero hard masked')
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('preflight dependencies changed')
    OUT.mkdir(parents=True)
    result = {'assertion_failures': 0, 'source_sample_records': len(samples),
              'source_hour_uses_verified': verified_rows, 'matching_carrier_max_error': maximum_identity_error,
              'forest_distribution_queries': len(query_leaves),
              'expanded_vs_indexed_crps_max_error': max_distribution_error,
              'positive_domain_zero_carbon_distinction': True, 'empty_positive_support_rejected': True,
              'asof_zero_activation_allowed': True, 'input_hashes': hashes,
              'scope': 'Algebraic/probability contracts, not predictive performance.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
