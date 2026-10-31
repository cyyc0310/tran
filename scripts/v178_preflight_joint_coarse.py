"""Numerical, data-access and kernel checks without held-out outcome scoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from frozen_recomposition import FrozenRecomposition
from hourly_fuel_data import HourlyFuelData
from joint_coarse_worlds import (JointCoarsePrediction, MonthlyWorldForests, rff,
                                 same_ef_projection, FORESTS, SETTINGS)
from monthly_history_features import MonthlyInformation
from v151_hourly_law import EF, sha
from v170_fuel_marginal_diagnosis import verify

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/v159-fuel-activity-source-pilot-r1'
PROTOCOL = ROOT / 'docs/V178_JOINT_COARSE_WORLD_PROTOCOL.md'


def toys():
    rng = np.random.default_rng(20261029)
    z = rng.normal(size=(12, 8))
    phi, bandwidth, frequency = rff(z)
    independent = np.asarray([[np.mean(np.cos((a - b) @ frequency)) for b in z] for a in z])
    gram_error = float(np.max(abs(phi @ phi.T - independent)))
    if gram_error > 1e-12 or bandwidth <= 0:
        raise ValueError('RFF Gram identity failed')
    comp = rng.dirichlet(np.ones(8), size=3)
    actual = rng.dirichlet(np.ones(8))
    w = np.asarray([.2, .3, .5])
    projected = same_ef_projection(comp)
    if max(abs(projected.sum(axis=1))) > 1e-12 or max(abs(projected @ EF)) > 1e-10:
        raise ValueError('same-EF projection changed total/EF')
    predictor = JointCoarsePrediction.__new__(JointCoarsePrediction)
    predictor.monthly = SimpleNamespace(shares=np.stack([comp[0], actual]),
                                        index=pd.DataFrame({'month': ['2024-01', '2024-02'], 'hours': [744, 696]}))
    predictor.frozen = SimpleNamespace(prior=np.full(8, .125))
    predictor.support = lambda profiles, B: comp.copy()
    forecast = {'local': 0, 'shares': comp[1]}
    detail = {'method': 'toy', 'source_profiles': json.dumps([0, 1, 2]), 'weights': json.dumps(w.tolist())}
    rows, fuels = predictor.score_coarse(1, [detail], forecast)
    expected = sum(w[i] * np.linalg.norm(comp[i] - actual) for i in range(3))
    expected -= .5 * sum(w[i] * w[j] * np.linalg.norm(comp[i] - comp[j]) for i in range(3) for j in range(3))
    energy_error = abs(rows[0]['energy_score'] - expected)
    crps_error = 0.
    for f in range(8):
        wanted = sum(w[i] * abs(comp[i, f] - actual[f]) for i in range(3))
        wanted -= .5 * sum(w[i] * w[j] * abs(comp[i, f] - comp[j, f]) for i in range(3) for j in range(3))
        crps_error = max(crps_error, abs(wanted - fuels[f]['crps']))
    if max(energy_error, crps_error) > 1e-12:
        raise ValueError('independent discrete coarse score differs')
    # Same coarse EF need not mean the same hourly CIF distribution.
    nuclear = np.zeros((2, 8))
    wind = nuclear.copy()
    nuclear[:, 0] = wind[:, 0] = 1
    nuclear[:, 3] = 1
    wind[:, 5] = [0, 2]
    anchor1, anchor2 = [float(a.sum(axis=0) @ EF / a.sum()) for a in (nuclear, wind)]
    if anchor1 != 500 or anchor2 != 500:
        raise ValueError('same-EF coarse toy failed')
    cif1, cif2 = [a @ EF / a.sum(axis=1) for a in (nuclear, wind)]
    if np.array_equal(cif1, cif2):
        raise ValueError('coarse/hourly ambiguity toy lost')
    return {'RFF_Gram_error': gram_error, 'independent_energy_error': energy_error,
            'independent_marginal_CRPS_error': crps_error, 'same_coarse_EF': anchor1,
            'nuclear_hour_CIF': cif1.tolist(), 'wind_hour_CIF': cif2.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default='r1')
    args = parser.parse_args()
    out = ROOT / f'results/v178-joint-coarse-preflight-{args.version}'
    if out.exists():
        raise FileExistsError(out)
    upstream = verify(BASE)
    paths = [Path(__file__), PROTOCOL, ROOT / 'scripts/joint_coarse_worlds.py',
             ROOT / 'scripts/cif_pit_recalibration.py', ROOT / 'scripts/portfolio_activity.py',
             ROOT / 'scripts/frozen_recomposition.py', ROOT / 'scripts/coarse_distribution_lifting.py',
             ROOT / 'scripts/scenario_reweighting.py', ROOT / 'scripts/monthly_history_features.py',
             ROOT / 'scripts/hourly_fuel_data.py', BASE / 'summary.json']
    paths.extend(BASE / name for name in upstream['output_hashes'])
    hashes = upstream['input_hashes'] | {str(p.relative_to(ROOT)): sha(p) for p in paths}
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('preflight inputs changed')
    toy = toys()
    data, monthly = HourlyFuelData(ROOT), MonthlyInformation(ROOT)
    records, metadata = [], {}
    kernel_error = 0.
    with threadpool_limits(limits=2):
        for fold in range(6):
            frozen = FrozenRecomposition(data, BASE, fold, 'asof_30d')
            predictor = JointCoarsePrediction(frozen, monthly)
            models = predictor.month_models
            if hasattr(monthly, 'raw'):
                raise ValueError('coarse model unexpectedly has hourly data')
            metadata[str(fold)] = models.metadata
            for q in frozen.queries.profile_id.iloc[[0, -1]]:
                q = int(q)
                before, context = models.probabilities(q)
                for key in FORESTS:
                    feature = 'history' if key.startswith('history_') else 'base'
                    response = key[len(feature) + 1:]
                    independent = before[key] @ models.targets[response]
                    predicted = models.models[key].predict(context[feature][None])[0]
                    kernel_error = max(kernel_error, float(max(abs(independent - predicted))))
                # Temporarily remove the hourly array, without examining held values.
                original_raw = data.raw
                try:
                    data.raw = None
                    laws, _, details = predictor.build(q)
                finally:
                    data.raw = original_raw
                if len(laws) != 14 or len(details) != 14 or len(SETTINGS) != 12:
                    raise ValueError('preflight method inventory')
                ids = monthly.groups[monthly.index.region.iloc[q]]
                ids = ids[(monthly.release.iloc[ids] > monthly.dates.iloc[q]).to_numpy()]
                saved = monthly.M[ids].copy(), monthly.shares[ids].copy()
                changed = np.random.default_rng(q).uniform(1, 1000, size=saved[0].shape)
                try:
                    monthly.M[ids] = changed
                    monthly.shares[ids] = changed / changed.sum(axis=1, keepdims=True)
                    fresh = MonthlyWorldForests(monthly, models.ids, frozen.fit, frozen.prior)
                    after, _ = fresh.probabilities(q)
                finally:
                    monthly.M[ids], monthly.shares[ids] = saved
                if any(not np.array_equal(before[key], after[key]) for key in FORESTS):
                    raise ValueError('fresh coarse model used unreleased query statistics')
                records.append({'fold': fold, 'profile_id': q, 'worlds': len(models.ids),
                                'kernel_error': kernel_error, 'fine_array_access_guard': True,
                                'fresh_future_mutation_equal': True})
            print(f'joint coarse preflight fold {fold}: {len(models.ids)} monthly worlds', flush=True)
    if kernel_error > 1e-12 or len(records) != 12:
        raise ValueError('kernel reproduction or preflight denominator')
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('preflight inputs changed during run')
    out.mkdir(parents=True)
    pd.DataFrame(records).to_csv(out / 'query_checks.csv', index=False)
    (out / 'model_metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    result = {'assertion_failures': 0, 'source_forests_fitted': 60, 'fresh_mutation_refits': 120,
              'fine_access_guard_cases': 12, 'fresh_future_mutation_cases': 12,
              'max_kernel_reproduction_error': kernel_error, 'toy': toy, 'held_CIF_scored': False,
              'input_hashes': hashes, 'output_hashes': {p.name: sha(p) for p in out.iterdir()}}
    (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if 'hash' not in k}, indent=2))


if __name__ == '__main__':
    main()
