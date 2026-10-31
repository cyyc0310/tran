"""Algebraic and source-law tests for matched source portfolio controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from frozen_recomposition import FrozenRecomposition
from hourly_fuel_data import HourlyFuelData
from monthly_history_features import MonthlyInformation
from portfolio_activity import PortfolioPrediction, portfolio_cells, permutation, MATCHED, SETTINGS
from v151_hourly_law import EF, sha
from v154_hourly_strong_baselines import CompressedLaw
from v170_fuel_marginal_diagnosis import verify

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/v159-fuel-activity-source-pilot-r1'
INVENTORY = ROOT / 'results/v176-aggregation-inventory-r1'
PROTOCOL = ROOT / 'docs/V177_PORTFOLIO_VALUE_PROTOCOL.md'


def toys():
    rng = np.random.default_rng(20261029)
    states = np.tile(np.arange(48), 5)
    g = rng.integers(0, 1000, size=(len(states), 2, 8)).astype(float)
    past = rng.uniform(1, 1000, size=(2, 8))
    prior = np.full(8, 1 / 8)
    cells, error, _ = portfolio_cells(g, past, states, prior, ['A', 'B'], '2024-06')
    B = .99 * past.sum(axis=0) / past.sum() + .01 * prior
    max_error = error
    for h in range(24):
        take = states % 24 == h
        expected = g[take].sum(axis=1).mean(axis=0) / past.sum()
        for mode in MATCHED:
            u, _, w = cells[mode][h]
            max_error = max(max_error, float(max(abs((w @ u) * B - expected))))
            fuel = u * B
            x = fuel @ EF / fuel.sum(axis=1)
            law = CompressedLaw(x, w)
            duplicated = CompressedLaw(np.repeat(x, 2), np.repeat(w / 2, 2))
            np.testing.assert_array_equal(law.x, duplicated.x)
            np.testing.assert_allclose(law.w, duplicated.w, atol=1e-15, rtol=0)
    for seed in (1, 2):
        order = permutation(['A', 'B'], '2024-06', states, seed)
        for state in range(48):
            ids = np.flatnonzero(states == state)
            np.testing.assert_array_equal(np.sort(order[ids]), ids)
            np.testing.assert_array_equal(g[order[ids], 1].sum(axis=0), g[ids, 1].sum(axis=0))
    # A scalar scaling does not change CIF; averaging CIFs is generally wrong.
    a, b = np.zeros(8), np.zeros(8)
    a[0], b[1] = 1, 9
    aggregate = float((a + b) @ EF / (a + b).sum())
    average_CIF = .5 * (float(a @ EF / a.sum()) + float(b @ EF / b.sum()))
    if aggregate != 460 or average_CIF != 700:
        raise ValueError('nonlinear CIF toy failed')
    if max_error > 1e-12:
        raise ValueError('independent portfolio means differ')
    return {'max_independent_mean_error': max_error, 'aggregate_CIF': aggregate,
            'arithmetic_average_CIF': average_CIF, 'joint_permutation_and_duplicate_atoms': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default='r1')
    args = parser.parse_args()
    out = ROOT / f'results/v177-portfolio-preflight-{args.version}'
    if out.exists():
        raise FileExistsError(out)
    hashes = {}
    archives = [BASE, INVENTORY]
    for directory in archives:
        saved = verify(directory)
        hashes |= saved['input_hashes']
        for name in ['summary.json', *saved['output_hashes']]:
            hashes[str((directory / name).relative_to(ROOT))] = sha(directory / name)
    dependencies = [Path(__file__), PROTOCOL, ROOT / 'scripts/portfolio_activity.py',
                    ROOT / 'scripts/cif_pit_recalibration.py', ROOT / 'scripts/coarse_distribution_lifting.py',
                    ROOT / 'scripts/scenario_reweighting.py', ROOT / 'scripts/frozen_recomposition.py',
                    ROOT / 'scripts/monthly_history_features.py', ROOT / 'scripts/fuel_activity_forest.py',
                    ROOT / 'scripts/hourly_fuel_data.py']
    hashes |= {str(p.relative_to(ROOT)): sha(p) for p in dependencies}
    for directory in archives:
        prior = json.loads((directory / 'summary.json').read_text())['input_hashes']
        if any(hashes[p] != h for p, h in prior.items()):
            raise ValueError('upstream dependency disagreement')
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('preflight input changed')
    toy = toys()
    data, monthly = HourlyFuelData(ROOT), MonthlyInformation(ROOT)
    records, proofs, scale = [], [], []
    started = time.monotonic()
    with threadpool_limits(limits=2):
        for fold in range(6):
            frozen = FrozenRecomposition(data, BASE, fold, 'asof_30d')
            predictor = PortfolioPrediction(frozen, monthly, INVENTORY, ROOT).prepare(fold)
            proofs.extend(predictor.source_proofs)
            scale.append(predictor.scale_error)
            for q in frozen.queries.profile_id.iloc[[0, -1]]:
                laws, _, details = predictor.build(int(q))
                if len(details) != len(SETTINGS):
                    raise ValueError('portfolio preflight methods')
                for name, cells in laws.items():
                    if len(cells) != 24 or any(not np.isfinite(c.w).all() or not (c.w > 0).all() for c in cells):
                        raise ValueError('preflight physical law')
                records.extend({'fold': fold, 'profile_id': int(q), **d} for d in details)
            print(f'portfolio preflight fold {fold}: {len(predictor.worlds)} source pair-months', flush=True)
    if len(proofs) != 5338 or len(records) != 144:
        raise ValueError('portfolio preflight inventory')
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('preflight inputs changed during run')
    out.mkdir(parents=True)
    pd.DataFrame(proofs).to_csv(out / 'source_worlds.csv', index=False)
    pd.DataFrame(records).to_csv(out / 'query_details.csv', index=False)
    result = {'assertion_failures': 0, 'source_pair_months': len(proofs), 'query_method_records': len(records),
              'max_source_matched_moment_error': max(p['matched_moment_error'] for p in proofs),
              'max_query_matched_moment_error': max(r['matched_moment_error'] for r in records),
              'max_proportional_activity_error': max(scale), 'toy': toy, 'seconds': time.monotonic() - started,
              'held_CIF_scored': False, 'input_hashes': hashes,
              'output_hashes': {p.name: sha(p) for p in out.iterdir()},
              'scope': 'Source construction and legal-query predictive laws only; no held hourly outcomes scored.'}
    (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if 'hash' not in k}, indent=2))


if __name__ == '__main__':
    main()
