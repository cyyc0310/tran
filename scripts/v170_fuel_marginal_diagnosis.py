"""Fuel-level error diagnosis; current-month alignment is explicitly oracle."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from frozen_recomposition import FrozenRecomposition
from fuel_activity_forest import carrier
from hourly_fuel_data import HourlyFuelData
from v151_hourly_law import sha
from v154_hourly_strong_baselines import CompressedLaw

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/v169-feasible-copula-r1'
BASE = ROOT / 'results/v159-fuel-activity-source-pilot-r1'
OUT = ROOT / 'results/v170-fuel-marginal-diagnosis-r1'
FUELS = ('coal', 'gas', 'oil', 'nuclear', 'hydro', 'wind', 'solar', 'other')
METHODS = ('conditional_activity_recomposition', 'world_gaussian')
METRICS = ('crps', 'coverage', 'lower_miss', 'upper_miss', 'width', 'zero_brier',
           'predicted_zero_probability', 'actual_zero_fraction', 'predicted_mean', 'actual_mean', 'absolute_mean_error')


def verify(path):
    summary = json.loads((path / 'summary.json').read_text())
    if summary['assertion_failures']:
        raise ValueError('failed archive')
    for name, digest in summary['output_hashes'].items():
        if sha(path / name) != digest:
            raise ValueError('archive changed')
    return summary


def score(cells, actual, clock):
    low = np.asarray([law.quantile(.05) for law in cells])[clock]
    high = np.asarray([law.quantile(.95) for law in cells])[clock]
    mean = np.asarray([law.mean for law in cells])[clock]
    pzero = np.asarray([float(law.w[law.x == 0].sum()) for law in cells])[clock]
    losses = np.empty(len(actual))
    for h in np.unique(clock):
        take = clock == h
        losses[take] = cells[int(h)].crps(actual[take])
    lower, upper = actual < low, actual > high
    return {'crps': float(losses.mean()), 'coverage': float((~lower & ~upper).mean()),
            'lower_miss': float(lower.mean()), 'upper_miss': float(upper.mean()), 'width': float((high - low).mean()),
            'zero_brier': float(np.mean((pzero - (actual == 0))**2)),
            'predicted_zero_probability': float(pzero.mean()), 'actual_zero_fraction': float((actual == 0).mean()),
            'predicted_mean': float(mean.mean()), 'actual_mean': float(actual.mean()),
            'absolute_mean_error': float(abs(mean.mean() - actual.mean()))}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    upstream = verify(SOURCE)
    dependencies = [Path(__file__), ROOT / 'docs/V170_FUEL_MARGINAL_DIAGNOSIS_PROTOCOL.md',
                    SOURCE / 'summary.json', ROOT / 'scripts/frozen_recomposition.py',
                    ROOT / 'scripts/fuel_activity_forest.py', ROOT / 'scripts/hourly_fuel_data.py',
                    ROOT / 'scripts/v151_hourly_law.py', ROOT / 'scripts/v154_hourly_strong_baselines.py']
    dependencies.extend(SOURCE / name for name in upstream['output_hashes'])
    hashes = upstream['input_hashes'] | {str(p.relative_to(ROOT)): sha(p) for p in dependencies}
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('input changed')
    data = HourlyFuelData(ROOT)
    outputs = []
    maximum_mean_error = maximum_truth_error = 0.
    for fold in range(6):
        frozen = FrozenRecomposition(data, BASE, fold, 'asof_30d')
        worlds = pd.read_csv(SOURCE / f'fold_{fold}/world_probabilities.csv')
        queries = frozen.queries.loc[frozen.queries.role.eq('test')]
        for rec in queries.itertuples(index=False):
            q = int(rec.profile_id)
            local = data.local(q, 'asof_30d')
            if local != rec.input_profile_id or data.release.iloc[local] > data.dates.iloc[q]:
                raise ValueError('query monthly timing')
            row = data.index.iloc[q]
            clock = data.states[int(row.offset):int(row.stop)] % 24
            occupancy = np.bincount(clock, minlength=24) / len(clock)
            base = carrier(data.shares[local], 'asof_30d', frozen.prior)
            selected = worlds.loc[worlds.profile_id.eq(q)]
            donors = selected.source_profile_id.to_numpy(int)
            if (selected.source_profile_id.duplicated().any() or not selected.role.eq('test').all()
                    or not selected.region.eq(rec.region).all() or not selected.month.eq(rec.month).all()
                    or not set(data.index.region.iloc[donors]).issubset(frozen.fit)):
                raise ValueError('world source/role identity')
            expected = []
            for region in frozen.fit:
                ids = frozen.bank[region]
                distance = abs(frozen.month[ids] - frozen.month[q])
                distance = np.minimum(distance, 12 - distance)
                expected.extend(ids[np.lexsort((ids, distance))[:2]])
            if not np.array_equal(donors, expected):
                raise ValueError('world inventory differs')
            # Build both legal distributions before consulting any query hourly labels.
            prepared = [frozen.activity.donor(int(d)) for d in donors]
            predictions = {}
            for method, column in zip(METHODS, ('conditional_weight', 'gaussian_weight')):
                prior = selected[column].to_numpy(float)
                if (prior <= 0).any() or abs(prior.sum() - 1) > 1e-12:
                    raise ValueError('world probabilities')
                fuel_cells = [[] for _ in FUELS]
                for hour in range(24):
                    fuel = np.concatenate([d[hour] for d in prepared]) * base
                    weights = np.concatenate([np.full(len(d[hour]), p / len(d[hour])) for d, p in zip(prepared, prior)])
                    for j in range(8):
                        fuel_cells[j].append(CompressedLaw(fuel[:, j], weights))
                predictions[method] = fuel_cells
            scale = data.M[local].sum() / len(clock)
            actual = data.raw[int(row.offset):int(row.stop)] / scale
            true_mean = data.M[q] / data.M[local].sum()
            truth_error = float(np.max(abs(actual.mean(axis=0) - true_mean)))
            maximum_truth_error = max(maximum_truth_error, truth_error)
            if truth_error > 1e-10:
                raise ValueError('coarse/hourly truth normalization mismatch')
            for method, fuel_cells in predictions.items():
                for j, cells in enumerate(fuel_cells):
                    predicted_mean = float(occupancy @ np.asarray([law.mean for law in cells]))
                    if predicted_mean <= 0 and true_mean[j] > 0:
                        raise ValueError('positive target fuel absent from entire predictive support')
                    factor = float(true_mean[j] / predicted_mean) if predicted_mean > 0 else 0.
                    aligned = [CompressedLaw(law.x * factor, law.w) for law in cells]
                    mean_error = abs(occupancy @ np.asarray([law.mean for law in aligned]) - true_mean[j])
                    maximum_mean_error = max(maximum_mean_error, float(mean_error))
                    if mean_error > 1e-10:
                        raise ValueError('oracle monthly mean not aligned')
                    if factor > 0:
                        for a, b in zip(cells, aligned):
                            if abs(a.w[a.x == 0].sum() - b.w[b.x == 0].sum()) > 1e-12:
                                raise ValueError('positive scaling changed zero mass')
                    identity = {'fold': fold, 'profile_id': q, 'region': rec.region, 'month': rec.month,
                                'hours': len(actual), 'fuel': FUELS[j], 'method': method,
                                'current_fuel_positive': bool(true_mean[j] > 0),
                                'input_fuel_positive': bool(data.M[local, j] > 0)}
                    for arm, laws in (('legal', cells), ('oracle_month_mean', aligned)):
                        outputs.append(identity | {'arm': arm, 'is_oracle': arm != 'legal',
                                                    'oracle_factor': factor if arm != 'legal' else 1.,
                                                    **score(laws, actual[:, j], clock)})
        print(f'completed marginal diagnosis fold {fold}: {len(queries)} months', flush=True)
    rows = pd.DataFrame(outputs)
    if (len(rows) != 7712 or rows.duplicated(['profile_id', 'method', 'arm', 'fuel']).any()
            or not np.isfinite(rows[list(METRICS)]).all().all()):
        raise ValueError('score denominator or finite check')
    regional = []
    for cohort, subset in (('all29', rows), ('current_fuel_positive', rows.loc[rows.current_fuel_positive])):
        for key, group in subset.groupby(['region', 'method', 'arm', 'fuel']):
            regional.append({'cohort': cohort, **dict(zip(['region', 'method', 'arm', 'fuel'], key)),
                             **{m: float(np.average(group[m], weights=group.hours)) for m in METRICS}})
    regional = pd.DataFrame(regional)
    macro = pd.DataFrame([{'cohort': key[0], 'method': key[1], 'arm': key[2], 'fuel': key[3], 'regions': len(group),
                          **{m: float(group[m].mean()) for m in METRICS}, 'worst_coverage': float(group.coverage.min())}
                         for key, group in regional.groupby(['cohort', 'method', 'arm', 'fuel'])])
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('dependency changed during diagnosis')
    OUT.mkdir(parents=True)
    for name, frame in (('test_month_metrics', rows), ('region_metrics', regional), ('macro_metrics', macro)):
        frame.to_csv(OUT / f'{name}.csv', index=False)
    result = {'assertion_failures': 0, 'score_rows': len(rows), 'test_months': rows.profile_id.nunique(),
              'max_oracle_mean_error': maximum_mean_error, 'max_truth_mean_error': maximum_truth_error,
              'input_hashes': hashes, 'output_hashes': {p.name: sha(p) for p in sorted(OUT.iterdir())},
              'scope': 'Fuel-marginal diagnosis only. Current-month mean alignment is oracle. Raw fuel intervals are uncalibrated; no new CIF forecast, target guarantee, or innovation gate.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(macro.loc[macro.cohort.eq('current_fuel_positive') & macro.method.eq('world_gaussian')].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
