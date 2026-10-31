"""Reconstruct all V170 scores, with additional independent CRPS integration."""
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
from v170_fuel_marginal_diagnosis import BASE, FUELS, METHODS, METRICS, SOURCE, score, verify

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'results/v170-fuel-marginal-diagnosis-r1'
OUT = ROOT / 'results/v170-fuel-marginal-audit-r1'


def cdf_crps(law, observed):
    """Integrate (F(x)-1{x>=y})² over the union of support and observation."""
    grid = np.unique(np.r_[law.x, observed])
    cumulative = np.r_[0., np.cumsum(law.w)]
    indices = np.searchsorted(law.x, grid[:-1], side='right')
    return float(np.diff(grid) @ (cumulative[indices] - (grid[:-1] >= observed))**2)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    summary = verify(ARCHIVE)
    verify(SOURCE)
    hashes = summary['input_hashes'] | {str((ARCHIVE / 'summary.json').relative_to(ROOT)): sha(ARCHIVE / 'summary.json'),
                                      str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('input changed')
    rows = pd.read_csv(ARCHIVE / 'test_month_metrics.csv')
    keys = ['profile_id', 'method', 'arm', 'fuel']
    if (len(rows) != 7712 or rows.duplicated(keys).any() or not np.array_equal(rows.is_oracle, rows.arm.ne('legal'))
            or not np.isfinite(rows[list(METRICS) + ['hours', 'oracle_factor']]).all().all()):
        raise ValueError('score identity/finiteness')
    saved = rows.set_index(keys)
    data = HourlyFuelData(ROOT)
    rebuilt, independent = [], []
    maximum_score = maximum_crps = 0.
    for fold in range(6):
        frozen = FrozenRecomposition(data, BASE, fold, 'asof_30d')
        worlds = pd.read_csv(SOURCE / f'fold_{fold}/world_probabilities.csv')
        queries = frozen.queries.loc[frozen.queries.role.eq('test')]
        sampled = set(queries.profile_id.iloc[[0, -1]])
        for rec in queries.itertuples(index=False):
            q = int(rec.profile_id)
            row = data.index.iloc[q]
            clock = data.states[int(row.offset):int(row.stop)] % 24
            occupancy = np.bincount(clock, minlength=24) / len(clock)
            local = data.local(q, 'asof_30d')
            base = carrier(data.shares[local], 'asof_30d', frozen.prior)
            selected = worlds.loc[worlds.profile_id.eq(q)]
            prepared = [frozen.activity.donor(int(d)) for d in selected.source_profile_id]
            # Legal source atoms are constructed independently of query hourly values.
            law_bank = {}
            for method, column in zip(METHODS, ('conditional_weight', 'gaussian_weight')):
                per_fuel = [[] for _ in FUELS]
                for h in range(24):
                    fuel = np.concatenate([cell[h] for cell in prepared]) * base
                    weights = np.concatenate([np.repeat(p / len(cell[h]), len(cell[h]))
                                              for p, cell in zip(selected[column], prepared)])
                    for j in range(8):
                        per_fuel[j].append(CompressedLaw(fuel[:, j], weights))
                law_bank[method] = per_fuel
            actual = data.raw[int(row.offset):int(row.stop)] / (data.M[local].sum() / len(clock))
            true_mean = data.M[q] / data.M[local].sum()
            for method, per_fuel in law_bank.items():
                for j, cells in enumerate(per_fuel):
                    predicted_mean = occupancy @ np.asarray([law.mean for law in cells])
                    if predicted_mean == 0 and true_mean[j] > 0:
                        raise ValueError('missing predictive positive support')
                    factor = float(true_mean[j] / predicted_mean) if predicted_mean > 0 else 0.
                    aligned = [CompressedLaw(law.x * factor, law.w) for law in cells]
                    for arm, laws in (('legal', cells), ('oracle_month_mean', aligned)):
                        old = saved.loc[(q, method, arm, FUELS[j])]
                        if (old.region != rec.region or old.month != rec.month or old.fold != fold
                                or old.hours != len(clock) or old.current_fuel_positive != (true_mean[j] > 0)
                                or old.input_fuel_positive != (data.M[local, j] > 0)):
                            raise ValueError('query score identity')
                        new = score(laws, actual[:, j], clock)
                        delta = np.asarray([new[key] - old[key] for key in METRICS])
                        if not np.isfinite(delta).all():
                            raise ValueError('nonfinite score delta')
                        maximum_score = max(maximum_score, float(np.max(abs(delta))))
                        if abs(old.oracle_factor - (1. if arm == 'legal' else factor)) > 1e-9 * max(1., factor):
                            raise ValueError('oracle scale mismatch')
                        if arm != 'legal' and abs(new['predicted_mean'] - true_mean[j]) > 1e-10:
                            raise ValueError('oracle moment mismatch')
                        rebuilt.append({**dict(zip(keys, (q, method, arm, FUELS[j]))),
                                        'region': rec.region, 'hours': len(clock),
                                        'current_fuel_positive': bool(true_mean[j] > 0), **new})
                        if q in sampled:
                            for hour_id in (0, len(clock) - 1):
                                law = laws[int(clock[hour_id])]
                                y = actual[hour_id, j]
                                independent_crps = cdf_crps(law, y)
                                direct = float(law.crps(np.array([y]))[0])
                                error = abs(independent_crps - direct)
                                maximum_crps = max(maximum_crps, error)
                                independent.append({'profile_id': q, 'method': method, 'arm': arm,
                                                    'fuel': FUELS[j], 'hour_id': hour_id,
                                                    'cdf_integral_crps': independent_crps,
                                                    'frozen_helper_crps': direct, 'absolute_error': error})
        print(f'audited marginal fold {fold}: {len(queries)} months', flush=True)
    rebuilt = pd.DataFrame(rebuilt)
    if len(rebuilt) != len(rows) or maximum_score > 1e-9 or maximum_crps > 1e-9:
        raise ValueError('score reconstruction or independent CRPS integration')
    regional = []
    for cohort, table in (('all29', rebuilt), ('current_fuel_positive', rebuilt.loc[rebuilt.current_fuel_positive])):
        for key, group in table.groupby(['region', 'method', 'arm', 'fuel']):
            regional.append({'cohort': cohort, **dict(zip(['region', 'method', 'arm', 'fuel'], key)),
                             **{m: float(np.average(group[m], weights=group.hours)) for m in METRICS}})
    regional = pd.DataFrame(regional)
    macro = pd.DataFrame([{'cohort': key[0], 'method': key[1], 'arm': key[2], 'fuel': key[3], 'regions': len(group),
                          **{m: float(group[m].mean()) for m in METRICS}, 'worst_coverage': float(group.coverage.min())}
                         for key, group in regional.groupby(['cohort', 'method', 'arm', 'fuel'])])
    for name, new, identity in [('region_metrics', regional, ['cohort', 'region', 'method', 'arm', 'fuel']),
                                 ('macro_metrics', macro, ['cohort', 'method', 'arm', 'fuel'])]:
        old = pd.read_csv(ARCHIVE / f'{name}.csv')
        pair = new.merge(old, on=identity, suffixes=('_new', '_old'), validate='one_to_one')
        numeric = list(METRICS) + (['worst_coverage', 'regions'] if name == 'macro_metrics' else [])
        if (len(pair) != len(new) or len(pair) != len(old) or not np.isfinite(old[numeric]).all().all()
                or any(max(abs(pair[m + '_new'] - pair[m + '_old'])) > 1e-9 for m in numeric)):
            raise ValueError('region or macro aggregation')
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('input changed during audit')
    verify(ARCHIVE)
    OUT.mkdir(parents=True)
    pd.DataFrame(independent).to_csv(OUT / 'independent_crps_checks.csv', index=False)
    result = {'assertion_failures': 0, 'rebuilt_score_rows': len(rebuilt), 'independent_crps_checks': len(independent),
              'max_score_error': maximum_score, 'max_independent_crps_error': maximum_crps,
              'input_hashes': hashes, 'output_hashes': {p.name: sha(p) for p in OUT.iterdir()},
              'scope': 'All marginal scores rebuilt with frozen scoring helper; selected pointwise CRPS additionally checked by independent CDF integration. Oracle current-month alignment remains diagnostic only.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('input_hashes', 'output_hashes')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
