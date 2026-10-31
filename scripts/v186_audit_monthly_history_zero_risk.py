"""Refit every V186 forest and independently expand every hourly Brier score."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from extended_monthly_information import ExtendedMonthlyInformation
from hourly_fuel_data import HourlyFuelData
from monthly_history_features import MonthlyInformation
from monthly_history_zero_risk import METHODS, OLD_ALIASES, check_inputs
from monthly_risk_features import NAMES
from v151_hourly_law import sha
from v170_fuel_marginal_diagnosis import FUELS, verify
from v171_two_scale_zero import METRICS, aggregate
from v179_audit_increment_support import compare_frames
from v186_monthly_history_zero_risk import ROOT, BASE, OLD, OUT as SOURCE, make_predictor

OUT = ROOT / 'results/v186-monthly-history-zero-risk-audit-r1'


def audit_fold(data, short, long, fold: int) -> dict:
    predictor = make_predictor(data, short, long, fold)
    folder = SOURCE / f'fold_{fold}'
    queries = predictor.frozen.queries
    saved_queries = pd.read_csv(folder / 'query_inputs.csv')
    pd.testing.assert_frame_equal(queries, saved_queries, check_exact=True)
    if predictor.fingerprint() != json.loads((folder / 'model_fingerprints.json').read_text()):
        raise ValueError('refitted forest fingerprints differ')
    if predictor.provenance != json.loads((folder / 'training_provenance.json').read_text()):
        raise ValueError('training provenance differs')
    for name in NAMES:
        with np.load(folder / f'{name}_training.npz', allow_pickle=False) as stored:
            expected = {'X': predictor.X[name], 'target': predictor.zero, 'raw_id': predictor.raw_ids,
                        'profile_id': predictor.profiles, 'region': np.asarray(predictor.regions, str)}
            if set(stored.files) != set(expected) or any(not np.array_equal(stored[k], value) for k, value in expected.items()):
                raise ValueError('source feature/zero-label/sample inventory differs')
    for q, (_, contexts, _) in predictor.contexts.items():
        region = short.index.region.iloc[q]
        if region not in predictor.frozen.fit or not short.index.month.iloc[q].startswith('2024'):
            raise ValueError('source role/year violation')
        for monthly, context in zip((short, long), contexts):
            if (q in context['prefix'] or not (monthly.release.iloc[context['prefix']] <= monthly.dates.iloc[q]).all()
                    or any(monthly.index.region.iloc[p] != region for p in context['prefix'])):
                raise ValueError('source feature prefix violation')
    for q in queries.profile_id.iloc[[0, -1]]:
        check_inputs(predictor, int(q))
    with np.load(folder / 'probabilities.npz', allow_pickle=False) as saved:
        ids, probabilities = saved['profile_id'].copy(), saved['hourly'].copy()
    if (not np.array_equal(ids, queries.profile_id.to_numpy(int))
            or probabilities.shape != (len(queries), 10, 48, 8) or not np.isfinite(probabilities).all()
            or probabilities.min() < 0 or probabilities.max() > 1):
        raise ValueError('complete probability axes/domain')
    rows = pd.read_csv(folder / 'hourly_scores.csv')
    if not np.isfinite(rows[list(METRICS)]).all().all() or rows.duplicated(['profile_id', 'method', 'fuel']).any():
        raise ValueError('score finite/key inventory')
    rebuilt = []
    for number, rec in enumerate(queries.itertuples(index=False)):
        q = int(rec.profile_id)
        fresh = predictor.predict(q)
        if any(not np.array_equal(fresh[name], probabilities[number, m]) for m, name in enumerate(METHODS)):
            raise ValueError('full query probability refit differs')
        if rec.role != 'test':
            continue
        row = data.index.iloc[q]
        states = data.states[int(row.offset):int(row.stop)]
        observed = (data.raw[int(row.offset):int(row.stop)] == 0).astype(float)
        local = data.local(q, 'asof_30d')
        for f, fuel in enumerate(FUELS):
            identity = {'fold': fold, 'profile_id': q, 'region': rec.region, 'month': rec.month,
                        'hours': len(states), 'fuel': fuel, 'input_fuel_positive': bool(data.M[local, f] > 0),
                        'current_fuel_positive': bool(data.M[q, f] > 0)}
            for m, name in enumerate(METHODS):
                p, y = probabilities[number, m, states, f], observed[:, f]
                # Independent algebraic expansion; no production score_probability.
                rebuilt.append(identity | {'method': name, 'brier': float(np.mean(p * p - 2 * p * y + y)),
                                            'mean_forecast': float(np.sum(p) / len(p)),
                                            'observed_fraction': float(np.sum(y) / len(y)),
                                            'calibration_bias': float((np.sum(p) - np.sum(y)) / len(y))})
    rebuilt = pd.DataFrame(rebuilt)
    error = compare_frames(rebuilt, rows, ['profile_id', 'method', 'fuel'])
    if len(rebuilt) != int(queries.role.eq('test').sum()) * 80:
        raise ValueError('independent score denominator')
    # All archived controls are matched to the same query/calendar/fuel cohort.
    old = pd.read_csv(OLD / f'fold_{fold}/hourly_scores.csv')
    old.method = old.method.map(OLD_ALIASES)
    pair = rows.merge(old, on=['profile_id', 'method', 'fuel'], validate='one_to_one', suffixes=('_a', '_b'))
    if len(pair) != int(queries.role.eq('test').sum()) * 64:
        raise ValueError('old control score denominator')
    if max(float(np.max(abs(pair[m + '_a'] - pair[m + '_b']))) for m in METRICS) > 1e-12:
        raise ValueError('old control scores differ')
    print(f'audited V186 fold {fold}: {len(rebuilt)} independent Brier rows', flush=True)
    return {'fold': fold, 'forests_refitted': 32, 'query_forecasts_rebuilt': len(queries),
            'independent_score_rows': len(rebuilt), 'old_score_rows_reproduced': len(pair), 'max_score_error': error}


def comparisons(regional: pd.DataFrame) -> pd.DataFrame:
    pairs = [(a, b) for a in NAMES for b in NAMES if a != b]
    pairs += [(a, b) for a in NAMES for b in ('world_gaussian', 'conditional_activity_recomposition',
                                             'month_persistence_conditional', 'month_history_conditional')]
    rng = np.random.default_rng(186)
    rows = []
    for (cohort, fuel), table in regional.groupby(['cohort', 'fuel']):
        for method, reference in pairs:
            a, b = [table.loc[table.method.eq(name)].sort_values('region') for name in (method, reference)]
            if not np.array_equal(a.region, b.region) or a.region.duplicated().any() or not len(a):
                raise ValueError('paired source inventory differs')
            difference = a.brier.to_numpy() - b.brier.to_numpy()
            boot = difference[rng.integers(0, len(a), size=(10000, len(a)))].mean(axis=1)
            baseline = float(b.brier.mean())
            rows.append({'cohort': cohort, 'fuel': fuel, 'method': method, 'reference': reference, 'regions': len(a),
                         'brier': float(a.brier.mean()), 'reference_brier': baseline,
                         'paired_brier_difference': float(difference.mean()),
                         'brier_reduction_percent': float(-100 * difference.mean() / baseline) if baseline > 0 else None,
                         'region_bootstrap_lo': float(np.quantile(boot, .025)),
                         'region_bootstrap_hi': float(np.quantile(boot, .975)),
                         'regions_better_brier': int((difference < -1e-12).sum())})
    if len(rows) != 448:
        raise ValueError('comparison denominator')
    return pd.DataFrame(rows)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    summary = verify(SOURCE)
    hashes = dict(summary['input_hashes'])
    for name in ('summary.json', *summary['output_hashes']):
        path = SOURCE / name
        hashes[str(path.relative_to(ROOT))] = sha(path)
    for path in (Path(__file__), ROOT / 'scripts/v179_audit_increment_support.py'):
        hashes[str(path.relative_to(ROOT))] = sha(path)
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('audit input changed')
    data, short = HourlyFuelData(ROOT), MonthlyInformation(ROOT)
    long = ExtendedMonthlyInformation(ROOT, ROOT / 'results/v183-coarse-history-export-r1/monthly_history.csv')
    with threadpool_limits(limits=2):
        checks = [audit_fold(data, short, long, fold) for fold in range(6)]
    rows = pd.concat([pd.read_csv(SOURCE / f'fold_{f}/hourly_scores.csv') for f in range(6)], ignore_index=True)
    if len(rows) != 19280 or (SOURCE / 'hourly_scores.csv').read_text() != rows.to_csv(index=False):
        raise ValueError('root producer serialization differs')
    regional, macro = aggregate(rows)
    compare_frames(regional, pd.read_csv(SOURCE / 'hourly_region_metrics.csv'), ['cohort', 'region', 'method', 'fuel'])
    compare_frames(macro, pd.read_csv(SOURCE / 'hourly_macro_metrics.csv'), ['cohort', 'method', 'fuel'])
    comparison = comparisons(regional)
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('audit input changed during run')
    OUT.mkdir()
    for name, frame in [('reconstruction', pd.DataFrame(checks)), ('comparisons', comparison)]:
        frame.to_csv(OUT / f'{name}.csv', index=False)
    result = {'assertion_failures': 0, 'forests_refitted': 192, 'query_forecasts_rebuilt': 482,
              'independent_brier_rows': sum(c['independent_score_rows'] for c in checks),
              'old_score_rows_reproduced': sum(c['old_score_rows_reproduced'] for c in checks),
              'max_score_error': max(c['max_score_error'] for c in checks), 'future_query_raw_guards': 12,
              'comparisons': len(comparison), 'input_hashes': hashes,
              'output_hashes': {p.name: sha(p) for p in OUT.iterdir()},
              'scope': 'Full 192-forest refit and all probability/score rebuild; independent Brier expansion. '
                       'Feature/tree helpers shared; old six controls frozen short-history probabilities, not refitted here. '
                       'Paired region bootstrap describes repeated source development only; no CIF result.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if 'hash' not in k}, indent=2), flush=True)


if __name__ == '__main__':
    main()
