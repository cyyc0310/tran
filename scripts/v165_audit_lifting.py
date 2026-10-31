"""Rebuild V165 complete coarse-law lifting and compare same-cohort controls."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from calendar_cif_law import CalendarLaw
from frozen_recomposition import FrozenRecomposition
from hourly_fuel_data import HourlyFuelData
from level_spread_diagnostics import score_laws
from monthly_history_features import MonthlyInformation
from coarse_distribution_lifting import (CoarseLawForecast as CoarseForecast, METHODS,
                                         DistributionScenarioPrediction as ScenarioPrediction)
from v151_hourly_law import EF, sha
from v154_hourly_strong_baselines import CompressedLaw
from monthly_history_features import monthly_context
from v159_fuel_activity_source_pilot import weighted_radius

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/v165-coarse-distribution-lifting-r1'
BASE = ROOT / 'results/v159-fuel-activity-source-pilot-r1'
OTHER = ROOT / 'results/v160-carbon-cdf-source-r1'
OUT = ROOT / 'results/v165-coarse-lifting-audit-r2'
METRICS = ['crps', 'mae', 'coverage', 'width', 'winkler', 'tolerant_coverage',
           'lower_miss', 'upper_miss', 'support_below', 'support_above']
KEYS = ['region', 'month', 'method', 'calibration']


def verify_archive(path):
    summary = json.loads((path / 'summary.json').read_text())
    if summary['assertion_failures']:
        raise ValueError('upstream failed')
    for name, digest in summary['output_hashes'].items():
        if sha(path / name) != digest:
            raise ValueError('archive output changed')
    return summary


def regional(rows):
    records = []
    for key, group in rows.groupby(['region', 'method', 'calibration']):
        records.append(dict(zip(['region', 'method', 'calibration'], key))
                       | {m: float(np.average(group[m], weights=group.hours)) for m in METRICS[:5]})
    return pd.DataFrame(records)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    summary = verify_archive(SOURCE)
    verify_archive(BASE)
    verify_archive(OTHER)
    hashes = summary['input_hashes'] | {'scripts/v165_audit_lifting.py': sha(Path(__file__)),
                                      str(SOURCE.relative_to(ROOT) / 'summary.json'): sha(SOURCE / 'summary.json'),
                                      str(OTHER.relative_to(ROOT) / 'summary.json'): sha(OTHER / 'summary.json')}
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('input changed')
    rows = pd.read_csv(SOURCE / 'test_month_metrics.csv')
    if (len(rows) != 4338 or rows.duplicated(KEYS).any()
            or not np.isfinite(rows[METRICS + ['radius', 'hours']]).all().all()
            or not rows.budget.eq('asof_30d').all()):
        raise ValueError('score denominator')
    data, monthly = HourlyFuelData(ROOT), MonthlyInformation(ROOT)
    max_score_error, max_forecast_error, max_radius_error = 0., 0., 0.
    max_anchor_error, max_oof_error, rebuilt, cal_contexts = 0., 0., 0, 0
    coarse_records, backtest_records = [], []
    with threadpool_limits(limits=2):
        for fold in range(6):
            block = SOURCE / f'fold_{fold}'
            saved = json.loads((block / 'summary.json').read_text())
            frozen = FrozenRecomposition(data, BASE, fold, 'asof_30d')
            model = CoarseForecast(monthly, frozen.fit)
            predictor = ScenarioPrediction(frozen, model)
            if saved['proper_regions'] != frozen.fit or saved['coarse_train_profiles'] != model.ids.tolist():
                raise ValueError('coarse training cohort mismatch')
            oof = pd.read_csv(block / 'coarse_oof.csv').sort_values('profile_id').reset_index(drop=True)
            rebuilt_oof = pd.DataFrame(model.oof).sort_values('profile_id').reset_index(drop=True)
            for key in ['profile_id', 'region', 'fold']:
                if not np.array_equal(oof[key], rebuilt_oof[key]):
                    raise ValueError('coarse grouped OOF identity mismatch')
            max_oof_error = max(max_oof_error, float(np.max(abs(oof.error - rebuilt_oof.error))))
            forecasts = pd.read_csv(block / 'forecasts.csv').set_index('profile_id')
            updates = pd.read_csv(block / 'scenario_updates.csv').set_index(['profile_id', 'method'])
            radii = pd.read_csv(block / 'calibration_radii.csv').set_index(['method', 'calibration'])
            errors = {method: {} for method in METHODS}
            for rec in frozen.queries.itertuples(index=False):
                q = int(rec.profile_id)
                forecast, S, B, donors, weights, anchors = predictor.inputs(q)
                # Independently form each scenario's coarse energy ratio.
                direct = []
                for d in donors:
                    local = data.local(int(d), 'asof_30d')
                    past = .99 * data.shares[local] + .01 * frozen.prior
                    fuel = B * data.M[d] / past
                    direct.append(float(sum(fuel * EF) / sum(fuel)))
                max_anchor_error = max(max_anchor_error, float(np.max(abs(anchors - direct))))
                stored = forecasts.loc[q]
                if stored.role != rec.role or stored.region != rec.region or stored.input_profile_id != forecast['local']:
                    raise ValueError('query forecast identity mismatch')
                for key in ['raw_anchor', 'anchor', 'past_anchor', 'sigma']:
                    max_forecast_error = max(max_forecast_error, abs(stored[key] - forecast[key]))
                if stored.fallback != forecast['fallback']:
                    raise ValueError('coarse fallback mismatch')
                if not np.allclose(json.loads(stored.forecast_shares), forecast['shares'], atol=1e-12, rtol=0):
                    raise ValueError('forecast composition mismatch')
                laws, _, details = predictor.build(q)
                for detail in details:
                    archived = updates.loc[(q, detail['method'])]
                    for key in detail:
                        if key == 'method':
                            continue
                        if key == 'convex_hull_clipped':
                            if archived[key] != detail[key]:
                                raise ValueError('hull flag mismatch')
                        elif isinstance(detail[key], str):
                            if archived[key] != detail[key]:
                                raise ValueError('serialized coarse law/backtest mismatch')
                        elif abs(archived[key] - detail[key]) > 1e-7:
                            raise ValueError('scenario update mismatch')
                    if detail['method'] == 'world_entropy_cap' and detail['kl'] > .1 + 1e-10:
                        raise ValueError('KL cap exceeded')
                actual = data.truth(q)
                row = data.index.iloc[q]
                clock = data.states[int(row.offset):int(row.stop)] % 24
                for detail in details:
                    for identity in json.loads(detail['local_backtest_identities']):
                        p = identity['observed_profile_id']
                        if data.index.region.iloc[p] != rec.region or data.release.iloc[p] > data.dates.iloc[q]:
                            raise ValueError('local backtest outcome unavailable')
                        context = monthly_context(monthly, p)
                        if context is None or context['prefix'] != identity['feature_prefix'] or context['local'] != identity['prediction_input_profile_id']:
                            raise ValueError('historical feature prefix mismatch')
                        if any(data.release.iloc[t] > data.dates.iloc[p] for t in identity['feature_prefix']):
                            raise ValueError('historical feature not released')
                        backtest_records.append({'fold': fold, 'role': rec.role, 'query_profile_id': q,
                                                 'method': detail['method'], **identity})
                    values = json.loads(detail['coarse_values'])
                    if values:
                        law = CompressedLaw(values, json.loads(detail['coarse_weights']))
                        truth = float(data.shares[q] @ EF)
                        lower, upper = law.quantile(.05), law.quantile(.95)
                        coarse_records.append({'fold': fold, 'role': rec.role, 'profile_id': q,
                                               'region': rec.region, 'month': rec.month, 'method': detail['method'],
                                               'truth': truth, 'mean': law.mean, 'crps': float(law.crps(np.array([truth]))[0]),
                                               'coverage': float(lower <= truth <= upper), 'width': upper - lower,
                                               'lower': lower, 'upper': upper, 'local_backtests': detail['local_backtests'],
                                               'outside_hull_mass': detail['outside_hull_mass']})
                support = [np.unique(law.x) for law in laws['conditional_activity_recomposition']]
                for name in METHODS[2:]:
                    if any(not np.array_equal(np.unique(law.x), old) for law, old in zip(laws[name], support)):
                        raise ValueError('probability update changed CIF atoms')
                for method, cells in laws.items():
                    if rec.role == 'calibration':
                        law = CalendarLaw(cells, clock)
                        distance = np.maximum.reduce([law.quantile(.05) - actual,
                                                      actual - law.quantile(.95), np.zeros(len(actual))])
                        errors[method].setdefault(rec.region, []).append(distance)
                    else:
                        for calibration in ['pooled', 'max_region']:
                            old = rows.loc[rows.profile_id.eq(q) & rows.method.eq(method)
                                           & rows.calibration.eq(calibration)]
                            if (len(old) != 1 or old.fold.iloc[0] != fold or old.hours.iloc[0] != len(actual)
                                    or old.region.iloc[0] != row.region or old.month.iloc[0] != row.month
                                    or old.budget.iloc[0] != 'asof_30d'):
                                raise ValueError('test record identity')
                            radius = float(radii.loc[(method, calibration), 'radius'])
                            score = score_laws(cells, actual, clock, radius)
                            max_score_error = max(max_score_error, max(abs(score[m] - old[m].iloc[0]) for m in METRICS))
                            if abs(old.radius.iloc[0] - radius) > 1e-10:
                                raise ValueError('radius key mismatch')
                            rebuilt += 1
                cal_contexts += int(rec.role == 'calibration')
            for method in METHODS:
                new_radii = weighted_radius(errors[method])
                for label, value in zip(['pooled', 'max_region'], new_radii):
                    max_radius_error = max(max_radius_error, abs(value - radii.loc[(method, label), 'radius']))
    if max(max_score_error, max_forecast_error, max_radius_error, max_anchor_error, max_oof_error) > 1e-7:
        raise ValueError('rebuild exceeded tolerance')
    r = regional(rows)
    stored_regions = pd.read_csv(SOURCE / 'region_metrics.csv')
    join = r.merge(stored_regions, on=['region', 'method', 'calibration'], suffixes=('_a', '_b'), validate='one_to_one')
    if len(join) != 29 * len(METHODS) * 2:
        raise ValueError('regional denominator')
    for metric in METRICS[:5]:
        if max(abs(join[metric + '_a'] - join[metric + '_b'])) > 1e-8:
            raise ValueError('regional aggregation')
    macro = pd.read_csv(SOURCE / 'macro_metrics.csv')
    if len(macro) != len(METHODS) * 4 or not np.isfinite(macro[METRICS + ['worst_coverage']]).all().all():
        raise ValueError('macro denominator/finite check')
    for rec in macro.itertuples(index=False):
        subset = rows.loc[rows.method.eq(rec.method) & rows.calibration.eq(rec.calibration)]
        if rec.cohort == 'nonconstant24':
            subset = subset.loc[~subset.region.isin(['CHPD', 'DOPD', 'GVL', 'SCL', 'SPA'])]
        elif rec.cohort != 'all29':
            raise ValueError('unknown macro cohort')
        source_values = subset.groupby('region').apply(
            lambda g: pd.Series({m: np.average(g[m], weights=g.hours) for m in METRICS}), include_groups=False)
        if len(source_values) != rec.regions:
            raise ValueError('macro region denominator')
        for metric in METRICS:
            if abs(source_values[metric].mean() - getattr(rec, metric)) > 1e-8:
                raise ValueError('macro aggregation')
        if abs(source_values.coverage.min() - rec.worst_coverage) > 1e-10:
            raise ValueError('macro worst-region coverage')
    comparisons, cohort_keys = [], set(rows[['region', 'month', 'calibration', 'hours']].itertuples(index=False, name=None))
    old = pd.read_csv(BASE / 'test_month_metrics.csv')
    other = pd.read_csv(OTHER / 'test_month_metrics.csv')
    prior_controls = pd.read_csv(ROOT / 'results/v164-scenario-reweighting-r1/test_month_metrics.csv')
    refs = pd.concat([prior_controls.loc[prior_controls.method.isin(['forecast_affine', 'forecast_composition', 'world_entropy', 'world_entropy_cap'])],
                      old.loc[old.budget.eq('asof_30d') & old.method.isin(['scalar_forest'])],
                      other.loc[other.budget.eq('asof_30d') & other.method.eq('activity_mmd_forest')]], ignore_index=True)
    for _, group in refs.groupby('method'):
        if set(group[['region', 'month', 'calibration', 'hours']].itertuples(index=False, name=None)) != cohort_keys:
            raise ValueError('strong reference cohort differs')
    all_regional = pd.concat([r, regional(refs)], ignore_index=True)
    rng = np.random.default_rng(165)
    for cohort in ['all29', 'nonconstant24']:
        table = all_regional if cohort == 'all29' else all_regional.loc[~all_regional.region.isin(['CHPD', 'DOPD', 'GVL', 'SCL', 'SPA'])]
        for calibration in ['pooled', 'max_region']:
            subset = table.loc[table.calibration.eq(calibration)]
            for method in METHODS[2:]:
                a = subset.loc[subset.method.eq(method)].sort_values('region')
                for reference in METHODS[:3] + ('scalar_forest', 'activity_mmd_forest', 'forecast_composition', 'world_entropy', 'world_entropy_cap', 'global_residual_nearest', 'global_residual_barycentric', 'qrf_barycentric', 'hetero_history_barycentric'):
                    if method == reference:
                        continue
                    b = subset.loc[subset.method.eq(reference)].sort_values('region')
                    if not np.array_equal(a.region, b.region):
                        raise ValueError('paired source cohort mismatch')
                    difference = a.crps.to_numpy() - b.crps.to_numpy()
                    bootstrap = difference[rng.integers(len(a), size=(3000, len(a)))].mean(axis=1)
                    comparisons.append({'cohort': cohort, 'calibration': calibration, 'method': method,
                                        'reference': reference, 'crps': a.crps.mean(),
                                        'crps_reduction_percent': 100 * (1 - a.crps.mean() / b.crps.mean()),
                                        'paired_crps_difference': difference.mean(),
                                        'region_bootstrap_lo': np.quantile(bootstrap, .025),
                                        'region_bootstrap_hi': np.quantile(bootstrap, .975),
                                        'regions_better_crps': int((difference < -1e-9).sum()),
                                        'coverage': a.coverage.mean(), 'worst_coverage': a.coverage.min(),
                                        'width': a.width.mean(), 'width_change_percent': 100 * (a.width.mean() / b.width.mean() - 1),
                                        'winkler': a.winkler.mean(),
                                        'joint_gate': bool(a.coverage.mean() >= .9 and a.coverage.min() >= .85
                                                          and a.crps.mean() <= .95 * b.crps.mean()
                                                          and a.width.mean() <= b.width.mean() + 1e-9
                                                          and a.winkler.mean() < b.winkler.mean())})
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('dependency changed during audit')
    verify_archive(SOURCE)
    OUT.mkdir(parents=True)
    pd.DataFrame(comparisons).to_csv(OUT / 'comparisons.csv', index=False)
    coarse_frame = pd.DataFrame(coarse_records)
    if len(coarse_frame) != 482 * 5 or not np.isfinite(coarse_frame[['truth', 'mean', 'crps', 'coverage', 'width']]).all().all():
        raise ValueError('coarse auxiliary denominator/finite check')
    coarse_frame.to_csv(OUT / 'coarse_month_metrics.csv', index=False)
    pd.DataFrame(backtest_records).to_csv(OUT / 'local_backtest_identities.csv', index=False)
    coarse_regional = coarse_frame.groupby(['role', 'region', 'method'])[['crps', 'coverage', 'width']].mean().reset_index()
    coarse_macro = coarse_regional.groupby(['role', 'method']).agg(crps=('crps', 'mean'), coverage=('coverage', 'mean'), width=('width', 'mean'), regions=('region', 'size')).reset_index()
    coarse_macro.to_csv(OUT / 'coarse_macro_metrics.csv', index=False)
    result = {'assertion_failures': 0, 'reconstructed_test_score_rows': rebuilt,
              'reconstructed_calibration_contexts': cal_contexts, 'verified_macro_rows': len(macro),
              'max_score_error': max_score_error,
              'max_forecast_error': max_forecast_error, 'max_radius_error': max_radius_error,
              'max_independent_world_anchor_error': max_anchor_error, 'max_oof_error': max_oof_error,
              'input_hashes': hashes, 'output_hashes': {p.name: sha(p) for p in sorted(OUT.iterdir())},
              'scope': 'Full pipeline reconstruction uses the frozen model/law helpers, not a second independent scoring implementation. Region bootstrap is descriptive development evidence, not untouched-region confirmation.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    frame = pd.DataFrame(comparisons)
    print(frame.loc[frame.cohort.eq('all29') & frame.calibration.eq('pooled')
                    & frame.reference.isin(['conditional_activity_recomposition', 'scalar_forest'])].to_string(index=False))


if __name__ == '__main__':
    main()
