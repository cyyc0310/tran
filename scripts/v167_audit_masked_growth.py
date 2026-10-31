"""Rebuild V167 missing-growth controls, calibration and matched references."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from calendar_cif_law import CalendarLaw
from masked_monthly_growth import MaskedMonthlyGrowth, METHODS, REFERENCE
from monthly_history_features import MonthlyInformation
from v167_masked_monthly_growth import check_mutation
from frozen_recomposition import FrozenRecomposition
from hourly_fuel_data import HourlyFuelData
from level_spread_diagnostics import score_laws
from v151_hourly_law import sha
from v159_fuel_activity_source_pilot import weighted_radius

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/v167-masked-monthly-growth-r1'
BASE = ROOT / 'results/v159-fuel-activity-source-pilot-r1'
OUT = ROOT / 'results/v167-masked-growth-audit-r1'
METRICS = ['crps', 'mae', 'coverage', 'width', 'winkler', 'tolerant_coverage',
           'lower_miss', 'upper_miss', 'support_below', 'support_above']


def verify(path):
    summary = json.loads((path / 'summary.json').read_text())
    if summary['assertion_failures']:
        raise ValueError('upstream failed')
    for name, digest in summary['output_hashes'].items():
        if sha(path / name) != digest:
            raise ValueError('archive changed')
    return summary


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    summary = verify(SOURCE)
    hashes = summary['input_hashes'] | {str(SOURCE.relative_to(ROOT) / 'summary.json'): sha(SOURCE / 'summary.json'),
                                      'scripts/v167_audit_masked_growth.py': sha(Path(__file__))}
    reference_sources = {'scalar_forest': BASE, 'activity_mmd_forest': ROOT / 'results/v160-carbon-cdf-source-r1',
                         'forecast_composition': ROOT / 'results/v164-scenario-reweighting-r1',
                         'world_entropy': ROOT / 'results/v164-scenario-reweighting-r1',
                         'hetero_history_barycentric': ROOT / 'results/v165-coarse-distribution-lifting-r1'}
    for directory in set(reference_sources.values()):
        verify(directory)
        for name in ['summary.json', 'test_month_metrics.csv']:
            hashes[str((directory / name).relative_to(ROOT))] = sha(directory / name)
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('input changed')
    rows = pd.read_csv(SOURCE / 'test_month_metrics.csv')
    if len(rows) != 6266 or rows.duplicated(['profile_id', 'method', 'radius_policy', 'calibration']).any():
        raise ValueError('score denominator')
    if not np.isfinite(rows[METRICS + ['radius', 'hours']]).all().all() or not rows.budget.eq('asof_30d').all():
        raise ValueError('score finite/budget')
    if not np.array_equal(rows.is_oracle, rows.method.str.startswith('oracle_')):
        raise ValueError('oracle flag mismatch')
    data = HourlyFuelData(ROOT)
    max_score, max_context, max_radius, rebuilt, contexts_rebuilt = 0., 0., 0., 0, 0
    enriched = []
    for fold in range(6):
        f = FrozenRecomposition(data, BASE, fold, 'asof_30d')
        monthly = MonthlyInformation(ROOT)
        diagnostic = MaskedMonthlyGrowth(f, monthly)
        for q in f.queries.profile_id.iloc[[0, -1]]:
            check_mutation(diagnostic, data, monthly, int(q))
        block = SOURCE / f'fold_{fold}'
        saved_context = pd.read_csv(block / 'contexts.csv').set_index('profile_id')
        saved_world = pd.read_csv(block / 'donor_exposures.csv').set_index(['profile_id', 'donor_profile_id'])
        saved_sensitivity = pd.read_csv(block / 'carrier_sensitivity.csv').set_index(['profile_id', 'method'])
        saved_radii = pd.read_csv(block / 'calibration_radii.csv').set_index(['method', 'calibration'])
        if not np.isfinite(saved_radii.radius).all() or set(saved_context.index) != set(f.queries.profile_id):
            raise ValueError('context/radius denominator')
        errors = {name: {} for name in METHODS}
        for rec in f.queries.itertuples(index=False):
            q = int(rec.profile_id)
            laws, context, worlds, sensitivity = diagnostic.build(q)
            saved = saved_context.loc[q]
            if saved.role != rec.role or saved.region != rec.region or saved.month != rec.month or saved.fold != fold:
                raise ValueError('context identity')
            values = np.array(list(context.values()), float)
            stored_values = np.array([saved[key] for key in context], float)
            if not np.isfinite(stored_values).all():
                raise ValueError('context nonfinite')
            max_context = max(max_context, float(np.max(abs(values - stored_values))))
            missing_both = 0.
            for world in worlds:
                archived = saved_world.loc[(q, world['donor_profile_id'])]
                if archived.role != rec.role or archived.region != rec.region or archived.month != rec.month or archived.fold != fold:
                    raise ValueError('donor exposure identity')
                for key, value in world.items():
                    if key == 'donor_profile_id':
                        continue
                    if isinstance(value, str):
                        if archived[key] != value:
                            raise ValueError('source monthly composition mismatch')
                    elif not np.isfinite(archived[key]) or abs(archived[key] - value) > 1e-8:
                        raise ValueError('donor exposure mismatch')
                d, past = world['donor_profile_id'], world['donor_input_profile_id']
                previous, current = data.shares[past], data.shares[d]
                S = data.shares[context['input_profile_id']]
                mask = (previous == 0) & (current == 0) & (S > 0)
                if not np.array_equal(json.loads(world['missing_mask']), mask):
                    raise ValueError('independent response mask mismatch')
                B = .99 * S + .01 * f.prior
                native = B * current / (.99 * previous + .01 * f.prior)
                growth = data.M[d].sum() / data.M[past].sum()
                expected = {'native_composition': native, 'share_composition': np.where(mask, B, native),
                            'energy_composition': np.where(mask, B / growth, native)}
                for key, value in expected.items():
                    if not np.allclose(json.loads(world[key]), value, rtol=0, atol=1e-10):
                        raise ValueError('independent growth-units construction mismatch')
                missing_both += world['conditional_probability'] * float(S @ mask)
            for entry in sensitivity:
                archived = saved_sensitivity.loc[(q, entry['method'])]
                for key in ['world_anchor_mean']:
                    if not np.isfinite(archived[key]) or abs(archived[key] - entry[key]) > 1e-8:
                        raise ValueError('carrier coarse sensitivity mismatch')
            row = data.index.iloc[q]
            clock = data.states[int(row.offset):int(row.stop)] % 24
            actual = data.truth(q)
            if rec.role == 'test':
                enriched.append({'fold': fold, 'profile_id': q, 'region': rec.region, 'month': rec.month,
                                 'hours': len(actual), **context,
                                 'independently_verified_missing_energy_proxy': missing_both})
            for method in METHODS:
                cells = laws[method]
                if rec.role == 'calibration':
                    law = CalendarLaw(cells, clock)
                    delta = np.maximum.reduce([law.quantile(.05) - actual,
                                               actual - law.quantile(.95), np.zeros(len(actual))])
                    errors[method].setdefault(rec.region, []).append(delta)
                    continue
                policies = ['recalibrated'] if method in METHODS[:3] else ['fixed', 'recalibrated']
                for policy in policies:
                    for calibration in ['pooled', 'max_region']:
                        old = rows.loc[rows.profile_id.eq(q) & rows.method.eq(method)
                                       & rows.radius_policy.eq(policy) & rows.calibration.eq(calibration)]
                        if (len(old) != 1 or old.region.iloc[0] != rec.region or old.month.iloc[0] != rec.month
                                or old.hours.iloc[0] != len(actual) or old.fold.iloc[0] != fold):
                            raise ValueError('test score identity')
                        radius = float(saved_radii.loc[(method, calibration), 'radius']) if policy == 'recalibrated' else f.radii[(REFERENCE[method], calibration)]
                        if abs(radius - old.radius.iloc[0]) > 1e-9:
                            raise ValueError('radius policy binding')
                        score = score_laws(cells, actual, clock, radius)
                        max_score = max(max_score, max(abs(score[m] - old[m].iloc[0]) for m in METRICS))
                        rebuilt += 1
            contexts_rebuilt += 1
        for method in METHODS:
            if set(errors[method]) != set(f.queries.loc[f.queries.role.eq('calibration'), 'region']):
                raise ValueError('calibration source cohort')
            for label, value in zip(['pooled', 'max_region'], weighted_radius(errors[method])):
                max_radius = max(max_radius, abs(value - saved_radii.loc[(method, label), 'radius']))
    if max(max_score, max_context, max_radius) > 1e-7:
        raise ValueError('reconstruction mismatch')
    group_keys = ['region', 'method', 'radius_policy', 'calibration']
    regional = rows.groupby(group_keys).apply(lambda g: pd.Series(
        {m: np.average(g[m], weights=g.hours) for m in METRICS}), include_groups=False).reset_index()
    old_regions = pd.read_csv(SOURCE / 'region_metrics.csv')
    matched = regional.merge(old_regions, on=group_keys, suffixes=('_a', '_b'), validate='one_to_one')
    if len(matched) != 754:
        raise ValueError('regional denominator')
    for m in METRICS:
        if not np.isfinite(matched[m + '_b']).all() or max(abs(matched[m + '_a'] - matched[m + '_b'])) > 1e-8:
            raise ValueError('regional aggregation')
    macro = pd.read_csv(SOURCE / 'macro_metrics.csv')
    if len(macro) != 52 or not np.isfinite(macro[METRICS + ['worst_coverage']]).all().all():
        raise ValueError('macro denominator/finite')
    for rec in macro.itertuples(index=False):
        table = regional.loc[regional.method.eq(rec.method) & regional.radius_policy.eq(rec.radius_policy)
                             & regional.calibration.eq(rec.calibration)]
        if rec.cohort == 'nonconstant24':
            table = table.loc[~table.region.isin(['CHPD', 'DOPD', 'GVL', 'SCL', 'SPA'])]
        elif rec.cohort != 'all29':
            raise ValueError('unknown macro cohort')
        if len(table) != rec.regions or abs(table.coverage.min() - rec.worst_coverage) > 1e-10:
            raise ValueError('macro source count/worst')
        if any(abs(table[m].mean() - getattr(rec, m)) > 1e-8 for m in METRICS):
            raise ValueError('macro aggregation')
    old = pd.read_csv(BASE / 'test_month_metrics.csv')
    old = old.loc[old.budget.eq('asof_30d') & old.method.isin(METHODS[:2])]
    new = rows.loc[rows.method.isin(METHODS[:2])]
    pair = new.merge(old, on=['region', 'month', 'method', 'calibration'], suffixes=('_a', '_b'), validate='one_to_one')
    if len(pair) != 964 or any(max(abs(pair[m + '_a'] - pair[m + '_b'])) > 1e-7 for m in METRICS[:5]):
        raise ValueError('old baseline binding')
    gaussian_old = pd.read_csv(ROOT / 'results/v164-scenario-reweighting-r1/test_month_metrics.csv')
    gaussian_old = gaussian_old.loc[gaussian_old.method.eq('world_gaussian')]
    gaussian = rows.loc[rows.method.eq('world_gaussian')].merge(gaussian_old,
        on=['region', 'month', 'method', 'calibration'], suffixes=('_a', '_b'), validate='one_to_one')
    if len(gaussian) != 482 or any(max(abs(gaussian[m + '_a'] - gaussian[m + '_b'])) > 1e-7 for m in METRICS[:5]):
        raise ValueError('V164 Gaussian score binding')
    all_regions = [regional]
    all_macro = [macro]
    cohort_keys = set(rows[['region', 'month', 'calibration', 'hours']].itertuples(index=False, name=None))
    for method, directory in reference_sources.items():
        reference_rows = pd.read_csv(directory / 'test_month_metrics.csv')
        reference_rows = reference_rows.loc[reference_rows.method.eq(method) & reference_rows.budget.eq('asof_30d')]
        if set(reference_rows[['region', 'month', 'calibration', 'hours']].itertuples(index=False, name=None)) != cohort_keys:
            raise ValueError('strong reference cohort differs')
        if not np.isfinite(reference_rows[METRICS[:5]]).all().all():
            raise ValueError('strong reference nonfinite')
        rr = reference_rows.groupby(['region', 'method', 'calibration']).apply(lambda g: pd.Series(
            {m: np.average(g[m], weights=g.hours) for m in METRICS[:5]}), include_groups=False).reset_index()
        rr['radius_policy'] = 'recalibrated'
        all_regions.append(rr)
        reference_macro = []
        for cohort, table in [('all29', rr), ('nonconstant24', rr.loc[~rr.region.isin(['CHPD', 'DOPD', 'GVL', 'SCL', 'SPA'])])]:
            for calibration, group in table.groupby('calibration'):
                reference_macro.append({'cohort': cohort, 'method': method, 'radius_policy': 'recalibrated',
                                        'calibration': calibration, 'regions': len(group),
                                        **{m: group[m].mean() for m in METRICS[:5]},
                                        'worst_coverage': group.coverage.min()})
        all_macro.append(pd.DataFrame(reference_macro))
    reference_macro = pd.concat(all_macro, ignore_index=True)
    reference_regional = pd.concat(all_regions, ignore_index=True)
    rng = np.random.default_rng(167)
    comparisons = []
    for rec in macro.itertuples(index=False):
        if rec.method in METHODS[:3]:
            continue
        for reference in ['positive_recomposition', 'conditional_activity_recomposition', 'world_gaussian',
                          'mask_reweighted', *reference_sources]:
            if reference == rec.method:
                continue
            baseline = reference_macro.loc[reference_macro.method.eq(reference) & reference_macro.cohort.eq(rec.cohort)
                                            & reference_macro.calibration.eq(rec.calibration)
                                            & reference_macro.radius_policy.eq('recalibrated')]
            if len(baseline) != 1:
                raise ValueError('strong reference macro binding')
            baseline = baseline.iloc[0]
            selected = reference_regional.loc[reference_regional.calibration.eq(rec.calibration)]
            if rec.cohort == 'nonconstant24':
                selected = selected.loc[~selected.region.isin(['CHPD', 'DOPD', 'GVL', 'SCL', 'SPA'])]
            a = selected.loc[selected.method.eq(rec.method) & selected.radius_policy.eq(rec.radius_policy)].sort_values('region')
            b = selected.loc[selected.method.eq(reference) & selected.radius_policy.eq('recalibrated')].sort_values('region')
            if not np.array_equal(a.region, b.region):
                raise ValueError('paired reference sources differ')
            delta = a.crps.to_numpy() - b.crps.to_numpy()
            boot = delta[rng.integers(len(delta), size=(3000, len(delta)))].mean(axis=1)
            comparisons.append({'cohort': rec.cohort, 'method': rec.method, 'is_oracle': False,
                            'radius_policy': rec.radius_policy, 'calibration': rec.calibration,
                            'reference': reference, 'crps': rec.crps,
                            'paired_crps_difference': float(delta.mean()),
                            'region_bootstrap_lo': float(np.quantile(boot, .025)),
                            'region_bootstrap_hi': float(np.quantile(boot, .975)),
                            'regions_better_crps': int((delta < -1e-9).sum()),
                            'crps_reduction_percent': 100 * (1 - rec.crps / baseline.crps),
                            'coverage': rec.coverage, 'worst_coverage': rec.worst_coverage,
                            'width': rec.width, 'width_change_percent': 100 * (rec.width / baseline.width - 1),
                            'winkler': rec.winkler,
                            'numeric_threshold_only': bool(rec.coverage >= .9 and rec.worst_coverage >= .85
                                                           and rec.crps <= .95 * baseline.crps
                                                           and rec.width <= baseline.width + 1e-9
                                                           and rec.winkler < baseline.winkler)})
    enriched = pd.DataFrame(enriched)
    exposure_columns = [c for c in enriched if 'proxy' in c or c == 'zero_past_world_probability']
    exposure = enriched.groupby('region').apply(lambda g: pd.Series(
        {m: np.average(g[m], weights=g.hours) for m in exposure_columns}), include_groups=False).reset_index()
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('dependency changed during audit')
    verify(SOURCE)
    OUT.mkdir(parents=True)
    for name, frame in [('comparisons', pd.DataFrame(comparisons)), ('test_exposures', enriched),
                        ('region_exposures', exposure)]:
        frame.to_csv(OUT / f'{name}.csv', index=False)
    result = {'assertion_failures': 0, 'rebuilt_score_rows': rebuilt, 'rebuilt_contexts': contexts_rebuilt,
              'verified_macro_rows': len(macro), 'baseline_records_reproduced': len(pair), 'gaussian_records_reproduced': len(gaussian),
              'future_month_mutation_checks': 12,
              'max_score_error': max_score, 'max_context_error': max_context, 'max_radius_error': max_radius,
              'input_hashes': hashes, 'output_hashes': {p.name: sha(p) for p in sorted(OUT.iterdir())},
              'scope': 'Full reconstruction uses frozen model/law helpers. Legal missing-response controls; q=1 remains a model assumption. Missing exposure is not a causal effect.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['input_hashes', 'output_hashes']}, indent=2))


if __name__ == '__main__':
    main()
