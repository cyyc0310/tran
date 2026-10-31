"""Audit the frozen joint-fuel probe and compare every arm, without selection."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from v151_hourly_law import sha

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / 'results/v158-joint-fuel-response-r1'
BASE = ROOT / 'results/v156-anchored-hourly-sampled-r1'
HOUR = ROOT / 'results/v157-matched-source-hours-r1'
OUT = ROOT / 'results/v158-joint-response-audit-r1'
KEYS = ['region', 'month', 'seed', 'budget', 'calibration']
METRICS = ['crps', 'mae', 'coverage', 'width', 'winkler']


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    archive_hashes = {}
    for archive in (NEW, BASE, HOUR):
        summary = json.loads((archive / 'summary.json').read_text())
        if summary['assertion_failures']:
            raise ValueError('upstream assertion failure')
        archive_hashes[str(archive.relative_to(ROOT))] = sha(archive / 'summary.json')
        for name, digest in summary['input_hashes'].items():
            if sha(ROOT / name) != digest:
                raise ValueError('input changed: ' + name)
        for name, digest in summary['output_hashes'].items():
            if sha(archive / name) != digest:
                raise ValueError('output changed: ' + name)
    for name in ('source_sampling.csv', 'region_roles.csv'):
        if (NEW / name).read_bytes() != (BASE / name).read_bytes():
            raise ValueError('source contract changed: ' + name)
    # V157 records its exact selection through the archived V154 draw.
    recorded = pd.read_csv(ROOT / 'results/v154-hourly-strong-baselines-r1/source_sampling.csv')
    current = pd.read_csv(NEW / 'source_sampling.csv')
    sample_keys = ['target_region', 'seed', 'budget', 'region']
    matched_samples = current.merge(recorded, on=sample_keys, suffixes=('_new', '_hour'),
                                    validate='one_to_one')
    if len(matched_samples) != 360:
        raise ValueError('recomposition source sample denominator')
    for row in matched_samples.itertuples(index=False):
        if json.loads(row.sampled_raw_hour_indices_new) != json.loads(row.sampled_raw_hour_indices_hour):
            raise ValueError('recomposition source samples differ')
    rows = pd.read_csv(NEW / 'target_month_metrics.csv')
    if len(rows) != 1440 or rows.duplicated(KEYS + ['method']).any():
        raise ValueError('score denominator')
    if not np.isfinite(rows[METRICS].to_numpy()).all():
        raise ValueError('nonfinite scores')
    old = pd.read_csv(BASE / 'target_month_metrics.csv')
    old = old.loc[old.method.eq('distribution_forest')]
    scalar = rows.loc[rows.method.eq('scalar')].merge(
        old, on=KEYS, suffixes=('_new', '_old'), validate='one_to_one')
    errors = {m: float(np.max(np.abs(scalar[m + '_new'] - scalar[m + '_old'])))
              for m in METRICS}
    if len(scalar) != 288 or max(errors.values()) > 1e-7:
        raise ValueError('scalar reproduction')
    asof = rows.loc[rows.budget.eq('asof_30d')]
    duplicate = asof.loc[asof.method.eq('joint_budget')].merge(
        asof.loc[asof.method.eq('joint_nonnegative')], on=KEYS,
        suffixes=('_budget', '_nonnegative'), validate='one_to_one')
    for m in METRICS:
        if not np.array_equal(duplicate[m + '_budget'], duplicate[m + '_nonnegative']):
            raise ValueError('asof zero support was applied')
    roles = pd.read_csv(NEW / 'region_roles.csv')
    index = pd.read_csv(ROOT / 'results/v144-canonical-profile-cache-r1/profile_index.csv')
    prototypes = pd.read_csv(NEW / 'response_prototypes.csv')
    if len(prototypes) != 12:
        raise ValueError('prototype denominator')
    for row in prototypes.itertuples(index=False):
        fit = set(roles.loc[roles.target_region.eq(row.target_region)
                            & roles.seed.eq(row.seed) & roles.role.eq('fit'), 'region'])
        chosen = np.asarray(json.loads(row.prototype_profile_ids), int)
        if len(chosen) != 5 or not set(index.region.iloc[chosen]).issubset(fit):
            raise ValueError('prototype source identity')
        if not index.month.iloc[chosen].str.startswith('2024').all():
            raise ValueError('prototype time identity')
    macro = pd.read_csv(NEW / 'macro_metrics.csv')
    hmacro = pd.read_csv(HOUR / 'macro_metrics.csv')
    hmacro = hmacro.loc[hmacro.method.eq('sampled')].copy()
    hmacro['method'] = 'hourly_recomposition'
    strong = pd.read_csv(BASE / 'macro_metrics.csv')
    strong = strong.loc[~strong.method.eq('distribution_forest')]
    allmacro = pd.concat([macro, hmacro, strong], ignore_index=True)
    hrows = pd.read_csv(HOUR / 'target_month_metrics.csv')
    hrows = hrows.loc[hrows.method.eq('sampled')].copy()
    hrows['method'] = 'hourly_recomposition'
    allrows = pd.concat([rows, hrows], ignore_index=True)
    comparisons = []
    for candidate, reference in [('joint_linear', 'scalar'),
                                  ('joint_nonnegative', 'joint_linear'),
                                  ('joint_budget', 'joint_nonnegative'),
                                  ('response_budget', 'joint_budget'),
                                  ('joint_budget', 'hourly_recomposition'),
                                  ('response_budget', 'hourly_recomposition'),
                                  ('response_budget', 'heteroscedastic_normal'),
                                  ('response_budget', 'conditional_quantile')]:
        a = allmacro.loc[allmacro.method.eq(candidate)]
        b = allmacro.loc[allmacro.method.eq(reference)]
        match = a.merge(b, on=['seed', 'budget', 'calibration'],
                        suffixes=('_candidate', '_reference'), validate='one_to_one')
        if len(match) != 8:
            raise ValueError('macro comparison denominator')
        for row in match.itertuples(index=False):
            comparisons.append({
                'candidate': candidate, 'reference': reference,
                'seed': row.seed, 'budget': row.budget, 'calibration': row.calibration,
                'crps_reduction_percent': 100 * (1 - row.crps_candidate / row.crps_reference),
                'coverage': row.coverage_candidate,
                'worst_coverage': row.worst_coverage_candidate,
                'width_change_percent': 100 * (row.width_candidate / row.width_reference - 1),
                'winkler_change_percent': 100 * (row.winkler_candidate / row.winkler_reference - 1),
                'crps_comparison_available': bool(np.isfinite(row.crps_reference)),
                'passes_coverage': row.coverage_candidate >= .9 and row.worst_coverage_candidate >= .85,
                'passes_joint_gate': ((row.coverage_candidate >= .9 and row.worst_coverage_candidate >= .85
                                      and row.crps_candidate <= .95 * row.crps_reference
                                      and row.width_candidate <= row.width_reference
                                      and row.winkler_candidate < row.winkler_reference)
                                     if np.isfinite(row.crps_reference) else None)})
    # Resample shared calendar months jointly across the three opened regions.
    # These are descriptive intervals, with no independent-target inference.
    paired = []
    for reference in ('joint_budget', 'hourly_recomposition'):
        a = allrows.loc[allrows.method.eq('response_budget') & allrows.calibration.eq('pooled')]
        b = allrows.loc[allrows.method.eq(reference) & allrows.calibration.eq('pooled')]
        match = a.merge(b, on=KEYS, suffixes=('_candidate', '_reference'), validate='one_to_one')
        for (seed, budget), group in match.groupby(['seed', 'budget']):
            if len(group) != 36 or not group.hours_candidate.eq(group.hours_reference).all():
                raise ValueError('paired target denominator')
            months = sorted(group.month.unique())
            weighted_deltas = np.zeros((3, len(months)))
            hours = np.zeros_like(weighted_deltas)
            for r, region in enumerate(sorted(group.region.unique())):
                g = group.loc[group.region.eq(region)].set_index('month').loc[months]
                hours[r] = g.hours_candidate
                weighted_deltas[r] = (g.crps_candidate - g.crps_reference) * hours[r]
            rng = np.random.default_rng(20261028)
            draws = rng.integers(0, len(months), size=(1000, len(months)))
            boot = (weighted_deltas[:, draws].sum(axis=2) / hours[:, draws].sum(axis=2)).mean(axis=0)
            paired.append({'reference': reference, 'seed': seed, 'budget': budget,
                           'response_minus_reference_crps': float((weighted_deltas.sum(axis=1) / hours.sum(axis=1)).mean()),
                           'descriptive_95_lower': float(np.quantile(boot, .025)),
                           'descriptive_95_upper': float(np.quantile(boot, .975))})
    OUT.mkdir(parents=True)
    pd.DataFrame(comparisons).to_csv(OUT / 'ablation_comparisons.csv', index=False)
    pd.DataFrame(paired).to_csv(OUT / 'paired_month_crps.csv', index=False)
    allmacro.to_csv(OUT / 'all_macro_metrics.csv', index=False)
    result = {'assertion_failures': 0, 'scalar_records_reproduced': len(scalar),
              'scalar_max_absolute_errors': errors, 'identical_asof_duplicate_records': len(duplicate),
              'prototype_sets_verified': len(prototypes), 'comparison_cells': len(comparisons),
              'paired_cells': len(paired), 'script_sha256': sha(Path(__file__)),
              'archive_summary_hashes': archive_hashes,
              'upstream_summary_sha256': sha(NEW / 'summary.json'),
              'output_hashes': {p.name: sha(p) for p in sorted(OUT.iterdir())},
              'scope': 'Opened-target description; source-role/hash/denominator verification, not independent confirmation.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(allmacro.loc[allmacro.calibration.eq('pooled')].to_string(index=False))


if __name__ == '__main__':
    main()
