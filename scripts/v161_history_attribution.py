"""Post-hoc coarse-history attribution; simple past-EF point gate is not a bound."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from monthly_history_features import MonthlyInformation
from v151_hourly_law import EF, sha

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/v161-monthly-history-value-r1'
AUDIT = ROOT / 'results/v161-monthly-history-audit-r1'
OUT = ROOT / 'results/v161-history-attribution-r2'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    source_summary = json.loads((SOURCE / 'summary.json').read_text())
    dependencies = dict(source_summary['input_hashes']) | {
        str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
        str((SOURCE / 'summary.json').relative_to(ROOT)): sha(SOURCE / 'summary.json'),
        str((AUDIT / 'summary.json').relative_to(ROOT)): sha(AUDIT / 'summary.json')}
    for name, digest in dependencies.items():
        if sha(ROOT / name) != digest:
            raise ValueError('attribution input changed')
    for name, digest in source_summary['output_hashes'].items():
        if sha(SOURCE / name) != digest:
            raise ValueError('source output changed')
    data = MonthlyInformation(ROOT)
    rows = pd.read_csv(SOURCE / 'predictions.csv')
    roles = pd.read_csv(ROOT / 'results/v159-fuel-activity-source-pilot-r1/region_roles.csv')
    # Explicitly bind metadata to profile IDs and the held-out role.
    for row in rows.itertuples(index=False):
        identity = data.index.iloc[row.profile_id]
        if row.region != identity.region or row.month != identity.month:
            raise ValueError('prediction metadata does not match profile')
        roster = roles.loc[roles.fold.eq(row.fold) & roles.region.eq(row.region)]
        if len(roster) != 1 or roster.iloc[0].role != 'test':
            raise ValueError('prediction is not in held-out source role')
    constant, past_anchor = {}, {}
    for q in rows.profile_id.unique():
        local = data.local(int(q), 'asof_30d')
        constant[q] = bool(np.ptp(EF[data.shares[local] > 0]) == 0)
        past_anchor[q] = float(data.shares[local] @ EF)
    rows['past_constant_factor'] = rows.profile_id.map(constant)
    augmented = [rows]
    for method in ('forest_base', 'forest_history', 'forest_noise', 'ridge_base', 'ridge_history', 'ridge_noise'):
        part = rows.loc[rows.method.eq(method)].copy()
        part['method'] = 'past_constant_gate_' + method
        mask = part.past_constant_factor
        part.loc[mask, 'prediction'] = part.loc[mask, 'profile_id'].map(past_anchor)
        part['error'] = part.prediction - part.actual_anchor
        augmented.append(part)
    allrows = pd.concat(augmented, ignore_index=True)
    strata = []
    for (method, is_constant), group in allrows.groupby(['method', 'past_constant_factor']):
        regional = group.groupby('region').error.apply(lambda e: e.abs().mean())
        strata.append({'method': method, 'past_constant_factor': bool(is_constant),
                       'mae': float(regional.mean()), 'regions': len(regional), 'months': len(group)})
    macro = allrows.groupby(['method', 'region']).error.apply(lambda e: e.abs().mean()).groupby('method').mean().rename('mae').reset_index()
    pairs = [('forest_history', 'forest_base'),
             ('forest_history', 'past_constant_gate_forest_base'),
             ('past_constant_gate_forest_history', 'past_constant_gate_forest_base'),
             ('past_constant_gate_forest_history', 'past_constant_gate_forest_noise')]
    comparisons = []
    for candidate, reference in pairs:
        a = allrows.loc[allrows.method.eq(candidate)]
        b = allrows.loc[allrows.method.eq(reference)]
        if set(a.profile_id) != set(b.profile_id) or len(a) != 241 or a.region.nunique() != 29:
            raise ValueError('paired query set/region completeness')
        joined = a.merge(b, on=['profile_id', 'region', 'month'], suffixes=('_a', '_b'), validate='one_to_one')
        if len(joined) != 241 or joined.region.nunique() != 29 or not joined.fold_a.eq(joined.fold_b).all():
            raise ValueError('paired metadata mismatch')
        joined['delta'] = joined.error_a.abs() - joined.error_b.abs()
        months = sorted(joined.month.unique())
        matrix = np.zeros((29, len(months)))
        count = np.zeros_like(matrix)
        for r, region in enumerate(sorted(joined.region.unique())):
            group = joined.loc[joined.region.eq(region)].set_index('month')
            for m, month in enumerate(months):
                if month in group.index:
                    matrix[r, m] = group.loc[month, 'delta']
                    count[r, m] = 1
        weights = np.random.default_rng(20261028).dirichlet(np.ones(len(months)), size=1000)
        distribution = ((matrix @ weights.T) / (count @ weights.T)).mean(axis=0)
        region_delta = joined.groupby('region').delta.mean()
        fold_delta = joined.groupby(['fold_a', 'region']).delta.mean().groupby('fold_a').mean()
        comparisons.append({'candidate': candidate, 'reference': reference,
                            'candidate_minus_reference_mae': float(region_delta.mean()),
                            'descriptive_95_lower': float(np.quantile(distribution, .025)),
                            'descriptive_95_upper': float(np.quantile(distribution, .975)),
                            'improved_regions': int((region_delta < 0).sum()),
                            'improved_folds': int((fold_delta < 0).sum())})
    if any(sha(ROOT / name) != digest for name, digest in dependencies.items()):
        raise ValueError('attribution input changed during computation')
    OUT.mkdir(parents=True)
    for name, frame in [('augmented_predictions', allrows), ('macro_mae', macro),
                        ('factor_strata', pd.DataFrame(strata)), ('paired_attribution', pd.DataFrame(comparisons))]:
        frame.to_csv(OUT / f'{name}.csv', index=False)
    result = {'assertion_failures': 0, 'verified_prediction_metadata_rows': len(rows),
              'source_summary_sha256': sha(SOURCE / 'summary.json'),
              'original_audit_sha256': sha(AUDIT / 'summary.json'), 'script_sha256': sha(Path(__file__)),
              'input_hashes': dependencies,
              'output_hashes': {p.name: sha(p) for p in sorted(OUT.iterdir())},
              'scope': 'Post-hoc development attribution; point persistence gate is not a future hard-support claim.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(macro.to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))


if __name__ == '__main__':
    main()
