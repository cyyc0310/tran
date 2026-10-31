"""Audit masking provenance/aggregation and refit every block on fixed cases."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from hourly_fuel_data import HourlyFuelData
from masked_shape_completion import (CompletionForest, METHODS, ShapePanel,
                                     proposals_for, score_completion)
from v151_hourly_law import EF, sha
from v168_masked_shape_value import METRICS, case_inventory, hidden_intervention

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/v159-fuel-activity-source-pilot-r1'
SOURCE = ROOT / 'results/v168-masked-shape-value-r1'
OUT = ROOT / 'results/v168-shape-value-audit-r1'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    summary = json.loads((SOURCE / 'summary.json').read_text())
    if summary['assertion_failures'] or summary['completed_blocks'] != 48:
        raise ValueError('incomplete source diagnostic')
    for name, digest in summary['output_hashes'].items():
        if sha(SOURCE / name) != digest:
            raise ValueError('source output changed')
    hashes = summary['input_hashes'] | {'scripts/v168_audit_shape_value.py': sha(Path(__file__)),
                                      str(SOURCE.relative_to(ROOT) / 'summary.json'): sha(SOURCE / 'summary.json')}
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('input changed')
    panel = ShapePanel(HourlyFuelData(ROOT), pd.read_csv(BASE / 'source_sampling.csv'))
    cases = case_inventory(panel)
    stored_cases = pd.read_csv(SOURCE / 'cases.csv', dtype={'mask_id': str})
    if stored_cases.to_csv(index=False) != cases.to_csv(index=False):
        raise ValueError('case inventory changed')
    rows = pd.read_csv(SOURCE / 'case_metrics.csv', dtype={'mask_id': str})
    keys = ['profile_id', 'mask_id', 'method']
    expected = cases.merge(pd.DataFrame({'method': METHODS}), how='cross')
    match = rows.merge(expected, on=keys, suffixes=('_score', '_case'), validate='one_to_one')
    if len(rows) != 5112 or len(match) != len(rows) or not np.isfinite(rows[METRICS]).all().all():
        raise ValueError('score denominator/finite')
    for key in ['fold', 'region', 'month', 'hours', 'mask_size', 'mixed_ef']:
        if not np.array_equal(match[key + '_score'], match[key + '_case']):
            raise ValueError('case score identity mismatch')
    a = rows.loc[rows.method.eq('conditional_joint')].set_index(['profile_id', 'mask_id'])
    b = rows.loc[rows.method.eq('conditional_independent')].set_index(['profile_id', 'mask_id']).loc[a.index]
    if max(abs(a.q_crps - b.q_crps)) > 1e-9:
        raise ValueError('joint/independent marginal score mismatch')
    single = a.mask_size.eq(1)
    if np.max(np.abs(a.loc[single, METRICS].to_numpy() - b.loc[single, METRICS].to_numpy())) > 1e-9:
        raise ValueError('single-mask methods differ')
    stress_parts, provenance_count, rebuilt_rows, max_rebuild = [], 0, 0, 0.
    max_stress_error, selected_cases = 0., []
    with threadpool_limits(limits=2):
        for (fold, mask_id), group in cases.groupby(['fold', 'mask_id']):
            mask = tuple(int(v) for v in mask_id.split('-'))
            block = SOURCE / f'fold_{fold}_mask_{mask_id}'
            block_summary = json.loads((block / 'summary.json').read_text())
            proper = json.loads((BASE / f'fold_{fold}_asof_30d/summary.json').read_text())['source_regions']
            if (block_summary['proper_regions'] != sorted(proper)
                    or block_summary['hidden_feature_interventions'] != len(group)
                    or block_summary['case_rows'] != len(group) * len(METHODS)):
                raise ValueError('block identity/count')
            proper_ids = np.flatnonzero(np.isin(panel.regions, proper))
            observed_ids = proper_ids[panel.present[proper_ids][:, mask].all(axis=1)]
            fits = json.loads((block / 'fit_models.json').read_text())
            if {item['model'] for item in fits} != {'coarse', 'conditional', 'log', 'imputed'}:
                raise ValueError('missing fitted control')
            for item in fits:
                ids = proper_ids if item['model'] == 'imputed' else observed_ids
                if (item['hours'] != len(ids) or item['regions'] != sorted(set(panel.regions[ids]))
                        or item['raw_ids_sha256'] != hashlib.sha256(panel.raw_ids[ids].tobytes()).hexdigest()
                        or item['pseudo_label_rows'] != int((~panel.present[ids][:, mask].all(axis=1)).sum())):
                    raise ValueError('training provenance/pseudo-label mismatch')
                provenance_count += 1
            saved_stress = pd.read_csv(block / 'particle_stress.csv', dtype={'mask_id': str})
            stress_metrics = [m + suffix for m in ['crps', 'coverage', 'width'] for suffix in ['_1024', '_4096']]
            if (len(saved_stress) != len(group) * 2 or not np.isfinite(saved_stress[stress_metrics]).all().all()
                    or not saved_stress.fold.eq(fold).all() or not saved_stress.mask_id.eq(mask_id).all()):
                raise ValueError('particle stress denominator')
            for case in group.itertuples(index=False):
                ids = panel.queries(int(case.profile_id))
                hidden_intervention(panel, ids[:2], mask)
                selected = saved_stress.loc[saved_stress.profile_id.eq(case.profile_id)]
                if (set(selected.method) != {'conditional_joint', 'conditional_independent'}
                        or not selected.raw_id.eq(int(panel.raw_ids[ids[0]])).all()):
                    raise ValueError('particle stress query identity')
            stress_parts.append(saved_stress)
            # Refit every model; score the first and last predeclared case in each
            # block. Selection depends on inventory order, never performance.
            model = CompletionForest(panel, proper, mask)
            if model.fallback != block_summary['model_fallback']:
                raise ValueError('model fallback flag mismatch')
            indices = sorted(set([0, len(group) - 1]))
            for case in group.iloc[indices].itertuples(index=False):
                q = int(case.profile_id)
                ids = panel.queries(q)
                distributions = model.distributions(ids)
                S = panel.data.shares[q]
                lo, hi = EF[S > 0].min(), EF[S > 0].max()
                collected = {method: [] for method in METHODS}
                for position, (index, distribution) in enumerate(zip(ids, distributions)):
                    raw_id = int(panel.raw_ids[index])
                    key = f'v168/{fold}/{mask_id}/{q}/{raw_id}'
                    proposals, _ = proposals_for(distribution, key)
                    raw = panel.data.raw[raw_id]
                    truth = float(np.clip(raw @ EF / raw.sum(), lo, hi))
                    for method, proposal in proposals.items():
                        collected[method].append(score_completion(S, panel.q[index], mask, proposal, panel.q[index], truth))
                    if position == 0:
                        fine, _ = proposals_for(distribution, key, 4096)
                        for method in ['conditional_joint', 'conditional_independent']:
                            score = score_completion(S, panel.q[index], mask, fine[method], panel.q[index], truth)
                            stored = saved_stress.loc[saved_stress.profile_id.eq(q) & saved_stress.method.eq(method)].iloc[0]
                            for metric in ['crps', 'coverage', 'width']:
                                max_stress_error = max(max_stress_error, abs(stored[metric + '_4096'] - score[metric]),
                                                       abs(stored[metric + '_1024'] - collected[method][0][metric]))
                for method, values in collected.items():
                    stored = rows.loc[rows.profile_id.eq(q) & rows.mask_id.eq(mask_id) & rows.method.eq(method)].iloc[0]
                    new = {m: float(np.mean([v[m] for v in values])) for m in METRICS}
                    max_rebuild = max(max_rebuild, max(abs(new[m] - stored[m]) for m in METRICS))
                    rebuilt_rows += 1
                selected_cases.append({'fold': fold, 'mask_id': mask_id, 'profile_id': q})
            print(f'audited fold={fold} mask={mask_id}', flush=True)
    if max(max_rebuild, max_stress_error) > 1e-7:
        raise ValueError('sampled refit reconstruction differs')
    regional, macro_records = [], []
    subsets = {'all29': rows, 'mixed_ef': rows.loc[rows.mixed_ef], 'contrast_pairs': rows.loc[rows.mask_size.eq(2)]}
    for cohort, table in subsets.items():
        per_month = table.groupby(['region', 'profile_id', 'method'])[METRICS].mean().reset_index()
        per_region = per_month.groupby(['region', 'method'])[METRICS].mean().reset_index()
        per_region['cohort'] = cohort
        regional.append(per_region)
        for method, g in per_region.groupby('method'):
            macro_records.append({'cohort': cohort, 'method': method, 'regions': len(g),
                                  **{m: g[m].mean() for m in METRICS}, 'worst_coverage': g.coverage.min()})
    regional, macro = pd.concat(regional), pd.DataFrame(macro_records)
    old_macro = pd.read_csv(SOURCE / 'macro_metrics.csv')
    paired = macro.merge(old_macro, on=['cohort', 'method'], suffixes=('_a', '_b'), validate='one_to_one')
    if len(paired) != 24:
        raise ValueError('macro denominator')
    for metric in METRICS + ['worst_coverage', 'regions']:
        if not np.isfinite(paired[metric + '_b']).all() or max(abs(paired[metric + '_a'] - paired[metric + '_b'])) > 1e-8:
            raise ValueError('macro aggregation mismatch')
    old_regions = pd.read_csv(SOURCE / 'region_metrics.csv')
    paired_regions = regional.merge(old_regions, on=['cohort', 'region', 'method'], suffixes=('_a', '_b'), validate='one_to_one')
    if len(paired_regions) != len(regional) or len(old_regions) != len(regional):
        raise ValueError('regional denominator')
    for metric in METRICS:
        if max(abs(paired_regions[metric + '_a'] - paired_regions[metric + '_b'])) > 1e-8:
            raise ValueError('regional aggregation mismatch')
    comparisons, rng = [], np.random.default_rng(168)
    for cohort, table in regional.groupby('cohort'):
        for method in ['conditional_joint', 'conditional_joint_log', 'conditional_joint_imputed']:
            for reference in ['flat', 'clock_independent', 'coarse_joint', 'conditional_mean',
                              'conditional_independent', 'conditional_joint', 'conditional_joint_imputed']:
                if method == reference:
                    continue
                a = table.loc[table.method.eq(method)].sort_values('region')
                b = table.loc[table.method.eq(reference)].sort_values('region')
                if not np.array_equal(a.region, b.region):
                    raise ValueError('paired comparison cohort mismatch')
                delta = a.crps.to_numpy() - b.crps.to_numpy()
                boot = delta[rng.integers(len(delta), size=(3000, len(delta)))].mean(axis=1)
                comparisons.append({'cohort': cohort, 'method': method, 'reference': reference,
                                    'regions': len(a), 'crps': a.crps.mean(), 'crps_reduction_percent': 100 * (1 - a.crps.mean() / b.crps.mean()),
                                    'paired_crps_difference': delta.mean(), 'region_bootstrap_lo': np.quantile(boot, .025),
                                    'region_bootstrap_hi': np.quantile(boot, .975), 'regions_better_crps': int((delta < -1e-9).sum()),
                                    'coverage': a.coverage.mean(), 'worst_coverage': a.coverage.min(),
                                    'width': a.width.mean(),
                                    'width_change_percent': (100 * (a.width.mean() / b.width.mean() - 1)
                                                             if b.width.mean() > 1e-12 else None),
                                    'winkler': a.winkler.mean()})
    stress = pd.concat(stress_parts, ignore_index=True)
    stress_summary = stress.groupby('method').apply(lambda g: pd.Series(
        {'queries': len(g), 'mean_crps_change': np.mean(g.crps_4096 - g.crps_1024),
         'mean_absolute_crps_change': np.mean(abs(g.crps_4096 - g.crps_1024)),
         'max_absolute_crps_change': np.max(abs(g.crps_4096 - g.crps_1024)),
         'mean_width_change': np.mean(g.width_4096 - g.width_1024),
         'coverage_changed_queries': int((g.coverage_4096 != g.coverage_1024).sum())}), include_groups=False).reset_index()
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('dependency changed during audit')
    for name, digest in summary['output_hashes'].items():
        if sha(SOURCE / name) != digest:
            raise ValueError('source output changed during audit')
    OUT.mkdir(parents=True)
    for name, table in [('comparisons', pd.DataFrame(comparisons)), ('particle_stress_summary', stress_summary),
                        ('refit_case_inventory', pd.DataFrame(selected_cases))]:
        table.to_csv(OUT / f'{name}.csv', index=False)
    result = {'assertion_failures': 0, 'verified_case_scores': len(rows), 'verified_training_provenance_records': provenance_count,
              'refitted_blocks': 48, 'recomputed_case_score_rows': rebuilt_rows, 'max_refit_score_error': max_rebuild,
              'max_sampled_stress_rebuild_error': max_stress_error, 'hidden_feature_interventions': len(cases),
              'verified_macro_rows': len(macro), 'input_hashes': hashes,
              'output_hashes': {p.name: sha(p) for p in sorted(OUT.iterdir())},
              'scope': 'All provenance/identities/aggregates checked. All 48 blocks (192 forests) refitted; first/last inventory cases per block rescored, remaining case scores not recomputed. Uses frozen completion/scoring helpers, not a second independent algorithm. Conditional source masking is not zero-hour forecasting.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(stress_summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
