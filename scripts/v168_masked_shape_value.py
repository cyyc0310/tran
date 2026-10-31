"""Source-world shape masking diagnostic; never a zero-hour target forecast."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from hourly_fuel_data import HourlyFuelData
from masked_shape_completion import (CompletionForest, METHODS, ShapePanel, masks_for,
                                     proposals_for, score_completion)
from v151_hourly_law import EF, sha

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/v159-fuel-activity-source-pilot-r1'
OUT = ROOT / 'results/v168-masked-shape-value-r1'
PROTOCOL = ROOT / 'docs/V168_MASKED_SHAPE_VALUE_PROTOCOL.md'
METRICS = ['q_crps', 'crps', 'mae', 'coverage', 'tolerant_coverage', 'width',
           'winkler', 'rejected_mass', 'flat_fallback']


def hidden_intervention(panel, ids, mask):
    before = panel.features(ids, mask)
    old = panel.q[np.ix_(ids, mask)].copy()
    try:
        panel.q[np.ix_(ids, mask)] = np.random.default_rng(168).uniform(100, 1000, old.shape)
        after = panel.features(ids, mask)
    finally:
        panel.q[np.ix_(ids, mask)] = old
    if not np.array_equal(before, after):
        raise ValueError('hidden shape leaked into completion input')


def case_inventory(panel):
    rows = []
    for fold in range(6):
        queries = pd.read_csv(BASE / f'fold_{fold}_asof_30d/query_inputs.csv')
        for rec in queries.loc[queries.role.eq('test')].itertuples(index=False):
            q = int(rec.profile_id)
            S = panel.data.shares[q]
            ids = panel.queries(q)
            if not np.all(panel.regions[ids] == rec.region):
                raise ValueError('case source identity')
            for mask in masks_for(S):
                rows.append({'fold': fold, 'profile_id': q, 'region': rec.region, 'month': rec.month,
                             'mask_id': '-'.join(map(str, mask)), 'mask_size': len(mask), 'hours': len(ids),
                             'mixed_ef': bool(np.ptp(EF[S > 0]) > 0),
                             'selected_raw_ids_sha256': hashlib.sha256(panel.raw_ids[ids].tobytes()).hexdigest()})
    result = pd.DataFrame(rows)
    if len(result) != 639 or result.hours.sum() != 40896:
        raise ValueError('predeclared case denominator')
    return result


def run_block(panel, fold, mask_id, cases, block, manifest_hash):
    mask = tuple(int(v) for v in mask_id.split('-'))
    metadata = json.loads((BASE / f'fold_{fold}_asof_30d/summary.json').read_text())
    proper = metadata['source_regions']
    if set(cases.region) & set(proper):
        raise ValueError('fit/test region overlap')
    model = CompletionForest(panel, proper, mask)
    if not set(panel.regions[model.ids]).issubset(proper):
        raise ValueError('training region leakage')
    records, stress = [], []
    max_label_error, max_clip, max_marginal_error, max_mean_error = 0., 0., 0., 0.
    intervention_count = 0
    for case in cases.itertuples(index=False):
        q = int(case.profile_id)
        ids = panel.queries(q)
        hidden_intervention(panel, ids[:2], mask)
        intervention_count += 1
        distribution = model.distributions(ids)
        S = panel.data.shares[q]
        lower, upper = float(EF[S > 0].min()), float(EF[S > 0].max())
        collected = {name: [] for name in METHODS}
        for position, (index, dist) in enumerate(zip(ids, distribution)):
            raw_id = int(panel.raw_ids[index])
            key = f'v168/{fold}/{mask_id}/{q}/{raw_id}'
            proposals, mean_error = proposals_for(dist, key)
            max_mean_error = max(max_mean_error, mean_error)
            actual_q = panel.q[index]
            raw = panel.data.raw[raw_id]
            actual_raw = float(raw @ EF / raw.sum())
            actual_reconstructed = float((S * EF) @ actual_q / (S @ actual_q))
            max_label_error = max(max_label_error, abs(actual_raw - actual_reconstructed))
            actual = float(np.clip(actual_raw, lower, upper))
            max_clip = max(max_clip, abs(actual - actual_raw))
            scores = {name: score_completion(S, actual_q, mask, proposal, actual_q, actual)
                      for name, proposal in proposals.items()}
            max_marginal_error = max(max_marginal_error, abs(scores['conditional_independent']['q_crps']
                                                            - scores['conditional_joint']['q_crps']))
            if len(mask) == 1:
                if max(abs(scores['conditional_independent'][m] - scores['conditional_joint'][m]) for m in METRICS) > 1e-10:
                    raise ValueError('single-fuel joint/independent mismatch')
            if position == 0:
                fine, _ = proposals_for(dist, key, 4096)
                for name in ['conditional_joint', 'conditional_independent']:
                    value = score_completion(S, actual_q, mask, fine[name], actual_q, actual)
                    stress.append({'fold': fold, 'profile_id': q, 'mask_id': mask_id, 'raw_id': raw_id,
                                   'method': name, 'crps_1024': scores[name]['crps'], 'crps_4096': value['crps'],
                                   'coverage_1024': scores[name]['coverage'], 'coverage_4096': value['coverage'],
                                   'width_1024': scores[name]['width'], 'width_4096': value['width']})
            for name in METHODS:
                collected[name].append(scores[name])
        for name, values in collected.items():
            records.append({'fold': fold, 'profile_id': q, 'region': case.region, 'month': case.month,
                            'mask_id': mask_id, 'mask_size': len(mask), 'mixed_ef': case.mixed_ef,
                            'hours': len(ids), 'method': name,
                            **{m: float(np.mean([v[m] for v in values])) for m in METRICS}})
    if max(max_label_error, max_clip, max_marginal_error) > 1e-9:
        raise ValueError('physical/marginal identity failed')
    rows = pd.DataFrame(records)
    if len(rows) != len(cases) * len(METHODS) or not np.isfinite(rows[METRICS]).all().all():
        raise ValueError('case scores denominator/finite')
    staging = Path(tempfile.mkdtemp(prefix='.v168-', dir=OUT.parent))
    rows.to_csv(staging / 'case_metrics.csv', index=False)
    pd.DataFrame(stress).to_csv(staging / 'particle_stress.csv', index=False)
    fit_summary = []
    for name, ids in model.training_ids.items():
        fit_summary.append({'model': name, 'hours': len(ids), 'regions': sorted(set(panel.regions[ids])),
                            'raw_ids_sha256': hashlib.sha256(panel.raw_ids[ids].tobytes()).hexdigest(),
                            'pseudo_label_rows': int((~panel.present[ids][:, mask].all(axis=1)).sum())})
    (staging / 'fit_models.json').write_text(json.dumps(fit_summary, indent=2, allow_nan=False) + '\n')
    summary = {'assertion_failures': 0, 'fold': fold, 'mask_id': mask_id, 'proper_regions': sorted(proper),
               'manifest_sha256': manifest_hash, 'case_rows': len(rows), 'model_fallback': model.fallback,
               'hidden_feature_interventions': intervention_count, 'max_raw_label_identity_error': max_label_error,
               'max_actual_physical_clip': max_clip, 'max_joint_independent_q_crps_error': max_marginal_error,
               'max_resampled_conditional_mean_error': max_mean_error,
               'output_hashes': {p.name: sha(p) for p in sorted(staging.iterdir())}}
    (staging / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    staging.rename(block)
    print(f'completed fold={fold} mask={mask_id}: {len(cases)} cases, {len(model.ids)} observable train hours', flush=True)


def aggregate(rows):
    regional_records, macros = [], []
    cohorts = {'all29': rows, 'mixed_ef': rows.loc[rows.mixed_ef],
               'contrast_pairs': rows.loc[rows.mask_size.eq(2)]}
    for cohort, data in cohorts.items():
        per_profile = data.groupby(['region', 'profile_id', 'method'])[METRICS].mean().reset_index()
        per_region = per_profile.groupby(['region', 'method'])[METRICS].mean().reset_index()
        per_region['cohort'] = cohort
        regional_records.append(per_region)
        for method, group in per_region.groupby('method'):
            macros.append({'cohort': cohort, 'method': method, 'regions': len(group),
                           **{m: float(group[m].mean()) for m in METRICS},
                           'worst_coverage': float(group.coverage.min())})
    return pd.concat(regional_records, ignore_index=True), pd.DataFrame(macros)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if OUT.exists() and not args.resume:
        raise FileExistsError(OUT)
    upstream = json.loads((BASE / 'summary.json').read_text())
    if upstream['assertion_failures']:
        raise ValueError('upstream failed')
    for name, digest in upstream['output_hashes'].items():
        if sha(BASE / name) != digest:
            raise ValueError('upstream changed')
    dependencies = [PROTOCOL, Path(__file__), ROOT / 'scripts/masked_shape_completion.py', BASE / 'summary.json']
    hashes = upstream['input_hashes'] | {str(p.relative_to(ROOT)): sha(p) for p in dependencies}
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('input changed')
    manifest = json.dumps({'input_hashes': hashes, 'methods': METHODS, 'particles': 1024,
                           'scope': 'Conditional source-world masking diagnostic, not zero-local-hour forecasting.'},
                          sort_keys=True, indent=2) + '\n'
    if OUT.exists():
        if (OUT / 'run_manifest.json').read_text() != manifest or (OUT / 'summary.json').exists():
            raise ValueError('invalid resume')
    else:
        OUT.mkdir(parents=True)
        (OUT / 'run_manifest.json').write_text(manifest)
    panel = ShapePanel(HourlyFuelData(ROOT), pd.read_csv(BASE / 'source_sampling.csv'))
    cases = case_inventory(panel)
    if (OUT / 'cases.csv').exists():
        if (OUT / 'cases.csv').read_text() != cases.to_csv(index=False):
            raise ValueError('resume cases differ')
    else:
        cases.to_csv(OUT / 'cases.csv', index=False)
    blocks = []
    with threadpool_limits(limits=2):
        for (fold, mask_id), group in cases.groupby(['fold', 'mask_id']):
            block = OUT / f'fold_{fold}_mask_{mask_id}'
            if block.exists():
                old = json.loads((block / 'summary.json').read_text())
                if old['manifest_sha256'] != sha(OUT / 'run_manifest.json') or old['assertion_failures']:
                    raise ValueError('checkpoint identity')
                for name, digest in old['output_hashes'].items():
                    if sha(block / name) != digest:
                        raise ValueError('checkpoint changed')
            else:
                run_block(panel, int(fold), mask_id, group, block, sha(OUT / 'run_manifest.json'))
            blocks.append(block)
    rows = pd.concat([pd.read_csv(b / 'case_metrics.csv') for b in blocks], ignore_index=True)
    if len(rows) != 5112 or len(blocks) != 48 or rows.duplicated(['profile_id', 'mask_id', 'method']).any():
        raise ValueError('aggregate case denominator')
    regional, macro = aggregate(rows)
    for name, frame in [('case_metrics', rows), ('region_metrics', regional), ('macro_metrics', macro)]:
        frame.to_csv(OUT / f'{name}.csv', index=False)
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('dependency changed during run')
    result = {'status': 'CONDITIONAL_SOURCE_SHAPE_MASKING_DIAGNOSTIC', 'assertion_failures': 0,
              'completed_blocks': len(blocks), 'score_rows': len(rows), 'input_hashes': hashes,
              'output_hashes': {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()},
              'limitations': ['Other source-world hourly fuel shapes and current coarse composition are supplied.',
                              'No claim of zero-local-hour target forecast performance or an original algorithm.',
                              'Repeated source development, shared 2024 time period and fixed-EF production proxy.']}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(macro.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
