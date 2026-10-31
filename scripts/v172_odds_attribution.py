"""Describe level drift and regional interval reallocation in frozen V172."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from v151_hourly_law import EF, sha
from v170_fuel_marginal_diagnosis import verify
from zero_odds_cif import REFERENCES, SETTINGS

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/v172-zero-odds-cif-r1'
AUDIT = ROOT / 'results/v172-zero-odds-audit-r1'
OUT = ROOT / 'results/v172-odds-attribution-r1'
DIAGNOSTICS = ['mean_kl', 'mean_ess_ratio', 'min_ess_ratio', 'mean_unreachable_zero_difference',
               'zero_error_before', 'zero_error_after', 'relative_fuel_mean_L1_shift',
               'generation_mean_relative_change', 'absolute_energy_anchor_shift']


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    summary = verify(SOURCE)
    verify(AUDIT)
    files = [Path(__file__), SOURCE / 'summary.json', AUDIT / 'summary.json']
    files.extend(SOURCE / name for name in summary['output_hashes'])
    hashes = summary['input_hashes'] | {str(p.relative_to(ROOT)): sha(p) for p in files}
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('input changed')
    rows = pd.read_csv(SOURCE / 'test_month_metrics.csv')
    hours = rows.groupby('profile_id').hours.first().to_dict()
    updates = pd.concat([pd.read_csv(SOURCE / f'fold_{fold}/zero_updates.csv') for fold in range(6)], ignore_index=True)
    updates = updates.loc[updates.role.eq('test')].copy()
    if len(updates) != 3856 or updates.duplicated(['profile_id', 'method']).any():
        raise ValueError('diagnostic denominator')
    records = []
    for rec in updates.itertuples(index=False):
        old, target, new = [np.asarray(json.loads(getattr(rec, key))) for key in ('prior_zero', 'target_zero', 'achieved_zero')]
        before, after = [np.asarray(json.loads(getattr(rec, key))) for key in ('fuel_before', 'fuel_after')]
        if before.sum() <= 0 or after.sum() <= 0:
            raise ValueError('invalid diagnostic generation moment')
        records.append({'profile_id': rec.profile_id, 'region': rec.region, 'method': rec.method, 'hours': hours[rec.profile_id],
                        **{k: getattr(rec, k) for k in DIAGNOSTICS[:4]},
                        'zero_error_before': float(abs(old - target).mean()), 'zero_error_after': float(abs(new - target).mean()),
                        'relative_fuel_mean_L1_shift': float(abs(before - after).sum() / before.sum()),
                        'generation_mean_relative_change': float(after.sum() / before.sum() - 1),
                        'absolute_energy_anchor_shift': float(abs(after @ EF / after.sum() - before @ EF / before.sum()))})
    diagnostics = pd.DataFrame(records)
    regional_diagnostics = pd.DataFrame([{'region': key[0], 'method': key[1],
                                         **{m: float(np.average(g[m], weights=g.hours)) for m in DIAGNOSTICS}}
                                        for key, g in diagnostics.groupby(['region', 'method'])])
    macro_diagnostics = regional_diagnostics.groupby('method')[DIAGNOSTICS].mean().reset_index()
    regional = pd.read_csv(SOURCE / 'region_metrics.csv')
    changes = []
    for rec in regional.loc[~regional.method.isin(REFERENCES)].itertuples(index=False):
        reference = SETTINGS[rec.method][0]
        old = regional.loc[regional.region.eq(rec.region) & regional.method.eq(reference)
                           & regional.calibration.eq(rec.calibration) & regional.radius_policy.eq('recalibrated')]
        if len(old) != 1:
            raise ValueError('reference identity')
        old = old.iloc[0]
        changes.append({'region': rec.region, 'method': rec.method, 'reference': reference,
                        'radius_policy': rec.radius_policy, 'calibration': rec.calibration,
                        'crps_difference': rec.crps - old.crps, 'coverage_before': old.coverage, 'coverage_after': rec.coverage,
                        'width_before': old.width, 'width_after': rec.width,
                        'width_change_percent': 100 * (rec.width / old.width - 1),
                        'winkler_difference': rec.winkler - old.winkler})
    changes = pd.DataFrame(changes)
    if len(changes) != 29 * 16 * 4 or not np.isfinite(diagnostics[DIAGNOSTICS]).all().all():
        raise ValueError('regional changes or diagnostic finiteness')
    macro_changes = []
    for key, group in changes.groupby(['method', 'radius_policy', 'calibration']):
        macro_changes.append(dict(zip(['method', 'radius_policy', 'calibration'], key)) | {
            'regions_wider': int((group.width_after > group.width_before + 1e-9).sum()),
            'regions_narrower': int((group.width_after < group.width_before - 1e-9).sum()),
            'regions_coverage_up_without_widening': int(((group.coverage_after > group.coverage_before + 1e-12)
                                                        & (group.width_after <= group.width_before + 1e-9)).sum())})
    macro = pd.read_csv(SOURCE / 'macro_metrics.csv')
    passing = macro.loc[macro.cohort.eq('all29') & ~macro.method.isin(REFERENCES)
                        & macro.coverage.ge(.9) & macro.worst_coverage.ge(.85)]
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('input changed during analysis')
    OUT.mkdir(parents=True)
    for name, frame in [('monthly_diagnostics', diagnostics), ('region_diagnostics', regional_diagnostics),
                        ('macro_diagnostics', macro_diagnostics), ('region_interval_changes', changes),
                        ('macro_interval_changes', pd.DataFrame(macro_changes)), ('coverage_only_candidates', passing)]:
        frame.to_csv(OUT / f'{name}.csv', index=False)
    result = {'assertion_failures': 0, 'diagnostic_rows': len(diagnostics), 'region_comparisons': len(changes),
              'coverage_only_candidates': len(passing), 'fixed_radius_coverage_only_candidates': int(passing.radius_policy.eq('fixed').sum()),
              'input_hashes': hashes, 'output_hashes': {p.name: sha(p) for p in OUT.iterdir()},
              'scope': 'Post-hoc source-development attribution. Energy-anchor shifts and regional widening are descriptive, not causal proof. Coverage-only candidates do not meet the full CIF gate.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('input_hashes', 'output_hashes')}, indent=2))


if __name__ == '__main__':
    main()
