"""Post-hoc source-optimization decrease versus realized held-source CRPS change."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from v151_hourly_law import sha
from v170_fuel_marginal_diagnosis import verify

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/v175-cif-crps-map-r1'
OUT = ROOT / 'results/v175-source-objective-diagnostic-r1'
INDEX = ROOT / 'results/v144-canonical-profile-cache-r1/profile_index.csv'
TIMES = ROOT / 'results/v153-calendar-times-r1/timestamps.npz'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    summary = verify(SOURCE)
    paths = [Path(__file__), SOURCE / 'summary.json', SOURCE / 'test_month_metrics.csv', INDEX, TIMES]
    paths.extend(SOURCE / f'fold_{k}/map_certificates.csv' for k in range(6))
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    index = pd.read_csv(INDEX).set_index('profile_id')
    with np.load(TIMES, allow_pickle=False) as saved:
        dates = pd.DatetimeIndex(pd.to_datetime(saved['unix_seconds'], unit='s', utc=True))
    states = dates.hour.to_numpy() + 24 * (dates.dayofweek.to_numpy() >= 5)
    rows = pd.read_csv(SOURCE / 'test_month_metrics.csv')
    rows = rows.loc[rows.radius_policy.eq('recalibrated') & rows.calibration.eq('pooled')]
    records = []
    for fold in range(6):
        cert = pd.read_csv(SOURCE / f'fold_{fold}/map_certificates.csv')
        cert = cert.loc[cert.role.eq('test')]
        for (q, method), group in cert.groupby(['profile_id', 'method']):
            if len(group) != 48 or set(group.state) != set(range(48)) or group.objective_gain.isna().any():
                raise ValueError('certificate application inventory')
            row = index.loc[q]
            weight = np.bincount(states[int(row.offset):int(row.stop)], minlength=48) / row.hours
            records.append({'fold': fold, 'profile_id': q, 'method': method,
                            'source_optimization_decrease': 1000 * float(weight @ group.sort_values('state').objective_gain.to_numpy()),
                            'source_objective': 'region_max_CRPS' if '_group_max_' in method else 'kernel_mean_CRPS'})
    paired = rows.merge(pd.DataFrame(records), on=['fold', 'profile_id', 'method'], validate='one_to_one')
    if len(paired) != 241 * 16:
        raise ValueError('source objective diagnostic denominator')
    paired['base_method'] = np.where(paired.method.str.startswith('gaussian'), 'world_gaussian', 'conditional_activity_recomposition')
    base = rows.loc[rows.method.isin(['world_gaussian', 'conditional_activity_recomposition']), ['profile_id', 'method', 'crps']]
    base = base.rename(columns={'method': 'base_method', 'crps': 'base_crps'})
    paired = paired.merge(base, on=['profile_id', 'base_method'], validate='many_to_one')
    paired['realized_CRPS_gain'] = paired.base_crps - paired.crps
    regional = pd.DataFrame([{'method': key[0], 'region': key[1], 'source_objective': group.source_objective.iloc[0],
                              **{p: float(np.average(group[p], weights=group.hours)) for p in
                                 ['source_optimization_decrease', 'realized_CRPS_gain']}}
                             for key, group in paired.groupby(['method', 'region'])])
    macro = regional.groupby(['method', 'source_objective'])[['source_optimization_decrease', 'realized_CRPS_gain']].mean().reset_index()
    if len(regional) != 16 * 29 or len(macro) != 16 or not np.isfinite(macro.select_dtypes('number')).all().all():
        raise ValueError('source diagnostic aggregate inventory')
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('source diagnostic input changed')
    OUT.mkdir(parents=True)
    for name, frame in [('query_gains', paired), ('region_gains', regional), ('macro_gains', macro)]:
        frame.to_csv(OUT / f'{name}.csv', index=False)
    result = {'assertion_failures': 0, 'query_rows': len(paired), 'input_hashes': hashes,
              'output_hashes': {p.name: sha(p) for p in OUT.iterdir()},
              'scope': 'Post-hoc descriptive optimization diagnostic. Source objective decreases are fit-mixture quantities, not identified target-risk gains; group-max and held macro-mean are different objectives. No causal attribution or new model selection.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(macro.to_string(index=False))


if __name__ == '__main__':
    main()
