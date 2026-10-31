"""Predictable-stratum summaries and source-month-zero decomposition of V170."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hourly_fuel_data import HourlyFuelData
from v151_hourly_law import sha
from v170_fuel_marginal_diagnosis import FUELS, SOURCE, verify

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'results/v170-fuel-marginal-diagnosis-r1'
OUT = ROOT / 'results/v170-predictable-strata-r1'
METRICS = ('predicted_zero_probability', 'actual_zero_fraction', 'zero_brier', 'coverage',
           'source_month_zero_probability', 'within_positive_source_month_zero_probability',
           'actual_month_zero', 'month_zero_brier', 'persistence_month_zero_brier')


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    summary = verify(ARCHIVE)
    reference = verify(SOURCE)
    files = [Path(__file__), ARCHIVE / 'summary.json', ARCHIVE / 'test_month_metrics.csv', SOURCE / 'summary.json']
    files.extend(SOURCE / name for name in reference['output_hashes'])
    hashes = summary['input_hashes'] | {str(p.relative_to(ROOT)): sha(p) for p in files}
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('input changed')
    data = HourlyFuelData(ROOT)
    rows = pd.read_csv(ARCHIVE / 'test_month_metrics.csv')
    rows = rows.loc[rows.arm.eq('legal')].copy()
    values = []
    for fold in range(6):
        worlds = pd.read_csv(SOURCE / f'fold_{fold}/world_probabilities.csv')
        for rec in rows.loc[rows.fold.eq(fold)].itertuples(index=False):
            q, j = int(rec.profile_id), FUELS.index(rec.fuel)
            local = data.local(q, 'asof_30d')
            if rec.input_fuel_positive != (data.M[local, j] > 0):
                raise ValueError('input stratum not measurable at forecast origin')
            selected = worlds.loc[worlds.profile_id.eq(q)]
            weight = selected['conditional_weight' if rec.method == 'conditional_activity_recomposition' else 'gaussian_weight'].to_numpy()
            month_zero = data.M[selected.source_profile_id.to_numpy(int), j] == 0
            probability = float(weight @ month_zero)
            residual = rec.predicted_zero_probability - probability
            if residual < -1e-12:
                raise ValueError('whole-month zero exceeds total zero probability')
            actual_zero = bool(data.M[q, j] == 0)
            persistence = bool(data.M[local, j] == 0)
            values.append(rec._asdict() | {'source_month_zero_probability': probability,
                          'within_positive_source_month_zero_probability': max(0., residual),
                          'actual_month_zero': float(actual_zero), 'month_zero_brier': (probability - actual_zero)**2,
                          'persistence_month_zero_brier': float(persistence != actual_zero)})
    records = pd.DataFrame(values)
    if len(records) != 3856 or records.duplicated(['profile_id', 'method', 'fuel']).any():
        raise ValueError('legal row denominator')
    regional = []
    for cohort, table in [('all29', records), ('input_fuel_positive', records.loc[records.input_fuel_positive])]:
        for key, group in table.groupby(['region', 'method', 'fuel']):
            regional.append({'cohort': cohort, **dict(zip(['region', 'method', 'fuel'], key)),
                             **{m: float(np.average(group[m], weights=group.hours)) for m in METRICS}})
    regional = pd.DataFrame(regional)
    macro = pd.DataFrame([{'cohort': key[0], 'method': key[1], 'fuel': key[2], 'regions': len(group),
                          **{m: float(group[m].mean()) for m in METRICS}}
                         for key, group in regional.groupby(['cohort', 'method', 'fuel'])])
    if (not np.isfinite(records[list(METRICS)]).all().all()
            or any(sha(ROOT / name) != digest for name, digest in hashes.items())):
        raise ValueError('finite or dependency check')
    OUT.mkdir(parents=True)
    for name, frame in [('month_components', records), ('region_metrics', regional), ('macro_metrics', macro)]:
        frame.to_csv(OUT / f'{name}.csv', index=False)
    result = {'assertion_failures': 0, 'legal_records': len(records), 'input_hashes': hashes,
              'output_hashes': {p.name: sha(p) for p in OUT.iterdir()},
              'scope': 'Post-hoc decomposition of frozen source forecasts, with forecast-origin measurable strata. Source whole-month zero is not certified capacity absence. No CIF model or gate result.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(macro.loc[macro.cohort.eq('input_fuel_positive') & macro.method.eq('world_gaussian')].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
