"""Rebuild all exported monthly margins with independent Decimal summation."""
from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import pandas as pd

from coarse_history_export import load_native_month_inputs, FUELS, _proxy_rows, MIN_MONTH_COVERAGE
from v151_hourly_law import sha
from v170_fuel_marginal_diagnosis import verify

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/v183-coarse-history-export-r1'
OUT = ROOT / 'results/v183-coarse-history-export-audit-r1'
CANONICAL = ROOT / 'results/v144-canonical-profile-cache-r1'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    source = verify(SOURCE)
    hashes = source['input_hashes'] | {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
                                      str((SOURCE / 'summary.json').relative_to(ROOT)): sha(SOURCE / 'summary.json')}
    for name in source['output_hashes']:
        hashes[str((SOURCE / name).relative_to(ROOT))] = sha(SOURCE / name)
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('monthly audit input changed')
    history = pd.read_csv(SOURCE / 'monthly_history.csv')
    qa = pd.read_csv(SOURCE / 'overlap_year_monthly_qa_only.csv')
    allowed = {'region', 'month', 'year', 'period_start', 'period_end', 'available_at',
               'observed_hours', 'calendar_hours', 'coverage', *[f'margin_{f}' for f in FUELS]}
    if set(history.columns) != allowed or set(qa.columns) != allowed or not set(history.year).issubset({2022, 2023}) or not qa.year.eq(2024).all():
        raise ValueError('monthly field/year whitelist')
    existing = pd.concat([history, qa], ignore_index=True)
    if existing.duplicated(['region', 'month']).any() or len(history) != source['historical_months_exported']:
        raise ValueError('monthly key/record inventory')
    regions = sorted(history.region.unique())
    roster = sorted(pd.read_csv(ROOT / 'results/v159-fuel-activity-source-pilot-r1/source_sampling.csv').region.unique())
    if regions != roster or len(regions) != 29:
        raise ValueError('historical source roster differs')
    canonical = json.loads((CANONICAL / 'summary.json').read_text())
    paths = [Path(p) for p in canonical['raw_source_hashes'] if Path(p).name.startswith('gen_')
             and any(y in Path(p).name for y in ('2022', '2023', '2024'))]
    native, schemas, duplicates = load_native_month_inputs(paths, regions)
    old_audits = json.loads((SOURCE / 'source_quality_audits.json').read_text())
    rows = []
    with localcontext() as context:
        context.prec = 64
        for region in regions:
            for year in (2022, 2023, 2024):
                selected = native.loc[native.region.eq(region) & native.valid_at.dt.year.eq(year)].copy()
                selected.index = pd.DatetimeIndex(selected.valid_at)
                hourly, audit = _proxy_rows(selected, region, schemas)
                if audit != old_audits[f'{region}/{year}']:
                    raise ValueError('source/year quality audit differs')
                hourly.index = pd.DatetimeIndex(hourly.time)
                for date, group in hourly.groupby(pd.Grouper(freq='MS')):
                    end = date + pd.offsets.MonthBegin(1)
                    hours = int((end - date) / pd.Timedelta(hours=1))
                    values = group.loc[group.valid, [f'generation_{f}' for f in FUELS]].to_numpy(float)
                    n = len(values)
                    if not n or n / hours < MIN_MONTH_COVERAGE:
                        continue
                    margins = [Fraction(sum((Decimal(str(float(v))) for v in values[:, f]), Decimal(0))) for f in range(8)]
                    if sum(margins) <= 0:
                        continue
                    rows.append({'region': region, 'month': date.strftime('%Y-%m-01'), 'year': year,
                                 'period_start': date.isoformat(), 'period_end': end.isoformat(),
                                 'available_at': (end + pd.Timedelta(days=30)).isoformat(),
                                 'observed_hours': n, 'calendar_hours': hours, 'coverage': n / hours,
                                 **{f'margin_{f}': str(v) for f, v in zip(FUELS, margins)}})
            print(f'Decimal monthly audit {region}', flush=True)
    fresh = pd.DataFrame(rows)
    keys = ['region', 'month']
    left, right = [t.sort_values(keys).reset_index(drop=True) for t in (fresh, existing)]
    if len(left) != len(right) or left.duplicated(keys).any():
        raise ValueError('complete monthly reconstruction inventory')
    for name in allowed:
        if name.startswith('margin_'):
            if any(Fraction(str(a)) != Fraction(str(b)) for a, b in zip(left[name], right[name])):
                raise ValueError(f'independent Decimal margins differ: {name}')
        elif name == 'coverage':
            if not np.isfinite(right[name]).all() or max(abs(left[name] - right[name])) > 1e-12:
                raise ValueError('monthly coverage metadata differs')
        elif not np.array_equal(left[name], right[name]):
            raise ValueError(f'monthly metadata differs: {name}')
    index = pd.read_csv(CANONICAL / 'profile_index.csv')
    expected = index.loc[index.region.isin(regions) & index.month.str.startswith('2024')]
    checks = pd.read_csv(SOURCE / 'canonical_overlap.csv')
    if set(checks.profile_id) != set(expected.profile_id) or checks.profile_id.duplicated().any() or len(checks) != 327:
        raise ValueError('canonical overlap inventory')
    lookup = fresh.set_index(keys)
    with np.load(CANONICAL / 'profiles.npz', allow_pickle=False) as packed:
        margins = packed['margin_fractions']
        for rec in expected.itertuples(index=False):
            row = lookup.loc[(rec.region, rec.month)]
            if row.observed_hours != rec.hours:
                raise ValueError('canonical observed hours differ')
            for f, fuel in enumerate(FUELS):
                if Fraction(row[f'margin_{fuel}']) != Fraction(str(margins[rec.profile_id, f])):
                    raise ValueError('canonical exact Decimal overlap differs')
    if len(old_audits) != 87 or duplicates != source['duplicate_native_rows_earliest_precedence']:
        raise ValueError('source quality scope differs')
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('monthly audit inputs changed during run')
    OUT.mkdir(parents=True)
    fresh.to_csv(OUT / 'monthly_reconstruction.csv', index=False)
    summary = {'assertion_failures': 0, 'monthly_records_rebuilt': len(fresh), 'historical_records_rebuilt': len(history),
               'independent_decimal_margin_checks': len(fresh) * 8, 'canonical_overlap_profiles': len(expected),
               'canonical_exact_margin_checks': len(expected) * 8, 'source_year_audits_rebuilt': len(old_audits),
               'monthly_field_whitelist_passed': True, 'hourly_arrays_exported': False, 'cif_computed': False,
               'input_hashes': hashes, 'output_hashes': {'monthly_reconstruction.csv': sha(OUT / 'monthly_reconstruction.csv')},
               'scope': 'Every exported historical/QA monthly record rebuilt with independent Decimal summation and exact canonical overlap. Native normalization and quality helper shared; not independent fuel-quality validation or public-vintage evidence.'}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if 'hash' not in k}, indent=2), flush=True)


if __name__ == '__main__':
    main()
