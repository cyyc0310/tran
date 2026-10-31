"""Export coarse-only history and verify canonical 2024 overlaps exactly."""
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import pandas as pd

from coarse_history_export import load_native_month_inputs, annual_margins, FUELS
from v151_hourly_law import sha
from v170_fuel_marginal_diagnosis import verify

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/v159-fuel-activity-source-pilot-r1'
CANONICAL = ROOT / 'results/v144-canonical-profile-cache-r1'
OUT = ROOT / 'results/v183-coarse-history-export-r1'
PROTOCOL = ROOT / 'docs/V183_COARSE_HISTORY_EXPORT_PROTOCOL.md'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    verify(BASE)
    verify(ROOT / 'results/v182-coarse-history-inventory-r2')
    canonical = json.loads((CANONICAL / 'summary.json').read_text())
    paths = [Path(p) for p in canonical['raw_source_hashes'] if Path(p).name.startswith('gen_')
             and any(year in Path(p).name for year in ('2022', '2023', '2024'))]
    if len(paths) != 6 or len({p.name for p in paths}) != 6:
        raise ValueError('six original historical snapshots required')
    hashes = {str(p): canonical['raw_source_hashes'][str(p)] for p in paths}
    dependencies = [PROTOCOL, Path(__file__), ROOT / 'scripts/coarse_history_export.py',
                    ROOT / 'src/transcif/data/eia930.py', ROOT / 'src/transcif/research/eia_proxy.py',
                    CANONICAL / 'summary.json', CANONICAL / 'profiles.npz', CANONICAL / 'profile_index.csv',
                    BASE / 'summary.json', BASE / 'source_sampling.csv']
    hashes |= {str(p.relative_to(ROOT)): sha(p) for p in dependencies}
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('canonical/history input snapshot changed')
    regions = sorted(pd.read_csv(BASE / 'source_sampling.csv').region.unique())
    if len(regions) != 29:
        raise ValueError('source roster differs')
    native, schemas, duplicates = load_native_month_inputs(paths, regions)
    monthly, audits = annual_margins(native, schemas, regions)
    if monthly.duplicated(['region', 'month']).any():
        raise ValueError('duplicate exported monthly record')
    index = pd.read_csv(CANONICAL / 'profile_index.csv')
    overlap = index.loc[index.region.isin(regions) & index.month.str.startswith('2024')]
    lookup = monthly.set_index(['region', 'month'])
    with np.load(CANONICAL / 'profiles.npz', allow_pickle=False) as packed:
        old = packed['margin_fractions']
        checks = []
        for row in overlap.itertuples(index=False):
            current = lookup.loc[(row.region, row.month)]
            if current.observed_hours != row.hours:
                raise ValueError(f'canonical overlap hours differ: {row.region}/{row.month}')
            for j, fuel in enumerate(FUELS):
                if Fraction(current[f'margin_{fuel}']) != Fraction(str(old[row.profile_id, j])):
                    raise ValueError(f'canonical overlap margin differs: {row.region}/{row.month}/{fuel}')
            checks.append({'profile_id': row.profile_id, 'region': row.region, 'month': row.month,
                           'observed_hours': row.hours, 'all_eight_exact_margins_match': True})
    history = monthly.loc[monthly.year.isin([2022, 2023])].copy()
    if history.empty or set(history.region) != set(regions):
        raise ValueError('coarse history source coverage missing')
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('coarse history input changed during export')
    OUT.mkdir(parents=True)
    for name, frame in [('monthly_history', history), ('canonical_overlap', pd.DataFrame(checks)),
                        ('overlap_year_monthly_qa_only', monthly.loc[monthly.year.eq(2024)])]:
        frame.to_csv(OUT / f'{name}.csv', index=False)
    (OUT / 'source_quality_audits.json').write_text(json.dumps(audits, indent=2, allow_nan=False) + '\n')
    result = {'assertion_failures': 0, 'source_regions': len(regions), 'historical_months_exported': len(history),
              'canonical_overlap_profiles': len(checks), 'canonical_exact_fuel_margin_checks': len(checks) * 8,
              'duplicate_native_rows_earliest_precedence': duplicates, 'year_source_audits': len(audits),
              'hourly_arrays_exported': False, 'cif_computed': False, 'models_fitted': False,
              'input_hashes': hashes, 'output_hashes': {p.name: sha(p) for p in OUT.iterdir()},
              'scope': 'Isolated retrospective canonical coarse-only history, 29 existing sources. Year-window quality matches V144 2024 exactly. Synthetic release/vintage and production-proxy limitations remain; not a performance or originality claim.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if 'hash' not in k}, indent=2), flush=True)


if __name__ == '__main__':
    main()
