"""Build all maps for legal source-law queries without reading held hourly labels."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from cif_crps_recalibration import CRPSRecalibration, SETTINGS
from frozen_recomposition import FrozenRecomposition
from hourly_fuel_data import HourlyFuelData
from monthly_history_features import MonthlyInformation
from v151_hourly_law import sha
from v170_fuel_marginal_diagnosis import verify

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/v159-fuel-activity-source-pilot-r1'
PIT = ROOT / 'results/v174-nested-pit-source-r1'
STATS = ROOT / 'results/v175-quadratic-source-r1'
OUT = ROOT / 'results/v175-source-map-preflight-r1'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    hashes = {}
    for directory in (BASE, PIT, STATS, ROOT / 'results/v175-quadratic-source-audit-r1'):
        s = verify(directory)
        hashes |= s['input_hashes']
        for p in ['summary.json', *s['output_hashes']]:
            hashes[str((directory / p).relative_to(ROOT))] = sha(directory / p)
    hashes |= {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), ROOT / 'scripts/cif_crps_recalibration.py']}
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('preflight dependency changed')
    data, monthly = HourlyFuelData(ROOT), MonthlyInformation(ROOT)
    records = []
    max_kernel = 0.
    started = time.monotonic()
    with threadpool_limits(limits=2):
        for fold in range(6):
            frozen = FrozenRecomposition(data, BASE, fold, 'asof_30d')
            predictor = CRPSRecalibration(frozen, monthly, PIT / f'fold_{fold}', STATS / f'fold_{fold}')
            for q in frozen.queries.profile_id.iloc[[0, -1]]:
                _, _, details, certificates = predictor.build(int(q))
                records.extend({'fold': fold, 'profile_id': int(q), **c} for c in certificates)
                max_kernel = max(max_kernel, max(d['kernel_pit_error'] for d in details))
            print(f'source map preflight fold {fold}', flush=True)
    if len(records) != 9216:
        raise ValueError('source map preflight denominator')
    success = [c for c in records if not c['used_identity_fallback']]
    for c in success:
        limit = 1e-6 if SETTINGS[c['method']][1] == 'group_max' else 1e-7
        if c['kkt_error'] is None or c['kkt_error'] > limit or c['feasibility_error'] > 1e-9:
            raise ValueError('preflight successful map not certified')
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('preflight dependency changed during run')
    OUT.mkdir(parents=True)
    pd.DataFrame(records).to_csv(OUT / 'certificates.csv', index=False)
    result = {'assertion_failures': 0, 'certificate_applications': len(records),
              'identity_fallback_applications': sum(c['used_identity_fallback'] for c in records),
              'max_kernel_PIT_error': max_kernel, 'seconds': time.monotonic() - started,
              'held_CIF_scored': False, 'input_hashes': hashes,
              'output_hashes': {'certificates.csv': sha(OUT / 'certificates.csv')}}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if 'hash' not in k}, indent=2))


if __name__ == '__main__':
    main()
