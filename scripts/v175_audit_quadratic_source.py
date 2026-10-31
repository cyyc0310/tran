"""Refit every nested model and independently integrate every CRPS coefficient."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from cif_crps_quadratic import K, unpack
from cif_pit_recalibration import GRID
from hourly_fuel_data import HourlyFuelData
from monthly_history_features import MonthlyInformation
from v151_hourly_law import sha
from v170_fuel_marginal_diagnosis import verify
from v175_quadratic_source import ROOT, OUT as SOURCE, build_fold

OUT = ROOT / 'results/v175-quadratic-source-audit-r1'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    summary = verify(SOURCE)
    hashes = summary['input_hashes'] | {str((SOURCE / 'summary.json').relative_to(ROOT)): sha(SOURCE / 'summary.json'),
                                      str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('source inputs changed')
    totals = {'coefficient_rows': 0, 'law_states': 0, 'max_independent_coefficient_error': 0.}

    def independent(law, actual, saved):
        cumulative = np.clip(np.cumsum(law.w)[:-1], 0, 1)
        # Dense basis interpolation and overlap integration, independent of the
        # production bincount/tail-integral implementation.
        eye = np.eye(K)
        basis = np.column_stack([np.interp(cumulative, GRID, row) for row in eye])
        width = np.diff(law.x)
        H = basis.T @ (width[:, None] * basis) / 1000
        overlap = np.clip(law.x[1:, None] - np.maximum(law.x[:-1, None], actual[None]), 0, width[:, None])
        b = (basis.T @ overlap).T / 1000
        old_H, old_b, old_k = unpack(saved)
        error = max(float(np.max(abs(old_H - H[None]))), float(np.max(abs(old_b - b))),
                    float(np.max(abs(old_k - abs(law.x[-1] - actual) / 1000))))
        if error > 1e-12:
            raise ValueError('independent coefficient integration differs')
        totals['coefficient_rows'] += len(actual)
        totals['law_states'] += 1
        totals['max_independent_coefficient_error'] = max(totals['max_independent_coefficient_error'], error)

    data, monthly = HourlyFuelData(ROOT), MonthlyInformation(ROOT)
    records = []
    with threadpool_limits(limits=2):
        for fold in range(6):
            block = SOURCE / f'fold_{fold}'
            old_summary = verify(block)
            arrays, provenance, contexts, fit, error = build_fold(data, monthly, fold, checker=independent)
            with np.load(block / 'quadratics.npz', allow_pickle=False) as old:
                if set(arrays) != set(old.files) or any(not np.array_equal(v, old[k]) for k, v in arrays.items()):
                    raise ValueError('complete sufficient statistics not reproduced')
            if (fit != old_summary['fit_regions'] or provenance != json.loads((block / 'provenance.json').read_text())
                    or contexts != json.loads((block / 'contexts.json').read_text())
                    or old_summary['manifest_sha256'] != sha(SOURCE / 'run_manifest.json')):
                raise ValueError('source identities/provenance/manifest differ')
            records.append({'fold': fold, 'source_rows': len(arrays['raw_id']), 'inner_models_rebuilt': 5,
                            'max_original_CRPS_identity_error': error})
            print(f'quadratic source audit fold {fold}: all entries rebuilt and integrated', flush=True)
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('source inputs changed during audit')
    OUT.mkdir(parents=True)
    (OUT / 'reconstruction.json').write_text(json.dumps(records, indent=2, allow_nan=False) + '\n')
    result = {'assertion_failures': 0, 'inner_models_rebuilt': 30,
              'source_rows': sum(r['source_rows'] for r in records), **totals,
              'input_hashes': hashes, 'output_hashes': {'reconstruction.json': sha(OUT / 'reconstruction.json')},
              'scope': 'All source statistics exactly rebuilt; every H/b/constant independently integrated with dense basis/overlap. No held-CIF scoring.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
