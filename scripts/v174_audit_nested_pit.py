"""Refit every nested physical model and verify all archived source PIT labels."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from cif_pit_recalibration import GRID, inner_roles, pit_targets, source_samples
from hourly_fuel_data import HourlyFuelData
from monthly_history_features import MonthlyInformation
from v151_hourly_law import sha
from v170_fuel_marginal_diagnosis import verify
from v174_nested_pit_archive import BASE, OUT as SOURCE, ROOT, build_fold, sha_bytes

OUT = ROOT / 'results/v174-nested-pit-audit-r1'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    summary = verify(SOURCE)
    hashes = summary['input_hashes'] | {str((SOURCE / 'summary.json').relative_to(ROOT)): sha(SOURCE / 'summary.json'),
                                      str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('OOF input changed')
    data, monthly = HourlyFuelData(ROOT), MonthlyInformation(ROOT)
    records = []
    with threadpool_limits(limits=2):
        for fold in range(6):
            block = SOURCE / f'fold_{fold}'
            saved_summary = verify(block)
            arrays, provenance, contexts, fit = build_fold(data, monthly, fold)
            if saved_summary['fit_regions'] != fit or saved_summary['manifest_sha256'] != sha(SOURCE / 'run_manifest.json'):
                raise ValueError('outer fit/manifest mismatch')
            with np.load(block / 'oof.npz', allow_pickle=False) as old:
                if set(old.files) != set(arrays):
                    raise ValueError('OOF axes differ')
                for key, array in arrays.items():
                    if not np.array_equal(array, old[key]):
                        raise ValueError(f'OOF rebuild differs: {fold}/{key}')
            if (provenance != json.loads((block / 'provenance.json').read_text())
                    or contexts != json.loads((block / 'contexts.json').read_text())):
                raise ValueError('training or legal-prefix provenance differs')
            samples = source_samples(BASE, fit)
            roles = inner_roles(fit)
            expected_raw = np.concatenate([samples[r] for r in sorted(fit)])
            if not np.array_equal(expected_raw, arrays['raw_id']):
                raise ValueError('raw source budget changed')
            for record in provenance:
                k = record['inner_fold']
                train = sorted(r for r in fit if roles[r] != k)
                held = sorted(r for r in fit if roles[r] == k)
                if (record['fit_regions'] != train or record['held_regions'] != held
                        or set(train) & set(held) or set(train) | set(held) != set(fit)
                        or record['fit_raw_index_sha256'] != sha_bytes(np.concatenate([samples[r] for r in train]))):
                    raise ValueError('inner training isolation mismatch')
                ids = record['coarse_training_profiles']
                if not set(monthly.index.region.iloc[ids]).issubset(train) or not monthly.index.month.iloc[ids].str.startswith('2024').all():
                    raise ValueError('coarse training source/year leakage')
            for context in contexts:
                q, prefix = context['profile_id'], context['prefix']
                if (context['local'] != data.local(q, 'asof_30d') or context['local'] != prefix[-1]
                        or not (monthly.release.iloc[prefix] <= monthly.dates.iloc[q]).all()
                        or set(monthly.index.region.iloc[prefix]) != {context['region']}
                        or roles[context['region']] != context['inner_fold']):
                    raise ValueError('released source prefix mismatch')
            max_error = 0.
            for j in range(2):
                L, R = arrays['lower'][:, j], arrays['upper'][:, j]
                # An independent expanded interval-CDF expression, no RNG jitter.
                width = R - L
                independent = np.empty((len(L), len(GRID) - 2))
                for k, u in enumerate(GRID[1:-1]):
                    value = np.asarray(u >= L, float)
                    positive = width > 0
                    value[positive] = (np.minimum(np.maximum(u, L[positive]), R[positive]) - L[positive]) / width[positive]
                    independent[:, k] = value
                max_error = max(max_error, float(np.max(abs(independent - pit_targets(L, R)))))
            if max_error > 1e-12:
                raise ValueError('independent PIT target formula differs')
            records.append({'fold': fold, 'source_hours': len(expected_raw), 'source_months': len(contexts),
                            'inner_models_rebuilt': 5, 'probability_entries_rebuilt': int(arrays['lower'].size * 2),
                            'max_independent_PIT_target_error': max_error})
            print(f'OOF audit fold {fold}: all source labels exactly rebuilt', flush=True)
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('input changed during OOF audit')
    OUT.mkdir(parents=True)
    (OUT / 'reconstruction.json').write_text(json.dumps(records, indent=2, allow_nan=False) + '\n')
    result = {'assertion_failures': 0, 'inner_models_rebuilt': 30,
              'selected_source_rows': sum(r['source_hours'] for r in records),
              'PIT_bound_entries_rebuilt': sum(r['probability_entries_rebuilt'] for r in records),
              'input_hashes': hashes, 'output_hashes': {'reconstruction.json': sha(OUT / 'reconstruction.json')},
              'scope': 'All nested models refitted and PIT bounds bit-exactly rebuilt using frozen helpers; independent analytic PIT-CDF target expression verified. No held CIF scores.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
