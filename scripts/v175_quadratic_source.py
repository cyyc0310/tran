"""Archive exact source-only CRPS sufficient statistics for nested OOF laws."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
from threadpoolctl import threadpool_limits

from cif_crps_quadratic import PACKED, statistics, value
from cif_pit_recalibration import (GRID, REFERENCES, NestedPhysical, PhysicalReferences,
                                   inner_roles, source_samples, pit_bounds)
from frozen_recomposition import FrozenRecomposition
from hourly_fuel_data import HourlyFuelData
from monthly_history_features import MonthlyInformation, monthly_context
from v151_hourly_law import EF, sha
from v170_fuel_marginal_diagnosis import verify
from v174_nested_pit_archive import sha_bytes

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/v159-fuel-activity-source-pilot-r1'
PIT = ROOT / 'results/v174-nested-pit-source-r1'
OUT = ROOT / 'results/v175-quadratic-source-r1'
PROTOCOL = ROOT / 'docs/V175_DIRECT_CRPS_MAP_PROTOCOL.md'


def build_fold(data, monthly, fold, checker=None):
    outer = FrozenRecomposition(data, BASE, fold, 'asof_30d')
    roles = inner_roles(outer.fit)
    samples = source_samples(BASE, outer.fit)
    raw = np.concatenate([samples[r] for r in sorted(outer.fit)])
    profiles = np.searchsorted(data.index.stop.to_numpy(), raw, side='right')
    regions = data.index.region.to_numpy()[profiles]
    inner = np.asarray([roles[r] for r in regions], int)
    stats = np.full((len(raw), 2, PACKED), np.nan)
    lower, upper = np.full((len(raw), 2), np.nan), np.full((len(raw), 2), np.nan)
    provenance, contexts = [], []
    max_error = 0.
    for k in range(5):
        train = sorted(r for r in outer.fit if roles[r] != k)
        held = sorted(r for r in outer.fit if roles[r] == k)
        queries = np.unique(profiles[inner == k])
        physical = NestedPhysical(data, BASE, train, queries)
        predictor = PhysicalReferences(physical, monthly)
        provenance.append({'inner_fold': k, 'fit_regions': train, 'held_regions': held,
                           'query_profiles': queries.tolist(), 'coarse_training_profiles': predictor.coarse.ids.tolist(),
                           'carrier_prior': physical.prior.tolist(),
                           'fit_raw_index_sha256': sha_bytes(np.concatenate([samples[r] for r in train]))})
        for number, q in enumerate(queries, start=1):
            q = int(q)
            laws, _ = predictor.build(q)
            context = monthly_context(monthly, q)
            contexts.append({'profile_id': q, 'region': data.index.region.iloc[q], 'inner_fold': k,
                             'local': context['local'], 'prefix': context['prefix']})
            take = np.flatnonzero(profiles == q)
            if not np.all(inner[take] == k) or data.index.region.iloc[q] not in held:
                raise ValueError('inner source identity')
            g = data.raw[raw[take]]
            actual = g @ EF / g.sum(axis=1)
            clock = data.states[raw[take]] % 24
            for j, reference in enumerate(REFERENCES):
                for state in np.unique(clock):
                    selected = clock == state
                    ids = take[selected]
                    law = laws[reference][state]
                    stats[ids, j] = statistics(law, actual[selected])
                    if checker is not None:
                        checker(law, actual[selected], stats[ids, j])
                    L, R, _, _ = pit_bounds(law, actual[selected])
                    lower[ids, j], upper[ids, j] = L, R
                    delta = value(stats[ids, j], GRID, 1.) * 1000 - law.crps(actual[selected])
                    max_error = max(max_error, float(max(abs(delta))))
            if number % 20 == 0 or number == len(queries):
                print(f'quadratic outer {fold} inner {k}: {number}/{len(queries)} months', flush=True)
    with np.load(PIT / f'fold_{fold}/oof.npz', allow_pickle=False) as old:
        if (not np.array_equal(raw, old['raw_id']) or not np.array_equal(profiles, old['profile_id'])
                or not np.array_equal(regions, old['region']) or not np.array_equal(inner, old['inner_fold'])
                or not np.array_equal(lower, old['lower']) or not np.array_equal(upper, old['upper'])):
            raise ValueError('V174 complete OOF laws/PIT not reproduced')
    if (not np.isfinite(stats).all() or stats.min() < -1e-12 or max_error > 1e-7
            or provenance != json.loads((PIT / f'fold_{fold}/provenance.json').read_text())
            or contexts != json.loads((PIT / f'fold_{fold}/contexts.json').read_text())):
        raise ValueError('quadratic identities or provenance failed')
    arrays = {'raw_id': raw, 'profile_id': profiles, 'region': np.asarray(regions, str),
              'inner_fold': inner, 'statistics': stats}
    return arrays, provenance, contexts, outer.fit, max_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if OUT.exists() and not args.resume:
        raise FileExistsError(OUT)
    archives = [BASE, PIT, ROOT / 'results/v174-nested-pit-audit-r1', ROOT / 'results/v175-quadratic-preflight-r2']
    hashes = {}
    for directory in archives:
        summary = verify(directory)
        hashes |= summary.get('input_hashes', {})
        for name in ['summary.json', *summary['output_hashes']]:
            hashes[str((directory / name).relative_to(ROOT))] = sha(directory / name)
    paths = [PROTOCOL, Path(__file__), ROOT / 'scripts/cif_crps_quadratic.py',
             ROOT / 'scripts/cif_pit_recalibration.py', ROOT / 'scripts/v174_nested_pit_archive.py']
    hashes |= {str(p.relative_to(ROOT)): sha(p) for p in paths}
    for directory in archives:
        prior = json.loads((directory / 'summary.json').read_text()).get('input_hashes', {})
        if any(hashes[p] != h for p, h in prior.items()):
            raise ValueError('upstream input disagreement')
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('input changed')
    manifest = json.dumps({'input_hashes': hashes, 'references': REFERENCES, 'packed_dimension': PACKED,
                           'held_CIF_scored': False}, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if OUT.exists():
        if (OUT / 'run_manifest.json').read_text() != manifest or (OUT / 'summary.json').exists():
            raise ValueError('invalid resume')
    else:
        OUT.mkdir(parents=True)
        (OUT / 'run_manifest.json').write_text(manifest)
    data, monthly = HourlyFuelData(ROOT), MonthlyInformation(ROOT)
    with threadpool_limits(limits=2):
        for fold in range(6):
            destination = OUT / f'fold_{fold}'
            if destination.exists():
                saved = verify(destination)
                if saved['manifest_sha256'] != sha(OUT / 'run_manifest.json'):
                    raise ValueError('saved block manifest')
                continue
            arrays, provenance, contexts, fit, max_error = build_fold(data, monthly, fold)
            staging = Path(tempfile.mkdtemp(prefix='.v175-stats-', dir=OUT.parent))
            np.savez_compressed(staging / 'quadratics.npz', **arrays)
            (staging / 'provenance.json').write_text(json.dumps(provenance, indent=2, allow_nan=False) + '\n')
            (staging / 'contexts.json').write_text(json.dumps(contexts, indent=2, allow_nan=False) + '\n')
            result = {'assertion_failures': 0, 'fold': fold, 'fit_regions': fit, 'source_rows': len(arrays['raw_id']),
                      'max_original_CRPS_identity_error': max_error, 'inner_models': 5,
                      'manifest_sha256': sha(OUT / 'run_manifest.json'),
                      'output_hashes': {p.name: sha(p) for p in staging.iterdir()}}
            (staging / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
            staging.rename(destination)
            print(f'completed quadratic fold {fold}: {len(arrays["raw_id"])} selected rows', flush=True)
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('input changed during statistics archive')
    result = {'assertion_failures': 0, 'completed_blocks': 6, 'inner_models': 30,
              'held_CIF_scored': False, 'input_hashes': hashes,
              'output_hashes': {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()},
              'scope': 'Exact source-only OOF CRPS sufficient statistics; all V174 PIT labels and training provenance reproduced.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
