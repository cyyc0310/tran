"""Generate PIT labels from inner region-held-out physical source models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from cif_pit_recalibration import (NestedPhysical, PhysicalReferences, REFERENCES, GRID,
                                    inner_roles, pit_bounds, source_samples)
from frozen_recomposition import FrozenRecomposition
from hourly_fuel_data import HourlyFuelData
from monthly_history_features import MonthlyInformation, monthly_context
from v151_hourly_law import EF, sha
from v170_fuel_marginal_diagnosis import verify

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/v159-fuel-activity-source-pilot-r1'
OUT = ROOT / 'results/v174-nested-pit-source-r1'
PROTOCOL = ROOT / 'docs/V174_CIF_RECALIBRATION_PROTOCOL.md'


def build_fold(data, monthly, fold):
    outer = FrozenRecomposition(data, BASE, fold, 'asof_30d')
    roles = inner_roles(outer.fit)
    samples = source_samples(BASE, outer.fit)
    raw = np.concatenate([samples[r] for r in sorted(outer.fit)])
    profiles = np.searchsorted(data.index.stop.to_numpy(), raw, side='right')
    regions = data.index.region.to_numpy()[profiles]
    inner = np.asarray([roles[r] for r in regions], int)
    lower, upper = np.full((len(raw), 2), np.nan), np.full((len(raw), 2), np.nan)
    snapped, outside = np.zeros((len(raw), 2), bool), np.zeros((len(raw), 2), bool)
    provenance, contexts = [], []
    for k in range(5):
        train = sorted(r for r in outer.fit if roles[r] != k)
        held = sorted(r for r in outer.fit if roles[r] == k)
        queries = np.unique(profiles[inner == k])
        physical = NestedPhysical(data, BASE, train, queries)
        predictor = PhysicalReferences(physical, monthly)
        provenance.append({'inner_fold': k, 'fit_regions': train, 'held_regions': held,
                           'query_profiles': queries.tolist(),
                           'coarse_training_profiles': predictor.coarse.ids.tolist(),
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
            # Only the original selected source hours may supply PIT outcomes.
            g = data.raw[raw[take]]
            actual = g @ EF / g.sum(axis=1)
            clock = data.states[raw[take]] % 24
            for j, reference in enumerate(REFERENCES):
                for state in np.unique(clock):
                    ids = take[clock == state]
                    L, R, changed, absent = pit_bounds(laws[reference][state], actual[clock == state])
                    lower[ids, j], upper[ids, j] = L, R
                    snapped[ids, j], outside[ids, j] = changed, absent
            if number % 20 == 0 or number == len(queries):
                print(f'outer {fold} inner {k}: {number}/{len(queries)} source months', flush=True)
    if (not np.isfinite(lower).all() or not np.isfinite(upper).all() or (lower < 0).any()
            or (upper > 1).any() or (upper < lower).any() or len(contexts) != len(np.unique(profiles))):
        raise ValueError('complete finite PIT archive')
    arrays = {'raw_id': raw, 'profile_id': profiles, 'region': np.asarray(regions, str), 'inner_fold': inner,
              'lower': lower, 'upper': upper, 'snapped': snapped, 'outside': outside}
    return arrays, provenance, contexts, outer.fit


def sha_bytes(ids):
    import hashlib
    return hashlib.sha256(np.asarray(ids, np.int64).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if OUT.exists() and not args.resume:
        raise FileExistsError(OUT)
    upstream = verify(BASE)
    extra = [PROTOCOL, Path(__file__), ROOT / 'scripts/cif_pit_recalibration.py',
             ROOT / 'scripts/coarse_distribution_lifting.py', ROOT / 'scripts/scenario_reweighting.py',
             ROOT / 'scripts/frozen_recomposition.py', ROOT / 'scripts/monthly_history_features.py',
             ROOT / 'scripts/fuel_activity_forest.py', ROOT / 'scripts/hourly_fuel_data.py',
             BASE / 'summary.json', BASE / 'source_sampling.csv']
    extra.extend(BASE / name for name in upstream['output_hashes'])
    hashes = upstream['input_hashes'] | {str(p.relative_to(ROOT)): sha(p) for p in extra}
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('input hash changed')
    manifest = json.dumps({'input_hashes': hashes, 'references': REFERENCES,
                           'grid': GRID.tolist(), 'scope': 'outer-proper inner-5-region OOF; no held CIF scoring'},
                          indent=2, sort_keys=True, allow_nan=False) + '\n'
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
                saved = json.loads((destination / 'summary.json').read_text())
                if saved['manifest_sha256'] != sha(OUT / 'run_manifest.json') or saved['assertion_failures']:
                    raise ValueError('invalid saved fold')
                for p, h in saved['output_hashes'].items():
                    if sha(destination / p) != h:
                        raise ValueError('saved output changed')
                continue
            arrays, provenance, contexts, fit = build_fold(data, monthly, fold)
            staging = Path(tempfile.mkdtemp(prefix='.v174-oof-', dir=OUT.parent))
            np.savez_compressed(staging / 'oof.npz', **arrays)
            (staging / 'provenance.json').write_text(json.dumps(provenance, indent=2, allow_nan=False) + '\n')
            (staging / 'contexts.json').write_text(json.dumps(contexts, indent=2, allow_nan=False) + '\n')
            summary = {'assertion_failures': 0, 'fold': fold, 'fit_regions': fit, 'source_hours': len(arrays['raw_id']),
                       'manifest_sha256': sha(OUT / 'run_manifest.json'), 'inner_models': 5,
                       'source_months': len(contexts), 'snapped': arrays['snapped'].sum(axis=0).tolist(),
                       'outside': arrays['outside'].sum(axis=0).tolist(),
                       'output_hashes': {p.name: sha(p) for p in staging.iterdir()}}
            (staging / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
            staging.rename(destination)
            print(f'completed OOF fold {fold}: {summary["source_hours"]} selected hours', flush=True)
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('inputs changed during OOF construction')
    summary = {'assertion_failures': 0, 'completed_blocks': 6, 'inner_models': 30,
               'input_hashes': hashes, 'outer_cal_test_scored': False,
               'output_hashes': {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()},
               'scope': 'Source-only nested OOF PIT supervision; neither a new method nor performance evidence.'}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
