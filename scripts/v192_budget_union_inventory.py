"""Metadata-only audit of per-origin versus cross-origin unique source labels."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from v151_hourly_law import sha

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/v192-reallocated-kappa-r1'
OUT = ROOT / 'results/v192-budget-union-inventory-r1'
ORIGINS = ('2024-04-01', '2024-05-01', '2024-06-01')


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    provenance = ROOT / 'results/v188-forward-zero-risk-r1/fold_0/training_provenance.json'
    matrix = ROOT / 'results/v188-forward-zero-risk-r1/fold_0/training_matrices.npz'
    roster = json.loads(provenance.read_text())['proper_regions']
    with np.load(matrix, allow_pickle=False) as p:
        names, counts = np.unique(p['region'], return_counts=True)
    budgets = dict(zip(names.tolist(), counts.tolist()))
    paths = [Path(__file__), ROOT / 'scripts/v151_hourly_law.py', provenance, matrix,
             SOURCE / 'summary.json', *sorted(SOURCE.glob('*/*_source.json'))]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    if len(paths) != 119:
        raise ValueError('source JSON file count')
    by_origin: dict[tuple, dict] = {}
    for path in paths[5:]:
        allocation = path.name.split('_')[0]
        origin, target = path.parent.name.split('_', 1)
        if allocation not in ('spread', 'recent') or origin not in ORIGINS or target not in roster:
            raise ValueError('unregistered identity')
        source = json.loads(path.read_text())
        if set(r['region'] for r in source) != set(roster) - {target}:
            raise ValueError('held region/source roster')
        local: dict[str, set] = {}
        for row in source:
            region = row['region']
            ids = set(row['raw_ids'])
            if len(ids) != len(row['raw_ids']) or local.setdefault(region, set()) & ids:
                raise ValueError('duplicate sample ID')
            local[region].update(ids)
        for region, ids in local.items():
            if len(ids) != budgets[region]:
                raise ValueError('per-fit region budget')
            key = (allocation, origin, region)
            if key in by_origin and by_origin[key] != ids:
                raise ValueError('sampling changes with held target')
            by_origin[key] = ids
    if len(by_origin) != 2 * 3 * 19:
        raise ValueError('allocation/origin/source grid')
    origin_rows = [{'allocation': a, 'origin': o, 'unique_labels_all_19_source_pool':
                    sum(len(by_origin[(a, o, r)]) for r in roster)}
                   for a in ('spread', 'recent') for o in ORIGINS]
    union_rows = [{'allocation': a, 'region': r, 'original_region_budget': budgets[r],
                   'cross_origin_unique_labels': len(set().union(*(by_origin[(a, o, r)] for o in ORIGINS)))}
                  for a in ('spread', 'recent') for r in roster]
    if any(row['unique_labels_all_19_source_pool'] != 17262 for row in origin_rows):
        raise ValueError('origin pool budget')
    totals = {a: sum(row['cross_origin_unique_labels'] for row in union_rows if row['allocation'] == a)
              for a in ('spread', 'recent')}
    if totals != {'spread': 27317, 'recent': 34532}:
        raise ValueError('cross-origin union changed')
    if any(sha(ROOT / key) != digest for key, digest in hashes.items()):
        raise ValueError('input changed')
    OUT.mkdir()
    pd.DataFrame(origin_rows).to_csv(OUT / 'by_origin.csv', index=False)
    pd.DataFrame(union_rows).to_csv(OUT / 'cross_origin.csv', index=False)
    summary = {'assertion_failures': 0, 'source_JSON_files_checked': 114,
               'per_origin_19_source_pool': 17262, 'per_LOO_fit_range': [16320, 16388],
               'cross_origin_unique_19_source_pool': totals,
               'equal_per_origin_fine_budget': True, 'equal_cross_origin_unique_budget': False,
               'models_fitted': False, 'new_raw_label_access': False,
               'scope': 'Original frozen V192 contract is per origin/fit. Cross-origin aggregate is not a fixed-global-unique-label experiment.',
               'input_hashes': hashes, 'output_hashes': {p.name: sha(p) for p in OUT.iterdir()}}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(totals, flush=True)


if __name__ == '__main__':
    main()
