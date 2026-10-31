"""Proper-source-only geometry of the response embedding; no target scores."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from hourly_fuel_data import HourlyFuelData, repair_response
from v151_hourly_law import EF, sha
from v154_hourly_strong_baselines import balanced

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/v156-anchored-hourly-sampled-r1'
OUT = ROOT / 'results/v158-response-dictionary-diagnostics-r2'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    paths = [Path(__file__), ROOT / 'scripts/hourly_fuel_data.py',
             ROOT / 'scripts/v158_joint_fuel_response.py', ROOT / 'scripts/v151_hourly_law.py',
             ROOT / 'scripts/v154_hourly_strong_baselines.py',
             ROOT / 'results/v144-canonical-profile-cache-r1/profiles.npz',
             ROOT / 'results/v144-canonical-profile-cache-r1/profile_index.csv',
             ROOT / 'results/v153-calendar-times-r1/timestamps.npz',
             BASE / 'summary.json', BASE / 'source_sampling.csv', BASE / 'region_roles.csv']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    archived = json.loads((BASE / 'summary.json').read_text())
    for name, digest in archived['input_hashes'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('upstream input changed: ' + name)
    for name, digest in archived['output_hashes'].items():
        if sha(BASE / name) != digest:
            raise ValueError('upstream output changed: ' + name)
    data = HourlyFuelData(ROOT)
    roles = pd.read_csv(BASE / 'region_roles.csv')
    sampling = pd.read_csv(BASE / 'source_sampling.csv')
    rows, spectra = [], []
    with threadpool_limits(limits=1):
        for (target, seed, budget), draw in sampling.groupby(['target_region', 'seed', 'budget']):
            fit = sorted(roles.loc[roles.target_region.eq(target) & roles.seed.eq(seed)
                                   & roles.role.eq('fit'), 'region'])
            draw = draw.set_index('region')
            sources = [data.source(r, budget, json.loads(draw.loc[r, 'sampled_raw_hour_indices'])) for r in fit]
            d = np.concatenate([s[2] for s in sources])
            y = np.concatenate([s[1] for s in sources])
            w = balanced(np.concatenate([np.full(len(s[1]), r) for r, s in zip(fit, sources)]))
            ids = np.unique(np.concatenate([s[3] for s in sources]))
            ordered = ids[np.lexsort((ids, data.shares[ids, 0]))]
            chosen = ordered[np.rint(np.linspace(0, len(ordered) - 1, 5)).astype(int)]
            responses = []
            for q in chosen:
                S = data.shares[q]
                response, fallbacks = repair_response(S, d, budget == 'same_month')
                linear = S @ EF + y
                candidate = S + d
                nonnegative = repair_response(S, d, False)[0]
                rows.append({'target_region': target, 'seed': int(seed), 'budget': budget,
                             'prototype_profile_id': int(q), 'training_rows': len(y),
                             'weighted_negative_fuel_fraction': float(np.average((candidate < 0).any(axis=1), weights=w)),
                             'weighted_changed_response_fraction': float(np.average(np.abs(response - linear) > 1e-8, weights=w)),
                             'weighted_repair_response_rms': float(np.sqrt(np.average((response - linear)**2, weights=w))),
                             'weighted_support_increment_rms': float(np.sqrt(np.average((response - nonnegative)**2, weights=w))),
                             'fallbacks': fallbacks})
                responses.append(response)
            response = np.column_stack(responses)
            mu = np.average(response, axis=0, weights=w)
            centered = response - mu
            sd = np.sqrt(np.average(centered**2, axis=0, weights=w))
            active = sd > 1e-8
            if not active.any():
                raise ValueError('no active response dimension')
            psi = centered[:, active] / sd[active]
            covariance = (psi * w[:, None]).T @ psi / w.sum()
            eigen = np.linalg.eigvalsh(covariance)
            ym = y - np.average(y, weights=w)
            if np.sum(w * ym**2) <= 1e-12:
                raise ValueError('degenerate scalar response variance')
            coefficient = np.sum(w[:, None] * ym[:, None] * psi, axis=0) / np.sum(w * ym**2)
            residual = psi - ym[:, None] * coefficient
            unexplained = np.average(np.sum(residual**2, axis=1), weights=w) / len(eigen)
            # Weighted within/between-node variance identity used in the novelty note.
            mask = np.arange(len(y)) % 2 == 0
            left = np.average(psi[mask], axis=0, weights=w[mask])
            right = np.average(psi[~mask], axis=0, weights=w[~mask])
            within = (np.sum(w[mask, None] * (psi[mask] - left)**2)
                      + np.sum(w[~mask, None] * (psi[~mask] - right)**2)) / w.sum()
            gain = w[mask].sum() * w[~mask].sum() / w.sum()**2 * np.sum((left - right)**2)
            error = abs(np.trace(covariance) - within - gain)
            if error > 1e-10:
                raise ValueError('weighted embedding split identity')
            spectra.append({'target_region': target, 'seed': int(seed), 'budget': budget,
                            'prototype_profile_ids': json.dumps(chosen.tolist()),
                            'active_dimensions': int(active.sum()),
                            'largest_eigen_variance_fraction': float(eigen[-1] / eigen.sum()),
                            'embedding_variance_unexplained_by_scalar_fraction': float(unexplained),
                            'split_identity_error': float(error)})
    if len(rows) != 60 or len(spectra) != 12:
        raise ValueError('proper source diagnostics denominator')
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('diagnostic input changed during computation')
    OUT.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(OUT / 'prototype_geometry.csv', index=False)
    pd.DataFrame(spectra).to_csv(OUT / 'dictionary_spectra.csv', index=False)
    summary = {'assertion_failures': 0, 'proper_source_contexts': 12, 'prototype_contexts': 60,
               'script_sha256': sha(Path(__file__)),
               'input_hashes': hashes,
               'output_hashes': {p.name: sha(p) for p in sorted(OUT.iterdir())},
               'scope': 'Proper-source embedding geometry only; no target/calibration labels or additional accuracy trial.'}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(pd.DataFrame(spectra).to_string(index=False))


if __name__ == '__main__':
    main()
