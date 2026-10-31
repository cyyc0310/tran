"""Explicit posthoc saved-prediction anchor/κ combination; no fitting or raw labels."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from v151_hourly_law import sha

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/v192-reallocated-kappa-r1'
AUDIT = ROOT / 'results/v192-reallocated-kappa-audit-r2'
OUT = ROOT / 'results/v192-kappa-anchor-posthoc-r1'
METHODS = ('decomp_ridge36', 'decomp_et36', 'dynamic_pooled', 'dynamic_random')


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    paths = [Path(__file__), ROOT / 'scripts/v151_hourly_law.py', SOURCE / 'scores.csv',
             SOURCE / 'summary.json', AUDIT / 'summary.json']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    audited = json.loads((AUDIT / 'summary.json').read_text())
    if audited['assertion_failures'] != 0 or audited['score_records_rebuilt'] != 912:
        raise ValueError('complete audit required')
    scores = pd.read_csv(SOURCE / 'scores.csv')
    key = ['target', 'origin', 'allocation']
    if len(scores) != 912 or scores.duplicated(key + ['method']).any():
        raise ValueError('score keys')
    anchor = scores.loc[scores.method.eq('anchor'), key + ['predicted_mu']].rename(columns={'predicted_mu': 'anchor'})
    frame = scores.loc[scores.method.isin(METHODS)].merge(anchor, on=key, validate='many_to_one')
    if len(frame) != 456:
        raise ValueError('posthoc case denominator')
    # Parent-requested exact algebraic combination; clipping is separate sensitivity.
    frame['combined_mu_unclipped'] = frame.anchor - frame.predicted_kappa
    frame['combined_mu_clipped'] = np.clip(frame.combined_mu_unclipped, 0, 1000)
    frame['combined_ae'] = abs(frame.combined_mu_unclipped - frame.actual_mu)
    frame['clipped_ae'] = abs(frame.combined_mu_clipped - frame.actual_mu)
    frame['anchor_ae'] = abs(frame.anchor - frame.actual_mu)
    frame['a_error'] = frame.anchor - frame.actual_a
    frame['kappa_error'] = frame.predicted_kappa - frame.actual_kappa
    if not np.allclose(frame.combined_mu_unclipped - frame.actual_mu,
                       frame.a_error - frame.kappa_error, rtol=0, atol=1e-10):
        raise ValueError('non-additive absolute-error decomposition')
    metrics = ['combined_ae', 'clipped_ae', 'anchor_ae']
    if not np.isfinite(frame[metrics + ['combined_mu_unclipped']]).all().all():
        raise ValueError('nonfinite diagnostic')
    regional = frame.groupby(['allocation', 'method', 'target'])[metrics].mean()
    boot = np.random.default_rng(192).integers(0, 19, size=(2000, 19))
    rows = []
    for (allocation, method), group in regional.groupby(level=[0, 1]):
        group = group.droplevel([0, 1]).sort_index()
        if len(group) != 19:
            raise ValueError('paired region denominator')
        delta = (group.clipped_ae - group.anchor_ae).to_numpy()
        lo, hi = np.quantile(delta[boot].mean(axis=1), [.025, .975])
        rows.append({'allocation': allocation, 'method': method, 'regions': 19, 'queries': 57,
                     'raw_algebraic_mae': float(group.combined_ae.mean()),
                     'clipped_combined_mae': float(group.clipped_ae.mean()),
                     'anchor_mae': float(group.anchor_ae.mean()), 'clipped_mae_delta': float(delta.mean()),
                     'regions_improved': int((delta < 0).sum()),
                     'descriptive_ci_low': float(lo), 'descriptive_ci_high': float(hi)})
    if any(sha(ROOT / k) != v for k, v in hashes.items()):
        raise ValueError('input changed')
    OUT.mkdir()
    frame.to_csv(OUT / 'cases.csv', index=False)
    pd.DataFrame(rows).to_csv(OUT / 'macro.csv', index=False)
    summary = {'assertion_failures': 0, 'posthoc': True, 'new_models_fitted': False,
               'new_raw_label_access': False, 'recomposed_cases': 456, 'combinations': 8,
               'unclipped_combinations_outside_0_1000': int(((frame.combined_mu_unclipped < 0)
                                                          | (frame.combined_mu_unclipped > 1000)).sum()),
               'source_probability_extension_triggers_changed': False,
               'scope': 'Saved point predictions only; κ correction error and future a change error are signed components, not additive absolute risks.',
               'input_hashes': hashes, 'output_hashes': {p.name: sha(p) for p in OUT.iterdir()}}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
