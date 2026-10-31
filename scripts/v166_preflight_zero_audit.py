"""Check missing-shape counterexample and source-carrier rescaling identities."""
from pathlib import Path
import json

import numpy as np

from coarse_oracle_zero_audit import CoarseOracleDiagnostic, carrier_laws
from frozen_recomposition import FrozenRecomposition
from hourly_fuel_data import HourlyFuelData
from v151_hourly_law import EF, sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/v166-zero-preflight-r2'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    # Two valid completions agree on every source observation and target margin.
    source_margin = np.array([1., 0.])
    target_margin = np.array([.5, .5])
    emissions = np.array([1000., 400.])
    flat, variable = np.ones((2, 2)), np.array([[1., 0.], [1., 2.]])
    assert np.array_equal(flat * source_margin, variable * source_margin)
    first, second = flat * target_margin, variable * target_margin
    assert np.array_equal(first.mean(axis=0), second.mean(axis=0))
    cif_first = first @ emissions / first.sum(axis=1)
    cif_second = second @ emissions / second.sum(axis=1)
    assert np.array_equal(cif_first, [700., 700.]) and np.array_equal(cif_second, [1000., 600.])
    data = HourlyFuelData(ROOT)
    frozen = FrozenRecomposition(data, ROOT / 'results/v159-fuel-activity-source-pilot-r1', 0, 'asof_30d')
    q = int(frozen.queries.profile_id.iloc[0])
    _, S, donors, _, _, _, weights = CoarseOracleDiagnostic(frozen).inputs(q)
    old = frozen.laws(q)['conditional_activity_recomposition']
    rebuilt = carrier_laws(frozen, donors, weights, S, .01)
    identity_error = max(max(abs(a.mean - b.mean), abs(a.quantile(.05) - b.quantile(.05)),
                             abs(a.quantile(.95) - b.quantile(.95))) for a, b in zip(old, rebuilt))
    if identity_error != 0:
        raise ValueError('reference carrier not exact')
    direct_error, cases = 0., 0
    for epsilon in [.001, .01, .1]:
        B = (1 - epsilon) * S + epsilon * frozen.prior
        for d in donors[:2]:
            source = frozen.activity.sources[data.index.region.iloc[d]]
            local = data.local(int(d), 'asof_30d')
            rows = np.flatnonzero(source['profiles'] == d)
            state = data.states[source['allowed'][rows]] % 24
            current_B = (1 - epsilon) * data.shares[local] + epsilon * frozen.prior
            old_B = .99 * data.shares[local] + .01 * frozen.prior
            for h in [0, 6, 12, 18]:
                ids = rows[state == h]
                if not len(ids):
                    ids = rows
                raw = data.raw[source['allowed'][ids]]
                mean_total = data.M[local].sum() / data.index.hours.iloc[d]
                direct = raw / mean_total / current_B * B
                indirect = frozen.activity.donor(int(d))[h] * (B * old_B / current_B)
                direct_cif = direct @ EF / direct.sum(axis=1)
                indirect_cif = indirect @ EF / indirect.sum(axis=1)
                direct_error = max(direct_error, float(np.max(abs(direct_cif - indirect_cif))))
                cases += 1
    if direct_error > 1e-9:
        raise ValueError('carrier source/target rescaling identity failed')
    OUT.mkdir(parents=True)
    result = {'assertion_failures': 0, 'counterexample_flat_cif': cif_first.tolist(),
              'counterexample_variable_cif': cif_second.tolist(), 'reference_law_max_error': identity_error,
              'direct_source_generation_cases': cases, 'direct_cif_max_error': direct_error,
              'script_sha256': sha(Path(__file__)), 'helper_sha256': sha(ROOT / 'scripts/coarse_oracle_zero_audit.py'),
              'scope': 'Synthetic construction and proper-source numerical identities; no held-out hourly scoring.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
