"""Direct moment/affine/generation-mixture identities for label-oracle diagnostics."""
from pathlib import Path
import json

import numpy as np

from frozen_recomposition import FrozenRecomposition
from level_spread_diagnostics import affine_laws, moments, score_laws, spread_ratio
from v151_hourly_law import EF, Law, sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/v163-moment-preflight-r1'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    dependencies = [Path(__file__), ROOT / 'scripts/level_spread_diagnostics.py', ROOT / 'scripts/frozen_recomposition.py',
                    ROOT / 'scripts/v163_level_spread_diagnostic.py', ROOT / 'scripts/fuel_activity_forest.py',
                    ROOT / 'scripts/calendar_cif_law.py', ROOT / 'scripts/v151_hourly_law.py',
                    ROOT / 'scripts/v154_hourly_strong_baselines.py', ROOT / 'scripts/v139_physical_reserve_calibration.py']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in dependencies}
    clock = np.array([0, 1, 1, 4, 23])
    laws = [Law([150 + h, 250 + 2 * h], [.3, .7]) for h in range(24)]
    mean, variance = moments(laws, clock)
    values = np.concatenate([laws[h].x for h in clock])
    weights = np.concatenate([laws[h].w / len(clock) for h in clock])
    if abs(mean - weights @ values) > 1e-10 or abs(variance - weights @ (values - mean)**2) > 1e-10:
        raise ValueError('mixture moments identity')
    moved, clipping, delta, scale = affine_laws(laws, clock, 13., 1.25, (0, 1000))
    changed_mean, changed_variance = moments(moved, clock)
    if clipping != 0 or abs(changed_mean - mean - 13) > 1e-10 or abs(changed_variance - 1.25**2 * variance) > 1e-8:
        raise ValueError('unclipped affine moments')
    clipped, mass, _, _ = affine_laws(laws, clock, 1000., 1., (0, 1000))
    if mass != 1 or any(not np.array_equal(law.x, [1000.]) for law in clipped):
        raise ValueError('clipping trace')
    gas = [Law([400 - 1e-13, 400, 400 + 1e-13], [.2, .6, .2]) for _ in range(24)]
    unchanged, _, used_delta, used_scale = affine_laws(gas, clock, 1e-13, 1 + 1e-15, (400, 400))
    if used_delta != 0 or used_scale != 1 or any(a is not b for a, b in zip(unchanged, gas)):
        raise ValueError('numerical identity changed atomic support')
    if spread_ratio(100, 0) != (1., True) or spread_ratio(0, 25) != (0., False) or spread_ratio(0, 0) != (1., False):
        raise ValueError('degenerate spread cases')
    point = [Law([400.], [1.]) for _ in range(24)]
    scores = score_laws(point, np.array([400., 400 + 1e-12]), np.array([0, 1]), 0.)
    if scores['coverage'] != .5 or scores['tolerant_coverage'] != 1:
        raise ValueError('numerical coverage diagnostic')
    a = np.zeros((2, 8))
    b = np.zeros((3, 8))
    a[:, :2] = [[1, 0], [0, 2]]
    b[:, :2] = [[0, 0], [2, 1], [1, 3]]
    base = np.array([.6, .4, 0, 0, 0, 0, 0, 0])
    class Bank:
        def donor(self, profile):
            return [a if profile == 0 else b] * 24
    generation, emissions = FrozenRecomposition._generation_moments(Bank(), [0, 1], [.3, .7], base)
    activity = np.concatenate([a, b])
    probability = np.r_[np.full(2, .3 / 2), np.full(3, .7 / 3)]
    fuel = activity * base
    g = fuel.sum(axis=1)
    positive = g > 0
    p = probability[positive] / probability[positive].sum()
    e = fuel[positive] @ EF
    c = e / g[positive]
    eg, ec, ee = p @ g[positive], p @ c, p @ e
    covariance = p @ ((g[positive] - eg) * (c - ec))
    if not np.allclose(generation, eg, atol=1e-12) or not np.allclose(emissions, ee, atol=1e-10):
        raise ValueError('positive-generation mixture normalization')
    covariance_error = abs(ec - ee / eg + covariance / eg)
    if covariance_error > 1e-10:
        raise ValueError('generator covariance identity')
    # The same CIF marginal can have different energy anchors: do not infer one from the other.
    c = np.array([0., 1000.])
    anchor_a = np.mean(c)
    changed_g = np.array([2., 1.])
    anchor_b = float(changed_g @ c / changed_g.sum())
    if anchor_a == anchor_b or np.mean(c) != 500:
        raise ValueError('latent-scale nonidentification counterexample')
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('preflight dependencies changed')
    OUT.mkdir(parents=True)
    result = {'assertion_failures': 0, 'oracle_affine_moments_checked': True,
              'identity_support_preserved': True, 'degenerate_spread_handling_checked': True,
              'generator_covariance_identity_error': covariance_error,
              'same_cif_marginal_distinct_energy_anchors': [anchor_a, anchor_b],
              'input_hashes': hashes, 'scope': 'Toy algebra only, no predictive accuracy or novelty evidence.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
