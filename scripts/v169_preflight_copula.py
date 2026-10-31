"""Synthetic admissibility and fixed-source identities before any new scoring."""
from pathlib import Path
import json

import numpy as np

from feasible_fuel_copula import FeasibleCopulaCache, feasible_indices
from frozen_recomposition import FrozenRecomposition
from hourly_fuel_data import HourlyFuelData
from v151_hourly_law import EF, sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/v169-feasible-copula-preflight-r4'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    activity = np.zeros((2, 8))
    activity[:, :2] = [[.5, 1.5], [1.5, .5]]
    indices, meta = feasible_indices(activity, 'toy', 41)
    shuffled = activity[indices, np.arange(8)]
    repeated, _ = feasible_indices(activity, 'toy', 41)
    assert np.array_equal(indices, repeated)
    assert np.array_equal(np.sort(shuffled, axis=0), np.sort(np.tile(activity, (meta['copies'], 1)), axis=0))
    B = np.array([.25, .75, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8])
    fuel, new_fuel = activity * B, shuffled * B
    mean_error = float(np.max(abs(fuel.mean(axis=0) - new_fuel.mean(axis=0))))
    anchor = float(fuel.mean(axis=0) @ EF / fuel.mean(axis=0).sum())
    new_anchor = float(new_fuel.mean(axis=0) @ EF / new_fuel.mean(axis=0).sum())
    old_cif, new_cif = fuel @ EF / fuel.sum(axis=1), new_fuel @ EF / new_fuel.sum(axis=1)
    assert mean_error < 1e-12 and abs(anchor - new_anchor) < 1e-10
    assert abs(old_cif.mean() - new_cif.mean()) > 1
    complementary = np.zeros((2, 8))
    complementary[:, :2] = [[1., 0.], [0., 1.]]
    forced, forced_meta = feasible_indices(complementary, 'complementary', 41)
    assert np.all((complementary[forced, np.arange(8)] > 0).sum(axis=1) == 1)
    assert forced_meta['changed_swaps'] == 0
    cache = FeasibleCopulaCache()
    three = np.zeros((3, 8))
    three[:, :2] = [[1., 1.], [1., 2.], [2., 1.]]
    cache.transform(three, 1, 0, 41)
    altered = three.copy()
    altered[:, 0] = [1., 2., 2.]
    cached, cached_meta = cache.transform(altered, 1, 0, 41)
    direct, direct_meta = feasible_indices(altered, '1/0', 41)
    assert (np.array_equal(cached, altered[direct, np.arange(8)]) and cached_meta == direct_meta
            and len(cache.indices) == 2)
    invalid = altered.copy()
    invalid[:, 7] = -1
    try:
        cache.transform(invalid, 1, 0, 41)
    except ValueError:
        pass
    else:
        raise AssertionError('cache hit bypassed input validation')
    data = HourlyFuelData(ROOT)
    frozen = FrozenRecomposition(data, ROOT / 'results/v159-fuel-activity-source-pilot-r1', 0, 'asof_30d')
    cases, max_relative_mean_error = 0, 0.
    for region in frozen.fit[:3]:
        profile = int(frozen.bank[region][0])
        for hour in [0, 6, 12, 18]:
            original = frozen.activity.donor(profile)[hour]
            for seed in [41, 83]:
                ids, _ = feasible_indices(original, f'{profile}/{hour}', seed)
                new = original[ids, np.arange(8)]
                error = np.max(abs(new.mean(axis=0) - original.mean(axis=0)) / np.maximum(1., abs(original.mean(axis=0))))
                max_relative_mean_error = max(max_relative_mean_error, float(error))
                cases += 1
    assert max_relative_mean_error < 1e-12
    OUT.mkdir(parents=True)
    result = {'assertion_failures': 0, 'toy_original_energy_anchor': anchor, 'toy_shuffled_energy_anchor': new_anchor,
              'toy_original_cif_time_mean': float(old_cif.mean()), 'toy_shuffled_cif_time_mean': float(new_cif.mean()),
              'complementary_zero_changed_swaps': forced_meta['changed_swaps'], 'proper_source_cases': cases,
              'different_equality_pattern_cache_entries': len(cache.indices),
              'invalid_cache_hit_rejected': True,
              'max_relative_fuel_mean_error': max_relative_mean_error,
              'script_sha256': sha(Path(__file__)), 'helper_sha256': sha(ROOT / 'scripts/feasible_fuel_copula.py'),
              'scope': 'Synthetic and proper-source admissibility checks; no held-out hourly performance scoring.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
