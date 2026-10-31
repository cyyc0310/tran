"""Algebraic tests for discrete PIT intervals and support-preserving CDF maps."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cif_pit_recalibration import GRID, check_maps, pit_bounds, pit_targets, recalibrate
from v151_hourly_law import sha
from v154_hourly_strong_baselines import CompressedLaw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/v174-pit-preflight-r1'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    law = CompressedLaw([0., 400., 1000.], [.2, .5, .3])
    lower, upper, snapped, outside = pit_bounds(law, np.asarray([-1., 0., 200., 400., 1001.]))
    np.testing.assert_allclose(lower, [0, 0, .2, .2, 1], atol=1e-14)
    np.testing.assert_allclose(upper, [0, .2, .2, .7, 1], atol=1e-14)
    np.testing.assert_array_equal(outside, [True, False, False, False, True])
    if snapped.any():
        raise ValueError('exact isolated atoms should not count as changed')
    target = pit_targets(lower, upper)
    independent = np.asarray([[min(1., max(0., (u - L) / (R - L))) if R > L else float(u >= L)
                               for u in GRID[1:-1]] for L, R in zip(lower, upper)])
    np.testing.assert_array_equal(target, independent)
    nearly = CompressedLaw([np.nextafter(400., 0), 400., np.nextafter(400., 1000)], [.2, .5, .3])
    L, R, change, absent = pit_bounds(nearly, np.asarray([400.]))
    np.testing.assert_array_equal(L, [0.])
    np.testing.assert_array_equal(R, [1.])
    if not change.all() or absent.any():
        raise ValueError('numerical atom grouping failed')
    np.testing.assert_allclose(pit_targets(L, R)[0], GRID[1:-1], atol=0, rtol=0)
    for alpha in (.5, 1.):
        if recalibrate(law, GRID.copy(), alpha) is not law:
            raise ValueError('identity map is not exact original law')
        for mapping in (GRID**2, np.sqrt(GRID), np.r_[0, np.ones(len(GRID) - 1)]):
            updated = recalibrate(law, mapping, alpha)
            expected = (1 - .95 * alpha) * law.w + .95 * alpha * np.diff(np.interp(np.r_[0, .2, .7, 1], GRID, mapping))
            np.testing.assert_allclose(updated.w, expected, atol=1e-14)
            np.testing.assert_array_equal(updated.x, law.x)
    invalid = GRID.copy()
    invalid[8] = .99
    try:
        check_maps(invalid)
    except ValueError:
        pass
    else:
        raise ValueError('nonmonotone map accepted')
    OUT.mkdir(parents=True)
    result = {'assertion_failures': 0, 'identity_exact': True, 'numerical_atom_grouping': True,
              'independent_PIT_target_formula': True, 'held_CIF_scored': False,
              'input_hashes': {str(p.relative_to(ROOT)): sha(p) for p in
                               [Path(__file__), ROOT / 'scripts/cif_pit_recalibration.py',
                                ROOT / 'docs/V174_CIF_RECALIBRATION_PROTOCOL.md']},
              'output_hashes': {}}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
