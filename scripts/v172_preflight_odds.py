"""Synthetic stability and proper-source law construction before V172 scoring."""
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from frozen_recomposition import FrozenRecomposition
from hourly_fuel_data import HourlyFuelData
from monthly_history_features import MonthlyInformation
from v151_hourly_law import sha
from zero_odds_cif import ZeroOddsCIF, odds_update

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/v172-zero-odds-preflight-r1'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    prior = np.array([.2, .8])
    zero = np.zeros((2, 8), bool)
    zero[0, 0] = True
    target = np.full(8, .5)
    target[0] = .7
    posterior = odds_update(prior, zero, target, 1.)
    assert abs(posterior[0] - (.95 * .7 + .05 * .2)) < 1e-12
    assert odds_update(prior, zero, target, 0.) is prior
    constant = np.zeros((2, 8), bool)
    assert odds_update(prior, constant, np.ones(8), 1.) is prior
    rare = np.array([1., 1e-310])
    for value in (0., 1.):
        stable = odds_update(rare, zero, np.full(8, value), 1.)
        assert np.isfinite(stable).all() and (stable > 0).all() and abs(stable.sum() - 1) < 1e-12
    with threadpool_limits(limits=2):
        data, monthly = HourlyFuelData(ROOT), MonthlyInformation(ROOT)
        frozen = FrozenRecomposition(data, ROOT / 'results/v159-fuel-activity-source-pilot-r1', 0, 'asof_30d')
        predictor = ZeroOddsCIF(frozen, monthly, ROOT / 'results/v171-two-scale-zero-r1/fold_0')
        q = int(frozen.queries.profile_id.iloc[0])
        laws, _, details = predictor.build(q)
        assert len(laws) == 18 and all(len(cells) == 48 for cells in laws.values()) and len(details) == 16
        assert all(np.isfinite(law.x).all() and np.isfinite(law.w).all() and (law.w > 0).all()
                   for cells in laws.values() for law in cells)
    OUT.mkdir(parents=True)
    result = {'assertion_failures': 0, 'one_fuel_smoothed_target': float(posterior[0]),
              'alpha_zero_identity': True, 'constant_support_identity': True, 'rare_class_stable': True,
              'legal_query_law_methods': len(laws), 'script_sha256': sha(Path(__file__)),
              'helper_sha256': sha(ROOT / 'scripts/zero_odds_cif.py'),
              'scope': 'Synthetic identities and legal law construction from proper source atoms; no held-out CIF scoring.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
