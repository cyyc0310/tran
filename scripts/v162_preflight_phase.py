"""Independent direct assignment/permutation checks for phase diagnostics."""
from itertools import permutations
from pathlib import Path
import json

import numpy as np
import pandas as pd

from calendar_cif_law import CalendarLaw
from phase_loss_tables import choose_phase, phase_scores
from v151_hourly_law import Law, sha
from v162_fixed_phase_diagnostic import METHODS, select_cases

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/v162-phase-preflight-r1'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    dependencies = [Path(__file__), ROOT / 'scripts/phase_loss_tables.py',
                    ROOT / 'scripts/v162_fixed_phase_diagnostic.py', ROOT / 'scripts/calendar_cif_law.py',
                    ROOT / 'scripts/v151_hourly_law.py', ROOT / 'scripts/v139_physical_reserve_calibration.py']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in dependencies}
    laws = [Law([15 * h, min(1000, 30 * h + 100)], [.3, .7]) for h in range(24)]
    actual, clock = np.array([0., 150., 500., 1000.]), np.array([0, 1, 1, 23])
    max_score_error, max_shuffle_error = 0., 0.
    for radius in (0., 17.5, 150.):
        scores = phase_scores(laws, actual, clock, radius)
        for phase in range(24):
            law = CalendarLaw(laws, (clock + phase) % 24)
            lo = np.maximum(0, law.quantile(.05) - radius)
            hi = np.minimum(1000, law.quantile(.95) + radius)
            direct = {'crps': law.crps(actual).mean(), 'mae': np.abs(actual - law.mean).mean(),
                      'coverage': ((actual >= lo) & (actual <= hi)).mean(), 'width': (hi - lo).mean(),
                      'winkler': np.mean(hi - lo + 20 * np.maximum(lo - actual, 0) + 20 * np.maximum(actual - hi, 0))}
            max_score_error = max(max_score_error, max(abs(value - scores[name][phase]) for name, value in direct.items()))
            shuffled = np.mean([law.crps(np.asarray(order)).mean() for order in permutations(actual)])
            max_shuffle_error = max(max_shuffle_error, abs(shuffled - scores['shuffle_crps'][phase]))
    if max_score_error > 1e-10 or max_shuffle_error > 1e-10:
        raise ValueError('direct phase/permutation identity')
    balanced = phase_scores(laws, np.linspace(0, 1000, 24), np.arange(24), 17.5)
    if np.ptp(balanced['width']) > 1e-10 or np.ptp(balanced['shuffle_crps']) > 1e-10:
        raise ValueError('uniform clock permutation invariant')
    if choose_phase(np.array([0, 1, 23]), np.array([0, 1e-11, 0])) != 0:
        raise ValueError('constant-law phase-zero tie preference')
    if choose_phase(np.array([1, 23]), np.array([0, 0])) != 23:
        raise ValueError('signed-phase tie rule')
    rows = []
    for region, months, best in [('A', [1, 7], 3), ('B', [7], 5)]:
        for month in months:
            for budget in ('same_month', 'asof_30d'):
                for method in METHODS:
                    for calibration in ('pooled', 'max_region'):
                        for phase in range(24):
                            distance = min((phase - best) % 24, (best - phase) % 24)
                            rows.append({'region': region, 'month': f'2024-{month:02d}-01', 'budget': budget,
                                         'method': method, 'calibration': calibration, 'phase': phase,
                                         'hours': 10 if month == 1 else 20, 'crps': float(distance**2),
                                         'mae': float(distance + 1), 'coverage': .9, 'width': 100.,
                                         'winkler': 100. + distance, 'calendar_width': 100., 'shuffle_crps': 30.})
    _, policies, selected, _, macro, _ = select_cases(pd.DataFrame(rows))
    fallback = policies.loc[policies.region.eq('B') & policies.case.eq('H1_phase_H2')]
    if not fallback.phase.eq(0).all() or not fallback.no_h1_fallback.eq(True).all():
        raise ValueError('missing H1 fallback')
    a = selected.loc[selected.case.eq('H1_phase_H2')]
    b = selected.loc[selected.case.eq('H2_unshifted')]
    keys = ['region', 'budget', 'method', 'calibration', 'month']
    if set(a[keys].itertuples(index=False, name=None)) != set(b[keys].itertuples(index=False, name=None)):
        raise ValueError('H2 cohort mismatch')
    if not policies.loc[policies.region.eq('A')].phase.eq(3).all():
        raise ValueError('known toy optimum not recovered')
    if any(sha(ROOT / name) != value for name, value in hashes.items()):
        raise ValueError('preflight inputs changed')
    OUT.mkdir(parents=True)
    result = {'assertion_failures': 0, 'direct_phase_score_cases': 72,
              'direct_score_max_error': max_score_error, 'exact_permutation_max_error': max_shuffle_error,
              'h2_cohort_and_no_h1_fallback_checked': True, 'input_hashes': hashes,
              'scope': 'Toy numerical/selection identities only, not predictive improvement.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
