"""Strict-boundary and independently optimized finite tail-count checks."""
from fractions import Fraction as F
from itertools import product

import numpy as np
import pytest
from scipy.optimize import linprog

from scripts import tail_count_v20 as m


def lp_counts(g, h, tau, bound=1000):
    result = set()
    n = len(g)
    for bits in product((0, 1), repeat=n):
        if not any(bits):
            answer = linprog(np.zeros(n), A_eq=[np.asarray(g, float)], b_eq=[float(h)],
                             bounds=[(0, float(tau))]*n, method='highs')
            if answer.success:
                result.add(0)
            continue
        objective = np.r_[np.zeros(n), -1.]
        rows, rhs, bounds = [], [], []
        for i, selected in enumerate(bits):
            if selected:
                row = np.zeros(n+1); row[i] = -1; row[-1] = 1
                rows.append(row); rhs.append(-float(tau))
                bounds.append((0, bound))
            else:
                bounds.append((0, float(tau)))
        answer = linprog(objective, A_ub=np.stack(rows), b_ub=rhs,
                         A_eq=[np.r_[np.asarray(g, float), 0]], b_eq=[float(h)],
                         bounds=bounds+[(0, bound-float(tau))], method='highs')
        if answer.success and answer.x[-1] > 1e-7:
            result.add(sum(bits))
    return tuple(sorted(result))


def test_strict_boundary_does_not_allow_five_exceedances_at_mean500():
    g = [F(1)]*6
    assert m.full_g_counts(g, F(3000), F(600)) == (0, 1, 2, 3, 4)
    assert m.moment_counts(6, F(6), F(6), F(3000), F(600)) == (0, 1, 2, 3, 4)
    assert m.coarse_counts(6, F(6), F(3000), F(600)) == (0, 1, 2, 3, 4, 5)
    assert 5 in m.full_g_counts(g, F(3000)+F(1, 10**30), F(600))


def test_equal_to_threshold_is_not_an_event_and_endpoints_are_exact():
    assert m.full_g_counts([F(1)]*6, F(600), F(600)) == (0,)
    assert m.full_g_counts([F(1)]*6, F(600)+F(1, 10**30), F(600)) == (0, 1)
    for h, expected in ((0, (0,)), (6000, (6,))):
        assert m.coarse_counts(6, F(6), F(h), F(600)) == expected
        assert m.moment_counts(6, F(6), F(6), F(h), F(600)) == expected
        assert m.full_g_counts([F(1)]*6, F(h), F(600)) == expected


def test_full_g_enumeration_matches_independent_strict_margin_lp():
    rng = np.random.default_rng(20026)
    for _ in range(20):
        g = [F(int(x)) for x in rng.integers(1, 10, 4)]
        c = [F(int(x)) for x in rng.choice([0, 200, 400, 600, 800, 1000], 4)]
        h = sum((a*b for a, b in zip(g, c)), F(0))
        tau = F(int(rng.choice([100, 300, 600, 900])))
        assert m.full_g_counts(g, h, tau) == lp_counts(g, h, tau)


def test_moment_outer_contains_truth_and_full_g_set_on_exhaustive_small_grid():
    for g in ([F(1)]*3, [F(1), F(2), F(9)]):
        for c in product(map(F, (0, 300, 600, 1000)), repeat=3):
            h = sum((a*b for a, b in zip(g, c)), F(0))
            for tau in map(F, (100, 300, 600, 900)):
                coarse = m.coarse_counts(3, sum(g), h, tau)
                outer = m.moment_counts(3, sum(g), sum(x*x for x in g), h, tau)
                sharp = m.full_g_counts(g, h, tau)
                assert set(sharp) <= set(outer) <= set(coarse)
                assert sum(x > tau for x in c) in sharp


def test_combination_retains_holes_and_description_requires_true_count():
    assert m.combine_blocks([(0, 2), (0, 2)]) == (0, 2, 4)
    assert m.describe((0,), 0)['no_exceedance_certified']
    with pytest.raises(ValueError):
        m.describe((0,), 1)


def test_invalid_or_incompatible_observations_fail():
    with pytest.raises(ValueError):
        m.moment_counts(6, F(6), F(5), F(100), F(600))
    with pytest.raises(ValueError):
        m.moment_counts(6, F(6), F(36), F(100), F(600))
    with pytest.raises(ValueError):
        m.full_g_counts([F(0), F(1)], F(100), F(600))
    with pytest.raises(ValueError):
        m.full_g_counts([F(1)]*24, F(12000), F(600))
