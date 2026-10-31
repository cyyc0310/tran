"""Exact small golden sets, strict endpoints and independently optimized witnesses."""
from fractions import Fraction as F
from itertools import combinations, product

import numpy as np
import pytest
from scipy.optimize import linprog
from scripts import risk_stats_v22 as m
from scripts import run_risk_stats_v22 as r


def subset_golden(g, h, tau, bound=F(1000)):
    answer = set()
    for k in range(len(g) + 1):
        for indices in combinations(range(len(g)), k):
            event_mass = sum((g[i] for i in indices), F(0))
            if (k == 0 and h <= tau * sum(g)) or (k > 0 and tau * event_mass < h
                    <= tau * sum(g) + (bound - tau) * event_mass):
                answer.add(k)
    return tuple(sorted(answer))


def min_g_lp_golden(n, total, minimum, h, tau, bound=10):
    """Independent variables (G_i, G_i*C_i), enumerate min-location and event set."""
    allowed = set()
    for k in range(n + 1):
        bits = [i < k for i in range(n)]  # Permutation symmetry; enumerate the min location.
        for min_index in range(n):
            rows, rhs = [], []
            for i, event in enumerate(bits):
                row = np.zeros(2 * n + 1)
                row[n + i], row[i] = 1, -(bound if event else tau)
                rows.append(row);rhs.append(0)
                if event:
                    row = np.zeros(2 * n + 1)
                    row[n + i], row[i], row[-1] = -1, tau, 1
                    rows.append(row);rhs.append(0)
            eq = np.zeros((3, 2 * n + 1))
            eq[0, :n], eq[1, n:2 * n], eq[2, min_index] = 1, 1, 1
            objective = np.zeros(2 * n + 1);objective[-1] = -1
            result = linprog(objective, A_ub=np.array(rows), b_ub=rhs,
                A_eq=eq, b_eq=[total, h, minimum], bounds=[(minimum, None)] * n
                + [(0, None)] * n + [(0, bound * total)], method='highs')
            if result.success and (k == 0 or result.x[-1] > 1e-8):
                allowed.add(k)
    return tuple(sorted(allowed))


def test_sorted_full_g_matches_independent_subset_enumeration():
    for g in ([F(1)], [F(1), F(9)], [F(1), F(2), F(9)], [F(1)] * 6):
        for c in product(map(F, (0, 300, 600, 1000)), repeat=len(g)):
            h = sum(a * b for a, b in zip(g, c))
            for tau in map(F, (100, 300, 600, 900)):
                counts = m.full_g_counts(g, h, tau)
                assert counts == subset_golden(g, h, tau)
                assert counts == tuple(range(min(counts), max(counts) + 1))
                assert sum(x > tau for x in c) in counts


def test_gmin_sharp_formula_matches_independent_small_linear_programs():
    for n, total, minimum in ((1, 2, 2), (2, 5, 1), (3, 3, 1), (3, 8, 1)):
        for tau in (3, 6):
            for h in (0, tau * total - 1, tau * total, tau * total + 1, 10 * total):
                assert m.minimum_g_counts(n, F(total), F(minimum), F(h), F(tau), F(10)) == (
                    min_g_lp_golden(n, total, minimum, h, tau))


def test_every_gmin_admitted_count_has_exact_extremal_construction():
    for n in range(1, 7):
        for total in map(F, (n, 2 * n + 1)):
            minimum = total if n == 1 else F(1)
            for tau in map(F, (3, 6)):
                for h in (F(0), tau * total - F(1, 10), tau * total,
                          tau * total + F(1, 10), 10 * total):
                    for k in m.minimum_g_counts(n, total, minimum, h, tau, F(10)):
                        g, c = m.minimum_g_witness(n, total, minimum, h, tau, k, F(10))
                        assert len(g) == n and min(g) == minimum and sum(g) == total
                        assert all(x > 0 for x in g) and all(0 <= x <= 10 for x in c)
                        assert sum(a * b for a, b in zip(g, c)) == h
                        assert sum(x > tau for x in c) == k


def test_strict_equalities_and_n1_n2():
    tiny = F(1, 10**40)
    assert m.full_g_counts([F(1)] * 6, F(3000), F(600)) == (0, 1, 2, 3, 4)
    assert m.full_g_counts([F(1)] * 6, F(3000) + tiny, F(600)) == (0, 1, 2, 3, 4, 5)
    assert m.minimum_g_counts(2, F(3), F(1), F(600), F(600)) == (0,)
    assert m.minimum_g_counts(2, F(3), F(1), F(600) + tiny, F(600)) == (0, 1)
    assert m.minimum_g_counts(2, F(3), F(1), F(2600), F(600)) == (1, 2)
    assert m.minimum_g_counts(2, F(3), F(1), F(2600) + tiny, F(600)) == (2,)
    for h, answer in ((0, (0,)), (600, (0,)), (601, (1,)), (1000, (1,))):
        assert m.minimum_g_counts(1, F(1), F(1), F(h), F(600)) == answer
        assert m.full_g_counts([F(1)], F(h), F(600)) == answer


def test_k0_certification_is_identical_to_full_g_and_threshold_times_exact_min():
    for g in ([F(1), F(2), F(9)], [F(1)] * 6, [F(1, 1000), F(2), F(5)]):
        total, minimum = sum(g), min(g)
        for tau in map(F, (100, 300, 600, 900)):
            for h in (F(0), tau * minimum, tau * minimum + F(1, 10**30), tau * total, 1000 * total):
                full = m.full_g_counts(g, h, tau)
                reduced = m.minimum_g_counts(len(g), total, minimum, h, tau)
                assert (full == (0,)) == (reduced == (0,)) == (h <= tau * minimum)
                assert set(full) <= set(reduced) <= set(m.coarse_counts(len(g), total, h, tau))


def test_gmin_and_g2_are_not_declared_nested_information_classes():
    # Same scalar count does not make the two different information maps nested.
    seen_gmin_stricter = seen_g2_stricter = False
    for g in ([F(1)] * 3, [F(1), F(2), F(9)], [F(1), F(4), F(4)]):
        total = sum(g)
        for c in product(map(F, (0, 300, 600, 1000)), repeat=3):
            h = sum(a * b for a, b in zip(g, c))
            for tau in map(F, (100, 300, 600, 900)):
                gmin = set(m.minimum_g_counts(3, total, min(g), h, tau))
                g2 = set(m.moment_counts(3, total, sum(x * x for x in g), h, tau))
                seen_gmin_stricter |= bool(g2 - gmin)
                seen_g2_stricter |= bool(gmin - g2)
    assert seen_gmin_stricter and seen_g2_stricter


@pytest.mark.parametrize('args', [(0, F(1), F(1), F(0), F(600)),
    (2, F(2), F(0), F(0), F(600)), (2, F(2), F(2), F(0), F(600)),
    (1, F(2), F(1), F(0), F(600)), (2, F(2), F(1), F(2001), F(600)),
    (2, F(2), F(1), F(0), F(0)), (2, F(2), F(1), F(0), F(1000)),
    (2, 2., F(1), F(0), F(600)), (True, F(1), F(1), F(0), F(600))])
def test_gmin_invalid_observations_fail(args):
    with pytest.raises(ValueError):
        m.minimum_g_counts(*args)


def test_fixed_g_domain_and_impossible_construction_fail():
    for values in ([], [F(0), F(1)], [1.0], [True]):
        with pytest.raises(ValueError):
            m.full_g_counts(values, F(0), F(600))
    with pytest.raises(ValueError):
        m.minimum_g_witness(2, F(3), F(1), F(0), F(600), 1)


def synthetic_fuel():
    return [[F(i % 3 + 1) / 2, F(0), F(0), F(0), F(i % 3 + 1) / 2,
             F(0), F(0), F(0)] for i in range(24)]


def test_one_synthetic_day_has_complete_registered_grid_and_k0_identity():
    daily, blocks = r.day_rows('BPAT', '2024-07-01T00:00:00+00:00', synthetic_fuel())
    assert len(daily) == 16 and len(blocks) == 64
    assert {(row['threshold'], row['budget']) for row in daily} == set(product(r.THRESHOLDS, r.BUDGETS))
    assert all(row['actual_count'] in map(int, row['counts'].split('|')) for row in daily + blocks)
    for tau in r.THRESHOLDS:
        rows = {row['budget']: row for row in daily if row['threshold'] == tau}
        assert rows['T8_Gmin']['no_exceedance_certified'] == rows['T8_G24']['no_exceedance_certified']


@pytest.mark.parametrize('digest', [None, '', 'x' * 64, 'A' * 64, '0' * 63])
def test_parent_digest_requires_explicit_lowercase64hex_before_any_reader(monkeypatch, digest):
    monkeypatch.setattr(r, 'read', lambda _: pytest.fail('Invalid digest reached metadata reader'))
    with pytest.raises(ValueError, match='Explicit64'):
        r.checked_parent(digest)


def test_unsealed_parent_is_rejected_before_source_or_metadata_access(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'MANIFEST', tmp_path / 'missing.json')
    monkeypatch.setattr(r, 'read', lambda _: pytest.fail('Unsealed parent reached metadata reader'))
    monkeypatch.setattr(r.prior, 'source_days', lambda _: pytest.fail('Unsealed parent reached source'))
    with pytest.raises(ValueError, match='Completed sealed Tail parent'):
        r.checked_parent('0' * 64)


def test_partial_attempt_preserved_before_parent_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'OUT', tmp_path)
    (tmp_path / 'block_counts.csv').write_text('partial')
    monkeypatch.setattr(r, 'checked_parent', lambda _: pytest.fail('Partial attempt reached parent'))
    with pytest.raises(FileExistsError, match='Preserve complete or partial'):
        r.prepare('0' * 64)


def test_inherited_curves_match_exactly_and_changes_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'PARENT', tmp_path)
    daily, blocks = r.day_rows('BPAT', '2024-07-01T00:00:00+00:00', synthetic_fuel())
    import pandas as pd
    day_table, block_table = pd.DataFrame(daily), pd.DataFrame(blocks)
    day_table[day_table.budget.isin(r.prior.BUDGETS)].to_csv(tmp_path / 'day_counts.csv', index=False)
    block_table[block_table.budget.isin(r.prior.BUDGETS)].to_csv(tmp_path / 'block_counts.csv', index=False)
    r.match_parent_curves(day_table, block_table)
    day_table.loc[0, 'counts'] = 'CORRUPTED_COUNTS'
    with pytest.raises(ValueError, match='Inherited exact parent count curves changed'):
        r.match_parent_curves(day_table, block_table)


def test_exact_inherited_mask_grid_required(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'PARENT', tmp_path)
    (tmp_path / 'day_counts.csv').write_text('region,day,threshold,budget,counts\nBPAT,2024-07-01,100,T8,0\n')
    with pytest.raises(ValueError, match='Exact inherited1000day'):
        r.inherited_mask()


def test_source_reader_discards_2025_numerical_poison(tmp_path):
    import csv
    path = tmp_path / 'BPAT_hourly.csv'
    fields = ['region', 'time', 'valid', *('generation_' + f for f in r.prior.FUELS)]
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields);writer.writeheader()
        for i, values in enumerate(synthetic_fuel()):
            writer.writerow(dict(region='BPAT', time=f'2024-07-01T{i:02d}:00:00+00:00', valid=True,
                **dict(zip(fields[3:], map(str, values)))))
        writer.writerow(dict(region='BPAT', time='2025-01-01T00:00:00+00:00', valid=True,
            **{key: '2025_MUST_NOT_BE_PARSED' for key in fields[3:]}))
    days = r.prior.source_days(path)
    assert len(days) == 1 and days[0][1] == synthetic_fuel()
