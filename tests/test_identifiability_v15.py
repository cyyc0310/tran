"""Structural bounds checked against independent linear programming."""
import numpy as np
import pytest
from scipy.optimize import linprog

from scripts.identifiability_v15 import (
    functional_bounds, group_only_bounds, marginal_emissions_bounds, oracle_bounds,
    read_evaluation, transport_witness,
)


def lp_value(rows, columns, ef, coefficients, maximize=False):
    h, f = len(rows), len(columns)
    constraints = np.zeros((h + f, h * f))
    for i in range(h):
        constraints[i, i*f:(i+1)*f] = 1
    for j in range(f):
        constraints[h+j, j::f] = 1
    objective = np.outer(coefficients, ef).ravel()
    result = linprog((-1 if maximize else 1)*objective, A_eq=constraints,
                     b_eq=np.r_[rows, columns], bounds=(0, None), method='highs')
    assert result.success, result.message
    return objective @ result.x


@pytest.mark.parametrize('seed', range(8))
def test_functional_and_marginal_bounds_match_lp(seed):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 3, (5, 4))
    a[:, 2] = 0
    a[2] = 0
    rows, cols = a.sum(1), a.sum(0)
    ef = np.array([1000., 400., 0., 400.])
    weights = rng.normal(size=5)
    actual = functional_bounds(rows, cols, ef, weights)
    expected = [lp_value(rows, cols, ef, weights, flag) for flag in (False, True)]
    np.testing.assert_allclose(actual, expected, atol=1e-8)
    lo, hi = marginal_emissions_bounds(rows, cols, ef)
    for i in range(5):
        w = np.eye(5)[i]
        np.testing.assert_allclose([lo[i], hi[i]],
                                  [lp_value(rows, cols, ef, w, flag) for flag in (False, True)],
                                  atol=1e-8)


@pytest.mark.parametrize('maximize', [False, True])
def test_witness_preserves_margins_and_attains_bound(maximize):
    a = np.array([[2., 1.], [1., 4.], [0., 0.]])
    weights, ef = np.array([1., -.5, 0.]), np.array([0., 1000.])
    witness = transport_witness(a.sum(1), a.sum(0), ef, weights, maximize)
    np.testing.assert_allclose(witness.sum(0), a.sum(0))
    np.testing.assert_allclose(witness.sum(1), a.sum(1))
    assert (witness >= 0).all()
    assert (witness @ ef) @ weights == pytest.approx(
        functional_bounds(a.sum(1), a.sum(0), ef, weights)[int(maximize)])


def test_same_fuel_and_hourly_totals_do_not_identify_mean_or_tail():
    ef = np.array([1000., 0.])
    a = np.array([[1., 0.], [0., 2.]])
    b = np.array([[0., 1.], [1., 1.]])
    np.testing.assert_array_equal(a.sum(0), b.sum(0))
    np.testing.assert_array_equal(a.sum(1), b.sum(1))
    ca, cb = a @ ef/a.sum(1), b @ ef/b.sum(1)
    assert ca.mean() == 500
    assert cb.mean() == 250
    assert ca.max() == 1000 and cb.max() == 500
    assert (a @ ef).sum()/a.sum() == (b @ ef).sum()/b.sum()


def test_equal_hourly_generation_still_leaves_tails_unidentified():
    a = np.array([[.5, .5], [.5, .5]])
    b = np.eye(2)
    ef = np.array([1000., 0.])
    assert (a @ ef).mean() == (b @ ef).mean() == 500
    assert (a @ ef).max() == 500 and (b @ ef).max() == 1000


def test_nested_blocks_contract_and_preserve_truth():
    rng = np.random.default_rng(12)
    a = rng.uniform(0, 10, (48, 8))
    ef = np.array([1000., 400., 800., 0., 0., 0., 0., 400.])
    previous = None
    for size in [48, 24, 6, 3, 1]:
        lo, hi, ml, mh = oracle_bounds(a, np.arange(48)//size, (tuple(range(8)),), ef)
        truth = (a @ ef)/a.sum(1)
        assert np.all(lo <= truth + 1e-9) and np.all(hi >= truth - 1e-9)
        assert np.all(ml <= truth.reshape(-1, 24).mean(1) + 1e-9)
        assert np.all(mh >= truth.reshape(-1, 24).mean(1) - 1e-9)
        if previous is not None:
            assert np.all(lo >= previous[0]-1e-8) and np.all(hi <= previous[1]+1e-8)
            assert np.all(ml >= previous[2]-1e-8) and np.all(mh <= previous[3]+1e-8)
        previous = lo, hi, ml, mh
    np.testing.assert_allclose(lo, truth, atol=1e-9)
    np.testing.assert_allclose(hi, truth, atol=1e-9)


def test_coal_and_emitting_channel_identifies_proxy_without_oil():
    rng = np.random.default_rng(17)
    a = rng.uniform(0, 2, (24, 8))
    a[:, 2] = 0
    ef = np.array([1000., 400., 800., 0., 0., 0., 0., 400.])
    groups = ((0,), (1, 2, 7), (3, 4, 5, 6))
    lo, hi, ml, mh = oracle_bounds(a, np.zeros(24, int), groups, ef)
    np.testing.assert_allclose(lo, hi, atol=1e-8)
    np.testing.assert_allclose(ml, mh, atol=1e-8)
    stress = ef.copy()
    stress[7] = 600
    sl, sh, _, _ = oracle_bounds(a, np.zeros(24, int), groups, stress)
    assert (sh-sl).mean() > 1


def test_removing_future_daily_fuel_totals_reveals_hidden_oil_ambiguity():
    rng = np.random.default_rng(27)
    a = rng.uniform(0, 2, (24, 8))
    a[:, 2] = 0
    ef = np.array([1000., 400., 800., 0., 0., 0., 0., 400.])
    groups = ((0,), (1, 2, 7), (3, 4, 5, 6))
    lo, hi, ml, mh = group_only_bounds(a, groups, ef)
    total = a.sum(1)
    emitting, coal = a[:, [0, 1, 2, 7]].sum(1)/total, a[:, 0]/total
    np.testing.assert_allclose(lo, 400*emitting+600*coal)
    np.testing.assert_allclose(hi, 800*emitting+200*coal)
    assert (hi-lo).mean() > 1
    assert mh[0]-ml[0] == pytest.approx((hi-lo).mean())
    day_lo, day_hi, _, _ = oracle_bounds(a, np.zeros(24), groups, ef)
    assert np.all(lo <= day_lo+1e-8) and np.all(hi >= day_hi-1e-8)


@pytest.mark.parametrize('rows,cols,ef,weights', [
    ([1], [2], [0], [1]), ([-1], [-1], [0], [1]),
    ([np.nan], [1], [0], [1]), ([1], [1], [np.inf], [1]),
    ([1], [1], [0], [1, 2]),
])
def test_bad_transport_inputs_rejected(rows, cols, ef, weights):
    with pytest.raises(ValueError):
        functional_bounds(rows, cols, ef, weights)


def test_bad_group_partition_rejected():
    a = np.ones((24, 2))
    with pytest.raises(ValueError):
        oracle_bounds(a, np.zeros(24), ((0,), (0, 1)), np.array([1., 0.]))


def test_reader_filters_year_before_numeric_parsing_and_rejects_duplicate(tmp_path):
    import pandas as pd
    cols = [f'generation_{f}' for f in
            ('coal', 'gas', 'oil', 'nuclear', 'hydro', 'wind', 'solar', 'other')]
    frame = pd.DataFrame({'time': pd.date_range('2025-01-01', periods=24, freq='h', tz='UTC'),
                          'region': 'BPAT', 'valid': True,
                          **{c: np.ones(24) for c in cols}})
    bad = frame.iloc[[0]].copy()
    bad['time'] = '2026-01-01 00:00:00+00:00'
    bad[cols] = 'DO_NOT_PARSE'
    path = tmp_path/'BPAT_hourly.csv'
    pd.concat([frame, bad]).to_csv(path, index=False)
    days, a = read_evaluation(path, 'BPAT', 2025)
    assert len(days) == 1 and a.shape == (24, 8)
    pd.concat([frame, frame.iloc[[0]]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match='unique'):
        read_evaluation(path, 'BPAT', 2025)


def test_source_choice_uses_main_ef_and_each_region_resolution():
    import pandas as pd
    from scripts.run_identifiability_v15 import SINGLE, DOUBLE, SOURCES, choose
    records = []
    for name, width in zip(SINGLE+DOUBLE, [50, 60, 70, 10, 5]):
        for i, region in enumerate(SOURCES):
            records.append(dict(ef='proxy', budget=name, region=region, hour_width=width,
                resolved_hour_fraction100=.9 if name in SINGLE and i == 0 else 1.,
                daily_mean_width=1.))
    region = pd.DataFrame(records)
    macro = region.groupby(['ef','budget']).hour_width.mean().reset_index()
    stress = macro.copy()
    stress['ef'] = 'algebraic_stress'
    stress['hour_width'] = np.arange(len(stress))
    selected = choose(region, pd.concat([macro, stress]))
    assert selected['single_channel'] == 'day_coal'
    assert selected['two_channels'] == 'day_emitting_coal'
    assert selected['smallest_tested_resolution_budget'] == 'day_emitting_coal'
    region['resolved_hour_fraction100'] = .5
    assert choose(region, macro)['smallest_tested_resolution_budget'] is None


def test_target_rejects_changed_freeze_before_parsing_labels(tmp_path, monkeypatch):
    import json
    from scripts import run_identifiability_v15 as runner
    monkeypatch.setattr(runner, 'OUT', tmp_path)
    (tmp_path/'freeze.json').write_text(json.dumps({'input_hashes': {}}))
    (tmp_path/'source_completed.json').write_text(json.dumps(dict(
        stage='source', version=15, freeze_sha256='changed', prediction_performed=False)))
    with pytest.raises(ValueError, match='Source receipt'):
        runner.run('target')


def test_during_run_freeze_mutation_cannot_be_bound_as_original(tmp_path, monkeypatch):
    import json
    import pandas as pd
    from scripts import run_identifiability_v15 as runner
    monkeypatch.setattr(runner, 'OUT', tmp_path)
    monkeypatch.setattr(runner, 'SOURCES', ('BPAT',))
    frozen = dict(input_hashes={}, source_year=2025, target_year=2026,
                  data={'source': {'BPAT': 'unused'}},
                  budgets=['past_only'], ef={'proxy': runner.EF.tolist()})
    def fake_prepare():
        (tmp_path/'freeze.json').write_text(json.dumps(frozen))
        return frozen
    def fake_read(*args):
        (tmp_path/'freeze.json').write_text(json.dumps({**frozen, 'target_year': 2027}))
        return pd.date_range('2025-01-01', periods=1145, freq='D', tz='UTC'), np.ones((1145*24, 8))
    monkeypatch.setattr(runner, 'prepare', fake_prepare)
    monkeypatch.setattr(runner, 'read_evaluation', fake_read)
    monkeypatch.setattr(runner, 'choose', lambda *args: {})
    with pytest.raises(ValueError, match='Changed frozen file: .*freeze.json'):
        runner.run('source')
    assert not (tmp_path/'source_completed.json').exists()
