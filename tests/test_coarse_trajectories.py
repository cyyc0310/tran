import numpy as np
import pandas as pd

from transcif.research.coarse_trajectories import MODES, crps, fit, predict, thermal_fraction, permuted_indices
from transcif.research.multicountry import GROUPS


def example():
    clock = pd.Series(pd.date_range('2022-04-01', periods=72, freq='h'))
    f = pd.DataFrame({'clock': clock, 'region': 'SOURCE', 'country': 'S'})
    f[list(GROUPS)] = np.tile([.5, .1, .1, .2, .1, 0.], (72, 1))
    fuel = np.tile([20., 30., 0., 0., 10., 10., 20., 10., 0.], (72, 1))
    fuel[:24, 0] *= 2; fuel[24:48, 6] *= 2
    return f, fuel


def test_source_month_normalization_reconstructs_own_coarse_shares():
    f, fuel = example(); bank = fit(f, fuel)
    means = np.array(bank['audit']['normalization'][0]['mean_power'])
    own = means / means.sum()
    expected = fuel[:, :4].sum(axis=1) / fuel.sum(axis=1)
    np.testing.assert_allclose(thermal_fraction(own, bank['q']).ravel(), expected)
    assert bank['audit']['retained_hours'] == len(f)


def test_query_batch_order_and_unused_target_labels_do_not_change_members():
    f, fuel = example(); bank = fit(f, fuel)
    q = f.copy(); q.clock += pd.DateOffset(years=1)
    q['target_cif'] = np.arange(len(q))
    a, _ = predict(bank, q, 32)
    q.target_cif = -999.
    order = np.arange(len(q))[::-1]
    b, _ = predict(bank, q.iloc[order], 32)
    assert set(a) == set(MODES)
    for mode in MODES:
        np.testing.assert_array_equal(a[mode], b[mode][order])
        assert np.isfinite(a[mode]).all()
    assert not np.array_equal(a['joint'], a['independent'])


def test_crps_matches_pairwise_definition():
    ensemble = np.array([[0., 2., 4.], [1., 1., 1.]])
    y = np.array([3., 0.])
    expected = np.abs(ensemble - y[:, None]).mean(axis=1) - .5 * np.abs(
        ensemble[:, :, None] - ensemble[:, None, :]).mean(axis=(1, 2))
    np.testing.assert_allclose(crps(ensemble, y), expected)


def test_copula_control_preserves_finite_donor_marginals_exactly():
    selected = np.array([0, 0, 0, 2, 4, 4, 7, 8, 9, 9])
    indices = permuted_indices(selected, np.random.default_rng(7))
    for column in indices.T:
        np.testing.assert_array_equal(np.sort(column), np.sort(selected))
    assert any(not np.array_equal(column, selected) for column in indices.T)


def test_lag_response_reconstructs_source_and_keeps_old_modes_exact():
    f, fuel = example()
    bank = fit(f, fuel, include_response=True)
    own = f.iloc[0][list(GROUPS)].to_numpy(float)
    actual = fuel[:, :4].sum(axis=1) / fuel.sum(axis=1)
    recovered = thermal_fraction(own, bank['q'] * bank['transition'][:, None, :])
    np.testing.assert_allclose(recovered.ravel(), actual)
    q = f.copy(); q.clock += pd.DateOffset(years=1)
    base, _ = predict(fit(f, fuel), q, 32)
    expanded, _ = predict(bank, q, 32)
    for mode in MODES:
        np.testing.assert_array_equal(base[mode], expanded[mode])
    for mode in ['response_coupled', 'response_shuffled', 'transition_only']:
        assert np.isfinite(expanded[mode]).all()
