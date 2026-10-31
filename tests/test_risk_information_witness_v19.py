"""Pure synthetic tests; never invoke the formal witness artifact writer."""
from __future__ import annotations

import json
import numpy as np
import pytest
from scripts import risk_information_witness_v19 as w


def test_same_full_coarse_observations_different_tail_peak_and_ramp():
    document = w.build_witnesses()
    pair = document['witnesses']['coarse_tail_peak_ramp']
    a, b = pair['left'], pair['right']
    assert a['observations'] == b['observations']
    assert a['observations']['D8'] == [12., 0., 0., 0., 12., 0., 0., 0.]
    assert a['observations']['T8_H_G'] == [[3., 6.]]*4
    assert a['observations']['G24'] == [1.]*24
    assert (a['risks']['exceedance_hours'], b['risks']['exceedance_hours']) == (0, 12)
    assert (a['risks']['peak_cif'], b['risks']['peak_cif']) == (500., 1000.)
    assert (a['risks']['max_adjacent_absolute_ramp'], b['risks']['max_adjacent_absolute_ramp']) == (0., 1000.)
    assert pair['max_abs_map_residual'] == 0
    for side in (a, b):
        generation = np.array(side['generation'])
        np.testing.assert_array_equal(generation.sum(1), 1.)
        assert np.all(generation >= 0)
        np.testing.assert_array_equal(generation[:, [1, 2, 3, 5, 6, 7]], 0.)


def test_permutation_preserves_coarse_and_tail_but_changes_time_localization():
    document = w.build_witnesses()
    reverse = document['witnesses']['coarse_tail_equal_but_time_location_reversed']
    permutation = document['witnesses']['coarse_tail_equal_but_time_location_permuted']
    for pair in (reverse, permutation):
        a, b = pair['left']['risks'], pair['right']['risks']
        assert a['exceedance_hours'] == b['exceedance_hours'] == 12
        assert a['peak_cif'] == b['peak_cif'] == 1000
        assert a['exceedance_hour_indices'] != b['exceedance_hour_indices']
    assert reverse['left']['risks']['first_peak_hour'] == 0
    assert reverse['right']['risks']['first_peak_hour'] == 3
    assert permutation['left']['risks']['nonzero_adjacent_ramp_count'] == 7
    assert permutation['right']['risks']['nonzero_adjacent_ramp_count'] == 23


def test_ramps_include_cross_block_boundaries_but_not_day_wrap_and_peak_ties_are_explicit():
    # Ramp only between two UTC6h blocks, not inside either block.
    a = w.path(np.r_[np.zeros(6), np.ones(18)])
    risk = w.risks(a)
    assert risk['max_adjacent_absolute_ramp'] == 1000
    assert risk['max_ramp_transition_start_hours'] == [5]
    assert risk['cyclic_boundary_absolute_ramp'] == 1000
    assert risk['peak_hour_indices'] == list(range(6, 24))
    # A gradual path has a huge wrap but a much smaller main adjacent ramp.
    risk = w.risks(w.path(np.linspace(0, 1, 24)))
    assert risk['max_adjacent_absolute_ramp'] == pytest.approx(1000/23)
    assert risk['cyclic_boundary_absolute_ramp'] == 1000


def test_four_exact_hourly_labels_still_fail_to_identify_remaining_tail():
    pair = w.build_witnesses()['witnesses']['sparse_four_labels_still_not_tail']
    a, b = pair['left'], pair['right']
    labels = np.array(w.OBSERVED_HOURS)
    np.testing.assert_array_equal(np.array(a['risks']['hourly_cif'])[labels], 500.)
    np.testing.assert_array_equal(np.array(b['risks']['hourly_cif'])[labels], 500.)
    assert a['risks']['exceedance_hours'] == 0
    assert b['risks']['exceedance_hours'] == 8
    assert not set(b['risks']['exceedance_hour_indices']) & set(labels)
    assert pair['observations_exactly_equal']


def test_full_affine_maps_match_direct_D8_T8_G24_and_report_actual_ranks():
    rng = np.random.default_rng(19);coal = rng.uniform(0, 1, 24)
    observations = w.observations(w.path(coal))
    direct = np.r_[observations['D8'], np.array(observations['T8_H_G']).ravel(), observations['G24']]
    base = w.observation_map()
    np.testing.assert_allclose(w.apply_map(coal, base), direct, atol=1e-14, rtol=0)
    labels = w.observation_map(labels=True)
    moments = w.observation_map(labels=True, time_moment=True)
    assert [(x['rank'], x['nullity']) for x in (base, labels, moments)] == [(4, 20), (8, 16), (9, 15)]
    assert [len(x['matrix']) for x in (base, labels, moments)] == [40, 44, 45]
    # D8 and constant G rows contribute no additional coal rank beyond T8 H.
    np.testing.assert_equal(np.linalg.matrix_rank(np.array(base['matrix'])[[8, 10, 12, 14]]), 4)


def test_independent_first_block_time_moment_still_has_feasible_tail_witness():
    document = w.build_witnesses()
    specification = document['maps']['plus_four_labels_and_one_time_moment']
    pair = document['witnesses']['additional_time_moment_still_not_tail']
    a = np.array(pair['left']['generation'])[:, 0]
    b = np.array(pair['right']['generation'])[:, 0]
    assert a[:6]@np.arange(6) == b[:6]@np.arange(6) == 7.5
    np.testing.assert_array_equal(a[list(w.OBSERVED_HOURS)], b[list(w.OBSERVED_HOURS)])
    np.testing.assert_array_equal(w.apply_map(a, specification), w.apply_map(b, specification))
    assert np.all((b >= 0) & (b <= 1))
    assert pair['left']['risks']['exceedance_hours'] == 0
    assert pair['right']['risks']['exceedance_hours'] == 2
    assert specification['rank'] == 9 and specification['nullity'] == 15


def test_parametric_two_hour_direction_preserves_sums_and_local_labels():
    coal = np.full(24, .5)
    changed = w.sum_preserving_perturbation(coal, w.OBSERVED_HOURS, 1, 2, .25)
    np.testing.assert_array_equal(changed.reshape(4, 6).sum(1), coal.reshape(4, 6).sum(1))
    np.testing.assert_array_equal(changed[list(w.OBSERVED_HOURS)], coal[list(w.OBSERVED_HOURS)])
    assert np.any(coal != changed)
    for first, second, epsilon in ((0, 2, .25), (1, 7, .25), (1, 2, 1.), (1, 1, .1)):
        with pytest.raises(ValueError):
            w.sum_preserving_perturbation(coal, w.OBSERVED_HOURS, first, second, epsilon)


def test_certified_support_cap_excludes_strict_tail_but_not_peak_or_ramp():
    bound = w.capacity_exclusion(.6)
    assert bound['hourly_cif_upper_bound'] == 600
    assert bound['strict_exceedance_excluded']
    risk = w.risks(w.path(np.tile([.6, 0.], 12)))
    assert risk['exceedance_hours'] == 0 and risk['peak_cif'] == 600 and risk['max_adjacent_absolute_ramp'] == 600
    assert not bound['real_capacity_certified'] and not bound['sufficient_for_future_prediction']
    assert not w.capacity_exclusion(.601)['strict_exceedance_excluded']


def test_mean_bounds_uniform_generation_endpoints_and_numerically_extreme_cv():
    assert w.mean_outer_bounds(400., 0.) == (400., 400.)
    assert w.mean_outer_bounds(0., 3.) == (0., 0.)
    assert w.mean_outer_bounds(1000., 3.) == (1000., 1000.)
    assert w._quadratic_mean_bounds(0., 1., 1000.) == (0., 500.)
    for cv in (1e-150, 1e-10, .5, 1e150, 1e308):
        low, high = w.mean_outer_bounds(400., cv)
        assert np.isfinite([low, high]).all() and 0 <= low <= 400 <= high <= 1000
    for mu, cv, support in ((-1, 1, 1000), (1001, 1, 1000), (1, -1, 1000), (1, 1, 0),
                            (np.nan, 1, 1000), (1, np.inf, 1000), (1, 1, np.inf)):
        with pytest.raises(ValueError):
            w.mean_outer_bounds(mu, cv, support)


def test_classical_mean_bounds_and_covariance_identity_on1000_positive_G_samples():
    result = w.synthetic_mean_checks()
    assert result['trials'] == 1000 and result['all_outer_bound_checks_passed']
    assert result['largest_generation_cv'] > 4
    assert result['largest_covariance_identity_abs_error'] < 1e-9
    # Equality example for Cauchy and bounded variance: C is binary and G affine in C.
    carbon = np.tile([0., 1000.], 12);generation = np.tile([.1, 1.9], 12)
    energy = float(np.average(carbon, weights=generation));cv = float(generation.std()/generation.mean())
    low, high = w.mean_outer_bounds(energy, cv)
    assert low == pytest.approx(500., abs=1e-10) and high >= 500
    # Equal G fixes the mean but the first exact witness still has different tails.
    pair = w.build_witnesses()['witnesses']['coarse_tail_peak_ramp']
    for side in ('left', 'right'):
        assert np.mean(pair[side]['risks']['hourly_cif']) == 500
    assert pair['left']['risks']['exceedance_hours'] != pair['right']['risks']['exceedance_hours']


def test_builder_is_deterministic_serializable_and_has_no_input_or_output_io(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Pure synthetic builder attempted filesystem input/output')
    monkeypatch.setattr(w.Path, 'open', forbidden)
    document = w.build_witnesses()
    assert document == w.build_witnesses()
    json.dumps(document, allow_nan=False)
    limits = document['limits']
    assert limits['all_inputs_synthetic'] and limits['noiseless_exact_contemporaneous_oracle']
    assert not any(limits[k] for k in ('regional_values_read', 'old_scores_read', 'future_forecast',
                   'new_theorem', 'real_information_value_measured', 'universally_minimal_budget_proved', 'target_or_unseen_opened'))


def test_mean_bound_float_guard_covers_decimal_roots_and_underflow_falls_back():
    from decimal import Decimal, localcontext
    with localcontext() as ctx:
        ctx.prec = 100
        mu, cv, bound = Decimal(1), Decimal(3), Decimal(1000)
        a = 1+cv*cv;b = 2*mu+cv*cv*bound
        root = (b+(b*b-4*a*mu*mu).sqrt())/(2*a)
        low, high = w.mean_outer_bounds(1., 3.)
        assert Decimal.from_float(high) > root
        assert low < 1 < high
    assert w.mean_outer_bounds(1e-300, 1e-300, 1e300) == (0., 1e300)
    # A classical equality case whose unguarded lower root can err several ulp.
    carbon = np.r_[np.full(22, 1000.), np.zeros(2)]
    fraction = 22/24
    generation = 1+.6138*(carbon/1000-fraction)/fraction
    energy = float(np.average(carbon, weights=generation))
    low, high = w.mean_outer_bounds(energy, float(generation.std()/generation.mean()))
    assert low <= carbon.mean() <= high
