import pickle

import numpy as np
import pandas as pd
import pytest

from transcif.research.dynamic_v2 import EF
from transcif.research.eia_proxy import FUELS
from transcif.research.fuel_transport_v7 import (
    BANDWIDTHS, FuelTrajectoryBank, GENERATION_COLUMNS, MODES, recompose_paths,
)


def mix(gas=.5, wind=.5, **other):
    result = np.zeros(8)
    for fuel, share in {"gas": gas, "wind": wind, **other}.items():
        result[FUELS.index(fuel)] = share
    return result


def frame(days=2, *, start="2024-01-01", gas=100., wind=100., joint=True):
    times = pd.date_range(start, periods=24 * days, freq="h", tz="UTC")
    row = pd.DataFrame({"time": times, "valid": True})
    for column in GENERATION_COLUMNS:
        row[column] = 0.
    hour = np.sin(2 * np.pi * np.arange(24) / 24)
    for day in range(days):
        # Day-average variation must survive normalization: for days=2 the
        # two daily factors are .5 and 1.5, with monthly mean exactly one.
        factor = .5 if day % 2 == 0 else 1.5
        pattern = 1 + (.4 if day % 2 == 0 else -.4) * hour
        row.loc[day * 24:(day + 1) * 24 - 1, "generation_gas"] = gas * factor * pattern
        row.loc[day * 24:(day + 1) * 24 - 1, "generation_wind"] = wind * factor * (pattern if joint else 2 - pattern)
    return row


def test_source_month_operator_identity_and_daily_variation_survives():
    bank = FuelTrajectoryBank(["A"]).fit({"A": frame()})
    restored = recompose_paths(bank.donor_mix_, bank.q_)
    np.testing.assert_allclose(restored, bank.raw_cif_, atol=1e-12)
    np.testing.assert_allclose(bank.q_.mean(axis=(0, 1)), np.ones(8), atol=1e-12)
    gas = FUELS.index("gas")
    np.testing.assert_allclose(bank.q_[:, :, gas].mean(axis=1), [.5, 1.5], atol=1e-12)
    generation = bank.q_ * bank.month_mean_generation_[:, None, :]
    energy_ratio = np.sum(generation * EF) / generation.sum()
    np.testing.assert_allclose(energy_ratio, bank.donor_mix_[0] @ EF)
    assert bank.audit_["monthly_shares_used"] is False
    assert "not constrained" in bank.audit_["energy_limit"]


def test_joint_coupling_and_independent_marginals_change_only_fuel_copula():
    bank = FuelTrajectoryBank(["A"]).fit({"A": frame()})
    options = dict(forecast_mix=[mix()], origins=["2025-01-15"], n_scenarios=5000, seed=17023)
    joint = bank.predict(mode="recompose_joint", **options)
    independent = bank.predict(mode="recompose_independent", **options)
    np.testing.assert_allclose(joint["ensemble"], 200., atol=1e-12)
    assert independent["ensemble"].std() > 20
    np.testing.assert_array_equal(joint["donor_weights"], independent["donor_weights"])
    draws = independent["sampled_donor_indices"][0]
    for fuel in range(8):
        assert .47 < np.mean(draws[:, fuel] == 0) < .53
    assert np.any(draws[:, FUELS.index("gas")] != draws[:, FUELS.index("wind")])


def test_direct_analog_uses_exact_same_weights_and_day_draws_as_joint():
    bank = FuelTrajectoryBank(["A"]).fit({"A": frame(joint=False)})
    options = dict(forecast_mix=[mix(gas=.1, wind=.9)], origins=["2025-01-15"], seed=19)
    direct = bank.predict(mode="direct_analog", **options)
    recomposed = bank.predict(mode="recompose_joint", **options)
    np.testing.assert_array_equal(direct["donor_weights"], recomposed["donor_weights"])
    np.testing.assert_array_equal(direct["sampled_donor_indices"], recomposed["sampled_donor_indices"])
    indices = direct["sampled_donor_indices"][0, :, 0]
    np.testing.assert_array_equal(direct["ensemble"][0], bank.raw_cif_[indices])
    assert not np.allclose(direct["point"], recomposed["point"])
    assert np.std(recomposed["point"]) > 1


def test_equal_region_prior_is_not_a_forced_equal_region_posterior():
    bank = FuelTrajectoryBank(["A", "B"]).fit({"A": frame(days=2, gas=900., wind=100.),
                                               "B": frame(days=6, gas=100., wind=900.)})
    posterior = bank.predict([mix(gas=.1, wind=.9)], ["2025-01-15"], bandwidth=.1)
    np.testing.assert_allclose(posterior["region_prior"], [[.5, .5]])
    np.testing.assert_allclose(posterior["donor_prior"][0, :2], .25)
    np.testing.assert_allclose(posterior["donor_prior"][0, 2:], 1 / 12)
    assert posterior["region_weights"][0, 1] > .999
    # Pooled removes only mix matching; identical month support restores the prior.
    pooled = bank.predict([mix(gas=.1, wind=.9)], ["2025-01-15"], mode="pooled_recompose", bandwidth=.1)
    np.testing.assert_allclose(pooled["region_weights"], [[.5, .5]])


def test_exclusion_happens_before_normalization_and_cannot_sample_heldout():
    bank = FuelTrajectoryBank().fit({"A": frame(gas=900., wind=100.), "B": frame(gas=100., wind=900.)})
    result = bank.predict([mix(gas=.1, wind=.9)], ["2025-01-15"], exclude_region="B", bandwidth=1e-6)
    np.testing.assert_array_equal(result["region_weights"], [[1., 0.]])
    assert np.all(bank.regions_[result["sampled_donor_indices"]] == "A")
    with pytest.raises(ValueError, match="no source donors"):
        FuelTrajectoryBank().fit({"A": frame()}).predict([mix()], ["2025-01-15"], exclude_region="A")


def test_missing_fuel_is_flat_prior_with_explicit_unsupported_target_mass():
    bank = FuelTrajectoryBank().fit({"A": frame()})
    solar = FUELS.index("solar")
    assert not bank.fuel_support_[:, solar].any()
    np.testing.assert_array_equal(bank.q_[:, :, solar], 1.)
    result = bank.predict([mix(gas=.3, wind=0., solar=.7)], ["2025-01-15"])
    np.testing.assert_allclose(result["unsupported_target_mass"], .7)
    np.testing.assert_allclose(result["sampled_unsupported_target_mass"], .7)
    assert result["weighted_fuel_support"][0, solar] == 0
    assert "no shape identified" in bank.audit_["missing_fuel_policy"]


@pytest.mark.parametrize("mode", MODES)
def test_output_bounded_label_free_and_point_is_exact_particle_mean(mode):
    bank = FuelTrajectoryBank().fit({"A": frame(joint=False)})
    result = bank.predict([mix(gas=.3, wind=.3, coal=.4)], ["2025-08-15"], mode=mode, n_scenarios=41)
    assert result["ensemble"].shape == (1, 41, 24)
    assert np.isfinite(result["ensemble"]).all()
    assert result["ensemble"].min() >= EF.min()
    assert result["ensemble"].max() <= EF.max()
    np.testing.assert_array_equal(result["point"], result["ensemble"].mean(axis=1))
    assert result["calendar_month_unseen"].item()
    assert not result["source_fit"]["target_hourly_labels_used"]
    if mode == "mean_profile":
        assert result["point_estimator"] == "ratio_of_weighted_mean_q_plugin"
        assert np.all(result["sampled_donor_indices"] == -1)


def test_seed_repetition_pickle_and_batch_order_resolution_invariance():
    bank = FuelTrajectoryBank().fit({"A": frame(joint=False), "B": frame(days=4)})
    origins = pd.DatetimeIndex(["2025-01-15", "2025-09-03"], tz="UTC")
    mixes = np.stack([mix(gas=.2, wind=.8), mix(gas=.7, wind=.3)])
    first = bank.predict(mixes, origins, mode="recompose_independent", seed=23)
    repeated = pickle.loads(pickle.dumps(bank)).predict(mixes[::-1], origins[::-1].as_unit("us"),
                                                       mode="recompose_independent", seed=23)
    single = bank.predict(mixes[:1], origins[:1], mode="recompose_independent", seed=23)
    for key in ("point", "ensemble", "sampled_donor_indices", "donor_weights"):
        np.testing.assert_array_equal(first[key], repeated[key][::-1])
        np.testing.assert_array_equal(first[key][:1], single[key])


def test_h2_values_never_enter_fit_and_2025_or_later_frames_are_rejected():
    base = frame()
    poison = frame(start="2024-07-01")
    for column in GENERATION_COLUMNS:
        poison[column] = "DO_NOT_PARSE_HELDOUT"
    clean = FuelTrajectoryBank().fit({"A": base})
    other = FuelTrajectoryBank().fit({"A": pd.concat([base, poison])})
    np.testing.assert_array_equal(clean.q_, other.q_)
    assert other.days_.max() < pd.Timestamp("2024-07-01", tz="UTC")
    poison["time"] += pd.DateOffset(years=1)
    with pytest.raises(ValueError, match="2025/later"):
        FuelTrajectoryBank().fit({"A": pd.concat([base, poison])})
    with pytest.raises(ValueError, match="cutoff"):
        FuelTrajectoryBank().fit({"A": base}, cutoff="2025-07-01")


def test_incomplete_or_invalid_days_do_not_enter_normalization_and_bad_valid_data_raise():
    rows = frame(days=3)
    rows.loc[0, "valid"] = False
    rows = rows.drop(index=24)
    bank = FuelTrajectoryBank().fit({"A": rows})
    assert len(bank.days_) == 1
    assert bank.days_[0] == pd.Timestamp("2024-01-03", tz="UTC")
    bad = frame()
    bad.loc[0, "generation_gas"] = np.nan
    with pytest.raises(ValueError, match="finite nonnegative"):
        FuelTrajectoryBank().fit({"A": bad})


def test_source_declaration_optional_monthly_boundary_and_zero_total_fallback():
    with pytest.raises(ValueError, match="declared source"):
        FuelTrajectoryBank(["A"]).fit({"A": frame(), "TARGET": frame()})
    monthly = pd.DataFrame({"region": ["TARGET"], "period_start": ["2024-01-01"], "period_end": ["2024-02-01"]})
    with pytest.raises(ValueError, match="source 2024H1"):
        FuelTrajectoryBank(["A"]).fit({"A": frame()}, monthly)
    paths, zero = recompose_paths(mix(gas=1., wind=0.), np.zeros((24, 8)), return_zero_total=True)
    np.testing.assert_array_equal(paths, 400.)
    assert zero.all()
    assert BANDWIDTHS == (.1, .3, 1.)
