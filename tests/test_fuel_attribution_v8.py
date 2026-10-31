import numpy as np
import pandas as pd
import pytest

from scripts.fuel_attribution_v8 import N_SAMPLES, SEED, generate_matched
from transcif.research.dynamic_v2 import EF
from transcif.research.eia_proxy import FUELS
from transcif.research.fuel_transport_v7 import FuelTrajectoryBank, GENERATION_COLUMNS


def mix(**shares):
    result = np.zeros(len(FUELS))
    for fuel, value in shares.items():
        result[FUELS.index(fuel)] = value
    return result


def frame(start, gas, wind, solar=0.):
    times = pd.date_range(start, periods=48, freq="h", tz="UTC")
    result = pd.DataFrame({"time": times, "valid": True})
    for column in GENERATION_COLUMNS:
        result[column] = 0.
    hours = np.tile(np.arange(24), 2)
    day = np.repeat([.7, 1.3], 24)
    result["generation_gas"] = gas * day * (1 + .35 * np.sin(hours * np.pi / 12))
    result["generation_wind"] = wind * day * (1 - .3 * np.sin(hours * np.pi / 12))
    result["generation_solar"] = solar * day * np.maximum(np.sin((hours - 6) * np.pi / 12), 0)
    return result


@pytest.fixture
def bank():
    return FuelTrajectoryBank().fit({
        "A": frame("2024-01-01", 90., 10.),
        "B": frame("2024-02-01", 20., 80.),
    })


@pytest.fixture
def coarse():
    return {
        "experts": np.stack([
            [mix(gas=.8, wind=.2), mix(gas=.3, wind=.7), mix(gas=.1, wind=.9)],
            [mix(gas=.2, wind=.8), mix(gas=.5, wind=.5), mix(gas=.9, wind=.1)],
        ]),
        "weights": {"0.03": np.array([[.1, .2, .7], [.6, .3, .1]])},
    }


@pytest.fixture
def days():
    return pd.DatetimeIndex(["2025-01-15", "2025-09-03"], tz="UTC")


@pytest.mark.parametrize("forecast,bandwidth", [
    ("latest", 1.), ("adaptive_mean", .3), ("adaptive_mixture", .3),
])
def test_legal_paths_match_v7_and_share_routing(bank, coarse, days, forecast, bandwidth):
    config = dict(forecast=forecast, bandwidth=bandwidth, temperature=.03)
    result = generate_matched(bank, coarse, days, config)
    audit = result["audit"]
    if forecast == "latest":
        mixtures = coarse["experts"][:, 0]
    elif forecast == "adaptive_mean":
        mixtures = np.einsum("ne,nef->nf", coarse["weights"]["0.03"], coarse["experts"])
    else:
        mixtures = None
    if mixtures is not None:
        joint = bank.predict(mixtures, days, mode="recompose_joint", bandwidth=bandwidth,
                             n_scenarios=N_SAMPLES, seed=SEED)
        direct = bank.predict(mixtures, days, mode="direct_analog", bandwidth=bandwidth,
                              n_scenarios=N_SAMPLES, seed=SEED)
        np.testing.assert_array_equal(result["recompose_joint"]["ensemble"], joint["ensemble"])
        np.testing.assert_array_equal(result["direct_analog"]["ensemble"], direct["ensemble"])
        np.testing.assert_array_equal(audit["sampled_donor_indices"], joint["sampled_donor_indices"][:, :, 0])
        np.testing.assert_array_equal(audit["donor_weights"][:, 0], joint["donor_weights"])
    else:
        components = [bank.predict(coarse["experts"][:, e], days,
                                   mode="recompose_joint", bandwidth=bandwidth,
                                   n_scenarios=N_SAMPLES, seed=SEED) for e in range(3)]
        expected_experts = np.stack([
            np.random.default_rng(np.random.SeedSequence([
                SEED, int(day.value // (86400 * 10**9)), 997
            ])).choice(3, N_SAMPLES, p=coarse["weights"]["0.03"][i])
            for i, day in enumerate(days)
        ])
        np.testing.assert_array_equal(audit["expert_choices"], expected_experts)
        for i in range(len(days)):
            for scenario in range(N_SAMPLES):
                e = expected_experts[i, scenario]
                d = components[e]["sampled_donor_indices"][i, scenario, 0]
                assert audit["sampled_donor_indices"][i, scenario] == d
                np.testing.assert_array_equal(result["recompose_joint"]["ensemble"][i, scenario],
                                              components[e]["ensemble"][i, scenario])
                np.testing.assert_array_equal(result["direct_analog"]["ensemble"][i, scenario],
                                              bank.raw_cif_[d])
    for name in ("direct_analog", "additive_anchor", "recompose_joint"):
        np.testing.assert_array_equal(result[name]["point"], result[name]["ensemble"].mean(axis=1))
    assert audit["preclip_additive_shape_max_abs_error"] < 1e-12


def test_additive_self_mix_identity_and_preclip_shape():
    bank = FuelTrajectoryBank().fit({"A": frame("2024-01-01", 90., 10.)})
    day = pd.DatetimeIndex(["2025-01-15"], tz="UTC")
    own_mix = bank.donor_mix_[0]
    coarse = {"experts": np.tile(own_mix, (1, 3, 1)), "weights": {"0.03": np.array([[1., 0., 0.]])}}
    result = generate_matched(bank, coarse, day, dict(forecast="latest", bandwidth=1., temperature=.03))
    np.testing.assert_array_equal(result["additive_anchor"]["ensemble"], result["direct_analog"]["ensemble"])
    donor = result["audit"]["sampled_donor_indices"]
    shift = (own_mix - bank.donor_mix_[donor]) @ EF
    np.testing.assert_allclose(shift, 0., atol=1e-12)
    assert result["audit"]["preclip_additive_shape_max_abs_error"] < 1e-12


def test_oracle_only_changes_mapping_and_is_label_free_without_argument(bank, coarse, days):
    config = dict(forecast="adaptive_mixture", bandwidth=.3, temperature=.03)
    legal = generate_matched(bank, coarse, days, config)
    oracle_mix = np.stack([mix(gas=.05, wind=.95), mix(gas=.95, wind=.05)])
    oracle = generate_matched(bank, coarse, days, config, oracle_mix=oracle_mix)
    for key in ("direct_analog", "additive_anchor", "recompose_joint"):
        for field in ("point", "ensemble"):
            np.testing.assert_array_equal(legal[key][field], oracle[key][field])
    for key in ("expert_choices", "sampled_donor_indices", "donor_weights", "draw_sha256"):
        np.testing.assert_array_equal(legal["audit"][key], oracle["audit"][key])
    assert not np.array_equal(oracle["oracle_additive_anchor"]["ensemble"], legal["additive_anchor"]["ensemble"])
    assert not np.array_equal(oracle["oracle_recompose_joint"]["ensemble"], legal["recompose_joint"]["ensemble"])
    np.testing.assert_array_equal(legal["direct_analog"]["ensemble"], oracle["direct_analog"]["ensemble"])


def test_ns_us_batch_order_and_zero_fuel(bank, coarse, days):
    solar = FUELS.index("solar")
    assert not bank.fuel_support_[:, solar].any()
    coarse["experts"][:, :, :] = np.stack([mix(gas=.2, solar=.8)] * 3)
    config = dict(forecast="adaptive_mixture", bandwidth=.3, temperature=.03)
    full = generate_matched(bank, coarse, days, config)
    reversed_coarse = {"experts": coarse["experts"][::-1],
                       "weights": {"0.03": coarse["weights"]["0.03"][::-1]}}
    reversed_ = generate_matched(bank, reversed_coarse, days[::-1].as_unit("us"), config)
    for key in ("direct_analog", "additive_anchor", "recompose_joint"):
        np.testing.assert_array_equal(full[key]["ensemble"], reversed_[key]["ensemble"][::-1])
    for key in ("sampled_donor_indices", "expert_choices"):
        np.testing.assert_array_equal(full["audit"][key], reversed_["audit"][key][::-1])
    assert np.isfinite(full["recompose_joint"]["ensemble"]).all()
    assert np.isfinite(full["additive_anchor"]["ensemble"]).all()
