import numpy as np
import pandas as pd
import pytest

from transcif.research.conditional_transfer_v6 import (
    ConditionalTransport, FEATURE_NAMES, MODES, information_features, information_feature_audit,
)
from transcif.research.eia_proxy import FUELS


def monthly(periods=72):
    starts = pd.date_range("2018-01-01", periods=periods, freq="MS", tz="UTC")
    frame = pd.DataFrame({"region": "R", "period_start": starts,
                          "period_end": starts + pd.offsets.MonthBegin(1),
                          "available_at": starts + pd.offsets.MonthBegin(1) + pd.Timedelta(days=30)})
    for fuel in FUELS:
        frame[f"fuelshare_{fuel}"] = 0.
    frame["fuelshare_coal"] = .4 + .2 * np.sin(2 * np.pi * starts.month.to_numpy() / 12)
    frame["fuelshare_gas"] = 1 - frame.fuelshare_coal
    return frame


def cohort():
    rng = np.random.default_rng(9021)
    dates = pd.DatetimeIndex([f"2024-{month:02d}-{day:02d}" for region in range(2)
                              for month in range(1, 13) for day in range(1, 11)], tz="UTC")
    regions = np.repeat(["A", "B"], 120)
    x = np.abs(rng.normal(size=(len(dates), len(FEATURE_NAMES)))) * 15
    x[:, 0] += 300
    x[:, 3] += 30
    x[:, 4] += 24
    x[:, 5] = rng.normal(size=len(dates)) * 20
    point = np.full((len(dates), 24), 300.)
    shape = np.arange(24.) - 11.5
    level_error = .5 * x[:, 5] + rng.normal(size=len(dates)) * (2 + x[:, 10])
    truth = point + level_error[:, None] + rng.normal(size=(len(dates), 1)) * shape
    return point, truth, x, regions, dates


def test_information_is_release_lawful_resolution_invariant_and_batch_consistent():
    frame = monthly()
    origins = pd.DatetimeIndex(["2023-08-03", "2023-08-29", "2023-09-10"], tz="UTC")
    actual, names = information_features(origins, "R", frame)
    assert names == FEATURE_NAMES
    future = frame.iloc[[-1]].copy()
    future.available_at = pd.Timestamp("2028-01-01", tz="UTC")
    future.fuelshare_coal = np.nan
    changed = pd.concat([frame, future])
    for column in ("period_start", "period_end", "available_at"):
        changed[column] = pd.to_datetime(changed[column], utc=True).dt.as_unit("us")
    other, _ = information_features(origins.as_unit("us"), "R", changed)
    np.testing.assert_array_equal(actual, other)
    singles = [information_features([origin], "R", frame)[0][0] for origin in origins]
    np.testing.assert_array_equal(actual, singles)
    delayed, _ = information_features(origins, "R", frame, 90)
    assert np.all(delayed[:, names.index("history_months")] < actual[:, names.index("history_months")])


def test_seasonal_predictability_is_distinct_from_raw_variation():
    x, names = information_features(["2024-03-15"], "R", monthly())
    values = dict(zip(names, x[0]))
    assert values["monthly_sd"] > 10
    assert values["seasonal_hindcast_rms"] < 1e-8
    assert values["harmonic_hindcast_sd"] < values["monthly_sd"] * .1
    assert values["latest_hindcast_sd"] > values["harmonic_hindcast_sd"] * 5


def test_hindcast_reconstructs_publication_time_instead_of_looking_back_from_now():
    frame = monthly(36)
    # This old row becomes available much later. It can be a current label but
    # cannot be a covariate in pseudo-forecasts that precede its publication.
    frame.loc[0, "available_at"] = pd.Timestamp("2021-01-15", tz="UTC")
    first, names = information_features(["2021-02-15"], "R", frame)
    altered = frame.copy()
    altered.loc[0, ["fuelshare_coal", "fuelshare_gas"]] = [.99, .01]
    second, _ = information_features(["2021-02-15"], "R", altered)
    # Month 2018-01 lies outside the last 24 hindcast labels; although released
    # now, its value cannot enter any of their pre-publication pseudo-prefixes.
    columns = [names.index(f"{m}_hindcast_{s}")
               for m in ("latest", "seasonal", "harmonic") for s in ("bias", "rms", "sd")]
    np.testing.assert_array_equal(first[:, columns], second[:, columns])


def test_empty_history_and_unreleased_bad_values():
    frame = monthly()
    frame.loc[:, "fuelshare_coal"] = np.nan
    x, names = information_features(["2017-01-01"], "R", frame)
    assert np.isfinite(x).all()
    assert x[0, names.index("history_months")] == 0
    assert x[0, names.index("age_days")] == 365
    with pytest.raises(ValueError, match="simplex"):
        information_features(["2024-01-01"], "R", frame)


def test_cold_start_fallbacks_do_not_count_as_valid_seasonal_or_harmonic():
    x, names = information_features(["2018-10-15"], "R", monthly())
    assert x[0, names.index("hindcast_months")] > 0
    assert x[0, names.index("seasonal_hindcast_valid_months")] == 0
    assert x[0, names.index("harmonic_hindcast_valid_months")] == 0
    for method in ("seasonal", "harmonic"):
        for stat in ("bias", "rms", "sd"):
            assert x[0, names.index(f"{method}_hindcast_{stat}")] == 0
    audit = information_feature_audit(x)
    assert audit["support"]["harmonic_hindcast_valid_months"]["zero_rows"] == 1
    assert audit["ranges"]["history_months"]["min"] > 0


@pytest.mark.parametrize("mode", MODES)
def test_modes_have_independent_bank_and_frozen_proper_fit(mode):
    point, truth, x, regions, dates = cohort()
    model = ConditionalTransport(["A", "B"], mode).fit(point, truth, x, regions, days=dates)
    poisoned = truth.copy()
    poisoned[dates.month > 6] += 500
    other = ConditionalTransport(["A", "B"], mode).fit(point, poisoned, x, regions, days=dates)
    pred = model.predict(point[:3], x[:3], seed=11)
    again = model.predict(point[:3], x[:3], seed=11)
    shifted = other.predict(point[:3], x[:3], seed=11)
    for key in ("corrected_point", "bias", "level_scale", "shape_scale"):
        np.testing.assert_array_equal(pred[key], shifted[key])
    np.testing.assert_array_equal(pred["ensemble"], again["ensemble"])
    assert not np.allclose(pred["ensemble"], shifted["ensemble"])
    assert pred["ensemble"].shape == (3, 128, 24)
    assert np.all(pred["lower"] <= pred["upper"])
    assert np.all(pred["level_scale"] >= 1)
    assert np.all(pred["shape_scale"] >= 1)
    assert model.audit_["proper_fit_days"] == model.audit_["bank_days"] == 120
    assert model.audit_["fit_year"] == 2024
    assert model.audit_["feature_audit"]["proper"]["rows"] == 120
    assert all(day.month > 6 for day in model.days_)
    assert all(count == 60 for count in model.audit_["proper_fit_region_counts"].values())
    # Full-bank intervals must not depend on finite ensemble draws.
    changed_seed = model.predict(point[:3], x[:3], seed=99, n_samples=17)
    np.testing.assert_array_equal(pred["lower"], changed_seed["lower"])


def test_fit_rejects_target_labels_dates_and_missing_source_split():
    point, truth, x, regions, dates = cohort()
    with pytest.raises(ValueError, match="source regions"):
        ConditionalTransport(["A"]).fit(point, truth, x, regions, days=dates)
    with pytest.raises(ValueError, match="2024"):
        ConditionalTransport(["A", "B"]).fit(point, truth, x, regions, days=dates + pd.DateOffset(years=1))
    with pytest.raises(ValueError, match="both Jan-Jun"):
        keep = dates.month < 7
        ConditionalTransport(["A", "B"]).fit(point[keep], truth[keep], x[keep], regions[keep], days=dates[keep])


def test_nohindcast_ablation_and_global_really_ignore_hindcasts():
    point, truth, x, regions, dates = cohort()
    changed = x.copy()
    changed[:, 5:] = 1e3
    for mode in ("global", "structured_nohindcast"):
        model = ConditionalTransport(["A", "B"], mode).fit(point, truth, x, regions, days=dates)
        first, second = model.predict(point[:3], x[:3]), model.predict(point[:3], changed[:3])
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])


def test_joint_pairs_and_equal_region_sampling_and_zero_multiplier():
    pattern = np.arange(24.) - 11.5
    levels = np.r_[1., 2., np.full(20, 3.), 30.]
    regions = np.array(["A", "B"] + ["A"] * 20 + ["B"])
    days = pd.DatetimeIndex(["2024-01-01", "2024-01-01"] + ["2024-07-01"] * 21, tz="UTC")
    point = np.zeros((len(levels), 24))
    truth = levels[:, None] * (1 + pattern)
    features = np.ones((len(levels), len(FEATURE_NAMES)))
    model = ConditionalTransport(["A", "B"], "global").fit(point, truth, features, regions, days=days)
    pred = model.predict(point[:1], features[:1], n_samples=10000)
    ensemble = pred["ensemble"][0]
    # The entire source daily path, including its level/shape association, survives.
    np.testing.assert_allclose(ensemble, ensemble.mean(axis=1)[:, None] * (1 + pattern), atol=1e-12)
    assert .47 < np.mean(ensemble.mean(axis=1) > 10) < .53
    zero = model.predict(point[:1], features[:1], multiplier=0)
    np.testing.assert_array_equal(zero["ensemble"], 0)
    np.testing.assert_array_equal(zero["lower"], zero["upper"])


def test_component_scales_change_level_without_distorting_transported_shape():
    point, truth, x, regions, dates = cohort()
    level = np.full(len(point), 3.)
    shape = np.full((len(point), 24), 4.)
    model = ConditionalTransport(["A", "B"]).fit(point, truth, x, regions, days=dates,
                                                    level_scales=level, shape_scales=shape)
    first = model.predict(point[:3], x[:3], level_scales=level[:3], shape_scales=shape[:3])
    second = model.predict(point[:3], x[:3], level_scales=level[:3] * 20, shape_scales=shape[:3])
    # NNLS may legitimately assign zero weight to the supplied base variance;
    # nondecrease is the model contract, while shape isolation is unconditional.
    assert np.all(second["level_scale"] >= first["level_scale"] - 1e-12)
    np.testing.assert_array_equal(second["shape_scale"], first["shape_scale"])
    zero_mean = lambda z: z - z.mean(axis=-1, keepdims=True)
    np.testing.assert_allclose(zero_mean(first["ensemble"]), zero_mean(second["ensemble"]), atol=1e-11)


@pytest.mark.parametrize("mode", ["loglinear", "extratrees"])
def test_generic_scale_models_receive_component_scales_and_freeze_transform_on_proper(mode):
    point, _, x, regions, dates = cohort()
    x[:] = 1.
    rng = np.random.default_rng(9022)
    scales = rng.uniform(3., 60., len(point))
    signs = rng.choice([-1., 1.], len(point))
    pattern = (np.arange(24.) - 11.5) / 5
    truth = point + signs[:, None] * scales[:, None] + scales[:, None] * pattern
    model = ConditionalTransport(["A", "B"], mode).fit(
        point, truth, x, regions, days=dates, level_scales=scales, shape_scales=scales)
    low = model.predict(point[:3], x[:3], level_scales=10., shape_scales=10.)
    high = model.predict(point[:3], x[:3], level_scales=50., shape_scales=50.)
    assert np.all(high["level_scale"] > low["level_scale"])
    assert np.all(high["shape_scale"] > low["shape_scale"])
    changed_scales = scales.copy()
    changed_scales[dates.month > 6] = 1e5
    other = ConditionalTransport(["A", "B"], mode).fit(
        point, truth, x, regions, days=dates, level_scales=changed_scales, shape_scales=changed_scales)
    for name in ("level_model_", "shape_model_"):
        for index in (0, 1):
            np.testing.assert_array_equal(getattr(model, name)[index], getattr(other, name)[index])
    second = other.predict(point[:3], x[:3], level_scales=50., shape_scales=50.)
    for key in ("level_scale", "shape_scale", "bias"):
        np.testing.assert_array_equal(high[key], second[key])
