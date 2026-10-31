import inspect
import pickle

import numpy as np
import pandas as pd
import pytest

from transcif.research.baselines_v7 import (
    DEFAULT_CQR_PARAMETERS, DEFAULT_TREE_PARAMETERS, DirectBaselines, equal_region_weights,
)


SMALL_TREE = {"n_estimators": 8, "min_samples_leaf": 2}
SMALL_CQR = {"max_iter": 8, "max_leaf_nodes": 5, "min_samples_leaf": 5}


def cohort():
    rng = np.random.default_rng(17023)
    x_fit, x_cal = rng.normal(size=(60, 5)), rng.normal(size=(30, 5))
    pattern = 12 * np.sin(2 * np.pi * np.arange(24) / 24)
    y_fit = 300 + 15 * x_fit[:, :1] + pattern + rng.normal(0., 2., (60, 24))
    y_cal = 300 + 15 * x_cal[:, :1] + pattern + rng.normal(0., 5., (30, 24))
    regions_fit = np.repeat(["A", "B"], [20, 40])
    regions_cal = np.repeat(["A", "B"], [10, 20])
    return x_fit, y_fit, regions_fit, x_cal, y_cal, regions_cal


def fit_model(*values):
    return DirectBaselines(tree_parameters=SMALL_TREE, cqr_parameters=SMALL_CQR).fit(*values)


def test_defaults_are_the_fixed_strong_comparators_and_cqr_has_no_fake_ensemble():
    assert DEFAULT_TREE_PARAMETERS == {"n_estimators": 150, "min_samples_leaf": 10,
                                       "random_state": 17023, "n_jobs": 1}
    assert DEFAULT_CQR_PARAMETERS["max_iter"] == 80
    assert DEFAULT_CQR_PARAMETERS["max_leaf_nodes"] == 15
    assert DEFAULT_CQR_PARAMETERS["l2_regularization"] == 10
    model = fit_model(*cohort())
    prediction = model.predict(cohort()[3][:2], ["2025-01-01", "2025-01-02"])
    assert set(prediction) == {"extratrees", "cqr"}
    assert set(prediction["cqr"]) == {"lower", "upper"}
    assert prediction["extratrees"]["ensemble"].shape == (2, 128, 24)
    assert all(value.shape == (2, 24) for value in prediction["cqr"].values())
    assert model.cqr_models_[0].loss == model.cqr_models_[1].loss == "quantile"
    assert model.cqr_models_[0].quantile == .05
    assert model.cqr_models_[1].quantile == .95
    assert model.audit_["cqr_interval_only"]


def test_fit_and_calibration_give_each_region_equal_probability_and_mean_one_estimator_weights():
    x = np.zeros((22, 3))
    regions = np.array(["A"] * 2 + ["B"] * 20)
    y = np.repeat(np.r_[np.full(2, 100.), np.full(20, 900.)][:, None], 24, axis=1)
    model = DirectBaselines(tree_parameters={**SMALL_TREE, "min_samples_leaf": 1},
                            cqr_parameters=SMALL_CQR).fit(x, y, regions, x, y, regions)
    np.testing.assert_allclose(model.predict(x[:1], ["2025-01-01"])["extratrees"]["point"], 500.)
    for region in ("A", "B"):
        assert model.fit_probability_weights_[regions == region].sum() == pytest.approx(.5)
        assert model.cal_probability_weights_[regions == region].sum() == pytest.approx(.5)
        hours = np.repeat(regions == region, 24)
        assert model.fit_hour_probability_weights_[hours].sum() == pytest.approx(.5)
    assert model.fit_estimator_weights_.mean() == pytest.approx(1.)
    np.testing.assert_array_equal(equal_region_weights(regions), model.fit_probability_weights_)


def test_source_calibration_noise_changes_uncertainty_without_refitting_source_models():
    rng = np.random.default_rng(17024)
    x_fit = rng.normal(size=(40, 3))
    x_cal = rng.normal(size=(20, 3))
    regions_fit, regions_cal = np.repeat(["A", "B"], 20), np.repeat(["A", "B"], 10)
    y_fit = np.full((40, 24), 300.)
    calm = np.full((20, 24), 300.)
    noisy = calm + rng.uniform(-60, 60, calm.shape)
    first = fit_model(x_fit, y_fit, regions_fit, x_cal, calm, regions_cal)
    second = fit_model(x_fit, y_fit, regions_fit, x_cal, noisy, regions_cal)
    a, b = first.predict(x_cal[:1], ["2025-01-01"]), second.predict(x_cal[:1], ["2025-01-01"])
    np.testing.assert_array_equal(a["extratrees"]["point"], b["extratrees"]["point"])
    assert b["extratrees"]["ensemble"].std() > a["extratrees"]["ensemble"].std() + 20
    assert (b["cqr"]["upper"] - b["cqr"]["lower"]).mean() > 80
    assert second.cqr_offset_ > first.cqr_offset_ + 40


def test_calibration_rows_do_not_fit_normalization_tree_or_quantile_functions():
    values = list(cohort())
    original = fit_model(*values)
    values[3] = values[3] + 1e6
    changed = fit_model(*values)
    np.testing.assert_array_equal(original.feature_center_, changed.feature_center_)
    np.testing.assert_array_equal(original.feature_scale_, changed.feature_scale_)
    test_x = cohort()[3][:3]
    np.testing.assert_array_equal(original.tree_.predict(original._transform(test_x)),
                                  changed.tree_.predict(changed._transform(test_x)))
    np.testing.assert_array_equal(original._quantiles(test_x), changed._quantiles(test_x))


def test_origin_seed_is_batch_order_resolution_invariant_and_pickle_preserves_output():
    values = cohort()
    model = fit_model(*values)
    origins = pd.DatetimeIndex(["2025-01-01", "2025-02-03", "2025-12-01"], tz="UTC")
    x = values[3][:3]
    a = model.predict(x, origins, seed=19)
    b = pickle.loads(pickle.dumps(model)).predict(x[::-1], origins[::-1].as_unit("us"), seed=19)
    single = model.predict(x[:1], origins[:1], seed=19)
    for method in a:
        for key in a[method]:
            np.testing.assert_array_equal(a[method][key], b[method][key][::-1])
            np.testing.assert_array_equal(a[method][key][:1], single[method][key])


def test_calibration_resampling_preserves_daily_pairs_and_equal_region_mass():
    x_fit, y_fit, regions_fit, _, _, _ = cohort()
    y_fit[:] = 300
    x_cal = np.zeros((11, 5))
    regions_cal = np.array(["A"] + ["B"] * 10)
    pattern = np.arange(24.) - 11.5
    y_cal = 300 + np.r_[-10., np.full(10, 10.)][:, None] + pattern
    model = fit_model(x_fit, y_fit, regions_fit, x_cal, y_cal, regions_cal)
    result = model.predict(x_cal[:1], ["2025-01-01"], n_scenarios=10000)["extratrees"]
    ensemble = result["ensemble"][0]
    np.testing.assert_allclose(ensemble - ensemble.mean(axis=1, keepdims=True), np.broadcast_to(pattern, ensemble.shape))
    assert .47 < np.mean(ensemble.mean(axis=1) > 300) < .53


def test_signed_cqr_rank_crossing_repair_bounds_and_label_free_predict():
    values = cohort()
    model = fit_model(*values)
    lo, hi = model._quantiles(values[3])
    scores = np.maximum(lo - values[4], values[4] - hi).ravel()
    rank = int(np.ceil((len(scores) + 1) * .9))
    assert model.cqr_offset_ == np.partition(scores, rank - 1)[rank - 1]
    # A signed negative offset can locally cross; engineering repair is explicit.
    model.cqr_offset_ = -1000.
    result = model.predict(values[3][:2], ["2026-01-01", "2026-01-02"])
    np.testing.assert_array_equal(result["cqr"]["lower"], result["cqr"]["upper"])
    for method in result.values():
        for array in method.values():
            assert np.isfinite(array).all() and array.min() >= 0 and array.max() <= 1000
    assert tuple(inspect.signature(model.predict).parameters) == ("x", "origins", "n_scenarios", "seed")
    with pytest.raises(TypeError):
        model.predict(values[3][:2], ["2026-01-01", "2026-01-02"], y_future=values[4][:2])


def test_split_regions_shapes_and_invalid_values_are_rejected():
    values = list(cohort())
    values[-1] = np.repeat(["A", "TARGET"], [10, 20])
    with pytest.raises(ValueError, match="both proper-fit"):
        fit_model(*values)
    values = list(cohort())
    values[3] = values[3][:, :2]
    with pytest.raises(ValueError, match="matching P"):
        fit_model(*values)
    values = list(cohort())
    values[4][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_model(*values)
