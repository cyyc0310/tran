"""Behavioral tests for weather-only increments and the joint-day QRF baseline."""
import numpy as np
import pandas as pd
import pytest

from scripts.weather_increment_v12 import (
    JointQuantileForest, ScaleModel, ShapeModel, delta_transport,
)
from scripts.weather_shapes_v11 import ScaleModel as FrozenScaleModel
from scripts.weather_shapes_v11 import ShapeModel as FrozenShapeModel


def profiles():
    rng = np.random.default_rng(17023)
    q = rng.uniform(0.1, 3.0, (3, 5, 24, 8))
    q[..., 0] = 1.0
    q[..., 1] = 0.0
    support = np.ones((3, 5, 8), dtype=bool)
    support[..., 0] = False
    return rng, q, support


def cohort():
    rng = np.random.default_rng(17023)
    x = rng.normal(size=(120, 3))
    x[:, 1] = np.nan
    # A clear, learnable conditional distribution with whole-day dependence.
    x[:60, 0] = -10.0
    x[60:, 0] = 10.0
    errors = (np.arange(120.0)[:, None] + np.arange(24.0)[None, :] / 100.0)
    regions = np.array(["BPAT"] * 30 + ["CISO"] * 90)
    days = pd.date_range("2024-07-01", periods=len(x), tz="UTC")
    return x, errors, regions, days


@pytest.fixture(scope="module")
def forest():
    x, errors, regions, days = cohort()
    return JointQuantileForest().fit(x, errors, regions, days=days, exclude_region="ERCO")


def test_frozen_v11_models_are_reused_without_forking():
    assert ShapeModel is FrozenShapeModel
    assert ScaleModel is FrozenScaleModel


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_transport_zero_strength_and_zero_delta_are_bytewise_identity(dtype):
    rng, q, support = profiles()
    q = q.astype(dtype)
    q[..., 0, 1] = -0.0
    prediction = rng.normal(size=q.shape)
    pairs = (
        (prediction, prediction * 2, prediction * 3, prediction * 4, 0.0),
        (prediction, prediction, prediction, prediction, 1.0),
        (np.full(q.shape, 3.0), np.ones(q.shape), np.full(q.shape, 4.0),
         np.full(q.shape, 2.0), 0.5),
    )
    for query_w, query_b, donor_w, donor_b, strength in pairs:
        result = delta_transport(q, query_w, query_b, donor_w, donor_b, support, strength)
        assert result.dtype == q.dtype
        assert result.tobytes() == q.tobytes()


@pytest.mark.parametrize("strength", [0.25, 0.5, 1.0])
def test_transport_preserves_each_joint_donor_daily_fuel_mean_zero_and_support(strength):
    rng, q, support = profiles()
    query_w = rng.normal(size=(3, 1, 24, 8)) * 8
    base = np.ones((24, 8))
    donor_w = rng.normal(size=q.shape)
    result = delta_transport(q, query_w, base, donor_w, base, support, strength)
    assert result.shape == q.shape
    assert np.isfinite(result).all() and (result >= 0).all()
    np.testing.assert_allclose(result.mean(axis=-2), q.mean(axis=-2), atol=2e-14)
    np.testing.assert_array_equal(result[..., 0], q[..., 0])
    np.testing.assert_array_equal(result[..., 1], q[..., 1])
    assert not np.allclose(result[..., 2], q[..., 2])


def test_transport_changes_exactly_query_weather_minus_donor_weather_increment():
    q = np.ones((24, 8))
    base = np.ones_like(q)
    query_w = base.copy()
    query_w[0, 2] += 0.4
    query_w[1, 2] -= 0.4
    donor_w = base.copy()
    donor_w[0, 2] += 0.2
    donor_w[1, 2] -= 0.2
    result = delta_transport(q, query_w, base, donor_w, base, np.ones(8), 0.5)
    expected = q.copy()
    expected[0, 2] += 0.1
    expected[1, 2] -= 0.1
    np.testing.assert_allclose(result, expected, atol=1e-14)
    np.testing.assert_array_equal(result[:, :2], q[:, :2])


@pytest.mark.parametrize("bad", [np.full((24, 8), np.nan), np.ones((23, 8)),
                                  np.full((24, 8), np.inf)])
def test_transport_validates_predictions_even_at_zero_strength(bad):
    q = np.ones((24, 8))
    with pytest.raises(ValueError):
        delta_transport(q, bad, q, q, q, np.ones(8), 0.0)


def test_transport_rejects_bad_support_amplitudes_broadcasts_and_strength():
    q = np.ones((24, 8))
    for flags in (np.full(8, 0.5), np.ones(7)):
        with pytest.raises(ValueError):
            delta_transport(q, q, q, q, q, flags)
    with pytest.raises(ValueError, match="unsupported"):
        delta_transport(q * 2, q, q, q, q, np.zeros(8), 0.0)
    for strength in (-1, 0.1, np.nan, np.inf):
        with pytest.raises(ValueError, match="strength"):
            delta_transport(q, q, q, q, q, np.ones(8), strength)
    with pytest.raises(ValueError, match="nonnegative"):
        delta_transport(-q, q, q, q, q, np.ones(8))
    with pytest.raises(ValueError, match="broadcast"):
        delta_transport(q, np.ones((2, 24, 8)), q, q, q, np.ones(8))


def test_qrf_forest_spec_proper_only_imputation_and_source_audit(forest):
    x, _, _, _ = cohort()
    assert forest.model_.n_estimators == 128
    assert forest.model_.min_samples_leaf == 20
    assert forest.model_.max_features == 0.7
    assert forest.model_.bootstrap is True
    assert forest.model_.random_state == 17023
    assert forest.model_.n_jobs == 1
    assert forest.model_.n_features_in_ == 2 * x.shape[1]
    before = forest.medians_.copy()
    weights = forest.predict_weights([[1e20, 1e20, np.nan], [0, np.nan, 0]])
    np.testing.assert_array_equal(before, forest.medians_)
    assert before[1] == 0.0
    assert np.isfinite(weights).all()
    assert forest.training_regions_ == ("BPAT", "CISO")
    assert forest.audit_["dates_validated"] is True
    assert forest.audit_["excluded_evaluation_region"] == "ERCO"
    assert forest.audit_["joint_day_sampling"] is True
    assert forest.audit_["coverage_guarantee"] is False


def test_qrf_weights_follow_region_balanced_leaf_formula_and_global_mix(forest):
    x, _, regions, _ = cohort()
    query = x[[0, -1]]
    leaves = forest.model_.apply(forest._design(x))
    query_leaves = forest.model_.apply(forest._design(query))
    region_mass = np.where(regions == "BPAT", 2.0, 2.0 / 3.0)
    expected = np.zeros((2, len(x)))
    for row in range(2):
        for tree in range(128):
            same = leaves[:, tree] == query_leaves[row, tree]
            expected[row, same] += region_mass[same] / region_mass[same].sum() / 128
    weights = forest.predict_weights(query)
    np.testing.assert_allclose(weights, expected, atol=1e-15)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    assert weights[0, :60].sum() > 0.8
    assert weights[1, 60:].sum() > 0.8
    np.testing.assert_allclose(forest.global_weights_[regions == "BPAT"].sum(), 0.5)
    np.testing.assert_allclose(forest.global_weights_[regions == "CISO"].sum(), 0.5)
    for fraction in (0.25, 0.5):
        mixed = forest.predict_weights(query, fraction)
        np.testing.assert_allclose(mixed, (1 - fraction) * weights
                                   + fraction * forest.global_weights_[None, :])
        assert np.all(mixed > 0)


def test_qrf_draw_samples_one_whole_calibration_day_and_reuses_paired_uniforms(forest):
    x, errors, _, _ = cohort()
    query = x[[0, -1]]
    uniforms = np.array([[0.0, 0.1, 0.5, 0.9, 1.0]] * 2)
    result = forest.draw(query, uniforms, 0.25)
    assert result.shape == (2, 5, 24)
    np.testing.assert_array_equal(result, forest.draw(query, uniforms, 0.25))
    weights = forest.predict_weights(query, 0.25)
    for row in range(2):
        cumulative = np.cumsum(weights[row])
        cumulative[-1] = 1.0
        indices = np.minimum(np.searchsorted(cumulative, uniforms[row], side="right"), 119)
        np.testing.assert_array_equal(result[row], errors[indices])
    # Every path has the unique known hour offsets of a complete training day.
    np.testing.assert_allclose(result - result[..., :1],
                               np.broadcast_to(np.arange(24.0) / 100.0, result.shape), atol=1e-13)


def test_qrf_weighted_quantiles_are_monotone_and_match_empirical_cdf(forest):
    x, errors, _, _ = cohort()
    query = x[[0, -1]]
    levels = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    result = forest.predict_quantiles(query, levels, 0.5)
    assert result.shape == (2, 5, 24)
    assert np.isfinite(result).all()
    assert np.all(np.diff(result, axis=1) >= 0)
    weights = forest.predict_weights(query, 0.5)
    for row in range(2):
        cumulative = np.cumsum(weights[row])
        cumulative[-1] = 1.0
        indices = np.searchsorted(cumulative, levels, side="left")
        np.testing.assert_array_equal(result[row], errors[indices])
    np.testing.assert_array_equal(result, forest.predict_quantiles(query, levels, 0.5))


def test_qrf_fit_and_prediction_are_deterministic(forest):
    x, errors, regions, days = cohort()
    repeated = JointQuantileForest().fit(x, errors, regions, days=days, exclude_region="ERCO")
    np.testing.assert_array_equal(forest.predict_weights(x[:3]), repeated.predict_weights(x[:3]))


def test_qrf_endpoint_draws_and_quantiles_ignore_zero_weight_days():
    x = np.repeat([[-1.0], [1.0]], 200, axis=0)
    errors = np.broadcast_to(np.arange(400.0)[:, None], (400, 24)).copy()
    regions = np.array(["BPAT", "CISO"] * 200)
    model = JointQuantileForest().fit(x, errors, regions)
    weights = model.predict_weights(x[:1])[0]
    assert np.all(weights[200:] == 0.0)
    endpoints = model.draw(x[:1], [[0.0, 1.0]])
    quantiles = model.predict_quantiles(x[:1], [0.0, 1.0])
    np.testing.assert_array_equal(endpoints[0], errors[[0, 199]])
    np.testing.assert_array_equal(quantiles, endpoints)


def test_qrf_guards_source_identity_exclusions_and_h2_temporal_split():
    x, errors, regions, days = cohort()
    with pytest.raises(ValueError, match="excluded"):
        JointQuantileForest().fit(x, errors, regions, exclude_region="BPAT")
    with pytest.raises(ValueError, match="known source"):
        JointQuantileForest().fit(x, errors, regions, exclude_region="AZPS")
    for labels in (np.where(regions == "BPAT", "AZPS", regions), ["BPAT"] * len(x)):
        with pytest.raises(ValueError, match="known source"):
            JointQuantileForest().fit(x, errors, labels)
    for bad_days in (days + pd.DateOffset(years=1), days - pd.DateOffset(months=6),
                     days[:-1], [pd.NaT] * len(x)):
        with pytest.raises(ValueError, match="2024H2"):
            JointQuantileForest().fit(x, errors, regions, days=bad_days)
    with pytest.raises(ValueError, match="errors"):
        JointQuantileForest().fit(x, np.full_like(errors, np.nan), regions)
    with pytest.raises(ValueError, match="features"):
        JointQuantileForest().fit(np.full_like(x, np.inf), errors, regions)


def test_qrf_prediction_guards_dimensions_uniforms_quantiles_and_fractions(forest):
    x, _, _, _ = cohort()
    with pytest.raises(RuntimeError, match="fit"):
        JointQuantileForest().predict_weights(x[:2])
    for bad_x in (np.ones((2, 4)), np.ones(3), np.ones((0, 3)), np.full((2, 3), np.inf)):
        with pytest.raises(ValueError):
            forest.predict_weights(bad_x)
    for bad in (np.ones((1, 2)), np.ones(2), np.full((2, 2), np.nan),
                np.full((2, 2), -0.01), np.full((2, 2), 1.01)):
        with pytest.raises(ValueError, match="uniforms"):
            forest.draw(x[:2], bad)
    for bad in ([], [[0.5]], [np.nan], [-0.1], [1.01]):
        with pytest.raises(ValueError, match="levels"):
            forest.predict_quantiles(x[:2], bad)
    for bad in (-1, 0.1, 1, np.nan):
        with pytest.raises(ValueError, match="global_fraction"):
            forest.predict_weights(x[:2], bad)
