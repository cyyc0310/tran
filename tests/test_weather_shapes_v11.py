"""Behavioral checks for within-fuel transport and source-only V11 references."""
import numpy as np
import pandas as pd
import pytest

from scripts.weather_shapes_v11 import (ShapeModel, ScaleModel, project_shapes,
                                       residual_paths, split_profiles, transport)


def fixture():
    rng = np.random.default_rng(17023)
    n = 60
    x = rng.normal(size=(n, 3))
    x[:, 1] = np.nan
    q = rng.uniform(.1, 3., (n, 24, 8))
    q[:, :, 0] = 1.  # unsupported flat prior
    q[:, :, 1] = 0.  # supported fuel, absent this day
    support = np.ones((n, 8), bool)
    support[:, 0] = False
    return x, q, np.array(["BPAT"] * 20 + ["CISO"] * 40), support


def test_projection_is_euclidean_simplex_and_identity_on_valid_shapes():
    values = np.zeros((2, 24, 8))
    values[:, 0] = 100.
    projected = project_shapes(values)
    np.testing.assert_allclose(projected[:, 0], 24.)
    np.testing.assert_array_equal(projected[:, 1:], 0.)
    np.testing.assert_allclose(projected.mean(axis=1), 1.)
    np.testing.assert_allclose(project_shapes(projected), projected)
    # Euclidean projection, unlike clipping/renormalization, subtracts a common
    # threshold from active hours.
    two = np.zeros((24, 8)); two[:2] = np.array([[20.], [10.]])
    answer = project_shapes(two)
    np.testing.assert_allclose(answer[:2], np.array([[17.]*8, [7.]*8]))


@pytest.mark.parametrize("keep", [False, True])
def test_transport_preserves_every_donor_daily_fuel_energy_and_support(keep):
    _, q, _, support = fixture()
    rng = np.random.default_rng(91)
    query = project_shapes(rng.normal(size=(24, 8)))
    donor_prediction = project_shapes(rng.normal(size=q.shape))
    result = transport(q, query, donor_prediction, support, keep_shape_residual=keep)
    assert result.shape == q.shape
    assert np.isfinite(result).all() and (result >= 0).all()
    np.testing.assert_allclose(result.mean(axis=1), q.mean(axis=1), atol=1e-12)
    np.testing.assert_array_equal(result[:, :, 0], 1.)
    np.testing.assert_array_equal(result[:, :, 1], 0.)


def test_split_reconstructs_profiles_and_transport_has_exact_identity():
    _, q, _, support = fixture()
    amplitude, shape = split_profiles(q)
    np.testing.assert_allclose(shape.mean(axis=1), 1.)
    np.testing.assert_allclose(amplitude[:, None] * shape, q)
    identity = transport(q, shape, shape, support)
    np.testing.assert_allclose(identity, q, atol=1e-12)
    with pytest.raises(ValueError, match="unsupported donor"):
        transport(q * 2, shape, shape, support)


def test_shape_model_fits_source_only_and_prediction_does_not_refit_imputer():
    x, q, regions, support = fixture()
    model = ShapeModel().fit(x, q, regions, support, exclude_region="ERCO",
                             days=pd.date_range("2024-02-01", periods=len(x), tz="UTC"))
    before = model.medians_.copy()
    predicted = model.predict(np.array([[1e10, 999., np.nan], [0., np.nan, 0.]]))
    assert predicted.shape == (2, 24, 8)
    assert (predicted >= 0).all()
    np.testing.assert_allclose(predicted.mean(axis=1), 1.)
    np.testing.assert_array_equal(predicted[:, :, :2], 1.)
    np.testing.assert_array_equal(before, model.medians_)
    assert before[1] == 0.
    assert model.training_regions_ == ("BPAT", "CISO")
    assert model.audit_["donor_predictions_crossfit_automatically"] is False
    assert model.models_[2].n_estimators == 64
    assert model.models_[2].min_samples_leaf == 20
    assert model.models_[2].n_jobs == 1
    # Root weighted mass is balanced by represented region despite 20/40 rows.
    root = model.models_[2].estimators_[0].tree_.weighted_n_node_samples[0]
    assert root == pytest.approx(len(x))


@pytest.mark.parametrize("kind", ["shape", "scale"])
def test_fit_rejects_target_excluded_and_future_labels(kind):
    x, q, regions, support = fixture()
    errors = q[:, :, 2] * 50
    def fit(labels, **kwargs):
        if kind == "shape":
            return ShapeModel().fit(x, q, labels, support, **kwargs)
        return ScaleModel().fit(x, errors, labels, **kwargs)
    with pytest.raises(ValueError, match="excluded"):
        fit(regions, exclude_region="BPAT")
    with pytest.raises(ValueError, match="known source"):
        fit(np.where(regions == "BPAT", "AZPS", regions))
    with pytest.raises(ValueError, match="future"):
        fit(regions, days=pd.date_range("2025-01-01", periods=len(x), tz="UTC"))
    with pytest.raises(ValueError, match="known source"):
        fit(regions, exclude_region="AZPS")


def test_two_channel_scale_floor_bounds_and_order():
    x, _, regions, _ = fixture()
    model = ScaleModel().fit(x, np.zeros((len(x), 24)), regions)
    np.testing.assert_allclose(model.predict(x[:3]), 5.)
    ramp = np.arange(24.) - 11.5
    model = ScaleModel().fit(x, np.broadcast_to(100.+ramp, (len(x), 24)), regions)
    scales = model.predict(x[:3])
    np.testing.assert_allclose(scales[:, 0], np.sqrt(100.**2+25.))
    np.testing.assert_allclose(scales[:, 1], np.sqrt(np.mean(ramp**2)+25.))
    model = ScaleModel().fit(x, np.full((len(x), 24), 1e5), regions)
    np.testing.assert_allclose(model.predict(x[:2])[:, 0], 500.)


def test_scale_date_guard_uses_crossfit_calibration_h2_not_shape_proper_h1():
    x, _, regions, _ = fixture()
    errors = np.zeros((len(x), 24))
    dates = pd.date_range("2024-07-01", periods=len(x), tz="UTC")
    model = ScaleModel().fit(x, errors, regions, days=dates)
    assert model.audit_["dates_validated"] is True
    assert model.audit_["fit_cohort"] == "source crossfit calibration 2024H2"
    with pytest.raises(ValueError, match="2024H2"):
        ScaleModel().fit(x, errors, regions,
                         days=pd.date_range("2024-02-01", periods=len(x), tz="UTC"))


def test_paired_residual_paths_keep_level_shape_and_joint_draw_identity():
    ramp = np.arange(24.) - 11.5
    residuals = np.stack((10.+ramp, -20.-2*ramp))
    indices = np.array([[0, 1, 0], [1, 0, 1]])
    center = np.full((2, 24), 300.)
    plain = residual_paths(center, residuals, indices)
    np.testing.assert_array_equal(plain, center[:, None] + residuals[indices])
    scaled = residual_paths(center, residuals, indices, scales=np.array([[20., 30.], [10., 40.]]),
                            calibration_scales=np.array([[10., 10.], [10., 10.]]))
    np.testing.assert_allclose(scaled[0, 0], 300.+20.+3*ramp)
    np.testing.assert_allclose(scaled[0, 1], 300.-40.-6*ramp)
    np.testing.assert_allclose(scaled[1, 1], 300.+10.+4*ramp)
    with pytest.raises(ValueError, match="both"):
        residual_paths(center, residuals, indices, scales=np.ones((2, 2)))
    with pytest.raises(ValueError, match="indices"):
        residual_paths(center, residuals, indices+2)


@pytest.mark.parametrize("bad", [np.full((24, 8), np.nan), np.full((24, 8), -1.), np.ones((23, 8))])
def test_profile_validation(bad):
    with pytest.raises(ValueError):
        split_profiles(bad)


def test_unfitted_prediction_guards():
    for model in (ShapeModel(), ScaleModel()):
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(np.ones((2, 3)))
