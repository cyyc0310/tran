import numpy as np
import pytest
from transcif.calibration.conformal import (
    compute_nonconformity_scores,
    conformal_interval_halfwidth,
    predict_with_interval,
    empirical_coverage,
)


def test_compute_nonconformity_scores_is_absolute_error():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    scores = compute_nonconformity_scores(y_true, y_pred)
    np.testing.assert_allclose(scores, [2.0, 2.0, 3.0])


def test_conformal_interval_halfwidth_raises_on_empty_input():
    with pytest.raises(ValueError):
        conformal_interval_halfwidth(np.array([]), coverage=0.9)


def test_conformal_prediction_achieves_target_coverage_on_held_out_data():
    rng = np.random.default_rng(123)
    calibration_errors = rng.normal(loc=0.0, scale=5.0, size=500)
    calibration_scores = np.abs(calibration_errors)
    halfwidth = conformal_interval_halfwidth(calibration_scores, coverage=0.9)

    test_true = rng.normal(loc=100.0, scale=1.0, size=1000)
    test_errors = rng.normal(loc=0.0, scale=5.0, size=1000)
    test_pred = test_true - test_errors

    lower, upper = predict_with_interval(test_pred, halfwidth)
    coverage = empirical_coverage(test_true, lower, upper)

    assert coverage >= 0.85
