import numpy as np
import pytest
from transcif.evaluation.metrics import rmse, mae, smape, cross_domain_degradation_rate


def test_rmse_zero_for_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0


def test_mae_known_value():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([12.0, 17.0])
    assert mae(y_true, y_pred) == pytest.approx(2.5)


def test_smape_known_value():
    y_true = np.array([100.0])
    y_pred = np.array([110.0])
    result = smape(y_true, y_pred)
    assert result == pytest.approx(2 * 10 / 210 * 100, rel=1e-6)


def test_cross_domain_degradation_rate_positive_when_worse():
    rate = cross_domain_degradation_rate(in_domain_metric=10.0, cross_domain_metric=15.0)
    assert rate == pytest.approx(50.0)


def test_cross_domain_degradation_rate_raises_on_zero_baseline():
    with pytest.raises(ValueError):
        cross_domain_degradation_rate(in_domain_metric=0.0, cross_domain_metric=5.0)
