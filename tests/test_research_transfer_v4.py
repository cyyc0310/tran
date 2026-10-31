import numpy as np
import pandas as pd
import pytest

from transcif.research.transfer_v4 import (
    KernelConfig, support_nodes, integrated_covariance, condition_level, anchored_level,
)
from transcif.research.eia_proxy import FUELS


def history():
    result = pd.DataFrame({"period_start": ["2022-01-01"], "period_end": ["2022-02-01"],
                           "available_at": ["2022-03-03"], "coverage": [1.]})
    for fuel in FUELS:
        result[f"fuelshare_{fuel}"] = float(fuel == "gas")
    return result


def test_integrated_kernel_is_positive_definite_and_preserves_constant():
    config = KernelConfig(seasonal_sd=0, drift_sd=0)
    nodes, w = support_nodes([0., 30.], [30., 365.])
    covariance = integrated_covariance(nodes, w, nodes, w, config)
    np.testing.assert_allclose(covariance, config.baseline_sd**2)
    full = integrated_covariance(nodes, w, nodes, w, KernelConfig())
    assert np.linalg.eigvalsh(full).min() > 0


def test_future_unreleased_and_invalid_values_cannot_change_prediction():
    origins = pd.DatetimeIndex(["2022-02-15", "2022-03-04"], tz="UTC")
    frame = history()
    mean, sd = condition_level(frame, origins, 200., KernelConfig())
    assert mean[0] == 200.
    assert abs(mean[1]-400) < abs(200-400)
    future = frame.copy()
    future.available_at = "2027-01-01"
    future.period_end = "2026-12-01"
    future["fuelshare_gas"] = np.nan
    changed, changed_sd = condition_level(pd.concat([frame, future]), origins, 200., KernelConfig())
    np.testing.assert_allclose(changed, mean)
    np.testing.assert_allclose(changed_sd, sd)


def test_permanent_level_does_not_decay_to_source_mean():
    config = KernelConfig(seasonal_sd=0, drift_sd=0, noise_sd=1)
    mean, sd = condition_level(history(), pd.DatetimeIndex(["2022-04-01", "2026-04-01"], tz="UTC"), 200., config)
    np.testing.assert_allclose(mean[0], mean[1])
    assert mean[1] > 399
    np.testing.assert_allclose(sd[0], sd[1])


def test_window_integration_differs_from_endpoint_and_predictive_variance_is_nonnegative():
    origins = pd.DatetimeIndex(["2022-03-04", "2022-09-01"], tz="UTC")
    m1, s1 = condition_level(history(), origins, 200., KernelConfig(baseline_sd=20))
    m2, _ = condition_level(history(), origins, 200., KernelConfig(integrate=False, baseline_sd=20))
    assert not np.allclose(m1, m2)
    assert np.isfinite(s1).all() and (s1 >= 0).all()


def test_invalid_released_simplex_fails_and_invalid_support_fails():
    frame = history()
    frame["fuelshare_gas"] = -1
    with pytest.raises(ValueError, match="simplex"):
        condition_level(frame, pd.DatetimeIndex(["2022-04-01"], tz="UTC"), 200., KernelConfig())
    with pytest.raises(ValueError, match="supports"):
        support_nodes([1], [0])


def test_mixed_datetime_resolutions_preserve_asof_and_kernel_timescale():
    origins = pd.DatetimeIndex(["2022-04-01"], tz="UTC")
    frame = history()
    predictions = [condition_level(frame, origins.as_unit(unit), 200., KernelConfig())
                   for unit in ("ns", "us", "ms")]
    for mean, sd in predictions[1:]:
        np.testing.assert_allclose(mean, predictions[0][0])
        np.testing.assert_allclose(sd, predictions[0][1])
    frame.period_start = "2026-01-01"
    frame.period_end = "2026-02-01"
    frame.available_at = "2027-01-01"
    for unit in ("ns", "us"):
        mean, _ = condition_level(frame, origins.as_unit(unit), 200., KernelConfig())
        assert mean[0] == 200.


def test_predictive_variance_matches_kernel_diagonal_without_observations():
    config = KernelConfig()
    _, sd = condition_level(history(), pd.DatetimeIndex(["2022-01-01"], tz="UTC"), 200., config)
    nodes = np.array([[19000.]])
    diagonal = integrated_covariance(nodes, np.ones(1), nodes, np.ones(1), config)[0, 0]
    np.testing.assert_allclose(sd[0]**2, diagonal)


def test_same_support_innovation_bridge_uses_only_released_annual_differences():
    old = history()
    recent = history()
    recent.period_start, recent.period_end, recent.available_at = "2023-01-01", "2023-02-01", "2023-03-03"
    recent["fuelshare_gas"], recent["fuelshare_coal"] = .9, .1
    frame = pd.concat([old, recent], ignore_index=True)
    origins = pd.DatetimeIndex(["2023-02-15", "2023-03-04"], tz="UTC")
    level, sd = anchored_level(frame, origins, np.array([300., 300.]), np.array([50., 50.]), 90)
    assert level[0] == 300. and sd[0] == 50.
    assert 300. < level[1] < 360.
    assert sd[1] >= 20.
    changed = recent.copy()
    changed.available_at = "2027-01-01"
    changed["fuelshare_gas"] = np.nan
    other, other_sd = anchored_level(pd.concat([frame, changed]), origins, np.array([300., 300.]), np.array([50., 50.]), 90)
    np.testing.assert_allclose(other, level)
    np.testing.assert_allclose(other_sd, sd)
