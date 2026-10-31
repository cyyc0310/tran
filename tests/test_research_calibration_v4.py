"""Calibration tests focus on ranks, split boundaries and coherent distributions."""

import numpy as np
import pytest

from transcif.research.calibration_v4 import (
    MODES,
    SourceCalibrator,
    combine_component_scenarios,
    decompose_residual_paths,
    split_conformal_quantile,
)
from transcif.research.weighted_metrics import distribution_metrics


def fit_paths(paths, regions=None, ages=None, scales=None, months=None):
    paths = np.asarray(paths, dtype=float)
    n = len(paths)
    return SourceCalibrator(["a", "b", "c", "d", "e"]).fit(
        np.zeros_like(paths), paths, np.ones_like(paths) if scales is None else scales,
        ["a"] * n if regions is None else regions,
        ["young"] * n if ages is None else ages,
        months=months, years=[2023] * n,
    )


def test_finite_sample_pooled_rank_and_insufficient_sample():
    assert split_conformal_quantile(np.arange(1, 10), 0.1) == 9
    assert split_conformal_quantile(np.arange(1, 11), 0.1) == 10
    assert np.isinf(split_conformal_quantile([1, 2], 0.1))
    paths = np.arange(1, 49).reshape(2, 24)
    cal = fit_paths(paths)
    assert cal.q_["pooled"] == 45  # ceil(49 * .9), not empirical 90th percentile.
    lower, upper = cal.intervals(np.zeros(24), np.full(24, 2))
    np.testing.assert_array_equal(lower, -90)
    np.testing.assert_array_equal(upper, 90)


def test_equal_region_prevents_long_region_dominance():
    paths = np.concatenate([np.ones((100, 24)), np.full((1, 24), 9)])
    cal = fit_paths(paths, ["a"] * 100 + ["b"])
    assert cal.q_["pooled"] == 1
    assert cal.q_["equal_region"] == 9
    draws = cal.ensemble(np.zeros(24), np.ones(24), n_samples=10000, seed=2)
    assert 0.47 < (draws[:, 0] == 9).mean() < 0.53


def test_source_domain_protection_month_blocks_and_prespecified_aggregation():
    paths = np.repeat(np.array([1, 1, 1, 1, 50])[:, None], 24, axis=1)
    cal = fit_paths(paths, ["a", "b", "c", "d", "e"])
    assert cal.q_["worstgrid_max"] == 50
    assert cal.q_["protected_domain80"] == 1
    np.testing.assert_array_equal(cal.intervals(np.zeros(24), np.ones(24), "protected_max"),
                                  cal.intervals(np.zeros(24), np.ones(24), "worstgrid_max"))
    # A rare high-error month cannot disappear in the long source-domain pool.
    monthly = fit_paths(np.r_[np.ones((100, 24)), np.full((1, 24), 30)],
                        months=["2023-01"] * 100 + ["2023-02"])
    assert monthly.q_["equal_region"] == 1
    assert monthly.q_["protected_domain80"] == 30


def test_age_conditioned_factor_has_global_floor_and_unknown_age_fallback():
    paths = np.r_[np.full((10, 24), 10), np.ones((1, 24))]
    cal = fit_paths(paths, ages=["old"] * 10 + ["young"])
    assert cal.q_by_age_["young"]["protected_domain80"] == cal.q_["protected_domain80"] == 10
    lower, upper = cal.intervals(np.zeros((2, 24)), np.ones((2, 24)),
                                 "protected_domain80", ages=["young", "unseen"])
    np.testing.assert_array_equal(lower, -10)
    np.testing.assert_array_equal(upper, 10)
    assert cal.audit_["calibration_year"] == 2023
    assert not cal.audit_["target_labels_used"]


def test_target_region_and_wrong_year_cannot_enter_calibration():
    zeros = np.zeros((1, 24))
    for region, year in [("target", 2023), ("a", 2022), ("a", 2024), ("a", 2025)]:
        with pytest.raises(ValueError):
            SourceCalibrator(["a"]).fit(zeros, zeros, np.ones_like(zeros), [region], ["young"], years=[year])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_inputs_fail_fast(bad):
    for field in ("point", "truth", "scales"):
        arrays = {"point": np.zeros((2, 24)), "truth": np.ones((2, 24)), "scales": np.ones((2, 24))}
        arrays[field][0, 0] = bad
        with pytest.raises(ValueError):
            SourceCalibrator(["a"]).fit(**arrays, regions=["a"] * 2, ages=["young"] * 2)
    with pytest.raises(ValueError):
        split_conformal_quantile([1, bad])


@pytest.mark.parametrize("bad_scale", [0, -1])
def test_scales_must_be_positive(bad_scale):
    with pytest.raises(ValueError):
        fit_paths(np.ones((2, 24)), scales=np.full((2, 24), bad_scale))


def test_signed_paths_preserve_dependence_and_raw_intervals_use_empirical_quantiles():
    wave = np.r_[np.ones(12), -np.ones(12)]
    paths = np.stack([-3 * wave, 2 * wave])
    cal = fit_paths(paths, scales=np.full((2, 24), 7))
    samples = cal.ensemble(np.zeros(24), np.full(24, 100), n_samples=2000, seed=3)
    # Raw baseline preserves physical residual magnitudes, signs and hour pairing.
    assert set(samples[:, 0]) == {-3.0, 2.0}
    np.testing.assert_array_equal(samples[:, :12], -samples[:, 12:])
    lower, upper = cal.intervals(np.zeros(24), np.full(24, 100), "source_residual")
    np.testing.assert_array_equal(lower[:12], -3)
    np.testing.assert_array_equal(upper[:12], 2)
    score = distribution_metrics(samples, np.ones(len(samples)), wave)
    assert score["crps"] >= 0
    assert score["width90"] == 5


def test_calibrated_paths_use_same_positive_normalization_and_deterministic_draws():
    paths = np.r_[np.ones((100, 24)), np.full((1, 24), -9)]
    cal = fit_paths(paths, ["a"] * 100 + ["b"])
    balanced = cal.ensemble(np.zeros(24), np.ones(24), "equal_region", n_samples=1000, seed=10)
    pooled = cal.ensemble(np.zeros(24), np.ones(24), "pooled", n_samples=1000, seed=10)
    np.testing.assert_allclose(pooled, balanced / 9)
    np.testing.assert_array_equal(pooled, cal.ensemble(np.zeros(24), np.ones(24), "pooled", n_samples=1000, seed=10))
    assert cal.ensemble(np.zeros((2, 24)), np.ones((2, 24)), n_samples=7).shape == (2, 7, 24)


def test_level_and_zero_mean_shape_have_exact_separate_scaling():
    residuals = np.arange(48).reshape(2, 24) - 10
    level, shape = decompose_residual_paths(residuals)
    np.testing.assert_allclose(shape.mean(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(combine_component_scenarios(np.zeros(24), level, shape), residuals)
    scenarios = combine_component_scenarios(np.ones(24), level, shape, level_scale=2, shape_scale=3)
    np.testing.assert_allclose(scenarios.mean(axis=1), 1 + 2 * level)
    np.testing.assert_allclose(scenarios - scenarios.mean(axis=1)[:, None], 3 * shape)
    with pytest.raises(ValueError, match="zero daily mean"):
        combine_component_scenarios(np.zeros(24), level, shape + 1)
    with pytest.raises(ValueError):
        combine_component_scenarios(np.zeros(24), level, shape, shape_scale=np.ones((2, 24)))


def test_zero_residuals_remain_degenerate_and_invalid_requests_raise():
    cal = fit_paths(np.zeros((2, 24)))
    for mode in MODES:
        np.testing.assert_array_equal(cal.ensemble(np.ones(24), np.ones(24), mode), 1)
    with pytest.raises(RuntimeError):
        SourceCalibrator(["a"]).intervals(np.zeros(24), np.ones(24))
    with pytest.raises(ValueError):
        cal.ensemble(np.zeros(24), np.ones(24), n_samples=0)
    with pytest.raises(ValueError):
        cal.intervals(np.zeros(24), np.ones(24), "unknown")
    with pytest.raises(ValueError):
        cal.intervals(np.zeros(24), np.ones(24), ages=[None])


def test_insufficient_rank_returns_unbounded_interval_but_cannot_score_infinite_ensemble():
    cal = SourceCalibrator(["a"], alpha=0.001).fit(
        np.zeros((1, 24)), np.ones((1, 24)), np.ones((1, 24)), ["a"], ["young"])
    lower, upper = cal.intervals(np.zeros(24), np.ones(24), "pooled")
    assert np.isneginf(lower).all() and np.isposinf(upper).all()
    with pytest.raises(ValueError, match="Too few calibration scores"):
        cal.ensemble(np.zeros(24), np.ones(24), "pooled")


def test_fitted_residual_bank_is_independent_of_later_caller_mutation():
    point = np.zeros((2, 24))
    truth = np.stack([np.full(24, -1), np.full(24, 2)])
    scales = np.ones((2, 24))
    cal = SourceCalibrator(["a"]).fit(point, truth, scales, ["a", "a"], ["young", "young"])
    truth[:] = 999
    scales[:] = 500
    samples = cal.ensemble(np.zeros(24), np.ones(24), n_samples=100)
    assert set(samples[:, 0]) == {-1.0, 2.0}
