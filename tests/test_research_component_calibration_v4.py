"""Behavioral checks for source-only paired level/shape residual transport."""

import numpy as np
import pytest

from transcif.research.component_calibration_v4 import COMPONENT_MODES, ComponentCalibrator
from transcif.research.weighted_metrics import weighted_quantile


WAVE = np.r_[np.ones(12), -np.ones(12)]


def fit_bank(paths=None, regions=None, *, level_scales=None, shape_scales=None):
    paths = np.stack([2 + WAVE, -3 + 2 * WAVE]) if paths is None else np.asarray(paths)
    n = len(paths)
    return ComponentCalibrator(["a", "b"]).fit(
        np.zeros_like(paths), paths, np.ones_like(paths),
        ["a"] * n if regions is None else regions, ["young"] * n,
        months=["2023-01"] * n, years=[2023] * n,
        level_scales=level_scales, shape_scales=shape_scales,
    )


def test_matching_component_scales_exactly_reconstruct_source_residuals():
    cal = fit_bank(level_scales=np.full(2, 2), shape_scales=np.full((2, 24), 4))
    paths = cal.ensemble(np.zeros(24), np.ones(24), level_scale=2,
                         shape_scale=np.full(24, 4), n_samples=1000, seed=3)
    expected = np.stack([2 + WAVE, -3 + 2 * WAVE])
    assert all(np.any(np.all(row == expected, axis=1)) for row in paths)
    np.testing.assert_allclose(cal.shape_paths_.mean(axis=1), 0, atol=1e-14)
    # Level and shape amplitudes remain tied to the same source day.
    mean = paths.mean(axis=1)
    shape_amplitude = (paths - mean[:, None])[:, 0]
    np.testing.assert_array_equal(shape_amplitude, np.where(mean == 2, 1, 2))


def test_expanding_level_uncertainty_cannot_expand_shape():
    cal = fit_bank()
    first = cal.ensemble(np.zeros(24), np.ones(24), level_scale=1, shape_scale=np.ones(24), seed=7)
    expanded = cal.ensemble(np.zeros(24), np.ones(24), level_scale=5, shape_scale=np.ones(24), seed=7)
    np.testing.assert_allclose(expanded.mean(axis=1), 5 * first.mean(axis=1))
    np.testing.assert_allclose(expanded - expanded.mean(axis=1)[:, None],
                               first - first.mean(axis=1)[:, None])


def test_shape_hourly_scale_uses_scalar_daily_rms_and_stays_zero_mean():
    cal = fit_bank()
    shape_scale = np.r_[np.ones(12), np.full(12, 3)]
    paths = cal.ensemble(np.zeros(24), np.ones(24), level_scale=0, shape_scale=shape_scale)
    np.testing.assert_allclose(paths.mean(axis=1), 0, atol=1e-13)
    np.testing.assert_allclose(paths[:, :12], -paths[:, 12:])
    assert np.all(np.isclose(paths[:, 0], np.sqrt(5)) | np.isclose(paths[:, 0], 2 * np.sqrt(5)))


def test_source_shape_normalization_uses_rms_and_preserves_zero_mean():
    scales = np.tile(np.r_[np.full(12, 3), np.full(12, 4)], (2, 1))
    cal = fit_bank(level_scales=np.array([2, 3]), shape_scales=scales)
    np.testing.assert_allclose(cal.level_paths_, [1, -1])
    np.testing.assert_allclose(cal.shape_paths_[0], WAVE / np.sqrt(12.5))
    np.testing.assert_allclose(cal.shape_paths_[1], 2 * WAVE / np.sqrt(12.5))


def test_regions_receive_equal_mass_and_deterministic_intervals_use_whole_bank():
    residuals = np.r_[np.full((100, 24), -2), np.full((1, 24), 10)]
    cal = fit_bank(residuals, ["a"] * 100 + ["b"])
    samples = cal.ensemble(np.zeros(24), np.ones(24), n_samples=10000, seed=2)
    assert 0.47 < (samples[:, 0] == 10).mean() < 0.53
    lower, upper = cal.intervals(np.zeros(24), np.ones(24))
    np.testing.assert_array_equal(lower, -2)
    np.testing.assert_array_equal(upper, 10)
    expected = weighted_quantile(residuals, np.r_[np.full(100, 0.5 / 100), 0.5], [0.05, 0.95])
    np.testing.assert_array_equal(np.stack([lower, upper]), expected)
    assert cal.audit_["region_day_counts"] == {"a": 100, "b": 1}


def test_prespecified_110_mode_is_exact_joint_inflation():
    cal = fit_bank()
    point = np.full(24, 100)
    base = cal.ensemble(point, np.ones(24), n_samples=1000, seed=4)
    inflated = cal.ensemble(point, np.ones(24), "component_transport_110", n_samples=1000, seed=4)
    np.testing.assert_allclose(inflated - point, 1.1 * (base - point))
    lower, upper = cal.intervals(point, np.ones(24))
    low110, up110 = cal.intervals(point, np.ones(24), "component_transport_110")
    np.testing.assert_allclose(low110 - point, 1.1 * (lower - point))
    np.testing.assert_allclose(up110 - point, 1.1 * (upper - point))
    assert cal.q_ == {"component_transport": 1.0, "component_transport_110": 1.1}


def test_ages_do_not_select_different_source_path_banks_and_batch_shapes_match():
    cal = fit_bank()
    point, scale = np.zeros((3, 24)), np.ones((3, 24))
    first = cal.ensemble(point, scale, ages=["young"] * 3, n_samples=7, seed=10)
    other = cal.ensemble(point, scale, ages=["old", "missing", "unknown"], n_samples=7, seed=10)
    np.testing.assert_array_equal(first, other)
    assert first.shape == (3, 7, 24)
    lower, upper = cal.intervals(point, scale, level_scale=np.array([1, 2, 3]), shape_scale=scale)
    assert lower.shape == upper.shape == (3, 24)
    assert not cal.audit_["age_conditioning"]


def test_optional_component_scale_fallback_matches_explicit_rms():
    cal = fit_bank()
    scale = np.r_[np.full(12, 3), np.full(12, 4)]
    fallback = cal.ensemble(np.zeros(24), scale, seed=11)
    explicit = cal.ensemble(np.zeros(24), scale, level_scale=np.sqrt(12.5), shape_scale=scale, seed=11)
    np.testing.assert_array_equal(fallback, explicit)
    assert cal.audit_["component_scale_fallback"] == {"level": True, "shape": True}


def test_source_only_and_year_boundary_are_enforced():
    zero, one = np.zeros((1, 24)), np.ones((1, 24))
    for region, year in [("target", 2023), ("a", 2022), ("a", 2024), ("a", 2025)]:
        with pytest.raises(ValueError):
            ComponentCalibrator(["a"]).fit(zero, zero, one, [region], ["young"], years=[year])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_component_scales_are_rejected(bad):
    with pytest.raises(ValueError):
        fit_bank(level_scales=np.array([bad, 1]))
    with pytest.raises(ValueError):
        fit_bank(shape_scales=np.full((2, 24), bad))
    cal = fit_bank()
    with pytest.raises(ValueError):
        cal.intervals(np.zeros(24), np.ones(24), level_scale=bad)
    with pytest.raises(ValueError):
        cal.ensemble(np.zeros(24), np.ones(24), shape_scale=np.full(24, bad))


def test_fit_denominators_must_be_positive_but_zero_target_uncertainty_is_valid():
    with pytest.raises(ValueError):
        fit_bank(level_scales=np.array([0, 1]))
    with pytest.raises(ValueError):
        fit_bank(shape_scales=np.zeros((2, 24)))
    cal = fit_bank()
    for mode in COMPONENT_MODES:
        paths = cal.ensemble(np.ones(24), np.ones(24), mode, level_scale=0, shape_scale=np.zeros(24))
        np.testing.assert_array_equal(paths, 1)


def test_invalid_requests_and_caller_mutations_cannot_corrupt_residual_bank():
    truth = np.stack([2 + WAVE, -3 + 2 * WAVE])
    level_scales, shape_scales = np.ones(2), np.ones((2, 24))
    cal = fit_bank(truth, level_scales=level_scales, shape_scales=shape_scales)
    expected = cal.ensemble(np.zeros(24), np.ones(24), seed=3)
    truth[:], level_scales[:], shape_scales[:] = 999, 999, 999
    np.testing.assert_array_equal(cal.ensemble(np.zeros(24), np.ones(24), seed=3), expected)
    with pytest.raises(RuntimeError):
        ComponentCalibrator(["a"]).intervals(np.zeros(24), np.ones(24))
    with pytest.raises(ValueError):
        cal.ensemble(np.zeros(24), np.ones(24), "unknown")
    with pytest.raises(ValueError):
        cal.ensemble(np.zeros(24), np.ones(24), n_samples=0)
    with pytest.raises(ValueError):
        cal.ensemble(np.zeros(24), np.ones(24), level_scale=np.ones((1, 1)))
