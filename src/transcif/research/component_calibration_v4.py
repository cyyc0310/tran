"""Source-only transport of paired level and zero-mean shape residuals.

This is an empirical residual-transport candidate, not a conformal method or a
coverage guarantee under unseen-domain shift. All source ages are pooled. Source
regions receive equal probability, then days receive equal probability within
their region. Changing target level uncertainty leaves transported daily shape
unchanged. The two multipliers below are fixed candidates selected on source
2024; target hourly labels must never enter fitting or multiplier selection.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .calibration_v4 import _daily, _labels, decompose_residual_paths
from .weighted_metrics import weighted_quantile


COMPONENT_MODES = ("component_transport", "component_transport_110")
MODES = COMPONENT_MODES
_MULTIPLIERS = {"component_transport": 1.0, "component_transport_110": 1.10}


def _rms(values: np.ndarray) -> np.ndarray:
    """Daily RMS without overflowing when finite component scales are large."""
    largest = np.max(values, axis=1)
    denominator = np.where(largest > 0, largest, 1)
    return largest * np.sqrt(np.mean((values / denominator[:, None]) ** 2, axis=1))


def _component_scales(scales: np.ndarray, level_scale, shape_scale, *, fitting: bool):
    n = len(scales)
    level = _rms(scales) if level_scale is None else np.asarray(level_scale, dtype=float)
    if level.ndim == 0:
        level = np.full(n, level.item())
    if level.shape != (n,) or not np.isfinite(level).all() or np.any(level < 0):
        raise ValueError("level_scale must be finite, nonnegative and have shape (N,) or be scalar.")
    shape, _ = _daily(scales if shape_scale is None else shape_scale, "shape_scale")
    if shape.shape == (1, 24) and n > 1:
        shape = np.broadcast_to(shape, (n, 24))
    if shape.shape != scales.shape or np.any(shape < 0):
        raise ValueError("shape_scale must be nonnegative with shape (N, 24) or (24,).")
    shape_rms = _rms(shape)
    if fitting and (np.any(level <= 0) or np.any(shape_rms <= 0)):
        raise ValueError("Source level_scale and daily shape RMS must be strictly positive.")
    return level, shape_rms


class ComponentCalibrator:
    """Jointly resample source daily level/shape pairs using separate scales.

    Fit accepts the same basic cohort fields as SourceCalibrator, plus optional
    ``level_scales`` (N,) and ``shape_scales`` (N,24). Shape scale is deliberately
    collapsed to a scalar daily RMS to preserve zero daily mean. If component
    scales are omitted, both fall back to the daily RMS of ``scales``; callers
    with conditional level uncertainties should pass their actual components.

    Years, when supplied, must be 2023. Without year metadata the caller remains
    responsible for the independent source-2023 calibration boundary. Ages and
    months are validated and audited but do not condition the residual bank.
    """

    def __init__(self, source_regions: Sequence, alpha: float = 0.1):
        if not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("alpha must lie strictly between zero and one.")
        if not len(source_regions):
            raise ValueError("Declare at least one source region.")
        self.source_regions = frozenset(_labels(source_regions, len(source_regions), "source_regions"))
        self.alpha = float(alpha)
        self.fitted = False

    def fit(self, point: np.ndarray, truth: np.ndarray, scales: np.ndarray,
            regions: Sequence, ages: Sequence, *, months: Sequence | None = None,
            years: Sequence | None = None, level_scales: np.ndarray | None = None,
            shape_scales: np.ndarray | None = None) -> ComponentCalibrator:
        point, _ = _daily(point, "point")
        truth, _ = _daily(truth, "truth")
        scales, _ = _daily(scales, "scales", positive=True)
        if point.shape != truth.shape or point.shape != scales.shape:
            raise ValueError("point, truth and scales must have matching shapes.")
        n = len(point)
        regions, ages = _labels(regions, n, "regions"), _labels(ages, n, "ages")
        if not set(regions).issubset(self.source_regions):
            raise ValueError("Calibration labels may come only from declared source regions.")
        if years is not None and np.any(_labels(years, n, "years") != "2023"):
            raise ValueError("Calibration years must contain only the independent source 2023 split.")
        months = _labels(months, n, "months") if months is not None else np.full(n, "all")
        level_scale, shape_scale = _component_scales(scales, level_scales, shape_scales, fitting=True)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            level, shape = decompose_residual_paths(truth - point)
            level_paths, shape_paths = level / level_scale, shape / shape_scale[:, None]
        if not np.isfinite(level_paths).all() or not np.isfinite(shape_paths).all():
            raise ValueError("Standardized component residuals overflowed; check component scales.")
        self.level_paths_, self.shape_paths_ = level_paths.copy(), shape_paths.copy()
        self.regions_, self.ages_, self.months_ = regions.copy(), ages.copy(), months.copy()
        self.row_weights_ = np.empty(n)
        self.domain_indices_ = []
        domains = np.unique(regions)
        for domain in domains:
            indices = np.flatnonzero(regions == domain)
            self.domain_indices_.append(indices)
            self.row_weights_[indices] = 1 / (len(domains) * len(indices))
        self.q_, self.q_by_age_ = _MULTIPLIERS.copy(), {}
        self.audit_ = {
            "n_days": n, "n_hours": 24 * n, "regions": sorted(set(regions)),
            "region_day_counts": {str(domain): int(np.sum(regions == domain)) for domain in domains},
            "ages": sorted(set(ages)), "months": sorted(set(months)),
            "calibration_year": 2023 if years is not None else None, "target_labels_used": False,
            "age_conditioning": False, "sampling": "equal region then equal source day; paired level/shape",
            "shape_normalization": "scalar daily RMS; preserves zero mean",
            "component_scale_fallback": {"level": level_scales is None, "shape": shape_scales is None},
            "multipliers": _MULTIPLIERS.copy(),
            "claim": "empirical source component transport; no unseen-domain coverage guarantee",
        }
        self.fitted = True
        return self

    def _prediction(self, point, scales, mode, ages, level_scale, shape_scale):
        if not self.fitted:
            raise RuntimeError("Fit the calibrator before predicting.")
        if mode not in COMPONENT_MODES:
            raise ValueError(f"Unknown component calibration mode {mode!r}; expected {COMPONENT_MODES}.")
        point, single = _daily(point, "point")
        scales, _ = _daily(scales, "scales", positive=True)
        if point.shape != scales.shape:
            raise ValueError("point and scales must have matching shapes.")
        if ages is not None:
            _labels(ages, len(point), "ages")
        level, shape = _component_scales(scales, level_scale, shape_scale, fitting=False)
        return point, level, shape, single

    def _transport(self, point, level_scale, shape_scale, indices, multiplier):
        with np.errstate(over="ignore", invalid="ignore"):
            residuals = (level_scale * self.level_paths_[indices, None]
                         + shape_scale * self.shape_paths_[indices])
            result = point + multiplier * residuals
        if not np.isfinite(result).all():
            raise ValueError("Transported component scenarios overflowed; check point and scales.")
        return result

    def intervals(self, point: np.ndarray, scales: np.ndarray, mode: str = "component_transport",
                  *, ages: Sequence | None = None, level_scale=None, shape_scale=None):
        """Exact equal-region full-bank marginal quantiles, not conformal bounds.

        Processes one target day at a time, so temporary memory is source-days x
        24 rather than target-days x source-days x 24. No Monte Carlo seed enters
        these deterministic intervals; ensemble intervals must be scored apart.
        """
        point, level, shape, single = self._prediction(point, scales, mode, ages, level_scale, shape_scale)
        lower, upper = np.empty_like(point), np.empty_like(point)
        indices = np.arange(len(self.level_paths_))
        for i in range(len(point)):
            paths = self._transport(point[i], level[i], shape[i], indices, _MULTIPLIERS[mode])
            lower[i], upper[i] = weighted_quantile(paths, self.row_weights_,
                                                 [self.alpha / 2, 1 - self.alpha / 2])
        return (lower[0], upper[0]) if single else (lower, upper)

    def ensemble(self, point: np.ndarray, scales: np.ndarray, mode: str = "component_transport",
                 *, ages: Sequence | None = None, level_scale=None, shape_scale=None,
                 n_samples: int = 128, seed: int | np.random.Generator = 0):
        """Return (N,S,24), or (S,24) for a single day, with uniform sample weights.

        The same sampled source-day index supplies both components. A fixed seed
        uses the same indices across component-scale and multiplier candidates,
        making level/shape isolation comparisons exact for a given batch order.
        """
        point, level, shape, single = self._prediction(point, scales, mode, ages, level_scale, shape_scale)
        if not isinstance(n_samples, (int, np.integer)) or isinstance(n_samples, bool) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer.")
        rng = np.random.default_rng(seed)
        result = np.empty((len(point), n_samples, 24))
        for i in range(len(point)):
            domain_draws = rng.integers(len(self.domain_indices_), size=n_samples)
            indices = np.empty(n_samples, dtype=int)
            for domain, available in enumerate(self.domain_indices_):
                positions = np.flatnonzero(domain_draws == domain)
                indices[positions] = rng.choice(available, size=len(positions))
            result[i] = self._transport(point[i], level[i], shape[i], indices, _MULTIPLIERS[mode])
        return result[0] if single else result
