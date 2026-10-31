"""Source-only residual calibration and coherent daily residual ensembles.

Scores are individual hourly absolute standardized errors, not daily maxima.
The pooled split quantile has the usual finite-sample rank correction, whose
coverage interpretation requires exchangeable calibration/test scores. Daily
dependence, age/month subgroup selection and unseen-domain shift do not satisfy
that assumption automatically. Protected modes are source-domain diagnostics
and conservative heuristics, not coverage guarantees for arbitrary new domains.
Ensemble CRPS and the explicit symmetric intervals are distinct estimands.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .weighted_metrics import weighted_quantile


MODES = ("source_residual", "pooled", "equal_region", "worstgrid_max", "protected_domain80")


def split_conformal_quantile(scores: np.ndarray, alpha: float = 0.1) -> float:
    """Return order statistic ceil((n+1)(1-alpha)); infinity if rank > n."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if not scores.size or not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError("Scores must be nonempty, finite and nonnegative.")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one.")
    rank = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    return float(np.partition(scores, rank - 1)[rank - 1]) if rank <= len(scores) else float("inf")


def _daily(values: np.ndarray, name: str, *, positive: bool = False) -> tuple[np.ndarray, bool]:
    values = np.asarray(values, dtype=float)
    single = values.ndim == 1
    if single:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 24 or not len(values) or not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite with shape (N, 24) or (24,).")
    if positive and np.any(values <= 0):
        raise ValueError(f"{name} must be strictly positive.")
    return values, single


def _labels(values: Sequence, n: int, name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.shape != (n,):
        raise ValueError(f"{name} must have shape (N,).")
    # Reject missing numeric and object labels before grouping.
    if any(v is None or str(v).strip() in ("", "nan", "NaT", "<NA>") for v in values):
        raise ValueError(f"{name} must not contain missing labels.")
    if values.dtype.kind in "fci" and not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite.")
    return values.astype(str)


class SourceCalibrator:
    """Fit residuals on an explicitly declared source-region calibration cohort.

    Numeric ages are treated as categorical strata: callers should pre-bin ages
    consistently. Unknown ages fall back to the complete source cohort. If years
    are supplied they must all equal 2023; without metadata the caller remains
    responsible for providing the independent 2023 calibration split. Months
    should identify calendar blocks (for example ``2023-01``). Protected factors
    first maximize monthly quantiles within each source domain, then take either
    the domain maximum (worstgrid baseline) or the inverse-CDF 80th percentile
    across equal domains. Only protected_domain80 uses age conditioning; its
    age-specific radius is floored at the global radius, including sparse ages.
    """

    def __init__(self, source_regions: Sequence, alpha: float = 0.1):
        if not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("alpha must lie strictly between zero and one.")
        if len(source_regions) == 0:
            raise ValueError("Declare at least one source region.")
        self.source_regions = frozenset(_labels(source_regions, len(source_regions), "source_regions"))
        self.alpha = float(alpha)
        self.fitted = False

    def _factors(self, mask: np.ndarray) -> dict[str, float]:
        paths = self.residual_paths_[mask]
        regions = self.regions_[mask]
        months = self.months_[mask]
        domains = np.unique(regions)
        row_weights = np.empty(len(paths))
        domain_factors = []
        for region in domains:
            domain = regions == region
            row_weights[domain] = 1 / (len(domains) * domain.sum())
            monthly = [split_conformal_quantile(np.abs(paths[domain & (months == month)]), self.alpha)
                       for month in np.unique(months[domain])]
            domain_factors.append(max(monthly))
        # Equal mass per region, then per day, then per hour.
        balanced = float(weighted_quantile(np.abs(paths).reshape(-1),
                                          np.repeat(row_weights / 24, 24), 1 - self.alpha))
        return {
            "source_residual": balanced,
            "pooled": split_conformal_quantile(np.abs(paths), self.alpha),
            "equal_region": balanced,
            "worstgrid_max": float(max(domain_factors)),
            "protected_domain80": float(np.sort(domain_factors)[int(np.ceil(0.8 * len(domains))) - 1]),
        }

    def fit(self, point: np.ndarray, truth: np.ndarray, scales: np.ndarray,
            regions: Sequence, ages: Sequence, *, months: Sequence | None = None,
            years: Sequence | None = None) -> SourceCalibrator:
        point, _ = _daily(point, "point")
        truth, _ = _daily(truth, "truth")
        scales, _ = _daily(scales, "scales", positive=True)
        if point.shape != truth.shape or point.shape != scales.shape:
            raise ValueError("point, truth and scales must have matching shapes.")
        n = len(point)
        regions = _labels(regions, n, "regions")
        ages = _labels(ages, n, "ages")
        if not set(regions).issubset(self.source_regions):
            raise ValueError("Calibration labels may come only from declared source regions.")
        if years is not None and np.any(_labels(years, n, "years") != "2023"):
            raise ValueError("Calibration years must contain only the independent source 2023 split.")
        months = _labels(months, n, "months") if months is not None else np.full(n, "all")
        residual_paths = (truth - point) / scales
        if not np.isfinite(residual_paths).all():
            raise ValueError("Standardized residuals overflowed; check scales.")
        # No partially fitted object remains usable if a new fit fails below.
        self.fitted = False
        self.residual_paths_ = residual_paths.copy()
        self.raw_residual_paths_ = (truth - point).copy()
        self.regions_, self.ages_, self.months_ = regions.copy(), ages.copy(), months.copy()
        self.q_ = self._factors(np.ones(n, dtype=bool))
        self.q_by_age_ = {age: self._factors(ages == age) for age in np.unique(ages)}
        for factors in self.q_by_age_.values():
            factors["protected_domain80"] = max(self.q_["protected_domain80"], factors["protected_domain80"])
        self.row_weights_ = np.empty(n)
        domains = np.unique(regions)
        for domain in domains:
            mask = regions == domain
            self.row_weights_[mask] = 1 / (len(domains) * mask.sum())
        self.raw_lower_, self.raw_upper_ = weighted_quantile(
            self.raw_residual_paths_, self.row_weights_, [self.alpha / 2, 1 - self.alpha / 2])
        self.audit_ = {"n_days": n, "n_hours": 24 * n, "regions": sorted(set(regions)),
                       "ages": sorted(set(ages)), "months": sorted(set(months)),
                       "calibration_year": 2023 if years is not None else None,
                       "target_labels_used": False,
                       "protected_claim": "source-domain protection heuristic; no unseen-domain guarantee"}
        self.fitted = True
        return self

    def _check(self, mode: str) -> str:
        if not self.fitted:
            raise RuntimeError("Fit the calibrator before predicting.")
        mode = "worstgrid_max" if mode == "protected_max" else mode
        if mode not in MODES:
            raise ValueError(f"Unknown calibration mode {mode!r}; expected one of {MODES}.")
        return mode

    def _prediction(self, point: np.ndarray, scales: np.ndarray, ages: Sequence | None):
        point, single = _daily(point, "point")
        scales, _ = _daily(scales, "scales", positive=True)
        if point.shape != scales.shape:
            raise ValueError("point and scales must have matching shapes.")
        age_labels = _labels(ages, len(point), "ages") if ages is not None else np.full(len(point), "")
        return point, scales, age_labels, single

    def intervals(self, point: np.ndarray, scales: np.ndarray, mode: str = "pooled",
                  *, ages: Sequence | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Symmetric q*scale bounds, or empirical raw residual bounds for baseline."""
        mode = self._check(mode)
        point, scales, age_labels, single = self._prediction(point, scales, ages)
        if mode == "source_residual":
            lower, upper = point + self.raw_lower_, point + self.raw_upper_
        else:
            q = np.asarray([(self.q_by_age_.get(age, self.q_) if mode == "protected_domain80" else self.q_)[mode]
                            for age in age_labels])
            lower, upper = point - q[:, None] * scales, point + q[:, None] * scales
        return (lower[0], upper[0]) if single else (lower, upper)

    def ensemble(self, point: np.ndarray, scales: np.ndarray, mode: str = "source_residual",
                 *, ages: Sequence | None = None, n_samples: int = 256,
                 seed: int | np.random.Generator = 0) -> np.ndarray:
        """Resample signed whole-day paths with equal probability per source domain.

        The source_residual baseline resamples physical truth-minus-point errors,
        ignoring target scales. Calibrated modes multiply standardized paths by
        q / the applicable equal-region
        absolute-error quantile. This approximately aligns absolute tail radii,
        retaining signs and within-day dependence; it does not force the 5th and
        95th ensemble quantiles to equal the explicit symmetric interval. Zero
        residual banks stay degenerate. The returned array is (N, S, 24), or
        (S, 24) for a single point vector; samples have uniform scoring weights.
        """
        mode = self._check(mode)
        if not isinstance(n_samples, (int, np.integer)) or isinstance(n_samples, bool) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer.")
        point, scales, age_labels, single = self._prediction(point, scales, ages)
        rng = np.random.default_rng(seed)
        result = np.empty((len(point), n_samples, 24))
        for i, age in enumerate(age_labels):
            conditional = mode == "protected_domain80" and age in self.q_by_age_
            mask = self.ages_ == age if conditional else np.ones(len(self.ages_), dtype=bool)
            available = np.flatnonzero(mask)
            regions = self.regions_[mask]
            domains = np.unique(regions)
            domain_draws = rng.choice(domains, size=n_samples)
            selected = np.empty(n_samples, dtype=int)
            for domain in domains:
                positions = np.flatnonzero(domain_draws == domain)
                selected[positions] = rng.choice(available[regions == domain], size=len(positions))
            factors = self.q_by_age_[age] if conditional else self.q_
            denominator = factors["equal_region"]
            multiplier = 1.0 if mode == "source_residual" or denominator == 0 else factors[mode] / denominator
            if mode != "source_residual" and not np.isfinite(factors[mode]):
                raise ValueError("Too few calibration scores for finite ensemble scaling at this alpha.")
            if mode == "source_residual":
                result[i] = point[i] + self.raw_residual_paths_[selected]
            else:
                result[i] = point[i] + self.residual_paths_[selected] * multiplier * scales[i]
        if not np.isfinite(result).all():
            raise ValueError("Ensemble scenarios overflowed; check point and scales.")
        return result[0] if single else result


def decompose_residual_paths(residual_paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split any (..., 24) residual array into level (...,) and zero-mean shape."""
    residual_paths = np.asarray(residual_paths, dtype=float)
    if residual_paths.ndim < 1 or residual_paths.shape[-1] != 24 or not residual_paths.size or not np.isfinite(residual_paths).all():
        raise ValueError("Residual paths must be finite with trailing dimension 24.")
    level = residual_paths.mean(axis=-1)
    return level, residual_paths - level[..., None]


def combine_component_scenarios(point: np.ndarray, level_paths: np.ndarray,
                                shape_paths: np.ndarray, *, level_scale: np.ndarray | float = 1.0,
                                shape_scale: np.ndarray | float = 1.0) -> np.ndarray:
    """Exactly combine separately scaled level and zero-mean shape scenarios.

    ``shape_paths`` is (..., 24), ``level_paths`` and each scale broadcast to its
    leading dimensions. Shape scales are constant within a day, preserving the
    zero-mean constraint. For batched scenarios, pass point as (N, 1, 24), and
    per-day scales as (N, 1). Sample level/shape pairs jointly to retain coupling.
    """
    point, level_paths, shape_paths = (np.asarray(x, dtype=float) for x in (point, level_paths, shape_paths))
    level_scale, shape_scale = np.asarray(level_scale, dtype=float), np.asarray(shape_scale, dtype=float)
    arrays = (point, level_paths, shape_paths, level_scale, shape_scale)
    if any(not np.isfinite(x).all() for x in arrays) or np.any(level_scale < 0) or np.any(shape_scale < 0):
        raise ValueError("Components must be finite and scales nonnegative.")
    if point.ndim < 1 or point.shape[-1] != 24 or shape_paths.ndim < 1 or shape_paths.shape[-1] != 24 or not shape_paths.size:
        raise ValueError("point and shape_paths must have trailing dimension 24.")
    if not np.allclose(shape_paths.mean(axis=-1), 0, atol=1e-10, rtol=0):
        raise ValueError("Shape residual paths must have zero daily mean.")
    try:
        level = np.broadcast_to(level_paths * level_scale, shape_paths.shape[:-1])
        scale = np.broadcast_to(shape_scale, shape_paths.shape[:-1])
        result = point + level[..., None] + shape_paths * scale[..., None]
    except ValueError as exc:
        raise ValueError("Component, point and scale shapes do not broadcast.") from exc
    if not np.isfinite(result).all():
        raise ValueError("Combined scenarios overflowed.")
    return result


Calibrator = SourceCalibrator
