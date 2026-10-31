"""Source-only weather shape transport and two-channel residual references.

These are standard regression/resampling mechanisms, not coverage guarantees.
Transport preserves each donor's daily fuel energy, not target monthly energy.
Callers must supply as-of features and geographically crossfit donor predictions
where required; in-sample fitted predictions are not crossfit automatically.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.ensemble import ExtraTreesRegressor

SOURCES = frozenset({"BPAT", "CISO", "ERCO", "FPL", "ISNE", "MISO", "NYIS"})
SEED = 17023
FLOOR = 5.0


def _profiles(value, name="q"):
    value = np.asarray(value, float)
    if (value.ndim < 2 or value.shape[-2:] != (24, 8)
            or not np.isfinite(value).all() or np.any(value < 0)):
        raise ValueError(f"{name} must be finite nonnegative (...,24,8)")
    return value


def split_profiles(q: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Return daily fuel amplitude and mean-one shape; zero fuel has flat shape."""
    q = _profiles(q)
    amplitude = q.mean(axis=-2)
    shape = np.divide(q, amplitude[..., None, :], out=np.ones_like(q),
                      where=amplitude[..., None, :] > 0)
    return amplitude, shape


def project_shapes(values: ArrayLike) -> np.ndarray:
    """Euclidean projection of each 24-hour fuel vector onto sum=24 simplex."""
    values = np.asarray(values, float)
    if values.ndim < 2 or values.shape[-2:] != (24, 8) or not np.isfinite(values).all():
        raise ValueError("shape values must be finite (...,24,8)")
    rows = np.moveaxis(values, -2, -1).reshape(-1, 24)
    ordered = np.sort(rows, axis=1)[:, ::-1]
    thresholds = (np.cumsum(ordered, axis=1) - 24.0) / np.arange(1, 25)
    rho = (ordered > thresholds).sum(axis=1) - 1
    threshold = thresholds[np.arange(len(rows)), rho]
    result = np.maximum(rows - threshold[:, None], 0)
    result = result.reshape((*values.shape[:-2], 8, 24))
    return np.moveaxis(result, -1, -2)


def transport(q: ArrayLike, query_shape: ArrayLike, donor_prediction: ArrayLike,
              support: ArrayLike, *, keep_shape_residual: bool = True) -> np.ndarray:
    """Replace predicted shape, optionally carrying the donor's shape residual.

    q is (samples,24,8) or (...,24,8). Predictions broadcast to q's shape;
    support broadcasts to (...,8). Unsupported fuels must have the bank's
    flat q=1 convention and remain flat. A zero daily amplitude stays zero.
    """
    q = _profiles(q)
    amplitude, shape = split_profiles(q)
    supported = np.asarray(support)
    if not np.isin(supported, [False, True]).all():
        raise ValueError("support must contain boolean flags")
    try:
        supported = np.broadcast_to(supported.astype(bool), amplitude.shape)
        query = np.broadcast_to(_profiles(query_shape, "query_shape"), q.shape)
        donor = np.broadcast_to(_profiles(donor_prediction, "donor_prediction"), q.shape)
    except ValueError as exc:
        raise ValueError("shape predictions/support cannot broadcast to donor profiles") from exc
    unsupported = np.broadcast_to(~supported[..., None, :], q.shape)
    if np.any(unsupported & ~np.isclose(q, 1., atol=1e-10, rtol=1e-10)):
        raise ValueError("unsupported donor fuels must use flat q=1")
    corrected = query + (shape - donor if keep_shape_residual else 0.)
    corrected = project_shapes(corrected)
    result = amplitude[..., None, :] * corrected
    return np.where(unsupported, 1., result)


def _matrix(x):
    x = np.asarray(x, float)
    if x.ndim != 2 or not len(x) or not x.shape[1] or np.isinf(x).any():
        raise ValueError("features must be a nonempty matrix, with NaN for missing")
    return x


def _region_weights(regions):
    counts = Counter(regions)
    return np.array([len(regions) / (len(counts) * counts[r]) for r in regions])


def _training(regions, n, exclude_region, days, *, calibration=False):
    regions = np.asarray(regions)
    if regions.shape != (n,) or not set(regions) <= SOURCES or len(set(regions)) < 2:
        raise ValueError("fit requires at least two known source regions")
    if exclude_region is not None:
        if exclude_region not in SOURCES:
            raise ValueError("excluded region must be a known source")
        if exclude_region in regions:
            raise ValueError("excluded evaluation region is present in proper fit")
    if days is not None:
        dates = pd.DatetimeIndex(pd.to_datetime(days, utc=True))
        start, end, split = (("2024-07-01", "2025-01-01", "2024H2") if calibration
                             else ("2024-01-01", "2024-07-01", "2024H1"))
        if (len(dates) != n or dates.hasnans
                or np.any(dates < pd.Timestamp(start, tz="UTC"))
                or np.any(dates >= pd.Timestamp(end, tz="UTC"))):
            raise ValueError(f"fit dates must lie in {split}; future labels forbidden")
    return regions


def _trees():
    return ExtraTreesRegressor(n_estimators=64, min_samples_leaf=20,
                               max_features=1.0, random_state=SEED, n_jobs=1)


class _ProperDesign:
    def _fit_design(self, x):
        self.medians_ = np.array([np.median(c[~np.isnan(c)]) if (~np.isnan(c)).any() else 0.
                                  for c in x.T])
        return self._design(x)

    def _design(self, x):
        x = _matrix(x)
        if x.shape[1] != len(self.medians_):
            raise ValueError("feature width differs from proper fit")
        missing = np.isnan(x)
        return np.column_stack((np.where(missing, self.medians_, x), missing.astype(float)))

    def _audit(self, regions, excluded, days):
        self.training_regions_ = tuple(sorted(set(regions)))
        self.audit_ = {"training_regions": list(self.training_regions_),
                       "proper_regions": dict(sorted(Counter(regions).items())),
                       "excluded_evaluation_region": excluded,
                       "proper_fit_only_imputation": True,
                       "equal_region_fit_weights": True,
                       "dates_validated": days is not None,
                       "seed": SEED}


class ShapeModel(_ProperDesign):
    """Per-fuel ExtraTrees shape prediction using active supported source days."""
    def fit(self, x: ArrayLike, q: ArrayLike, regions: Sequence[str] | np.ndarray,
            supports: ArrayLike, *, exclude_region: str | None = None,
            days: pd.DatetimeIndex | None = None) -> ShapeModel:
        self.fitted_ = False
        x, q = _matrix(x), _profiles(q)
        if q.shape != (len(x), 24, 8):
            raise ValueError("q must have one (24,8) profile per proper row")
        regions = _training(regions, len(x), exclude_region, days)
        supports = np.asarray(supports)
        if supports.shape != (len(x), 8) or not np.isin(supports, [False, True]).all():
            raise ValueError("supports must be boolean (rows,8)")
        amplitude, shapes = split_profiles(q)
        design = self._fit_design(x)
        self.models_, counts = [], []
        for fuel in range(8):
            active = supports[:, fuel].astype(bool) & (amplitude[:, fuel] > 0)
            counts.append(int(active.sum()))
            model = None
            if active.any():
                model = _trees().fit(design[active], shapes[active, :, fuel],
                                     sample_weight=_region_weights(regions[active]))
            self.models_.append(model)
        self._audit(regions, exclude_region, days)
        self.audit_["supported_active_days_per_fuel"] = counts
        self.audit_["donor_predictions_crossfit_automatically"] = False
        self.fitted_ = True
        return self

    def predict(self, x: ArrayLike) -> np.ndarray:
        if not getattr(self, "fitted_", False):
            raise RuntimeError("fit ShapeModel before prediction")
        design = self._design(x)
        shapes = np.ones((len(design), 24, 8))
        for fuel, model in enumerate(self.models_):
            if model is not None:
                shapes[:, :, fuel] = model.predict(design)
        return project_shapes(shapes)


class ScaleModel(_ProperDesign):
    """Conditional RMS reference, fitted to crossfit 2024H2 errors, bounded 5..500.

    Feature imputation uses only this scale-fit cohort. Caller must exclude
    inner calibration regions when obtaining out-of-fold calibration scales.
    """
    def fit(self, x: ArrayLike, errors: ArrayLike, regions: Sequence[str] | np.ndarray,
            *, exclude_region: str | None = None,
            days: pd.DatetimeIndex | None = None) -> ScaleModel:
        self.fitted_ = False
        x, errors = _matrix(x), np.asarray(errors, float)
        if errors.shape != (len(x), 24) or not np.isfinite(errors).all():
            raise ValueError("errors must be finite (rows,24)")
        regions = _training(regions, len(x), exclude_region, days, calibration=True)
        level = errors.mean(axis=1)
        shape = errors - level[:, None]
        target = np.log(np.column_stack((level ** 2, np.mean(shape ** 2, axis=1))) + FLOOR ** 2)
        self.model_ = _trees().fit(self._fit_design(x), target,
                                    sample_weight=_region_weights(regions))
        self._audit(regions, exclude_region, days)
        self.audit_["fit_cohort"] = "source crossfit calibration 2024H2"
        self.audit_.update({"scale_floor": FLOOR, "scale_ceiling": 500.,
                            "channels": ["level", "shape"]})
        self.fitted_ = True
        return self

    def predict(self, x: ArrayLike) -> np.ndarray:
        if not getattr(self, "fitted_", False):
            raise RuntimeError("fit ScaleModel before prediction")
        log_variance = self.model_.predict(self._design(x))
        return np.exp(.5 * np.clip(log_variance, 2*np.log(FLOOR), 2*np.log(500.)))


def residual_paths(center: ArrayLike, residuals: ArrayLike, indices: ArrayLike, *,
                   scales: ArrayLike | None = None,
                   calibration_scales: ArrayLike | None = None) -> np.ndarray:
    """Add paired daily level/shape errors; caller supplies frozen draw indices.

    center is (N,24), residuals (C,24), indices (N,S). Optional scale matrices
    are (N,2) and (C,2); both must be supplied. No clipping is performed here.
    """
    center, residuals = np.asarray(center, float), np.asarray(residuals, float)
    indices = np.asarray(indices)
    if (center.ndim != 2 or center.shape[1] != 24 or residuals.ndim != 2
            or residuals.shape[1] != 24 or not len(residuals)
            or not np.isfinite(center).all() or not np.isfinite(residuals).all()
            or indices.ndim != 2 or indices.shape[0] != len(center)
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0) or np.any(indices >= len(residuals))):
        raise ValueError("invalid center, residuals or paired residual indices")
    level = residuals.mean(axis=1)
    shape = residuals - level[:, None]
    level_draws, shape_draws = level[indices], shape[indices]
    if (scales is None) != (calibration_scales is None):
        raise ValueError("query and calibration scales must both be supplied")
    if scales is not None:
        scales, calibration_scales = np.asarray(scales, float), np.asarray(calibration_scales, float)
        if (scales.shape != (len(center), 2) or calibration_scales.shape != (len(residuals), 2)
                or not np.isfinite(scales).all() or not np.isfinite(calibration_scales).all()
                or np.any(scales <= 0) or np.any(calibration_scales <= 0)):
            raise ValueError("scales must be finite positive (rows,2)")
        ratio = scales[:, None, :] / calibration_scales[indices]
        level_draws = level_draws * ratio[..., 0]
        shape_draws = shape_draws * ratio[..., 1, None]
    return center[:, None, :] + level_draws[..., None] + shape_draws
