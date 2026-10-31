"""Release-matched coarse hindcasts and source-only conditional residual transport.

Hindcast errors measure unpredictable coarse innovations separately from raw
seasonal variation. Conditional variance regression, ridge bias correction and
empirical residual transport are existing statistical tools; their combination
here is an experimental mechanism, not a new regression algorithm or a coverage
guarantee under target-domain shift. Target hourly observations are never inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from .calibration_v4 import _daily, _labels, decompose_residual_paths
from .dynamic_v2 import EF
from .transfer_v4 import FUEL_COLS
from .weighted_metrics import weighted_quantile


FEATURE_NAMES = ("latest_level", "monthly_sd", "annual_rms", "age_days", "history_months") + tuple(
    f"{method}_hindcast_{stat}" for method in ("latest", "seasonal", "harmonic")
    for stat in ("bias", "rms", "sd")
) + ("hindcast_months", "seasonal_hindcast_valid_months", "harmonic_hindcast_valid_months")
MODES = ("global", "loglinear", "extratrees", "structured", "structured_nohindcast")
_FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def _released(history, origin):
    return (history.loc[(history.available_at <= origin) & (history.period_end <= origin)]
            .sort_values(["period_end", "available_at"], kind="stable")
            .drop_duplicates("period_start", keep="last"))


def _values(rows):
    shares = rows[FUEL_COLS].to_numpy(dtype=float)
    if (not np.isfinite(shares).all() or np.any(shares < 0)
            or not np.allclose(shares.sum(axis=1), 1, atol=1e-5)):
        raise ValueError("Released shares require finite simplex rows.")
    return shares @ EF


def _harmonic_design(starts):
    month = np.asarray([stamp.year * 12 + stamp.month for stamp in starts], dtype=float)
    phase = 2 * np.pi * month / 12
    return np.column_stack([np.ones(len(month)), np.sin(phase), np.cos(phase),
                            np.sin(2 * phase), np.cos(2 * phase)])


def _coarse_predictions(rows, target):
    values = _values(rows)
    seasonal = values[-1]
    matches = np.flatnonzero(rows.period_start.dt.month.to_numpy() == target.month)
    if len(matches):
        seasonal = values[matches[-1]]
    design = _harmonic_design(rows.period_start)
    # Fixed ridge and a latest-level fallback avoid fitting an annual cycle from
    # less than a year's support. Neither hyperparameter is target-selected.
    harmonic = values[-1]
    if len(rows) >= 12:
        penalty = np.diag([0., 1., 1., 1., 1.])
        coef = np.linalg.solve(design.T @ design + penalty, design.T @ values)
        harmonic = float((_harmonic_design([target]) @ coef)[0])
    return np.asarray([values[-1], seasonal, harmonic]), np.asarray([True, bool(len(matches)), len(rows) >= 12])


def information_features(origins, region, monthly, extra_delay=0):
    """Return (N,F) lawful features and their fixed ordered names.

    Each currently released month j supplies a coarse label. Its predictor is
    reconstructed at j's exact mid-period using only releases available then,
    including the same additional publication delay as the live forecast.
    Revisions are selected as of each respective time. Last 24 eligible label
    months are used; each pseudo-forecast uses at most 36 released months.
    Seasonal/harmonic statistics exclude their latest-level fallbacks; explicit
    valid-month counts distinguish missing support from genuinely zero error.
    A batch-local cache shares work between origins with the same release set.
    Timestamp comparisons use pandas UTC values, never raw resolution-dependent
    integer representations. Values in unreleased rows are not inspected.
    """
    if not np.isfinite(extra_delay) or extra_delay < 0:
        raise ValueError("extra_delay must be finite and nonnegative.")
    origins = pd.DatetimeIndex(pd.to_datetime(origins, utc=True))
    if origins.hasnans:
        raise ValueError("origins must be nonmissing.")
    history = monthly.loc[monthly.region.eq(region)].copy().reset_index(drop=True)
    for column in ("period_start", "period_end", "available_at"):
        history[column] = pd.to_datetime(history[column], utc=True)
        if history[column].isna().any():
            raise ValueError(f"{column} must be nonmissing.")
    history["available_at"] += pd.Timedelta(days=float(extra_delay))
    result = np.zeros((len(origins), len(FEATURE_NAMES)))
    cache, hindcasts = {}, {}
    for i, origin in enumerate(origins):
        rows = _released(history, origin)
        key = tuple(rows.index)
        if key not in cache:
            vector = np.zeros(len(FEATURE_NAMES))
            vector[_FEATURE_INDEX["age_days"]] = 365.
            if len(rows):
                recent = rows.tail(36)
                values = _values(recent)
                vector[:3] = values[-1], np.std(values[-12:]), 0.
                lookup = {(stamp.year, stamp.month): value
                          for stamp, value in zip(recent.period_start, values)}
                annual = [value - lookup[(stamp.year - 1, stamp.month)]
                          for stamp, value in zip(recent.period_start, values)
                          if (stamp.year - 1, stamp.month) in lookup]
                if annual:
                    vector[2] = np.sqrt(np.mean(np.square(annual)))
                vector[4] = len(rows)
                errors = []
                for label in rows.tail(24).itertuples():
                    # Cache includes label version because an as-of revision may
                    # change its error; the pseudo-prefix itself is always lawful.
                    if label.Index not in hindcasts:
                        pseudo = label.period_start + (label.period_end - label.period_start) / 2
                        prefix = _released(history, pseudo).tail(36)
                        if prefix.empty:
                            hindcasts[label.Index] = None
                        else:
                            prediction, valid = _coarse_predictions(prefix, label.period_start)
                            error = _values(history.loc[[label.Index]])[0] - prediction
                            hindcasts[label.Index] = np.where(valid, error, np.nan)
                    if hindcasts[label.Index] is not None:
                        errors.append(hindcasts[label.Index])
                if errors:
                    errors = np.asarray(errors)
                    for method in range(3):
                        e = errors[:, method]
                        e = e[np.isfinite(e)]
                        if len(e):
                            vector[5 + method * 3:8 + method * 3] = (
                                np.mean(e), np.sqrt(np.mean(e**2)), np.std(e))
                        if method > 0:
                            vector[14 + method] = len(e)
                    vector[14] = len(errors)
            cache[key] = vector
        result[i] = cache[key]
        if len(rows):
            result[i, 3] = (origin - rows.iloc[-1].period_end).total_seconds() / 86400
    return result, FEATURE_NAMES


def _region_weights(regions):
    domains = np.unique(regions)
    return np.asarray([1 / (len(domains) * np.sum(regions == region)) for region in regions])


def information_feature_audit(features):
    """Truth-free ranges/support report usable on proper/bank/selection/target.

    Callers supply an already lawful feature matrix; this never inspects labels
    or adapts any fitted transform. Counts are per forecast row, so repeated
    origins under different release delays remain distinct information sets.
    """
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not len(x) or not np.isfinite(x).all():
        raise ValueError(f"features must be nonempty finite (N,{len(FEATURE_NAMES)}).")
    support_names = ("history_months", "hindcast_months", "seasonal_hindcast_valid_months",
                     "harmonic_hindcast_valid_months")
    return {"rows": len(x), "ranges": {
                name: {"min": float(x[:, i].min()), "max": float(x[:, i].max()),
                       "mean": float(x[:, i].mean())} for i, name in enumerate(FEATURE_NAMES)},
            "support": {name: {"zero_rows": int(np.sum(x[:, _FEATURE_INDEX[name]] == 0)),
                "positive_fraction": float(np.mean(x[:, _FEATURE_INDEX[name]] > 0))}
                for name in support_names}}


def _component_inputs(n, level, shape):
    level = np.ones(n) if level is None else np.asarray(level, dtype=float)
    if level.ndim == 0:
        level = np.full(n, level.item())
    shape = np.ones(n) if shape is None else np.asarray(shape, dtype=float)
    if shape.ndim == 0:
        shape = np.full(n, shape.item())
    if shape.ndim == 1 and shape.shape == (24,) and n != 24:
        shape = np.broadcast_to(shape, (n, 24))
    if shape.ndim == 2 and shape.shape == (n, 24):
        if not np.isfinite(shape).all() or np.any(shape < 0):
            raise ValueError("shape_scales must be finite and nonnegative.")
        shape = np.sqrt(np.mean(shape**2, axis=1))
    if (level.shape != (n,) or shape.shape != (n,) or not np.isfinite(level).all()
            or not np.isfinite(shape).all() or np.any(level < 0) or np.any(shape < 0)):
        raise ValueError("Component scales must be finite nonnegative (N,) or shape (N,24).")
    return level, shape


class ConditionalTransport:
    """Separate source-proper fit from an independent paired residual bank.

    ``global`` uses constant component scales and zero location correction.
    ``loglinear`` uses ridge log-absolute component errors on all features.
    ``extratrees`` uses fixed 100-tree absolute-error regressors on all features.
    ``structured`` predicts signed level bias from signed coarse errors, then
    fits two nonnegative ridge variance regressions with component-specific base
    variances and centered coarse innovation variances. ``structured_nohindcast``
    removes every hindcast variable, retaining raw monthly/annual variation.
    Component scales are always bounded to [1, 1e6] in emission-intensity units;
    bias is bounded to [-1e6, 1e6]. Shape uses a scalar daily RMS so transported
    shapes retain exactly zero daily mean, rather than hourwise rescaling.
    Jan-Jun of fit_year (default 2024) fits all coefficients; Jul-Dec of that
    year only populates the bank. Starting in 2024 gives the coarse hindcasts
    a longer lawful prefix than the cold-start 2023 historical comparator.
    Each declared source must occur in both halves. No target labels are accepted.
    """

    def __init__(self, source_regions, mode="structured", alpha=.1, fit_year=2024):
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}; expected {MODES}.")
        if not len(source_regions):
            raise ValueError("Declare at least one source region.")
        if not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("alpha must lie strictly between zero and one.")
        if not isinstance(fit_year, (int, np.integer)) or isinstance(fit_year, bool) or fit_year < 2000:
            raise ValueError("fit_year must be an integer calendar year >= 2000.")
        self.source_regions = frozenset(_labels(source_regions, len(source_regions), "source_regions"))
        self.mode, self.alpha, self.fit_year, self.fitted = mode, float(alpha), int(fit_year), False

    def _features(self, features, n):
        x = np.asarray(features, dtype=float)
        if x.shape != (n, len(FEATURE_NAMES)) or not np.isfinite(x).all():
            raise ValueError(f"features must be finite (N,{len(FEATURE_NAMES)}) in FEATURE_NAMES order.")
        return x

    def _bias_design(self, x):
        if self.mode == "structured":
            columns = ["latest_hindcast_bias", "seasonal_hindcast_bias", "harmonic_hindcast_bias"]
            values = x[:, [_FEATURE_INDEX[name] for name in columns]]
        elif self.mode == "structured_nohindcast":
            values = x[:, :5]
        else:
            values = x
        return values

    def _variance_design(self, x, component_scale):
        # Predictable seasonal amplitude is deliberately absent from the main
        # mechanism: only centered hindcast errors represent coarse innovation.
        names = (["seasonal_hindcast_sd", "harmonic_hindcast_sd"]
                 if self.mode == "structured" else ["monthly_sd", "annual_rms"])
        innovation = np.maximum(x[:, [_FEATURE_INDEX[name] for name in names]], 0)
        age = np.maximum(x[:, _FEATURE_INDEX["age_days"]], 0) / 90
        return np.column_stack([np.ones(len(x)), component_scale**2,
                                innovation**2, innovation**2 * age[:, None]])

    @staticmethod
    def _ridge(design, response, weights, penalty=1.):
        root = np.sqrt(weights * len(weights))
        regularizer = np.eye(design.shape[1]) * penalty
        regularizer[0, 0] = 0
        return np.linalg.solve((design * root[:, None]).T @ (design * root[:, None]) + regularizer,
                               design.T @ (response * root**2))

    def _fit_scale(self, x, component_scale, residual, weights):
        magnitude = max(float(np.sqrt(np.sum(weights * residual**2))), 1.)
        if self.mode == "global":
            return (magnitude,)
        if self.mode in ("loglinear", "extratrees"):
            # Generic comparators receive the same point-model component scale
            # as structured variance regression. Learn this transform on proper
            # rows only, separately for level and shape; bank rows cannot adapt it.
            raw = np.column_stack([x, np.log1p(component_scale)])
            center = np.sum(weights[:, None] * raw, axis=0)
            norm = np.maximum(np.sqrt(np.sum(weights[:, None] * (raw - center)**2, axis=0)), 1.)
            standardized = (raw - center) / norm
        if self.mode == "extratrees":
            model = self._tree()
            model.fit(standardized, np.abs(residual),
                      sample_weight=weights * len(weights))
            return center, norm, model
        if self.mode == "loglinear":
            design = np.column_stack([np.ones(len(x)), standardized])
            coef = self._ridge(design, np.log(np.maximum(np.abs(residual), .1)), weights, penalty=10.)
            return center, norm, coef
        design = self._variance_design(x, component_scale)
        norm = np.maximum(np.sqrt(np.sum(weights[:, None] * design**2, axis=0)), 1.)
        design = design / norm
        root = np.sqrt(weights * len(weights))
        # Existing nonnegative least squares with fixed ridge regularization.
        # Normalizing the response makes the fixed penalty unit-independent.
        fit = lsq_linear(np.vstack([design * root[:, None], np.eye(design.shape[1])]),
                         np.r_[residual**2 / magnitude**2 * root, np.zeros(design.shape[1])],
                         bounds=(0., np.inf))
        if not fit.success:
            raise RuntimeError(f"Nonnegative variance regression failed: {fit.message}")
        return norm, fit.x, magnitude

    def _scale(self, model, x, component_scale):
        if self.mode == "global":
            return np.full(len(x), np.clip(model[0], 1., 1e6))
        if self.mode in ("loglinear", "extratrees"):
            center, norm, estimator = model
            standardized = (np.column_stack([x, np.log1p(component_scale)]) - center) / norm
        if self.mode == "extratrees":
            return np.clip(estimator.predict(standardized), 1., 1e6)
        if self.mode == "loglinear":
            design = np.column_stack([np.ones(len(x)), standardized])
            return np.exp(np.clip(design @ estimator, 0., np.log(1e6)))
        norm, coef, magnitude = model
        variance = self._variance_design(x, component_scale) / norm @ coef
        return np.clip(np.sqrt(np.maximum(variance, 0)) * magnitude, 1., 1e6)

    @staticmethod
    def _tree():
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(n_estimators=100, min_samples_leaf=30,
                                   random_state=17023, n_jobs=1)

    def _conditional(self, x, level, shape):
        if self.mode == "global":
            bias = np.zeros(len(x))
        elif self.mode == "extratrees":
            bias = self.bias_tree_.predict((x - self.feature_center_) / self.feature_norm_)
        else:
            design = np.column_stack([np.ones(len(x)),
                (self._bias_design(x) - self.bias_center_) / self.bias_norm_])
            bias = design @ self.bias_coef_
        bias = np.clip(bias, -1e6, 1e6)
        level = self._scale(self.level_model_, x, level)
        shape = self._scale(self.shape_model_, x, shape)
        if not all(np.isfinite(value).all() for value in (bias, level, shape)):
            raise ValueError("Conditional components overflowed; check feature magnitudes.")
        return bias, level, shape

    def fit(self, point, truth, features, regions, *, days, level_scales=None, shape_scales=None):
        self.fitted = False
        point, _ = _daily(point, "point")
        truth, _ = _daily(truth, "truth")
        if truth.shape != point.shape:
            raise ValueError("point and truth must have matching shapes.")
        n = len(point)
        x = self._features(features, n)
        regions = _labels(regions, n, "regions")
        if not set(regions).issubset(self.source_regions):
            raise ValueError("Only declared source regions may enter fitting.")
        dates = pd.DatetimeIndex(pd.to_datetime(days, utc=True))
        if len(dates) != n or dates.hasnans or np.any(dates.year != self.fit_year):
            raise ValueError(f"days must identify only the independent source {self.fit_year} split.")
        proper = dates.month <= 6
        bank = ~proper
        if set(regions[proper]) != self.source_regions or set(regions[bank]) != self.source_regions:
            raise ValueError("Every source region must occur in both Jan-Jun proper-fit and Jul-Dec bank.")
        level_base, shape_base = _component_inputs(n, level_scales, shape_scales)
        level_error, shape_error = decompose_residual_paths(truth - point)
        if not np.isfinite(level_error).all() or not np.isfinite(shape_error).all():
            raise ValueError("Source residual components overflowed.")
        weights = _region_weights(regions[proper])
        self.feature_center_ = np.sum(weights[:, None] * x[proper], axis=0)
        self.feature_norm_ = np.maximum(np.sqrt(np.sum(weights[:, None] *
            (x[proper] - self.feature_center_)**2, axis=0)), 1.)
        if self.mode == "extratrees":
            self.bias_tree_ = self._tree()
            design = (x[proper] - self.feature_center_) / self.feature_norm_
            self.bias_tree_.fit(design, level_error[proper], sample_weight=weights * len(weights))
            proper_bias = self.bias_tree_.predict(design)
        elif self.mode != "global":
            bx = self._bias_design(x[proper])
            self.bias_center_ = np.sum(weights[:, None] * bx, axis=0)
            self.bias_norm_ = np.maximum(np.sqrt(np.sum(weights[:, None] *
                (bx - self.bias_center_)**2, axis=0)), 1.)
            design = np.column_stack([np.ones(proper.sum()), (bx - self.bias_center_) / self.bias_norm_])
            self.bias_coef_ = self._ridge(design, level_error[proper], weights, penalty=10.)
            proper_bias = design @ self.bias_coef_
        else:
            proper_bias = 0.
        self.level_model_ = self._fit_scale(x[proper], level_base[proper],
                                          level_error[proper] - proper_bias, weights)
        self.shape_model_ = self._fit_scale(x[proper], shape_base[proper],
                                          np.sqrt(np.mean(shape_error[proper]**2, axis=1)), weights)
        bias, level, shape = self._conditional(x[bank], level_base[bank], shape_base[bank])
        self.level_paths_ = (level_error[bank] - bias) / level
        self.shape_paths_ = shape_error[bank] / shape[:, None]
        if not np.isfinite(self.level_paths_).all() or not np.isfinite(self.shape_paths_).all():
            raise ValueError("Source standardized residual components overflowed.")
        self.regions_, self.days_ = regions[bank].copy(), dates[bank].copy()
        self.row_weights_ = _region_weights(self.regions_)
        self.domain_indices_ = [np.flatnonzero(self.regions_ == r) for r in sorted(self.source_regions)]
        self.audit_ = {"mode": self.mode, "proper_fit_days": int(proper.sum()), "bank_days": int(bank.sum()),
            "fit_year": self.fit_year,
            "feature_audit": {"proper": information_feature_audit(x[proper]),
                              "bank": information_feature_audit(x[bank])},
            "proper_fit_start": str(dates[proper].min()), "proper_fit_end": str(dates[proper].max()),
            "bank_start": str(dates[bank].min()), "bank_end": str(dates[bank].max()),
            "proper_fit_region_counts": {r: int(np.sum(regions[proper] == r)) for r in sorted(self.source_regions)},
            "bank_region_counts": {r: int(np.sum(regions[bank] == r)) for r in sorted(self.source_regions)},
            "source_regions": sorted(self.source_regions), "target_labels_used": False,
            "scale_bounds": [1., 1e6], "bias_bounds": [-1e6, 1e6],
            "generic_scale_features": FEATURE_NAMES + ("log1p_component_scale",),
            "parameters": {"bias_ridge": 10., "loglinear_ridge": 10., "variance_ridge": 1.,
                "tree_n_estimators": 100, "tree_min_samples_leaf": 30, "tree_random_state": 17023},
            "features": FEATURE_NAMES, "sampling": "equal region, then equal bank row; paired level/shape",
            "claim": "empirical conditional source transport; no unseen-domain coverage guarantee"}
        self.fitted = True
        return self

    def predict(self, point, features, *, level_scales=None, shape_scales=None,
                n_samples=128, seed=17023, multiplier=1.):
        if not self.fitted:
            raise RuntimeError("Fit ConditionalTransport before prediction.")
        point, _ = _daily(point, "point")
        x = self._features(features, len(point))
        if not isinstance(n_samples, (int, np.integer)) or isinstance(n_samples, bool) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer.")
        if not np.isfinite(multiplier) or multiplier < 0:
            raise ValueError("multiplier must be finite and nonnegative.")
        level_base, shape_base = _component_inputs(len(point), level_scales, shape_scales)
        bias, level, shape = self._conditional(x, level_base, shape_base)
        corrected = point + bias[:, None]
        ensemble = np.empty((len(point), n_samples, 24))
        lower, upper = np.empty_like(point), np.empty_like(point)
        rng = np.random.default_rng(seed)
        for i in range(len(point)):
            paths = corrected[i] + multiplier * (level[i] * self.level_paths_[:, None]
                                                 + shape[i] * self.shape_paths_)
            lower[i], upper[i] = weighted_quantile(paths, self.row_weights_,
                                                  [self.alpha / 2, 1 - self.alpha / 2])
            domains = rng.integers(len(self.domain_indices_), size=n_samples)
            draws = np.empty(n_samples, dtype=int)
            for domain, available in enumerate(self.domain_indices_):
                positions = np.flatnonzero(domains == domain)
                draws[positions] = rng.choice(available, size=len(positions))
            ensemble[i] = paths[draws]
        return {"ensemble": ensemble, "lower": lower, "upper": upper,
                "corrected_point": corrected, "level_scale": level, "shape_scale": shape, "bias": bias}
