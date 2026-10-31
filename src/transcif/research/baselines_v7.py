"""Same-information direct ExtraTrees and interval-only CQR source baselines.

The caller supplies only source proper-fit and independent source-calibration
cohorts. This array API has no date metadata and therefore cannot itself enforce
the caller's 2024H1/H2 split. Prediction accepts features/calendar only, never
target outcomes. CQR's exchangeability interpretation is not a guarantee under
temporal dependence or unseen-domain shift.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from .calibration_v4 import _daily, _labels


DEFAULT_TREE_PARAMETERS = {"n_estimators": 150, "min_samples_leaf": 10,
                           "random_state": 17023, "n_jobs": 1}
DEFAULT_CQR_PARAMETERS = {"max_iter": 80, "max_leaf_nodes": 15,
                          "l2_regularization": 10., "random_state": 17023,
                          "early_stopping": False}
BOUNDS = (0., 1000.)


def equal_region_weights(regions):
    """Probability mass 1/(R*n_region) for each source day, summing to one."""
    regions = np.asarray(regions)
    if regions.ndim != 1 or not len(regions):
        raise ValueError("regions must be nonempty one-dimensional labels.")
    labels = _labels(regions, len(regions), "regions")
    domains = np.unique(labels)
    result = np.empty(len(labels))
    for domain in domains:
        mask = labels == domain
        result[mask] = 1 / (len(domains) * mask.sum())
    return result


def _features(values, name, n_features=None):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 2 or not values.shape[0] or not values.shape[1]
            or not np.isfinite(values).all() or (n_features is not None and values.shape[1] != n_features)):
        raise ValueError(f"{name} must be a nonempty finite (N,P) feature matrix with matching P.")
    return values


class DirectBaselines:
    """Multivariate tree point + paired daily residual bank, and hourly CQR.

    Defaults are the predeclared V7 comparators. Parameter overrides support
    small behavior tests; callers must freeze any overrides in their protocol.
    Fit probability weights give each region mass 1/R. Estimator sample weights
    are rescaled to mean one to retain the stated HGB L2 units; this global
    factor leaves equal-region relative weighting unchanged.
    """

    def __init__(self, *, alpha=.1, tree_parameters=None, cqr_parameters=None):
        if not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("alpha must lie strictly between zero and one.")
        if cqr_parameters is not None and ({"loss", "quantile"} & set(cqr_parameters)):
            raise ValueError("CQR loss/quantiles are fixed by alpha, not parameter overrides.")
        self.alpha = float(alpha)
        self.tree_parameters = {**DEFAULT_TREE_PARAMETERS, **(tree_parameters or {})}
        self.cqr_parameters = {**DEFAULT_CQR_PARAMETERS, **(cqr_parameters or {})}
        self.fitted = False

    def _transform(self, x):
        return (x - self.feature_center_) / self.feature_scale_

    def _hourly_design(self, x):
        hour = np.tile(np.arange(24), len(x))
        phase = 2 * np.pi * hour / 24
        return np.column_stack([np.repeat(self._transform(x), 24, axis=0),
                                np.sin(phase), np.cos(phase), np.sin(2 * phase), np.cos(2 * phase)])

    def _quantiles(self, x):
        design = self._hourly_design(x)
        with threadpool_limits(limits=1):
            values = np.stack([model.predict(design) for model in self.cqr_models_])
        # Quantile functions can cross locally. Use the ordered envelope both
        # when calculating source calibration scores and during prediction.
        return np.sort(values, axis=0).reshape(2, len(x), 24)

    def fit(self, x_fit, y_fit, regions_fit, x_cal, y_cal, regions_cal):
        self.fitted = False
        x_fit = _features(x_fit, "x_fit")
        x_cal = _features(x_cal, "x_cal", x_fit.shape[1])
        y_fit, _ = _daily(y_fit, "y_fit")
        y_cal, _ = _daily(y_cal, "y_cal")
        if len(y_fit) != len(x_fit) or len(y_cal) != len(x_cal):
            raise ValueError("Features and daily outcomes must have matching row counts in each split.")
        if np.any(y_fit < BOUNDS[0]) or np.any(y_fit > BOUNDS[1]) or np.any(y_cal < BOUNDS[0]) or np.any(y_cal > BOUNDS[1]):
            raise ValueError("Source CIF outcomes must lie in [0,1000].")
        fit_regions = _labels(regions_fit, len(x_fit), "regions_fit")
        cal_regions = _labels(regions_cal, len(x_cal), "regions_cal")
        if set(fit_regions) != set(cal_regions):
            raise ValueError("Every source region must occur in both proper-fit and calibration cohorts.")
        self.source_regions_ = tuple(sorted(set(fit_regions)))
        self.fit_probability_weights_ = equal_region_weights(fit_regions)
        self.cal_probability_weights_ = equal_region_weights(cal_regions)
        self.feature_count_ = x_fit.shape[1]
        weights = self.fit_probability_weights_
        self.feature_center_ = np.sum(weights[:, None] * x_fit, axis=0)
        self.feature_scale_ = np.maximum(np.sqrt(np.sum(weights[:, None] *
            (x_fit - self.feature_center_)**2, axis=0)), 1e-6)
        if not np.isfinite(self.feature_center_).all() or not np.isfinite(self.feature_scale_).all():
            raise ValueError("Source feature normalization overflowed.")
        self.tree_ = ExtraTreesRegressor(**self.tree_parameters)
        # Mean-one estimator weights are important for HGB's fixed L2 penalty.
        estimator_weights = weights * len(x_fit)
        self.fit_estimator_weights_ = estimator_weights.copy()
        self.fit_hour_probability_weights_ = np.repeat(weights / 24, 24)
        with threadpool_limits(limits=1):
            self.tree_.fit(self._transform(x_fit), y_fit, sample_weight=estimator_weights)
            cal_point = np.clip(self.tree_.predict(self._transform(x_cal)), *BOUNDS)
        self.residual_paths_ = y_cal - cal_point
        self.residual_regions_ = cal_regions.copy()
        self.cqr_models_ = []
        design = self._hourly_design(x_fit)
        for quantile in (self.alpha / 2, 1 - self.alpha / 2):
            model = HistGradientBoostingRegressor(loss="quantile", quantile=quantile, **self.cqr_parameters)
            with threadpool_limits(limits=1):
                model.fit(design, y_fit.ravel(), sample_weight=np.repeat(estimator_weights, 24))
            self.cqr_models_.append(model)
        lower, upper = self._quantiles(x_cal)
        scores = np.maximum(lower - y_cal, y_cal - upper).ravel()
        rank = int(np.ceil((len(scores) + 1) * (1 - self.alpha)))
        if rank > len(scores):
            raise ValueError("Too few calibration hours for a finite CQR rank.")
        self.cqr_offset_ = float(np.partition(scores, rank - 1)[rank - 1])
        self.audit_ = {"source_regions": list(self.source_regions_), "proper_rows": len(x_fit),
            "calibration_rows": len(x_cal), "calibration_hours": len(scores), "feature_count": self.feature_count_,
            "proper_region_counts": {r: int(np.sum(fit_regions == r)) for r in self.source_regions_},
            "calibration_region_counts": {r: int(np.sum(cal_regions == r)) for r in self.source_regions_},
            "fit_weight": "probability 1/(R*n_region); estimator probability*N, mean one",
            "hour_weight": "day probability/24; estimator hour probability*(24*N)",
            "feature_transform_fit": "proper-fit x only; equal-region weighted center and scale",
            "tree_parameters": deepcopy(self.tree_parameters), "cqr_parameters": deepcopy(self.cqr_parameters),
            "cqr_quantiles": [self.alpha / 2, 1 - self.alpha / 2], "cqr_signed_offset": self.cqr_offset_,
            "cqr_rank": rank, "cqr_calibration": "pooled hourly signed max(lower-y,y-upper), finite-sample rank",
            "tree_resampling": "whole paired source calibration day; equal region probability",
            "cqr_interval_only": True, "bounds": list(BOUNDS), "target_outcomes_accepted_by_predict": False,
            "date_boundary": "caller must enforce source proper/calibration dates; array API has no date metadata",
            "coverage_guarantee_claimed": False}
        self.fitted = True
        return self

    @staticmethod
    def _rng(seed, origin):
        stamp = int(origin.value)
        return np.random.default_rng(np.random.SeedSequence([
            int(seed) & 0xffffffff, int(seed) >> 32, stamp & 0xffffffff, (stamp >> 32) & 0xffffffff]))

    def predict(self, x, origins, n_scenarios=128, seed=17023):
        if not self.fitted:
            raise RuntimeError("Fit DirectBaselines before prediction.")
        x = _features(x, "x", self.feature_count_)
        origins = pd.DatetimeIndex(pd.to_datetime(origins, utc=True))
        if len(origins) != len(x) or origins.hasnans:
            raise ValueError("origins must be nonmissing and match feature rows.")
        if not isinstance(n_scenarios, (int, np.integer)) or isinstance(n_scenarios, bool) or n_scenarios < 1:
            raise ValueError("n_scenarios must be a positive integer.")
        if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or not 0 <= seed < 2**64:
            raise ValueError("seed must be a nonnegative integer less than 2**64.")
        with threadpool_limits(limits=1):
            point = np.clip(self.tree_.predict(self._transform(x)), *BOUNDS)
        ensemble = np.empty((len(x), n_scenarios, 24))
        cdf = np.cumsum(self.cal_probability_weights_)
        cdf[-1] = 1.
        for i, origin in enumerate(origins):
            indices = np.searchsorted(cdf, self._rng(seed, origin).random(n_scenarios), side="right")
            ensemble[i] = np.clip(point[i] + self.residual_paths_[indices], *BOUNDS)
        lower, upper = self._quantiles(x)
        lower, upper = lower - self.cqr_offset_, upper + self.cqr_offset_
        # Negative signed offsets can collapse a narrow target interval. Repair
        # to its midpoint before clipping; do not invent a distribution/CRPS.
        midpoint = (lower + upper) / 2
        crossed = lower > upper
        lower = np.clip(np.where(crossed, midpoint, lower), *BOUNDS)
        upper = np.clip(np.where(crossed, midpoint, upper), *BOUNDS)
        return {"extratrees": {"point": point, "ensemble": ensemble},
                "cqr": {"lower": lower, "upper": upper}}
