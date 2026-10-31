"""Weather-only shape increments and a standard joint-day QRF reference.

ShapeModel and ScaleModel are the frozen V11 implementations. The quantile
forest is a standard multioutput quantile-regression-forest extension: its
leaf weights define a distribution over complete calibration-day residual
vectors. It is a baseline, not a novel method or a CQR coverage guarantee.
Callers are responsible for obtaining geographically out-of-fold donor and
calibration predictions before passing them to these mechanisms.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.ensemble import RandomForestRegressor

from scripts.weather_shapes_v11 import (
    SEED,
    SOURCES,
    ScaleModel,
    ShapeModel,
    project_shapes,
    split_profiles,
)

STRENGTHS = (0.0, 0.25, 0.5, 1.0)
GLOBAL_FRACTIONS = (0.0, 0.25, 0.5)


def _finite_profiles(value: ArrayLike, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim < 2 or result.shape[-2:] != (24, 8) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite (...,24,8)")
    return result


def _registered(value: float, choices: tuple[float, ...], name: str) -> float:
    if not np.isscalar(value) or value not in choices:
        raise ValueError(f"{name} must be one of {choices}")
    return float(value)


def delta_transport(
    q: ArrayLike,
    query_W: ArrayLike,
    query_B: ArrayLike,
    donor_W: ArrayLike,
    donor_B: ArrayLike,
    support: ArrayLike,
    strength: float = 1.0,
) -> np.ndarray:
    """Apply only a weather-minus-base shape increment to joint donor profiles.

    q has shape (...,24,8). Predictions broadcast to q; support broadcasts to
    (...,8). Donor predictions must be geographically out of fold. The update
    is project(shape(q) + strength * ((query_W-query_B)-(donor_W-donor_B))).
    Each donor fuel's daily amplitude, unsupported flat q=1 convention, and
    zero-amplitude fuels are preserved. Zero strength or an identically zero
    increment returns a copy of q with its original dtype and bytes intact.
    """
    original = np.asarray(q)
    profiles = _finite_profiles(q, "q")
    if np.any(profiles < 0):
        raise ValueError("q must be nonnegative")
    strength = _registered(strength, STRENGTHS, "strength")
    supported = np.asarray(support)
    if not np.isin(supported, [False, True]).all():
        raise ValueError("support must contain boolean flags")
    try:
        supported = np.broadcast_to(supported.astype(bool), profiles.shape[:-2] + (8,))
        predictions = [np.broadcast_to(_finite_profiles(value, name), profiles.shape)
                       for value, name in ((query_W, "query_W"), (query_B, "query_B"),
                                           (donor_W, "donor_W"), (donor_B, "donor_B"))]
    except ValueError as exc:
        raise ValueError("predictions/support must be finite and broadcast to q (...,24,8)") from exc
    unsupported = np.broadcast_to(~supported[..., None, :], profiles.shape)
    if np.any(unsupported & (profiles != 1.0)):
        raise ValueError("unsupported donor fuels must use flat q=1")
    if strength == 0.0:
        return original.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        increment = (predictions[0] - predictions[1]) - (predictions[2] - predictions[3])
    if not np.isfinite(increment).all():
        raise ValueError("weather-minus-base increment must be finite")
    if not np.any(increment):
        return original.copy()
    amplitude, shape = split_profiles(profiles)
    corrected = amplitude[..., None, :] * project_shapes(shape + strength * increment)
    # Retain exact donor values for every unaffected fuel, including zero fuel.
    changed = supported & (amplitude > 0) & np.any(increment != 0.0, axis=-2)
    return np.where(changed[..., None, :], corrected, original)


def _matrix(x: ArrayLike) -> np.ndarray:
    result = np.asarray(x, dtype=float)
    if (result.ndim != 2 or not len(result) or not result.shape[1]
            or np.isinf(result).any()):
        raise ValueError("features must be a nonempty matrix, with NaN for missing")
    return result


class JointQuantileForest:
    """Standard QRF leaf distribution over whole 24-hour calibration errors.

    The multioutput forest uses 128 trees, leaf size 20, feature fraction .7,
    bootstrap sampling and seed 17023. Region-balanced training weights are
    also used inside each leaf. Imputation and missing indicators are fitted
    on this proper risk-fit cohort only. No hours are sampled independently.
    """

    def fit(
        self,
        x: ArrayLike,
        errors: ArrayLike,
        regions: Sequence[str] | np.ndarray,
        *,
        days: Sequence[object] | pd.DatetimeIndex | None = None,
        exclude_region: str | None = None,
    ) -> JointQuantileForest:
        """Fit to source-only, geographically crossfit 2024H2 daily errors."""
        self.fitted_ = False
        features = _matrix(x)
        residuals = np.asarray(errors, dtype=float)
        if residuals.shape != (len(features), 24) or not np.isfinite(residuals).all():
            raise ValueError("errors must be finite (rows,24)")
        labels = np.asarray(regions)
        if (labels.shape != (len(features),) or not set(labels) <= SOURCES
                or len(set(labels)) < 2):
            raise ValueError("fit requires at least two known source regions")
        if exclude_region is not None:
            if exclude_region not in SOURCES:
                raise ValueError("excluded region must be a known source")
            if exclude_region in labels:
                raise ValueError("excluded evaluation region is present in proper fit")
        if days is not None:
            dates = pd.DatetimeIndex(pd.to_datetime(days, utc=True))
            if (len(dates) != len(features) or dates.hasnans
                    or np.any(dates < pd.Timestamp("2024-07-01", tz="UTC"))
                    or np.any(dates >= pd.Timestamp("2025-01-01", tz="UTC"))):
                raise ValueError("fit dates must lie in 2024H2; future labels forbidden")
        self.n_features_in_ = features.shape[1]
        self.medians_ = np.array([
            np.median(column[~np.isnan(column)]) if (~np.isnan(column)).any() else 0.0
            for column in features.T
        ])
        counts = Counter(labels)
        region_weights = np.array([
            len(labels) / (len(counts) * counts[region]) for region in labels
        ])
        self.global_weights_ = region_weights / region_weights.sum()
        self.errors_ = residuals.copy()
        self.model_ = RandomForestRegressor(
            n_estimators=128, min_samples_leaf=20, max_features=0.7,
            bootstrap=True, random_state=SEED, n_jobs=1,
        )
        design = self._design(features)
        self.model_.fit(design, residuals, sample_weight=region_weights)
        # Store only the O(trees * calibration rows) leaf memberships, never a
        # query-by-calibration-by-tree tensor. Apply all proper rows per tree.
        leaves = self.model_.apply(design)
        self.leaf_maps_: list[dict[int, tuple[np.ndarray, np.ndarray]]] = []
        for tree in range(leaves.shape[1]):
            leaf_map = {}
            order = np.argsort(leaves[:, tree], kind="stable")
            boundaries = np.flatnonzero(np.diff(leaves[order, tree])) + 1
            for indices in np.split(order, boundaries):
                mass = region_weights[indices]
                leaf_map[int(leaves[indices[0], tree])] = (indices, mass / mass.sum())
            self.leaf_maps_.append(leaf_map)
        self.error_orders_ = np.argsort(self.errors_, axis=0, kind="stable")
        self.sorted_errors_ = np.take_along_axis(self.errors_, self.error_orders_, axis=0)
        self.training_regions_ = tuple(sorted(counts))
        self.audit_ = {
            "training_regions": list(self.training_regions_),
            "proper_regions": dict(sorted(counts.items())),
            "excluded_evaluation_region": exclude_region,
            "proper_fit_only_imputation": True,
            "equal_region_fit_weights": True,
            "equal_region_leaf_weights": True,
            "leaf_distribution": "all proper rows in leaf, normalized region weights; no bootstrap multiplicity",
            "forest_predictive_mean_equivalence_claimed": False,
            "dates_validated": days is not None,
            "fit_cohort": "source crossfit calibration 2024H2",
            "seed": SEED,
            "method": "standard multioutput quantile regression forest extension",
            "joint_day_sampling": True,
            "coverage_guarantee": False,
            "n_estimators": 128,
            "min_samples_leaf": 20,
            "max_features": 0.7,
            "bootstrap": True,
            "global_fractions": list(GLOBAL_FRACTIONS),
        }
        self.fitted_ = True
        return self

    def _design(self, x: ArrayLike) -> np.ndarray:
        features = _matrix(x)
        if features.shape[1] != self.n_features_in_:
            raise ValueError("feature width differs from proper fit")
        missing = np.isnan(features)
        return np.column_stack((np.where(missing, self.medians_, features), missing.astype(float)))

    def predict_weights(self, x: ArrayLike, global_fraction: float = 0.0) -> np.ndarray:
        """Return (query rows, calibration days) QRF weights with optional shrinkage."""
        if not getattr(self, "fitted_", False):
            raise RuntimeError("fit JointQuantileForest before prediction")
        fraction = _registered(global_fraction, GLOBAL_FRACTIONS, "global_fraction")
        leaves = self.model_.apply(self._design(x))
        weights = np.zeros((len(leaves), len(self.errors_)))
        for tree, leaf_map in enumerate(self.leaf_maps_):
            for leaf in np.unique(leaves[:, tree]):
                rows = np.flatnonzero(leaves[:, tree] == leaf)
                indices, masses = leaf_map[int(leaf)]
                weights[np.ix_(rows, indices)] += masses
        weights /= len(self.leaf_maps_)
        weights /= weights.sum(axis=1, keepdims=True)
        if fraction:
            weights = (1.0 - fraction) * weights + fraction * self.global_weights_[None, :]
        return weights

    def draw(self, x: ArrayLike, uniforms: ArrayLike, global_fraction: float = 0.0) -> np.ndarray:
        """Invert the day-weight CDF using supplied paired (N,S) uniforms in [0,1]."""
        weights = self.predict_weights(x, global_fraction)
        draws = np.asarray(uniforms, dtype=float)
        if (draws.ndim != 2 or draws.shape[0] != len(weights) or not draws.shape[1]
                or not np.isfinite(draws).all() or np.any((draws < 0) | (draws > 1))):
            raise ValueError("uniforms must be finite (query rows,samples) in [0,1]")
        cumulative = np.cumsum(weights, axis=1)
        indices = np.empty(draws.shape, dtype=np.intp)
        for row in range(len(weights)):
            last_positive = np.flatnonzero(weights[row] > 0)[-1]
            cumulative[row, last_positive:] = 1.0
            indices[row] = np.minimum(np.searchsorted(cumulative[row], draws[row], side="right"),
                                      last_positive)
        return self.errors_[indices]

    def predict_quantiles(
        self, x: ArrayLike, levels: ArrayLike, global_fraction: float = 0.0,
    ) -> np.ndarray:
        """Return (N,Q,24) marginal weighted empirical quantiles; no CQR claim."""
        weights = self.predict_weights(x, global_fraction)
        quantiles = np.asarray(levels, dtype=float)
        if (quantiles.ndim != 1 or not len(quantiles) or not np.isfinite(quantiles).all()
                or np.any((quantiles < 0) | (quantiles > 1))):
            raise ValueError("levels must be a finite nonempty vector in [0,1]")
        result = np.empty((len(weights), len(quantiles), 24))
        for hour in range(24):
            sorted_weights = weights[:, self.error_orders_[:, hour]]
            cumulative = np.cumsum(sorted_weights, axis=1)
            for row in range(len(weights)):
                positive = np.flatnonzero(sorted_weights[row] > 0)
                cumulative[row, positive[-1]:] = 1.0
                indices = np.searchsorted(cumulative[row], quantiles, side="left")
                # At level zero use the smallest value with positive mass.
                indices[quantiles == 0] = positive[0]
                indices[quantiles == 1] = positive[-1]
                result[row, :, hour] = self.sorted_errors_[indices, hour]
        return result
