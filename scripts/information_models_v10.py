"""Source-only, equal-information daily CIF references for V10.

The caller supplies as-of features and released harmonic-mix EF anchors. This
module accepts source proper-fit and source calibration labels only; an outer
source evaluation region must be excluded from both via ``exclude_region``.
It never accepts evaluation truth at prediction time.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

SOURCE_REGIONS = frozenset({"BPAT", "CISO", "ERCO", "FPL", "ISNE", "MISO", "NYIS"})
SEED = 17023
HOURS = 24
MAX_CIF = 1000.0


def _matrix(value: Any, label: str, *, columns: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or not len(array) or not array.shape[1]:
        raise ValueError(f"{label} must be a nonempty two-dimensional feature matrix")
    if columns is not None and array.shape[1] != columns:
        raise ValueError(f"{label} has the wrong feature width")
    if np.isinf(array).any():
        raise ValueError(f"{label} contains infinity; use NaN for missing features")
    return array


def _truth(value: Any, rows: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (rows, HOURS) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite (days, 24) hourly CIF")
    if np.any((array < 0) | (array > MAX_CIF)):
        raise ValueError(f"{label} must lie in [0, 1000]")
    return array


def _anchors(value: Any, rows: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (rows,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite daily scalar anchor per row")
    if np.any((array < 0) | (array > MAX_CIF)):
        raise ValueError(f"{label} must lie in [0, 1000]")
    return array


def _regions(value: Any, rows: int, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (rows,) or not all(isinstance(v, str) and v in SOURCE_REGIONS for v in array):
        raise ValueError(f"{label} must name only the seven declared source regions")
    return array.astype(str)


def _equal_region_weights(regions: np.ndarray) -> np.ndarray:
    counts = Counter(regions)
    # Mean weight is one; each represented region has the same total mass.
    return np.array([len(regions) / (len(counts) * counts[region]) for region in regions], dtype=float)


class InfoModel:
    """Ridge or ExtraTrees on 24-hour residuals around a lawful daily anchor."""

    def __init__(self, kind: str):
        if kind not in {"ridge", "extra_trees"}:
            raise ValueError("kind must be 'ridge' or 'extra_trees'")
        self.kind = kind
        self.fitted = False

    def _design(self, x: np.ndarray) -> np.ndarray:
        missing = np.isnan(x)
        filled = np.where(missing, self.medians_[None, :], x)
        return np.concatenate((filled, missing.astype(float)), axis=1)

    def fit(self, xfit: Any, yfit: Any, regionsfit: Any, anchorfit: Any,
            xcal: Any, ycal: Any, regionscal: Any, anchorcal: Any,
            *, exclude_region: str | None = None) -> "InfoModel":
        """Fit on source 2024H1 proper labels; calibrate on source 2024H2.

        The caller makes the temporal split and passes only as-of features. The
        optional outer held-out source region is rejected from BOTH splits.
        """
        self.fitted = False
        fit_x = _matrix(xfit, "xfit")
        cal_x = _matrix(xcal, "xcal", columns=fit_x.shape[1])
        fit_y = _truth(yfit, len(fit_x), "yfit")
        cal_y = _truth(ycal, len(cal_x), "ycal")
        fit_anchor = _anchors(anchorfit, len(fit_x), "anchorfit")
        cal_anchor = _anchors(anchorcal, len(cal_x), "anchorcal")
        fit_regions = _regions(regionsfit, len(fit_x), "regionsfit")
        cal_regions = _regions(regionscal, len(cal_x), "regionscal")
        if exclude_region is not None:
            if exclude_region not in SOURCE_REGIONS:
                raise ValueError("exclude_region must be a declared source region")
            if exclude_region in fit_regions or exclude_region in cal_regions:
                raise ValueError("Outer evaluation region is present in source fit/calibration")
        if len(set(fit_regions)) < 2 or len(set(cal_regions)) < 2:
            raise ValueError("Proper fit and calibration each need at least two source regions")

        # Source proper-fit rows alone determine every transform parameter.
        self.medians_ = np.array([
            float(np.median(column[~np.isnan(column)])) if np.any(~np.isnan(column)) else 0.0
            for column in fit_x.T
        ])
        self.all_missing_columns_ = np.isnan(fit_x).all(axis=0)
        fit_design = self._design(fit_x)
        weights = _equal_region_weights(fit_regions)
        self.scaler_ = StandardScaler().fit(fit_design, sample_weight=weights)
        transformed_fit = self.scaler_.transform(fit_design)
        if self.kind == "ridge":
            model = Ridge(alpha=30.0)
        else:
            model = ExtraTreesRegressor(n_estimators=150, min_samples_leaf=10,
                                        random_state=SEED, n_jobs=1)
        model.fit(transformed_fit, fit_y - fit_anchor[:, None], sample_weight=weights)
        self.model_ = model

        transformed_cal = self.scaler_.transform(self._design(cal_x))
        cal_point = np.clip(cal_anchor[:, None] + model.predict(transformed_cal), 0.0, MAX_CIF)
        self.residuals_ = cal_y - cal_point
        self.calibration_regions_ = tuple(sorted(set(cal_regions)))
        self.calibration_indices_ = {
            region: np.flatnonzero(cal_regions == region) for region in self.calibration_regions_
        }
        self.audit_ = {
            "kind": self.kind,
            "allowed_source_regions": sorted(SOURCE_REGIONS),
            "proper_regions": dict(sorted(Counter(fit_regions).items())),
            "calibration_regions": dict(sorted(Counter(cal_regions).items())),
            "excluded_evaluation_region": exclude_region,
            "proper_fit_only_transform": True,
            "equal_region_fit_weight": True,
            "equal_region_calibration_resampling": True,
            "all_missing_proper_columns": np.flatnonzero(self.all_missing_columns_).tolist(),
            "input_features": int(fit_x.shape[1]),
            "transformed_features": int(transformed_fit.shape[1]),
            "proper_days": int(len(fit_x)),
            "calibration_days": int(len(cal_x)),
            "residual_definition": "source calibration hourly truth minus clipped source model point",
            "point_definition": "clipped anchor plus raw model residual prediction; not ensemble mean",
            "seed": SEED,
        }
        self.fitted = True
        return self

    @staticmethod
    def _rng(origin: pd.Timestamp) -> np.random.Generator:
        stamp = int(origin.value)
        return np.random.default_rng(np.random.SeedSequence([
            SEED, stamp & 0xffffffff, (stamp >> 32) & 0xffffffff,
        ]))

    def predict(self, x: Any, anchors: Any, origins: Any,
                n_samples: int = 128) -> dict[str, Any]:
        """Predict without evaluation truth; resample complete source error days."""
        if not self.fitted:
            raise RuntimeError("Fit the source model before prediction")
        features = _matrix(x, "x", columns=len(self.medians_))
        anchor = _anchors(anchors, len(features), "anchors")
        dates = pd.DatetimeIndex(pd.to_datetime(origins, utc=True))
        if len(dates) != len(features) or dates.hasnans:
            raise ValueError("origins must contain one valid UTC timestamp per prediction row")
        if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)) or n_samples < 2:
            raise ValueError("n_samples must be an integer >= 2")
        raw = anchor[:, None] + self.model_.predict(self.scaler_.transform(self._design(features)))
        point = np.clip(raw, 0.0, MAX_CIF)
        ensemble = np.empty((len(features), int(n_samples), HOURS), dtype=float)
        region_count = len(self.calibration_regions_)
        for i, origin in enumerate(dates):
            uniforms = self._rng(origin).random((int(n_samples), 2))
            region_ids = np.minimum((uniforms[:, 0] * region_count).astype(int), region_count - 1)
            residual_ids = np.empty(int(n_samples), dtype=int)
            for region_index, region in enumerate(self.calibration_regions_):
                positions = np.flatnonzero(region_ids == region_index)
                candidates = self.calibration_indices_[region]
                selected = np.minimum((uniforms[positions, 1] * len(candidates)).astype(int), len(candidates) - 1)
                residual_ids[positions] = candidates[selected]
            ensemble[i] = np.clip(point[i, None, :] + self.residuals_[residual_ids], 0.0, MAX_CIF)
        return {"point": point, "ensemble": ensemble, "audit": dict(self.audit_)}
