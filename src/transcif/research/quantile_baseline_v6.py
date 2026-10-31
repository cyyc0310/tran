"""An interval-only CQR comparator with temporal source fitting/calibration.

The usual exchangeability coverage theorem does not apply automatically to
dependent hourly data or unseen regions. No synthetic ensemble/CRPS is invented
from the two quantile endpoints.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


class QuantileBaseline:
    def __init__(self, source_regions, alpha=.1, fit_year=2024):
        self.source_regions = tuple(source_regions)
        if not self.source_regions or len(set(self.source_regions)) != len(self.source_regions):
            raise ValueError("Declare unique source regions")
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0,1)")
        self.alpha, self.fit_year = float(alpha), int(fit_year)

    @staticmethod
    def _design(point, features, level_scales, shape_scales):
        point, features = np.asarray(point, float), np.asarray(features, float)
        if point.ndim != 2 or point.shape[1] != 24 or features.ndim != 2 or len(point) != len(features):
            raise ValueError("Expected daily points and matching features")
        n = len(point)
        level = np.asarray(level_scales, float)
        shape = np.asarray(shape_scales, float)
        if level.shape != (n,) or shape.shape != (n, 24):
            raise ValueError("Expected level (N,) and shape (N,24) scales")
        hours = np.tile(np.arange(24), n)
        design = np.column_stack((np.repeat(features, 24, axis=0), point.ravel(),
                                  np.repeat(level, 24), shape.ravel(),
                                  np.sin(2*np.pi*hours/24), np.cos(2*np.pi*hours/24),
                                  np.sin(4*np.pi*hours/24), np.cos(4*np.pi*hours/24)))
        if not np.isfinite(design).all() or (level < 0).any() or (shape < 0).any():
            raise ValueError("Finite features and nonnegative scales required")
        return design

    def fit(self, point, truth, features, regions, *, days, level_scales, shape_scales):
        point, truth = np.asarray(point, float), np.asarray(truth, float)
        regions = np.asarray(regions)
        days = pd.DatetimeIndex(pd.to_datetime(days, utc=True))
        if point.shape != truth.shape or not np.isfinite(truth).all() or regions.shape != (len(point),) or len(days) != len(point):
            raise ValueError("Mismatched finite source arrays")
        if set(regions) != set(self.source_regions) or np.any(days.year != self.fit_year):
            raise ValueError("Only the declared source fitting year is legal")
        proper = days.month <= 6
        bank = ~proper
        if any(not np.any(mask & (regions == r)) for mask in (proper, bank) for r in self.source_regions):
            raise ValueError("Every source needs proper and calibration observations")
        design = self._design(point, features, level_scales, shape_scales)
        train = np.repeat(proper, 24)
        cal = np.repeat(bank, 24)
        residual = (truth-point).ravel()
        counts = {r: np.sum(proper & (regions == r)) for r in self.source_regions}
        weights = np.repeat([np.sum(proper)/(len(counts)*counts[r]) for r in regions[proper]], 24)
        self.models_ = []
        for quantile in (self.alpha/2, 1-self.alpha/2):
            model = GradientBoostingRegressor(loss="quantile", alpha=quantile, n_estimators=80,
                max_depth=2, min_samples_leaf=100, learning_rate=.05, random_state=17023)
            model.fit(design[train], residual[train], sample_weight=weights)
            self.models_.append(model)
        endpoints = np.sort(np.stack([m.predict(design[cal]) for m in self.models_]), axis=0)
        scores = np.maximum(endpoints[0]-residual[cal], residual[cal]-endpoints[1])
        rank = int(np.ceil((len(scores)+1)*(1-self.alpha)))
        if rank > len(scores):
            raise ValueError("Too few calibration scores for a finite interval")
        self.offset_ = float(np.partition(scores, rank-1)[rank-1])
        self.audit_ = {"fit_year": self.fit_year, "proper_months": [1,6], "calibration_months": [7,12],
            "proper_days": int(proper.sum()), "calibration_days": int(bank.sum()),
            "source_regions": list(self.source_regions), "offset": self.offset_,
            "calibration": "pooled hourly signed CQR scores; finite-sample rank", "interval_only": True,
            "coverage_guarantee_claimed": False, "target_hourly_labels_used": False}
        return self

    def intervals(self, point, features, *, level_scales, shape_scales):
        if not hasattr(self, "models_"):
            raise RuntimeError("Fit before prediction")
        point = np.asarray(point, float)
        design = self._design(point, features, level_scales, shape_scales)
        ends = np.sort(np.stack([m.predict(design) for m in self.models_]), axis=0)
        lower, upper = ends[0]-self.offset_, ends[1]+self.offset_
        # A negative CQR offset can cross locally narrow endpoints. Collapse to
        # their midpoint; report this as engineering repair, not a new theorem.
        midpoint = (lower+upper)/2
        crossed = lower > upper
        lower = np.where(crossed, midpoint, lower)
        upper = np.where(crossed, midpoint, upper)
        return point+lower.reshape(point.shape), point+upper.reshape(point.shape)
