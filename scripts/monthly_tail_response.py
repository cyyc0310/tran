"""Source-only recomposition features; no target responses or calibration fitting."""
from __future__ import annotations

import numpy as np
import pandas as pd

STAT_NAMES = ("low_mean", "high_mean", "low_median", "high_median", "low_sd", "high_sd",
              "low_q05", "high_q95", "unsupported_mass", "effective_regions")
RESPONSE_FEATURES = tuple(f"rsp_{mode}_{name}" for mode in ("pooled", "conditional") for name in STAT_NAMES)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    order = np.argsort(values, kind="stable")
    index = min(int(np.searchsorted(np.cumsum(weights[order]), probability, side="left")), len(order) - 1)
    return float(values[order[index]])


class MonthlyTailResponses:
    """Apply fixed donor selection/weighting to a pre-audited response cache.

    Each query is represented by monthly shares and calendar month only.
    The caller supplies its proper fit-region roster; the query's entire region
    is always removed, including when generating proper-fit training features.
    """

    def __init__(self, metadata: pd.DataFrame, low: np.ndarray, high: np.ndarray):
        self.metadata = metadata.sort_values("profile_id").reset_index(drop=True).copy()
        n = len(self.metadata)
        if not self.metadata.profile_id.eq(np.arange(n)).all() or low.shape != (2, n, n) or high.shape != low.shape:
            raise ValueError("canonical profile-indexed response arrays required")
        self.low, self.high = low, high
        self.shares = self.metadata[[f"fuelshare_{fuel}" for fuel in
            ("coal", "gas", "oil", "nuclear", "hydro", "wind", "solar", "other")]].to_numpy(float)
        if not np.isfinite(self.shares).all() or not np.allclose(self.shares.sum(axis=1), 1, atol=1e-12):
            raise ValueError("finite monthly simplexes required")
        self.groups = {region: np.asarray(ids, dtype=int) for region, ids in self.metadata.groupby("region").indices.items()}
        self.month_numbers = self.metadata.month.str.slice(5, 7).astype(int).to_numpy()
        self.month_labels = self.metadata.month.to_numpy(str)
        self.hours = self.metadata.hours.to_numpy(float)
        self._selection_cache: dict[tuple, tuple] = {}

    def _selection(self, query_id: int, fit_regions: tuple[str, ...]):
        key = (query_id, fit_regions)
        if key in self._selection_cache:
            return self._selection_cache[key]
        query = self.metadata.iloc[query_id]
        regions = sorted(set(fit_regions) - {query.region})
        if not regions or not set(regions).issubset(self.groups):
            raise ValueError("nonempty known proper-source regions required")
        ids, prior, labels = [], [], []
        month = self.month_numbers[query_id]
        for region in regions:
            available = self.groups[region]
            distance = np.abs(self.month_numbers[available] - month)
            distance = np.minimum(distance, 12 - distance)
            order = np.lexsort((available, self.month_labels[available], distance))
            chosen = available[order[:2]]
            if len(chosen) != 2:
                raise ValueError("two complete donor-month profiles required per region")
            ids.extend(int(i) for i in chosen)
            labels.extend([region] * len(chosen))
            prior.extend([1 / (len(regions) * len(chosen))] * len(chosen))
        ids, prior, labels = np.asarray(ids), np.asarray(prior), np.asarray(labels)
        if query.region in set(labels) or not set(labels).issubset(fit_regions):
            raise ValueError("query region entered its response donor bank")
        squared = .5 * ((np.sqrt(self.shares[ids]) - np.sqrt(self.shares[query_id])) ** 2).sum(axis=1)
        conditional = prior * np.exp(-squared / .3**2)
        conditional /= conditional.sum()
        unsupported = (self.shares[query_id] * (self.shares[ids] == 0)).sum(axis=1)
        result = ids, prior / prior.sum(), conditional, unsupported, labels
        self._selection_cache[key] = result
        return result

    def features(self, query_id: int, tau: int, fit_regions: tuple[str, ...]) -> tuple[dict, dict]:
        if tau not in (600, 900):
            raise ValueError("registered threshold required")
        ids, prior, conditional, unsupported, labels = self._selection(query_id, tuple(sorted(fit_regions)))
        t = (600, 900).index(tau)
        low = self.low[t, query_id, ids] / self.hours[ids]
        high = self.high[t, query_id, ids] / self.hours[ids]
        features = {}
        for mode, weights in (("pooled", prior), ("conditional", conditional)):
            lmean, hmean = float(weights @ low), float(weights @ high)
            lsd = float(np.sqrt(max(0.0, weights @ (low - lmean) ** 2)))
            hsd = float(np.sqrt(max(0.0, weights @ (high - hmean) ** 2)))
            region_mass = np.asarray([weights[labels == region].sum() for region in np.unique(labels)])
            values = (lmean, hmean, weighted_quantile(low, weights, .5), weighted_quantile(high, weights, .5),
                      lsd, hsd, weighted_quantile(low, weights, .05), weighted_quantile(high, weights, .95),
                      float(weights @ unsupported), float(1 / (region_mass @ region_mass)))
            features.update({f"rsp_{mode}_{name}": value for name, value in zip(STAT_NAMES, values)})
        if not np.isfinite(list(features.values())).all():
            raise ValueError("nonfinite source response feature")
        provenance = {"profile_id": query_id, "tau": tau, "donor_ids": ids.tolist(),
                      "donor_regions": sorted(set(labels)), "pooled_weights": prior.tolist(),
                      "conditional_weights": conditional.tolist()}
        return features, provenance
