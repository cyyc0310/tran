"""Fixed-donor-law risk curves in a virtual coal-composition coordinate."""
from __future__ import annotations

from fractions import Fraction as F

import numpy as np
import pandas as pd

from monthly_tail_response import MonthlyTailResponses
from v144_cache_composition_responses import exact_coefficients, rational_sign
from v144_prepare_profile_cache import FACTORS

DELTAS = np.linspace(0, 1, 257)


def row_quantile(values: np.ndarray, weights: np.ndarray, probability: float):
    order = np.argsort(values, axis=1, kind="stable")
    sorted_values = np.take_along_axis(values, order, axis=1)
    index = np.minimum((np.cumsum(weights[order], axis=1) < probability).sum(axis=1), len(weights) - 1)
    return sorted_values[np.arange(len(values)), index]


class CriticalCompositionResponses:
    def __init__(self, engine: MonthlyTailResponses, index: pd.DataFrame, raw: np.ndarray, margin_strings: np.ndarray):
        self.engine, self.index, self.raw = engine, index, raw
        self.margins = [[F(str(value)) for value in row] for row in margin_strings]
        self.shapes: dict[int, np.ndarray] = {}
        self.identity_checks, self.noncoal_exact_decisions, self.critical_tie_corrections = 0, 0, 0

    def _raw_shape(self, donor_id: int):
        row = self.index.iloc[donor_id]
        raw = self.raw[int(row.offset):int(row.stop)]
        if donor_id not in self.shapes:
            means = np.asarray([float(x) / row.hours for x in self.margins[donor_id]])
            self.shapes[donor_id] = np.divide(raw, means, out=np.ones_like(raw), where=means > 0)
        return raw, self.shapes[donor_id], int(row.hours)

    def _critical(self, query_id: int, donor_id: int, tau: int, factor: str):
        if self.index.region.iloc[query_id] == self.index.region.iloc[donor_id]:
            raise ValueError("query-region hourly profile entered its own curve")
        raw, shape, hours = self._raw_shape(donor_id)
        coefficients = np.asarray(FACTORS[factor]) - tau
        p = self.engine.shares[query_id, 0]
        if p >= 1:
            raise ValueError("pure-coal query requires the declared physical fallback")
        if np.all(coefficients <= 0):
            critical = np.full(hours, np.inf)
        else:
            ratios = self.engine.shares[query_id, 1:] / (1 - p)
            a = coefficients[0] * shape[:, 0]
            b = shape[:, 1:] @ (coefficients[1:] * ratios)
            absolute = shape[:, 1:] @ (np.abs(coefficients[1:]) * ratios)
            near = np.flatnonzero((np.abs(b) <= 64 * np.finfo(float).eps * absolute) & (absolute > 0))
            if len(near):
                query_noncoal = list(self.margins[query_id])
                query_noncoal[0] = F(0)
                terms, constant = exact_coefficients(query_noncoal, self.margins[donor_id], coefficients, hours)
                total_noncoal = sum(query_noncoal, F(0))
                for h in near:
                    exact = sum((F(str(float(x))) * term for x, term in zip(raw[h], terms)), constant)
                    b[h] = float(exact * hours / total_noncoal)
                    self.noncoal_exact_decisions += 1
            critical = np.full(hours, np.inf)
            critical[b > 0] = -np.inf
            critical[(b == 0) & (a > 0)] = 0.0
            crossing = (b < 0) & (a > 0)
            critical[crossing] = -b[crossing] / (a[crossing] - b[crossing])
            ties = np.flatnonzero(np.isfinite(critical) & (np.abs(critical - p) <= 128 * np.finfo(float).eps))
            if len(ties):
                terms, constant = exact_coefficients(self.margins[query_id], self.margins[donor_id], coefficients, hours)
                for h in ties:
                    event = rational_sign(raw[h], terms, constant)
                    before = critical[h] < p
                    critical[h] = np.nextafter(p, -np.inf) if event else p
                    self.critical_tie_corrections += bool(before) != event
        actual = int(np.count_nonzero(critical < p))
        t = (600, 900).index(tau)
        expected = int((self.engine.low if factor == "low" else self.engine.high)[t, query_id, donor_id])
        self.identity_checks += 1
        if actual != expected:
            raise ValueError(f"critical response identity differs: q={query_id},d={donor_id},tau={tau},factor={factor},{actual}!={expected}")
        return np.sort(critical), hours

    def curves(self, query_id: int, fit_regions: tuple[str, ...]):
        ids, _, weights, _, labels = self.engine._selection(query_id, tuple(sorted(fit_regions)))
        p = self.engine.shares[query_id, 0]
        left, right = np.maximum(0, p - DELTAS), np.minimum(np.nextafter(1.0, 0), p + DELTAS)
        curves = np.empty((2, len(DELTAS), 2))
        for t, tau in enumerate((600, 900)):
            lower, upper = np.empty((len(DELTAS), len(ids))), np.empty((len(DELTAS), len(ids)))
            for j, donor_id in enumerate(ids):
                lc, hours = self._critical(query_id, int(donor_id), tau, "low")
                hc, _ = self._critical(query_id, int(donor_id), tau, "high")
                lower[:, j] = np.searchsorted(lc, left, side="left") / hours
                upper[:, j] = np.searchsorted(hc, right, side="left") / hours
            curves[t, :, 0] = row_quantile(lower, weights, .05)
            curves[t, :, 1] = row_quantile(upper, weights, .95)
            features, _ = self.engine.features(query_id, tau, fit_regions)
            if (abs(curves[t, 0, 0] - features["rsp_conditional_low_q05"]) > 1e-12
                    or abs(curves[t, 0, 1] - features["rsp_conditional_high_q95"]) > 1e-12):
                raise ValueError("delta-zero band differs from the frozen conditional response")
        if (np.any(np.diff(curves[:, :, 0], axis=1) > 1e-12)
                or np.any(np.diff(curves[:, :, 1], axis=1) < -1e-12)
                or np.any(curves[:, :, 0] > curves[:, :, 1] + 1e-12)):
            raise ValueError("critical-composition delta family is not nested")
        return curves, {"donor_ids": ids.tolist(), "donor_regions": sorted(set(labels)), "weights": weights.tolist()}
