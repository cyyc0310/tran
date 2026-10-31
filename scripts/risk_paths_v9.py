"""Lawful coarse-composition pushforward and source-crossfit fine residuals.

Residual replacement is a research candidate, not a coverage guarantee. Its
identifying assumption is that whole-day errors given true coarse composition
transfer between source regions and a new region after conditioning on mix and
calendar. No target hourly data or future monthly composition is accepted.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from transcif.research.dynamic_v2 import EF
from transcif.research.fuel_transport_v7 import FuelTrajectoryBank, recompose_paths

METHODS = ("latest", "seasonal", "v7_mixture", "harmonic_ridge", "seasonal_kalman", "release_bridge")
RESIDUAL_METHODS = METHODS[2:]
BANDWIDTH = .3
SEED = 17023
_RESIDUAL_CACHE = {}


def configs() -> list[dict[str, Any]]:
    rows = []
    for round_ in (1, 2):
        for forecast in METHODS:
            modes = ["additive", "recompose"]
            if round_ == 2 and forecast in RESIDUAL_METHODS:
                modes += ["residual_global", "residual_conditional"]
            for mode in modes:
                proposed = mode == "residual_conditional" or (forecast == "release_bridge" and mode != "residual_global")
                rows.append(dict(round=round_, name=f"{forecast}--{mode}", mode=mode,
                    forecast=forecast, family="proposed" if proposed else "standard",
                    simple_anchor=mode == "additive",
                    standard_dynamic=forecast in ("harmonic_ridge", "seasonal_kalman") and not proposed))
    return rows


def _simplex(value, shape, label):
    value = np.asarray(value, float)
    if (value.shape != shape or not np.isfinite(value).all() or np.any(value < 0)
            or not np.allclose(value.sum(-1), 1., atol=1e-8, rtol=1e-8)):
        raise ValueError(f"Invalid {label} composition shape/simplex")
    return value


def _bank_key(bank):
    h = hashlib.sha256()
    for value in (bank.q_, bank.raw_cif_, bank.donor_mix_, bank.months_):
        h.update(np.ascontiguousarray(value).tobytes())
    h.update("|".join(map(str, bank.regions_)).encode())
    return h.hexdigest()


def crossfit_residuals(bank: FuelTrajectoryBank) -> np.ndarray:
    """Crossfit each source region against all other regions in THIS bank.

    An outer source-LODO bank has already excluded the evaluation region, so it
    cannot re-enter through the inner residual construction. Source actual
    region-month composition is a training covariate, never a target input.
    """
    key = _bank_key(bank)
    if key in _RESIDUAL_CACHE:
        return _RESIDUAL_CACHE[key]
    if len(np.unique(bank.regions_)) < 2:
        raise ValueError("Crossfit risk needs at least two training regions")
    residual = np.empty_like(bank.raw_cif_)
    for region in np.unique(bank.regions_):
        eligible = bank.regions_ != region
        for month in np.unique(bank.months_[bank.regions_ == region]):
            ids = np.flatnonzero((bank.regions_ == region) & (bank.months_ == month))
            # A fitted bank has one source year; validate the common mix rather
            # than silently using a representative for nonidentical records.
            mix = bank.donor_mix_[ids[0]]
            if not np.allclose(bank.donor_mix_[ids], mix, atol=1e-12, rtol=1e-12):
                raise ValueError("Crossfit grouping requires one mix per source region/month")
            weights, *_ = bank._weights(mix, pd.Timestamp(bank.days_[ids[0]]),
                                        "recompose_joint", BANDWIDTH, eligible)
            mean_q = np.einsum("d,dhf->hf", weights, bank.q_)
            center = recompose_paths(mix, mean_q)
            residual[ids] = bank.raw_cif_[ids] - center
    residual.setflags(write=False)
    _RESIDUAL_CACHE[key] = residual
    return residual


def _prediction(raw, audit):
    paths = np.clip(raw, 0., 1000.)
    return dict(point=paths.mean(1), ensemble=paths,
                audit={**audit, "raw_clip_fraction": float(np.mean(raw != paths))})


def predict_paths(bank: FuelTrajectoryBank, forecasts: dict[str, dict[str, Any]],
                  days: pd.DatetimeIndex | list[str]) -> dict[str, dict[str, Any]]:
    """Same donor uniforms for additive and ratio; no hourly truth argument.

    R1 maps each coarse joint draw through one central-mix-routed donor. R2
    maps coarse draws through the weighted mean fuel path, replacing donor
    variability with crossfit whole-day errors rather than adding both. Global
    errors are a standard reference; conditional errors use the SAME lawful
    Hellinger/calendar donor weights as the point mapping. Both preserve paired
    daily level and shape errors. Neither mode uses target-specific thresholds.
    """
    origins = pd.DatetimeIndex(pd.to_datetime(days, utc=True))
    if not len(origins) or origins.hasnans or np.any(origins < bank.cutoff_):
        raise ValueError("Nonempty origins must follow fitted bank cutoff")
    if set(forecasts) != set(METHODS):
        raise ValueError("Complete fixed coarse method grid required")
    n = len(origins)
    residuals = crossfit_residuals(bank)
    eligible = np.ones(len(bank.days_), dtype=bool)
    global_weights = np.zeros(len(bank.days_))
    regions = np.unique(bank.regions_)
    for region in regions:
        mask = bank.regions_ == region
        global_weights[mask] = 1. / (len(regions) * mask.sum())
    result = {}
    for method in METHODS:
        central = _simplex(forecasts[method]["mix"], (n, 8), method)
        draws = np.asarray(forecasts[method]["draws"], float)
        if draws.ndim != 3 or draws.shape[1] < 2:
            raise ValueError("Coarse draws must be (days, samples>=2, fuels)")
        samples = draws.shape[1]
        draws = _simplex(draws, (n, samples, 8), method + " draws")
        modes = ["additive", "recompose"]
        if method in RESIDUAL_METHODS:
            modes += ["residual_global", "residual_conditional"]
        raw = {mode: np.empty((n, samples, 24)) for mode in modes}
        min_ess, unsupported = float("inf"), []
        for i, origin in enumerate(origins):
            weights, *_ = bank._weights(central[i], origin, "recompose_joint", BANDWIDTH, eligible)
            uniforms = bank._rng(SEED, origin).random((samples, 8))
            cdf = np.cumsum(weights); cdf[-1] = 1.
            donor = np.searchsorted(cdf, uniforms[:, 0], side="right")
            raw["additive"][i] = bank.raw_cif_[donor] + ((draws[i] - bank.donor_mix_[donor]) @ EF)[:, None]
            raw["recompose"][i] = recompose_paths(draws[i], bank.q_[donor])
            if method in RESIDUAL_METHODS:
                mean_q = np.einsum("d,dhf->hf", weights, bank.q_)
                center_draws = recompose_paths(draws[i], np.broadcast_to(mean_q, (samples, 24, 8)))
                global_cdf = np.cumsum(global_weights); global_cdf[-1] = 1.
                # Separate uniforms from R1 donor draw, common across both
                # residual references so only the conditional weights change.
                global_ids = np.searchsorted(global_cdf, uniforms[:, 1], side="right")
                conditional_ids = np.searchsorted(cdf, uniforms[:, 1], side="right")
                raw["residual_global"][i] = center_draws + residuals[global_ids]
                raw["residual_conditional"][i] = center_draws + residuals[conditional_ids]
            min_ess = min(min_ess, float(1 / np.sum(weights**2)))
            unsupported.append(float(central[i] @ (1 - weights @ bank.fuel_support_)))
        audit = dict(source_regions=list(map(str, regions)), target_hourly_used=False,
            routing="central forecast Hellinger+calendar; fixed while propagating coarse draws",
            min_effective_donors=min_ess, unsupported_target_mass=float(np.mean(unsupported)),
            crossfit="inner leave-source-region-out within current outer bank, 2024H1 only",
            residual_risk_assumption="source conditional errors given actual composition transfer to forecast-composition draws",
            coarse_audit=forecasts[method]["audit"])
        for mode in modes:
            result[f"{method}--{mode}"] = _prediction(raw[mode], audit)
    return result
