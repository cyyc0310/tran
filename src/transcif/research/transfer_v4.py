"""Historical-support conditioning with separate coarse level and hourly shape.

The GP is a research candidate, not an identified physical model. Monthly fuel
intensity is energy weighted; observation noise includes its mismatch to a
uniform-time average. Only released historical rows condition future levels.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import softmax

from .dynamic_v2 import EF, daily_profiles, _solve, _predict
from .eia_proxy import FUELS
from .structural_v3 import fit_model, delayed, features, zero_mean

VARIANTS = ("round1", "round2")
DAY_NS = 86400 * 10**9
FUEL_COLS = [f"fuelshare_{fuel}" for fuel in FUELS]


@dataclass(frozen=True)
class KernelConfig:
    tau: float = 120.
    seasonal_sd: float = 60.
    drift_sd: float = 45.
    noise_sd: float = 15.
    integrate: bool = True
    baseline_sd: float = 250.


def support_nodes(starts, ends, integrate=True):
    """Four-point quadrature of a uniform-time historical window."""
    starts, ends = np.asarray(starts, float), np.asarray(ends, float)
    if starts.shape != ends.shape or np.any(ends <= starts):
        raise ValueError("Positive, aligned observation supports required")
    if integrate:
        nodes, weights = np.polynomial.legendre.leggauss(4)
        return (starts[:, None] + (ends-starts)[:, None] * (nodes+1)/2), weights/2
    # Explicit ablation: represent an aggregate by its endpoint, not its support.
    return ends[:, None], np.ones(1)


def periodic_design(times):
    phase = 2 * np.pi * np.asarray(times) / 365.25
    return np.stack([np.sin(phase), np.cos(phase), np.sin(2*phase), np.cos(2*phase)], axis=-1) / np.sqrt(2)


def integrated_covariance(a, aw, b, bw, config):
    fa = np.einsum("ikd,k->id", periodic_design(a), aw)
    fb = np.einsum("ikd,k->id", periodic_design(b), bw)
    distance = np.abs(a[:, None, :, None] - b[None, :, None, :])
    transient = np.einsum("ijkl,k,l->ij", np.exp(-distance/config.tau), aw, bw)
    return config.baseline_sd**2 + config.seasonal_sd**2 * (fa @ fb.T) + config.drift_sd**2 * transient


def condition_level(history, origins, prior_level, config):
    """Return conditional mean/SD, keeping permanent and transient covariances.

    Rows after their release are the only observations admitted. Grouping by
    eligible row identity avoids repeated factorizations within a release month.
    """
    origins = pd.DatetimeIndex(pd.to_datetime(origins, utc=True)).as_unit("ns")
    history = history.copy()
    for col in ("available_at", "period_start", "period_end"):
        history[col] = pd.to_datetime(history[col], utc=True).dt.as_unit("ns")
    history = history.sort_values(["period_end", "available_at"])
    means = np.full(len(origins), float(prior_level))
    prior_var = config.baseline_sd**2 + config.seasonal_sd**2 + config.drift_sd**2
    stds = np.full(len(origins), np.sqrt(prior_var))
    if history.empty:
        return means, stds
    end = history.period_end.astype("int64").to_numpy()/DAY_NS
    start = history.period_start.astype("int64").to_numpy()/DAY_NS
    released = history.available_at.astype("int64").to_numpy()/DAY_NS
    t = origins.asi8/DAY_NS
    allowed = (released[None, :] <= t[:, None]) & (end[None, :] <= t[:, None])
    groups = {}
    for index, mask in enumerate(allowed):
        # A fixed cap makes the numerical cost bounded and is common to models.
        key = tuple(np.flatnonzero(mask)[-48:])
        groups.setdefault(key, []).append(index)
    for key, positions in groups.items():
        if not key:
            continue
        ix = np.asarray(key)
        rows = history.iloc[ix]
        mix = rows[FUEL_COLS].to_numpy(float)
        if not np.isfinite(mix).all() or (mix < 0).any() or not np.allclose(mix.sum(axis=1), 1, atol=1e-5):
            raise ValueError("Released fuel shares must be finite simplex values")
        nodes, weights = support_nodes(start[ix], end[ix], config.integrate)
        coverage = rows.coverage.to_numpy(float)
        if not np.isfinite(coverage).all() or np.any((coverage <= 0) | (coverage > 1)):
            raise ValueError("Observation coverage must be in (0,1]")
        covariance = integrated_covariance(nodes, weights, nodes, weights, config)
        covariance += np.diag(config.noise_sd**2/coverage + 1e-5)
        future = t[positions, None]
        cross = integrated_covariance(future, np.ones(1), nodes, weights, config)
        factor = np.linalg.cholesky(covariance)
        response = np.linalg.solve(factor.T, np.linalg.solve(factor, mix @ EF-prior_level))
        means[positions] = prior_level + cross @ response
        v = np.linalg.solve(factor, cross.T)
        stds[positions] = np.sqrt(np.maximum(prior_var-np.sum(v*v, axis=0), 0))
    return means, stds


def history_descriptors(origins, region, monthly, seasonal):
    """All routing signals use released coarse history, never hourly outcomes."""
    x, latest, rolling, same_month, age, valid = features(origins, region, monthly, seasonal)
    origins = pd.DatetimeIndex(origins)
    part = monthly[monthly.region.eq(region)].copy()
    for col in ("period_start", "period_end", "available_at"):
        part[col] = pd.to_datetime(part[col], utc=True)
    counts, trend, support, cover = [], [], [], []
    for origin in origins:
        rows = part[(part.period_end <= origin) & (part.available_at <= origin)].sort_values("period_end")
        counts.append(len(rows))
        if len(rows):
            values = rows[FUEL_COLS].to_numpy(float) @ EF
            trend.append(float(values[-1]-np.median(values[-3:])))
            support.append((rows.iloc[-1].period_end-rows.iloc[-1].period_start).total_seconds()/86400)
            cover.append(float(rows.iloc[-1].coverage))
        else:
            trend.append(0.); support.append(0.); cover.append(0.)
    gate = np.column_stack([np.ones(len(origins)), age/120, np.minimum(counts, 36)/36,
                            (same_month-latest)/100, np.abs(same_month-latest)/100,
                            np.asarray(trend)/100, np.asarray(support)/90, cover, valid])
    return x, gate, age, valid


def fit_gate(x, expert_levels, truth, penalty=10.):
    """A generic source-trained softmax mixture, retained as a strong control."""
    center, scale = x.mean(axis=0), np.maximum(x.std(axis=0), .2)
    center[0], scale[0] = 0., 1.
    z = (x-center)/scale
    w = np.zeros((z.shape[1], expert_levels.shape[1]))
    experts, y = expert_levels/100, truth/100
    for _ in range(250):
        probs = softmax(z @ w, axis=1)
        prediction = (probs*experts).sum(axis=1)
        score = np.clip(prediction-y, -1, 1)
        gradient = z.T @ (score[:, None]*probs*(experts-prediction[:, None]))/len(y)
        gradient += penalty/len(y)*w
        w -= .1*gradient
    return center, scale, w


def configurations(variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown frozen candidate round: {variant}")
    return {"harmonic_ridge": KernelConfig(seasonal_sd=75, drift_sd=0, noise_sd=20),
            "local_level": KernelConfig(seasonal_sd=0, drift_sd=60, noise_sd=15),
            "support_gp_short": KernelConfig(tau=45),
            "support_gp_medium": KernelConfig(tau=120),
            "support_gp_long": KernelConfig(tau=365, drift_sd=25),
            "endpoint_gp": KernelConfig(tau=120, integrate=False)}


def anchored_level(history, origins, anchor, fallback_sd, tau):
    """Transport same-support annual innovations onto a seasonal anchor.

    Annual differences cancel persistent seasonal/region structure. Their noisy
    integrated-window OU bridge is extrapolated separately, so a stale latest
    observation cannot erase the anchor. This is a candidate construction, not
    an assumption that the future repeats last year's dispatch.
    """
    history = history.copy()
    for column in ("period_start", "period_end", "available_at"):
        history[column] = pd.to_datetime(history[column], utc=True).dt.as_unit("ns")
    history = history.sort_values("period_end")
    means, scales = np.asarray(anchor).copy(), np.asarray(fallback_sd).copy()
    config = KernelConfig(tau=tau, seasonal_sd=0, drift_sd=30, baseline_sd=10, noise_sd=15)
    for i, origin in enumerate(pd.DatetimeIndex(origins)):
        part = history[(history.period_end <= origin) & (history.available_at <= origin)]
        lookup = {(row.period_start.year, row.period_start.month): row for row in part.itertuples()}
        deltas, starts, ends, coverages = [], [], [], []
        for row in part.itertuples():
            old = lookup.get((row.period_start.year-1, row.period_start.month))
            if old is None:
                continue
            # Match complete calendar-month support. Reject arbitrary aggregates
            # whose year/month identity alone does not identify the same window.
            if row.period_end != row.period_start + pd.offsets.MonthBegin(1) or old.period_end != old.period_start + pd.offsets.MonthBegin(1):
                continue
            current_mix = np.array([getattr(row, c) for c in FUEL_COLS])
            old_mix = np.array([getattr(old, c) for c in FUEL_COLS])
            if not np.isfinite(current_mix).all() or not np.isfinite(old_mix).all():
                raise ValueError("Released innovation shares must be finite")
            deltas.append((current_mix-old_mix) @ EF)
            starts.append(row.period_start.value/DAY_NS); ends.append(row.period_end.value/DAY_NS)
            coverages.append(min(row.coverage, old.coverage))
        if not deltas:
            continue
        nodes, weights = support_nodes(starts[-12:], ends[-12:])
        k = integrated_covariance(nodes, weights, nodes, weights, config)
        k += np.diag(config.noise_sd**2/np.asarray(coverages[-12:])+1e-5)
        future = np.array([[origin.value/DAY_NS]])
        cross = integrated_covariance(future, np.ones(1), nodes, weights, config)[0]
        factor = np.linalg.cholesky(k)
        coefficients = np.linalg.solve(factor.T, np.linalg.solve(factor, np.asarray(deltas[-12:])))
        means[i] += cross @ coefficients
        solved = np.linalg.solve(factor, cross)
        # The seasonal anchor is observed historical intensity, not noiseless
        # future truth; its own uncertainty remains in addition to bridge state.
        scales[i] = np.sqrt(20**2 + max(config.baseline_sd**2+config.drift_sd**2-solved@solved, 0))
    return means, scales


@dataclass
class TransferModel:
    structural: object
    monthly: pd.DataFrame
    extra_delay: int
    variant: str
    prior_level: float
    shape_sd: np.ndarray
    pooled_fit: tuple
    gate_fit: tuple

    def predict(self, origins, region, monthly):
        origins = pd.DatetimeIndex(pd.to_datetime(origins, utc=True))
        lawful = delayed(monthly, self.extra_delay)
        base = self.structural.predict(origins, region, monthly)
        x, gate, age, valid = history_descriptors(origins, region, lawful, self.structural.seasonal)
        age = np.where(valid, age, 365.)
        shape = zero_mean(base["latest"])
        analog = zero_mean(base["analog_shape_half"])
        points = {"latest": base["latest"], "seasonal": base["seasonal"],
                  "joint_half": base["seasonal"].mean(axis=1)[:, None]+analog,
                  "analog_half": base["analog_shape_half"],
                  "pooled_ridge": _predict(self.pooled_fit, np.column_stack([x, gate]))}
        history = lawful[lawful.region.eq(region)]
        conditional_sd = {}
        for name, config in configurations(self.variant).items():
            level, sd = condition_level(history, origins, self.prior_level, config)
            points[name] = level[:, None] + analog
            conditional_sd[name] = sd
        if self.variant == "round2":
            fallback = np.sqrt(25**2+np.minimum(age, 365)*4.)
            for tau in (90, 365):
                name = f"innovation_bridge_{tau}"
                level, sd = anchored_level(history, origins, base["seasonal"].mean(axis=1), fallback, tau)
                points[name] = level[:, None]+analog
                conditional_sd[name] = sd
            points["support_gp_balanced"] = .5*(points["support_gp_medium"]+points["support_gp_long"])
            conditional_sd["support_gp_balanced"] = np.sqrt(.5*(conditional_sd["support_gp_medium"]**2+conditional_sd["support_gp_long"]**2)
                + .25*(points["support_gp_medium"].mean(axis=1)-points["support_gp_long"].mean(axis=1))**2)
        expert_names = ("latest", "seasonal", "harmonic_ridge", "local_level")
        experts = np.stack([points[k] for k in expert_names], axis=1)
        center, scale, weights = self.gate_fit
        probs = softmax((gate-center)/scale @ weights, axis=1)
        points["softmax_moe"] = np.einsum("bk,bkh->bh", probs, experts)
        # Identical age/shape scale baseline prevents special treatment of a
        # candidate during calibration. GP models add conditional level variance.
        fallback_sd = np.sqrt(25**2 + np.minimum(age, 365)*4.)
        shape_scale = np.broadcast_to(self.shape_sd, (len(origins), 24)).copy()
        if self.variant == "round2":
            # Source-compatible shape spread is a separate uncertainty signal.
            # A global half-variance floor prevents sparse nearest neighbours
            # from declaring a new region's hourly shape certain.
            bank = self.structural
            norm = np.maximum(bank.analog_x.std(axis=0), .05)
            distances = (((x[:, None, :]-bank.analog_x[None, :, :])/norm)**2).mean(axis=2)
            k = min(32, len(bank.analog_x))
            indices = np.argpartition(distances, k-1, axis=1)[:, :k]
            weights = 1/(np.take_along_axis(distances, indices, axis=1)+.1)
            weights /= weights.sum(axis=1, keepdims=True)
            neighbor_shapes = bank.analog_shape[indices]
            center_shape = np.sum(weights[:, :, None]*neighbor_shapes, axis=1)
            variance = np.sum(weights[:, :, None]*(neighbor_shapes-center_shape[:, None, :])**2, axis=1)
            shape_scale = np.sqrt(.5*self.shape_sd[None, :]**2+.5*variance)
        result = {}
        for name, point in points.items():
            level_sd = np.sqrt(15**2 + conditional_sd[name]**2) if name in conditional_sd else fallback_sd
            hourly_scale = np.sqrt(level_sd[:, None]**2 + shape_scale**2)
            result[name] = {"point": point, "scale": hourly_scale, "age": age,
                            "level_scale": level_sd, "shape_scale": shape_scale.copy()}
        return result


def fit_transfer(frames, monthly, extra_delay=0, variant="round1"):
    """Fit source hourly statistics; target coarse history stays as-of at predict."""
    configurations(variant)
    if len(frames) < 2:
        raise ValueError("At least two training source regions required")
    for frame in frames.values():
        if not pd.to_datetime(frame.time, utc=True).dt.year.eq(2022).all():
            raise ValueError("V4 prediction fitting is restricted to source year 2022")
    structural = fit_model(frames, monthly, cutoff="2023-01-01T00:00:00Z", delay_days=extra_delay)
    records = [(r, *daily_profiles(f)) for r, f in sorted(frames.items())]
    truth = np.concatenate([y for _, _, y in records])
    prior = float(truth.mean())
    shape_sd = np.maximum(np.std(zero_mean(truth), axis=0), 10.)
    xx, gates, levels, targets = [], [], [], []
    lawful = delayed(monthly, extra_delay)
    for region, days, y in records:
        x, gate, _, _ = history_descriptors(days, region, lawful, structural.seasonal)
        base = structural.predict(days, region, monthly)
        expert = [base["latest"].mean(axis=1), base["seasonal"].mean(axis=1)]
        for name in ("harmonic_ridge", "local_level"):
            prediction, _ = condition_level(lawful[lawful.region.eq(region)], days, prior, configurations(variant)[name])
            expert.append(prediction)
        xx.append(np.column_stack([x, gate])); gates.append(gate)
        levels.append(np.column_stack(expert)); targets.append(y)
    pooled = _solve(np.concatenate(xx), truth)
    gating = fit_gate(np.concatenate(gates), np.concatenate(levels), truth.mean(axis=1))
    return TransferModel(structural, monthly, int(extra_delay), variant, prior, shape_sd, pooled, gating)
