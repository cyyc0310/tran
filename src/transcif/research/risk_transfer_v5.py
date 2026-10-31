"""Predictable coarse-innovation scaling, keeping level and shape separate."""
import numpy as np
import pandas as pd
from scipy.special import softmax

from .dynamic_v2 import EF
from .structural_v3 import delayed, zero_mean
from .transfer_v4 import FUEL_COLS, configurations, condition_level, history_descriptors

POINTS = ("support_gp_medium", "harmonic_ridge", "softmax_moe")


def coarse_risk(origins, region, monthly, extra_delay=0):
    """Forecast-time scale proxy; no target hourly observations enter."""
    history = delayed(monthly.loc[monthly.region.eq(region)], extra_delay)
    for column in ("period_start", "period_end", "available_at"):
        history[column] = pd.to_datetime(history[column], utc=True)
    history = history.sort_values("period_end")
    origins = pd.DatetimeIndex(pd.to_datetime(origins, utc=True))
    risks, ages, counts = [], [], []
    for origin in origins:
        rows = history[(history.period_end <= origin) & (history.available_at <= origin)].tail(24)
        counts.append(len(rows))
        if len(rows) < 3:
            risks.append(0.); ages.append(365. if rows.empty else (origin-rows.iloc[-1].period_end).total_seconds()/86400)
            continue
        mix = rows[FUEL_COLS].to_numpy(float)
        if not np.isfinite(mix).all() or (mix < 0).any() or not np.allclose(mix.sum(axis=1), 1, atol=1e-5):
            raise ValueError("Released shares require finite simplex rows")
        values = mix @ EF
        age = (origin-rows.iloc[-1].period_end).total_seconds()/86400
        lookup = {(row.period_start.year, row.period_start.month): value
                  for row, value in zip(rows.itertuples(), values)}
        innovations = [value-lookup[(row.period_start.year-1, row.period_start.month)]
                       for row, value in zip(rows.itertuples(), values)
                       if (row.period_start.year-1, row.period_start.month) in lookup]
        # RMS includes persistent changes; centering annual changes would erase
        # a steady structural drift that may be dangerous at the forecast time.
        annual = float(np.sqrt(np.mean(np.square(innovations)))) if innovations else 0.
        same_month = [value for row, value in zip(rows.itertuples(), values) if row.period_start.month == origin.month]
        disagreement = abs(values[-1]-same_month[-1]) if same_month else 0.
        dispersion = float(np.std(values[-12:]))
        risk = np.sqrt(.5*dispersion**2+annual**2+.25*disagreement**2)*np.sqrt(1+age/60)
        risks.append(risk); ages.append(age)
    return np.asarray(risks), np.asarray(ages), np.asarray(counts)


def risk_scales(origins, region, monthly, prediction, extra_delay=0, beta=1.):
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("Risk beta must be finite and nonnegative")
    risk, _, count = coarse_risk(origins, region, monthly, extra_delay)
    result = {key: np.asarray(value).copy() for key, value in prediction.items()}
    result["level_scale"] = np.sqrt(result["level_scale"]**2+(beta*risk)**2)
    result["scale"] = np.sqrt(result["level_scale"][:, None]**2+result["shape_scale"]**2)
    result["coarse_risk"] = risk
    result["risk_history_months"] = count
    return result


def predict_base(model, origins, region, monthly):
    """Exact three-method subset of the frozen V4 round2 predictor."""
    if model.variant != "round2":
        raise ValueError("Expected the frozen round2 model")
    origins = pd.DatetimeIndex(pd.to_datetime(origins, utc=True))
    lawful = delayed(monthly, model.extra_delay)
    bank = model.structural
    base = bank.predict(origins, region, monthly)
    x, gate, age, valid = history_descriptors(origins, region, lawful, bank.seasonal)
    age = np.where(valid, age, 365.)
    analog = zero_mean(base["analog_shape_half"])
    points, stds = {}, {}
    history = lawful[lawful.region.eq(region)]
    for name in ("support_gp_medium", "harmonic_ridge", "local_level"):
        level, sd = condition_level(history, origins, model.prior_level, configurations("round2")[name])
        points[name] = level[:, None]+analog
        stds[name] = np.sqrt(15**2+sd**2)
    experts = np.stack([base["latest"], base["seasonal"], points["harmonic_ridge"], points["local_level"]], axis=1)
    center, scale, weights = model.gate_fit
    probabilities = softmax((gate-center)/scale @ weights, axis=1)
    points["softmax_moe"] = np.einsum("bk,bkh->bh", probabilities, experts)
    stds["softmax_moe"] = np.sqrt(25**2+np.minimum(age, 365)*4.)
    norm = np.maximum(bank.analog_x.std(axis=0), .05)
    distances = (((x[:, None, :]-bank.analog_x[None, :, :])/norm)**2).mean(axis=2)
    k = min(32, len(bank.analog_x))
    indices = np.argpartition(distances, k-1, axis=1)[:, :k]
    weights = 1/(np.take_along_axis(distances, indices, axis=1)+.1)
    weights /= weights.sum(axis=1, keepdims=True)
    neighbors = bank.analog_shape[indices]
    center = np.sum(weights[:, :, None]*neighbors, axis=1)
    variance = np.sum(weights[:, :, None]*(neighbors-center[:, None, :])**2, axis=1)
    shape_scale = np.sqrt(.5*model.shape_sd[None, :]**2+.5*variance)
    return {name: {"point": points[name], "level_scale": stds[name], "shape_scale": shape_scale.copy(),
                   "scale": np.sqrt(stds[name][:, None]**2+shape_scale**2), "age": age.copy()}
            for name in POINTS}
