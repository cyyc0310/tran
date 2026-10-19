"""Fixed-energy contiguous-load scheduling on production-CIF proxy costs."""

import numpy as np


def choose_block(samples, duration=4, risk="mean", alpha=.9):
    """samples (...,S,H); return a feasible block start per leading item."""
    samples = np.asarray(samples, float)
    if (samples.ndim < 2 or duration < 1 or duration > samples.shape[-1]
            or not np.isfinite(samples).all() or not 0 < alpha < 1):
        raise ValueError("finite scenarios, valid duration and alpha in (0,1) required")
    costs = np.lib.stride_tricks.sliding_window_view(samples, duration, axis=-1).mean(-1)
    if risk == "mean":
        score = costs.mean(-2)
    elif risk == "cvar":
        # Fractional top-tail mass: exact for this discrete empirical distribution.
        size = samples.shape[-2]
        mass = (1 - alpha) * size
        weights = np.clip(mass - np.arange(size), 0, 1)
        score = (np.sort(costs, axis=-2)[..., ::-1, :] * weights[:, None]).sum(-2) / mass
    else:
        raise ValueError("risk must be mean or cvar")
    return score.argmin(-1)


def scheduling_metrics(truth, samples, duration=4, risk="mean"):
    truth = np.asarray(truth, float)
    start = choose_block(samples, duration, risk)
    feasible = np.lib.stride_tricks.sliding_window_view(truth, duration, axis=-1).mean(-1)
    selected = np.take_along_axis(feasible, start[..., None], axis=-1).squeeze(-1)
    uniform = truth.mean(-1)
    return {"proxy_cost_g_per_kwh": float(selected.mean()),
            "oracle_regret_g_per_kwh": float((selected - feasible.min(-1)).mean()),
            "worse_than_uniform_fraction": float((selected > uniform).mean()),
            "duration_hours": duration, "risk": risk}
