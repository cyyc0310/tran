"""Fixed-energy contiguous-load scheduling on production-CIF proxy costs."""

import numpy as np


def block_scores(samples, duration=4, risk="mean", alpha=.9):
    """Cost per feasible block for each leading item of (...,S,H)."""
    samples = np.asarray(samples, float)
    if (samples.ndim < 2 or samples.shape[-2] < 1 or
            not isinstance(duration, (int, np.integer)) or duration < 1 or duration > samples.shape[-1]
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
    return score


def choose_block(samples, duration=4, risk="mean", alpha=.9):
    """samples (...,S,H); return a feasible block start per leading item."""
    return block_scores(samples, duration, risk, alpha).argmin(-1)


def choose_robust_block(components, duration=4, risk="cvar", alpha=.9):
    """Minimize max_k risk(P_k) over a FINITE set of mechanism laws.

    components (...,K,S,H). In particular, max component CVaR is not claimed
    to equal worst CVaR over every possible mixture of component laws.
    """
    components = np.asarray(components, float)
    if components.ndim < 3 or components.shape[-3] < 1:
        raise ValueError("nonempty mechanism scenarios (...,K,S,H) required")
    return block_scores(components, duration, risk, alpha).max(-2).argmin(-1)


def mechanism_decisions(components, supported, duration=4, alpha=.9, max_shortage=.01):
    cif = components["cif"]
    if cif.ndim != 4 or np.shape(supported) != (len(cif),) or not 0 <= max_shortage <= 1:
        raise ValueError("components (N,K,S,H), support mask (N,) and shortage limit in [0,1] required")
    mixed = cif.reshape(len(cif), -1, cif.shape[-1])
    shortage = (components["unserved"] / np.maximum(components["demand"], 1e-9)).mean((2, 3)).max(1)
    return {"mixture_mean": choose_block(mixed, duration, "mean", alpha),
        "mixture_cvar": choose_block(mixed, duration, "cvar", alpha),
        "worst_mechanism_mean": choose_robust_block(cif, duration, "mean", alpha),
        "worst_mechanism_cvar": choose_robust_block(cif, duration, "cvar", alpha),
        "accepted": np.asarray(supported, bool) & (shortage <= max_shortage),
        "max_mechanism_unserved_fraction": shortage}


def _tail_mean(values, alpha=.9):
    mass = (1 - alpha) * len(values)
    weights = np.clip(mass - np.arange(len(values)), 0, 1)
    return float((np.sort(values)[::-1] * weights).sum() / mass)


def scheduling_metrics(truth, samples, duration=4, risk="mean"):
    return {**score_blocks(truth, choose_block(samples, duration, risk), duration), "risk": risk}


def score_blocks(truth, start, duration=4, accepted=None):
    """Always report all proposed decisions, plus explicitly selective results."""
    truth = np.asarray(truth, float)
    start = np.asarray(start)
    if (truth.ndim != 2 or len(truth) < 1 or not np.isfinite(truth).all()
            or start.shape != truth.shape[:-1] or not np.issubdtype(start.dtype, np.integer)
            or duration < 1 or duration > truth.shape[-1]
            or (start < 0).any() or (start > truth.shape[-1] - duration).any()):
        raise ValueError("finite truth (N,H), feasible integer starts (N,) and duration required")
    feasible = np.lib.stride_tricks.sliding_window_view(truth, duration, axis=-1).mean(-1)
    selected = np.take_along_axis(feasible, start[..., None], axis=-1).squeeze(-1)
    uniform = truth.mean(-1)
    accepted = np.ones(len(truth), dtype=bool) if accepted is None else np.asarray(accepted, bool)
    if accepted.shape != start.shape:
        raise ValueError("acceptance mask must match windows")
    regret = selected - feasible.min(-1)
    return {"proxy_cost_g_per_kwh": float(selected.mean()),
            "realized_proxy_cost_cvar90_g_per_kwh": _tail_mean(selected),
            "oracle_regret_g_per_kwh": float(regret.mean()),
            "worse_than_uniform_fraction": float((selected > uniform).mean()),
            "duration_hours": duration, "recommendation_rate": float(accepted.mean()),
            "accepted_windows": int(accepted.sum()),
            "accepted_proxy_cost_g_per_kwh": float(selected[accepted].mean()) if accepted.any() else None,
            "accepted_oracle_regret_g_per_kwh": float(regret[accepted].mean()) if accepted.any() else None}
