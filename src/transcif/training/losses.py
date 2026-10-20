"""Proper Monte Carlo scores and source-only physical supervision."""

import torch


def fair_crps(samples, truth):
    """Unbiased MC CRPS estimator for iid scenario draws, shape B,S,H."""
    s = samples.shape[1]
    if s < 2:
        raise ValueError("at least two scenarios required")
    order = samples.sort(dim=1).values
    coeff = (2 * torch.arange(1, s + 1, device=samples.device) - s - 1)[None, :, None]
    spread = (order * coeff).sum(1) / (s * (s - 1))
    return ((samples - truth[:, None]).abs().mean(1) - spread).mean()


def energy_score(samples, truth):
    """Joint path energy score with unbiased off-diagonal MC spread."""
    s, h = samples.shape[1:]
    if s < 2:
        raise ValueError("at least two scenarios required")
    fit = torch.linalg.vector_norm(samples - truth[:, None], dim=-1).mean()
    # The matrix-multiplication shortcut can produce nonzero self-distances
    # and lose translation invariance for large, nearly equal CIF paths.
    spread = torch.cdist(samples, samples, compute_mode="donot_use_mm_for_euclid_dist").sum() / (
        samples.shape[0] * s * (s - 1))
    return (fit - spread / 2) / h**0.5


def dispatch_loss(forecast, labels):
    """CIF probability scores + mean fuel-share error + explicit shortage penalty.

    Labels are source-domain observations, never passed to the model.
    """
    cif = labels["cif"]
    true_gen = labels["generation"]
    true_share = true_gen / true_gen.sum(-1, keepdim=True).clamp_min(1e-9)
    predicted = forecast.generation / forecast.generation.sum(-1, keepdim=True).clamp_min(1e-9)
    share_error = (predicted.mean(1) - true_share).abs().mean()
    shortage = (forecast.unserved / forecast.demand.clamp_min(1e-9)).mean()
    crps = fair_crps(forecast.cif, cif)
    energy = energy_score(forecast.cif, cif)
    loss = crps + 0.25 * energy + 50 * share_error + 1000 * shortage
    return loss, {"crps": float(crps.detach()), "energy": float(energy.detach()),
                  "share_mae": float(share_error.detach()), "unserved_fraction": float(shortage.detach())}


def joint_dispatch_loss(forecast, labels, capacity):
    """Proper CIF scores plus joint normalized source-generation energy score.

    This supervises a generation distribution even when multiple dispatch
    mechanisms have the same CIF, without penalizing a noisy sample median.
    The shortage penalty remains a modeling assumption, not a proper score.
    """
    scale = capacity.sum(-1).clamp_min(1.)
    samples = (forecast.generation / scale[:, None, None, None]).flatten(2)
    truth = (labels["generation"] / scale[:, None, None]).flatten(1)
    generation = energy_score(samples, truth)
    crps = fair_crps(forecast.cif, labels["cif"])
    energy = energy_score(forecast.cif, labels["cif"])
    shortage = (forecast.unserved / forecast.demand.clamp_min(1e-9)).mean()
    loss = crps + .25 * energy + 500 * generation + 1000 * shortage
    return loss, {"crps": float(crps.detach()), "energy": float(energy.detach()),
        "generation_energy_score": float(generation.detach()), "unserved_fraction": float(shortage.detach())}


def point_dispatch_loss(forecast, point, labels, capacity, estimator="physical"):
    """MAE-first estimator-aligned supervision plus probability/production losses.

    Componentwise generation summaries are auxiliary statistics, not an
    assertion that the vector of marginal medians is a feasible dispatch.
    Every sampled path continues to pass through the physical decoder.
    """
    if estimator == "physical":
        point_cif, point_generation = point.cif[:, 0], point.generation[:, 0]
    elif estimator in ("mean", "median"):
        summarize = (lambda x: x.mean(1)) if estimator == "mean" else (lambda x: x.quantile(.5, dim=1))
        point_cif, point_generation = summarize(forecast.cif), summarize(forecast.generation)
    else:
        raise ValueError("point estimator must be physical, mean or median")
    point_mae = (point_cif - labels["cif"]).abs().mean()
    # Absolute source production supervision identifies scale hidden by fuel-share ratios.
    production = (point_generation - labels["generation"]).abs()
    production = (production / capacity.sum(-1)[:, None, None].clamp_min(1.)).mean()
    shortage = (forecast.unserved / forecast.demand.clamp_min(1e-9)).mean()
    if estimator == "physical":
        shortage = .5 * (shortage + (point.unserved / point.demand.clamp_min(1e-9)).mean())
    crps = fair_crps(forecast.cif, labels["cif"])
    energy = energy_score(forecast.cif, labels["cif"])
    loss = point_mae + .1 * crps + .025 * energy + 500 * production + 1000 * shortage
    return loss, {"point_mae": float(point_mae.detach()), "crps": float(crps.detach()),
        "energy": float(energy.detach()), "normalized_generation_mae": float(production.detach()),
        "unserved_fraction": float(shortage.detach())}
