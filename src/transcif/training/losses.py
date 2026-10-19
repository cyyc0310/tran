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
    spread = torch.cdist(samples, samples).sum() / (samples.shape[0] * s * (s - 1))
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


def point_dispatch_loss(forecast, point, labels, capacity):
    """MAE-first physical point head; probability and source generation auxiliary tasks."""
    point_mae = (point.cif[:, 0] - labels["cif"]).abs().mean()
    # Absolute source production supervision identifies scale hidden by fuel-share ratios.
    production = (point.generation[:, 0] - labels["generation"]).abs()
    production = (production / capacity.sum(-1)[:, None, None].clamp_min(1.)).mean()
    shortage = .5 * ((point.unserved / point.demand.clamp_min(1e-9)).mean() +
                      (forecast.unserved / forecast.demand.clamp_min(1e-9)).mean())
    crps = fair_crps(forecast.cif, labels["cif"])
    energy = energy_score(forecast.cif, labels["cif"])
    loss = point_mae + .1 * crps + .025 * energy + 500 * production + 1000 * shortage
    return loss, {"point_mae": float(point_mae.detach()), "crps": float(crps.detach()),
        "energy": float(energy.detach()), "normalized_generation_mae": float(production.detach()),
        "unserved_fraction": float(shortage.detach())}
