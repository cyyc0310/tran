"""Finite feasible mechanisms: predictions, checkpoint I/O and diagnostics."""

import numpy as np
import torch

from transcif.data.dataset import FUELS
from transcif.models.dispatch import ProbabilisticDispatchNet
from transcif.training.fit import predict


def predict_components(models, examples, scenarios=32, seed=0, latent_scale=1.):
    if not models:
        raise ValueError("at least one mechanism required")
    forecasts = [predict(model, examples, scenarios, seed + 10000 * i, latent_scale=latent_scale)
                 for i, model in enumerate(models)]
    # N,K,S,H[,F]; keep mechanism identity for robust decisions.
    return {key: np.stack([f[key] for f in forecasts], axis=1) for key in forecasts[0]}


def mixture(components):
    """Equal-weight empirical mixture, N,K,S,... -> N,K*S,... ."""
    return {key: values.mean(1) if key == "point_cif" else values.reshape(values.shape[0], -1, *values.shape[3:])
            for key, values in components.items()}


def uncertainty_decomposition(cif):
    cif = np.asarray(cif, float)
    if cif.ndim != 4 or min(cif.shape) < 1 or not np.isfinite(cif).all():
        raise ValueError("finite component CIF (N,K,S,H) required")
    within = cif.var(axis=2).mean(axis=1)
    between = cif.mean(axis=2).var(axis=1)
    return {"within_mechanism_variance": within, "between_mechanism_variance": between,
            "total_variance": within + between}


def independent_hour_ablation(cif, seed=0):
    """Randomly permute scenario indices per hour; exactly preserve marginals.

    This diagnoses dependence at the CIF level, not a new physical simulation.
    A shared mechanism index can still induce between-mechanism dependence.
    """
    cif = np.asarray(cif)
    if cif.ndim not in (3, 4) or min(cif.shape) < 1:
        raise ValueError("CIF (N,S,H) or (N,K,S,H) required")
    rng = np.random.default_rng(seed)
    order = np.argsort(rng.random(cif.shape), axis=-2)
    return np.take_along_axis(cif, order, axis=-2)


def save_ensemble(path, models, metadata, transfer):
    checkpoint = {**metadata, "schema_version": 2, "kind": "dispatch_mechanism_ensemble",
        "fuels": list(FUELS), "transfer": transfer,
        "members": [{"model_config": m.model_config,
                     "state_dict": m.state_dict()} for m in models]}
    torch.save(checkpoint, path)


def load_ensemble(checkpoint, device="cpu"):
    if (checkpoint.get("schema_version") != 2 or
            checkpoint.get("kind") != "dispatch_mechanism_ensemble" or
            checkpoint.get("fuels") != list(FUELS) or not checkpoint.get("members")):
        raise ValueError("incompatible mechanism ensemble checkpoint")
    models = []
    for member in checkpoint["members"]:
        model = ProbabilisticDispatchNet(**member["model_config"]).to(device)
        model.load_state_dict(member["state_dict"])
        model.eval()
        models.append(model)
    return models
