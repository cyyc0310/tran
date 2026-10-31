"""Batched public-input inference, independent of training and label loading."""

import numpy as np
import torch

from transcif.data.dataset import stack_inputs


def predict(model, examples, num_scenarios=64, seed=0, batch_size=32,
            latent_scale=1., deterministic=False):
    if not examples:
        raise ValueError("no public examples to predict")
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    outputs = {key: [] for key in ("cif", "generation", "demand", "unserved", "curtailment")}
    explicit_point = model.point_forecast or model.point_estimator != "mean"
    if explicit_point:
        outputs["point_cif"] = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            public = stack_inputs(examples[start:start + batch_size], device)
            forecast = model(public, num_scenarios, generator, latent_scale, deterministic)
            for key in outputs:
                if key != "point_cif":
                    outputs[key].append(getattr(forecast, key).cpu().numpy())
            if explicit_point:
                if model.point_estimator == "physical":
                    point = forecast if deterministic else model(public, 2, deterministic=True)
                    point_value = point.cif[:, 0]
                else:
                    point_value = (forecast.cif.quantile(.5, dim=1) if model.point_estimator == "median"
                                   else forecast.cif.mean(1))
                outputs["point_cif"].append(point_value.cpu().numpy())
    return {key: np.concatenate(values) for key, values in outputs.items()}
