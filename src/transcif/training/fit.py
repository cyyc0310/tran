"""Source-only fitting; this module has no target labels or target calibration."""

from copy import deepcopy
import math

import numpy as np
import torch

from transcif.data.dataset import stack_inputs
from transcif.evaluation.metrics import ensemble_crps
from transcif.models.dispatch import ProbabilisticDispatchNet
from transcif.training.losses import dispatch_loss, point_dispatch_loss


def predict(model, examples, num_scenarios=64, seed=0, batch_size=32,
            latent_scale=1., deterministic=False):
    if not examples:
        raise ValueError("no public examples to predict")
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    outputs = {key: [] for key in ("cif", "generation", "demand", "unserved", "curtailment")}
    if model.point_forecast:
        outputs["point_cif"] = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            forecast = model(stack_inputs(examples[start:start + batch_size], device),
                             num_scenarios, generator, latent_scale, deterministic)
            for key in outputs:
                if key != "point_cif":
                    outputs[key].append(getattr(forecast, key).cpu().numpy())
            if model.point_forecast:
                point = forecast if deterministic else model(
                    stack_inputs(examples[start:start + batch_size], device), 2, deterministic=True)
                outputs["point_cif"].append(point.cif[:, 0].cpu().numpy())
    return {key: np.concatenate(values) for key, values in outputs.items()}


def fit(train_examples, train_labels, validation_examples, validation_labels,
        steps=250, seed=0, batch_size=16, scenarios=16, lr=1e-3, device="cpu",
        objective="crps", selection_metric=None, model_config=None,
        schedule="constant", sampler="uniform"):
    if steps < 1 or batch_size < 1 or scenarios < 2 or lr <= 0:
        raise ValueError("positive training options and at least two scenarios required")
    if (not train_examples or not validation_examples or len(train_examples) != len(train_labels)
            or len(validation_examples) != len(validation_labels)):
        raise ValueError("nonempty aligned source training and validation sets required")
    selection_metric = selection_metric or ("mae" if objective == "point" else "crps")
    if (objective not in ("point", "crps") or selection_metric not in ("mae", "rmse", "crps")
            or schedule not in ("constant", "cosine") or sampler not in ("uniform", "region_balanced")):
        raise ValueError("valid objective, selection_metric, schedule and sampler required")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = {**(model_config or {}), "point_forecast": objective == "point"}
    model = ProbabilisticDispatchNet(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    truth_val = np.stack([y["cif"] for y in validation_labels])
    region_indices = {name: np.array([i for i, e in enumerate(train_examples) if e.region == name])
                      for name in sorted({e.region for e in train_examples})}
    val_regions = sorted({e.region for e in validation_examples})
    best, best_step, best_state = float("inf"), 0, None
    history = []
    interval = max(1, min(25, steps // 5))
    for step in range(1, steps + 1):
        model.train()
        if sampler == "region_balanced":
            names = rng.choice(list(region_indices), batch_size)
            indices = [rng.choice(region_indices[name]) for name in names]
        else:
            indices = rng.choice(len(train_examples), min(batch_size, len(train_examples)), replace=False)
        if schedule == "cosine":
            warmup = min(20, max(1, steps // 10))
            fraction = max(0., (step - warmup) / max(1, steps - warmup))
            rate = min(1., step / warmup) * (.1 + .9 * (1 + math.cos(math.pi * fraction)) / 2)
            optimizer.param_groups[0]["lr"] = lr * rate
        public = stack_inputs([train_examples[i] for i in indices], device)
        labels = {key: torch.as_tensor(np.stack([train_labels[i][key] for i in indices]),
                                       dtype=torch.float32, device=device)
                  for key in ("cif", "generation")}
        optimizer.zero_grad()
        forecast = model(public, num_scenarios=scenarios)
        if objective == "point":
            point = model(public, num_scenarios=2, deterministic=True)
            loss, diagnostic = point_dispatch_loss(forecast, point, labels, public.capacity)
        else:
            loss, diagnostic = dispatch_loss(forecast, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if step % interval == 0 or step == steps:
            validation = predict(model, validation_examples, num_scenarios=32, seed=seed + 10000)
            point_val = validation.get("point_cif", validation["cif"].mean(1))
            val_metrics = {"crps": ensemble_crps(truth_val, validation["cif"]),
                "mae": float(np.abs(truth_val - point_val).mean()),
                "rmse": float(np.square(truth_val - point_val).mean()**.5)}
            # MAE/RMSE checkpoint selection weights source regions equally.
            scores = []
            for name in val_regions:
                mask = np.array([e.region == name for e in validation_examples])
                error = truth_val[mask] - point_val[mask]
                scores.append(float(np.abs(error).mean()) if selection_metric == "mae"
                              else float(np.square(error).mean()**.5))
            score = val_metrics["crps"] if selection_metric == "crps" else float(np.mean(scores))
            history.append({"step": step, **{"source_validation_" + k: v for k, v in val_metrics.items()},
                "selection_score": score, "gradient_norm": float(grad_norm),
                "lr": optimizer.param_groups[0]["lr"], **diagnostic})
            if score < best:
                best, best_step = score, step
                best_state = deepcopy(model.state_dict())
                best_metrics = val_metrics
    model.load_state_dict(best_state)
    model.eval()
    return model, {"selected_step": best_step, **{"source_validation_" + k: v for k, v in best_metrics.items()},
        "selection_metric": selection_metric, "selection_score": best, "objective": objective,
        "model_config": model.model_config, "schedule": schedule, "sampler": sampler,
        "batch_size": batch_size, "training_scenarios": scenarios, "learning_rate": lr, "history": history}
