"""Source-only fitting; this module has no target labels or target calibration."""

from copy import deepcopy

import numpy as np
import torch

from transcif.data.dataset import stack_inputs
from transcif.evaluation.metrics import ensemble_crps
from transcif.models.dispatch import ProbabilisticDispatchNet
from transcif.training.losses import dispatch_loss


def predict(model, examples, num_scenarios=64, seed=0, batch_size=32):
    if not examples:
        raise ValueError("no public examples to predict")
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    outputs = {key: [] for key in ("cif", "generation", "demand", "unserved", "curtailment")}
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            forecast = model(stack_inputs(examples[start:start + batch_size], device),
                             num_scenarios, generator)
            for key in outputs:
                outputs[key].append(getattr(forecast, key).cpu().numpy())
    return {key: np.concatenate(values) for key, values in outputs.items()}


def fit(train_examples, train_labels, validation_examples, validation_labels,
        steps=250, seed=0, batch_size=16, scenarios=16, lr=1e-3, device="cpu"):
    if steps < 1 or batch_size < 1 or scenarios < 2 or lr <= 0:
        raise ValueError("positive training options and at least two scenarios required")
    if (not train_examples or not validation_examples or len(train_examples) != len(train_labels)
            or len(validation_examples) != len(validation_labels)):
        raise ValueError("nonempty aligned source training and validation sets required")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ProbabilisticDispatchNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    truth_val = np.stack([y["cif"] for y in validation_labels])
    best, best_step, best_state = float("inf"), 0, None
    history = []
    interval = max(1, min(25, steps // 5))
    for step in range(1, steps + 1):
        model.train()
        indices = rng.choice(len(train_examples), min(batch_size, len(train_examples)), replace=False)
        public = stack_inputs([train_examples[i] for i in indices], device)
        labels = {key: torch.as_tensor(np.stack([train_labels[i][key] for i in indices]),
                                       dtype=torch.float32, device=device)
                  for key in ("cif", "generation")}
        optimizer.zero_grad()
        forecast = model(public, num_scenarios=scenarios)
        loss, diagnostic = dispatch_loss(forecast, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if step % interval == 0 or step == steps:
            validation = predict(model, validation_examples, num_scenarios=32, seed=seed + 10000)
            score = ensemble_crps(truth_val, validation["cif"])
            history.append({"step": step, "source_validation_crps": score, **diagnostic})
            if score < best:
                best, best_step = score, step
                best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return model, {"selected_step": best_step, "source_validation_crps": best, "history": history}
