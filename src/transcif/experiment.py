"""One explicit cross-grid, forward-time experiment with matched input baselines."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transcif.data.asof import utc
from transcif.data.dataset import FUELS, load_bundle, source_names
from transcif.evaluation.metrics import forecast_metrics
from transcif.evaluation.scheduling import scheduling_metrics
from transcif.models.baselines import PublicRidge, mix_constant, residual_scenarios
from transcif.training.fit import fit, predict


def run(manifest, output, target=None, steps=250, seed=0, scenarios=64,
        leave_jurisdiction_out=False, device="cpu", score_target=True):
    path, output = Path(manifest), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists() or (output / "model.pt").exists():
        raise ValueError("output already contains a run; choose a fresh directory")
    doc, grids = load_bundle(path)
    target = target or doc.get("target")
    if target not in grids:
        raise ValueError("declare an existing target region")
    sources = source_names(grids, target, leave_jurisdiction_out)
    origins = utc(doc["origins"], "origins")
    if origins.has_duplicates or not origins.is_monotonic_increasing:
        raise ValueError("origins must be sorted and unique")
    train_until = utc([doc["train_until"]])[0]
    val_until = utc([doc["validation_until"]])[0]
    if train_until >= val_until:
        raise ValueError("training cutoff must precede validation cutoff")
    horizon = int(doc.get("horizon", 24))
    if horizon < 1:
        raise ValueError("positive horizon required")
    end = origins + pd.Timedelta(hours=horizon)
    masks = {"train": end <= train_until,
             "validation": (origins >= train_until) & (end <= val_until),
             "test": origins >= val_until}
    if not all(mask.any() for mask in masks.values()):
        raise ValueError("empty temporal split")
    data, labels = {}, {}
    for split, names, cutoff in (("train", sources, train_until),
                                  ("validation", sources, val_until)):
        data[split], labels[split] = [], []
        for name in names:
            for origin in origins[masks[split]]:
                example = grids[name].example(origin, horizon)
                # Both the label interval and its publication must precede cutoff.
                observed = grids[name].labels(example, available_by=cutoff)
                data[split].append(example)
                labels[split].append(observed)
    # The outer target contributes PUBLIC INPUTS only until inference finishes.
    data["test"] = [grids[target].example(o, horizon) for o in origins[masks["test"]]]
    model, training = fit(data["train"], labels["train"], data["validation"], labels["validation"],
                           steps=steps, seed=seed, device=device)
    ridge = PublicRidge().fit(data["train"], labels["train"])
    val_truth = np.stack([y["cif"] for y in labels["validation"]])
    residuals = val_truth - ridge.predict(data["validation"])
    prediction = predict(model, data["test"], scenarios, seed + 20000)
    ridge_draws = residual_scenarios(ridge.predict(data["test"]), residuals,
                                    data["test"], scenarios, seed)
    constant = mix_constant(data["test"])
    # Source-derived uncertainty for the constant uses its OWN residuals.
    const_residuals = val_truth - mix_constant(data["validation"])
    constant_draws = residual_scenarios(constant, const_residuals, data["test"], scenarios, seed)
    torch.save({"schema_version": 1, "fuels": list(FUELS),
                "accounting": doc["accounting"], "training_track": doc["track"],
                "cost_currency": doc["cost_currency"],
                "model_config": {"hidden": model.hidden,
                                 "dispatch_temperature": model.dispatch_temperature},
                "state_dict": model.state_dict(), "sources": sources, "target_excluded": target,
                "train_until": str(train_until), "validation_until": str(val_until),
                "seed": seed}, output / "model.pt")
    np.savez_compressed(output / "forecasts.npz", **prediction, ridge=ridge_draws,
                        mix_constant=constant_draws,
                        origins=np.array([str(e.origin) for e in data["test"]]))
    metrics = {}
    scored = score_target and grids[target].labels_path is not None
    if scored:
        # Labels enter only here, after all predictions and fitting.
        truth = np.stack([grids[target].labels(e)["cif"] for e in data["test"]])
        for name, draws in (("probabilistic_dispatch", prediction["cif"]),
                             ("public_ridge_source_residuals", ridge_draws),
                             ("published_mix_source_residuals", constant_draws)):
            metrics[name] = {**forecast_metrics(truth, draws),
                             "scheduling_mean": scheduling_metrics(truth, draws, min(4, horizon), "mean"),
                             "scheduling_cvar": scheduling_metrics(truth, draws, min(4, horizon), "cvar")}
    shortage_fraction = float(
        np.mean(prediction["unserved"] / np.maximum(prediction["demand"], 1e-9)))
    # Fingerprint only declared, used files. Prediction-only runs do not even
    # read target labels for hashing, nor unrelated CSVs in the bundle tree.
    used = set()
    for region in doc["regions"]:
        if region["name"] not in sources + [target]:
            continue
        keys = ["releases", "forecasts"]
        if region["name"] in sources or scored:
            keys.append("labels")
        used.update(path.parent / region[key] for key in keys if region.get(key))
    hashes = {str(p.relative_to(path.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(used)}
    source_root = Path(__file__).parent
    summary = {"schema_version": 1, "track": doc["track"], "accounting": doc["accounting"],
        "cost_currency": doc["cost_currency"],
        "label_kind": {name: grid.label_kind for name, grid in grids.items()},
        "target": target, "sources": sources, "source_coverage": {
            name: sorted(grids[name].coverage) for name in sources},
        "target_coverage": sorted(grids[target].coverage),
        "train_until": str(train_until), "validation_until": str(val_until),
        "n_windows": {key: len(values) for key, values in data.items()},
        "source_regions_only_for_fitting_and_model_selection": True,
        "steps": steps, "seed": seed, "scenarios": scenarios, "training": training,
        "model_parameters": sum(p.numel() for p in model.parameters()), "metrics": metrics,
        "target_scored": scored, "unserved_forecast_load_fraction": shortage_fraction,
        "runtime": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "pandas")},
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "input_sha256": hashes,
        "source_sha256": {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(source_root.rglob("*.py"))},
        "test_input_provenance": [e.provenance for e in data["test"]],
        "limitations": ["single-bus production model; no imports/storage/ramps/unit commitment",
                        "scheduling scores are average-CIF proxy costs, not avoided emissions",
                        "source-only probabilistic transfer has no distribution-free target coverage guarantee"]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary
