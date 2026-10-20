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
from transcif.evaluation.scheduling import scheduling_metrics, mechanism_decisions, score_blocks
from transcif.models.baselines import PublicRidge, PublicBoosting, PublicGenerationBoosting, mix_constant, residual_scenarios
from transcif.models.ensemble import (predict_components, mixture, uncertainty_decomposition,
                                     independent_hour_ablation, save_ensemble)
from transcif.training.fit import predict
from transcif.training.transfer import fit_source_model, fit_mechanisms, support_diagnostics


def run(manifest, output, target=None, steps=250, seed=0, scenarios=64,
        leave_jurisdiction_out=False, device="cpu", score_target=True,
        mechanism_ensemble=False, inner_unit="region", scale_candidates=(.5, 1., 2., 4.),
        training_options=None, tree_baseline=False):
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
    model, training = fit_source_model(data["train"], labels["train"], data["validation"], labels["validation"],
                           {name: grids[name] for name in sources},
                           steps=steps, seed=seed, device=device, **(training_options or {}))
    transfer, inner_models = None, []
    if mechanism_ensemble:
        inner_models, transfer = fit_mechanisms(data["train"], labels["train"],
            data["validation"], labels["validation"], {name: grids[name] for name in sources},
            steps=steps, seed=seed, device=device, unit=inner_unit, scale_candidates=scale_candidates,
            training_options=training_options)
    # Match the total empirical scenario budget across stochastic methods.
    draw_budget = scenarios * (len(inner_models) + 1)
    ridge = PublicRidge().fit(data["train"], labels["train"])
    val_truth = np.stack([y["cif"] for y in labels["validation"]])
    residuals = val_truth - ridge.predict(data["validation"])
    prediction = predict(model, data["test"], draw_budget, seed + 20000)
    ridge_draws = residual_scenarios(ridge.predict(data["test"]), residuals,
                                    data["test"], draw_budget, seed)
    constant = mix_constant(data["test"])
    # Source-derived uncertainty for the constant uses its OWN residuals.
    const_residuals = val_truth - mix_constant(data["validation"])
    constant_draws = residual_scenarios(constant, const_residuals, data["test"], draw_budget, seed)
    metadata = {"accounting": doc["accounting"], "training_track": doc["track"],
                "cost_currency": doc["cost_currency"], "sources": sources, "target_excluded": target,
                "train_until": str(train_until), "validation_until": str(val_until), "seed": seed,
                "training_options": training_options or {}}
    torch.save({**metadata, "schema_version": 1, "fuels": list(FUELS),
                "model_config": model.model_config,
                "state_dict": model.state_dict()}, output / "model.pt")
    ridge_point = ridge.predict(data["test"])
    extra = {"public_ridge_point_cif": ridge_point, "published_mix_point_cif": constant}
    transfer_summary, decisions = None, None
    points = {"public_ridge_source_residuals": ridge_point, "published_mix_source_residuals": constant}
    if "point_cif" in prediction:
        points["probabilistic_dispatch"] = prediction["point_cif"]
    if tree_baseline:
        tree = PublicBoosting(seed).fit(data["train"], labels["train"])
        tree_point = tree.predict(data["test"])
        extra["public_boosting_source_residuals"] = residual_scenarios(tree_point,
            val_truth - tree.predict(data["validation"]), data["test"], draw_budget, seed)
        extra["public_boosting_point_cif"] = tree_point
        points["public_boosting_source_residuals"] = tree_point
        generation_tree = PublicGenerationBoosting(seed).fit(data["train"], labels["train"])
        generation_point = generation_tree.predict(data["test"])
        extra["public_generation_boosting_point_cif"] = generation_point
        extra["public_generation_boosting_source_residuals"] = residual_scenarios(generation_point,
            val_truth - generation_tree.predict(data["validation"]), data["test"], draw_budget, seed)
        points["public_generation_boosting_source_residuals"] = generation_point
    duration = min(4, horizon)
    if transfer is not None:
        components = predict_components([model] + inner_models, data["test"], scenarios,
                                        seed + 40000, transfer["latent_scale"])
        mixed = mixture(components)
        uncertainty = uncertainty_decomposition(components["cif"])
        support = support_diagnostics(data["test"], transfer["support"])
        decisions = mechanism_decisions(components, support["supported"], duration)
        scaled = predict(model, data["test"], draw_budget, seed + 20000,
                         latent_scale=transfer["latent_scale"])
        deterministic = predict(model, data["test"], 2, seed, deterministic=True)
        extra.update({"mechanism_mixture": mixed["cif"], "source_scaled_dispatch": scaled["cif"],
            "independent_hour_ablation": independent_hour_ablation(mixed["cif"], seed),
            "deterministic_dispatch": deterministic["cif"],
            "source_scaled_unserved": scaled["unserved"], "source_scaled_demand": scaled["demand"],
            "deterministic_unserved": deterministic["unserved"],
            "deterministic_demand": deterministic["demand"]})
        if "point_cif" in mixed:
            points.update({"mechanism_mixture": mixed["point_cif"],
                "source_scaled_dispatch": scaled["point_cif"],
                "independent_hour_ablation": mixed["point_cif"],
                "deterministic_dispatch": deterministic["point_cif"]})
            extra["mechanism_point_cif"] = mixed["point_cif"]
        transfer["member_names"] = ["pooled"] + ["leave_" + "+".join(g) for g in transfer["groups"]]
        transfer["decision_policy"] = {"duration_hours": duration, "alpha": .9,
                                       "max_shortage": .01, "ambiguity_set": "finite_mechanisms"}
        save_ensemble(output / "ensemble.pt", [model] + inner_models, metadata, transfer)
        np.savez_compressed(output / "mechanisms.npz", **components, **uncertainty,
            **{"support_" + k: v for k, v in support.items()},
            **{"decision_" + k: v for k, v in decisions.items()})
        transfer_summary = {**transfer, "mechanisms": len(inner_models) + 1,
            "scenarios_per_mechanism": scenarios,
            "unsupported_window_fraction": float((~support["supported"]).mean()),
            "recommendation_rate": float(decisions["accepted"].mean()),
            "mean_pointwise_variance": {k: float(v.mean()) for k, v in uncertainty.items()}}
    np.savez_compressed(output / "forecasts.npz", **prediction, **extra, ridge=ridge_draws,
                        mix_constant=constant_draws,
                        origins=np.array([str(e.origin) for e in data["test"]]))
    metrics = {}
    scored = score_target and grids[target].labels_path is not None
    if scored:
        # Labels enter only here, after all predictions and fitting.
        truth = np.stack([grids[target].labels(e)["cif"] for e in data["test"]])
        distributions = [("probabilistic_dispatch", prediction["cif"]),
                             ("public_ridge_source_residuals", ridge_draws),
                             ("published_mix_source_residuals", constant_draws)]
        distributions += [(name, extra[name]) for name in ("mechanism_mixture", "source_scaled_dispatch",
                            "independent_hour_ablation", "deterministic_dispatch",
                            "public_boosting_source_residuals", "public_generation_boosting_source_residuals") if name in extra]
        for name, draws in distributions:
            metrics[name] = {**forecast_metrics(truth, draws, points.get(name)),
                             "scheduling_mean": scheduling_metrics(truth, draws, min(4, horizon), "mean"),
                             "scheduling_cvar": scheduling_metrics(truth, draws, min(4, horizon), "cvar")}
        if decisions is not None:
            metrics["mechanism_mixture"]["finite_mechanism_policies"] = {
                name: score_blocks(truth, decisions[name], duration, decisions["accepted"])
                for name in ("mixture_mean", "mixture_cvar", "worst_mechanism_mean", "worst_mechanism_cvar")}
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
        "steps": steps, "seed": seed, "scenarios": draw_budget, "training": training,
        "transfer": transfer_summary,
        "tree_baseline": tree_baseline,
        "baseline_training": ({"public_boosting": {"models": 1, "max_iter_per_model": 300,
            "features": 31, "supervision": "source CIF"},
            "public_generation_boosting": {"models": 6, "max_iter_per_model": 300,
            "features": 38, "supervision": "source generation / public capacity",
            "constraint": "capacity bounded, not demand balanced"}} if tree_baseline else {}),
        "model_parameters": sum(p.numel() for p in model.parameters()), "metrics": metrics,
        "target_scored": scored, "unserved_forecast_load_fraction": shortage_fraction,
        "runtime": {name: importlib.metadata.version(name) for name in
                    (("torch", "numpy", "pandas", "scikit-learn") if tree_baseline else ("torch", "numpy", "pandas"))},
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "input_sha256": hashes,
        "source_sha256": {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(source_root.rglob("*.py"))},
        "test_input_provenance": [e.provenance for e in data["test"]],
        "limitations": ["single-bus production model; no imports/storage/ramps/unit commitment",
                        "scheduling scores are average-CIF proxy costs, not avoided emissions",
                        "source-only probabilistic transfer has no distribution-free target coverage guarantee"]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary
