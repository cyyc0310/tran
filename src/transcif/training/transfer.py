"""Source-domain cross-fitting, latent-scale selection and metadata support."""

import json

import numpy as np

from transcif.evaluation.metrics import ensemble_crps
from transcif.training.fit import fit, predict


def independent_groups(grids, unit="region"):
    """Connected components of overlapping coverage, optionally entire countries.

    Transitive overlap is kept together, so an aggregate cannot bridge folds.
    The caller passes SOURCE grids only.
    """
    if unit not in ("region", "jurisdiction"):
        raise ValueError("inner unit must be region or jurisdiction")
    remaining, groups = set(grids), []
    while remaining:
        group, frontier = set(), [min(remaining)]
        while frontier:
            name = frontier.pop()
            if name in group:
                continue
            group.add(name)
            remaining.discard(name)
            for other in sorted(remaining):
                if (grids[name].coverage & grids[other].coverage or
                    (unit == "jurisdiction" and
                     grids[name].jurisdiction == grids[other].jurisdiction)):
                    frontier.append(other)
        groups.append(sorted(group))
    return groups


def _held_metrics(model, examples, labels, seed, scenarios=32):
    forecast = predict(model, examples, scenarios, seed)
    point = forecast.get("point_cif", forecast["cif"].mean(1))
    truth = np.stack([label["cif"] for label in labels])
    scores = []
    for region in sorted({example.region for example in examples}):
        mask = np.array([example.region == region for example in examples])
        error = truth[mask] - point[mask]
        scores.append({"mae": float(np.abs(error).mean()),
                       "rmse": float(np.square(error).mean() ** .5),
                       "crps": ensemble_crps(truth[mask], forecast["cif"][mask])})
    return {key: float(np.mean([score[key] for score in scores])) for key in scores[0]}


def fit_source_model(train_examples, train_labels, validation_examples, validation_labels,
                     source_grids, steps=250, seed=0, device="cpu",
                     selection_protocol="temporal", selection_unit="region",
                     model_candidates=None, **options):
    """Choose public model configuration/LR without any outer-target labels.

    In source_group mode, each candidate is checkpointed on other sources'
    temporal validation, then scored on a held independent source group. The
    winning hyperparameters are refit on all supplied sources. A mechanism's
    permanently held group must be excluded from source_grids and both splits.
    """
    if selection_protocol not in ("temporal", "source_group"):
        raise ValueError("selection_protocol must be temporal or source_group")
    if selection_protocol == "temporal" and model_candidates is None:
        return fit(train_examples, train_labels, validation_examples, validation_labels,
                   steps=steps, seed=seed, device=device, **options)
    if (not train_examples or not validation_examples or
            len(train_examples) != len(train_labels) or len(validation_examples) != len(validation_labels)):
        raise ValueError("nonempty aligned source splits required")
    if any(e.region not in source_grids for e in train_examples + validation_examples):
        raise ValueError("model selection accepts declared source regions only")
    groups = independent_groups(source_grids, selection_unit) if selection_protocol == "source_group" else []
    if selection_protocol == "source_group" and len(groups) < 2:
        raise ValueError("source-group selection needs at least two independent source groups")
    base = options.get("model_config") or {}
    configs = [base] if model_candidates is None else [base | cfg for cfg in model_candidates]
    if not configs or len({json.dumps(cfg, sort_keys=True, allow_nan=False) for cfg in configs}) != len(configs):
        raise ValueError("nonempty unique model candidates required")
    rates = options.pop("lr_candidates", None)
    rates = [options.get("lr", 1e-3)] if rates is None else list(rates)
    if not rates or len(set(rates)) != len(rates) or any(not np.isfinite(r) or r <= 0 for r in rates):
        raise ValueError("nonempty unique finite positive learning-rate candidates required")
    metric = options.get("selection_metric") or ("mae" if options.get("objective") in ("point", "joint") else "crps")
    if metric not in ("mae", "rmse", "crps"):
        raise ValueError("valid source selection metric required")
    folds = []
    for held in groups:
        train = [i for i, e in enumerate(train_examples) if e.region not in held]
        val = [i for i, e in enumerate(validation_examples) if e.region not in held]
        test = [i for i, e in enumerate(validation_examples) if e.region in held]
        if not train or not val or not test or any(
                not any(validation_examples[i].region == name for i in test) for name in held):
            raise ValueError("each source-selection fold needs fit, checkpoint and held-group windows")
        folds.append((held, train, val, test))
    records, best_model, best_training = [], None, None
    for config in configs:
        for rate in rates:
            candidate_options = {**options, "model_config": config, "lr": rate}
            if selection_protocol == "temporal":
                model, training = fit(train_examples, train_labels, validation_examples, validation_labels,
                    steps=steps, seed=seed, device=device, **candidate_options)
                score, fold_records = training["selection_score"], []
                if best_training is None or score < best_training["selection_score"]:
                    best_model, best_training = model, training
            else:
                fold_records = []
                for fold, (held, train, val, test) in enumerate(folds):
                    model, training = fit([train_examples[i] for i in train], [train_labels[i] for i in train],
                        [validation_examples[i] for i in val], [validation_labels[i] for i in val],
                        steps=steps, seed=seed + 1000 * (fold + 1), device=device, **candidate_options)
                    held_scores = _held_metrics(model, [validation_examples[i] for i in test],
                        [validation_labels[i] for i in test], seed + 50000 + fold)
                    fold_records.append({"held_regions": held,
                        "training_regions": sorted({train_examples[i].region for i in train}),
                        "checkpoint_regions": sorted({validation_examples[i].region for i in val}),
                        "selected_step": training["selected_step"],
                        "checkpoint_score": training["selection_score"],
                        "checkpoint_history": training.get("history", []), "held_metrics": held_scores})
                score = float(np.mean([record["held_metrics"][metric] for record in fold_records]))
            records.append({"model_config": config, "learning_rate": rate,
                            "selection_score": score, "folds": fold_records,
                            **({"history": training.get("history", [])} if not groups else {})})
    selected = int(np.argmin([record["selection_score"] for record in records]))
    if selection_protocol == "source_group":
        winner = records[selected]
        best_model, best_training = fit(train_examples, train_labels, validation_examples, validation_labels,
            steps=steps, seed=seed, device=device,
            **{**options, "model_config": winner["model_config"], "lr": winner["learning_rate"]})
    fits = len(records) * (len(groups) if groups else 1) + (1 if groups else 0)
    best_training["source_model_selection"] = {"protocol": selection_protocol,
        "grouping": selection_unit, "metric": metric, "groups": groups,
        "selected_candidate": selected, "selection_score": records[selected]["selection_score"],
        "candidates": records, "selection_scenarios": 32,
        "total_model_fits": fits, "total_training_steps": fits * steps,
        "limitation": "source-selected score is not an unbiased outer-target performance estimate"}
    return best_model, best_training


def metadata_vector(example):
    x = example.values
    return np.r_[x["capacity"] / x["capacity"].sum(), x["mix"]].astype(float)


def fit_support(examples, groups):
    """Reference configuration distance, using public SOURCE metadata only.

    The cutoff is the maximum group-wise 95th percentile of distances to other
    independent groups. This is a descriptive support heuristic, not a test
    with a guaranteed false-positive rate.
    """
    if len(groups) < 2:
        raise ValueError("at least two independent source groups required")
    vectors, group_ids = [], []
    for i, group in enumerate(groups):
        rows = [metadata_vector(e) for e in examples if e.region in group]
        if not rows:
            raise ValueError("each support group needs public examples")
        unique = np.unique(np.stack(rows), axis=0)
        vectors.extend(unique)
        group_ids.extend([i] * len(unique))
    vectors, ids = np.asarray(vectors), np.asarray(group_ids)
    cutoffs = []
    for i in range(len(groups)):
        distances = np.linalg.norm(vectors[ids == i, None] - vectors[None, ids != i], axis=-1) / 2**.5
        cutoffs.append(float(np.quantile(distances.min(-1), .95)))
    return {"kind": "capacity_and_mix_l2", "groups": groups,
            "vectors": vectors.tolist(), "group_ids": ids.tolist(),
            "distance_cutoff": max(max(cutoffs), 1e-6),
            "source_group_distance_p95": cutoffs}


def support_diagnostics(examples, support):
    if support["kind"] != "capacity_and_mix_l2":
        raise ValueError("unknown metadata support metric")
    vectors = np.stack([metadata_vector(e) for e in examples])
    references = np.asarray(support["vectors"], float)
    distances = np.linalg.norm(vectors[:, None] - references[None], axis=-1) / 2**.5
    nearest = distances.argmin(-1)
    distance = distances[np.arange(len(examples)), nearest]
    ratio = distance / support["distance_cutoff"]
    return {"distance": distance, "distance_ratio": ratio, "supported": ratio <= 1.,
            "nearest_source_group": np.asarray(support["group_ids"])[nearest]}


def fit_mechanisms(train_examples, train_labels, validation_examples, validation_labels,
                   source_grids, steps=250, seed=0, device="cpu", unit="region",
                   scale_candidates=(.5, 1., 2., 4.), scenarios=32, training_options=None):
    """Each held source group is absent from fitting AND checkpoint selection.

    Its forward-time validation labels are then used to select one shared
    latent scale across folds. Those diagnostics are selection scores, not
    unbiased estimates of performance after scale selection.
    """
    candidates = np.asarray(scale_candidates, dtype=float)
    if (candidates.ndim != 1 or len(candidates) < 1 or not np.isfinite(candidates).all()
            or (candidates <= 0).any() or len(np.unique(candidates)) != len(candidates)):
        raise ValueError("unique finite positive scale candidates required")
    if (len(train_examples) != len(train_labels) or
            len(validation_examples) != len(validation_labels) or scenarios < 2):
        raise ValueError("aligned source examples/labels and scenarios>=2 required")
    if any(e.region not in source_grids for e in train_examples + validation_examples):
        raise ValueError("cross-fitting accepts declared source regions only")
    groups = independent_groups(source_grids, unit)
    if len(groups) < 2:
        raise ValueError("mechanism ensemble needs at least two independent source groups")
    models, folds, scores = [], [], []
    for i, held in enumerate(groups):
        train = [j for j, e in enumerate(train_examples) if e.region not in held]
        val = [j for j, e in enumerate(validation_examples) if e.region not in held]
        test = [j for j, e in enumerate(validation_examples) if e.region in held]
        if not train or not val or not test:
            raise ValueError("each inner fold needs training, selection and held-group windows")
        model, training = fit_source_model([train_examples[j] for j in train], [train_labels[j] for j in train],
            [validation_examples[j] for j in val], [validation_labels[j] for j in val],
            {name: grid for name, grid in source_grids.items() if name not in held},
            steps=steps, seed=seed + 1000 * (i + 1), device=device, **(training_options or {}))
        held_examples = [validation_examples[j] for j in test]
        truth = np.stack([validation_labels[j]["cif"] for j in test])
        fold_scores = []
        for scale in candidates:
            # Common random numbers across candidate scales within the fold.
            draws = predict(model, held_examples, scenarios, seed + 30000 + i,
                            latent_scale=float(scale))["cif"]
            region_scores = []
            for name in held:
                mask = np.array([e.region == name for e in held_examples])
                if not mask.any():
                    raise ValueError("every held source region needs validation windows")
                region_scores.append(ensemble_crps(truth[mask], draws[mask]))
            fold_scores.append(float(np.mean(region_scores)))
        models.append(model)
        scores.append(fold_scores)
        folds.append({"held_regions": held,
            "training_regions": sorted({train_examples[j].region for j in train}),
            "selection_regions": sorted({validation_examples[j].region for j in val}),
            "selected_step": training["selected_step"],
            "selection_metric": training["selection_metric"],
            "selection_score": training["selection_score"],
            "learning_rate": training["learning_rate"],
            **({"learning_rate_search": training["learning_rate_search"]}
               if "learning_rate_search" in training else {}),
            **({"source_model_selection": training["source_model_selection"]}
               if "source_model_selection" in training else {}),
            "within_source_validation_crps": training["source_validation_crps"],
            "held_group_crps_by_scale": fold_scores})
    macro = np.mean(scores, axis=0)
    selected = int(np.argmin(macro))
    support = fit_support(train_examples + validation_examples, groups)
    return models, {"inner_unit": unit, "groups": groups, "folds": folds,
        "scale_candidates": candidates.tolist(), "candidate_group_macro_crps": macro.tolist(),
        "latent_scale": float(candidates[selected]), "selection_scenarios": scenarios,
        "support": support}
