"""Source-domain cross-fitting, latent-scale selection and metadata support."""

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
        model, training = fit([train_examples[j] for j in train], [train_labels[j] for j in train],
            [validation_examples[j] for j in val], [validation_labels[j] for j in val],
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
            "within_source_validation_crps": training["source_validation_crps"],
            "held_group_crps_by_scale": fold_scores})
    macro = np.mean(scores, axis=0)
    selected = int(np.argmin(macro))
    support = fit_support(train_examples + validation_examples, groups)
    return models, {"inner_unit": unit, "groups": groups, "folds": folds,
        "scale_candidates": candidates.tolist(), "candidate_group_macro_crps": macro.tolist(),
        "latent_scale": float(candidates[selected]), "selection_scenarios": scenarios,
        "support": support}
