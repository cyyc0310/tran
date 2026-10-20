from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import transcif.training.transfer as module


def test_group_search_uses_unseen_source_scores_and_keeps_overlap_together(bundle, monkeypatch):
    _, doc, grids = bundle
    sources = {name: grids[name] for name in ("source_coal", "source_gas", "source_wind")}
    sources["source_gas"] = replace(sources["source_gas"], coverage=sources["source_coal"].coverage)
    train = [g.example(doc["origins"][0]) for g in sources.values()]
    val = [g.example(doc["origins"][1]) for g in sources.values()]
    labels = [{"cif": np.full(24, 10.)} for _ in train]
    fitted = []

    def fake_fit(train, labels, val, val_labels, lr, **kwargs):
        fitted.append(({e.region for e in train}, {e.region for e in val}))
        # Temporal validation prefers the wrong candidate for unseen sources.
        return SimpleNamespace(lr=lr), {"selected_step": 1,
            "selection_score": 0. if lr == .001 else 100., "learning_rate": lr}

    def fake_predict(model, examples, scenarios, seed):
        point = np.full((len(examples), 24), 0. if model.lr == .001 else 10.)
        return {"point_cif": point, "cif": np.repeat(point[:, None], scenarios, axis=1)}

    monkeypatch.setattr(module, "fit", fake_fit)
    monkeypatch.setattr(module, "predict", fake_predict)
    model, record = module.fit_source_model(train, labels, val, labels, sources,
        steps=2, objective="point", selection_protocol="source_group", lr_candidates=[.001, .01])
    assert model.lr == .01
    search = record["source_model_selection"]
    assert search["selection_score"] == 0
    assert search["groups"] == [["source_coal", "source_gas"], ["source_wind"]]
    assert search["total_model_fits"] == 5
    assert search["total_training_steps"] == 10
    for candidate in search["candidates"]:
        for fold in candidate["folds"]:
            held = set(fold["held_regions"])
            assert not held.intersection(fold["training_regions"] + fold["checkpoint_regions"])
    assert fitted[-1] == (set(sources), set(sources))


def test_source_group_search_refuses_missing_independent_domains(bundle):
    _, doc, grids = bundle
    grid = grids["source_coal"]
    example = grid.example(doc["origins"][0])
    labels = [grid.labels(example)]
    with pytest.raises(ValueError, match="two independent"):
        module.fit_source_model([example], labels, [example], labels, {grid.name: grid},
                                selection_protocol="source_group")
    with pytest.raises(ValueError, match="declared source"):
        module.fit_source_model([example], labels, [example], labels, {},
                                selection_protocol="source_group")
