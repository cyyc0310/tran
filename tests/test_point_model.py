import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from transcif.data.dataset import stack_inputs
from transcif.evaluation.metrics import forecast_metrics
from transcif.models.context import ContextEncoder
from transcif.physics.dispatch import capped_dispatch
from transcif.models.dispatch import ProbabilisticDispatchNet
from transcif.training.fit import fit, predict
from transcif.experiment import run


@pytest.mark.parametrize("load", [1.3, 1.8, 5.])
def test_temperature_implicit_gradient(load):
    args = [torch.tensor(x, dtype=torch.double, requires_grad=True)
            for x in ([1., 2., 1.5], [-.4, .3, .4], load, .3)]
    assert torch.autograd.gradcheck(lambda *x: capped_dispatch(*x), tuple(args),
                                    eps=1e-6, atol=1e-5, rtol=1e-4)


def test_context_sees_full_issued_horizon():
    torch.manual_seed(4)
    encoder = ContextEncoder(38, 16)
    features = torch.randn(1, 48, 38, requires_grad=True)
    encoder(features, torch.randn(1, 21))[0, 0, 0].backward()
    assert features.grad[0, -1].abs().sum() > 0


def test_context_can_represent_below_fifty_percent_availability(bundle):
    _, doc, grids = bundle
    example = grids["source_coal"].example(doc["origins"][0])
    example.values["capacity"][:] = [0, 0, 0, 1000, 0, 0]
    example.values["demand"][:] = 600
    model = ProbabilisticDispatchNet(backbone="context", point_forecast=True)
    with torch.no_grad():
        model.availability_bias[3] = -2
    result = model(stack_inputs([example]), 2, deterministic=True)
    assert (result.generation[..., 3] < 500).all()
    torch.testing.assert_close(result.generation.sum(-1) + result.unserved, result.demand,
                               atol=1e-3, rtol=1e-5)


def test_explicit_point_metric_keeps_probability_scores_separate():
    truth = np.ones((2, 4))
    draws = np.zeros((2, 8, 4))
    metrics = forecast_metrics(truth, draws, point=truth)
    assert metrics["mae"] == metrics["rmse"] == 0
    assert metrics["crps"] == metrics["scenario_mean_mae"] == 1
    assert metrics["point_definition"] == "explicit_point"


@pytest.mark.parametrize("backbone", ["context", "shared"])
def test_point_training_selects_source_mae_and_retains_physical_scenarios(bundle, backbone):
    _, doc, grids = bundle
    grid = grids["source_coal"]
    train = [grid.example(o) for o in doc["origins"][:4]]
    val = [grid.example(o) for o in doc["origins"][4:6]]
    model, result = fit(train, [grid.labels(e) for e in train], val, [grid.labels(e) for e in val],
        steps=3, objective="point", model_config={"backbone": backbone, "hidden": 16},
        schedule="cosine", sampler="region_balanced")
    assert result["selection_metric"] == "mae"
    assert result["selection_score"] == min(h["selection_score"] for h in result["history"])
    assert model.point_forecast
    assert model.raw_temperature.grad is not None and torch.isfinite(model.raw_temperature.grad)
    pred = predict(model, val, 16, seed=3)
    assert pred["point_cif"].shape == (2, 24)
    np.testing.assert_allclose(pred["generation"].sum(-1) + pred["unserved"], pred["demand"],
                               atol=1e-3, rtol=1e-5)
    # Probability temperature/scale changes must not move the supervised point head.
    wide = predict(model, val, 16, seed=3, latent_scale=4.)
    np.testing.assert_array_equal(pred["point_cif"], wide["point_cif"])


@pytest.mark.parametrize("backbone", ["context", "shared"])
def test_point_ensemble_checkpoint_cli_and_target_isolation(bundle, tmp_path, backbone):
    manifest, doc, grids = bundle
    options = {"objective": "point", "model_config": {"backbone": backbone, "hidden": 16},
               "schedule": "cosine", "sampler": "region_balanced"}
    if backbone == "shared":
        options["lr_candidates"] = [.001, .01]
    a, b = tmp_path / "a", tmp_path / "b"
    summary = run(manifest, a, steps=2, scenarios=4, mechanism_ensemble=True,
                  scale_candidates=(1., 2.), training_options=options)
    assert summary["training"]["selection_metric"] == "mae"
    if backbone == "shared":
        search = summary["training"]["learning_rate_search"]
        assert search["total_training_steps"] == 4
        assert summary["training"]["selection_score"] == min(
            candidate["selection_score"] for candidate in search["candidates"])
    assert summary["metrics"]["probabilistic_dispatch"]["point_definition"] == "explicit_point"
    grids["target_mixed"].labels_path.unlink()
    run(manifest, b, steps=2, scenarios=4, mechanism_ensemble=True,
        scale_candidates=(1., 2.), training_options=options, score_target=False)
    with np.load(a / "forecasts.npz") as left, np.load(b / "forecasts.npz") as right:
        assert all(np.array_equal(left[k], right[k]) for k in left)
        assert left["point_cif"].shape[-1] == left["mechanism_point_cif"].shape[-1] == 24
    script = Path(__file__).resolve().parents[1] / "scripts" / "predict.py"
    result = subprocess.run([sys.executable, str(script), "--manifest", str(manifest),
        "--checkpoint", str(a / "ensemble.pt"), "--target", "target_mixed",
        "--origin", doc["origins"][-1], "--scenarios", "4", "--output", str(tmp_path / "pred.json")],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert len(json.loads((tmp_path / "pred.json").read_text())["point_g_per_kwh"]) == 24


def test_tree_reference_runs_without_target_labels(bundle, tmp_path):
    pytest.importorskip("sklearn")
    manifest, _, grids = bundle
    grids["target_mixed"].labels_path.unlink()
    run(manifest, tmp_path / "tree", steps=1, scenarios=4, score_target=False,
        training_options={"objective": "point"}, tree_baseline=True)
    with np.load(tmp_path / "tree" / "forecasts.npz") as predictions:
        assert predictions["public_boosting_point_cif"].shape == (3, 24)
        assert np.isfinite(predictions["public_boosting_source_residuals"]).all()


def test_mae_selected_distribution_exports_median_not_mean(bundle):
    _, doc, grids = bundle
    grid = grids["source_coal"]
    examples = [grid.example(o) for o in doc["origins"][:3]]
    labels = [grid.labels(e) for e in examples]
    model, history = fit(examples[:2], labels[:2], examples[2:], labels[2:],
        steps=1, objective="crps", selection_metric="mae")
    assert not model.point_forecast
    assert history["selection_metric"] == "mae"
    assert model.point_estimator == "median"
    forecast = predict(model, examples[2:], 64, seed=7)
    np.testing.assert_allclose(forecast["point_cif"], np.median(forecast["cif"], axis=1), atol=1e-4)
    assert model.model_config["point_estimator"] == "median"
