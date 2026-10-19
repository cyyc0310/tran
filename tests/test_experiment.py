import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from transcif.experiment import run


def test_target_label_poisoning_cannot_change_training_or_predictions(bundle, tmp_path):
    manifest, doc, grids = bundle
    first = tmp_path / "first"
    summary = run(manifest, first, steps=2, scenarios=8, seed=3)
    assert summary["track"] == "synthetic"
    assert summary["sources"] == ["source_coal", "source_gas", "source_wind"]
    assert summary["training"]["selected_step"] in (1, 2)
    assert summary["n_windows"] == {"train": 21, "validation": 6, "test": 3}
    target = grids["target_mixed"]
    labels = pd.read_csv(target.labels_path)
    labels.cif_g_per_kwh = 100000.
    labels.to_csv(target.labels_path, index=False)
    second = tmp_path / "second"
    poisoned = run(manifest, second, steps=2, scenarios=8, seed=3)
    a = torch.load(first / "model.pt", weights_only=True)["state_dict"]
    b = torch.load(second / "model.pt", weights_only=True)["state_dict"]
    assert all(torch.equal(a[key], b[key]) for key in a)
    with np.load(first / "forecasts.npz") as a, np.load(second / "forecasts.npz") as b:
        assert all(np.array_equal(a[key], b[key]) for key in a)
    assert poisoned["metrics"]["probabilistic_dispatch"]["mae"] > 90000
    assert summary["training"] == poisoned["training"]
    with pytest.raises(ValueError, match="already contains"):
        run(manifest, first, steps=1)

    # Actual CLI inference must work after the target labels have been deleted.
    target.labels_path.unlink()
    output = tmp_path / "prediction.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "predict.py"
    result = subprocess.run([sys.executable, str(script), "--checkpoint", str(first / "model.pt"),
        "--manifest", str(manifest), "--target", target.name,
        "--origin", doc["origins"][-1], "--scenarios", "8", "--output", str(output)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    prediction = json.loads(output.read_text())
    assert np.array(prediction["scenarios"]).shape == (8, 24)
    assert prediction["training_track"] == prediction["input_track"] == "synthetic"


@pytest.mark.parametrize("use_ensemble", [False, True])
def test_training_and_forecasts_succeed_without_target_labels(bundle, tmp_path, monkeypatch, use_ensemble):
    manifest, _, grids = bundle
    target_labels = grids["target_mixed"].labels_path
    # Also forbid an incidental raw read for hashing while the file exists.
    original = Path.read_bytes
    def guarded_read(path):
        if path == target_labels:
            raise AssertionError("prediction-only run opened target labels")
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    output = tmp_path / "unscored"
    summary = run(manifest, output, steps=1, scenarios=4, score_target=False,
                  mechanism_ensemble=use_ensemble, scale_candidates=(1., 2.))
    assert not summary["target_scored"]
    assert summary["metrics"] == {}
    assert (output / "model.pt").exists()
    assert (output / "forecasts.npz").exists()
    assert (output / "summary.json").exists()


def test_ensemble_target_poisoning_and_label_free_cli(bundle, tmp_path):
    manifest, doc, grids = bundle
    first, second = tmp_path / "a", tmp_path / "b"
    summary = run(manifest, first, steps=2, scenarios=4, seed=8,
                  mechanism_ensemble=True, scale_candidates=(1., 2.))
    assert summary["scenarios"] == 16
    transfer = summary["transfer"]
    assert transfer["latent_scale"] in (1., 2.)
    assert transfer["mechanisms"] == 4
    for fold in transfer["folds"]:
        assert not (set(fold["held_regions"]) & set(fold["training_regions"]))
        assert not (set(fold["held_regions"]) & set(fold["selection_regions"]))
        assert "target_mixed" not in fold["held_regions"] + fold["training_regions"] + fold["selection_regions"]
    a = torch.load(first / "ensemble.pt", weights_only=True)
    labels = pd.read_csv(grids["target_mixed"].labels_path)
    labels.cif_g_per_kwh = 100000.
    labels.to_csv(grids["target_mixed"].labels_path, index=False)
    poisoned = run(manifest, second, steps=2, scenarios=4, seed=8,
                   mechanism_ensemble=True, scale_candidates=(1., 2.))
    b = torch.load(second / "ensemble.pt", weights_only=True)
    assert a["transfer"] == b["transfer"]
    for ma, mb in zip(a["members"], b["members"]):
        assert all(torch.equal(ma["state_dict"][key], mb["state_dict"][key]) for key in ma["state_dict"])
    for file in ("forecasts.npz", "mechanisms.npz"):
        with np.load(first / file) as left, np.load(second / file) as right:
            assert all(np.array_equal(left[k], right[k]) for k in left)
    assert poisoned["metrics"]["mechanism_mixture"]["mae"] > 90000
    metric = summary["metrics"]
    assert metric["mechanism_mixture"]["crps"] == pytest.approx(metric["independent_hour_ablation"]["crps"])
    assert metric["mechanism_mixture"]["mean_90_width"] == pytest.approx(
        metric["independent_hour_ablation"]["mean_90_width"])
    grids["target_mixed"].labels_path.unlink()
    output = tmp_path / "ensemble-inference.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "predict.py"
    result = subprocess.run([sys.executable, str(script), "--checkpoint", str(first / "ensemble.pt"),
        "--manifest", str(manifest), "--target", "target_mixed", "--origin", doc["origins"][-1],
        "--scenarios", "4", "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    prediction = json.loads(output.read_text())
    assert np.shape(prediction["scenarios"]) == (16, 24)
    assert len(prediction["schedule_proposals"]) == 4
    assert all((p["recommended_start_hour"] is not None) == prediction["recommendation_accepted"]
               for p in prediction["schedule_proposals"].values())


@pytest.mark.parametrize("rates", [None, [.001, .01]])
def test_inner_held_source_never_changes_its_own_model_or_checkpoint_selection(bundle, tmp_path, rates):
    manifest, _, grids = bundle
    first, second = tmp_path / "before", tmp_path / "poisoned-source"
    run(manifest, first, steps=2, scenarios=4, seed=4,
        mechanism_ensemble=True, scale_candidates=(1., 2.), training_options={"lr_candidates": rates})
    coal = grids["source_coal"].labels_path
    labels = pd.read_csv(coal)
    labels.cif_g_per_kwh = 100000.
    labels.to_csv(coal, index=False)
    run(manifest, second, steps=2, scenarios=4, seed=4,
        mechanism_ensemble=True, scale_candidates=(1., 2.), training_options={"lr_candidates": rates})
    a, b = (torch.load(p / "ensemble.pt", weights_only=True) for p in (first, second))
    index = a["transfer"]["member_names"].index("leave_source_coal")
    left, right = a["members"][index]["state_dict"], b["members"][index]["state_dict"]
    assert all(torch.equal(left[k], right[k]) for k in left)
    assert a["transfer"]["folds"][index - 1]["selected_step"] == b["transfer"]["folds"][index - 1]["selected_step"]
    for key in ("learning_rate", "learning_rate_search"):
        assert a["transfer"]["folds"][index - 1].get(key) == b["transfer"]["folds"][index - 1].get(key)


def test_manifest_without_target_labels_is_supported(bundle, tmp_path):
    manifest, doc, grids = bundle
    grids["target_mixed"].labels_path.unlink()
    doc["regions"][-1]["labels"] = None
    manifest.write_text(json.dumps(doc))
    summary = run(manifest, tmp_path / "no-target-labels", steps=1, scenarios=4)
    assert not summary["target_scored"] and summary["metrics"] == {}
