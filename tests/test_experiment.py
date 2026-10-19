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


def test_training_and_forecasts_succeed_without_target_labels(bundle, tmp_path, monkeypatch):
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
    summary = run(manifest, output, steps=1, scenarios=4, score_target=False)
    assert not summary["target_scored"]
    assert summary["metrics"] == {}
    assert (output / "model.pt").exists()
    assert (output / "forecasts.npz").exists()
    assert (output / "summary.json").exists()


def test_manifest_without_target_labels_is_supported(bundle, tmp_path):
    manifest, doc, grids = bundle
    grids["target_mixed"].labels_path.unlink()
    doc["regions"][-1]["labels"] = None
    manifest.write_text(json.dumps(doc))
    summary = run(manifest, tmp_path / "no-target-labels", steps=1, scenarios=4)
    assert not summary["target_scored"] and summary["metrics"] == {}
