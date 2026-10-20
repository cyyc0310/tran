import numpy as np
import pytest
import json
from pathlib import Path
import subprocess
import sys

from transcif.evaluation.comparison import paired_block_comparison
from transcif.benchmark import benchmark


def test_identical_methods_have_zero_paired_interval_without_independent_hour_resampling():
    errors = np.random.default_rng(3).normal(size=(3, 2, 14, 24))
    result = paired_block_comparison(errors, errors, block_windows=7, draws=100)
    for metric in result["metrics"].values():
        assert metric["candidate_minus_reference"] == 0
        assert metric["percentile_95_interval"] == [0, 0]


def test_country_and_seed_aggregation_matches_macro_metric_definition():
    # Two grids in country A and one in B; seeds are averaged AFTER RMSE.
    errors = np.array([[1., 3.], [1., 3.], [10., 14.]])[:, :, None, None] * np.ones((1, 1, 14, 24))
    result = paired_block_comparison(np.zeros_like(errors), errors,
                                    groups=["A", "A", "B"], block_windows=7, draws=100)
    assert result["independent_groups"] == 2
    assert result["training_seeds"] == 2
    for metric in result["metrics"].values():
        assert metric["candidate"] == 7  # mean(country A=2, country B=12)
        assert metric["reference"] == 0
        assert metric["candidate_minus_reference"] == 7
    with pytest.raises(ValueError, match="block length"):
        paired_block_comparison(errors, errors, block_windows=15)


def test_comparison_cli_matches_benchmark_and_rejects_changed_truth(bundle, tmp_path):
    manifest, _, grids = bundle
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    targets = ["source_coal", "source_gas"]
    a = benchmark(manifest, reference, targets, seeds=[0], steps=1, scenarios=4)
    b = benchmark(manifest, candidate, targets, seeds=[0], steps=1, scenarios=4,
                  training_options={"objective": "point", "model_config": {"point_estimator": "median"}})
    script = Path(__file__).resolve().parents[1] / "scripts" / "compare_benchmarks.py"
    output = tmp_path / "comparison.json"
    cmd = [sys.executable, str(script), "--reference", str(reference), "--candidate", str(candidate),
           "--manifest", str(manifest), "--block-windows", "2", "--draws", "100", "--output", str(output)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    comparison = json.loads(output.read_text())
    for metric, values in comparison["metrics"].items():
        assert values["reference"] == pytest.approx(a["target_macro"]["probabilistic_dispatch"][metric], abs=1e-4)
        assert values["candidate"] == pytest.approx(b["target_macro"]["probabilistic_dispatch"][metric], abs=1e-4)
    labels = grids[targets[0]].labels_path
    labels.write_text(labels.read_text() + "\n")
    cmd[-1] = str(tmp_path / "changed.json")
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0
    assert "input file differs" in result.stderr
