import json

import pytest

import transcif.benchmark as module


def test_benchmark_averages_seeds_then_targets_and_jurisdictions(bundle, tmp_path, monkeypatch):
    manifest, doc, _ = bundle
    doc["regions"][1]["jurisdiction"] = doc["regions"][0]["jurisdiction"]
    manifest.write_text(json.dumps(doc))
    targets = ["source_coal", "source_gas", "source_wind"]
    values = dict(zip(targets, [10., 30., 100.]))
    calls = []
    def fake_run(manifest, folder, target, seed, **options):
        calls.append((target, seed))
        return {"target_scored": True, "transfer": None,
                "metrics": {"model": {k: values[target] + seed for k in module.METRICS}}}
    monkeypatch.setattr(module, "run", fake_run)
    result = module.benchmark(manifest, tmp_path / "benchmark", targets, seeds=[0, 2], steps=1)
    assert len(calls) == 6
    assert result["per_target_seed_average"]["source_coal"]["model"]["crps"] == 11
    assert result["target_macro"]["model"]["crps"] == pytest.approx(143 / 3)
    assert result["jurisdiction_macro"]["model"]["crps"] == 61
    assert (tmp_path / "benchmark" / "benchmark.json").exists()
    with pytest.raises(ValueError, match="empty"):
        module.benchmark(manifest, tmp_path / "benchmark", targets, seeds=[0])


def test_benchmark_rejects_duplicate_or_overlapping_targets(bundle, tmp_path):
    manifest, doc, _ = bundle
    with pytest.raises(ValueError, match="unique"):
        module.benchmark(manifest, tmp_path / "duplicate", ["source_coal"] * 2)
    doc["regions"][1]["coverage"] = doc["regions"][0]["coverage"]
    manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="disjoint coverage"):
        module.benchmark(manifest, tmp_path / "overlap", ["source_coal", "source_gas"])
