"""V5 checks for frozen point bytes, seven-source calibration and global sealing."""
import importlib.util
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location("risk_v5_runner", Path(__file__).parents[1] / "scripts/run_risk_transfer_v5.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class FakeModel:
    def predict(self, origins, region, monthly):
        n = len(origins)
        return {method: {"point": np.zeros((n, 24)), "scale": np.ones((n, 24)), "age": np.full(n, 30.),
            "level_scale": np.ones(n), "shape_scale": np.ones((n, 24))} for method in runner.POINTS}


FIT_CALLS = []


class FakeCalibrator:
    def __init__(self, source_regions, alpha):
        assert tuple(source_regions) == runner.SOURCES

    def fit(self, point, truth, scales, regions, ages, months, years, **kwargs):
        assert set(regions) == set(runner.SOURCES)
        assert set(years) == {2023}
        assert set(ages) == {"le60"}
        FIT_CALLS.append(set(regions))
        return self

    def intervals(self, point, scales, mode, **kwargs):
        return point - scales, point + scales

    def ensemble(self, point, scales, mode, n_samples, seed, **kwargs):
        return np.repeat(point[:, None], n_samples, axis=1)


def fake_inputs(tmp_path):
    root, data, baseline = (tmp_path / name for name in ("repo", "source_data", "old_v4_source"))
    (root / "src/transcif/research").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "docs").mkdir()
    for name in runner.BASE_DEPENDENCIES:
        (root / name).write_text("original point dependency " + name)
    (root / "src/transcif/research/risk_transfer_v5.py").write_text("new risk module")
    for filename in ("run_transfer_v4.py", "run_risk_transfer_v5.py"):
        (root / "scripts" / filename).write_text("frozen runner " + filename)
    (root / runner.PROTOCOL).write_text("frozen V5 protocol")
    data.mkdir()
    (data / "monthly.csv").write_text("region,total_mwh\nBPAT,1\n")
    for region in runner.SOURCES:
        (data / f"{region}_hourly.csv").write_text("time,region,value\n2023-01-01T00:00:00Z," + region + ",1\n2024-01-01T00:00:00Z," + region + ",2\n2025-01-01T00:00:00Z," + region + ",FORBIDDEN_SOURCE_2025\n")
    runner.V4.freeze_inputs(root, data, baseline)
    (baseline / "fit").mkdir()
    records = []
    for delay in runner.DELAYS:
        name = f"fit/round2-{delay}.pkl"
        with (baseline / name).open("wb") as handle:
            pickle.dump({"model": FakeModel(), "calibrators": "must never be reused"}, handle)
        records.append({"variant": "round2", "delay": delay, "path": name})
    runner.V4.write_json(baseline / "run_config.json", {"fit_regions": list(runner.V4.FIT_REGIONS), "fit_year": 2022, "records": records})
    artifacts = {record["path"]: runner.V4.sha(baseline / record["path"]) for record in records}
    artifacts["run_config.json"] = runner.V4.sha(baseline / "run_config.json")
    runner.V4.write_json(baseline / "fit_seal.json", {"prefit_sha256": runner.V4.sha(baseline / "prefit_freeze.json"), "artifacts": artifacts})
    return root, data, baseline


def test_original_point_dependencies_are_guarded_but_new_risk_module_is_allowed(tmp_path):
    root, _, baseline = fake_inputs(tmp_path)
    runner.verify_base(baseline, root)
    (root / "src/transcif/research/risk_transfer_v5.py").write_text("new V5 risk design")
    runner.verify_base(baseline, root)  # New risk code is frozen separately in V5.
    (root / runner.BASE_DEPENDENCIES[0]).write_text("changed V4 point model")
    with pytest.raises(ValueError, match="Original point-model dependency changed"):
        runner.verify_base(baseline, root)


def test_risk_transform_must_preserve_point_and_inputs(monkeypatch):
    origins = pd.date_range("2024-01-01", periods=1, tz="UTC")
    point = FakeModel().predict(origins, "BPAT", None)[runner.POINTS[0]]
    def illegal(*args, **kwargs):
        result = args[3]
        result["point"] += 1
        return result
    monkeypatch.setitem(sys.modules, "transcif.research.risk_transfer_v5", SimpleNamespace(risk_scales=illegal))
    with pytest.raises(ValueError, match="may not alter"):
        runner.transform(origins, "BPAT", None, point, 0, "adaptive_component110_beta1")
    assert not point["point"].any()


def test_seven_source_calibration_then_all_new_targets_sealed_before_labels(tmp_path, monkeypatch):
    FIT_CALLS.clear()
    root, data, baseline = fake_inputs(tmp_path)
    def risk_scales(origins, region, monthly, prediction, extra_delay, beta):
        assert region in runner.SOURCES + runner.HOLDOUTS
        result = {key: np.asarray(value).copy() for key, value in prediction.items()}
        for key in ("scale", "level_scale", "shape_scale"):
            result[key] *= beta
        return result
    monkeypatch.setitem(sys.modules, "transcif.research.risk_transfer_v5", SimpleNamespace(risk_scales=risk_scales))
    monkeypatch.setitem(sys.modules, "transcif.research.calibration_v4", SimpleNamespace(SourceCalibrator=FakeCalibrator))
    monkeypatch.setitem(sys.modules, "transcif.research.component_calibration_v4", SimpleNamespace(ComponentCalibrator=FakeCalibrator))
    # Avoid coupling the end-to-end guard test to the expensive original models.
    monkeypatch.setattr(runner, "base_points", lambda model, origins, region, monthly: model.predict(origins, region, monthly))
    def profiles(frame):
        truth = np.zeros((len(frame), 24))
        truth[:, :3] = 1.
        return pd.DatetimeIndex(frame.time), truth
    monkeypatch.setattr(runner.V4, "daily_profiles", profiles)
    original_read = runner.V4.read_year
    parsed = []
    active_output = [tmp_path / "source_v5"]
    def guarded_read(path, year):
        if year >= 2024:
            seal = json.loads((active_output[0] / "all_predictions_sealed.json").read_text())
            assert len(seal["entries"]) == (315 if year == 2024 else 180)
        parsed.append((Path(path).stem, year))
        return original_read(path, year)
    monkeypatch.setattr(runner.V4, "read_year", guarded_read)
    source = active_output[0]
    config = runner.fit_phase(data, baseline, source, root=root, n_samples=4)
    assert len(FIT_CALLS) == 45
    assert config["calibration_regions"] == list(runner.SOURCES)
    assert config["calibration_year"] == 2023
    assert all(year == 2023 for _, year in parsed)
    assert len(parsed) == 7
    runner.predict_seal_phase(source, end="2024-01-01")
    assert len(parsed) == 7
    runner.score_phase(source)
    assert sum(year == 2024 for _, year in parsed) == 7
    selection = source / "scores/selected_config.json"
    selected = runner.V4.load_source_selection(selection)
    assert selected["status"] == "passed"
    assert selected["selection_regions"] == list(runner.SOURCES)
    assert "delay" not in selected["configuration"]
    external = tmp_path / "new_four"
    external.mkdir()
    (external / "monthly.csv").write_text("region,total_mwh\nSCL,1\n")
    for region in runner.HOLDOUTS:
        (external / f"{region}_hourly.csv").write_text(f"time,region,value\n2025-01-01T00:00:00Z,{region},3\n")
    runner.V4.write_json(external / "audit.json", {"external_regions": list(runner.HOLDOUTS),
        "final_external_year": 2025, "cif_scores_inspected": False, "cif_summaries_computed": False,
        "target_hourly_used_for_fit_or_calibration": False, "protocol_sha256_before": "same", "protocol_sha256_after": "same",
        "derived_hashes": {p.name: runner.V4.sha(p) for p in external.glob("*.csv")}})
    final = tmp_path / "external_v5"
    active_output[0] = final
    before = len(parsed)
    runner.fit_phase(data, baseline, final, root=root, external_data=external, selection=selection, n_samples=4)
    assert len(parsed) == before
    assert len(FIT_CALLS) == 45  # Neither source nor external labels recalibrate final artifacts.
    for record in config["records"]:
        assert runner.V4.sha(final / record["base_path"]) == record["original_base_sha256"]
        assert runner.V4.sha(final / record["path"]) == runner.V4.sha(source / record["path"])
    runner.predict_seal_phase(final, study="holdout")
    assert len(parsed) == before
    runner.score_phase(final)
    assert sum(year == 2025 for _, year in parsed) == 4
    assert {name for name, year in parsed if year == 2025} == {region + "_hourly" for region in runner.HOLDOUTS}
    assert (final / "scores/fixed_primary_metrics.csv").exists()
    assert (final / "preprojection_boundary_audit.csv").exists()
    original_beta = runner.CONFIGS["adaptive_component110_beta1"]
    with monkeypatch.context() as temporary:
        temporary.setitem(runner.CONFIGS, "adaptive_component110_beta1", (9., original_beta[1]))
        with pytest.raises(ValueError, match="grid differs"):
            runner.fit_phase(data, baseline, tmp_path / "changed_beta", root=root, external_data=external, selection=selection, n_samples=4)
        with pytest.raises(ValueError, match="grid differs"):
            runner.predict_seal_phase(source)
    runner_path = root / "scripts/run_risk_transfer_v5.py"
    original_runner_bytes = runner_path.read_bytes()
    runner_path.write_text("same src model modules but changed V5 runner")
    with pytest.raises(ValueError, match="V5 runner changed"):
        runner.fit_phase(data, baseline, tmp_path / "changed_runner", root=root, external_data=external, selection=selection, n_samples=4)
    runner_path.write_bytes(original_runner_bytes)
    (root / "src/transcif/research/risk_transfer_v5.py").write_text("changed risk policy after source selection")
    with pytest.raises(ValueError, match="risk/calibration code or environment changed"):
        runner.fit_phase(data, baseline, tmp_path / "bad_policy", root=root, external_data=external, selection=selection, n_samples=4)


def test_public_support_projection_records_unbounded_mass_without_changing_point():
    point = np.full((1, 24), -3.)
    ensemble = np.stack((np.full((1, 24), -2.), np.full((1, 24), 1002.)), axis=1)
    lower, upper = np.full((1, 24), -4.), np.full((1, 24), 1004.)
    clipped, lo, hi, audit = runner.project_distribution(ensemble, lower, upper)
    assert clipped.min() == 0 and clipped.max() == 1000
    assert lo.min() == 0 and hi.max() == 1000
    assert audit["preprojection_ensemble_below_zero"].tolist() == [.5]
    assert audit["preprojection_ensemble_above_max"].tolist() == [.5]
    assert audit["preprojection_interval_endpoints_below_zero"].tolist() == [.5]
    assert audit["preprojection_interval_endpoints_above_max"].tolist() == [.5]
    assert point.min() == -3  # Projection takes distributions only, never the raw point.
