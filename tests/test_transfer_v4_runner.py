"""Failure-oriented checks for prefit freezing and global pre-label sealing."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location("transfer_v4_runner", Path(__file__).parents[1] / "scripts/run_transfer_v4.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
BASE_MODES = ("source_residual", "pooled", "equal_region", "worstgrid_max", "protected_domain80")
COMPONENT_MODES = ("component_transport", "component_transport_110")


def frozen_run(tmp_path):
    root = tmp_path / "repo"
    (root / "src/transcif/research").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "docs").mkdir()
    (root / "src/transcif/research/model.py").write_text("source code")
    (root / "scripts/run_transfer_v4.py").write_text("runner code")
    (root / runner.PROTOCOL).write_text("fixed protocol")
    data = tmp_path / "data"
    data.mkdir()
    (data / "X_hourly.csv").write_text("time,value\n2025-01-01,1\n")
    output = tmp_path / "run"
    manifest = runner.freeze_inputs(root, data, output)
    (output / "fit.pkl").write_bytes(b"fitted")
    runner.write_json(output / "fit_seal.json", {"prefit_sha256": runner.sha(output / "prefit_freeze.json"),
        "artifacts": {"fit.pkl": runner.sha(output / "fit.pkl")}})
    return root, data, output, manifest


def prediction_entry(output, region="X", method="a"):
    name = f"{region}-{method}.npz"
    np.savez(output / name, point=np.zeros((1, 24)))
    return {"region": region, "delay": 0, "variant": "round1", "method": method,
            "calibration": "pooled", "path": name}


def entry_key(entry):
    return tuple(entry[key] for key in ("region", "delay", "variant", "method", "calibration"))


@pytest.mark.parametrize("changed", ["code", "data", "snapshot", "fit"])
def test_changed_input_or_fit_prevents_any_target_label_read(tmp_path, monkeypatch, changed):
    root, data, output, _ = frozen_run(tmp_path)
    entry = prediction_entry(output)
    runner.seal_predictions(output, [entry_key(entry)], [entry])
    locations = {"code": root / "src/transcif/research/model.py", "data": data / "X_hourly.csv",
                 "snapshot": output / "snapshot/src/transcif/research/model.py", "fit": output / "fit.pkl"}
    locations[changed].write_text("changed")
    opened = []
    monkeypatch.setattr(runner, "read_year", lambda *args: opened.append(args))
    with pytest.raises(ValueError, match="changed"):
        runner.LabelAccess(output).read(data / "X_hourly.csv", 2025)
    assert opened == []


def test_partial_global_grid_cannot_unlock_labels(tmp_path, monkeypatch):
    _, data, output, _ = frozen_run(tmp_path)
    first = prediction_entry(output, region="X")
    second = prediction_entry(output, region="Y")
    expected = [entry_key(first), entry_key(second)]
    with pytest.raises(ValueError, match="incomplete"):
        runner.seal_predictions(output, expected, [first])
    assert not (output / "all_predictions_sealed.json").exists()
    opened = []
    monkeypatch.setattr(runner, "read_year", lambda *args: opened.append(args))
    with pytest.raises(FileNotFoundError):
        runner.LabelAccess(output)
    assert opened == []
    runner.seal_predictions(output, expected, [first, second])
    runner.LabelAccess(output).read(data / "X_hourly.csv", 2025)
    assert len(opened) == 1


def test_every_target_method_is_verified_before_first_label(tmp_path, monkeypatch):
    _, data, output, _ = frozen_run(tmp_path)
    entries = [prediction_entry(output, region=region, method=method) for region in ("X", "Y") for method in ("a", "b")]
    runner.seal_predictions(output, [entry_key(e) for e in entries], entries)
    access = runner.LabelAccess(output)
    (output / entries[-1]["path"]).write_bytes(b"late mutation")
    opened = []
    monkeypatch.setattr(runner, "read_year", lambda *args: opened.append(args))
    with pytest.raises(ValueError, match="Sealed artifact changed"):
        access.read(data / "X_hourly.csv", 2025)
    assert opened == []


def test_manifest_mutation_cannot_rebaseline_frozen_inputs(tmp_path):
    _, _, output, manifest = frozen_run(tmp_path)
    entry = prediction_entry(output)
    runner.seal_predictions(output, [entry_key(entry)], [entry])
    manifest["data_hashes"] = {}
    (output / "prefit_freeze.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Prefit manifest changed"):
        runner.LabelAccess(output)


def test_run_and_global_seal_are_immutable(tmp_path):
    root, data, output, _ = frozen_run(tmp_path)
    with pytest.raises(FileExistsError):
        runner.freeze_inputs(root, data, output)
    entry = prediction_entry(output)
    runner.seal_predictions(output, [entry_key(entry)], [entry])
    with pytest.raises(FileExistsError):
        runner.seal_predictions(output, [entry_key(entry)], [entry])


def test_year_filter_excludes_poisoned_target_numeric_fields(tmp_path, monkeypatch):
    path = tmp_path / "years.csv"
    path.write_text("time,generation\n2022-01-01T00:00:00Z,1.5\n2025-01-01T00:00:00Z,FORBIDDEN_TARGET_LABEL\n")
    original = pd.read_csv
    def guarded_read_csv(buffer, *args, **kwargs):
        text = buffer.getvalue()
        assert "FORBIDDEN_TARGET_LABEL" not in text
        assert "2025-" not in text
        return original(buffer, *args, **kwargs)
    monkeypatch.setattr(pd, "read_csv", guarded_read_csv)
    frame = runner.read_year(path, 2022)
    assert frame.generation.tolist() == [1.5]


def test_analytic_and_empirical_intervals_are_separately_scored():
    truth = np.ones(24)
    point = np.zeros(24)
    metrics = runner.score_day(point, np.zeros((8, 24)), -np.ones(24), np.ones(24) * 2, truth)
    assert metrics["empirical_coverage90"] == 0
    assert metrics["analytic_coverage90"] == 1
    assert metrics["empirical_interval_score90"] == 20
    assert metrics["analytic_interval_score90"] == 3
    assert metrics["raw_point_mae"] == 1
    assert metrics["raw_shape_mae"] == 0
    assert metrics["raw_level_mae"] == 1


def selection_tables(coverages=(.9, .9), crps=(1., 2.)):
    rows = []
    for method, coverage, score in zip(("a", "b"), coverages, crps):
        for region in runner.CAL_REGIONS:
            for delay in runner.DELAYS:
                rows.append({"region": region, "delay": delay, "variant": "round1", "method": method,
                    "calibration": "pooled", "empirical_coverage90": coverage, "crps": score,
                    "empirical_width90": 4., "empirical_interval_score90": 5.})
    regional = pd.DataFrame(rows)
    macro = regional.groupby(["delay", "variant", "method", "calibration"]).mean(numeric_only=True).reset_index()
    return macro, regional


def test_selection_rejects_all_undercoverage_without_claiming_winner():
    macro, regional = selection_tables((.7, .8))
    selected = runner.select_config(macro, regional)
    assert selected["status"] == "failed"
    assert "configuration" not in selected
    assert "best_effort_diagnostic_only" in selected


def test_selection_is_uniform_across_lags_and_checks_worst_region():
    macro, regional = selection_tables()
    # The mean still clears .85, but a single region/delay is unprotected.
    regional.loc[(regional.method == "a") & (regional.region == "CISO") & (regional.delay == 0), "empirical_coverage90"] = .7
    selected = runner.select_config(macro, regional)
    assert selected["configuration"]["method"] == "b"
    assert "delay" not in selected["configuration"]


def test_raw_point_metrics_do_not_hide_bounds_by_clipping():
    point = np.r_[-2., np.ones(22), 1100.]
    metric = runner.score_day(point, np.repeat(point[None], 4, axis=0), point, point, np.ones(24))
    assert metric["raw_point_below_zero"] == 1 / 24
    assert metric["raw_point_above_max"] == 1 / 24
    assert metric["ensemble_above_max"] == 1 / 24
    assert metric["raw_point_mae"] > 40


class FakeModel:
    def predict(self, origins, region, monthly):
        n = len(origins)
        return {"method": {"point": np.zeros((n, 24)), "scale": np.ones((n, 24)),
            "age": np.ones(n), "level_scale": np.ones(n), "shape_scale": np.ones((n, 24))}}


class FakeCalibrator:
    def __init__(self, source_regions, alpha):
        self.source_regions = source_regions

    def fit(self, point, truth, scales, regions, ages, months, years):
        assert set(regions) == set(runner.CAL_REGIONS)
        assert set(years) == {2023}
        assert set(ages) <= set(runner.AGE_BIN_LABELS)
        return self

    def intervals(self, point, scales, mode, ages):
        return point - 1, point + 1

    def ensemble(self, point, scales, mode, ages, n_samples, seed):
        return np.repeat(point[:, None], n_samples, axis=1)


class FakeComponentCalibrator(FakeCalibrator):
    def fit(self, *args, level_scales, shape_scales, **kwargs):
        assert level_scales.shape == (len(shape_scales),)
        assert shape_scales.shape[1] == 24
        return super().fit(*args, **kwargs)

    def intervals(self, point, scales, mode, ages, level_scale, shape_scale):
        assert level_scale.shape == (len(point),)
        assert shape_scale.shape == point.shape
        return super().intervals(point, scales, mode, ages)

    def ensemble(self, point, scales, mode, ages, n_samples, seed, level_scale, shape_scale):
        assert level_scale.shape == (len(point),)
        assert shape_scale.shape == point.shape
        return super().ensemble(point, scales, mode, ages, n_samples, seed)


def test_complete_stage_order_with_independent_fit_and_calibration(tmp_path, monkeypatch):
    """An actual miniature staged run must never parse 2024 before global sealing."""
    import sys
    from types import SimpleNamespace
    root, data, unused, _ = frozen_run(tmp_path)
    output = tmp_path / "integration"
    for region in runner.FIT_REGIONS + runner.CAL_REGIONS:
        (data / f"{region}_hourly.csv").write_text("time,value\n2022-01-01T00:00:00Z,1\n2023-01-01T00:00:00Z,2\n2024-01-01T00:00:00Z,3\n2025-01-01T00:00:00Z,POISON\n")
    (data / "monthly.csv").write_text("region,total_mwh\nCISO,1\n")
    fit_calls = []
    def fit_transfer(frames, monthly, extra_delay, variant):
        assert set(frames) == set(runner.FIT_REGIONS)
        assert all(set(frame.time.dt.year) == {2022} for frame in frames.values())
        fit_calls.append((extra_delay, variant))
        return FakeModel()
    monkeypatch.setitem(sys.modules, "transcif.research.transfer_v4", SimpleNamespace(fit_transfer=fit_transfer))
    monkeypatch.setitem(sys.modules, "transcif.research.calibration_v4", SimpleNamespace(SourceCalibrator=FakeCalibrator, MODES=BASE_MODES))
    monkeypatch.setitem(sys.modules, "transcif.research.component_calibration_v4", SimpleNamespace(ComponentCalibrator=FakeComponentCalibrator, COMPONENT_MODES=COMPONENT_MODES))
    monkeypatch.setattr(runner, "daily_profiles", lambda frame: (pd.DatetimeIndex(frame.time), np.zeros((len(frame), 24))))
    original_read = runner.read_year
    parsed = []
    def read_guard(path, year):
        if year == 2024:
            assert (output / "all_predictions_sealed.json").is_file()
            assert len(json.loads((output / "all_predictions_sealed.json").read_text())["entries"]) == 21
        parsed.append((Path(path).stem, year))
        return original_read(path, year)
    monkeypatch.setattr(runner, "read_year", read_guard)
    runner.fit_phase(data, output, ("round1",), delays=(0,), n_samples=4, root=root)
    assert fit_calls == [(0, "round1")]
    config = json.loads((output / "run_config.json").read_text())
    assert config["calibration_modes"] == list(BASE_MODES + COMPONENT_MODES)
    assert config["ensemble_dtype"] == "float32"
    assert not any(year >= 2024 for _, year in parsed)
    runner.predict_seal_phase(output, end="2024-01-01")
    assert not any(year >= 2024 for _, year in parsed)
    with np.load(next((output / "predictions").glob("*.npz"))) as archive:
        assert archive["ensemble"].dtype == np.dtype("float32")
    macro = runner.score_phase(output)
    assert len(macro) == 7
    assert sum(year == 2024 for _, year in parsed) == 3
    assert not any(year == 2025 for _, year in parsed)
    selected = json.loads((output / "scores/selected_config.json").read_text())
    assert selected["status"] == "failed"  # Degenerate ensemble overcovers, as the gate must report.


def test_age_binning_has_stable_boundaries_and_rejects_invalid_ages():
    assert runner.age_bins([0, 60, 61, 120, 121, 240, 241, 1000]).tolist() == [
        "le60", "le60", "61_120", "61_120", "121_240", "121_240", "gt240", "gt240"]
    for values in ([-1], [np.nan], [np.inf]):
        with pytest.raises(ValueError, match="finite nonnegative"):
            runner.age_bins(values)


def test_prefit_captures_runtime_and_external_audit_hash(tmp_path):
    root, data, _, _ = frozen_run(tmp_path)
    external = tmp_path / "external"
    external.mkdir()
    (external / "monthly.csv").write_text("region,total_mwh\nSWPP,2\n")
    (external / "audit.json").write_text("{}")
    output = tmp_path / "both"
    manifest = runner.freeze_inputs(root, data, output, external_data=external)
    assert manifest["environment"]["packages"]["pandas"] == pd.__version__
    assert str(external / "audit.json") in manifest["data_hashes"]
    assert str(external / "monthly.csv") in manifest["data_hashes"]
    (external / "audit.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="Frozen input changed"):
        runner.verify_frozen_inputs(output)


def test_tail_errors_and_path_coverage_are_explicit():
    truth = np.zeros(24)
    truth[0], truth[1] = -2., 2.
    result = runner.interval_metrics(-np.ones(24), np.ones(24), truth)
    assert result["lower_tail_miss"] == 1 / 24
    assert result["upper_tail_miss"] == 1 / 24
    assert result["path_coverage90"] == 0
    assert runner.interval_metrics(-np.ones(24), np.ones(24), np.zeros(24))["path_coverage90"] == 1


def test_source_selected_fitted_bytes_are_reused_for_all_new_regions(tmp_path, monkeypatch):
    """A miniature external run verifies binding without reading real BA outcomes."""
    import sys
    from types import SimpleNamespace
    root, data, _, _ = frozen_run(tmp_path)
    source = tmp_path / "source"
    for region in runner.FIT_REGIONS + runner.CAL_REGIONS:
        (data / f"{region}_hourly.csv").write_text("time,value\n2022-01-01T00:00:00Z,1\n2023-01-01T00:00:00Z,2\n2024-01-01T00:00:00Z,3\n")
    (data / "monthly.csv").write_text("region,total_mwh\nCISO,1\n")
    fit_calls = []
    def fit_transfer(*args, **kwargs):
        fit_calls.append(kwargs)
        return FakeModel()
    monkeypatch.setitem(sys.modules, "transcif.research.transfer_v4", SimpleNamespace(fit_transfer=fit_transfer))
    monkeypatch.setitem(sys.modules, "transcif.research.calibration_v4", SimpleNamespace(SourceCalibrator=FakeCalibrator, MODES=BASE_MODES))
    monkeypatch.setitem(sys.modules, "transcif.research.component_calibration_v4", SimpleNamespace(ComponentCalibrator=FakeComponentCalibrator, COMPONENT_MODES=COMPONENT_MODES))
    def profiles(frame):
        truth = np.zeros((len(frame), 24))
        truth[:, :3] = 1.  # Empirical coverage 21/24 passes the declared gate.
        return pd.DatetimeIndex(frame.time), truth
    monkeypatch.setattr(runner, "daily_profiles", profiles)
    runner.fit_phase(data, source, ("round1",), delays=(0,), n_samples=4, root=root)
    runner.predict_seal_phase(source, end="2024-01-01")
    runner.score_phase(source)
    selection = source / "scores/selected_config.json"
    assert runner.load_source_selection(selection)["status"] == "passed"
    external = tmp_path / "external"
    external.mkdir()
    (external / "monthly.csv").write_text("region,total_mwh\nSWPP,2\n")
    for region in runner.HOLDOUT_REGIONS:
        (external / f"{region}_hourly.csv").write_text("time,value\n2025-01-01T00:00:00Z,9\n")
    audit = {"external_regions": list(runner.HOLDOUT_REGIONS), "final_external_year": 2025,
        "cif_scores_inspected": False, "cif_summaries_computed": False,
        "target_hourly_used_for_fit_or_calibration": False,
        "protocol_sha256_before": "same", "protocol_sha256_after": "same",
        "derived_hashes": {path.name: runner.sha(path) for path in external.glob("*.csv")}}
    runner.write_json(external / "audit.json", audit)
    output = tmp_path / "external_run"
    parsed = []
    original_read = runner.read_year
    def guarded_read(path, year):
        if year == 2025:
            assert (output / "all_predictions_sealed.json").exists()
            assert len(json.loads((output / "all_predictions_sealed.json").read_text())["entries"]) == 42
            assert Path(path).parent == external
        parsed.append(year)
        return original_read(path, year)
    monkeypatch.setattr(runner, "read_year", guarded_read)
    runner.fit_phase(data, output, ("round1",), delays=(0,), n_samples=4, root=root,
                     external_data=external, selection=selection)
    assert len(fit_calls) == 1  # External mode never refits or recalibrates.
    assert parsed == []
    assert runner.sha(source / "fit/round1-0.pkl") == runner.sha(output / "fit/round1-0.pkl")
    runner.predict_seal_phase(output, study="holdout")
    assert parsed == []
    runner.score_phase(output)
    assert parsed == [2025] * 6
    primary = pd.read_csv(output / "scores/fixed_primary_metrics.csv")
    assert len(primary) == 1
    assert primary.method.tolist() == ["method"]
    # A changed model closure cannot silently borrow an earlier selection.
    (root / "src/transcif/research/model.py").write_text("changed fitted model definition")
    with pytest.raises(ValueError, match="code closure differs"):
        runner.fit_phase(data, tmp_path / "changed_model_run", ("round1",), delays=(0,), n_samples=4,
                         root=root, external_data=external, selection=selection)
