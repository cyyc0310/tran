"""Small behavioral checks for source-only V7 selection and sealed scoring."""
import copy
import importlib.util
import inspect
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fuel_transport_v7_runner_tests", ROOT / "scripts/run_fuel_transport_v7.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def score_table():
    rows = []
    for config in runner.configs():
        mode = config["mode"]
        for factor in ((1.,) if mode == "cqr" else runner.FACTORS):
            for region in ("A", "B"):
                for delay in (0, 30):
                    for semester in ("H1", "H2"):
                        # The best primary uses bw=.3; the best independently
                        # selected direct control uses bw=.1. Exact controls
                        # must retain the primary's configuration instead.
                        cost = {"recompose_joint": 10., "direct_analog": 8.,
                                "recompose_independent": 9., "pooled_recompose": 11.,
                                "fresh_anchor": 9.5, "extratrees": 9., "cqr": np.nan}[mode]
                        if config.get("bandwidth") == .3 and mode == "recompose_joint":
                            cost -= 1
                        if config.get("bandwidth") == .1 and mode == "direct_analog":
                            cost -= 1
                        if config.get("forecast") == "adaptive_mean":
                            cost += .1
                        coverage = .92 if mode == "cqr" else {.75: .8, 1.: .89, 1.25: .92, 1.5: .98}[factor]
                        rows.append({"region": region, "delay": delay, "round": config["round"],
                            "name": config["name"], "mode": mode, "factor": factor, "semester": semester,
                            "days": 3, "crps": cost, "empirical_coverage90": coverage,
                            "empirical_width90": 20., "empirical_interval_score90": 25.})
    return pd.DataFrame(rows)


def test_selection_uses_h1_only_even_when_h2_strongly_favors_another_candidate():
    table = score_table()
    expected, _ = runner.select(table)
    altered = table.copy()
    stress = altered.semester.eq("H2")
    altered.loc[stress, "empirical_coverage90"] = .92
    altered.loc[stress, "crps"] = np.where(altered.loc[stress, "name"].str.contains("b0.1"), 0., 1e6)
    altered.loc[stress, "empirical_interval_score90"] = 1e6
    actual, _ = runner.select(altered)
    assert actual == expected
    assert set(actual) == {"1", "2", "joint"}
    assert actual["1"]["primary"]["configuration"]["factor"] == 1.25
    assert actual["1"]["primary"]["configuration"]["name"].endswith("b0.3")


def test_exact_mechanism_controls_keep_primary_forecast_bandwidth_and_factor():
    selected, _ = runner.select(score_table())
    for round_ in ("1", "2"):
        primary = selected[round_]["primary"]["configuration"]
        for role, mode in (("exact_direct", "direct_analog"),
                           ("exact_independent", "recompose_independent"),
                           ("exact_pooled", "pooled_recompose")):
            actual = selected[round_][role]["configuration"]
            assert actual == {**primary, "name": primary["name"].replace("recompose_joint", mode)}
    assert selected["1"]["baseline"]["configuration"]["name"].endswith("b0.1")
    assert selected["1"]["exact_direct"]["configuration"]["name"].endswith("b0.3")
    assert selected["1"]["simple_anchor"]["configuration"]["name"] == "fresh_anchor"


def test_failed_coverage_keeps_diagnostic_choice_without_qualifying_it():
    table = score_table()
    table.loc[:, "empirical_coverage90"] = .7
    selected, ranking = runner.select(table)
    assert (ranking.violation > 0).all()
    for round_ in ("1", "2"):
        for role in ("primary", "baseline", "simple_anchor", "cqr",
                     "exact_direct", "exact_independent", "exact_pooled"):
            assert selected[round_][role]["status"] == "failed"
            assert selected[round_][role]["configuration"]
        assert selected[round_]["unconstrained_baseline"]["status"] == "unconstrained"
    assert selected["joint"]["primary"]["status"] == "failed"


class FakeStructural:
    def predict(self, days, region, monthly):
        return {"analog_shape_half": np.full((len(days), 24), 30.)}


class FakeBaseline:
    def __init__(self):
        self.fit_arguments = None

    def fit(self, *arguments):
        self.fit_arguments = arguments
        return self

    def predict(self, features, days, n_scenarios, seed):
        n = len(days)
        return {"extratrees": {"point": np.full((n, 24), 100.),
                               "ensemble": np.full((n, n_scenarios, 24), 100.)},
                "cqr": {"lower": np.full((n, 24), 90.), "upper": np.full((n, 24), 110.)}}


class FakeBank:
    def __init__(self):
        self.calls = []
        self.fit_regions = None

    def fit(self, frames):
        self.fit_regions = set(frames)
        return self

    def predict(self, mix, days, **options):
        self.calls.append((np.asarray(mix).copy(), pd.DatetimeIndex(days), options.copy()))
        level = np.asarray(mix) @ np.arange(8.)
        point = np.repeat(level[:, None], 24, axis=1)
        return {"point": point, "ensemble": np.repeat(point[:, None], options["n_scenarios"], axis=1),
                "unsupported_target_mass": np.zeros(len(days)), "effective_donors": np.full(len(days), 2.),
                "zero_generation_fallback_fraction": np.zeros(len(days))}


def fake_coarse(n):
    experts = np.broadcast_to(np.eye(8)[:3], (n, 3, 8)).copy()
    return {"features": np.arange(n * 4.).reshape(n, 4), "experts": experts,
            "weights": {str(t): np.broadcast_to([.1, .2, .7], (n, 3)).copy() for t in (.03, .1)},
            "anchor": np.full(n, 5.)}


def test_fit_bundle_excludes_lodo_region_from_both_h1_fit_and_h2_calibration(monkeypatch):
    monkeypatch.setattr(runner, "SOURCES", ("A", "B", "HELD"))
    monkeypatch.setattr(runner, "FuelTrajectoryBank", FakeBank)
    monkeypatch.setattr(runner, "DirectBaselines", FakeBaseline)
    fitted_regions = []

    def fit_structural(frames, monthly, **kwargs):
        fitted_regions.append(set(frames))
        assert kwargs["cutoff"] == "2024-07-01T00:00:00Z"
        return FakeStructural()

    monkeypatch.setattr(runner, "fit_model", fit_structural)
    proper, cal = {r: object() for r in runner.SOURCES}, {r: object() for r in runner.SOURCES}

    def records(month):
        result = {}
        for region in runner.SOURCES:
            result[region] = {"days": pd.date_range(f"2024-{month:02d}-01", periods=2, tz="UTC"),
                "truth": np.full((2, 24), "FORBIDDEN_HELD_LABEL") if region == "HELD" else np.full((2, 24), 100.),
                "coarse": fake_coarse(2)}
        return result

    bundle = runner.fit_bundle(proper, cal, None, 30, "HELD", records(1), records(7))
    assert bundle["training_regions"] == ("A", "B")
    assert bundle["bank"].fit_regions == {"A", "B"}
    assert fitted_regions == [{"A", "B"}]
    arguments = bundle["baseline"].fit_arguments
    assert set(arguments[2]) == set(arguments[5]) == {"A", "B"}
    assert arguments[1].dtype.kind == arguments[4].dtype.kind == "f"
    assert set(bundle["residual_regions"]) == {"A", "B"}
    np.testing.assert_array_equal(bundle["residuals"], 65.)


def test_raw_predictions_need_no_outcomes_and_use_matched_common_coarse_inputs():
    assert tuple(inspect.signature(runner.raw_predictions).parameters) == ("bundle", "monthly", "region", "days", "coarse")
    days = pd.date_range("2026-01-01", periods=2, tz="UTC")
    coarse, bank = fake_coarse(2), FakeBank()
    bundle = {"bank": bank, "baseline": FakeBaseline(), "structural": FakeStructural(),
              "training_regions": ("A", "B"), "residual_regions": np.array(["A", "B"]),
              "residuals": np.zeros((2, 24))}
    outputs = list(runner.raw_predictions(bundle, None, "TARGET", days, coarse))
    assert len(outputs) == len(runner.configs())
    for bandwidth in (.1, .3, 1.):
        latest = [(mix, options) for mix, _, options in bank.calls if options["bandwidth"] == bandwidth
                  and np.array_equal(mix, coarse["experts"][:, 0])]
        assert {options["mode"] for _, options in latest} == {
            "recompose_joint", "direct_analog", "recompose_independent", "pooled_recompose"}
        assert all(options["seed"] == runner.SEED and options["n_scenarios"] == runner.N_SAMPLES for _, options in latest)
    key = next(c["name"] for c in runner.configs() if c.get("forecast") == "adaptive_mixture"
               and c["mode"] == "recompose_joint")
    mixture = next(result for config, result, _ in outputs if config["name"] == key)
    expected = np.stack([runner.day_rng(day, 997).choice(3, runner.N_SAMPLES, p=[.1, .2, .7]) for day in days])
    np.testing.assert_array_equal(mixture["ensemble"][:, :, 0], expected)
    np.testing.assert_array_equal(mixture["point"], mixture["ensemble"].mean(axis=1))


def choices_fixture(tmp_path):
    output = tmp_path / "run"
    scores = output / "source_scores"
    scores.mkdir(parents=True)
    score_table().to_csv(scores / "region_metrics.csv", index=False)
    selections, _ = runner.select(pd.read_csv(scores / "region_metrics.csv"))
    runner.write(output / "source_prediction_seal.json", {"entries": [], "fits": {}})
    choice = {"selections": selections,
              "source_score_hashes": {"region_metrics.csv": runner.sha(scores / "region_metrics.csv")},
              "source_seal_sha256": runner.sha(output / "source_prediction_seal.json"),
              "target_model_scores_read": False}
    runner.write(output / "source_choices.json", choice)
    return output, choice


def test_committed_source_choices_recompute_from_h1_and_tampering_is_rejected(tmp_path):
    output, choice = choices_fixture(tmp_path)
    runner.validate_choices(output, choice)
    for field, value in (("factor", .75), ("name", "fresh_anchor"), ("status", "failed")):
        changed = copy.deepcopy(choice)
        record = changed["selections"]["joint"]["primary"]
        (record if field == "status" else record["configuration"])[field] = value
        with pytest.raises(ValueError, match="choice changed"):
            runner.validate_choices(output, changed)
    metrics = output / "source_scores/region_metrics.csv"
    original = metrics.read_bytes()
    metrics.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="Source scores changed"):
        runner.validate_choices(output, choice)
    metrics.write_bytes(original)
    (output / "source_prediction_seal.json").write_text("{}")
    with pytest.raises(ValueError, match="Source seal changed"):
        runner.validate_choices(output, choice)


def test_target_rejects_rewritten_sourcechoice_before_generating_predictions_or_reading_labels(tmp_path, monkeypatch):
    output, choice = choices_fixture(tmp_path)
    choice["selections"]["1"]["primary"]["configuration"]["factor"] = .75
    runner.write(output / "source_choices.json", choice)
    monkeypatch.setattr(runner, "verify", lambda path: {"source": str(tmp_path / "source"), "target": str(tmp_path / "target")})

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid source choices must be rejected before prediction or target scoring")

    monkeypatch.setattr(runner, "raw_predictions", forbidden)
    monkeypatch.setattr(runner, "score_entries", forbidden)
    with pytest.raises(ValueError, match="choice changed"):
        runner.target_phase(output)


def archives(tmp_path):
    output = tmp_path / "run"
    (output / "predictions").mkdir(parents=True)
    days = pd.DatetimeIndex(["2025-07-01", "2025-01-01", "2025-01-02"], tz="UTC").as_unit("us")
    point = np.repeat(np.array([100., 200., 999.])[:, None], 24, axis=1)
    outputs = [({"round": 1, "name": "joint", "mode": "recompose_joint"},
                {"point": point, "ensemble": np.repeat(point[:, None], runner.N_SAMPLES, axis=1)}, {}),
               ({"round": 1, "name": "cqr", "mode": "cqr"}, {"lower": point - 1, "upper": point + 1}, {})]
    entries = runner.save_predictions(output, "source", "A", 0, days, outputs)
    return output, entries


def truth_spy(monkeypatch):
    monkeypatch.setattr(runner, "read_year", lambda path, year: year)

    def truth(year):
        assert year == 2025
        # Reverse order relative to archive; matching must use UTC timestamp,
        # not position, and the unobserved archive day must be skipped.
        days = pd.DatetimeIndex(["2025-01-01", "2025-07-01"], tz="UTC")
        y = np.repeat(np.array([200., 100.])[:, None], 24, axis=1)
        return days, y

    monkeypatch.setattr(runner, "daily_profiles", truth)


def test_save_score_matches_dates_not_positions_and_keeps_cqr_interval_only(tmp_path, monkeypatch):
    output, entries = archives(tmp_path)
    truth_spy(monkeypatch)
    table = runner.score_entries(output, entries, tmp_path, 2025, output / "scores")
    probabilistic = table[table["mode"].eq("recompose_joint")]
    assert set(probabilistic.semester) == {"H1", "H2"}
    assert probabilistic.days.eq(1).all()
    np.testing.assert_array_equal(probabilistic.crps, 0.)
    np.testing.assert_array_equal(probabilistic.point_level_mae, 0.)
    assert table.loc[table["mode"].eq("cqr"), "crps"].isna().all()
    counts = pd.read_csv(output / "scores/day_counts.csv")
    assert counts.days.eq(2).all()
    seal = json.loads((output / "scores/score_seal.json").read_text())
    assert all(runner.sha(output / "scores" / name) == digest for name, digest in seal.items())


def test_prediction_bytes_are_immutable_and_score_rejects_changed_archive(tmp_path, monkeypatch):
    output, entries = archives(tmp_path)
    reads = []
    monkeypatch.setattr(runner, "read_year", lambda *args: reads.append(args))
    monkeypatch.setattr(runner, "daily_profiles", lambda frame: (pd.DatetimeIndex(["2025-01-01"], tz="UTC"),
                                                               np.zeros((1, 24))))
    archive = output / entries[0]["path"]
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Prediction bytes changed"):
        runner.score_entries(output, entries, tmp_path, 2025, output / "scores")
    assert not reads
    with pytest.raises(FileExistsError):
        archives(tmp_path)


def test_frozen_code_or_inputs_changed_rejected_before_reuse(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    frozen = tmp_path / "coarse-input.csv"
    frozen.write_text("region,value\nA,1\n")
    runner.write(output / "prefit_freeze.json", {"code": {}, "inputs": {str(frozen): runner.sha(frozen)},
        "configs": runner.configs(), "factors": list(runner.FACTORS), "delays": list(runner.DELAYS),
        "seed": runner.SEED, "samples": runner.N_SAMPLES, "environment": runner.environment()})
    runner.verify(output)
    manifest = json.loads((output / "prefit_freeze.json").read_text())
    modified = copy.deepcopy(manifest)
    modified["environment"]["python"] = "changed interpreter"
    runner.write(output / "prefit_freeze.json", modified)
    with pytest.raises(ValueError, match="[Ee]nvironment"):
        runner.verify(output)
    runner.write(output / "prefit_freeze.json", manifest)
    frozen.write_text("region,value\nA,2\n")
    with pytest.raises(ValueError, match="Frozen input/code changed"):
        runner.verify(output)


def test_target_hourly_read_waits_for_the_complete_prediction_seal(tmp_path, monkeypatch):
    output, choice = choices_fixture(tmp_path)
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir(); target.mkdir(); (output / "fits").mkdir(); (output / "predictions").mkdir()
    pd.DataFrame({"region": ["S"]}).to_csv(source / "monthly.csv", index=False)
    pd.DataFrame({"region": ["T"]}).to_csv(target / "monthly.csv", index=False)
    fitted = output / "fits/full-0.pkl"
    fitted.write_bytes(pickle.dumps({"frozen": "source-only artifact"}))
    runner.write(output / "source_prediction_seal.json", {"entries": [], "fits": {fitted.name: runner.sha(fitted)}})
    choice["source_seal_sha256"] = runner.sha(output / "source_prediction_seal.json")
    runner.write(output / "source_choices.json", choice)
    monkeypatch.setattr(runner, "verify", lambda path: {"source": str(source), "target": str(target)})
    monkeypatch.setattr(runner, "TARGETS", ("T1", "T2"))
    monkeypatch.setattr(runner, "DELAYS", (0,))
    monkeypatch.setattr(runner, "N_SAMPLES", 2)
    tiny_grid = [{"round": 1, "name": "joint", "mode": "recompose_joint"},
                 {"round": 1, "name": "cqr", "mode": "cqr"}]
    monkeypatch.setattr(runner, "configs", lambda: tiny_grid)
    monkeypatch.setattr(runner, "make_features", lambda monthly, region, days, delay: fake_coarse(len(days)))

    def raw(bundle, monthly, region, days, coarse):
        point = np.full((len(days), 24), 100.)
        yield tiny_grid[0], {"point": point, "ensemble": np.repeat(point[:, None], 2, axis=1)}, {}
        yield tiny_grid[1], {"lower": point - 1, "upper": point + 1}, {}

    monkeypatch.setattr(runner, "raw_predictions", raw)
    reads = []

    def read_target(path, year):
        assert Path(path).parent == target and year == 2026
        seal = json.loads((output / "target_prediction_seal.json").read_text())
        assert seal["all_grid_complete"] and not seal["target_labels_read_in_this_prediction_phase"]
        assert len(seal["entries"]) == 4
        assert all(runner.sha(output / entry["path"]) == entry["sha256"] for entry in seal["entries"])
        reads.append(path)
        return None

    monkeypatch.setattr(runner, "read_year", read_target)
    monkeypatch.setattr(runner, "daily_profiles", lambda frame: (pd.DatetimeIndex(["2026-01-01"], tz="UTC"),
                                                               np.full((1, 24), 100.)))
    runner.target_phase(output)
    assert len(reads) == 2
    assert (output / "completed.json").exists()
    with pytest.raises(FileExistsError, match="immutable"):
        runner.target_phase(output)
