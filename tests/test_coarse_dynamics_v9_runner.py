"""V9 sealing and source-only selection contract tests."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


RUNNER = load("test_v9_runner", ROOT / "scripts/run_coarse_dynamics_v9.py")
ANALYSIS = load("test_v9_analysis", ROOT / "scripts/analyze_coarse_dynamics_v9.py")


def small_grid():
    rows = []
    for round_ in (1, 2):
        for name, mode, family, simple, dynamic in (
            ("proposed", "recompose", "proposed", False, False),
            ("additive", "additive", "standard", True, False),
            ("kalman", "recompose", "standard", False, True),
        ):
            rows.append({"round": round_, "name": name, "mode": mode, "family": family,
                         "simple_anchor": simple, "standard_dynamic": dynamic})
    return rows


def source_table(grid):
    rows = []
    for config in grid:
        for half in ("H1", "H2"):
            coverage = .92 if config["name"] != "kalman" else .88
            rows.append({**{k: config[k] for k in ("round", "name", "mode")},
                "region": "A", "delay": 0, "factor": 1., "semester": half,
                "days": 180, "crps": {"proposed": 2., "additive": 3., "kalman": 1.}[config["name"]]
                + (100 if half == "H2" and config["name"] == "proposed" else 0),
                "empirical_coverage90": coverage, "empirical_width90": 12.,
                "empirical_interval_score90": 14.})
    return pd.DataFrame(rows)


def test_source_selection_h1_only_and_failed_fallback(monkeypatch):
    monkeypatch.setattr(RUNNER.V7, "SOURCES", ("A",))
    monkeypatch.setattr(RUNNER.V7, "DELAYS", (0,))
    roles, ranking = RUNNER.selection(source_table(small_grid()), small_grid())
    assert roles["1"]["primary"]["configuration"]["name"] == "proposed"
    assert roles["1"]["baseline"]["configuration"]["name"] == "additive"
    assert roles["1"]["standard_dynamic"]["status"] == "failed"
    assert roles["joint"]["primary"]["metrics"]["crps"] == 2.
    assert ranking.loc[ranking.name.eq("proposed"), "crps"].eq(2.).all()
    with pytest.raises(ValueError, match="incomplete"):
        RUNNER.selection(source_table(small_grid()).iloc[1:], small_grid())


def test_real_coarse_module_loads_and_saves_lawful_archive(tmp_path, monkeypatch):
    monthly = pd.read_csv(ROOT / "results/heldout-2025-data/monthly.csv")
    days = pd.date_range("2025-01-01", periods=2, tz="UTC")
    methods = RUNNER.coarse().forecast_monthly(monthly, "BPAT", days,
                                                extra_delay=30, n_samples=4, seed=17023)
    assert set(methods) == set(RUNNER.coarse().METHODS)
    (tmp_path / "coarse_predictions").mkdir()
    monkeypatch.setattr(RUNNER.V7, "N_SAMPLES", 4)
    entries = RUNNER._save_coarse(tmp_path, "source", "BPAT", 30, days, methods,
                                  RUNNER.coarse().METHODS)
    assert len(entries) == 6
    assert all((tmp_path / entry["path"]).exists() for entry in entries)


def test_coarse_to_risk_to_sealed_grid_integration(tmp_path, monkeypatch):
    class TinyBank:
        cutoff_ = pd.Timestamp("2024-07-01", tz="UTC")
        regions_ = np.array(["train1", "train2"])
        source_regions_ = ("train1", "train2")
        months_ = np.array([1, 1])
        days_ = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]")
        donor_mix_ = np.array([[.7, .3, 0, 0, 0, 0, 0, 0],
                               [.3, .7, 0, 0, 0, 0, 0, 0]])
        q_ = np.ones((2, 24, 8))
        raw_cif_ = np.stack([np.full(24, 200.), np.full(24, 300.)])
        fuel_support_ = np.ones((2, 8))

        def _weights(self, mix, origin, mode, bandwidth, eligible):
            weights = np.asarray(eligible, float)
            return weights / weights.sum(), None

        @staticmethod
        def _rng(seed, origin):
            return np.random.default_rng(np.random.SeedSequence([seed, pd.Timestamp(origin).dayofyear]))

    monthly = pd.read_csv(ROOT / "results/heldout-2025-data/monthly.csv")
    days = pd.date_range("2025-01-01", periods=2, tz="UTC")
    methods = RUNNER.coarse().forecast_monthly(monthly, "BPAT", days,
                                                extra_delay=30, n_samples=4, seed=17023)
    predictions = RUNNER.risk().predict_paths(TinyBank(), methods, days)
    grid = RUNNER.risk().configs()
    (tmp_path / "predictions").mkdir()
    (tmp_path / "coarse_predictions").mkdir()
    monkeypatch.setattr(RUNNER.V7, "SOURCES", ("BPAT",))
    monkeypatch.setattr(RUNNER.V7, "DELAYS", (30,))
    monkeypatch.setattr(RUNNER.V7, "N_SAMPLES", 4)
    entries = RUNNER._save_predictions(tmp_path, "source", "BPAT", 30, days, predictions, grid)
    coarse_entries = RUNNER._save_coarse(tmp_path, "source", "BPAT", 30, days,
                                          methods, RUNNER.coarse().METHODS)
    assert len(entries) == 32 and len(coarse_entries) == 6
    seal = {"entries": entries, "coarse_entries": coarse_entries,
            "complete_grid": True, "coarse_complete_grid": True}
    monkeypatch.setattr(RUNNER.pd, "date_range", lambda *args, **kwargs: days)
    RUNNER.check_entries(tmp_path, {"configs": grid, "coarse_methods": list(methods)}, "source", seal)
    RUNNER.write(tmp_path / "source_prediction_seal.json", seal)
    RUNNER.score_coarse(tmp_path, coarse_entries, monthly, tmp_path / "source_coarse_scores")
    by_month = pd.read_csv(tmp_path / "source_coarse_scores/monthly_metrics.csv")
    assert len(by_month) == 6
    assert np.isfinite(by_month[["mix_energy_score32", "ef_point_mae", "ef_crps"]]).all().all()


def test_missing_actual_month_is_explicit_without_dropping_forecasts(tmp_path):
    (tmp_path / "coarse_predictions").mkdir()
    days = pd.DatetimeIndex(["2025-01-31", "2025-02-01"], tz="UTC")
    mix = np.tile([1., *([0.] * 7)], (2, 1))
    path = tmp_path / "coarse_predictions" / "one.npz"
    np.savez_compressed(path, origins=days.as_unit("ns").asi8, mix=mix,
                        draws=np.repeat(mix[:, None], 4, axis=1))
    entry = {"region": "BPAT", "delay": 0, "method": "latest",
             "path": "coarse_predictions/one.npz", "sha256": RUNNER.sha(path)}
    RUNNER.write(tmp_path / "source_prediction_seal.json", {"coarse_entries": [entry]})
    monthly = pd.DataFrame([{"region": "BPAT", "period_start": "2025-01-01T00:00:00Z",
                             **{f"fuelshare_{name}": float(i == 0) for i, name in enumerate(
                                 ("coal", "gas", "oil", "nuclear", "hydro", "wind", "solar", "other"))}}])
    RUNNER.score_coarse(tmp_path, [entry], monthly, tmp_path / "source_coarse_scores")
    rows = pd.read_csv(tmp_path / "source_coarse_scores/day_metrics.csv")
    support = pd.read_csv(tmp_path / "source_coarse_scores/month_support.csv")
    assert len(rows) == 2 and rows.scored.tolist() == [True, False]
    assert rows.loc[1, ["mix_mae", "mix_crps", "mix_energy_score32", "ef_point_mae", "ef_crps"]].isna().all()
    assert support.label_available.tolist() == [True, False]
    assert len(pd.read_csv(tmp_path / "source_coarse_scores/monthly_metrics.csv")) == 1


def test_coarse_macro_balances_regions_with_uneven_month_labels(tmp_path):
    (tmp_path / "coarse_predictions").mkdir()
    days = pd.DatetimeIndex(["2025-01-15", "2025-02-15"], tz="UTC")
    mix = np.tile([1., *([0.] * 7)], (2, 1))
    entries = []
    for region in ("BPAT", "CISO"):
        path = tmp_path / "coarse_predictions" / f"{region}.npz"
        np.savez_compressed(path, origins=days.as_unit("ns").asi8, mix=mix,
                            draws=np.repeat(mix[:, None], 4, axis=1))
        entries.append({"region": region, "delay": 0, "method": "latest",
                        "path": str(path.relative_to(tmp_path)), "sha256": RUNNER.sha(path)})
    RUNNER.write(tmp_path / "source_prediction_seal.json", {"coarse_entries": entries})
    fuels = ("coal", "gas", "oil", "nuclear", "hydro", "wind", "solar", "other")
    def truth(region, month, fuel):
        return {"region": region, "period_start": f"2025-{month:02d}-01T00:00:00Z",
                **{f"fuelshare_{name}": float(name == fuel) for name in fuels}}
    monthly = pd.DataFrame([truth("BPAT", 1, "coal"), truth("BPAT", 2, "gas"),
                            truth("CISO", 1, "coal")])
    RUNNER.score_coarse(tmp_path, entries, monthly, tmp_path / "source_coarse_scores")
    macro = pd.read_csv(tmp_path / "source_coarse_scores/macro_metrics.csv")
    assert macro.loc[0, "mix_mae"] == pytest.approx(.0625)


def test_real_coarse_grid_verifier_handles_sparse_source_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(RUNNER.V7, "SOURCES", ("BPAT",))
    monkeypatch.setattr(RUNNER.V7, "DELAYS", (0,))
    (tmp_path / "coarse_predictions").mkdir()
    source = tmp_path / "source"
    source.mkdir()
    days = pd.date_range("2025-01-01", "2026-01-01", inclusive="left", tz="UTC")
    mix = np.tile([1., *([0.] * 7)], (len(days), 1))
    archive = tmp_path / "coarse_predictions" / "latest.npz"
    np.savez_compressed(archive, origins=days.as_unit("ns").asi8, mix=mix,
                        draws=np.repeat(mix[:, None], 4, axis=1))
    entry = {"region": "BPAT", "delay": 0, "method": "latest",
             "path": str(archive.relative_to(tmp_path)), "sha256": RUNNER.sha(archive)}
    RUNNER.write(tmp_path / "source_prediction_seal.json", {"coarse_entries": [entry]})
    monthly = pd.DataFrame([{"region": "BPAT", "period_start": "2025-01-01T00:00:00Z",
                             **{f"fuelshare_{name}": float(i == 0) for i, name in enumerate(
                                 ("coal", "gas", "oil", "nuclear", "hydro", "wind", "solar", "other"))}}])
    monthly.to_csv(source / "monthly.csv", index=False)
    RUNNER.score_coarse(tmp_path, [entry], monthly, tmp_path / "source_coarse_scores")
    RUNNER.check_coarse_score_grid(tmp_path,
                                   {"source": str(source), "coarse_methods": ["latest"]}, "source")
    support = pd.read_csv(tmp_path / "source_coarse_scores/month_support.csv")
    assert support.label_available.sum() == 1
    assert len(support) == 12


def test_prediction_grid_missing_and_tampered(tmp_path, monkeypatch):
    monkeypatch.setattr(RUNNER.V7, "SOURCES", ("A",))
    monkeypatch.setattr(RUNNER.V7, "DELAYS", (0,))
    monkeypatch.setattr(RUNNER.V7, "N_SAMPLES", 2)
    (tmp_path / "predictions").mkdir()
    (tmp_path / "coarse_predictions").mkdir()
    days = pd.date_range("2025-01-01", "2026-01-01", inclusive="left", tz="UTC")
    origins = days.as_unit("ns").asi8
    config = {"round": 1, "name": "one", "mode": "recompose", "family": "proposed"}
    pred = tmp_path / "predictions" / "one.npz"
    coarse = tmp_path / "coarse_predictions" / "one.npz"
    np.savez_compressed(pred, origins=origins, point=np.zeros((len(days), 24)),
                        ensemble=np.zeros((len(days), 2, 24)))
    mix = np.tile([1., *([0.] * 7)], (len(days), 1))
    np.savez_compressed(coarse, origins=origins, mix=mix,
                        draws=np.repeat(mix[:, None], 2, axis=1))
    frozen = {"configs": [config], "coarse_methods": ["latest"]}
    seal = {"entries": [{**config, "region": "A", "delay": 0, "path": "predictions/one.npz",
                         "sha256": RUNNER.sha(pred), "interval_only": False}],
            "coarse_entries": [{"region": "A", "delay": 0, "method": "latest",
                                "path": "coarse_predictions/one.npz", "sha256": RUNNER.sha(coarse)}],
            "complete_grid": True, "coarse_complete_grid": True}
    RUNNER.check_entries(tmp_path, frozen, "source", seal)
    with pytest.raises(ValueError, match="Incomplete"):
        RUNNER.check_entries(tmp_path, frozen, "source", {**seal, "coarse_entries": []})
    pred.write_bytes(pred.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="changed"):
        RUNNER.check_entries(tmp_path, frozen, "source", seal)


def test_score_grid_requires_every_factor_and_semester(tmp_path, monkeypatch):
    monkeypatch.setattr(RUNNER.V7, "SOURCES", ("A",))
    monkeypatch.setattr(RUNNER.V7, "DELAYS", (0,))
    monkeypatch.setattr(RUNNER.V7, "FACTORS", (.75, 1.))
    folder = tmp_path / "source_scores"
    folder.mkdir()
    rows = []
    for factor in RUNNER.V7.FACTORS:
        for semester in ("H1", "H2"):
            rows.append({"region": "A", "delay": 0, "round": 1, "name": "one",
                         "factor": factor, "semester": semester, "days": 180,
                         "crps": 1., "empirical_coverage90": .92,
                         "empirical_width90": 1., "empirical_interval_score90": 1.})
    pd.DataFrame(rows).to_csv(folder / "region_metrics.csv", index=False)
    RUNNER.check_score_grid(tmp_path, {"configs": [{"round": 1, "name": "one"}]}, "source")
    pd.DataFrame(rows[:-1]).to_csv(folder / "region_metrics.csv", index=False)
    with pytest.raises(ValueError, match="Incomplete"):
        RUNNER.check_score_grid(tmp_path, {"configs": [{"round": 1, "name": "one"}]}, "source")


def test_freeze_captures_original_closure_and_rejects_input_mutation(tmp_path, monkeypatch):
    upstream = tmp_path / "prior"
    for folder in ("fits", "source_scores", "target_scores"):
        (upstream / folder).mkdir(parents=True)
        (upstream / folder / "artifact").write_bytes(b"frozen")
    for name in ("prefit_freeze.json", "source_choices.json", "source_prediction_seal.json",
                 "target_prediction_seal.json", "completed.json"):
        (upstream / name).write_text("{}")
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    original_input = source / "monthly.csv"
    original_input.write_text("original")
    original_code = tmp_path / "old.py"
    original_code.write_text("original code")
    old = {"source": str(source), "target": str(target),
           "inputs": {str(original_input): RUNNER.sha(original_input)},
           "code": {str(original_code): RUNNER.sha(original_code)}}
    monkeypatch.setattr(RUNNER.A7, "verify_run", lambda _: {"frozen": old})
    output = tmp_path / "new"
    frozen = RUNNER.freeze(output, upstream)
    assert str(original_input) in frozen["dependencies"]
    assert str(upstream / "fits/artifact") in frozen["dependencies"]
    RUNNER.verify(output)
    original_input.write_text("changed")
    with pytest.raises(ValueError, match="Frozen bytes changed"):
        RUNNER.verify(output)


def test_target_completion_requires_source_choice_and_seal(tmp_path, monkeypatch):
    frozen = {"dependencies": {}, "code": {}, "configs": [], "inputs": {}}
    (tmp_path / "prefit_freeze.json").write_text(json.dumps(frozen))
    (tmp_path / "target_prediction_seal.json").write_text(json.dumps({
        "created_utc": "2026-01-02T00:00:00+00:00", "source_choices_sha256": "bad",
        "hourly_labels_read_in_prediction_phase": False,
        "actual_monthly_mix_used_for_scoring_before_seal": False,
        "monthly_forecast_vintages_read_preseal": True}))
    (tmp_path / "source_choices.json").write_text(json.dumps({"created_utc": "2026-01-01T00:00:00+00:00"}))
    for folder in ("target_scores", "target_coarse_scores"):
        (tmp_path / folder).mkdir()
    score_files = RUNNER.SCORE_FILES
    for name in score_files:
        (tmp_path / "target_scores" / name).write_text("x")
    RUNNER.write(tmp_path / "target_scores/score_seal.json",
                 {name: RUNNER.sha(tmp_path / "target_scores" / name) for name in score_files})
    for name in RUNNER.COARSE_SCORE_FILES:
        (tmp_path / "target_coarse_scores" / name).write_text("x")
    RUNNER.write(tmp_path / "target_coarse_scores/score_seal.json",
                 {name: RUNNER.sha(tmp_path / "target_coarse_scores" / name)
                  for name in RUNNER.COARSE_SCORE_FILES})
    RUNNER.write(tmp_path / "target_completed.json", {
        "created_utc": "2026-01-03T00:00:00+00:00",
        "prefit_freeze_sha256": RUNNER.sha(tmp_path / "prefit_freeze.json"),
        "prediction_seal_sha256": RUNNER.sha(tmp_path / "target_prediction_seal.json"),
        "score_seal_sha256": RUNNER.sha(tmp_path / "target_scores/score_seal.json"),
        "coarse_score_seal_sha256": RUNNER.sha(tmp_path / "target_coarse_scores/score_seal.json"),
        "source_choices_sha256": RUNNER.sha(tmp_path / "source_choices.json"),
        "target_selection_performed": False, "target_training_performed": False})
    monkeypatch.setattr(RUNNER, "verify", lambda _: frozen)
    monkeypatch.setattr(RUNNER, "check_entries", lambda *_: None)
    monkeypatch.setattr(RUNNER, "check_score_grid", lambda *_: None)
    monkeypatch.setattr(RUNNER, "check_coarse_score_grid", lambda *_: None)
    monkeypatch.setattr(RUNNER, "verify_phase_source", lambda *_: None, raising=False)
    original = RUNNER.verify_phase
    monkeypatch.setattr(RUNNER, "verify_phase", lambda output, stage: None if stage == "source" else original(output, stage))
    monkeypatch.setattr(RUNNER, "validate_choices", lambda *_: None)
    with pytest.raises(ValueError, match="Target detached"):
        RUNNER.verify_phase(tmp_path, "target")
    (tmp_path / "target_prediction_seal.json").write_text(
        (tmp_path / "target_prediction_seal.json").read_text() + " ")
    with pytest.raises(ValueError, match="prediction seal binding"):
        RUNNER.verify_phase(tmp_path, "target")


def test_gate_requires_all_three_references():
    roles = {"joint": {"primary": {"status": "passed"}}}
    macro = pd.DataFrame([{"scope": "joint", "role": "primary", "semester": "H1",
                           "empirical_coverage90": .92, "worst_coverage": .87}])
    rows = []
    for ref in ("simple_anchor", "standard_dynamic", "old_v8_fixed_ratio"):
        for metric, difference in (("crps", -1.), ("empirical_interval_score90", -1.),
                                   ("empirical_width90", 0.)):
            rows.append({"scope": "joint", "comparison": "primary_minus_" + ref,
                         "stratum": "overall", "metric": metric, "primary_mean": 8.,
                         "reference_mean": 10., "difference_primary_minus_reference": difference,
                         "ci95_high": -.1})
    pairs = pd.DataFrame(rows)
    assert ANALYSIS.holdout_trigger(roles, macro, macro, pairs, old_available=True)["new_holdout_triggered"]
    pairs.loc[pairs.comparison.eq("primary_minus_standard_dynamic") & pairs.metric.eq("crps"), "ci95_high"] = .1
    assert not ANALYSIS.holdout_trigger(roles, macro, macro, pairs, old_available=True)["new_holdout_triggered"]
