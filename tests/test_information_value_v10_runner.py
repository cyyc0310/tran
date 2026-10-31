"""Small, label-free checks of V10 scoring and information boundaries."""

import json

import numpy as np
import pandas as pd
import pytest

from scripts import run_information_value_v10 as runner


def test_empirical_scores_match_two_draw_manual_case():
    point = np.full((1, 24), 2.)
    truth = np.full((1, 24), 1.)
    ensemble = np.stack((np.zeros((1, 24)), np.full((1, 24), 2.)), axis=1)
    scores = runner.score_arrays(point, ensemble, truth)
    assert scores["crps"] == pytest.approx([.5])
    assert scores["empirical_coverage90"] == pytest.approx([1.])
    assert scores["empirical_width90"] == pytest.approx([2.])
    assert scores["empirical_interval_score90"] == pytest.approx([2.])
    assert scores["empirical_path_coverage90"] == pytest.approx([1.])
    assert scores["point_level_mae"] == pytest.approx([1.])
    assert scores["point_shape_mae"] == pytest.approx([0.])


def test_weather_track_keeps_all_evaluation_days_and_later_proper_start():
    assert runner._evaluation_days("source", True)[0] == pd.Timestamp("2025-01-01", tz="UTC")
    assert runner._evaluation_days("target", True)[0] == pd.Timestamp("2026-01-01", tz="UTC")
    assert len(runner._evaluation_days("source", True)) == 181
    assert len(runner._evaluation_days("target", True)) == 181


def test_feature_record_rejects_late_generation_even_when_values_look_valid(monkeypatch):
    origin = pd.DatetimeIndex(["2025-01-08"], tz="UTC")
    mix = np.zeros((1, 23))
    mix[0, 0] = 1.

    def fake_features(*args, **kwargs):
        return {"base": mix, "range": np.zeros((1, 9)), "shape": np.zeros((1, 27)),
                "audit": {"rows": [{"origin": origin[0].isoformat(),
                                    "max_hour_end_allowed": "2025-01-01T00:00:00+00:00",
                                    "max_hour_end_used": "2025-01-02T00:00:00+00:00",
                                    "released_available_at": None,
                                    "released_period_end": None}]}}

    monkeypatch.setattr(runner, "build_features", fake_features)
    with pytest.raises(ValueError, match="availability"):
        runner._feature_record(pd.DataFrame(), pd.DataFrame(), None,
                               {"BPAT": (1., 2.)}, "BPAT", origin, 7)


def test_source_selection_uses_lag7_h1_only_and_records_failure():
    rows = []
    for region in runner.SOURCES:
        for lag in runner.LAGS:
            for budget in ("B", "R"):
                for model in runner.MODELS:
                    for factor in runner.FACTORS:
                        good = budget == "R" and model == "ridge" and factor == 1.
                        rows.append({"stage": "source", "region": region, "lag": lag,
                                     "budget": budget, "model": model, "factor": factor,
                                     "semester": "H1", "crps": 1. if good else 2.,
                                     "empirical_coverage90": .92 if good else .80,
                                     "empirical_width90": 2.,
                                     "empirical_interval_score90": 3.,
                                     "empirical_lower_tail_miss": .04,
                                     "empirical_upper_tail_miss": .04,
                                     "empirical_path_coverage90": .2,
                                     "point_level_mae": 1., "point_shape_mae": 1.})
    table = pd.DataFrame(rows)
    choice = runner.selection(table, ("B", "R"))
    assert choice["primary"]["budget"] == "R"
    assert choice["primary"]["model"] == "ridge"
    assert choice["primary"]["factor"] == 1.
    assert choice["primary"]["status"] == "passed"
    assert choice["selections"]["B"]["status"] == "failed"
    table.loc[(table.lag == 30) & table.budget.eq("B"), "crps"] = -1000.
    assert runner.selection(table, ("B", "R")) == choice
    table = table.loc[~((table.region == runner.SOURCES[0]) & table.budget.eq("R") &
                        table.model.eq("ridge") & table.factor.eq(1.) & table.lag.eq(7))]
    with pytest.raises(ValueError, match="grid incomplete"):
        runner.selection(table, ("B", "R"))


def test_evaluation_labels_cannot_be_read_before_prediction_seal(tmp_path, monkeypatch):
    def forbidden(*args):
        raise AssertionError("evaluation hourly labels were parsed")

    monkeypatch.setattr(runner, "_truth_lookup", forbidden)
    with pytest.raises(ValueError, match="Prediction seal required"):
        runner.score_phase(tmp_path, {}, "source", {"entries": []})
    runner.write(tmp_path / "source_prediction_seal.json", {"entries": []})
    with pytest.raises(ValueError, match="Prediction seal required"):
        runner.score_phase(tmp_path, {}, "source", {"entries": [1]})


def test_weather_registry_metadata_binding_is_required(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "audit.json").write_text(json.dumps({"publication_vintage_known": False}))
    points = tmp_path / "points.json"
    points.write_text(json.dumps({"schema_version": 1, "points": [
        {"region": r, "latitude": 1., "longitude": 2., "proxy": "fixture"}
        for r in runner.SOURCES + runner.TARGETS]}))
    weather = tmp_path / "daily_weather.csv"
    weather.write_text("region,origin\n")
    meta = tmp_path / "metadata.json"
    meta.write_text(json.dumps({"schema_version": 1, "points_sha256": "bad",
                                "daily_weather_sha256": runner.sha(weather),
                                "target_hourly_labels_read": False,
                                "fixed_lead_hours": 72}))
    old = tmp_path / "v9.json"
    old.write_text(json.dumps({"source": str(data), "target": str(data)}))
    monkeypatch.setattr(runner, "V9_ROOT", tmp_path)
    (tmp_path / "prefit_freeze.json").write_text(old.read_text())
    monkeypatch.setattr(runner, "_v9_background_paths", lambda: {})
    monkeypatch.setattr(runner, "_code_paths", lambda: [])
    monkeypatch.setattr(runner, "_weather", lambda _: pd.DataFrame({"region": runner.SOURCES + runner.TARGETS}))
    monkeypatch.setattr(runner.Path, "is_file", lambda self: True)
    with pytest.raises(ValueError, match="point registry mismatch"):
        runner.freeze(tmp_path / "new-output", data, weather_csv=weather, points_json=points)
