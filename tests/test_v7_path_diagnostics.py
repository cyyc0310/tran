import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("v7_path_diagnostics_tests", ROOT / "scripts/diagnose_paths_v7.py")
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)


def test_energy_score_exact_finite_pairwise_normalized_l2():
    particles = np.stack([np.zeros(24), np.full(24, 2.)])
    truth = np.ones(24)
    assert diagnostics.energy_score(particles, truth) == pytest.approx(.5)
    assert diagnostics.energy_score(np.repeat(truth[None], 128, axis=0), truth) == 0
    # Direct ordered-pair formula includes every self-pair in denominator S².
    rng = np.random.default_rng(17)
    x, y = rng.normal(size=(7, 24)), rng.normal(size=24)
    exact = np.linalg.norm(x - y, axis=1).mean() / np.sqrt(24)
    exact -= .5 * np.linalg.norm(x[:, None] - x[None], axis=2).mean() / np.sqrt(24)
    assert diagnostics.energy_score(x, y) == pytest.approx(exact)
    assert diagnostics.energy_score(x[::-1], y) == pytest.approx(exact)


def test_variogram_distinguishes_shape_from_pure_level_shift():
    truth = 200 + 10 * np.sin(2 * np.pi * np.arange(24) / 24)
    shifted = np.repeat((truth + 5)[None], 128, axis=0)
    assert diagnostics.variogram_score(shifted, truth) == pytest.approx(0., abs=1e-20)
    assert diagnostics.energy_score(shifted, truth) == pytest.approx(5.)
    point = np.roll(truth, 3)
    summary = diagnostics.shape_summary(point, truth)
    assert summary["point_daymin_circular_stable"] == 3
    assert summary["point_shape_cosine"] == pytest.approx(np.cos(2 * np.pi * 3 / 24))
    assert summary["point_shape_rms"] == pytest.approx(10 / np.sqrt(2))
    assert diagnostics.circular_hour_distance(23, 1) == 2


def test_variogram_uses_all_276_uniform_unordered_pairs():
    rng = np.random.default_rng(18)
    particles, truth = rng.normal(size=(5, 24)), rng.normal(size=24)
    terms = [(abs(truth[i] - truth[j])**.5 - np.mean(abs(particles[:, i] - particles[:, j])**.5))**2
             for i in range(24) for j in range(i + 1, 24)]
    assert len(terms) == 276
    assert diagnostics.variogram_score(particles, truth) == pytest.approx(np.mean(terms))


def test_flat_and_tied_minima_do_not_claim_identified_phase():
    truth = 300 + 20 * np.sin(2 * np.pi * np.arange(24) / 24)
    flat = diagnostics.shape_summary(np.full(24, 300.), truth)
    assert flat["point_flat"] and not flat["point_phase_valid"]
    assert flat["point_min_ties"] == 24
    assert np.isnan(flat["point_shape_cosine"])
    assert np.isnan(flat["point_daymin_circular_stable"])
    assert np.isfinite(flat["point_daymin_circular_first"])
    tied = truth.copy()
    tied[17] = tied[18]
    result = diagnostics.shape_summary(tied, truth)
    assert not result["point_flat"] and result["point_min_ties"] == 2
    assert not result["point_phase_valid"]


def test_raw_point_and_projected_scenario_mean_remain_distinct():
    truth = 300 + 20 * np.sin(2 * np.pi * np.arange(24) / 24)
    result = diagnostics.daily_metrics(truth + 10, np.repeat(truth[None], 128, axis=0), truth)
    assert result["point_level_bias"] == pytest.approx(10.)
    assert result["ensemble_mean_level_bias"] == pytest.approx(0.)
    assert result["point_shape_mae"] == pytest.approx(0., abs=1e-12)
    assert result["energy_score"] == pytest.approx(0.)
    raw = np.repeat(np.array([0., 200., 1000.])[None, :, None], 24, axis=2)
    projected = diagnostics.scale_scenarios(raw, 2.)
    np.testing.assert_array_equal(projected[0, :, 0], [0., 0., 1000.])
    assert raw.mean() == 400 and projected.mean() == pytest.approx(1000 / 3)


def fake_completed_run(tmp_path, monkeypatch):
    run, target = tmp_path / "run", tmp_path / "target"
    (run / "predictions").mkdir(parents=True)
    (run / "source_scores").mkdir()
    (run / "target_scores").mkdir()
    target.mkdir()
    hourly = target / "PACE_hourly.csv"
    hourly.write_text("ARTIFICIAL HOURLY FILE; PARSING IS INTERCEPTED BY TEST")
    for relative in ("prefit_freeze.json", "completed.json", "source_choices.json",
                     "source_prediction_seal.json", "source_scores/score_seal.json", "target_scores/score_seal.json"):
        (run / relative).write_text("{}")
    truth_days = pd.DatetimeIndex(["2026-01-01", "2026-01-02"], tz="UTC")
    truth = np.array([200., 300.])[:, None] + 8 * np.sin(2 * np.pi * np.arange(24) / 24)
    roles, entries, expected_days = {}, [], []
    for scope in diagnostics.SCOPES:
        roles[scope] = {}
        for role in diagnostics.ROLES:
            identity = "exact_direct" if role == "baseline" else role
            config = {"round": int(scope), "name": identity, "factor": 1.25}
            roles[scope][role] = {"configuration": config, "status": "failed", "metrics": {}}
            if role == "baseline":
                continue
            # Archive and truth order differ, exercising timestamp joins.
            point = truth[::-1] + 5
            ensemble = np.repeat((point - 2)[:, None], 128, axis=1)
            path = run / "predictions" / f"{scope}-{identity}.npz"
            np.savez(path, origins=truth_days[::-1].as_unit("ns").asi8, point=point, ensemble=ensemble)
            entries.append({"region": "PACE", "delay": 0, "round": int(scope), "name": identity,
                            "path": str(path.relative_to(run)), "sha256": diagnostics.ANALYSIS.sha(path), "interval_only": False})
            expected_days.extend({"region": "PACE", "delay": 0, **config, "origin": str(day)} for day in truth_days)
    diagnostics.ANALYSIS.write_json(run / "target_prediction_seal.json", {"entries": entries})
    verified = {"source": {}, "target": {"day": pd.DataFrame(expected_days)}, "roles": roles,
        "target_seal": {"entries": entries}, "source_seal": {},
        "frozen": {"code": {}, "inputs": {str(hourly): diagnostics.ANALYSIS.sha(hourly)}, "target": str(target)}}
    state = {"verified": False, "hourly_reads": 0, "daily_calls": 0, "figure_calls": 0}

    def verify(root, source_only):
        assert root == run and source_only is False
        for entry in entries:
            if diagnostics.ANALYSIS.sha(run / entry["path"]) != entry["sha256"]:
                raise ValueError("Prediction bytes changed")
        state["verified"] = True
        return copy.deepcopy(verified)

    monkeypatch.setattr(diagnostics.ANALYSIS, "verify_run", verify)
    monkeypatch.setattr(diagnostics.ANALYSIS.RUNNER, "TARGETS", ("PACE",))
    monkeypatch.setattr(diagnostics.ANALYSIS.RUNNER, "DELAYS", (0,))

    def read(path, year):
        assert state["verified"] and Path(path) == hourly and year == 2026
        state["hourly_reads"] += 1
        return None

    monkeypatch.setattr(diagnostics.ANALYSIS.RUNNER, "read_year", read)
    monkeypatch.setattr(diagnostics.ANALYSIS.RUNNER, "daily_profiles", lambda frame: (truth_days, truth))
    original = diagnostics.daily_metrics

    def counted(*args, **kwargs):
        state["daily_calls"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(diagnostics, "daily_metrics", counted)
    monkeypatch.setattr(diagnostics, "figures", lambda *args: state.update(figure_calls=state["figure_calls"] + 1))
    return run, verified, state


def test_diagnostics_cache_configuration_but_preserve_roles_and_seal_outputs(tmp_path, monkeypatch):
    run, verified, state = fake_completed_run(tmp_path, monkeypatch)
    output = tmp_path / "new-diagnostics"
    metadata = diagnostics.diagnose(run, output)
    assert state["hourly_reads"] == 1 and state["figure_calls"] == 1
    assert metadata["selected_configuration_cells_cached"] == 10
    assert metadata["role_configuration_cells_reported"] == 12
    assert state["daily_calls"] == 20
    daily = pd.read_csv(output / "day_path_metrics.csv")
    assert len(daily) == 24 and set(daily.role) == set(diagnostics.ROLES)
    assert daily.source_status.eq("failed").all()
    np.testing.assert_allclose(daily.point_level_bias, 5.)
    np.testing.assert_allclose(daily.ensemble_mean_level_bias, 3.)
    macro = pd.read_csv(output / "macro_path_metrics.csv")
    assert macro.n_region_delay_cells.eq(1).all() and macro.n_days_total.eq(2).all()
    profiles = pd.read_csv(output / "mean_profiles.csv")
    assert len(profiles) == 12 * 24
    assert metadata["all_prediction_archives_verified_before_label_read"]
    assert metadata["target_selection_performed"] is False
    assert all(diagnostics.ANALYSIS.sha(output / name) == digest for name, digest in metadata["output_hashes"].items())
    assert (output / "diagnostics_metadata.sha256").read_text().strip() == diagnostics.ANALYSIS.sha(output / "diagnostics_metadata.json")
    with pytest.raises(FileExistsError):
        diagnostics.diagnose(run, output)


def test_incomplete_or_changed_run_rejected_before_hourly_label_access(tmp_path, monkeypatch):
    run, verified, state = fake_completed_run(tmp_path, monkeypatch)
    entry = verified["target_seal"]["entries"][-1]
    path = run / entry["path"]
    path.write_bytes(path.read_bytes() + b"changed")
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="Prediction bytes changed"):
        diagnostics.diagnose(run, output)
    assert state["hourly_reads"] == 0 and not output.exists()


def test_retained_day_mismatch_cannot_silently_change_evaluation_sample(tmp_path, monkeypatch):
    run, verified, state = fake_completed_run(tmp_path, monkeypatch)
    original = diagnostics.ANALYSIS.verify_run

    def wrong_days(*args, **kwargs):
        result = original(*args, **kwargs)
        result["target"]["day"]["origin"] = "2026-02-01"
        return result

    monkeypatch.setattr(diagnostics.ANALYSIS, "verify_run", wrong_days)
    output = tmp_path / "different-support"
    with pytest.raises(ValueError, match="different retained days"):
        diagnostics.diagnose(run, output)
    assert not (output / "diagnostics_metadata.json").exists()


def test_input_hashes_rechecked_after_analysis_before_metadata_completion(tmp_path, monkeypatch):
    run, verified, state = fake_completed_run(tmp_path, monkeypatch)
    original = diagnostics.ANALYSIS.RUNNER.read_year

    def changed_on_read(path, year):
        result = original(path, year)
        Path(path).write_text("input changed during analysis")
        return result

    monkeypatch.setattr(diagnostics.ANALYSIS.RUNNER, "read_year", changed_on_read)
    output = tmp_path / "changed-during-analysis"
    with pytest.raises(ValueError, match="input changed during"):
        diagnostics.diagnose(run, output)
    assert not (output / "diagnostics_metadata.json").exists()
