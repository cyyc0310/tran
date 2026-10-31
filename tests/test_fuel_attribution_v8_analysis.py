"""Tests for source-fixed V8 attribution analysis and its no-oracle-selection boundary."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("v8_attribution_analysis_under_test", ROOT / "scripts/analyze_fuel_attribution_v8.py")
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


def _record(round_, mode, factor, status="passed"):
    prefix = "latest" if round_ == 1 else "adaptive_mixture-t0.03"
    suffix = "b1.0" if round_ == 1 else "b0.3"
    return {"configuration": {"round": round_, "name": f"{prefix}-{mode}-{suffix}", "factor": factor},
            "status": status, "metrics": {"crps": 20.}}


def choices():
    selections = {}
    for round_ in (1, 2):
        selections[str(round_)] = {
            "primary": _record(round_, "recompose_joint", 1.25),
            "baseline": _record(round_, "additive_anchor", 1.5),
            "exact_additive": _record(round_, "additive_anchor", 1.25),
            "exact_direct": _record(round_, "direct_analog", 1.25),
        }
    selections["joint"] = {role: selections["2"][role].copy() for role in selections["2"]}
    return {"selections": selections}


def test_fixed_roles_keep_exact_mapping_factor_and_joint_source_choice():
    roles = A.fixed_roles(choices())
    assert len(A.planned_legal(roles)) == 9
    assert all(item["primary"]["factor"] == item["reference"]["factor"]
               for item in A.planned_legal(roles) if item["comparison"].startswith("primary_minus_exact"))
    planned = A.planned_oracle(roles)
    assert len(planned) == 6
    assert planned[0]["reference"]["name"] == planned[0]["primary"]["name"].replace(
        "recompose_joint", "oracle_recompose_joint")
    assert all(item["primary"]["factor"] == item["reference"]["factor"] for item in planned)


def test_exact_control_or_joint_tampering_rejected():
    altered = choices()
    altered["selections"]["1"]["exact_direct"]["configuration"]["factor"] = .75
    with pytest.raises(ValueError, match="primary factor"):
        A.fixed_roles(altered)
    altered = choices()
    altered["selections"]["joint"]["primary"] = _record(2, "recompose_joint", .75)
    with pytest.raises(ValueError):
        A.fixed_roles(altered)


def _days(config, score, *, oracle=False):
    rows = []
    for region in A.V7.RUNNER.TARGETS:
        for delay in A.V7.RUNNER.DELAYS:
            for month in range(1, 7):
                rows.append({"region": region, "delay": delay, **config,
                    "mode": "oracle_recompose_joint" if oracle else "recompose_joint",
                    "origin": f"2026-{month:02d}-10 00:00:00+00:00", "semester": "H1",
                    "crps": score + month / 10, "empirical_coverage90": .91,
                    "empirical_width90": 100., "empirical_interval_score90": score + 5,
                    "raw_level_mae": 3., "raw_shape_mae": 4., "point_level_mae": 3., "point_shape_mae": 4.})
    return pd.DataFrame(rows)


def test_month_bootstrap_same_day_and_oracle_separate():
    roles = A.fixed_roles(choices())
    legal_config = roles["2"]["primary"]["configuration"]
    oracle_config = {**legal_config, "name": legal_config["name"].replace("recompose_joint", "oracle_recompose_joint")}
    add_config = roles["2"]["exact_additive"]["configuration"]
    oracle_add = {**add_config, "name": add_config["name"].replace("additive_anchor", "oracle_additive_anchor")}
    legal = pd.concat([_days(legal_config, 20.), _days(add_config, 20.)], ignore_index=True)
    hindsight = pd.concat([_days(oracle_config, 19., oracle=True), _days(oracle_add, 19., oracle=True)], ignore_index=True)
    paired = A.oracle_legal_days(legal, hindsight, {"1": roles["2"], "2": roles["2"], "joint": roles["2"]})
    assert len(paired) == 2 * len(legal)
    comparison = A.planned_oracle({"2": roles["2"]})[0]
    result = A.compare_days(paired, [comparison], n_bootstrap=100)
    overall = result[(result.stratum == "overall") & (result.metric == "crps")].iloc[0]
    assert overall.difference_primary_minus_reference == pytest.approx(1.)
    assert overall.ci95_low == pytest.approx(1.)
    assert overall.n_calendar_blocks == 6
    with pytest.raises(ValueError, match="different scored days"):
        A.oracle_legal_days(legal, hindsight.iloc[1:], {"2": roles["2"]})


def test_source_h2_is_descriptive_and_failed_gate_blocks_holdout():
    roles = A.fixed_roles(choices())
    source = pd.DataFrame([
        {"scope": "joint", "role": "primary", "semester": "H1", "empirical_coverage90": .92, "worst_coverage": .86},
        {"scope": "joint", "role": "primary", "semester": "H2", "empirical_coverage90": .1, "worst_coverage": 0.},
    ])
    legal = pd.DataFrame([
        {"scope": "joint", "role": "primary", "semester": "H1", "crps": 90., "empirical_coverage90": .91,
         "worst_coverage": .86, "empirical_interval_score90": 80., "empirical_width90": 90.},
        {"scope": "joint", "role": "baseline", "semester": "H1", "crps": 100., "empirical_coverage90": .91,
         "worst_coverage": .86, "empirical_interval_score90": 100., "empirical_width90": 100.},
    ])
    pairs = pd.DataFrame([{"scope": "joint", "comparison": "primary_minus_baseline", "stratum": "overall",
                           "metric": "crps", "ci95_high": -1.}])
    old = pd.DataFrame([{"scope": "joint", "comparison": "v8_primary_minus_v7_fresh_anchor",
        "stratum": "overall", "metric": metric, "primary_mean": 90., "reference_mean": 100.,
        "difference_primary_minus_reference": -10., "ci95_high": -1.}
        for metric in ("crps", "empirical_interval_score90", "empirical_width90")])
    status = A.holdout_trigger(roles, source, legal, pairs, old)
    assert status["new_holdout_triggered"] is True
    assert status["scope"].startswith("Development")
    roles["joint"]["primary"]["status"] = "failed"
    assert A.holdout_trigger(roles, source, legal, pairs, old)["new_holdout_triggered"] is False
    assert A.holdout_trigger(roles, source, legal, pairs)["new_holdout_triggered"] is False


def test_incomplete_stage_stops_before_any_score_read(tmp_path, monkeypatch):
    run = tmp_path / "incomplete"
    run.mkdir()
    output = tmp_path / "analysis"

    class IncompleteRunner:
        @staticmethod
        def verify(_):
            return {}

        @staticmethod
        def verify_phase(_, stage):
            raise ValueError(f"{stage} completion missing")

    monkeypatch.setattr(A, "runner", lambda: IncompleteRunner)
    monkeypatch.setattr(pd, "read_csv", lambda *_args, **_kwargs: pytest.fail("partial scores were read"))
    with pytest.raises(ValueError, match="completion missing"):
        A.analyze(run, output, through="source")
    assert not output.exists()
