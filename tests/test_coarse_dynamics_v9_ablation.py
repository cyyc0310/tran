"""Source-fixed matched mechanism ablation tests."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/analyze_coarse_dynamics_v9_ablation.py"
spec = importlib.util.spec_from_file_location("test_v9_ablation", PATH)
ABLATION = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ABLATION
spec.loader.exec_module(ABLATION)


def roles(joint_round=2):
    return {
        "1": {"primary": {"configuration": {"round": 1, "name": "release_bridge--recompose", "factor": 1.25}}},
        "2": {"primary": {"configuration": {"round": 2, "name": "v7_mixture--residual_conditional", "factor": 1.}}},
        "joint": {"primary": {"configuration": {"round": joint_round,
            "name": "v7_mixture--residual_conditional" if joint_round == 2 else "release_bridge--recompose",
            "factor": .75}}},
    }


def test_plan_is_same_forecast_factor_and_canonical_r2():
    grid = ABLATION.RUNNER.configurations()
    methods = list(ABLATION.RUNNER.coarse().METHODS)
    planned = ABLATION.plan(roles(), grid, methods)
    assert len(planned) == 26
    assert sum(row["source_primary_aligned"] for row in planned) == 3
    for row in planned:
        assert row["primary"]["round"] == row["reference"]["round"] == row["round"]
        assert row["primary"]["factor"] == row["reference"]["factor"] == row["factor"]
        assert row["primary"]["name"].split("--")[0] == row["reference"]["name"].split("--")[0] == row["forecast"]
        if row["mechanism"] == "residual_conditional_minus_global":
            assert row["round"] == 2
            assert row["primary"]["name"].endswith("residual_conditional")
            assert row["reference"]["name"].endswith("residual_global")
    assert len(ABLATION.plan(roles(joint_round=1), grid, methods)) == 22


def test_plan_rejects_changed_or_incomplete_source_identity():
    grid = ABLATION.RUNNER.configurations()
    methods = list(ABLATION.RUNNER.coarse().METHODS)
    broken = roles()
    broken["2"]["primary"]["configuration"]["name"] = "absent"
    with pytest.raises(ValueError, match="absent"):
        ABLATION.plan(broken, grid, methods)
    grid = [row for row in grid if not (row["round"] == 2 and row["name"] == "v7_mixture--residual_global")]
    with pytest.raises(ValueError, match="incomplete"):
        ABLATION.plan(roles(), grid, methods)


def test_matched_comparison_uses_same_days_and_prespecified_direction():
    item = next(row for row in ABLATION.plan(roles(), ABLATION.RUNNER.configurations(),
                                              list(ABLATION.RUNNER.coarse().METHODS))
                if row["scope"] == "2" and row["mechanism"] == "residual_conditional_minus_global"
                and row["forecast"] == "v7_mixture")
    rows = []
    for region in ABLATION.RUNNER.V7.TARGETS:
        for delay in ABLATION.RUNNER.V7.DELAYS:
            for month in range(1, 7):
                for config, crps in ((item["primary"], 1.), (item["reference"], 2.)):
                    rows.append({"region": region, "delay": delay, "origin": f"2026-{month:02d}-15T00:00:00Z",
                                 "round": config["round"], "name": config["name"], "factor": config["factor"],
                                 "mode": config["name"].split("--")[-1], "crps": crps,
                                 "empirical_coverage90": .92, "empirical_width90": 10.,
                                 "empirical_interval_score90": crps + 1.,
                                 "point_level_mae": crps, "point_shape_mae": crps})
    day = pd.DataFrame(rows)
    result = ABLATION.compare(day, [item], n_bootstrap=100)
    overall = result[result.stratum.eq("overall") & result.metric.eq("crps")].iloc[0]
    assert overall.difference_primary_minus_reference == pytest.approx(-1.)
    assert overall.ci95_high == pytest.approx(-1.)
    assert overall.source_primary_aligned
    with pytest.raises(ValueError, match="different scored days"):
        ABLATION.compare(day.iloc[:-1], [item], n_bootstrap=100)
