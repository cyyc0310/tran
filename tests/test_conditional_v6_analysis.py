"""Actual V6 runner role selection must agree with sealed-score analysis."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


runner = module("actual_conditional_runner", "run_conditional_transfer_v6.py")
analysis = module("actual_conditional_analysis", "analyze_conditional_transfer_v6.py")
anchored = module("actual_anchored_runner", "run_anchored_transfer_v6.py")


def declared_runner(variant):
    return runner if variant == "conditional_v6" else anchored.R6


def synthetic_scores(case, variant="conditional_v6"):
    rows, intervals = [], []
    for region in runner.SOURCES:
        for delay in runner.DELAYS:
            for method_index, method in enumerate(runner.POINTS):
                for config_index, calibration in enumerate(analysis.probability_calibrations(variant)):
                    coverage = .9
                    if case == "all_failed" or (case == "primary_failed" and calibration.startswith("structured_x")):
                        coverage = .7
                    elif case == "summed_tolerance" and calibration.startswith("structured_x"):
                        # Separate threshold tolerances would pass the macro;
                        # runner's summed violation is deliberately stricter.
                        coverage = .85 - 7e-13
                    rows.append({"region": region, "delay": delay, "variant": variant,
                        "method": method, "calibration": calibration,
                        "crps": .001 if calibration.startswith("noanchor_") else 10. + config_index / 10 + method_index,
                        "empirical_coverage90": coverage, "empirical_width90": 100. + config_index,
                        "empirical_interval_score90": 120. + config_index,
                        "empirical_lower_tail_miss": (1 - coverage) / 2,
                        "empirical_upper_tail_miss": (1 - coverage) / 2,
                        "empirical_path_coverage90": coverage / 2})
                coverage = .7 if case == "all_failed" else .9
                intervals.append({"region": region, "delay": delay, "variant": variant,
                    "method": method, "calibration": "cqr", "empirical_coverage90": coverage,
                    "empirical_width90": 110. + method_index, "empirical_interval_score90": 130. + method_index,
                    "empirical_lower_tail_miss": (1 - coverage) / 2,
                    "empirical_upper_tail_miss": (1 - coverage) / 2,
                    "empirical_path_coverage90": coverage / 2})
    def frames(records):
        regional = pd.DataFrame(records)
        numeric = [column for column in regional.select_dtypes(include="number") if column != "delay"]
        macro = regional.groupby(["delay", *analysis.KEYS])[numeric].mean().reset_index()
        return macro, regional
    return *frames(rows), *frames(intervals)


def bound_selection(tmp_path, frames, variant="conditional_v6"):
    scores = tmp_path / "scores"
    scores.mkdir()
    names = ("macro_metrics", "region_metrics", "interval_macro_metrics", "interval_region_metrics")
    for name, frame in zip(names, frames):
        frame.to_csv(scores / (name + ".csv"), index=False)
    (tmp_path / "fit_seal.json").write_text(json.dumps({"artifacts": {"fit/test.pkl": "synthetic"}}))
    (tmp_path / "all_predictions_sealed.json").write_text("{}")
    frozen = {"root": str(tmp_path), "code_hashes": {}, "environment": {"python": "synthetic"}}
    (tmp_path / "prefit_freeze.json").write_text(json.dumps(frozen))
    selection = {"source_year": 2025, "selection_regions": list(runner.SOURCES),
        "selections": declared_runner(variant).select_roles(*frames), "fit_artifact_hashes": {"fit/test.pkl": "synthetic"},
        "fit_seal_sha256": analysis.V4.sha(tmp_path / "fit_seal.json"),
        "prediction_global_seal_sha256": analysis.V4.sha(tmp_path / "all_predictions_sealed.json"),
        "model_closure_hashes": {}, "environment": frozen["environment"]}
    selection.update({name + "_sha256": analysis.V4.sha(scores / (name + ".csv")) for name in names})
    return scores, selection


@pytest.mark.parametrize("case", ("all_passed", "primary_failed", "all_failed", "summed_tolerance"))
@pytest.mark.parametrize("variant", analysis.VARIANTS)
def test_actual_runner_choices_and_statuses_validate_without_reselection(tmp_path, case, variant):
    assert len(analysis.probability_calibrations(variant)) == (22 if variant == "conditional_v6" else 32)
    frames = synthetic_scores(case, variant)
    scores, selection = bound_selection(tmp_path, frames, variant)
    ranked = analysis.rank_probability(*frames[:2])
    intervals = analysis.rank_intervals(*frames[2:])
    roles = analysis.validate_selection(tmp_path, scores, {}, selection, ranked, intervals)
    assert roles == declared_runner(variant).select_roles(*frames)
    assert roles["unconstrained_baseline"]["status"] == "unconstrained"
    assert roles["historical_v5"]["status"] == "unconstrained"
    if case in ("primary_failed", "all_failed", "summed_tolerance"):
        assert roles["primary"]["status"] == "failed"
    else:
        assert roles["primary"]["status"] == "passed"
    assert roles["primary"]["configuration"]["calibration"].startswith("structured_x")
    if variant == "anchored_v6":
        assert ranked.iloc[0].calibration.startswith("noanchor_")
        assert not roles["baseline"]["configuration"]["calibration"].startswith("noanchor_")
        assert not roles["unconstrained_baseline"]["configuration"]["calibration"].startswith("noanchor_")
    if case == "all_failed":
        assert all(roles[role]["status"] == "failed" for role in ("primary", "baseline", "cqr"))


def test_failed_role_cannot_be_relabelled_as_passed(tmp_path):
    frames = synthetic_scores("all_failed")
    scores, selection = bound_selection(tmp_path, frames)
    selection["selections"]["primary"]["status"] = "passed"
    with pytest.raises(ValueError, match="coverage-gate result"):
        analysis.validate_selection(tmp_path, scores, {}, selection,
            analysis.rank_probability(*frames[:2]), analysis.rank_intervals(*frames[2:]))


def test_synchronous_h1_bootstrap_preserves_fixed_effect_and_rejects_bad_pairing():
    primary = {"variant": runner.VARIANT, "method": runner.POINTS[0], "calibration": "structured_x1"}
    baseline = {**primary, "calibration": "global_x1"}
    rows = [{"region": region, "delay": delay, "origin": f"2026-{month:02d}-15T00:00Z", **config, "crps": value}
        for region in runner.HOLDOUTS for delay in runner.DELAYS for month in range(1, 7)
        for config, value in ((primary, 3.), (baseline, 1.))]
    days = pd.DataFrame(rows)
    table = analysis.paired_month_bootstrap(days, primary, baseline, n_bootstrap=100)
    assert table.difference_primary_minus_reference.eq(2).all()
    assert table.ci95_low.eq(2).all() and table.ci95_high.eq(2).all()
    assert table.n_calendar_blocks.eq(6).all()
    days.loc[0, "origin"] = "2025-01-15T00:00Z"
    with pytest.raises(ValueError, match="different scored days"):
        analysis.paired_month_bootstrap(days, primary, baseline, n_bootstrap=100)


@pytest.mark.parametrize("external", (False, True))
@pytest.mark.parametrize("variant", analysis.VARIANTS)
def test_sealed_grid_accepts_only_exact_source2025_or_target2026h1(tmp_path, external, variant):
    """Exercise actual runner grids, seals and disjoint CQR score support."""
    write = analysis.V4.write_json
    sha = analysis.V4.sha
    frames = list(synthetic_scores("all_passed", variant))
    regions = runner.HOLDOUTS if external else runner.SOURCES
    if external:
        frames[1] = frames[1].loc[frames[1].region.ne(runner.SOURCES[-1])].copy()
        frames[3] = frames[3].loc[frames[3].region.ne(runner.SOURCES[-1])].copy()
        mapping = dict(zip(runner.SOURCES, runner.HOLDOUTS))
        for i in (1, 3):
            frames[i]["region"] = frames[i].region.map(mapping)
    year = 2026 if external else 2025
    origins = pd.date_range(f"{year}-01-01", "2026-06-30" if external else "2025-12-31", tz="UTC")
    expected = [(region, delay, variant, method, calibration)
        for region in regions for delay in runner.DELAYS for method in runner.POINTS
        for calibration in (*analysis.probability_calibrations(variant), "cqr")]
    entries = [{**dict(zip(("region", "delay", *analysis.KEYS), key)), "path": f"predictions/{i}.npz"}
               for i, key in enumerate(expected)]
    config = declared_runner(variant).grid_parameters()
    assert config["calibration_modes"] == list(analysis.probability_calibrations(variant))
    write(tmp_path / "run_config.json", config)
    write(tmp_path / "prefit_freeze.json", {})
    (tmp_path / "fit.bin").write_bytes(b"synthetic fitted artifact")
    write(tmp_path / "fit_seal.json", {"prefit_sha256": sha(tmp_path / "prefit_freeze.json"),
        "artifacts": {"fit.bin": sha(tmp_path / "fit.bin"), "run_config.json": sha(tmp_path / "run_config.json")}})
    plan = {"study": "holdout" if external else "development", "year": year, "targets": list(regions),
            "origins": [str(day) for day in origins], "expected": expected}
    write(tmp_path / "evaluation_plan.json", plan)
    write(tmp_path / "all_predictions_sealed.json", {"plan_sha256": sha(tmp_path / "evaluation_plan.json"),
        "prefit_sha256": sha(tmp_path / "prefit_freeze.json"), "fit_seal_sha256": sha(tmp_path / "fit_seal.json"),
        "target_labels_read": False, "entries": entries, "expected": expected,
        "artifacts": {entry["path"]: "synthetic prediction commitment" for entry in entries}})
    write(tmp_path / "evaluation_commit.json", {"plan_sha256": sha(tmp_path / "evaluation_plan.json"),
        "global_seal_sha256": sha(tmp_path / "all_predictions_sealed.json")})
    scores = tmp_path / "scores"
    scores.mkdir()
    for name, frame in zip(("macro_metrics", "region_metrics", "interval_macro_metrics", "interval_region_metrics"), frames):
        frame.to_csv(scores / (name + ".csv"), index=False)
    for name in ("day_metrics.csv", "day_counts.csv", "interval_day_metrics.csv"):
        (scores / name).write_text("synthetic\n")
    if external:
        (scores / "fixed_primary_metrics.csv").write_text("synthetic\n")
    else:
        write(scores / "selected_config.json", {})
    write(scores / "postrun_verification.json", {"all_frozen_hashes_match": True,
        "target_label_access_after_global_seal": True,
        "global_seal_sha256": sha(tmp_path / "all_predictions_sealed.json"),
        "score_hashes": {path.name: sha(path) for path in scores.iterdir()}})
    _, _, verified_plan, _, _ = analysis.verified_scores(tmp_path, external=external, variant=variant)
    assert verified_plan["year"] == year
    assert len(verified_plan["origins"]) == (181 if external else 365)
    other_variant = next(value for value in analysis.VARIANTS if value != variant)
    with pytest.raises(ValueError, match="frozen grid"):
        analysis.verified_scores(tmp_path, external=external, variant=other_variant)
    plan["origins"] = plan["origins"][:-1]
    write(tmp_path / "evaluation_plan.json", plan)
    with pytest.raises(ValueError, match="exact source year or confirmation half-year"):
        analysis.verified_scores(tmp_path, external=external, variant=variant)
