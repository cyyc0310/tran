"""Synthetic producer-to-analysis integration; no real target metrics are read."""
import copy
import importlib.util
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fuel_v7_analysis_tests", ROOT / "scripts/analyze_fuel_transport_v7.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)
runner = analysis.RUNNER


def synthetic_days(stage):
    rows = []
    regions = runner.SOURCES if stage == "source" else runner.TARGETS
    year, months = (2025, range(1, 13)) if stage == "source" else (2026, range(1, 7))
    for config in runner.configs():
        mode = config["mode"]
        for factor in ((1.,) if mode == "cqr" else runner.FACTORS):
            for r, region in enumerate(regions):
                for delay in runner.DELAYS:
                    for month in months:
                        cost = {"recompose_joint": 10., "direct_analog": 8., "recompose_independent": 9.,
                                "pooled_recompose": 11., "fresh_anchor": 9.5, "extratrees": 9., "cqr": np.nan}[mode]
                        if config.get("bandwidth") == .3 and mode == "recompose_joint":
                            cost -= 1
                        if config.get("bandwidth") == .1 and mode == "direct_analog":
                            cost -= 1
                        if config["round"] == 2:
                            cost -= .5
                        if config.get("forecast") == "adaptive_mixture":
                            cost += .1
                        if config.get("temperature", .03) > .03:
                            cost += .2
                        if month > 6 and mode == "recompose_joint":
                            cost += 1000  # H2 cannot select a different primary.
                        if stage == "target":
                            cost = 50 - cost  # Target-optimal choice differs deliberately.
                        coverage = .92 if mode == "cqr" else {.75: .8, 1.: .89, 1.25: .92, 1.5: .98}[factor]
                        row = dict(region=region, delay=delay, round=config["round"], name=config["name"], mode=mode,
                            factor=factor, origin=f"{year}-{month:02d}-01 00:00:00+00:00", semester="H1" if month <= 6 else "H2",
                            crps=cost + month / 10 + r / 100 if mode != "cqr" else np.nan,
                            empirical_coverage90=coverage, empirical_width90=20 * factor,
                            empirical_interval_score90=25 * factor, empirical_lower_tail_miss=(1 - coverage) / 2,
                            empirical_upper_tail_miss=(1 - coverage) / 2, empirical_path_coverage90=coverage / 2,
                            point_level_mae=cost / 2, point_shape_mae=cost / 3)
                        rows.append(row)
    return pd.DataFrame(rows)


def save_scores(root, stage, days):
    folder = root / (stage + "_scores")
    folder.mkdir()
    metrics = [m for m in analysis.METRICS if m in days]
    days.to_csv(folder / "day_metrics.csv", index=False)
    grouped = days.groupby(analysis.CELL)
    region = grouped[metrics].mean().reset_index()
    region["days"] = grouped.size().to_numpy()
    region.to_csv(folder / "region_metrics.csv", index=False)
    region.groupby([*analysis.IDENTITY, "semester"])[metrics].mean().reset_index().to_csv(folder / "macro_metrics.csv", index=False)
    days.groupby(["region", "round", "name", "delay", "factor"]).size().reset_index(name="days").to_csv(folder / "day_counts.csv", index=False)
    analysis.write_json(folder / "score_seal.json", {name: analysis.sha(folder / name) for name in analysis.SCORE_FILES})
    return pd.read_csv(folder / "region_metrics.csv")


def save_predictions(root, stage):
    regions = runner.SOURCES if stage == "source" else runner.TARGETS
    entries = []
    for region in regions:
        for delay in runner.DELAYS:
            for config in runner.configs():
                path = root / "predictions" / f"{stage}-{region}-{delay}-r{config['round']}-{config['name']}.npz"
                path.write_bytes(b"synthetic archive bytes; tests never parse outcomes")
                support = {} if config["mode"] in ("fresh_anchor", "extratrees", "cqr") else {
                    "unsupported_target_mass": .1, "mean_effective_donors": 20., "min_effective_donors": 3.,
                    "zero_generation_fallback_fraction": 0., "ess_scope": "Expert-weighted component ESS; NOT exact mixture ESS"}
                entries.append(dict(region=region, delay=delay, **config, path=str(path.relative_to(root)), sha256=analysis.sha(path),
                                    support=support, interval_only=config["mode"] == "cqr"))
    if stage == "source":
        fits = {f"lodo-{r}-{d}.pkl" for r in runner.SOURCES for d in runner.DELAYS} | {f"full-{d}.pkl" for d in runner.DELAYS}
        for name in fits:
            (root / "fits" / name).write_bytes(b"synthetic fit; never unpickled")
        seal = dict(entries=entries, fits={name: analysis.sha(root / "fits" / name) for name in fits})
    else:
        seal = dict(entries=entries, created_utc="2026-10-23T10:00:01+00:00", source_choices_sha256=analysis.sha(root / "source_choices.json"),
                    all_grid_complete=True, target_labels_read_in_this_prediction_phase=False, target_already_viewed_in_V6=True)
    analysis.write_json(root / (stage + "_prediction_seal.json"), seal)


@pytest.fixture(scope="module")
def complete_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic-v7-run")
    (root / "predictions").mkdir()
    (root / "fits").mkdir()
    dummy_input = root / "input.csv"
    dummy_input.write_text("Frozen dummy input, no labels\n")
    analysis.write_json(root / "prefit_freeze.json", dict(code={str(analysis.RUNNER_PATH): analysis.sha(analysis.RUNNER_PATH)},
        inputs={str(dummy_input): analysis.sha(dummy_input)}, configs=runner.configs(), delays=list(runner.DELAYS), factors=list(runner.FACTORS)))
    save_predictions(root, "source")
    table = save_scores(root, "source", synthetic_days("source"))
    choices, ranking = runner.select(table)
    ranking.to_csv(root / "source_scores/ranking.csv", index=False)
    analysis.write_json(root / "source_choices.json", dict(created_utc="2026-10-23T10:00:00+00:00", selections=choices,
        source_score_hashes={p.name: analysis.sha(p) for p in (root / "source_scores").iterdir()},
        source_seal_sha256=analysis.sha(root / "source_prediction_seal.json"), target_model_scores_read=False))
    save_predictions(root, "target")
    save_scores(root, "target", synthetic_days("target"))
    analysis.write_json(root / "completed.json", dict(created_utc="2026-10-23T10:00:02+00:00",
        source_choices_sha256=analysis.sha(root / "source_choices.json"), target_seal_sha256=analysis.sha(root / "target_prediction_seal.json"),
        target_score_seal_sha256=analysis.sha(root / "target_scores/score_seal.json"), frozen_hashes_verified=True))
    return root


def test_actual_runner_source_choices_pass_full_analysis_verification(complete_run):
    verified = analysis.verify_run(complete_run)
    actual, _ = runner.select(verified["source"]["region"])
    assert verified["roles"] == actual
    for scope in ("1", "2", "joint"):
        assert actual[scope]["primary"]["status"] == "passed"
        assert actual[scope]["unconstrained_baseline"]["status"] == "unconstrained"
        assert actual[scope]["primary"]["configuration"]["factor"] == 1.25
    assert actual["joint"]["primary"]["configuration"]["round"] == 2


def test_failed_roles_cannot_be_disguised_as_passed(complete_run):
    choice = analysis.read_json(complete_run / "source_choices.json")
    table = pd.read_csv(complete_run / "source_scores/region_metrics.csv")
    altered = copy.deepcopy(choice)
    altered["selections"]["joint"]["primary"]["status"] = "failed"
    with pytest.raises(ValueError, match="choices differ"):
        analysis.validate_selection(complete_run, altered, table)
    table["empirical_coverage90"] = .7
    selected, _ = runner.select(table)
    assert selected["joint"]["primary"]["status"] == "failed"
    assert selected["joint"]["unconstrained_baseline"]["status"] == "unconstrained"


def test_source_h2_stress_does_not_change_selection(complete_run):
    table = pd.read_csv(complete_run / "source_scores/region_metrics.csv")
    expected, _ = runner.select(table)
    table.loc[table.semester.eq("H2"), "crps"] = -1e8
    assert runner.select(table)[0] == expected
    scores = analysis.verified_scores(complete_run, "source")
    macro, _, _ = analysis.fixed_tables(scores, expected)
    h1 = macro.query("scope == '2' and role == 'primary' and semester == 'H1'").iloc[0]
    h2 = macro.query("scope == '2' and role == 'primary' and semester == 'H2'").iloc[0]
    assert h2.crps > h1.crps + 900


@pytest.mark.parametrize("field", ["source_choices_sha256", "target_seal_sha256", "target_score_seal_sha256"])
def test_completion_hash_binding_is_required(complete_run, tmp_path, field):
    # Only manifests are copied; archives resolve inside the copied root via symlinks.
    root = tmp_path / "run"
    shutil.copytree(complete_run, root, symlinks=True)
    complete = analysis.read_json(root / "completed.json")
    complete[field] = "0" * 64
    analysis.write_json(root / "completed.json", complete)
    with pytest.raises(ValueError, match="binding differs"):
        analysis.verify_run(root)


def test_source_score_and_prediction_byte_changes_are_rejected(complete_run, tmp_path):
    root = tmp_path / "run"
    shutil.copytree(complete_run, root)
    score = root / "source_scores/macro_metrics.csv"
    score.write_bytes(score.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Changed sealed bytes"):
        analysis.verified_scores(root, "source")
    seal = analysis.read_json(root / "target_prediction_seal.json")
    (root / seal["entries"][0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Prediction bytes changed"):
        analysis.verified_prediction_seal(root, "target")


def test_identity_keeps_round_mode_and_factor_distinct(complete_run):
    scores = analysis.verified_scores(complete_run, "target")
    config = dict(round=2, name="adaptive_mean-t0.03-recompose_joint-b0.3", factor=1.25)
    group = analysis.subset(scores["day"], config)
    assert group["round"].unique().tolist() == [2]
    assert group["mode"].unique().tolist() == ["recompose_joint"]
    assert group.factor.unique().tolist() == [1.25]
    other = analysis.subset(scores["day"], {**config, "factor": .75})
    assert not np.isclose(group.empirical_width90.mean(), other.empirical_width90.mean())


def test_bootstrap_exact_controls_and_cqr_remain_separate(complete_run):
    scores = analysis.verified_scores(complete_run, "target")
    roles = analysis.read_json(complete_run / "source_choices.json")["selections"]
    comparisons = analysis.planned_comparisons(roles)
    exact = next(c for c in comparisons if c["scope"] == "2" and c["comparison"] == "primary_minus_exact_independent")
    assert exact["primary"]["name"].replace("recompose_joint", "recompose_independent") == exact["reference"]["name"]
    assert exact["primary"]["factor"] == exact["reference"]["factor"]
    result = analysis.paired_month_bootstrap(scores["day"], exact["primary"], exact["reference"], ["crps"], n_bootstrap=100)
    overall = result[result.stratum.eq("overall")].iloc[0]
    assert overall.difference_primary_minus_reference == pytest.approx(.0, abs=1e-12)
    assert overall.ci95_low == pytest.approx(overall.ci95_high, abs=1e-12)
    assert overall.n_calendar_blocks == 6
    cqr = next(c for c in comparisons if c["scope"] == "joint" and c["comparison"] == "primary_minus_cqr")
    assert "crps" not in cqr["metrics"]
    interval = analysis.paired_month_bootstrap(scores["day"], cqr["primary"], cqr["reference"], cqr["metrics"], n_bootstrap=100)
    assert "crps" not in interval.metric.values
    with pytest.raises(ValueError, match="CQR has no CRPS"):
        analysis.paired_month_bootstrap(scores["day"], cqr["primary"], cqr["reference"], ["crps"], n_bootstrap=100)


def test_partial_pairing_is_rejected_and_support_ess_scope_preserved(complete_run):
    scores = analysis.verified_scores(complete_run, "target")
    roles = analysis.read_json(complete_run / "source_choices.json")["selections"]
    p, r = roles["1"]["primary"]["configuration"], roles["1"]["exact_direct"]["configuration"]
    days = scores["day"]
    index = analysis.subset(days, r).index[0]
    with pytest.raises(ValueError, match="different scored days"):
        analysis.paired_month_bootstrap(days.drop(index), p, r, ["crps"], n_bootstrap=100)
    _, fixed = analysis.support_tables(analysis.read_json(complete_run / "target_prediction_seal.json"), roles)
    assert "NOT exact mixture ESS" in fixed.ess_scope.dropna().iloc[0]
    assert fixed.query("scope == '2' and role == 'primary'").factor.unique().tolist() == [1.25]


def test_complete_analysis_writes_fixed_roles_without_target_selection(complete_run, tmp_path):
    output = tmp_path / "analysis"
    meta = analysis.analyze(complete_run, output)
    assert meta["target_selection_performed"] is False
    assert meta["bootstrap"]["n_bootstrap"] == 5000
    assert len(list(output.glob("*_fixed_role_summary.png"))) == 3
    macro = pd.read_csv(output / "target_fixed_roles_macro.csv")
    assert macro[macro.role.eq("cqr")].crps.isna().all()
    assert set(macro.scope.astype(str)) == {"1", "2", "joint"}
    selected = meta["selections"]["joint"]["primary"]["configuration"]
    assert macro.query("scope == 'joint' and role == 'primary'")["name"].iloc[0] == selected["name"]
    for name, digest in meta["output_hashes"].items():
        assert analysis.sha(output / name) == digest
    with pytest.raises(FileExistsError):
        analysis.analyze(complete_run, output)


def test_source_only_does_not_require_or_read_target_artifacts(complete_run, tmp_path):
    root = tmp_path / "run"
    shutil.copytree(complete_run, root)
    (root / "completed.json").unlink()
    (root / "target_prediction_seal.json").unlink()
    shutil.rmtree(root / "target_scores")
    meta = analysis.analyze(root, tmp_path / "source-analysis", source_only=True)
    assert meta["source_only"] is True
    assert "target_prediction_seal_sha256" not in meta


def test_historical_v6_background_requires_same_data_and_exact_day_pairing(complete_run, tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    frozen = analysis.read_json(complete_run / "prefit_freeze.json")
    confirmation_hashes = {}
    frozen["inputs"] = {}
    for region in runner.TARGETS:
        path = root / f"{region}_hourly.csv"
        path.write_text("Synthetic labels are not parsed\n")
        confirmation_hashes[path.name] = analysis.sha(path)
        frozen["inputs"][str(path)] = analysis.sha(path)
    analysis.write_json(root / "prefit_freeze.json", frozen)
    scores = analysis.verified_scores(complete_run, "target")
    roles = analysis.read_json(complete_run / "source_choices.json")["selections"]
    days = analysis.subset(scores["day"], roles["joint"]["primary"]["configuration"])
    joint = tmp_path / "joint"
    joint.mkdir()
    freeze = dict(fixed_roles={}, external_runs=[], confirmation_data_hashes=confirmation_hashes)
    for role, variant in (("primary", "conditional_v6"), ("source2_simple_anchor", "anchored_v6")):
        run = tmp_path / variant
        (run / "scores").mkdir(parents=True)
        analysis.write_json(run / "all_predictions_sealed.json", {"synthetic": True})
        config = dict(variant=variant, method="fixed_point", calibration="fixed_spread")
        old = days[["region", "delay", "origin", "crps", *analysis.INTERVAL]].assign(**config)
        path = run / "scores/day_metrics.csv"
        old.to_csv(path, index=False)
        seal_hash = analysis.sha(run / "all_predictions_sealed.json")
        analysis.write_json(run / "scores/postrun_verification.json", dict(all_frozen_hashes_match=True,
            target_label_access_after_global_seal=True, global_seal_sha256=seal_hash, score_hashes={"day_metrics.csv": analysis.sha(path)}))
        freeze["fixed_roles"][role] = dict(configuration=config)
        freeze["external_runs"].append(dict(root=str(run), variant=variant, global_prediction_seal_sha256=seal_hash))
    analysis.write_json(joint / "joint_freeze.json", freeze)
    (joint / "joint_freeze.sha256").write_text(analysis.sha(joint / "joint_freeze.json") + "  joint_freeze.json\n")
    output = tmp_path / "history-analysis"
    output.mkdir()
    hashes = analysis.historical_background(root, scores["day"], roles, output, joint)
    assert str(joint / "joint_freeze.json") in hashes
    result = pd.read_csv(output / "historical_v6_same_day_background.csv")
    assert np.allclose(result.difference_primary_minus_reference, 0)
    assert result.interpretation.str.contains("different training budgets").all()
    frozen["inputs"][str(root / "AZPS_hourly.csv")] = "0" * 64
    analysis.write_json(root / "prefit_freeze.json", frozen)
    with pytest.raises(ValueError, match="hourly data differ"):
        analysis.historical_background(root, scores["day"], roles, output, joint)
    frozen["inputs"][str(root / "AZPS_hourly.csv")] = confirmation_hashes["AZPS_hourly.csv"]
    analysis.write_json(root / "prefit_freeze.json", frozen)
    old_daily = tmp_path / "conditional_v6/scores/day_metrics.csv"
    old_daily.write_bytes(old_daily.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Changed sealed bytes"):
        analysis.historical_background(root, scores["day"], roles, output, joint)
