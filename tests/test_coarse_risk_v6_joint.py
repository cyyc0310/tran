"""Joint choices are source-only and cannot be changed by target scores."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("joint_v6_test", ROOT / "scripts/analyze_coarse_risk_v6_joint.py")
J = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(J)


def bundles(tmp_path):
    result = []
    for index, variant in enumerate(J.VARIANTS):
        root = tmp_path / ("source" + str(index + 1))
        (root / "scores").mkdir(parents=True)
        (root / "fit").mkdir()
        (root / "fit/model.pkl").write_bytes(b"synthetic fitted model")
        J.V4.write_json(root / "run_config.json", {"variants": [variant],
            "calibration_modes": list(J.A.probability_calibrations(variant)),
            "methods": list(J.A.POINTS), "delays": list(J.A.DELAYS), "source_selection": None})
        fit = {"artifacts": {name: J.V4.sha(root / name) for name in ("fit/model.pkl", "run_config.json")}}
        J.V4.write_json(root / "fit_seal.json", fit)
        selections, ranks = {}, []
        for role in J.A.ROLES:
            calibration = {"primary": "structured_x1p25", "baseline": "global_x1p25",
                "unconstrained_baseline": "extratrees_x1", "historical_v5": "historical_v5_risk0p5", "cqr": "cqr"}[role]
            config = {"variant": variant, "method": J.A.POINTS[0], "calibration": calibration}
            metrics = {"crps": 10. + index, "empirical_coverage90": .9, "empirical_width90": 50.,
                "empirical_interval_score90": 15. + index, "empirical_lower_tail_miss": .05,
                "empirical_upper_tail_miss": .05, "empirical_path_coverage90": .5,
                "worst_region_delay_coverage90": .88, "gate_violation": 0.}
            selections[role] = {"status": "unconstrained" if role in ("unconstrained_baseline", "historical_v5") else "passed",
                                "configuration": config, "metrics": metrics}
            ranks.append({**config, **metrics, "gate_passed": True})
        noanchor = {**ranks[0], "calibration": "noanchor_x1p25", "crps": 999.}
        ranks.append(noanchor)
        selected = {"source_year": 2025, "selection_regions": list(J.A.SOURCES), "selections": selections}
        J.V4.write_json(root / "scores/selected_config.json", selected)
        result.append({"root": root, "variant": variant, "selection": selected,
            "ranked": pd.DataFrame(ranks), "interval_ranked": pd.DataFrame([ranks[-2]]),
            "metadata": {}, "selection_sha256": J.V4.sha(root / "scores/selected_config.json"),
            "frozen_code_protocol_hashes": {"protocol.md": "synthetic"}})
    return result


def external(tmp_path, bundle, *, snapshot=True, label_revision=False):
    root = tmp_path / ("external-" + bundle["variant"])
    (root / "predictions").mkdir(parents=True)
    (root / "fit").mkdir()
    (root / "fit/model.pkl").write_bytes((bundle["root"] / "fit/model.pkl").read_bytes())
    source_fit = J.A.read_json(bundle["root"] / "fit_seal.json")
    config = {"variants": [bundle["variant"]], "calibration_modes": list(J.A.probability_calibrations(bundle["variant"])),
        "methods": list(J.A.POINTS), "delays": list(J.A.DELAYS), "source_selection": bundle["selection"],
        "source_provenance": {"fitted_bytes_identical": True, "source_run": str(bundle["root"]),
            "source_fit_seal_sha256": J.V4.sha(bundle["root"] / "fit_seal.json"),
            "source_fit_artifact_hashes": source_fit["artifacts"]}}
    J.V4.write_json(root / "run_config.json", config)
    snapshot_hashes = {}
    if snapshot:
        folder = root / "snapshot/source"
        folder.mkdir(parents=True)
        J.V4.write_json(folder / "selected_config.json", bundle["selection"])
        snapshot_hashes["snapshot/source/selected_config.json"] = J.V4.sha(folder / "selected_config.json")
    label_data = tmp_path / "same-prepared-confirmation"
    label_hashes = {str(label_data / name): f"{i:064x}" for i, name in enumerate(
        ("monthly.csv", *(f"{region}_hourly.csv" for region in J.A.HOLDOUTS)))}
    if label_revision:
        label_hashes[str(label_data / "monthly.csv")] = "f" * 64
    J.V4.write_json(root / "prefit_freeze.json", {"code_hashes": {}, "snapshot_hashes": snapshot_hashes,
        "external_data": str(label_data), "data_hashes": label_hashes})
    fit = {"prefit_sha256": J.V4.sha(root / "prefit_freeze.json"),
           "artifacts": {name: J.V4.sha(root / name) for name in ("fit/model.pkl", "run_config.json")}}
    J.V4.write_json(root / "fit_seal.json", fit)
    expected = [(region, delay, bundle["variant"], method, calibration)
        for region in J.A.HOLDOUTS for delay in J.A.DELAYS for method in J.A.POINTS
        for calibration in (*J.A.probability_calibrations(bundle["variant"]), "cqr")]
    entries, artifacts = [], {}
    for i, key in enumerate(expected):
        name = f"predictions/{i}.npz"
        (root / name).write_bytes(b"synthetic sealed prediction bytes")
        artifacts[name] = J.V4.sha(root / name)
        entries.append({**dict(zip(("region", "delay", *J.A.KEYS), key)), "path": name,
                        "kind": "interval_only" if key[-1] == "cqr" else "probabilistic"})
    J.V4.write_json(root / "evaluation_plan.json", {"study": "holdout", "year": 2026, "targets": list(J.A.HOLDOUTS),
        "origins": [str(day) for day in pd.date_range("2026-01-01", "2026-06-30", tz="UTC")],
        "expected": expected, "selection": bundle["selection"], "label_data": str(label_data)})
    J.V4.write_json(root / "all_predictions_sealed.json", {"target_labels_read": False,
        "plan_sha256": J.V4.sha(root / "evaluation_plan.json"), "prefit_sha256": J.V4.sha(root / "prefit_freeze.json"),
        "fit_seal_sha256": J.V4.sha(root / "fit_seal.json"), "fit_artifacts": fit["artifacts"],
        "entries": entries, "expected": expected, "artifacts": artifacts})
    J.V4.write_json(root / "evaluation_commit.json", {"plan_sha256": J.V4.sha(root / "evaluation_plan.json"),
        "global_seal_sha256": J.V4.sha(root / "all_predictions_sealed.json")})
    # This poison label file is never parsed by joint fixation.
    (root / "unopened_hourly_labels.csv").write_text("not a valid target label table\n")
    return root


def test_passed_gate_beats_lower_source_score_and_ablation_is_exact(tmp_path):
    source = bundles(tmp_path)
    source[1]["selection"]["selections"]["primary"]["metrics"]["crps"] = .01
    source[1]["selection"]["selections"]["primary"]["status"] = "failed"
    source[1]["selection"]["selections"]["primary"]["metrics"]["gate_violation"] = .01
    roles = J.joint_roles(source)
    assert roles["primary"]["variant"] == "conditional_v6"
    assert roles["primary"]["status"] == "passed"
    assert roles["source2_noanchor"]["configuration"]["calibration"] == "noanchor_x1p25"
    assert roles["source2_noanchor"]["source_metrics"]["crps"] == 999.
    assert roles["source2_simple_anchor"]["configuration"]["calibration"] == "global_x1p25"
    assert roles["source2_simple_anchor"]["configuration"]["method"] == roles["source2_primary"]["configuration"]["method"]


def test_simple_anchor_keeps_primary_point_and_factor_when_generic_baseline_differs(tmp_path):
    source = bundles(tmp_path)
    source[1]["selection"]["selections"]["baseline"]["configuration"] = {
        "variant": "anchored_v6", "method": J.A.POINTS[1], "calibration": "global_x1"}
    source[1]["selection"]["selections"]["baseline"]["metrics"]["crps"] = .001
    roles = J.joint_roles(source)
    assert roles["baseline"]["configuration"]["method"] == J.A.POINTS[1]
    assert roles["source2_simple_anchor"]["configuration"] == {
        "variant": "anchored_v6", "method": J.A.POINTS[0], "calibration": "global_x1p25"}


def test_all_failed_minimizes_violation_then_score_and_retains_failed(tmp_path):
    source = bundles(tmp_path)
    for bundle, gap in zip(source, (.03, .01)):
        for role in ("primary", "baseline", "cqr"):
            bundle["selection"]["selections"][role]["status"] = "failed"
            bundle["selection"]["selections"][role]["metrics"]["gate_violation"] = gap
    roles = J.joint_roles(source)
    for role in ("primary", "baseline", "cqr"):
        assert roles[role]["variant"] == "anchored_v6"
        assert roles[role]["status"] == "failed"


def test_stable_ties_use_variant_then_method_and_calibration(tmp_path):
    source = bundles(tmp_path)
    for role in J.A.ROLES:
        source[1]["selection"]["selections"][role]["metrics"] = deepcopy(source[0]["selection"]["selections"][role]["metrics"])
    assert all(J.joint_roles(source)[role]["variant"] == "anchored_v6" for role in J.A.ROLES)


@pytest.mark.parametrize("variant_index", (0, 1))
def test_unscored_external_seal_complete_grids_and_scores_rejection(tmp_path, variant_index):
    bundle = bundles(tmp_path)[variant_index]
    root = external(tmp_path, bundle)
    verified = J.sealed_external(root, bundle, require_unscored=True)
    assert verified["prediction_count"] == (1242 if variant_index == 0 else 1782)
    assert verified["scores_absent_verified_at_freeze"]
    (root / "scores").mkdir()
    with pytest.raises(ValueError, match="scores directories to be absent"):
        J.sealed_external(root, bundle, require_unscored=True)


def test_external_prefit_must_embed_source_selection(tmp_path):
    bundle = bundles(tmp_path)[0]
    root = external(tmp_path, bundle, snapshot=False)
    with pytest.raises(ValueError, match="prefit snapshot"):
        J.sealed_external(root, bundle, require_unscored=True)


def test_prediction_mutation_is_detected_without_reading_labels(tmp_path):
    bundle = bundles(tmp_path)[0]
    root = external(tmp_path, bundle)
    (root / "predictions/0.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash binding"):
        J.sealed_external(root, bundle, require_unscored=True)


def test_joint_freeze_is_immutable_and_binds_both_unscored_runs(tmp_path, monkeypatch):
    source = bundles(tmp_path)
    target = [external(tmp_path, bundle) for bundle in source]
    monkeypatch.setattr(J, "source_bundle", lambda path, variant: next(bundle for bundle in source if bundle["variant"] == variant))
    protocol = tmp_path / "joint_protocol.md"
    protocol.write_text("Frozen joint mechanism comparison protocol.\n")
    monkeypatch.setattr(J, "PROTOCOL", protocol)
    output = tmp_path / "joint"
    frozen = J.freeze_joint([bundle["root"] for bundle in source], target, output)
    assert frozen["target_scores_read"] is False
    assert frozen["both_external_score_directories_absent"] is True
    assert frozen["protocol_sha256"] == J.V4.sha(protocol)
    assert (output / "snapshot/joint_protocol.md").read_bytes() == protocol.read_bytes()
    assert J.verify_joint_freeze(output) == frozen
    assert all(not (root / "scores").exists() for root in target)
    with pytest.raises(FileExistsError):
        J.freeze_joint([bundle["root"] for bundle in source], target, output)
    protocol.write_text("Changed after joint fixation.\n")
    with pytest.raises(ValueError, match="protocol changed after fixation"):
        J.verify_joint_freeze(output)
    (output / "joint_freeze.json").write_text("{}")
    with pytest.raises(ValueError, match="joint freeze hash"):
        J.verify_joint_freeze(output)


def test_joint_freeze_rejects_different_confirmation_revisions_without_reading_labels(tmp_path, monkeypatch):
    source = bundles(tmp_path)
    target = [external(tmp_path, bundle, label_revision=bool(index)) for index, bundle in enumerate(source)]
    monkeypatch.setattr(J, "source_bundle", lambda path, variant: next(bundle for bundle in source if bundle["variant"] == variant))
    with pytest.raises(ValueError, match="identical confirmation hourly and monthly data hashes"):
        J.freeze_joint([bundle["root"] for bundle in source], target, tmp_path / "joint")
    assert not (tmp_path / "joint").exists()


def test_analyze_uses_frozen_cross_variant_roles_even_when_target_prefers_another(tmp_path, monkeypatch):
    source = bundles(tmp_path)
    roles = J.joint_roles(source)
    output = tmp_path / "joint"
    output.mkdir()
    (output / "joint_freeze.json").write_text("synthetic immutable joint freeze")
    frozen = {"created_utc": "2020-01-01T00:00:00+00:00", "sources": [{"root": str(bundle["root"])} for bundle in source],
              "fixed_roles": roles, "external_runs": [], "uncertainty_scope": "six descriptive blocks",
              "protocol_sha256": "synthetic joint protocol", "confirmation_data_hashes": {"monthly.csv": "same"}}
    targets = {}
    for bundle in source:
        variant = bundle["variant"]
        root = tmp_path / ("target-scores-" + variant)
        scores = root / "scores"
        scores.mkdir(parents=True)
        (scores / "postrun_verification.json").write_text("{}")
        configs = {tuple(record["configuration"][key] for key in J.A.KEYS) for record in roles.values()
                   if record["variant"] == variant}
        records = []
        for config_tuple in configs:
            config = dict(zip(J.A.KEYS, config_tuple))
            for region in J.A.HOLDOUTS:
                for delay in J.A.DELAYS:
                    for month in range(1, 7):
                        metrics = {"empirical_coverage90": .9, "empirical_width90": 50., "empirical_interval_score90": 60.,
                            "empirical_lower_tail_miss": .05, "empirical_upper_tail_miss": .05, "empirical_path_coverage90": .5}
                        if config["calibration"] != "cqr":
                            metrics["crps"] = 100. if variant == "conditional_v6" else 1.
                            if variant == "anchored_v6" and config["calibration"].startswith("global_"):
                                metrics["crps"] = 5.
                            elif config["calibration"].startswith("noanchor_"):
                                metrics["crps"] = 4.
                        records.append({**config, "region": region, "delay": delay,
                            "origin": f"2026-{month:02d}-15T00:00Z", **metrics})
        frame = pd.DataFrame(records)
        for interval in (False, True):
            part = frame.loc[frame.calibration.eq("cqr") == interval].copy()
            if interval:
                part = part.drop(columns="crps")
            prefix = "interval_" if interval else ""
            part.to_csv(scores / (prefix + "day_metrics.csv"), index=False)
            numeric = [column for column in part.select_dtypes(include="number") if column != "delay"]
            part.groupby(["region", "delay", *J.A.KEYS])[numeric].mean().reset_index().to_csv(scores / (prefix + "region_metrics.csv"), index=False)
            part.groupby(["delay", *J.A.KEYS])[numeric].mean().reset_index().to_csv(scores / (prefix + "macro_metrics.csv"), index=False)
        targets[variant] = root
        frozen["external_runs"].append({"root": str(root), "source_selection_sha256": bundle["selection_sha256"],
            "confirmation_data_hashes": frozen["confirmation_data_hashes"],
            "scores_absent_verified_at_freeze": True, **{key: "bound" for key in
                ("global_prediction_seal_sha256", "prefit_sha256", "fit_seal_sha256", "evaluation_plan_sha256")}})
    monkeypatch.setattr(J, "verify_joint_freeze", lambda output: frozen)
    monkeypatch.setattr(J, "source_bundle", lambda path, variant: next(bundle for bundle in source if bundle["variant"] == variant))
    monkeypatch.setattr(J.A, "verified_scores", lambda root, **kwargs: (Path(root), Path(root) / "scores", {}, {}, {}))
    monkeypatch.setattr(J, "sealed_external", lambda root, bundle, **kwargs: next(item for item in frozen["external_runs"] if item["root"] == str(root)))
    monkeypatch.setattr(J.A, "paired_figure", lambda *args: None)
    monkeypatch.setattr(J.A, "external_analysis", lambda *args, **kwargs: {"external_selection_performed": False})
    monkeypatch.setattr(J, "plot_roles", lambda *args: None)
    J.analyze_joint(output)
    macro = pd.read_csv(output / "analysis/fixed_roles_macro_metrics.csv")
    primary = macro.loc[macro.role.eq("primary")].iloc[0]
    assert primary.variant == "conditional_v6" and primary.crps == 100.
    assert macro.loc[macro.role.eq("source2_primary"), "crps"].iloc[0] == 1.
    cqr = pd.read_csv(output / "analysis/joint_fixed_cqr_interval_comparison.csv")
    assert "crps" not in set(cqr.metric)
    comparison = pd.read_csv(output / "analysis/joint_fixed_paired_month_bootstrap.csv")
    assert comparison.n_bootstrap.eq(5000).all() and comparison.n_calendar_blocks.eq(6).all()
    simple_anchor = pd.read_csv(output / "analysis/source2_primary_minus_exact_simple_anchor.csv")
    assert set(simple_anchor.reference_role) == {"source2_simple_anchor"}
    assert simple_anchor.n_bootstrap.eq(5000).all() and simple_anchor.n_calendar_blocks.eq(6).all()
    assert simple_anchor.loc[simple_anchor.metric.eq("crps"), "difference_primary_minus_reference"].eq(-4).all()
    noanchor = pd.read_csv(output / "analysis/source2_primary_minus_exact_noanchor.csv")
    assert noanchor.loc[noanchor.metric.eq("crps"), "difference_primary_minus_reference"].eq(-3).all()
    for variant in J.VARIANTS:
        assert (output / "analysis/individual_rounds" / variant / "selection.json").is_file()
    with pytest.raises(FileExistsError):
        J.analyze_joint(output)
