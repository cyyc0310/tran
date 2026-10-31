"""Source-only V5 reference fixation and sealed four-region external analysis.

    python scripts/analyze_risk_transfer_v5.py --source results/risk-transfer-v5-source2024 \
        --output results/risk-transfer-v5-source-analysis
    python scripts/analyze_risk_transfer_v5.py --source results/risk-transfer-v5-source2024 \
        --external results/risk-transfer-v5-external2025 \
        --baseline-selection results/risk-transfer-v5-source-analysis/source_only_baseline_selection.json \
        --output results/risk-transfer-v5-external-analysis

Reads sealed score CSVs and protocol metadata only, never raw target labels.
Risk-beta tables are descriptive; external scores cannot change fixed choices.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


HELPER_PATH = Path(__file__).with_name("analyze_transfer_v4.py")
_SPEC = importlib.util.spec_from_file_location("risk_v5_analysis_helpers", HELPER_PATH)
V4 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(V4)
SOURCES = ("BPAT", "CISO", "ERCO", "FPL", "ISNE", "MISO", "NYIS")
HOLDOUTS = ("SCL", "WACM", "SEC", "SCEG")
POINTS = ("support_gp_medium", "harmonic_ridge", "softmax_moe")
DELAYS = (0, 30, 90)
BETAS = {"seven_raw": None, "seven_component110": None,
         "adaptive_component110_beta0p5": .5, "adaptive_component110_beta1": 1.,
         "adaptive_component110_beta2": 2.}
BASELINE_MODES = ("seven_raw", "seven_component110")


def verified_scores(path, *, external=False):
    """Verify V5 score hashes and pre-label global metadata without raw inputs."""
    root = V4.run_root(path)
    plan = json.loads((root / "evaluation_plan.json").read_text())
    targets, year, study = (HOLDOUTS, 2025, "holdout") if external else (SOURCES, 2024, "development")
    if (plan.get("study"), plan.get("year"), tuple(plan.get("targets", ()))) != (study, year, targets):
        raise ValueError("V5 analysis requires the exact predeclared regions/year/study.")
    config = json.loads((root / "run_config.json").read_text())
    expected_config = {"study_version": 5, "variants": ["risk_v5"], "methods": list(POINTS),
                       "delays": list(DELAYS), "calibration_modes": list(BETAS),
                       "calibration_regions": list(SOURCES), "calibration_year": 2023,
                       "selection_regions": list(SOURCES), "selection_year": 2024,
                       "external_regions": list(HOLDOUTS), "external_year": 2025,
                       "fit_regions": ["BPAT", "ERCO", "MISO", "NYIS"], "fit_year": 2022}
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise ValueError("Frozen V5 model/grid/split metadata differs from the declared study.")
    expected = {(region, delay, "risk_v5", method, calibration) for region in targets
                for delay in DELAYS for method in POINTS for calibration in BETAS}
    seal = json.loads((root / "all_predictions_sealed.json").read_text())
    for value, path in ((seal.get("plan_sha256"), root / "evaluation_plan.json"),
                        (seal.get("prefit_sha256"), root / "prefit_freeze.json"),
                        (seal.get("fit_seal_sha256"), root / "fit_seal.json")):
        if value != V4.sha(path):
            raise ValueError("V5 global prediction seal metadata changed.")
    entries = seal.get("entries", [])
    entry_keys = [tuple(entry[key] for key in ("region", "delay", *V4.KEYS)) for entry in entries]
    if (seal.get("target_labels_read") is not False or len(entries) != len(expected)
            or len(set(entry_keys)) != len(entries) or set(entry_keys) != expected
            or {tuple(key) for key in seal.get("expected", [])} != expected
            or {tuple(key) for key in plan.get("expected", [])} != expected
            or {entry["path"] for entry in entries} != set(seal.get("artifacts", {}))):
        raise ValueError("V5 prediction grid is incomplete, duplicate, changed or lacks its pre-label seal.")
    commit = json.loads((root / "evaluation_commit.json").read_text())
    if (commit.get("plan_sha256") != V4.sha(root / "evaluation_plan.json")
            or commit.get("global_seal_sha256") != V4.sha(root / "all_predictions_sealed.json")):
        raise ValueError("V5 evaluation commit differs from global seal.")
    fit_seal = json.loads((root / "fit_seal.json").read_text())
    if fit_seal.get("prefit_sha256") != V4.sha(root / "prefit_freeze.json"):
        raise ValueError("V5 fitted artifacts lack their prefit binding.")
    for name, digest in fit_seal.get("artifacts", {}).items():
        if V4.sha(root / name) != digest:
            raise ValueError("V5 fitted bytes changed after selection.")
    scores = root / "scores"
    verification = json.loads((scores / "postrun_verification.json").read_text())
    if (verification.get("all_frozen_hashes_match") is not True
            or verification.get("target_label_access_after_global_seal") is not True
            or verification.get("global_seal_sha256") != V4.sha(root / "all_predictions_sealed.json")):
        raise ValueError("V5 score postrun verification failed.")
    required = {"macro_metrics.csv", "region_metrics.csv", "day_metrics.csv"}
    if external:
        required.add("fixed_primary_metrics.csv")
    hashes = verification.get("score_hashes", {})
    if not required.issubset(hashes):
        raise ValueError("V5 score hash verification is incomplete.")
    for name, digest in hashes.items():
        if Path(name).name != name or V4.sha(scores / name) != digest:
            raise ValueError("V5 scored CSV changed: " + name)
    for filename, keys, expected_keys in (
        ("macro_metrics.csv", ["delay", *V4.KEYS], {key[1:] for key in expected}),
        ("region_metrics.csv", ["region", "delay", *V4.KEYS], expected),
    ):
        frame = pd.read_csv(scores / filename, usecols=keys)
        if len(frame) != len(expected_keys) or set(frame[keys].itertuples(index=False, name=None)) != expected_keys:
            raise ValueError("V5 scored macro/region support differs from the globally sealed grid.")
    return root, scores, plan


def baseline_selection(ranked):
    """Only two predeclared seven-source transports compete as references."""
    baseline = ranked.loc[ranked.calibration.isin(BASELINE_MODES)]
    if len(baseline) != len(POINTS) * len(BASELINE_MODES):
        raise ValueError("Missing or extra V5 baseline candidates.")
    passing = baseline.loc[baseline.gate_passed].sort_values(["crps", *V4.KEYS], kind="stable")
    closest = baseline.sort_values(["gate_violation", "crps", *V4.KEYS], kind="stable").iloc[0]
    unconstrained = baseline.sort_values(["crps", *V4.KEYS], kind="stable").iloc[0]
    winner = None if passing.empty else passing.iloc[0]
    return {"status": "constraint_failed" if winner is None else "passed",
            "configuration": None if winner is None else V4.configuration(winner),
            "reference_configuration": V4.configuration(closest if winner is None else winner),
            "reference_role": "coverage_failed_reference" if winner is None else "source_selected_qualified_baseline",
            "unconstrained_configuration": V4.configuration(unconstrained),
            "unconstrained_role": "source_only_unconstrained_crps_baseline",
            "best_by_violation": {"configuration": V4.configuration(closest),
                                  "gate_violation": float(closest.gate_violation),
                                  "worst_region_delay_coverage90": float(closest.worst_region_delay_coverage90)},
            "baseline_methods": list(POINTS), "baseline_calibrations": list(BASELINE_MODES),
            "coverage_gap_limit": .05, "worst_region_delay_floor": .85,
            "selection_rule": "Source equal-region/equal-delay coverage gate, then minimum CRPS; failed reference is diagnostic only"}


def descriptive_tables(macro, ranked, output, prefix):
    table = ranked.copy()
    table["risk_beta"] = table.calibration.map(BETAS)
    table["is_baseline"] = table.calibration.isin(BASELINE_MODES)
    table["descriptive_only"] = True
    table.to_csv(output / f"{prefix}_risk_beta_descriptive.csv", index=False)
    metrics = [metric for metric in V4.METRICS if metric in macro]
    keys = ["delay", "variant", "method"]
    reference = macro.loc[macro.calibration.eq("seven_component110"), keys + metrics]
    risk = macro.loc[~macro.calibration.isin(BASELINE_MODES)]
    merged = risk.merge(reference, on=keys, suffixes=("_risk", "_reference"), validate="many_to_one")
    result = merged[keys + ["calibration"]].copy()
    result["risk_beta"] = result.calibration.map(BETAS)
    for metric in metrics:
        result[metric + "_delta_risk_minus_component"] = merged[metric + "_risk"] - merged[metric + "_reference"]
    result.to_csv(output / f"{prefix}_risk_vs_component.csv", index=False)


def region_figure(regional, output, prefix, primary, reference):
    import matplotlib.pyplot as plt
    # Every predeclared beta for the fixed primary point, plus the fixed reference.
    configs = [(str(calibration), {**primary, "calibration": calibration}) for calibration in BETAS]
    if reference not in [config for _, config in configs]:
        configs.append(("Source-fixed reference: " + reference["method"], reference))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    labels = ["Hourly coverage", "90% interval width (gCO2/kWh)", "CRPS (gCO2/kWh)"]
    colors = ["#000000", "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]
    for i, (label, config) in enumerate(configs):
        group = V4.subset(regional, config).groupby("region", sort=True).mean(numeric_only=True)
        for ax, metric, ylabel in zip(axes, ["empirical_coverage90", "empirical_width90", "crps"], labels):
            ax.plot(group.index, group[metric], marker=["o", "s", "^", "D", "v", "x"][i],
                    color=colors[i], linewidth=1, label=label)
            ax.set_ylabel(ylabel)
            ax.tick_params(axis="x", rotation=35)
    axes[0].axhline(.9, color="black", linestyle="--", linewidth=.7)
    axes[0].legend(frameon=False, fontsize=6.5, loc="best")
    fig.suptitle(f"{prefix}: all predeclared betas for fixed point {primary['method']}; descriptive", fontsize=10)
    V4.save_figure(fig, output, prefix + "_by_region")


def source_analysis(path, output):
    root, scores, plan = verified_scores(path)
    macro, regional = pd.read_csv(scores / "macro_metrics.csv"), pd.read_csv(scores / "region_metrics.csv")
    ranked = V4.rank_candidates(macro, regional)
    selected_path = scores / "selected_config.json"
    selected = json.loads(selected_path.read_text())
    for name, field in (("macro_metrics.csv", "macro_metrics_sha256"), ("region_metrics.csv", "region_metrics_sha256")):
        if selected.get(field) != V4.sha(scores / name):
            raise ValueError("V5 source primary selection score binding changed.")
    passing = ranked.loc[ranked.gate_passed].sort_values(["crps", *V4.KEYS], kind="stable")
    primary = None if passing.empty else V4.configuration(passing.iloc[0])
    if (selected.get("status") != ("failed" if primary is None else "passed")
            or selected.get("configuration") != primary
            or tuple(selected.get("selection_regions", ())) != SOURCES
            or selected.get("selection_year") != 2024):
        raise ValueError("V5 fixed primary does not match the source-only seven-region gate.")
    baseline = baseline_selection(ranked)
    ranked.to_csv(output / "source_candidate_ranking.csv", index=False)
    descriptive_tables(macro, ranked, output, "source")
    V4.figures(ranked, output, "source_v5", primary=primary, reference=baseline["reference_configuration"])
    if primary is not None:
        region_figure(regional, output, "source_v5", primary, baseline["reference_configuration"])
    metadata = {"root": str(root), "selection_regions": list(SOURCES),
                "artifacts": {str(scores / name): V4.sha(scores / name) for name in
                              ("macro_metrics.csv", "region_metrics.csv", "selected_config.json")}}
    return {"study_version": 5, "created_utc": datetime.now(timezone.utc).isoformat(),
            "analysis_script_sha256": V4.sha(__file__), "analysis_helper_sha256": V4.sha(HELPER_PATH),
            "final_source_root": str(root), "source_rounds": [metadata],
            "primary_configuration": primary, "primary_status": selected["status"],
            "primary_selection_sha256": V4.sha(selected_path), "selection_year": 2024,
            "target_scores_used": False, "external_regions": list(HOLDOUTS),
            "calibration_split": "2023 temporal separation; source domains overlap the original 2022 point-fit domains",
            **baseline}


def external_analysis(path, baseline_path, expected, output, n_bootstrap, seed):
    root = V4.run_root(path)
    frozen = V4.validate_frozen_baseline(baseline_path, expected, root)
    if frozen.get("study_version") != 5 or frozen.get("external_regions") != list(HOLDOUTS):
        raise ValueError("Expected the pre-fixed V5 four-region baseline object.")
    if frozen.get("analysis_helper_sha256") != V4.sha(HELPER_PATH):
        raise ValueError("V4 analysis helper changed after the V5 baseline freeze.")
    root, scores, plan = verified_scores(root, external=True)
    primary = frozen["primary_configuration"]
    if primary is None or plan.get("selection", {}).get("configuration") != primary:
        raise ValueError("External evaluation lacks a passed fixed source-only primary.")
    fixed = pd.read_csv(scores / "fixed_primary_metrics.csv")
    if len(V4.subset(fixed, primary)) != len(fixed) or set(fixed.delay) != set(DELAYS):
        raise ValueError("V5 fixed primary metrics changed configuration or delay support.")
    macro, regional = pd.read_csv(scores / "macro_metrics.csv"), pd.read_csv(scores / "region_metrics.csv")
    ranked = V4.rank_candidates(macro, regional)
    ranked.to_csv(output / "external_descriptive_candidates.csv", index=False)
    descriptive_tables(macro, ranked, output, "external")
    # Only two fixed reference families are needed from the potentially large daily CSV.
    references = [(frozen["reference_role"], frozen["reference_configuration"]),
                  (frozen["unconstrained_role"], frozen["unconstrained_configuration"])]
    needed = [primary, *(config for _, config in references)]
    chunks = []
    for chunk in pd.read_csv(scores / "day_metrics.csv", chunksize=50000):
        mask = np.zeros(len(chunk), dtype=bool)
        for config in needed:
            mask |= np.logical_and.reduce([chunk[key].eq(value) for key, value in config.items()])
        chunks.append(chunk.loc[mask])
    days = pd.concat(chunks, ignore_index=True)
    for config in needed:
        selected = V4.subset(days, config)
        if set(zip(selected.region, selected.delay)) != {(region, delay) for region in HOLDOUTS for delay in DELAYS}:
            raise ValueError("V5 paired daily score support omits a fixed external region/delay.")
    comparisons = []
    for role, reference in references:
        table = V4.paired_month_bootstrap(days, primary, reference, n_bootstrap=n_bootstrap, seed=seed)
        table.insert(0, "reference_role", role)
        for key, value in reference.items():
            table["reference_" + key] = value
        comparisons.append(table)
    combined = pd.concat(comparisons, ignore_index=True)
    combined.to_csv(output / "external_fixed_paired_month_bootstrap.csv", index=False)
    fixed.to_csv(output / "external_fixed_primary_metrics.csv", index=False)
    V4.figures(ranked, output, "external_v5", primary=primary, reference=frozen["reference_configuration"])
    V4.paired_figure(combined, output)
    region_figure(regional, output, "external_v5", primary, frozen["reference_configuration"])
    V4.write_json(output / "external_analysis_metadata.json", {
        "study_version": 5, "baseline_selection_sha256": V4.sha(baseline_path),
        "global_prediction_seal_sha256": V4.sha(root / "all_predictions_sealed.json"),
        "external_selection_performed": False, "risk_beta_tables": "descriptive only; all predeclared betas retained",
        "bootstrap": "synchronous 12 calendar-month blocks; region-delay sum/count then equal-group macro",
        "uncertainty_scope": "conditional on the four fixed observed regions; unadjusted percentile 95% intervals",
        "reference_status": frozen["status"], "n_bootstrap": n_bootstrap, "seed": seed})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--external", type=Path)
    parser.add_argument("--baseline-selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=17023)
    args = parser.parse_args(argv)
    if args.external is not None and args.baseline_selection is None:
        parser.error("--external requires a source-only --baseline-selection fixed before external freeze")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output / ".matplotlib-cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(args.output / ".cache"))
    expected = source_analysis(args.source, args.output)
    if args.external is None:
        path = args.output / "source_only_baseline_selection.json"
        V4.write_json(path, expected)
        path.with_suffix(".sha256").write_text(V4.sha(path) + "  " + path.name + "\n")
        print(json.dumps({"baseline_selection": str(path.resolve()), "sha256": V4.sha(path),
                          "status": expected["status"], "primary": expected["primary_configuration"],
                          "reference": expected["reference_configuration"]}))
    else:
        external_analysis(args.external, args.baseline_selection, expected, args.output, args.bootstrap, args.seed)
        print(json.dumps({"analysis": str(args.output.resolve()), "external_selection_performed": False}))


if __name__ == "__main__":
    main()
