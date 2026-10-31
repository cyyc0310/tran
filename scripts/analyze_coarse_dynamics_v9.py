"""Source-fixed V9 analysis and seen-2026H1 development continuation gate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    try:
        spec.loader.exec_module(value)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return value


RUNNER_PATH = ROOT / "scripts/run_coarse_dynamics_v9.py"
RUNNER = module("v9_runner_for_analysis", RUNNER_PATH)
V7 = RUNNER.A7
INTERVAL = V7.INTERVAL
METRICS = V7.METRICS
FOCUS = ("LGEE", "PACE", "AZPS", "PSCO")
ROLES = ("primary", "baseline", "simple_anchor", "standard_dynamic")


def subset(table, config):
    return V7.subset(table, config)


def verified_scores(root: Path, stage: str) -> dict[str, Any]:
    RUNNER.verify_phase(root, stage)
    folder = root / f"{stage}_scores"
    day = pd.read_csv(folder / "day_metrics.csv")
    region = pd.read_csv(folder / "region_metrics.csv")
    macro = pd.read_csv(folder / "macro_metrics.csv")
    if day.empty or day.duplicated(["region", "delay", "round", "name", "factor", "origin"]).any():
        raise ValueError("Empty or duplicated V9 score days")
    metrics = [m for m in METRICS if m in day.columns]
    if not set(["crps", *INTERVAL]).issubset(metrics) or not np.isfinite(day[metrics]).all().all():
        raise ValueError("Missing or nonfinite V9 scores")
    keys = ["region", "delay", "round", "name", "mode", "factor", "semester"]
    grouped = day.groupby(keys, sort=True)
    expected = grouped[metrics].mean().reset_index()
    expected["days"] = grouped.size().to_numpy()
    V7.assert_equal_table(region, expected, keys, [*metrics, "days"])
    V7.assert_equal_table(macro, region.groupby(["round", "name", "mode", "factor", "semester"])[metrics].mean().reset_index(),
                          ["round", "name", "mode", "factor", "semester"], metrics)
    return {"day": day, "region": region, "macro": macro, "metrics": metrics}


def fixed_tables(scores: dict[str, Any], roles: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    macros, regional, by_region = [], [], []
    for scope, selected in roles.items():
        for role in ROLES:
            record = selected[role]
            config = record["configuration"]
            cells = subset(scores["region"], config).assign(scope=scope, role=role, source_status=record["status"])
            if cells.empty:
                raise ValueError("Missing fixed role scores")
            regional.append(cells)
            for semester, half in cells.groupby("semester", sort=True):
                context = {"scope": scope, "role": role, "source_status": record["status"],
                           **config, "semester": semester}
                macros.append({**context, **half[scores["metrics"]].mean().to_dict(),
                               "worst_coverage": float(half.empirical_coverage90.min()),
                               "n_region_delay_groups": len(half), "n_scored_days": int(half.days.sum())})
                for region, group in half.groupby("region", sort=True):
                    by_region.append({**context, "region": region,
                                      **group[scores["metrics"]].mean().to_dict(),
                                      "worst_coverage": float(group.empirical_coverage90.min()),
                                      "n_delay_groups": len(group), "n_scored_days": int(group.days.sum())})
    return pd.DataFrame(macros), pd.concat(regional, ignore_index=True), pd.DataFrame(by_region)


def planned_comparisons(roles: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for scope, selected in roles.items():
        primary = selected["primary"]["configuration"]
        for role in ("baseline", "simple_anchor", "standard_dynamic"):
            records.append({"scope": scope, "comparison": "primary_minus_" + role,
                            "primary": primary, "reference": selected[role]["configuration"]})
    return records


def compare(days: pd.DataFrame, roles: dict[str, Any], *, n_bootstrap: int = 5000,
            seed: int = 17023) -> pd.DataFrame:
    rows = []
    metrics = [m for m in ("crps", *INTERVAL, "point_level_mae", "point_shape_mae") if m in days]
    for record in planned_comparisons(roles):
        paired = V7.paired_month_bootstrap(days, record["primary"], record["reference"], metrics,
                                            n_bootstrap=n_bootstrap, seed=seed)
        rows.append(paired.assign(scope=record["scope"], comparison=record["comparison"],
                    primary_configuration=json.dumps(record["primary"], sort_keys=True),
                    reference_configuration=json.dumps(record["reference"], sort_keys=True)))
    return pd.concat(rows, ignore_index=True)


def old_v8_ratio(v9_root: Path, target_day: pd.DataFrame, roles: dict[str, Any], v8_root: Path,
                 *, n_bootstrap: int = 5000, seed: int = 17023) -> tuple[pd.DataFrame, dict[str, str]]:
    v8_root = Path(v8_root).resolve()
    v8 = module("v9_verified_v8_ratio", ROOT / "scripts/run_fuel_attribution_v8.py")
    v8.verify_phase(v8_root, "legal")
    v8_freeze = RUNNER.read(v8_root / "prefit_freeze.json")
    v9_freeze = RUNNER.read(v9_root / "prefit_freeze.json")
    for region in RUNNER.V7.TARGETS:
        name = region + "_hourly.csv"
        prior = [digest for path, digest in v8_freeze["inputs"].items() if Path(path).name == name]
        current = [digest for path, digest in v9_freeze["inputs"].items() if Path(path).name == name]
        if prior != current or len(prior) != 1:
            raise ValueError("V8 fixed ratio target outcome bytes differ")
    choices = RUNNER.read(v8_root / "source_choices.json")
    old_config = choices["selections"]["joint"]["primary"]["configuration"]
    old = subset(pd.read_csv(v8_root / "legal_scores/day_metrics.csv"), old_config)
    results = []
    for scope, selected in roles.items():
        new = subset(target_day, selected["primary"]["configuration"])
        left = new.assign(round=9, name="v9_" + scope, mode="v9", factor=1.)
        right = old.assign(round=8, name="v8_fixed_ratio", mode="v8", factor=1.)
        paired = V7.paired_month_bootstrap(pd.concat([left, right], ignore_index=True),
                   {"round": 9, "name": "v9_" + scope, "factor": 1.},
                   {"round": 8, "name": "v8_fixed_ratio", "factor": 1.},
                   ["crps", *INTERVAL], n_bootstrap=n_bootstrap, seed=seed)
        results.append(paired.assign(scope=scope, comparison="primary_minus_old_v8_fixed_ratio",
                       primary_configuration=json.dumps(selected["primary"]["configuration"], sort_keys=True),
                       reference_configuration=json.dumps(old_config, sort_keys=True)))
    return pd.concat(results, ignore_index=True), {
        str(v8_root / name): RUNNER.sha(v8_root / name) for name in
        ("prefit_freeze.json", "source_choices.json", "legal_completed.json",
         "legal_prediction_seal.json", "legal_scores/score_seal.json")}


def holdout_trigger(roles: dict[str, Any], source_macro: pd.DataFrame, target_macro: pd.DataFrame,
                    pairs: pd.DataFrame, *, old_available: bool) -> dict[str, Any]:
    def row(table, role):
        found = table[(table.scope.eq("joint")) & (table.role.eq(role)) & table.semester.eq("H1")]
        if len(found) != 1:
            raise ValueError("Joint H1 role missing")
        return found.iloc[0]
    source = row(source_macro, "primary")
    primary = row(target_macro, "primary")
    checks = {"source_coverage_90_to_95pct": bool(roles["joint"]["primary"]["status"] == "passed"
              and .90 <= source.empirical_coverage90 <= .95 and source.worst_coverage >= .85),
              "target_coverage_at_least_90pct": bool(primary.empirical_coverage90 >= .90),
              "target_worst_region_delay_coverage_at_least_85pct": bool(primary.worst_coverage >= .85)}
    references = ["simple_anchor", "standard_dynamic", "old_v8_fixed_ratio"]
    for reference in references:
        name = "primary_minus_" + reference
        for metric in ("crps", "empirical_interval_score90", "empirical_width90"):
            found = pairs[(pairs.scope.eq("joint")) & pairs.comparison.eq(name) &
                          pairs.stratum.eq("overall") & pairs.metric.eq(metric)]
            if len(found) != 1:
                if reference == "old_v8_fixed_ratio" and not old_available:
                    checks[f"{reference}_available"] = False
                    continue
                raise ValueError("Missing fixed paired comparison: " + name + "/" + metric)
            item = found.iloc[0]
            if metric == "crps":
                checks[f"crps_5pct_below_{reference}"] = bool(item.primary_mean <= .95 * item.reference_mean)
                checks[f"six_month_crps_ci_upper_below_zero_vs_{reference}"] = bool(item.ci95_high < 0)
            elif metric == "empirical_interval_score90":
                checks[f"interval_score_better_than_{reference}"] = bool(item.difference_primary_minus_reference < 0)
            else:
                checks[f"width_no_more_than_{reference}"] = bool(item.difference_primary_minus_reference <= 0)
    return {"checks": checks, "new_holdout_triggered": bool(old_available and all(checks.values())),
            "scope": "Development gate on previously viewed target regions; no unseen holdout evaluated"}


def analyze(root: Path, output: Path, *, through: str = "target", v8_run: Path | None = None,
            n_bootstrap: int = 5000, seed: int = 17023) -> dict[str, Any]:
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new V9 analysis directory")
    if through not in ("source", "target"):
        raise ValueError("Unknown analysis stage")
    frozen = RUNNER.verify(root)
    source = verified_scores(root, "source")
    choice = RUNNER.read(root / "source_choices.json")
    RUNNER.validate_choices(root, choice)
    roles = choice["selections"]
    if set(roles) != {"1", "2", "joint"} or any(set(selected) != set(ROLES) for selected in roles.values()):
        raise ValueError("Incomplete source-fixed V9 roles")
    stages = {"source": source}
    if through == "target":
        stages["target"] = verified_scores(root, "target")
    output.mkdir(parents=True)
    macros = {}
    coarse_support = {}
    for stage, scores in stages.items():
        macro, regional, by_region = fixed_tables(scores, roles)
        macros[stage] = macro
        macro.to_csv(output / f"{stage}_fixed_roles_macro.csv", index=False)
        regional.to_csv(output / f"{stage}_fixed_roles_region_delay.csv", index=False)
        by_region.to_csv(output / f"{stage}_fixed_roles_region.csv", index=False)
        by_region[by_region.region.isin(FOCUS)].to_csv(output / f"{stage}_focus_regions.csv", index=False)
        scores["macro"].to_csv(output / f"{stage}_all_candidates_descriptive_macro.csv", index=False)
        pd.read_csv(root / f"{stage}_coarse_scores/macro_metrics.csv").to_csv(
            output / f"{stage}_coarse_forecast_macro.csv", index=False)
        monthly_coarse = pd.read_csv(root / f"{stage}_coarse_scores/monthly_metrics.csv")
        monthly_coarse.to_csv(output / f"{stage}_coarse_forecast_region_delay_month.csv", index=False)
        coarse_metrics = ["mix_mae", "mix_crps", "mix_energy_score32", "ef_point_mae", "ef_crps"]
        by_delay = monthly_coarse.groupby(["region", "delay", "method", "semester"], sort=True)[
            coarse_metrics].mean().reset_index()
        by_delay.groupby(["region", "method", "semester"], sort=True)[
            coarse_metrics].mean().reset_index().to_csv(
                output / f"{stage}_coarse_forecast_region.csv", index=False)
        support = pd.read_csv(root / f"{stage}_coarse_scores/month_support.csv")
        support.to_csv(output / f"{stage}_coarse_month_support.csv", index=False)
        unique = support.drop_duplicates(["region", "month", "semester"])
        if unique.duplicated(["region", "month"]).any():
            raise ValueError("Coarse support has duplicate region-months")
        coarse_support[stage] = {
            "observed_region_months": int(unique.label_available.sum()),
            "missing_region_months": int((~unique.label_available).sum()),
            "observed_months_by_region_semester": {
                str(region) + "/" + semester: int(group.label_available.sum())
                for (region, semester), group in unique.groupby(["region", "semester"], sort=True)}}
    historical = {}
    if through == "target":
        target = stages["target"]
        paired = compare(target["day"], roles, n_bootstrap=n_bootstrap, seed=seed)
        if v8_run is not None:
            old, historical = old_v8_ratio(root, target["day"], roles, v8_run,
                                           n_bootstrap=n_bootstrap, seed=seed)
            paired = pd.concat([paired, old], ignore_index=True)
        paired.to_csv(output / "target_fixed_paired_month_bootstrap.csv", index=False)
        RUNNER.write(output / "holdout_trigger.json", holdout_trigger(
            roles, macros["source"], macros["target"], paired, old_available=v8_run is not None))
    (output / "READ_ME.md").write_text(
        "# V9 lawful coarse dynamics development analysis\n\n"
        "2025H1 LODO selects configurations and factors; 2025H2 is stress evidence only. "
        "The 2026H1 regions have been viewed in earlier rounds and are a development retest. "
        "All target roles and baselines retain their source choices. Coarse forecasts are "
        "sealed before the actual monthly composition is used for scoring. "
        "Target paired ranges use synchronous six-calendar-month resampling with equal "
        "region/delay weights. They describe these seen regions, not unseen geographic generalization.\n")
    manifest = [root / "prefit_freeze.json", root / "source_choices.json", RUNNER_PATH,
                Path(__file__).resolve()]
    for stage in stages:
        manifest.extend((root / f"{stage}_completed.json", root / f"{stage}_prediction_seal.json",
                         root / f"{stage}_scores/score_seal.json",
                         root / f"{stage}_coarse_scores/score_seal.json"))
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "run": str(root),
                "through": through, "selections": roles, "target_selection_performed": False,
                "target_training_performed": False,
                "bootstrap": {"n_bootstrap": n_bootstrap, "seed": seed, "calendar_month_blocks": 6},
                "coarse_label_support": coarse_support,
                "scope": frozen["target_scope"], "frozen_input_hashes": frozen["inputs"],
                "frozen_code_hashes": frozen["code"], "upstream_dependency_hashes": frozen["dependencies"],
                "historical_input_hashes": historical,
                "input_hashes": {str(path): RUNNER.sha(path) for path in manifest},
                "output_hashes": {path.name: RUNNER.sha(path) for path in sorted(output.iterdir()) if path.is_file()}}
    RUNNER.write(output / "analysis_metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--through", choices=("source", "target"), default="target")
    parser.add_argument("--v8-run", type=Path, default=ROOT / "results/fuel-attribution-v8")
    parser.add_argument("--skip-v8-background", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    analyze(args.run, args.output, through=args.through,
            v8_run=None if args.skip_v8_background or args.through == "source" else args.v8_run,
            n_bootstrap=args.bootstrap)


if __name__ == "__main__":
    main()
