"""Source-fixed V8 attribution analysis on previously viewed 2026H1 regions.

The illegal oracle is a diagnostic paired to its legal counterpart.  It never
selects a factor, a method, or a deployable forecast.  Month-bootstrap ranges
describe time variation in the six seen regions only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


V7 = _module("v8_uses_v7_analysis", ROOT / "scripts/analyze_fuel_transport_v7.py")
RUNNER_PATH = ROOT / "scripts/run_fuel_attribution_v8.py"
V8 = None  # Loaded after the runner has been installed; permits unit tests of pure helpers.
INTERVAL = V7.INTERVAL
METRICS = V7.METRICS
IDENTITY = ["round", "name", "mode", "factor"]
FOCUS = ("PACE", "AZPS", "PSCO", "LGEE")
SCORE_FILES = ("day_metrics.csv", "region_metrics.csv", "macro_metrics.csv", "day_counts.csv", "score_seal.json")


def runner():
    global V8
    if V8 is None:
        V8 = _module("v8_attribution_runner_for_analysis", RUNNER_PATH)
    return V8


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def config_of(record):
    return record["configuration"]


def fixed_roles(choices):
    roles = choices["selections"]
    if set(roles) != {"1", "2", "joint"}:
        raise ValueError("Source choices must contain both rounds and joint roles")
    required = {"primary", "baseline", "exact_additive", "exact_direct"}
    for scope, records in roles.items():
        if not required.issubset(records):
            raise ValueError(f"Missing source-fixed attribution role in {scope}")
        primary = config_of(records["primary"])
        if scope != "joint" and int(primary["round"]) != int(scope):
            raise ValueError("Round primary identity differs")
        if "recompose_joint" not in primary["name"]:
            raise ValueError("Primary must be the legal ratio mapping")
        for role, mode in (("exact_additive", "additive_anchor"), ("exact_direct", "direct_analog")):
            expected = {**primary, "name": primary["name"].replace("recompose_joint", mode)}
            if config_of(records[role]) != expected:
                raise ValueError(f"{role} must match the primary factor and base identity")
        if "additive_anchor" not in config_of(records["baseline"])["name"]:
            raise ValueError("Baseline must use additive mapping")
        if scope == "joint":
            for role in ("primary", "baseline"):
                if config_of(records[role]) not in [config_of(roles[str(r)][role]) for r in (1, 2)]:
                    raise ValueError("Joint role must retain a source-selected round role")
    return roles


def subset(frame, config):
    return V7.subset(frame, config)


def fixed_tables(scores, roles):
    """Equal region/delay macro summaries; preserve failed source statuses."""
    regional, macro, by_region = [], [], []
    metrics = [metric for metric in METRICS if metric in scores["day"].columns]
    for scope, records in roles.items():
        for role, record in records.items():
            config = config_of(record)
            cells = subset(scores["region"], config).assign(scope=scope, role=role, source_status=record["status"])
            regional.append(cells)
            for semester, half in cells.groupby("semester", sort=True):
                context = {"scope": scope, "role": role, "source_status": record["status"], **config,
                           "mode": half["mode"].iloc[0], "semester": semester}
                macro.append({**context, **half[metrics].mean().to_dict(),
                              "worst_coverage": float(half.empirical_coverage90.min()),
                              "n_region_delay_groups": len(half), "n_scored_days": int(half.days.sum())})
                for region, group in half.groupby("region", sort=True):
                    by_region.append({**context, "region": region, **group[metrics].mean().to_dict(),
                                      "worst_coverage": float(group.empirical_coverage90.min()),
                                      "n_delay_groups": len(group), "n_scored_days": int(group.days.sum())})
    return pd.DataFrame(macro), pd.concat(regional, ignore_index=True), pd.DataFrame(by_region)


def planned_legal(roles):
    """Predefined comparisons; each exact control inherits primary's factor."""
    records = []
    for scope, selected in roles.items():
        primary = config_of(selected["primary"])
        for role in ("baseline", "exact_additive", "exact_direct"):
            records.append({"scope": scope, "comparison": "primary_minus_" + role,
                            "primary": primary, "reference": config_of(selected[role]),
                            "metrics": ["crps", *INTERVAL, "raw_level_mae", "raw_shape_mae", "point_level_mae", "point_shape_mae"]})
    return records


def planned_oracle(roles):
    """Oracle pairs are fixed by the legal identity and never enter selection."""
    records = []
    for scope, selected in roles.items():
        for role in ("primary", "exact_additive"):
            legal = config_of(selected[role])
            mode = "recompose_joint" if role == "primary" else "additive_anchor"
            oracle = {**legal, "name": legal["name"].replace(mode, "oracle_" + mode)}
            records.append({"scope": scope, "comparison": "legal_" + role + "_minus_oracle",
                            "primary": legal, "reference": oracle,
                            "metrics": ["crps", *INTERVAL, "raw_level_mae", "raw_shape_mae", "point_level_mae", "point_shape_mae"]})
    return records


def compare_days(days, comparisons, *, n_bootstrap=5000, seed=17023):
    rows = []
    for record in comparisons:
        metrics = [metric for metric in record["metrics"] if metric in days]
        paired = V7.paired_month_bootstrap(days, record["primary"], record["reference"], metrics,
                                           n_bootstrap=n_bootstrap, seed=seed)
        rows.append(paired.assign(scope=record["scope"], comparison=record["comparison"],
            primary_configuration=json.dumps(record["primary"], sort_keys=True),
            reference_configuration=json.dumps(record["reference"], sort_keys=True)))
    return pd.concat(rows, ignore_index=True)


def oracle_legal_days(legal, oracle, roles):
    """Make pairable days while retaining the phase name and exact configuration."""
    paired = []
    for record in planned_oracle(roles):
        lawful, hindsight = subset(legal, record["primary"]), subset(oracle, record["reference"])
        keys = ["region", "delay", "origin"]
        if lawful.duplicated(keys).any() or hindsight.duplicated(keys).any():
            raise ValueError("Duplicate oracle/legal scoring day")
        merged = lawful[keys].merge(hindsight[keys], on=keys, how="outer", indicator=True, validate="one_to_one")
        if not merged._merge.eq("both").all():
            raise ValueError("Oracle and legal mapping have different scored days")
        paired.extend((lawful, hindsight))
    combined = pd.concat(paired, ignore_index=True)
    identity = ["region", "delay", "round", "name", "factor", "origin"]
    if combined.groupby(identity, dropna=False).nunique(dropna=False).gt(1).any().any():
        raise ValueError("Repeated joint/round oracle scores differ")
    return combined.drop_duplicates(identity).reset_index(drop=True)


def holdout_trigger(roles, source_macro, legal_macro, legal_pairs, v7_pairs=None):
    """Prespecified development continuation gate, never a holdout conclusion."""
    def one(table, role):
        rows = table[(table.scope.eq("joint")) & (table.role.eq(role)) & (table.semester.eq("H1"))]
        if len(rows) != 1:
            raise ValueError("Missing joint H1 fixed role")
        return rows.iloc[0]

    source = one(source_macro, "primary")
    primary, additive = one(legal_macro, "primary"), one(legal_macro, "baseline")
    p = legal_pairs[(legal_pairs.scope.eq("joint")) &
                    (legal_pairs.comparison.eq("primary_minus_baseline")) &
                    (legal_pairs.stratum.eq("overall")) & (legal_pairs.metric.eq("crps"))]
    if len(p) != 1:
        raise ValueError("Missing legal primary/additive six-month comparison")
    checks = {
        "source_primary_gate_passed": bool(roles["joint"]["primary"]["status"] == "passed" and
            .9 <= source.empirical_coverage90 <= .95 and source.worst_coverage >= .85),
        "target_macro_coverage_at_least_90pct": bool(primary.empirical_coverage90 >= .9),
        "target_worst_cell_coverage_at_least_85pct": bool(primary.worst_coverage >= .85),
        "crps_5pct_below_source_additive": bool(primary.crps <= .95 * additive.crps),
        "interval_score_below_source_additive": bool(primary.empirical_interval_score90 < additive.empirical_interval_score90),
        "width_no_more_than_source_additive": bool(primary.empirical_width90 <= additive.empirical_width90),
        "six_month_crps_upper_below_zero_vs_additive": bool(p.iloc[0].ci95_high < 0),
    }
    if v7_pairs is not None:
        def old(metric):
            rows = v7_pairs[(v7_pairs.scope.eq("joint")) &
                            (v7_pairs.comparison.eq("v8_primary_minus_v7_fresh_anchor")) &
                            (v7_pairs.stratum.eq("overall")) & (v7_pairs.metric.eq(metric))]
            if len(rows) != 1:
                raise ValueError("Missing V7 fresh-anchor same-day comparison")
            return rows.iloc[0]
        crps, interval, width = (old(metric) for metric in
            ("crps", "empirical_interval_score90", "empirical_width90"))
        checks.update({
            "crps_5pct_below_v7_fresh_anchor": bool(crps.primary_mean <= .95 * crps.reference_mean),
            "interval_score_below_v7_fresh_anchor": bool(interval.difference_primary_minus_reference < 0),
            "width_no_more_than_v7_fresh_anchor": bool(width.difference_primary_minus_reference <= 0),
            "six_month_crps_upper_below_zero_vs_v7_fresh_anchor": bool(crps.ci95_high < 0),
        })
    return {"checks": checks, "new_holdout_triggered": bool(v7_pairs is not None and all(checks.values())),
            "simplification_signal_additive_crps_lower": bool(additive.crps < primary.crps),
            "scope": "Development continuation gate only; no new holdout evaluated"}


def _v7_background(v8_root, legal_days, roles, output, v7_run, *, n_bootstrap, seed):
    """Historical same-day controls; V7 training budget is identical by reuse."""
    old = V7.verify_run(Path(v7_run).resolve())
    v8_freeze = read_json(v8_root / "prefit_freeze.json")
    old_freeze = old["frozen"]
    current_inputs = v8_freeze.get("inputs", v8_freeze["dependencies"])
    for region in V7.RUNNER.TARGETS:
        name = region + "_hourly.csv"
        new_hashes = [digest for path, digest in current_inputs.items() if Path(path).name == name]
        old_hashes = [digest for path, digest in old_freeze["inputs"].items() if Path(path).name == name]
        if len(new_hashes) != 1 or new_hashes != old_hashes:
            raise ValueError("V8 and V7 target hourly input bytes differ")
    choice = old["roles"]
    selected = choice["joint"]
    references = [("fresh_anchor", config_of(selected["simple_anchor"])),
                  ("extratrees_factor_1p5", {"round": 2, "name": "extratrees", "factor": 1.5}),
                  ("cqr", config_of(selected["cqr"]))]
    rows = []
    for scope, records in roles.items():
        legal = subset(legal_days, config_of(records["primary"])).assign(round=8, name=f"v8_{scope}_primary", mode="v8_primary", factor=1.)
        for label, old_config in references:
            reference = subset(old["target"]["day"], old_config).assign(round=7, name="v7_" + label, mode="v7_" + label, factor=1.)
            combined = pd.concat([legal, reference], ignore_index=True)
            metrics = INTERVAL if label == "cqr" else ["crps", *INTERVAL, "raw_level_mae", "raw_shape_mae"]
            paired = V7.paired_month_bootstrap(combined, {"round": 8, "name": f"v8_{scope}_primary", "factor": 1.},
                {"round": 7, "name": "v7_" + label, "factor": 1.}, metrics, n_bootstrap=n_bootstrap, seed=seed)
            rows.append(paired.assign(scope=scope, comparison="v8_primary_minus_v7_" + label,
                interpretation="Same seen days; V7 fits reused; CQR supplies interval metrics only"))
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(output / "v7_same_day_background.csv", index=False)
    return {str(Path(v7_run).resolve() / name): V7.sha(Path(v7_run).resolve() / name)
            for name in ("prefit_freeze.json", "source_choices.json", "completed.json", "target_scores/score_seal.json")}


def _verified_stage(root, stage):
    runner().verify_phase(root, stage)
    folder = root / (stage + "_scores")
    seal = read_json(folder / "score_seal.json")
    V7.check_hashes(folder, seal, set(SCORE_FILES) - {"score_seal.json"})
    day = pd.read_csv(folder / "day_metrics.csv")
    region = pd.read_csv(folder / "region_metrics.csv")
    macro = pd.read_csv(folder / "macro_metrics.csv")
    if day.empty or day.duplicated(["region", "delay", "round", "name", "factor", "origin"]).any():
        raise ValueError("Empty or duplicate daily scoring grid")
    metrics = [metric for metric in METRICS if metric in day]
    if not set(["crps", *INTERVAL]).issubset(metrics) or not np.isfinite(day[metrics]).all().all():
        raise ValueError("Missing or nonfinite V8 scores")
    keys = ["region", "delay", *IDENTITY, "semester"]
    grouped = day.groupby(keys, sort=True)
    expected = grouped[metrics].mean().reset_index()
    expected["days"] = grouped.size().to_numpy()
    V7.assert_equal_table(region, expected, keys, [*metrics, "days"])
    V7.assert_equal_table(macro, region.groupby([*IDENTITY, "semester"])[metrics].mean().reset_index(),
                          [*IDENTITY, "semester"], metrics)
    return {"day": day, "region": region, "macro": macro, "metrics": metrics}


def analyze(root, output, *, through="oracle", v7_run=None, n_bootstrap=5000, seed=17023):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new V8 analysis output directory")
    if through not in ("source", "legal", "oracle"):
        raise ValueError("Unknown completed stage")
    freeze = runner().verify(root)
    source = _verified_stage(root, "source")
    choice = read_json(root / "source_choices.json")
    runner().validate_choices(root, choice)
    roles = fixed_roles(choice)
    stages = {"source": source}
    if through in ("legal", "oracle"):
        stages["legal"] = _verified_stage(root, "legal")
    if through == "oracle":
        stages["oracle"] = _verified_stage(root, "oracle")
    output.mkdir(parents=True)
    for stage, scores in stages.items():
        if stage == "oracle":
            scores["macro"].to_csv(output / "oracle_all_diagnostic_macro.csv", index=False)
            scores["region"].to_csv(output / "oracle_all_diagnostic_region_delay.csv", index=False)
            continue
        macro, regional, by_region = fixed_tables(scores, roles)
        macro.to_csv(output / f"{stage}_fixed_roles_macro.csv", index=False)
        regional.to_csv(output / f"{stage}_fixed_roles_region_delay.csv", index=False)
        by_region.to_csv(output / f"{stage}_fixed_roles_region.csv", index=False)
        if stage == "legal":
            by_region[by_region.region.isin(FOCUS)].to_csv(output / "focus_regions_coverage_efficiency.csv", index=False)
            legal_pairs = compare_days(scores["day"], planned_legal(roles), n_bootstrap=n_bootstrap, seed=seed)
            legal_pairs.to_csv(output / "legal_paired_month_bootstrap.csv", index=False)
            legal_macro = macro
        elif stage == "source":
            source_macro = macro
    if through == "oracle":
        pairs = oracle_legal_days(stages["legal"]["day"], stages["oracle"]["day"], roles)
        compare_days(pairs, planned_oracle(roles), n_bootstrap=n_bootstrap, seed=seed).to_csv(
            output / "oracle_legal_paired_month_bootstrap.csv", index=False)
    historical = {}
    v7_pairs = None
    if "legal" in stages and v7_run is not None:
        historical = _v7_background(root, stages["legal"]["day"], roles, output, v7_run,
                                    n_bootstrap=n_bootstrap, seed=seed)
        v7_pairs = pd.read_csv(output / "v7_same_day_background.csv")
    if "legal" in stages:
        write_json(output / "holdout_trigger.json", holdout_trigger(roles, source_macro, legal_macro,
                                                            legal_pairs, v7_pairs))
    (output / "READ_ME.md").write_text(
        "# V8 source-fixed attribution analysis\n\n"
        "Source roles and spread factors come only from 2025H1 LODO. 2025H2 is stress evidence. "
        "The six 2026H1 regions were previously viewed and remain development retests. "
        "Oracle outcomes use future actual monthly mix and are illegal for selection or deployment. "
        "All oracle differences pair the same days and fixed legal configuration. "
        "Differences are first method minus reference. "
        f"The {n_bootstrap:,} synchronous six-calendar-month bootstrap draws (seed {seed}) "
        "describe time variation in these six seen regions only; "
        "they are not geographic generalization intervals. Coverage/width is reported together "
        "with CRPS, interval score, and level/shape errors.\n")
    manifest = [root / "prefit_freeze.json", root / "source_choices.json", RUNNER_PATH,
                Path(__file__).resolve(), ROOT / "scripts/analyze_fuel_transport_v7.py"]
    for stage in stages:
        manifest.extend((root / f"{stage}_completed.json", root / f"{stage}_prediction_seal.json",
                         root / f"{stage}_scores/score_seal.json"))
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "run": str(root), "through": through,
        "selections": roles, "target_selection_performed": False, "oracle_selection_performed": False,
        "target_training_performed": False, "bootstrap": {"n_bootstrap": n_bootstrap, "seed": seed,
            "calendar_month_blocks": 6, "scope": "Conditional descriptive intervals for six seen regions"},
        "frozen_input_hashes": freeze.get("inputs", {}), "frozen_code_hashes": freeze.get("code", {}),
        "upstream_dependency_hashes": freeze.get("dependencies", {}),
        "input_hashes": {str(path): V7.sha(path) for path in manifest}, "historical_input_hashes": historical,
        "output_hashes": {path.name: V7.sha(path) for path in sorted(output.iterdir()) if path.is_file()}}
    write_json(output / "analysis_metadata.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--through", choices=("source", "legal", "oracle"), default="oracle")
    parser.add_argument("--v7-run", type=Path, default=ROOT / "results/fuel-transport-v7")
    args = parser.parse_args()
    analyze(args.run, args.output, through=args.through, v7_run=args.v7_run)


if __name__ == "__main__":
    main()
