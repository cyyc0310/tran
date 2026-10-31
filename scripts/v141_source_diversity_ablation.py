"""V141: add donor regimes while preserving V140 target/calibration roles."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import v139_physical_reserve_calibration as v139
import v140_monthly_conditional_risk as v140

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V141_SOURCE_DIVERSITY_ABLATION_PROTOCOL.md"
PREPARED = ROOT / "results/v141-source-expansion-input-r1"
V140 = ROOT / "results/v140-monthly-conditional-risk-r1"
OUT = ROOT / "results/v141-source-diversity-ablation-r1"
SCENARIOS = ("plus_all", "plus_high_coal", "plus_matched_low_coal")
MODELS = ("direct_cqr_all", "direct_cqr_active", "occupancy_cqr")


def original_budget_frame(source: pd.DataFrame) -> pd.DataFrame:
    source = source.copy()
    source["physical_lower"] = [getattr(r, f"monthly_lower_{r.tau}") for r in source.itertuples(index=False)]
    source["physical_upper"] = [getattr(r, f"monthly_upper_{r.tau}") for r in source.itertuples(index=False)]
    source["physical_lower_fraction"] = source.physical_lower / source.hours
    source["physical_upper_fraction"] = source.physical_upper / source.hours
    source["physical_width_fraction"] = (source.physical_upper - source.physical_lower) / source.hours
    return source


def selected_donors(inventory: pd.DataFrame, held_out: str, seed: int) -> dict[str, list[str]]:
    eligible = inventory.loc[inventory.eligible_donor]
    order = lambda names: sorted(names, key=lambda region: hashlib.sha256(f"{seed}/{held_out}/{region}".encode()).hexdigest())
    high = order(eligible.loc[eligible.coal_heavy, "region"])
    low = order(eligible.loc[~eligible.coal_heavy, "region"])
    matched_count = min(len(high), len(low))
    if matched_count == 0:
        raise ValueError("no matched source composition groups")
    return {"plus_all": sorted(eligible.region), "plus_high_coal": high[:matched_count],
            "plus_matched_low_coal": low[:matched_count]}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    protocol_hash = v139.sha(PROTOCOL)
    prep_summary = json.loads((PREPARED / "summary.json").read_text())
    if prep_summary["protocol_sha256"] != protocol_hash or prep_summary["exact_count_failures"]:
        raise ValueError("input preparation not valid for the frozen protocol")
    if v139.sha(PREPARED / "additional_monthly_panel.csv") != prep_summary["panel_sha256"]:
        raise ValueError("prepared panel hash changed")
    source = original_budget_frame(pd.read_csv(V140 / "monthly_feature_panel.csv"))
    additions = pd.read_csv(PREPARED / "additional_monthly_panel.csv")
    inventory = pd.read_csv(PREPARED / "donor_inventory.csv")
    role_table = pd.read_csv(V140 / "region_roles.csv")
    archived = pd.read_csv(V140 / "outer_fold_rows.csv")
    archived = archived.loc[archived.budget.eq("monthly")].copy()
    original_rows = archived.copy()
    original_rows["source_scenario"] = "original"
    config = json.loads((V140 / "summary.json").read_text())
    v140.FEATURES = list(config["common_features"])
    if config["model_params"] != v140.PARAMS:
        raise ValueError("model hyperparameters changed since V140")
    if not np.isfinite(additions[v140.FEATURES].to_numpy(float)).all():
        raise ValueError("invalid additional monthly features")
    if set(source.region) & set(additions.region):
        raise ValueError("additional source overlaps original identities")
    outputs, memberships, radii, failures = [], [], [], []
    for held_out in sorted(source.region.unique()):
        for seed in v140.SEEDS:
            roles = role_table.loc[role_table.target_region.eq(held_out) & role_table.seed.eq(seed)]
            base_fit_regions = roles.loc[roles.role.eq("fit"), "region"].tolist()
            cal_regions = roles.loc[roles.role.eq("calibration"), "region"].tolist()
            rosters = selected_donors(inventory, held_out, seed)
            for scenario, donor_regions in rosters.items():
                fit_roster = base_fit_regions + donor_regions
                if (held_out in fit_roster or held_out in cal_regions or set(fit_roster) & set(cal_regions)
                        or len(base_fit_regions) != 12 or len(cal_regions) != 12):
                    raise ValueError("target/fit/calibration role leakage")
                for region in fit_roster:
                    memberships.append({"target_region": held_out, "seed": seed, "source_scenario": scenario,
                                        "source_region": region,
                                        "source_role": "original_fit" if region in base_fit_regions else "additional_fit"})
                for tau in (600, 900):
                    frame = source.loc[source.tau.eq(tau)]
                    fit = pd.concat([frame.loc[frame.region.isin(base_fit_regions)],
                                     additions.loc[additions.region.isin(donor_regions) & additions.tau.eq(tau)]],
                                    ignore_index=True, sort=False)
                    calibration = frame.loc[frame.region.isin(cal_regions)]
                    target = frame.loc[frame.region.eq(held_out)]
                    if (not np.isfinite(fit[v140.FEATURES].to_numpy(float)).all()
                            or fit.region.nunique() != len(fit_roster)
                            or not fit.physical_lower.le(fit.factor_box_K_lower).all()
                            or not fit.physical_upper.ge(fit.factor_box_K_upper).all()):
                        raise ValueError("invalid fit panel or donor role count")
                    for method in MODELS:
                        normalized = method == "occupancy_cqr"
                        models = v140.fitted_models(fit, method)
                        ca, cb = v140.projected_predictions(calibration, method, models)
                        scores = pd.DataFrame({"region": calibration.region.to_numpy(),
                                               "score": v140.score(calibration, ca, cb, normalized)})
                        region_scores = scores.groupby("region").score.max().sort_values()
                        a, b = v140.projected_predictions(target, method, models)
                        if len(region_scores) != 12:
                            raise ValueError("calibration region count changed")
                        for alpha in v140.ALPHAS:
                            rank = math.ceil((len(region_scores) + 1) * (1 - alpha))
                            radius = float(region_scores.iloc[rank - 1]) if rank <= len(region_scores) else 1.0
                            radii.append({"target_region": held_out, "seed": seed, "tau": tau, "method": method,
                                          "source_scenario": scenario, "alpha": alpha, "rank": rank,
                                          "calibration_regions": len(region_scores), "radius": radius,
                                          "fit_regions": len(fit_roster), "fit_region_months": len(fit)})
                            scale = ((target.physical_upper - target.physical_lower).to_numpy(float)
                                     if normalized else target.hours.to_numpy(float))
                            lower = np.maximum(target.physical_lower.to_numpy(int), np.floor(a - radius * scale + 1e-9).astype(int))
                            upper = np.minimum(target.physical_upper.to_numpy(int), np.ceil(b + radius * scale - 1e-9).astype(int))
                            for j, row in enumerate(target.itertuples(index=False)):
                                lo, hi = int(lower[j]), int(upper[j])
                                if not row.physical_lower <= lo <= hi <= row.physical_upper:
                                    failures.append({"region": held_out, "month": row.month, "tau": tau,
                                                     "seed": seed, "source_scenario": scenario, "method": method,
                                                     "alpha": alpha, "check": "physical_subset"})
                                record = {"region": held_out, "month": row.month, "tau": tau, "seed": seed,
                                          "budget": "monthly", "source_scenario": scenario, "method": method,
                                          "alpha": alpha, "hours": row.hours, "lower": lo, "upper": hi,
                                          "width": hi - lo, "covered": lo <= row.factor_box_K_lower and hi >= row.factor_box_K_upper,
                                          "empty_fallback": False, "capacity_unavailable": row.capacity_unavailable,
                                          "factor_box_K_lower": row.factor_box_K_lower,
                                          "factor_box_K_upper": row.factor_box_K_upper,
                                          "robust_interval_score90": (hi - lo + 20 * max(lo - row.factor_box_K_lower, 0)
                                                                       + 20 * max(row.factor_box_K_upper - hi, 0))}
                                for rate in v139.RATES:
                                    required = math.ceil(rate * row.hours - 1e-10)
                                    pos, neg = lo >= required, hi < required
                                    record[f"positive_{rate}"] = pos
                                    record[f"negative_{rate}"] = neg
                                    record[f"abstain_{rate}"] = not (pos or neg)
                                    record[f"error_{rate}"] = ((pos and row.factor_box_K_lower < required)
                                                               or (neg and row.factor_box_K_upper >= required))
                                outputs.append(record)
        print(f"source expansion target {held_out}", flush=True)
    rows = pd.concat([original_rows, pd.DataFrame(outputs)], ignore_index=True)
    expected_new = len(source) * len(v140.SEEDS) * len(v140.ALPHAS) * len(SCENARIOS) * len(MODELS)
    keys = ["region", "month", "tau", "seed", "source_scenario", "method", "alpha"]
    if len(outputs) != expected_new or rows.duplicated(keys).any():
        raise ValueError("target denominator/duplicate failure")
    reproduced = rows.loc[rows.source_scenario.eq("original"), archived.columns].reset_index(drop=True)
    if not reproduced.equals(archived.reset_index(drop=True)):
        raise ValueError("archived original predictions changed")
    for key, group in rows.groupby(keys[:-1]):
        ordered = group.sort_values("alpha", ascending=False)
        narrow, wide = ordered.iloc[0], ordered.iloc[-1]
        if wide.lower > narrow.lower or wide.upper < narrow.upper:
            failures.append({"key": str(key), "check": "alpha_nesting"})
    summaries = []
    for (scenario, seed), group in rows.groupby(["source_scenario", "seed"]):
        summary = v139.summarize(group)
        stress = group.groupby(["method", "alpha", "tau", "region"]).robust_interval_score90.mean()
        stress = stress.groupby(["method", "alpha", "tau"]).mean().reset_index()
        summary = summary.merge(stress, on=["method", "alpha", "tau"], validate="one_to_one")
        summary["source_scenario"], summary["seed"] = scenario, seed
        summaries.append(summary)
    frontier = pd.concat(summaries, ignore_index=True)
    sc = rows.loc[rows.region.eq("SC")].groupby(["source_scenario", "seed", "tau", "method", "alpha"]).agg(
        coverage=("covered", "mean"), width=("width", "mean"), interval_score90=("robust_interval_score90", "mean")).reset_index()
    if protocol_hash != v139.sha(PROTOCOL):
        raise RuntimeError("protocol changed during scoring")
    OUT.mkdir(parents=True)
    rows.to_csv(OUT / "outer_fold_rows.csv", index=False)
    frontier.to_csv(OUT / "region_equal_frontier.csv", index=False)
    sc.to_csv(OUT / "SC_frontier.csv", index=False)
    pd.DataFrame(memberships).to_csv(OUT / "fit_membership.csv", index=False)
    pd.DataFrame(radii).to_csv(OUT / "calibration_radii.csv", index=False)
    pd.DataFrame(failures).to_csv(OUT / "assertion_failures.csv", index=False)
    paths = [PROTOCOL, V140 / "monthly_feature_panel.csv", V140 / "outer_fold_rows.csv", V140 / "region_roles.csv",
             V140 / "summary.json", PREPARED / "additional_monthly_panel.csv", PREPARED / "donor_inventory.csv", PREPARED / "summary.json"]
    payload = {"status": "FROZEN_TARGET_ROLE_SOURCE_DIVERSITY_DEVELOPMENT_ABLATION", "target_region_months": 589,
               "scored_rows": len(rows), "new_scored_rows": len(outputs), "assertion_failures": len(failures),
               "archived_original_predictions_exactly_reproduced": True,
               "source_scenarios": SCENARIOS, "eligible_donor_inventory": json.loads(inventory.to_json(orient="records")),
               "frontier": json.loads(frontier.to_json(orient="records")),
               "input_hashes": {str(p.relative_to(ROOT)): v139.sha(p) for p in paths},
               "script_sha256": v139.sha(Path(__file__)), "v140_helper_sha256": v139.sha(Path(v140.__file__)),
               "limitations": ["Opened-source development experiment; source histories added to fitting only.",
                               "Increased source support is a data gain, not an original algorithm.",
                               "Monthly target-period aggregates and fixed EF-box proxy labels, not prospective Chinese CIF.",
                               "The original hourly forecast/CRPS gate is not evaluated here."]}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(frontier.loc[frontier.alpha.eq(.1), ["source_scenario", "seed", "tau", "method", "macro_coverage",
           "macro_width", "possible_positive_coverage", "robust_interval_score90"]].to_string(index=False))
    print(sc.loc[sc.alpha.eq(.1)].to_string(index=False))
    print(f"assertion failures: {len(failures)}")


if __name__ == "__main__":
    main()
