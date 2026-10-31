"""V100: evaluate lagged coarse-capacity tail sets without target-hour peak screening."""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction as F
from pathlib import Path

import pandas as pd
import v90_coarse_capacity_identification as v90

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v100-no-peak-screen-capacity-r1"
PROTOCOL = ROOT / "docs/V100_NO_PEAK_SCREEN_CAPACITY_PROTOCOL.md"
ROSTER = ROOT / "results/eia-ba-quality-census-v52-r1/quality_census.json"
RATES = (0.01, 0.05, 0.10)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def robust_decision(counts, actual, required):
    if not counts:
        return {"lower": None, "upper": None, "width": None, "positive": False, "negative": False,
                "abstain": True, "covered": False, "certificate_valid": False, "certificate_error": False}
    lo, hi = min(counts), max(counts)
    covered = actual in counts
    positive, negative = lo >= required, hi < required
    error = (positive and actual < required) or (negative and actual >= required)
    return {"lower": lo, "upper": hi, "width": hi - lo, "positive": positive, "negative": negative,
            "abstain": not (positive or negative), "covered": covered,
            "certificate_valid": covered and not error, "certificate_error": error or ((positive or negative) and not covered)}


def finalize_existing_outputs():
    """Recover summary JSON after a completed run whose CSVs survived serialization failure."""
    required = [OUT / "all_energy_feasible_frontier.csv", OUT / "nonselective_cohort.csv", OUT / "frontier_summary.csv"]
    if not all(path.is_file() for path in required) or (OUT / "summary.json").exists():
        raise FileExistsError(f"output directory exists but is not a resumable completed run: {OUT}")
    rows = pd.read_csv(required[0])
    cohort = pd.read_csv(required[1])
    v90_summary = json.loads((ROOT / "results/v90-coarse-capacity-identification-r5/summary.json").read_text())
    counts = cohort.groupby(["year", "monthly_group_energy_feasible", "capacity_peak_screen_pass"], dropna=False).size().rename("region_months").reset_index()
    rows["region_month"] = rows.region + "/" + rows.month
    metrics = ["actual_K_covered", "set_width", "positive", "negative", "abstain", "certificate_error", "certificate_valid"]
    micro = rows.groupby(["method", "tau", "rate"]).agg(
        region_months=("region_month", "nunique"), rows=("actual_K", "size"),
        set_coverage=("actual_K_covered", "mean"), mean_width=("set_width", "mean"),
        positive=("positive", "mean"), negative=("negative", "mean"), abstain=("abstain", "mean"),
        certificate_error=("certificate_error", "mean"), certificate_valid=("certificate_valid", "mean"),
    ).reset_index()
    region = rows.groupby(["method", "region", "tau", "rate"]).agg(
        region_months=("region_month", "nunique"), set_coverage=("actual_K_covered", "mean"),
        mean_width=("set_width", "mean"), positive=("positive", "mean"), negative=("negative", "mean"),
        abstain=("abstain", "mean"), certificate_error=("certificate_error", "mean"),
        certificate_valid=("certificate_valid", "mean"),
    ).reset_index()
    macro = region.groupby(["method", "tau", "rate"]).agg(
        regions=("region", "nunique"), mean_region_months=("region_months", "mean"),
        set_coverage=("set_coverage", "mean"), mean_width=("mean_width", "mean"),
        positive=("positive", "mean"), negative=("negative", "mean"), abstain=("abstain", "mean"),
        certificate_error=("certificate_error", "mean"), certificate_valid=("certificate_valid", "mean"),
    ).reset_index()
    peak_region = rows[rows.capacity_peak_screen_pass].groupby(["method", "region", "tau", "rate"]).agg(
        peak_screen_months=("region_month", "nunique"), peak_screen_coverage=("actual_K_covered", "mean"),
        peak_screen_mean_width=("set_width", "mean"), peak_screen_positive=("positive", "mean"),
        peak_screen_certificate_error=("certificate_error", "mean"),
    ).reset_index()
    peak_macro = peak_region.groupby(["method", "tau", "rate"]).agg(
        peak_screen_regions=("region", "nunique"), peak_screen_mean_months=("peak_screen_months", "mean"),
        peak_screen_coverage=("peak_screen_coverage", "mean"), peak_screen_mean_width=("peak_screen_mean_width", "mean"),
        peak_screen_positive=("peak_screen_positive", "mean"),
        peak_screen_certificate_error=("peak_screen_certificate_error", "mean"),
    ).reset_index()
    macro = macro.merge(peak_macro, on=["method", "tau", "rate"], how="left")
    micro.to_csv(OUT / "frontier_summary_month_weighted.csv", index=False)
    macro.to_csv(OUT / "frontier_summary_region_equal.csv", index=False)
    macro.to_csv(OUT / "frontier_summary.csv", index=False)
    frontier_frame = macro
    frontier = [{key: (None if pd.isna(value) else value) for key, value in record.items()}
                for record in frontier_frame.to_dict("records")]
    payload = {"status": "NO_TARGET_HOURLY_PEAK_SCREEN_RETROSPECTIVE_ALL_CANDIDATE_FRAME",
               "nonselective_months_with_coverage_ge_80pct": int(cohort.shape[0]),
               "monthly_group_energy_feasible_months": int(cohort.monthly_group_energy_feasible.sum()),
               "scored_region_months": int(rows[["region", "month"]].drop_duplicates().shape[0]),
               "cohort_counts": counts.to_dict("records"), "frontier": frontier,
               "hourly_source_sha256": v90_summary["hourly_source_sha256"],
               "capacity_source_sha256": v90_summary["capacity_source_sha256"],
               "protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__)),
               "limitations": ["Already-opened EIA regions and fixed-EF proxy; this is not unseen-region or China validation.",
                               "Lagged annual capacity is a proxy; month-energy feasibility is pre-outcome, but observed K remains retrospective evaluation.",
                               "The group capacity set is a conservative outer approximation intersected with V66; it is not claimed sharp.",
                               "Hourly coverage under 80% is excluded by predeclared data-quality rule; peak-capacity violations remain in the main cohort and count against coverage."]}
    (OUT / "summary.json").write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return payload


def main():
    if OUT.exists():
        payload = finalize_existing_outputs()
        print("recovered completed CSV run summary", payload["scored_region_months"])
        print(pd.read_csv(OUT / "frontier_summary_region_equal.csv").to_string(index=False))
        return
    roster = json.loads(ROSTER.read_text())
    if tuple(roster["quality_qualified_regions_alphabetical"]) != v90.REGIONS:
        raise ValueError("V90 roster mismatch")
    capacities, cap_hashes = {}, {}
    for year in (2023, 2024):
        capacities[year], cap_hashes[str(year)] = v90.capacities(year)
    source = v90.source_data
    source.ALL_REGIONS = tuple(dict.fromkeys(source.ALL_REGIONS + v90.REGIONS))
    hourly, _, audits, input_hashes = source.prepare_regions()
    rows, cohort_rows = [], []
    for region in v90.REGIONS:
        frame = hourly[region].copy()
        time = pd.to_datetime(frame.time, utc=True)
        frame = frame.loc[time.ge("2024-01-01") & time.lt("2026-01-01") & frame.valid.astype(bool)].copy()
        frame["time"] = pd.to_datetime(frame.time, utc=True)
        frame["month"] = frame.time.dt.strftime("%Y-%m-01")
        for month, group in frame.groupby("month", sort=True):
            T = len(group)
            year = int(month[:4])
            vintage = year - 1
            coverage = T / (pd.Timestamp(month).days_in_month * 24)
            margins = {fuel: sum((F(str(float(x))) for x in group[f"generation_{fuel}"]), F(0)) for fuel in v90.FUELS}
            cap_row = capacities[vintage].loc[region] if region in capacities[vintage].index else pd.Series(dtype=float)
            fine_caps = {fuel: float(cap_row.get(fuel, 0)) for fuel in v90.FUELS}
            group_caps = {name: sum(fine_caps[fuel] for fuel in members) for name, members in v90.GROUPS.items()}
            group_mwh = {name: sum(float(margins[fuel]) for fuel in members) for name, members in v90.GROUPS.items()}
            energy_ok = all(group_mwh[name] <= T * group_caps[name] + 1e-6 for name in v90.GROUPS)
            if coverage < 0.8:
                continue
            cohort_rows.append({"region": region, "month": month, "year": year, "capacity_vintage": vintage,
                                "hours": T, "coverage": coverage, "monthly_group_energy_feasible": energy_ok})
            if not energy_ok:
                continue
            vals = []
            for _, obs in group.iterrows():
                generation = [F(str(float(obs[f"generation_{fuel}"]))) for fuel in v90.FUELS]
                total = sum(generation, F(0))
                emissions = sum((generation[i] * v90.EF[v90.FUELS[i]] for i in range(len(v90.FUELS))), F(0))
                vals.append((total, emissions))
            for tau in v90.TAUS:
                actual = sum(emissions > F(tau) * total for total, emissions in vals)
                monthly_set = v90.v66_set(margins, tau, T)
                candidate = v90.capacity_group_set(margins, group_caps, tau, T, v90.GROUPS)
                if candidate is None:
                    raise ValueError("monthly-energy-feasible capacity set unexpectedly infeasible")
                cap_set = tuple(k for k in candidate if k in monthly_set)
                # Construct the original V90 peak-screen condition only for paired descriptive reporting.
                peak_ok = all(float(group[[f"generation_{f}" for f in members]].sum(axis=1).max()) <= group_caps[name] + 1e-6
                              for name, members in v90.GROUPS.items())
                for rate in RATES:
                    req = int((F(str(rate)) * T).__ceil__())
                    for method, counts in (("monthly_margin", monthly_set), ("lagged_coarse_capacity", cap_set)):
                        d = robust_decision(counts, actual, req)
                        rows.append({"region": region, "month": month, "tau": tau, "rate": rate, "required_hours": req,
                                     "actual_K": actual, "method": method, "capacity_peak_screen_pass": peak_ok,
                                     "set_lower": d["lower"], "set_upper": d["upper"], "set_width": d["width"],
                                     "positive": d["positive"], "negative": d["negative"], "abstain": d["abstain"],
                                     "actual_K_covered": d["covered"], "certificate_valid": d["certificate_valid"],
                                     "certificate_error": d["certificate_error"]})
    OUT.mkdir(parents=True)
    row_df, cohort_df = pd.DataFrame(rows), pd.DataFrame(cohort_rows)
    row_df.to_csv(OUT / "all_energy_feasible_frontier.csv", index=False)
    cohort_df.to_csv(OUT / "nonselective_cohort.csv", index=False)
    cohort_summary = cohort_df.groupby(["year", "monthly_group_energy_feasible", "capacity_peak_screen_pass"], dropna=False).size().rename("region_months").reset_index() if "capacity_peak_screen_pass" in cohort_df else None
    # Peak screening is paired at the scored-month level; join it from the long frontier.
    peak = row_df.drop_duplicates(["region", "month"])[["region", "month", "capacity_peak_screen_pass"]]
    cohort_df = cohort_df.merge(peak, on=["region", "month"], how="left")
    cohort_df.to_csv(OUT / "nonselective_cohort.csv", index=False)
    cohort_summary = cohort_df.groupby(["year", "monthly_group_energy_feasible", "capacity_peak_screen_pass"], dropna=False).size().rename("region_months").reset_index()
    row_df["region_month"] = row_df.region + "/" + row_df.month
    region = row_df.groupby(["method", "region", "tau", "rate"]).agg(
        region_months=("region_month", "nunique"), set_coverage=("actual_K_covered", "mean"),
        mean_width=("set_width", "mean"), positive=("positive", "mean"), negative=("negative", "mean"),
        abstain=("abstain", "mean"), certificate_error=("certificate_error", "mean"),
        certificate_valid=("certificate_valid", "mean"),
    ).reset_index()
    macro = region.groupby(["method", "tau", "rate"]).agg(
        regions=("region", "nunique"), mean_region_months=("region_months", "mean"),
        set_coverage=("set_coverage", "mean"), mean_width=("mean_width", "mean"),
        positive=("positive", "mean"), negative=("negative", "mean"), abstain=("abstain", "mean"),
        certificate_error=("certificate_error", "mean"), certificate_valid=("certificate_valid", "mean"),
    ).reset_index()
    peak_region = row_df[row_df.capacity_peak_screen_pass].groupby(["method", "region", "tau", "rate"]).agg(
        peak_screen_months=("region_month", "nunique"), peak_screen_coverage=("actual_K_covered", "mean"),
        peak_screen_mean_width=("set_width", "mean"), peak_screen_positive=("positive", "mean"),
        peak_screen_certificate_error=("certificate_error", "mean"),
    ).reset_index()
    peak_macro = peak_region.groupby(["method", "tau", "rate"]).agg(
        peak_screen_regions=("region", "nunique"), peak_screen_mean_months=("peak_screen_months", "mean"),
        peak_screen_coverage=("peak_screen_coverage", "mean"), peak_screen_mean_width=("peak_screen_mean_width", "mean"),
        peak_screen_positive=("peak_screen_positive", "mean"),
        peak_screen_certificate_error=("peak_screen_certificate_error", "mean"),
    ).reset_index()
    macro = macro.merge(peak_macro, on=["method", "tau", "rate"], how="left")
    macro.to_csv(OUT / "frontier_summary.csv", index=False)
    payload = {"status": "NO_TARGET_HOURLY_PEAK_SCREEN_RETROSPECTIVE_ALL_CANDIDATE_FRAME",
               "nonselective_months_with_coverage_ge_80pct": int(cohort_df.shape[0]),
               "monthly_group_energy_feasible_months": int(cohort_df.monthly_group_energy_feasible.sum()),
               "scored_region_months": int(row_df[["region", "month"]].drop_duplicates().shape[0]),
               "cohort_counts": cohort_summary.to_dict("records"), "frontier": macro.to_dict("records"),
               "hourly_source_sha256": input_hashes, "capacity_source_sha256": cap_hashes,
               "protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__)), "source_audits": audits,
               "limitations": ["Already-opened EIA regions and fixed-EF proxy; this is not unseen-region or China validation.",
                               "Lagged annual capacity is a proxy; month-energy feasibility is pre-outcome, but observed K remains retrospective evaluation.",
                               "The group capacity set is a conservative outer approximation intersected with V66; it is not claimed sharp.",
                               "Hourly coverage under 80% is excluded by predeclared data-quality rule; peak-capacity violations remain in the main cohort and count against coverage."]}
    (OUT / "summary.json").write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(cohort_summary.to_string(index=False))
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
