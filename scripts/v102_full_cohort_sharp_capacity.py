"""V102: compare exact group-capacity identification with V100 on the full feasible cohort."""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction as F
from pathlib import Path

import pandas as pd
import v90_coarse_capacity_identification as v90
from scripts.v92_sharp_coarse_capacity_identification import sharp_group_set

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v102-full-cohort-sharp-capacity-r1"
PROTOCOL = ROOT / "docs/V102_FULL_COHORT_SHARP_CAPACITY_PROTOCOL.md"
COHORT = ROOT / "results/v100-no-peak-screen-capacity-r1/nonselective_cohort.csv"
V100_ROWS = ROOT / "results/v100-no-peak-screen-capacity-r1/all_energy_feasible_frontier.csv"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {OUT}")
    eligible = pd.read_csv(COHORT)
    eligible = eligible[eligible.monthly_group_energy_feasible.astype(bool)]
    cohort = {(r.region, str(r.month)[:10]) for r in eligible.itertuples(index=False)}
    v100 = pd.read_csv(V100_ROWS)
    v100 = v100[v100.method.eq("lagged_coarse_capacity")].drop_duplicates(["region", "month", "tau"])
    v100["month_key"] = pd.to_datetime(v100.month, utc=True).dt.strftime("%Y-%m-01")
    data = v90.source_data
    data.ALL_REGIONS = tuple(dict.fromkeys(data.ALL_REGIONS + v90.REGIONS))
    hourly, _, audits, input_hashes = data.prepare_regions()
    cap, cap_hashes = {}, {}
    for year in (2023, 2024):
        cap[year], cap_hashes[str(year)] = v90.capacities(year)
    rows = []
    for region in v90.REGIONS:
        frame = hourly[region].copy()
        times = pd.to_datetime(frame.time, utc=True)
        frame = frame.loc[times.ge("2024-01-01") & times.lt("2026-01-01") & frame.valid.astype(bool)].copy()
        frame["time"] = pd.to_datetime(frame.time, utc=True)
        frame["month_key"] = frame.time.dt.strftime("%Y-%m-01")
        for month, group in frame.groupby("month_key", sort=True):
            if (region, month) not in cohort:
                continue
            T, vintage = len(group), int(month[:4]) - 1
            margins = {fuel: sum((F(str(float(x))) for x in group[f"generation_{fuel}"]), F(0)) for fuel in v90.FUELS}
            cap_row = cap[vintage].loc[region] if region in cap[vintage].index else pd.Series(dtype=float)
            # Match V100's frozen float aggregation order for identical input bounds.
            group_caps = {name: sum(float(cap_row.get(fuel, 0)) for fuel in fuels)
                          for name, fuels in v90.GROUPS.items()}
            values = []
            for _, obs in group.iterrows():
                generation = [F(str(float(obs[f"generation_{fuel}"]))) for fuel in v90.FUELS]
                total = sum(generation, F(0))
                emissions = sum((generation[i] * v90.EF[v90.FUELS[i]] for i in range(len(v90.FUELS))), F(0))
                values.append((total, emissions))
            for tau in v90.TAUS:
                actual = sum(emissions > F(tau) * total for total, emissions in values)
                exact_set = sharp_group_set(margins, group_caps, tau, T)
                base_set = v90.v66_set(margins, tau, T)
                relaxed = v90.capacity_group_set(margins, group_caps, tau, T, v90.GROUPS)
                outer_set = tuple(k for k in relaxed if k in base_set) if relaxed is not None else ()
                prior = v100[(v100.region.eq(region)) & v100.month_key.eq(month) & v100.tau.eq(tau)]
                if prior.empty:
                    raise ValueError(f"V100 comparator missing {region}/{month}/tau={tau}")
                approx_lower, approx_upper = int(prior.set_lower.iloc[0]), int(prior.set_upper.iloc[0])
                if not exact_set or actual not in exact_set or not set(exact_set).issubset(base_set):
                    raise ValueError(f"exact-set coverage/nesting failure {region}/{month}/tau={tau}")
                if not set(exact_set).issubset(outer_set):
                    raise ValueError(f"exact set is not contained in relaxed outer set {region}/{month}/tau={tau}")
                if min(exact_set) < approx_lower or max(exact_set) > approx_upper:
                    raise ValueError(f"exact set outside V100 at {region}/{month}/tau={tau}: exact=({min(exact_set)},{max(exact_set)}), V100=({approx_lower},{approx_upper}), relaxed=({min(outer_set)},{max(outer_set)}), T={T}")
                for rate in v90.RATES:
                    exact_width = max(exact_set) - min(exact_set)
                    outer_width = max(outer_set) - min(outer_set)
                    rows.append({"region": region, "month": month, "tau": tau, "rate": rate, "hours": T, "actual_K": actual,
                                 "capacity_peak_screen_pass": bool(eligible.loc[(eligible.region.eq(region)) & eligible.month.astype(str).str[:10].eq(month), "capacity_peak_screen_pass"].iloc[0]),
                                 "exact_lower": min(exact_set), "exact_upper": max(exact_set), "exact_width": exact_width,
                                 "outer_lower": min(outer_set), "outer_upper": max(outer_set), "outer_width": outer_width,
                                 "v100_lower": approx_lower, "v100_upper": approx_upper,
                                 "recomputed_outer_matches_v100": (min(outer_set), max(outer_set)) == (approx_lower, approx_upper),
                                 "same_endpoint": (min(exact_set), max(exact_set)) == (approx_lower, approx_upper),
                                 "same_integer_set": exact_set == outer_set, "actual_K_covered": actual in exact_set,
                                 "positive_certificate": min(exact_set) >= int((F(str(rate)) * T).__ceil__()),
                                 "negative_certificate": max(exact_set) < int((F(str(rate)) * T).__ceil__())})
    OUT.mkdir(parents=True)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "exact_vs_relaxed_frontier.csv", index=False)
    region = result.groupby(["region", "tau", "rate", "month"]).agg(
        exact_width=("exact_width", "mean"), outer_width=("outer_width", "mean"),
        same_endpoint=("same_endpoint", "all"), same_integer_set=("same_integer_set", "all"),
        actual_K_covered=("actual_K_covered", "all"), peak_screen_pass=("capacity_peak_screen_pass", "all"),
    ).reset_index().groupby(["region", "tau", "rate"]).agg(
        region_months=("month", "nunique"), exact_width=("exact_width", "mean"), outer_width=("outer_width", "mean"),
        endpoint_match_rate=("same_endpoint", "mean"), integer_set_match_rate=("same_integer_set", "mean"),
        coverage=("actual_K_covered", "mean"), peak_pass_rate=("peak_screen_pass", "mean"),
    ).reset_index()
    macro = region.groupby(["tau", "rate"]).agg(
        regions=("region", "nunique"), mean_region_months=("region_months", "mean"),
        exact_width=("exact_width", "mean"), outer_width=("outer_width", "mean"),
        endpoint_match_rate=("endpoint_match_rate", "mean"), integer_set_match_rate=("integer_set_match_rate", "mean"),
        coverage=("coverage", "mean"), peak_pass_rate=("peak_pass_rate", "mean"),
    ).reset_index()
    macro.to_csv(OUT / "region_equal_summary.csv", index=False)
    payload = {"status": "EXACT_SHARP_GROUP_CAPACITY_VS_OUTER_SET_FULL_V100_COHORT",
               "region_months": int(result[["region", "month"]].drop_duplicates().shape[0]),
               "scored_rows": len(result), "frontier": macro.to_dict("records"),
               "capacity_source_sha256": cap_hashes, "hourly_source_sha256": input_hashes, "source_audits": audits,
               "protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__)),
               "limitations": ["Previously opened EIA regions, fixed-EF generation proxy; this is not blind, unseen-region, or China validation.",
                               "Sharpness is conditional on exact detailed monthly fuel margins, lagged shared group caps, positive output in both blocks, and no cross-hour constraints.",
                               "The 513-month cohort is selected only by hourly coverage and monthly-total feasibility; target-hour peak failures remain included."]}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
