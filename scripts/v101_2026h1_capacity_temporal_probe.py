"""V101: transport frozen V100 capacity identification to the 2026H1 time block."""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction as F
from pathlib import Path

import pandas as pd
import v90_coarse_capacity_identification as v90
from v100_no_peak_screen_capacity import robust_decision

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v101-2026h1-capacity-temporal-probe-r1"
PROTOCOL = ROOT / "docs/V101_2026H1_CAPACITY_TEMPORAL_PROBE_PROTOCOL.md"
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
    return value.item() if hasattr(value, "item") else value


def main():
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {OUT}")
    roster = json.loads(ROSTER.read_text())
    if tuple(roster["quality_qualified_regions_alphabetical"]) != v90.REGIONS:
        raise ValueError("frozen V52 region roster mismatch")
    caps, cap_hash = v90.capacities(2025)
    source = v90.source_data
    source.ALL_REGIONS = tuple(dict.fromkeys(source.ALL_REGIONS + v90.REGIONS))
    hourly, _, audits, input_hashes = source.prepare_regions()
    rows, cohort = [], []
    for region in v90.REGIONS:
        frame = hourly[region].copy()
        times = pd.to_datetime(frame.time, utc=True)
        frame = frame.loc[times.ge("2026-01-01") & times.lt("2026-07-01") & frame.valid.astype(bool)].copy()
        frame["time"] = pd.to_datetime(frame.time, utc=True)
        frame["month"] = frame.time.dt.strftime("%Y-%m-01")
        for month, group in frame.groupby("month", sort=True):
            T = len(group)
            coverage = T / (pd.Timestamp(month).days_in_month * 24)
            if coverage < 0.8:
                continue
            margins = {fuel: sum((F(str(float(x))) for x in group[f"generation_{fuel}"]), F(0)) for fuel in v90.FUELS}
            cap_row = caps.loc[region] if region in caps.index else pd.Series(dtype=float)
            group_caps = {name: sum(float(cap_row.get(f, 0)) for f in fuels) for name, fuels in v90.GROUPS.items()}
            group_mwh = {name: sum(float(margins[f]) for f in fuels) for name, fuels in v90.GROUPS.items()}
            energy_ok = all(group_mwh[name] <= T * group_caps[name] + 1e-6 for name in v90.GROUPS)
            observed_peak_ok = all(float(group[[f"generation_{f}" for f in fuels]].sum(axis=1).max()) <= group_caps[name] + 1e-6
                                   for name, fuels in v90.GROUPS.items())
            cohort.append({"region": region, "month": month, "hours": T, "coverage": coverage,
                           "monthly_group_energy_feasible": energy_ok, "observed_hourly_peak_pass": observed_peak_ok})
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
                    raise ValueError("monthly energy passed but capacity set is infeasible")
                cap_set = tuple(k for k in candidate if k in monthly_set)
                for rate in RATES:
                    required = int((F(str(rate)) * T).__ceil__())
                    for method, counts in (("monthly_margin", monthly_set), ("lagged_coarse_capacity", cap_set)):
                        result = robust_decision(counts, actual, required)
                        rows.append({"region": region, "month": month, "tau": tau, "rate": rate, "actual_K": actual,
                                     "method": method, "capacity_peak_screen_pass": observed_peak_ok,
                                     "set_lower": result["lower"], "set_upper": result["upper"], "set_width": result["width"],
                                     "positive": result["positive"], "negative": result["negative"], "abstain": result["abstain"],
                                     "actual_K_covered": result["covered"], "certificate_error": result["certificate_error"]})
    if not cohort:
        raise ValueError("no 2026H1 months passed coverage rule")
    OUT.mkdir(parents=True)
    cohort_df, result_df = pd.DataFrame(cohort), pd.DataFrame(rows)
    cohort_df.to_csv(OUT / "cohort.csv", index=False)
    result_df.to_csv(OUT / "frontier_rows.csv", index=False)
    cohort_counts = cohort_df.groupby(["monthly_group_energy_feasible", "observed_hourly_peak_pass"]).size().rename("region_months").reset_index()
    region = result_df.groupby(["method", "region", "tau", "rate"]).agg(
        region_months=("month", "nunique"), set_coverage=("actual_K_covered", "mean"), mean_width=("set_width", "mean"),
        positive=("positive", "mean"), negative=("negative", "mean"), abstain=("abstain", "mean"),
        certificate_error=("certificate_error", "mean"),
    ).reset_index()
    macro = region.groupby(["method", "tau", "rate"]).agg(
        regions=("region", "nunique"), mean_region_months=("region_months", "mean"),
        set_coverage=("set_coverage", "mean"), mean_width=("mean_width", "mean"), positive=("positive", "mean"),
        negative=("negative", "mean"), abstain=("abstain", "mean"), certificate_error=("certificate_error", "mean"),
    ).reset_index()
    micro = result_df.groupby(["method", "tau", "rate"]).agg(
        region_months=("month", "nunique"), rows=("actual_K", "size"), set_coverage=("actual_K_covered", "mean"),
        mean_width=("set_width", "mean"), positive=("positive", "mean"), negative=("negative", "mean"),
        abstain=("abstain", "mean"), certificate_error=("certificate_error", "mean"),
    ).reset_index()
    macro.to_csv(OUT / "frontier_region_equal.csv", index=False)
    micro.to_csv(OUT / "frontier_pooled.csv", index=False)
    payload = {"status": "TEMPORAL_TRANSPORT_PROBE_NOT_BLIND_2026H1_REUSED_IN_OTHER_EXPERIMENTS",
               "coverage_qualified_region_months": int(cohort_df.shape[0]),
               "capacity_feasible_region_months": int(cohort_df.monthly_group_energy_feasible.sum()),
               "observed_peak_feasible_region_months": int(cohort_df.observed_hourly_peak_pass.sum()),
               "scored_region_months": int(result_df.loc[result_df.method.eq("lagged_coarse_capacity"), ["region", "month"]].drop_duplicates().shape[0]),
               "cohort_counts": cohort_counts.to_dict("records"), "frontier_region_equal": macro.to_dict("records"),
               "frontier_pooled": micro.to_dict("records"), "capacity_source_sha256": {"2025": cap_hash},
               "hourly_source_sha256": input_hashes, "source_audits": audits,
               "protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__)),
               "limitations": ["2026H1 outcomes were already used in unrelated predictor experiments; this is not a sealed blind confirmation.",
                               "Previously opened regions and fixed-EF hourly generation proxy; not real CIF or China validation.",
                               "Monthly target generation is an after-period input, so this evaluates retrospective within-month identification/nowcasting, not a month-ahead forecast.",
                               "Capacity-infeasible months are unavailable and count as abstentions when rates are interpreted over the full cohort."]}
    (OUT / "summary.json").write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(cohort_counts.to_string(index=False))
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
