"""V103: exact hourly-tail identified-set envelope under fuel-factor uncertainty."""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction as F
from pathlib import Path

import pandas as pd
import v90_coarse_capacity_identification as v90
from scripts.v92_sharp_coarse_capacity_identification import sharp_group_set

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v103-factor-uncertainty-sharp-set-r1"
PROTOCOL = ROOT / "docs/V103_FACTOR_UNCERTAINTY_SHARP_SET_PROTOCOL.md"
COHORT = ROOT / "results/v100-no-peak-screen-capacity-r1/nonselective_cohort.csv"
TAUS = (600, 900)
RATES = (0.01, 0.05, 0.10)
DELTA = 0.10
NOMINAL = dict(v90.EF)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_set(margins, caps, tau, hours, factors):
    old = dict(v90.EF)
    try:
        v90.EF.clear()
        v90.EF.update(factors)
        return sharp_group_set(margins, caps, tau, hours)
    finally:
        v90.EF.clear()
        v90.EF.update(old)


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
    eligible = pd.read_csv(COHORT)
    eligible = eligible[eligible.monthly_group_energy_feasible.astype(bool)]
    cohort = {(row.region, str(row.month)[:10]) for row in eligible.itertuples(index=False)}
    data = v90.source_data
    data.ALL_REGIONS = tuple(dict.fromkeys(data.ALL_REGIONS + v90.REGIONS))
    hourly, _, audits, input_hashes = data.prepare_regions()
    capacities, cap_hashes = {}, {}
    for year in (2023, 2024):
        capacities[year], cap_hashes[str(year)] = v90.capacities(year)
    low = {fuel: ef * (1 - DELTA) if fuel in ("coal", "gas", "oil", "other") else ef for fuel, ef in NOMINAL.items()}
    high = {fuel: ef * (1 + DELTA) if fuel in ("coal", "gas", "oil", "other") else ef for fuel, ef in NOMINAL.items()}
    rows = []
    for region in v90.REGIONS:
        frame = hourly[region].copy()
        times = pd.to_datetime(frame.time, utc=True)
        frame = frame.loc[times.ge("2024-01-01") & times.lt("2026-01-01") & frame.valid.astype(bool)].copy()
        frame["time"] = pd.to_datetime(frame.time, utc=True)
        frame["month"] = frame.time.dt.strftime("%Y-%m-01")
        for month, group in frame.groupby("month", sort=True):
            if (region, month) not in cohort:
                continue
            T, vintage = len(group), int(month[:4]) - 1
            margins = {fuel: sum((F(str(float(x))) for x in group[f"generation_{fuel}"]), F(0)) for fuel in v90.FUELS}
            caprow = capacities[vintage].loc[region] if region in capacities[vintage].index else pd.Series(dtype=float)
            fine_caps = {fuel: float(caprow.get(fuel, 0)) for fuel in ("coal", "gas", "oil", "nuclear", "hydro", "wind", "solar", "other")}
            group_caps = {name: sum(fine_caps[fuel] for fuel in fuels) for name, fuels in v90.GROUPS.items()}
            values = []
            for _, obs in group.iterrows():
                generation = [F(str(float(obs[f"generation_{fuel}"]))) for fuel in v90.FUELS]
                total = sum(generation, F(0))
                carbon = sum((generation[i] * NOMINAL[v90.FUELS[i]] for i in range(len(v90.FUELS))), F(0))
                values.append((total, carbon))
            for tau in TAUS:
                actual = sum(carbon > F(tau) * total for total, carbon in values)
                nominal_set = evaluate_set(margins, group_caps, tau, T, NOMINAL)
                low_set = evaluate_set(margins, group_caps, tau, T, low)
                high_set = evaluate_set(margins, group_caps, tau, T, high)
                robust_lower, robust_upper = min(low_set), max(high_set)
                if not (robust_lower <= min(nominal_set) <= max(nominal_set) <= robust_upper):
                    raise ValueError(f"factor hull fails nominal set containment at {region}/{month}/tau={tau}")
                if not robust_lower <= actual <= robust_upper:
                    raise ValueError(f"factor hull fails observed proxy K coverage at {region}/{month}/tau={tau}")
                for rate in RATES:
                    required = int((F(str(rate)) * T).__ceil__())
                    rows.append({"region": region, "month": month, "tau": tau, "rate": rate, "hours": T, "actual_nominal_K": actual,
                                 "capacity_peak_screen_pass": bool(eligible.loc[(eligible.region.eq(region)) & eligible.month.astype(str).str[:10].eq(month), "capacity_peak_screen_pass"].iloc[0]),
                                 "nominal_lower": min(nominal_set), "nominal_upper": max(nominal_set), "nominal_width": max(nominal_set) - min(nominal_set),
                                 "low_EF_lower": min(low_set), "low_EF_upper": max(low_set), "high_EF_lower": min(high_set), "high_EF_upper": max(high_set),
                                 "robust_lower": robust_lower, "robust_upper": robust_upper, "robust_width": robust_upper - robust_lower,
                                 "nominal_positive": min(nominal_set) >= required, "nominal_negative": max(nominal_set) < required,
                                 "robust_positive": robust_lower >= required, "robust_negative": robust_upper < required,
                                 "robust_abstain": not (robust_lower >= required or robust_upper < required), "actual_covered": True})
    OUT.mkdir(parents=True)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "factor_robust_rows.csv", index=False)
    region = result.groupby(["region", "month", "tau", "rate"]).agg(
        nominal_width=("nominal_width", "mean"), robust_width=("robust_width", "mean"),
        nominal_positive=("nominal_positive", "mean"), nominal_negative=("nominal_negative", "mean"),
        robust_positive=("robust_positive", "mean"), robust_negative=("robust_negative", "mean"),
        robust_abstain=("robust_abstain", "mean"), actual_covered=("actual_covered", "all"),
        peak_pass=("capacity_peak_screen_pass", "all"),
    ).reset_index().groupby(["region", "tau", "rate"]).agg(
        region_months=("month", "nunique"), nominal_width=("nominal_width", "mean"), robust_width=("robust_width", "mean"),
        nominal_positive=("nominal_positive", "mean"), nominal_negative=("nominal_negative", "mean"),
        robust_positive=("robust_positive", "mean"), robust_negative=("robust_negative", "mean"),
        robust_abstain=("robust_abstain", "mean"), actual_coverage=("actual_covered", "mean"),
        peak_pass_rate=("peak_pass", "mean"),
    ).reset_index()
    macro = region.groupby(["tau", "rate"]).agg(
        regions=("region", "nunique"), mean_region_months=("region_months", "mean"),
        nominal_width=("nominal_width", "mean"), robust_width=("robust_width", "mean"),
        nominal_positive=("nominal_positive", "mean"), nominal_negative=("nominal_negative", "mean"),
        robust_positive=("robust_positive", "mean"), robust_negative=("robust_negative", "mean"),
        robust_abstain=("robust_abstain", "mean"), actual_coverage=("actual_coverage", "mean"),
        peak_pass_rate=("peak_pass_rate", "mean"),
    ).reset_index()
    pooled = result.groupby(["tau", "rate"]).agg(
        region_months=("month", "nunique"), nominal_width=("nominal_width", "mean"), robust_width=("robust_width", "mean"),
        nominal_positive=("nominal_positive", "mean"), robust_positive=("robust_positive", "mean"),
        robust_negative=("robust_negative", "mean"), robust_abstain=("robust_abstain", "mean"),
        actual_coverage=("actual_covered", "mean"),
    ).reset_index()
    macro.to_csv(OUT / "region_equal_frontier.csv", index=False)
    pooled.to_csv(OUT / "pooled_frontier.csv", index=False)
    payload = {"status": "SHARP_ENDPOINT_ENVELOPE_OVER_FUEL_EF_BOX_NOT_OFFICIAL_FACTOR_UNCERTAINTY",
               "relative_factor_radius": DELTA, "ef_nominal": NOMINAL, "ef_lower": low, "ef_upper": high,
               "region_months": int(result[["region", "month"]].drop_duplicates().shape[0]), "scored_rows": len(result),
               "frontier_region_equal": macro.to_dict("records"), "frontier_pooled": pooled.to_dict("records"),
               "capacity_source_sha256": cap_hashes, "hourly_source_sha256": input_hashes, "source_audits": audits,
               "protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__)),
               "limitations": ["The ±10% EF box is a stress scenario, not an estimated or official factor uncertainty interval.",
                               "The robust interval is the sharp endpoint hull from factor-box monotonicity; not every interior K is proven jointly feasible under one global EF vector.",
                               "Fixed-EF EIA proxy labels and previously opened regions; no real China hourly CIF validation or blind confirmation.",
                               "Monthly energy and lagged annual capacity are target-month inputs; this is retrospective identification, not month-ahead forecasting."]}
    (OUT / "summary.json").write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
