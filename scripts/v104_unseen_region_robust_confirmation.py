"""V104: frozen factor-robust exact-set confirmation on five unseen EIA regions."""
from __future__ import annotations

import hashlib
import json
import sys
from fractions import Fraction as F
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v90_coarse_capacity_identification as v90  # noqa: E402
from scripts.v92_sharp_coarse_capacity_identification import sharp_group_set  # noqa: E402

REGIONS = ("CPLE", "FPC", "IPCO", "PNM", "SOCO")
TAUS = (600, 900)
RATES = (0.01, 0.05, 0.10)
DELTA = 0.10
OUT = ROOT / "results/v104-unseen-region-robust-confirmation-r1"
PROTOCOL = ROOT / "docs/V104_UNSEEN_REGION_ROBUST_CONFIRMATION_PROTOCOL.md"
CENSUS = ROOT / "results/eia-ba-quality-census-v52-r1/quality_census.json"


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


def exact_set(margins, caps, tau, hours, factors):
    prior = dict(v90.EF)
    try:
        v90.EF.clear()
        v90.EF.update(factors)
        return sharp_group_set(margins, caps, tau, hours)
    finally:
        v90.EF.clear()
        v90.EF.update(prior)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {OUT}")
    census = json.loads(CENSUS.read_text())
    if census["status"] != "QUALITY_ONLY_NO_CIF_SCORES_OR_REGION_RANKING":
        raise ValueError("unexpected selection census status")
    coverage = {
        region: float(census["periods"][region]["2026"]["complete_proxy_coverage"])
        for region in REGIONS
    }
    if any(value < 0.98 for value in coverage.values()):
        raise ValueError(f"frozen region roster fails quality threshold: {coverage}")

    # Expand the established loader only to the quality-selected unseen roster.
    v90.REGIONS = REGIONS
    v90.source_data.ALL_REGIONS = tuple(
        dict.fromkeys(v90.source_data.ALL_REGIONS + REGIONS)
    )
    hourly, _, audits, hourly_hashes = v90.source_data.prepare_regions()
    capacity, capacity_hash = v90.capacities(2025)

    nominal = dict(v90.EF)
    low = {fuel: ef * (1 - DELTA) if fuel in ("coal", "gas", "oil", "other") else ef
           for fuel, ef in nominal.items()}
    high = {fuel: ef * (1 + DELTA) if fuel in ("coal", "gas", "oil", "other") else ef
            for fuel, ef in nominal.items()}
    rows, availability = [], []
    for region in REGIONS:
        frame = hourly[region].copy()
        times = pd.to_datetime(frame.time, utc=True)
        frame = frame.loc[times.ge("2026-01-01") & times.lt("2026-07-01")
                          & frame.valid.astype(bool)].copy()
        frame["time"] = pd.to_datetime(frame.time, utc=True)
        frame["month"] = frame.time.dt.strftime("%Y-%m-01")
        for month in pd.date_range("2026-01-01", "2026-06-01", freq="MS", tz="UTC"):
            month_key = month.strftime("%Y-%m-01")
            group = frame.loc[frame.month.eq(month_key)]
            expected = month.days_in_month * 24
            hours = len(group)
            caps_row = capacity.loc[region] if region in capacity.index else pd.Series(dtype=float)
            fuel_caps = {fuel: float(caps_row.get(fuel, 0)) for fuel in
                         ("coal", "gas", "oil", "nuclear", "hydro", "wind", "solar", "other")}
            group_caps = {name: sum(fuel_caps[fuel] for fuel in fuels)
                          for name, fuels in v90.GROUPS.items()}
            margins = {fuel: sum((F(str(float(x))) for x in group[f"generation_{fuel}"]), F(0))
                       for fuel in v90.FUELS}
            energy_ok = hours > 0 and all(
                sum(float(margins[fuel]) for fuel in fuels) <= hours * group_caps[name] + 1e-6
                for name, fuels in v90.GROUPS.items()
            )
            eligible = hours / expected >= 0.80 and energy_ok
            availability.append({"region": region, "month": month_key, "hours": hours,
                                 "expected_hours": expected, "coverage": hours / expected,
                                 "capacity_vintage": 2025, "group_energy_feasible": bool(energy_ok),
                                 "scored": bool(eligible), **{f"capacity_{g}_mw": value
                                                               for g, value in group_caps.items()}})
            if not eligible:
                continue

            observed = []
            for _, obs in group.iterrows():
                generation = [F(str(float(obs[f"generation_{fuel}"]))) for fuel in v90.FUELS]
                total = sum(generation, F(0))
                emissions = sum((generation[i] * nominal[v90.FUELS[i]]
                                 for i in range(len(v90.FUELS))), F(0))
                observed.append((total, emissions))
            for tau in TAUS:
                actual = sum(emissions > F(tau) * total for total, emissions in observed)
                base = exact_set(margins, group_caps, tau, hours, nominal)
                low_set = exact_set(margins, group_caps, tau, hours, low)
                high_set = exact_set(margins, group_caps, tau, hours, high)
                lower, upper = min(low_set), max(high_set)
                if not (lower <= min(base) <= max(base) <= upper):
                    raise ValueError(f"robust hull excludes nominal set: {region}/{month_key}/{tau}")
                if not lower <= actual <= upper:
                    raise ValueError(f"robust hull excludes observed proxy K: {region}/{month_key}/{tau}")
                for rate in RATES:
                    required = int((F(str(rate)) * hours).__ceil__())
                    base_lower, base_upper = min(base), max(base)
                    rows.append({"region": region, "month": month_key, "tau": tau, "rate": rate,
                                 "hours": hours, "required_hours": required, "actual_nominal_K": actual,
                                 "nominal_lower": base_lower, "nominal_upper": base_upper,
                                 "nominal_width": base_upper - base_lower,
                                 "low_EF_lower": min(low_set), "low_EF_upper": max(low_set),
                                 "high_EF_lower": min(high_set), "high_EF_upper": max(high_set),
                                 "robust_lower": lower, "robust_upper": upper,
                                 "robust_width": upper - lower,
                                 "nominal_positive": base_lower >= required,
                                 "nominal_negative": base_upper < required,
                                 "nominal_abstain": not (base_lower >= required or base_upper < required),
                                 "robust_positive": lower >= required,
                                 "robust_negative": upper < required,
                                 "robust_abstain": not (lower >= required or upper < required),
                                 "actual_covered": True})

    scored = pd.DataFrame(rows)
    if scored.empty:
        raise ValueError("No eligible unseen-region months")
    OUT.mkdir(parents=True, exist_ok=False)
    scored.to_csv(OUT / "region_month_scores.csv", index=False)
    avail = pd.DataFrame(availability)
    avail.to_csv(OUT / "availability_and_capacity.csv", index=False)

    region_month = scored.groupby(["region", "month", "tau", "rate"]).agg(
        nominal_width=("nominal_width", "first"), robust_width=("robust_width", "first"),
        nominal_positive=("nominal_positive", "mean"), nominal_negative=("nominal_negative", "mean"),
        nominal_abstain=("nominal_abstain", "mean"), robust_positive=("robust_positive", "mean"),
        robust_negative=("robust_negative", "mean"), robust_abstain=("robust_abstain", "mean"),
        actual_coverage=("actual_covered", "mean"),
    ).reset_index()
    per_region = region_month.groupby(["region", "tau", "rate"]).agg(
        region_months=("month", "nunique"), nominal_width=("nominal_width", "mean"),
        robust_width=("robust_width", "mean"), nominal_positive=("nominal_positive", "mean"),
        nominal_negative=("nominal_negative", "mean"), nominal_abstain=("nominal_abstain", "mean"),
        robust_positive=("robust_positive", "mean"), robust_negative=("robust_negative", "mean"),
        robust_abstain=("robust_abstain", "mean"), actual_coverage=("actual_coverage", "mean"),
    ).reset_index()
    per_region.to_csv(OUT / "per_region_frontier.csv", index=False)
    macro = per_region.groupby(["tau", "rate"]).agg(
        regions=("region", "nunique"), mean_region_months=("region_months", "mean"),
        nominal_width=("nominal_width", "mean"), robust_width=("robust_width", "mean"),
        nominal_positive=("nominal_positive", "mean"), nominal_negative=("nominal_negative", "mean"),
        nominal_abstain=("nominal_abstain", "mean"), robust_positive=("robust_positive", "mean"),
        robust_negative=("robust_negative", "mean"), robust_abstain=("robust_abstain", "mean"),
        actual_coverage=("actual_coverage", "mean"),
    ).reset_index()
    macro.to_csv(OUT / "region_equal_frontier.csv", index=False)
    pooled = scored.groupby(["tau", "rate"]).agg(
        region_months=("month", "nunique"), nominal_width=("nominal_width", "mean"),
        robust_width=("robust_width", "mean"), nominal_positive=("nominal_positive", "mean"),
        nominal_negative=("nominal_negative", "mean"), nominal_abstain=("nominal_abstain", "mean"),
        robust_positive=("robust_positive", "mean"), robust_negative=("robust_negative", "mean"),
        robust_abstain=("robust_abstain", "mean"), actual_coverage=("actual_covered", "mean"),
    ).reset_index()
    pooled.to_csv(OUT / "pooled_frontier.csv", index=False)
    payload = {
        "status": "BLIND_REGION_TRANSFER_FACTOR_ROBUST_RETROSPECTIVE_IDENTIFICATION",
        "regions": list(REGIONS), "quality_only_2026_complete_proxy_coverage": coverage,
        "scored_region_months": int(scored[["region", "month"]].drop_duplicates().shape[0]),
        "scored_rows": len(scored), "eligible_regions": sorted(scored.region.unique().tolist()),
        "unavailable_region_months": int((~avail.scored).sum()),
        "availability_by_region": avail.groupby("region").agg(
            candidate_months=("month", "nunique"), scored_months=("scored", "sum"),
            mean_label_coverage=("coverage", "mean"), group_energy_feasible_months=("group_energy_feasible", "sum")
        ).reset_index().to_dict("records"),
        "region_equal_frontier": macro.to_dict("records"), "pooled_frontier": pooled.to_dict("records"),
        "thermal_factor_relative_stress_radius": DELTA, "nominal_ef": nominal,
        "low_ef": low, "high_ef": high, "capacity_vintage": 2025,
        "capacity_source_sha256": capacity_hash, "hourly_source_sha256": hourly_hashes,
        "source_audits": audits, "quality_census_sha256": sha(CENSUS),
        "protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__)),
        "limitations": [
            "This is region-heldout transfer within 2026H1, not a future-time blind holdout.",
            "Labels are fixed-factor EIA generation proxies rather than measured CIF.",
            "The ±10% factor box is a stress scenario, not an estimated factor uncertainty interval.",
            "Target-month monthly energy totals and lagged capacity are inputs; this is retrospective identification, not month-ahead prediction.",
            "Infeasible or low-coverage months are unavailable and must be counted as abstentions in deployment-level coverage.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False,
                                                   allow_nan=False) + "\n")
    print("availability", payload["availability_by_region"])
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
