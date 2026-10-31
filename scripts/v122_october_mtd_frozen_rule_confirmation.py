"""V122: apply the frozen support-gap rule to October 2026 month-to-date."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from fractions import Fraction as F
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v54_unseen_local_ef_confirmation as data  # noqa: E402
import v90_coarse_capacity_identification as v90  # noqa: E402
from scripts.v92_sharp_coarse_capacity_identification import sharp_group_set  # noqa: E402

MONTH = "2026-10-01"
REGION_NWMT = "NWMT"
TAUS = (600, 900)
RATES = (0.01, 0.05, 0.10)
DELTA = 0.10
ALPHA = 0.10
SUPPORT_TAIL = 0.20
OUT = ROOT / "results/v122-october-mtd-frozen-rule-confirmation-r1"
PROTOCOL = ROOT / "docs/V122_OCTOBER_MTD_FROZEN_RULE_CONFIRMATION_PROTOCOL.md"
CALIBRATION_ROWS = ROOT / "results/v111-factor-robust-monthly-anchor-r1/source_calibration_rows.csv"
SUPPORT_FEATURES = ROOT / "results/v120-support-gap-region-loo-cv-r2/source_outer_fold_features.csv"
NATIVE = ROOT / "data/native-eia-2026/EIA930_BALANCE_2026_Jul_Dec.csv"
CENSUS = ROOT / "results/eia-ba-quality-census-v52-r1/quality_census.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def factors_box():
    nominal = dict(v90.EF)
    low = {fuel: ef * (1 - DELTA) if fuel in ("coal", "gas", "oil", "other") else ef
           for fuel, ef in nominal.items()}
    high = {fuel: ef * (1 + DELTA) if fuel in ("coal", "gas", "oil", "other") else ef
            for fuel, ef in nominal.items()}
    return {"low": low, "nominal": nominal, "high": high}


def fold_calibration(calibration: pd.DataFrame, features: pd.DataFrame, region: str):
    radii, boundaries, audits = {}, {}, {}
    for tau in TAUS:
        train = calibration.loc[(calibration.tau == tau) & (calibration.region != region)]
        score = train.groupby("region").worst_case_absolute_error.max().sort_values().to_numpy()
        n = len(score); rank = math.ceil((n + 1) * (1 - ALPHA))
        if rank > n:
            raise ValueError(f"non-finite anchor radius for {region}, tau={tau}")
        radii[tau] = float(score[rank - 1])
        feat = features.loc[(features.tau == tau) & (features.region != region)]
        region_max = feat.groupby("region").support_gap.max().sort_values().to_numpy()
        nf = len(region_max); frank = math.ceil((nf + 1) * (1 - SUPPORT_TAIL))
        if frank > nf:
            raise ValueError(f"non-finite support boundary for {region}, tau={tau}")
        boundaries[tau] = float(region_max[frank - 1])
        audits[tau] = {"calibration_regions": n, "calibration_rank": rank,
                       "support_regions": nf, "support_rank": frank}
    return radii, boundaries, audits


def decide(lo, hi, hours, rate):
    req = int((F(str(rate)) * hours).__ceil__())
    return lo >= req, hi < req, not (lo >= req or hi < req), req


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {OUT}")
    census = json.loads(CENSUS.read_text())
    regions = tuple(sorted(region for region, periods in census["periods"].items()
                           if float(periods["2026"]["complete_proxy_coverage"]) >= 0.90) + [REGION_NWMT])
    h1_coverage = {region: float(census["periods"][region]["2026"]["complete_proxy_coverage"])
                   for region in regions}
    v90.REGIONS = regions
    v90.source_data.ALL_REGIONS = tuple(dict.fromkeys(v90.source_data.ALL_REGIONS + regions))
    old_paths = list(data.default_native_paths())
    data.default_native_paths = lambda: old_paths + [NATIVE]
    data.STOP = pd.Timestamp("2026-11-01T00:00:00Z")
    data.ALL_REGIONS = tuple(dict.fromkeys(data.ALL_REGIONS + regions))
    hourly, _, audits, hourly_hashes = data.prepare_regions()
    capacity, capacity_hash = v90.capacities(2025)
    calibration = pd.read_csv(CALIBRATION_ROWS)
    features = pd.read_csv(SUPPORT_FEATURES)
    ef_vectors = factors_box()

    rows, availability = [], []
    for region in regions:
        radii, boundaries, fold_audits = fold_calibration(calibration, features, region)
        frame = hourly[region].copy()
        frame["time"] = pd.to_datetime(frame.time, utc=True)
        frame = frame.loc[frame.time.ge(MONTH) & frame.time.lt("2026-11-01")
                          & frame.valid.astype(bool)].copy()
        hours = len(frame)
        expected = pd.Period(MONTH, freq="M").days_in_month * 24
        coverage = hours / expected
        eligible = coverage >= 0.80
        frame["month"] = frame.time.dt.strftime("%Y-%m-01")
        caprow = capacity.loc[region] if region in capacity.index else pd.Series(dtype=float)
        fine_caps = {fuel: float(caprow.get(fuel, 0)) for fuel in
                     ("coal", "gas", "oil", "nuclear", "hydro", "wind", "solar", "other")}
        group_caps = {name: sum(fine_caps[fuel] for fuel in members)
                      for name, members in v90.GROUPS.items()}
        margins = {fuel: sum((F(str(float(x))) for x in frame[f"generation_{fuel}"]), F(0))
                   for fuel in v90.FUELS}
        energy_ok = hours > 0 and all(
            sum(margins[fuel] for fuel in members) <= F(hours) * F(str(group_caps[name]))
            for name, members in v90.GROUPS.items())
        cap_available = eligible and energy_ok
        availability.append({"region": region, "hours": hours, "expected_hours": expected,
                             "coverage": coverage, "eligible": eligible,
                             "capacity_feasible": bool(cap_available), "h1_coverage": h1_coverage[region]})
        if not eligible:
            print(f"availability-only {region}: {hours}/{expected}", flush=True)
            continue
        actual, anchor_base, monthly, capacity_sets = {}, {}, {}, {}
        total_margins = sum(margins.values(), F(0))
        for name, ef in ef_vectors.items():
            exact_ef = {fuel: F(str(float(ef[fuel]))) for fuel in v90.FUELS}
            mean_ci = (sum((margins[fuel] * exact_ef[fuel] for fuel in v90.FUELS), F(0)) / total_margins
                       if total_margins > 0 else F(0))
            exact_hourly = []
            for _, hr in frame.iterrows():
                gens = {fuel: F(str(float(hr[f"generation_{fuel}"]))) for fuel in v90.FUELS}
                exact_hourly.append((gens, sum(gens.values(), F(0))))
            for tau in TAUS:
                count = 0
                for gens, gen_total in exact_hourly:
                    if gen_total <= 0:
                        continue
                    carbon = sum((gens[fuel] * exact_ef[fuel] for fuel in v90.FUELS), F(0))
                    count += carbon > F(tau) * gen_total
                actual[(name, tau)] = count
                anchor_base[(name, tau)] = float(mean_ci > tau)
                old_ef = dict(v90.EF)
                try:
                    v90.EF.clear(); v90.EF.update(ef)
                    monthly[(name, tau)] = v90.v66_set(margins, tau, hours)
                    if cap_available:
                        capacity_sets[(name, tau)] = sharp_group_set(margins, group_caps, tau, hours)
                finally:
                    v90.EF.clear(); v90.EF.update(old_ef)
        for tau in TAUS:
            k_low = min(actual[(name, tau)] for name in ef_vectors)
            k_high = max(actual[(name, tau)] for name in ef_vectors)
            base_low = min(anchor_base[(name, tau)] for name in ef_vectors)
            base_high = max(anchor_base[(name, tau)] for name in ef_vectors)
            anchor_lower = max(0, math.ceil(max(0.0, base_low - radii[tau]) * hours - 1e-12))
            anchor_upper = min(hours, math.floor(min(1.0, base_high + radii[tau]) * hours + 1e-12))
            monthly_lower = min(min(monthly[(name, tau)]) for name in ef_vectors)
            monthly_upper = max(max(monthly[(name, tau)]) for name in ef_vectors)
            row = {"region": region, "month": MONTH, "tau": tau, "hours": hours,
                   "coverage": coverage, "capacity_available": bool(cap_available),
                   "h1_coverage": h1_coverage[region], "calibration_radius": radii[tau],
                   "calibration_regions": fold_audits[tau]["calibration_regions"],
                   "support_boundary": boundaries[tau], "support_regions": fold_audits[tau]["support_regions"],
                   "support_rank": fold_audits[tau]["support_rank"],
                   "actual_nominal_K": actual[("nominal", tau)],
                   "factor_box_K_lower": k_low, "factor_box_K_upper": k_high,
                   "anchor_lower": anchor_lower, "anchor_upper": anchor_upper,
                   "anchor_width": anchor_upper - anchor_lower,
                   "anchor_factor_box_covered": anchor_lower <= k_low and anchor_upper >= k_high,
                   "monthly_lower": monthly_lower, "monthly_upper": monthly_upper,
                   "monthly_width": monthly_upper - monthly_lower,
                   "monthly_factor_box_covered": monthly_lower <= k_low and monthly_upper >= k_high}
            if cap_available:
                cap_lower = min(min(capacity_sets[(name, tau)]) for name in ef_vectors)
                cap_upper = max(max(capacity_sets[(name, tau)]) for name in ef_vectors)
                inter_lower, inter_upper = max(anchor_lower, cap_lower), min(anchor_upper, cap_upper)
                if inter_lower > inter_upper:
                    inter_lower, inter_upper = cap_lower, cap_upper
                gap = max(0.0, cap_upper / hours - base_high)
                fallback = tau == 900 and gap > boundaries[tau]
                if tau == 600 or fallback:
                    frozen_lower, frozen_upper = cap_lower, cap_upper
                    policy = "capacity_only_tau600" if tau == 600 else "capacity_fallback_support_exceeded"
                else:
                    frozen_lower, frozen_upper = inter_lower, inter_upper
                    policy = "anchor_capacity_intersection"
                row.update({"capacity_lower": cap_lower, "capacity_upper": cap_upper,
                            "capacity_width": cap_upper - cap_lower,
                            "capacity_factor_box_covered": cap_lower <= k_low and cap_upper >= k_high,
                            "intersection_lower": inter_lower, "intersection_upper": inter_upper,
                            "intersection_width": inter_upper - inter_lower,
                            "intersection_factor_box_covered": inter_lower <= k_low and inter_upper >= k_high,
                            "support_gap": gap, "support_fallback": fallback,
                            "frozen_policy": policy, "frozen_lower": frozen_lower,
                            "frozen_upper": frozen_upper, "frozen_width": frozen_upper - frozen_lower,
                            "frozen_factor_box_covered": frozen_lower <= k_low and frozen_upper >= k_high})
            else:
                for key in ("capacity_lower", "capacity_upper", "capacity_width", "capacity_factor_box_covered",
                            "intersection_lower", "intersection_upper", "intersection_width",
                            "intersection_factor_box_covered", "support_gap", "support_fallback",
                            "frozen_lower", "frozen_upper", "frozen_width", "frozen_factor_box_covered"):
                    row[key] = None
                row["frozen_policy"] = "unavailable_capacity"
            for rate in RATES:
                for method, prefix in (("anchor", "anchor"), ("monthly", "monthly"),
                                       ("capacity", "capacity"), ("intersection", "intersection"),
                                       ("frozen", "frozen")):
                    lo, hi = row.get(f"{method}_lower"), row.get(f"{method}_upper")
                    if lo is None or hi is None:
                        for state in ("positive", "negative", "abstain"):
                            row[f"{prefix}_{state}_{rate:g}"] = None
                    else:
                        pos, neg, abstain, req = decide(int(lo), int(hi), hours, rate)
                        row[f"{prefix}_positive_{rate:g}"] = pos
                        row[f"{prefix}_negative_{rate:g}"] = neg
                        row[f"{prefix}_abstain_{rate:g}"] = abstain
                        row[f"required_{rate:g}"] = req
            rows.append(row)
        print(f"scored {region}", flush=True)

    detail = pd.DataFrame(rows); avail = pd.DataFrame(availability)
    OUT.mkdir(parents=True, exist_ok=False)
    detail.to_csv(OUT / "target_rows.csv", index=False)
    avail.to_csv(OUT / "availability.csv", index=False)
    frontier = []
    for tau, group in detail.groupby("tau"):
        eligible_rows = group[group.capacity_available]
        record = {"tau": int(tau), "scored_regions": int(group.region.nunique()),
                  "capacity_feasible_regions": int(eligible_rows.region.nunique()),
                  "nonzero_proxy_upper_regions": int((eligible_rows.factor_box_K_upper > 0).sum())}
        for method in ("anchor", "monthly", "capacity", "intersection", "frozen"):
            width = f"{method}_width"
            cover = f"{method}_factor_box_covered"
            subset = group if method in ("anchor", "monthly") else eligible_rows
            record[f"{method}_mean_width"] = float(subset[width].mean()) if len(subset) else None
            record[f"{method}_coverage"] = float(subset[cover].mean()) if len(subset) else None
        for rate in RATES:
            for method in ("anchor", "monthly", "capacity", "intersection", "frozen"):
                subset = group if method in ("anchor", "monthly") else eligible_rows
                for state in ("positive", "negative", "abstain"):
                    record[f"{method}_{state}_{rate:g}"] = float(subset[f"{method}_{state}_{rate:g}"].mean()) if len(subset) else None
        frontier.append(record)
    payload = {"status": "FROZEN_RULE_OCTOBER_MTD_TEMPORAL_CONFIRMATION",
               "regions": list(regions), "candidate_count": len(regions),
               "eligible_regions": int(avail.eligible.sum()),
               "capacity_feasible_regions": int(avail.capacity_feasible.sum()),
               "month_to_date_last_timestamp_utc": str(max(pd.to_datetime(hourly[r].time, utc=True).max() for r in regions)),
               "h1_completeness": h1_coverage, "frontier": frontier,
               "support_tail": SUPPORT_TAIL, "calibration_alpha": ALPHA,
               "native_sha256": sha(NATIVE), "capacity_sha256": capacity_hash,
               "calibration_rows_sha256": sha(CALIBRATION_ROWS),
               "support_features_sha256": sha(SUPPORT_FEATURES),
               "protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__)),
               "hourly_source_hashes": hourly_hashes, "source_audits": audits,
               "limitations": ["Partial October month-to-date nowcast/retrospective identification, not month-ahead forecast.",
                               "Method was developed after Q3 was opened; October target outcomes were not used in design.",
                               "Several regions overlap the historical source pool, though each is excluded from its own fold.",
                               "Fixed-factor U.S. proxy and capacity analogue do not validate China real CIF."]}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(pd.DataFrame(frontier).to_string(index=False))


if __name__ == "__main__":
    main()
