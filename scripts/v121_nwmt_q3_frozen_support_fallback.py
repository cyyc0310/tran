"""V121: frozen V120 support-gap fallback on the unscored NWMT Q3 cohort."""
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

REGION = "NWMT"
MONTHS = ("2026-07-01", "2026-08-01", "2026-09-01")
TAUS = (600, 900)
RATES = (0.01, 0.05, 0.10)
DELTA = 0.10
ALPHA = 0.10
SUPPORT_TAIL = 0.20
OUT = ROOT / "results/v121-nwmt-q3-frozen-support-fallback-r1"
PROTOCOL = ROOT / "docs/V121_NWMT_Q3_FROZEN_SUPPORT_FALLBACK_PROTOCOL.md"
CALIBRATION_ROWS = ROOT / "results/v111-factor-robust-monthly-anchor-r1/source_calibration_rows.csv"
SUPPORT_FEATURES = ROOT / "results/v120-support-gap-region-loo-cv-r2/source_outer_fold_features.csv"
NATIVE = ROOT / "data/native-eia-2026/EIA930_BALANCE_2026_Jul_Dec.csv"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def factors_box():
    nominal = dict(v90.EF)
    low = {fuel: ef * (1 - DELTA) if fuel in ("coal", "gas", "oil", "other") else ef
           for fuel, ef in nominal.items()}
    high = {fuel: ef * (1 + DELTA) if fuel in ("coal", "gas", "oil", "other") else ef
            for fuel, ef in nominal.items()}
    return {"low": low, "nominal": nominal, "high": high}


def radius(calibration: pd.DataFrame, tau: int) -> tuple[float, int, int]:
    train = calibration.loc[~calibration.region.eq(REGION) & calibration.tau.eq(tau)]
    scores = train.groupby("region").worst_case_absolute_error.max().sort_values().to_numpy()
    n = len(scores); rank = math.ceil((n + 1) * (1 - ALPHA))
    if rank > n:
        raise ValueError("non-finite anchor calibration radius")
    return float(scores[rank - 1]), n, rank


def support_boundary(features: pd.DataFrame, tau: int) -> tuple[float, int, int]:
    train = features.loc[(features.tau == tau) & (features.region != REGION)]
    region_max = train.groupby("region").support_gap.max().sort_values().to_numpy()
    n = len(region_max); rank = math.ceil((n + 1) * (1 - SUPPORT_TAIL))
    if rank > n:
        raise ValueError("non-finite support boundary")
    return float(region_max[rank - 1]), n, rank


def decide(lo: int, hi: int, hours: int, rate: float):
    required = int((F(str(rate)) * hours).__ceil__())
    return lo >= required, hi < required, not (lo >= required or hi < required), required


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {OUT}")
    v90.REGIONS = (REGION,)
    v90.source_data.ALL_REGIONS = tuple(dict.fromkeys(v90.source_data.ALL_REGIONS + (REGION,)))
    old_paths = list(data.default_native_paths())
    data.default_native_paths = lambda: old_paths + [NATIVE]
    data.STOP = pd.Timestamp("2026-10-01T00:00:00Z")
    data.ALL_REGIONS = tuple(dict.fromkeys(data.ALL_REGIONS + (REGION,)))
    hourly, _, audits, hourly_hashes = data.prepare_regions()
    capacity, capacity_hash = v90.capacities(2025)
    calibration = pd.read_csv(CALIBRATION_ROWS)
    features = pd.read_csv(SUPPORT_FEATURES)
    ef_vectors = factors_box()
    radii, radius_audits = {}, {}
    boundaries, boundary_audits = {}, {}
    for tau in TAUS:
        radii[tau], n, rank = radius(calibration, tau)
        radius_audits[tau] = {"regions": n, "rank": rank}
        boundaries[tau], n, rank = support_boundary(features, tau)
        boundary_audits[tau] = {"regions": n, "rank": rank}

    frame = hourly[REGION].copy()
    frame["time"] = pd.to_datetime(frame.time, utc=True)
    frame = frame.loc[frame.time.ge("2026-07-01") & frame.time.lt("2026-10-01")
                      & frame.valid.astype(bool)].copy()
    frame["month"] = frame.time.dt.strftime("%Y-%m-01")
    caprow = capacity.loc[REGION] if REGION in capacity.index else pd.Series(dtype=float)
    fine_caps = {fuel: float(caprow.get(fuel, 0)) for fuel in
                 ("coal", "gas", "oil", "nuclear", "hydro", "wind", "solar", "other")}
    group_caps = {name: sum(fine_caps[fuel] for fuel in members)
                  for name, members in v90.GROUPS.items()}

    rows, availability = [], []
    for month_key in MONTHS:
        group = frame.loc[frame.month.eq(month_key)]
        hours = len(group); expected = pd.Period(month_key, freq="M").days_in_month * 24
        coverage = hours / expected
        eligible = coverage >= 0.80
        margins = {fuel: sum((F(str(float(x))) for x in group[f"generation_{fuel}"]), F(0))
                   for fuel in v90.FUELS}
        energy_ok = hours > 0 and all(
            sum(margins[fuel] for fuel in members) <= F(hours) * F(str(group_caps[name]))
            for name, members in v90.GROUPS.items())
        cap_available = eligible and energy_ok
        availability.append({"region": REGION, "month": month_key, "hours": hours,
                             "expected_hours": expected, "coverage": coverage,
                             "eligible": eligible, "capacity_feasible": cap_available})
        if not eligible:
            continue
        observed_k, anchor_means, monthly_sets, capacity_sets = {}, {}, {}, {}
        total_margins = sum(margins.values(), F(0))
        for name, ef in ef_vectors.items():
            exact_ef = {fuel: F(str(float(ef[fuel]))) for fuel in v90.FUELS}
            mean_ci = (sum((margins[fuel] * exact_ef[fuel] for fuel in v90.FUELS), F(0)) / total_margins
                       if total_margins > 0 else F(0))
            for tau in TAUS:
                # Recompute exact K for each threshold, with strict inequality.
                observed_k[(name, tau)] = 0
                for _, hourly_row in group.iterrows():
                    gens = {fuel: F(str(float(hourly_row[f"generation_{fuel}"]))) for fuel in v90.FUELS}
                    gen_total = sum(gens.values(), F(0))
                    if gen_total <= 0:
                        continue
                    carbon = sum((gens[fuel] * exact_ef[fuel] for fuel in v90.FUELS), F(0))
                    observed_k[(name, tau)] += carbon > F(tau) * gen_total
                anchor_means[(name, tau)] = float(mean_ci > tau)
                old_ef = dict(v90.EF)
                try:
                    v90.EF.clear(); v90.EF.update(ef)
                    monthly_sets[(name, tau)] = v90.v66_set(margins, tau, hours)
                    if cap_available:
                        capacity_sets[(name, tau)] = sharp_group_set(margins, group_caps, tau, hours)
                finally:
                    v90.EF.clear(); v90.EF.update(old_ef)

        for tau in TAUS:
            factor_low = min(observed_k[(name, tau)] for name in ef_vectors)
            factor_high = max(observed_k[(name, tau)] for name in ef_vectors)
            base_low = min(anchor_means[(name, tau)] for name in ef_vectors)
            base_high = max(anchor_means[(name, tau)] for name in ef_vectors)
            anchor_lo = math.ceil(max(0.0, base_low - radii[tau]) * hours - 1e-12)
            anchor_hi = math.floor(min(1.0, base_high + radii[tau]) * hours + 1e-12)
            monthly_lo = min(min(monthly_sets[(name, tau)]) for name in ef_vectors)
            monthly_hi = max(max(monthly_sets[(name, tau)]) for name in ef_vectors)
            row = {"region": REGION, "month": month_key, "tau": tau, "hours": hours,
                   "coverage": coverage, "capacity_feasible": cap_available,
                   "calibration_radius": radii[tau], "calibration_regions": radius_audits[tau]["regions"],
                   "calibration_rank": radius_audits[tau]["rank"],
                   "actual_nominal_K": observed_k[("nominal", tau)],
                   "anchor_base_lower_fraction": base_low,
                   "anchor_base_upper_fraction": base_high,
                   "monthly_lower": monthly_lo, "monthly_upper": monthly_hi,
                   "monthly_width": monthly_hi - monthly_lo,
                   "factor_box_K_lower": factor_low, "factor_box_K_upper": factor_high,
                   "anchor_lower": anchor_lo, "anchor_upper": anchor_hi,
                   "anchor_width": anchor_hi - anchor_lo,
                   "anchor_factor_box_covered": anchor_lo <= factor_low and anchor_hi >= factor_high}
            if cap_available:
                cap_lo = min(min(capacity_sets[(name, tau)]) for name in ef_vectors)
                cap_hi = max(max(capacity_sets[(name, tau)]) for name in ef_vectors)
                inter_lo, inter_hi = max(anchor_lo, cap_lo), min(anchor_hi, cap_hi)
                if inter_lo > inter_hi:
                    inter_lo, inter_hi = cap_lo, cap_hi
                gap = max(0.0, cap_hi / hours - base_high)
                fallback = tau == 900 and gap > boundaries[tau]
                if tau == 600:
                    chosen_lo, chosen_hi, policy = cap_lo, cap_hi, "capacity_only_frozen_tau600"
                elif fallback:
                    chosen_lo, chosen_hi, policy = cap_lo, cap_hi, "capacity_fallback_support_exceeded"
                else:
                    chosen_lo, chosen_hi, policy = inter_lo, inter_hi, "anchor_capacity_intersection"
                row.update({"capacity_lower": cap_lo, "capacity_upper": cap_hi,
                            "capacity_width": cap_hi - cap_lo,
                            "capacity_factor_box_covered": cap_lo <= factor_low and cap_hi >= factor_high,
                            "intersection_lower": inter_lo, "intersection_upper": inter_hi,
                            "intersection_width": inter_hi - inter_lo,
                            "intersection_factor_box_covered": inter_lo <= factor_low and inter_hi >= factor_high,
                            "support_gap": gap, "support_boundary": boundaries[tau],
                            "support_fallback": fallback, "frozen_policy": policy,
                            "frozen_lower": chosen_lo, "frozen_upper": chosen_hi,
                            "frozen_width": chosen_hi - chosen_lo,
                            "frozen_factor_box_covered": chosen_lo <= factor_low and chosen_hi >= factor_high})
            else:
                row.update({"capacity_lower": None, "capacity_upper": None, "capacity_width": None,
                            "capacity_factor_box_covered": None, "intersection_lower": None,
                            "intersection_upper": None, "intersection_width": None,
                            "intersection_factor_box_covered": None, "support_gap": None,
                            "support_boundary": boundaries[tau], "support_fallback": None,
                            "frozen_policy": "unavailable_capacity", "frozen_lower": None,
                            "frozen_upper": None, "frozen_width": None,
                            "frozen_factor_box_covered": None})
            for rate in RATES:
                for method in ("anchor", "monthly", "capacity", "intersection", "frozen"):
                    lo, hi = ((anchor_lo, anchor_hi) if method == "anchor" else
                              (monthly_lo, monthly_hi) if method == "monthly" else
                              (row.get("capacity_lower"), row.get("capacity_upper")) if method == "capacity" else
                              (row.get("intersection_lower"), row.get("intersection_upper")) if method == "intersection" else
                              (row.get("frozen_lower"), row.get("frozen_upper")))
                    if lo is None or hi is None:
                        row[f"{method}_positive_{rate:g}"] = None
                        row[f"{method}_negative_{rate:g}"] = None
                        row[f"{method}_abstain_{rate:g}"] = None
                    else:
                        p, n, a, req = decide(int(lo), int(hi), hours, rate)
                        row[f"{method}_positive_{rate:g}"] = p
                        row[f"{method}_negative_{rate:g}"] = n
                        row[f"{method}_abstain_{rate:g}"] = a
                        row[f"required_{rate:g}"] = req
            rows.append(row)

    detail = pd.DataFrame(rows); avail = pd.DataFrame(availability)
    OUT.mkdir(parents=True, exist_ok=False)
    detail.to_csv(OUT / "target_rows.csv", index=False)
    avail.to_csv(OUT / "availability.csv", index=False)
    payload = {"status": "FROZEN_METHOD_TEMPORAL_TARGET_CONFIRMATION_NWMT_Q3",
               "region": REGION, "months": list(MONTHS), "candidate_slots": len(MONTHS),
               "eligible_slots": int(avail.eligible.sum()), "capacity_feasible_slots": int(avail.capacity_feasible.sum()),
               "h1_complete_proxy_coverage": 0.845304,
               "support_tail": SUPPORT_TAIL,
               "support_boundaries": boundaries, "support_boundary_audits": boundary_audits,
               "anchor_radii": radii, "anchor_radius_audits": radius_audits,
               "rows": detail.to_dict("records"),
               "native_file_sha256": sha(NATIVE), "capacity_source_sha256": capacity_hash,
               "calibration_rows_sha256": sha(CALIBRATION_ROWS),
               "support_features_sha256": sha(SUPPORT_FEATURES),
               "hourly_source_hashes": hourly_hashes, "source_audits": audits,
               "protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__)),
               "limitations": ["NWMT historical records were in the source pool but are excluded from this target fold.",
                               "One region and three months; not a wholly unseen BA identity.",
                               "Retrospective use of contemporaneous monthly margins, not a month-ahead forecast.",
                               "Fixed-factor U.S. proxy, not Chinese real CIF."]}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(detail[["month", "tau", "actual_nominal_K", "factor_box_K_lower", "factor_box_K_upper",
                  "capacity_lower", "capacity_upper", "intersection_lower", "intersection_upper",
                  "support_gap", "support_boundary", "support_fallback", "frozen_policy",
                  "frozen_lower", "frozen_upper", "frozen_factor_box_covered"]].to_string(index=False))


if __name__ == "__main__":
    main()
