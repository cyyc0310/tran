"""V113: frozen future-time confirmation of factor-robust tail intervals."""
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

REGIONS = ("CPLE", "FPC", "IPCO", "PNM", "SOCO")
MONTHS = ("2026-07-01", "2026-08-01", "2026-09-01")
TAUS = (600, 900)
RATES = (0.01, 0.05, 0.10)
DELTA = 0.10
OUT = ROOT / "results/v113-2026q3-future-time-confirmation-r1"
PROTOCOL = ROOT / "docs/V113_2026Q3_FUTURE_TIME_CONFIRMATION_PROTOCOL.md"
NATIVE = ROOT / "data/native-eia-2026/EIA930_BALANCE_2026_Jul_Dec.csv"
V112_SUMMARY = ROOT / "results/v112-worst-region-robust-anchor-r1/summary.json"
V112_PROTOCOL = ROOT / "docs/V112_WORST_REGION_ROBUST_ANCHOR_PROTOCOL.md"
CAPACITY_FILE = ROOT / "data/research-v87/eia8602025.zip"


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


def factors_box():
    nominal = dict(v90.EF)
    low = {fuel: ef * (1 - DELTA) if fuel in ("coal", "gas", "oil", "other") else ef
           for fuel, ef in nominal.items()}
    high = {fuel: ef * (1 + DELTA) if fuel in ("coal", "gas", "oil", "other") else ef
            for fuel, ef in nominal.items()}
    return {"low": low, "nominal": nominal, "high": high}


def interval_decision(lower: int, upper: int, hours: int, rate: float):
    required = int((F(str(rate)) * hours).__ceil__())
    positive = lower >= required
    negative = upper < required
    return required, positive, negative, not (positive or negative)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {OUT}")
    if not NATIVE.exists():
        raise FileNotFoundError(NATIVE)
    previous_paths = list(data.default_native_paths())
    data.default_native_paths = lambda: previous_paths + [NATIVE]
    data.STOP = pd.Timestamp("2026-10-01T00:00:00Z")
    data.ALL_REGIONS = tuple(dict.fromkeys(data.ALL_REGIONS + REGIONS))
    v90.REGIONS = REGIONS
    v90.source_data.ALL_REGIONS = data.ALL_REGIONS
    hourly, _, audits, hourly_hashes = data.prepare_regions()
    capacities, capacity_sha = v90.capacities(2025)
    factor_vectors = factors_box()
    v112 = json.loads(V112_SUMMARY.read_text())
    radii = {int(tau): float(value) for tau, value in
             v112["max_region_q90_radius_by_tau"].items()}
    if sorted(v112["target_regions"]) != sorted(REGIONS):
        raise ValueError("V112 calibration source roster does not match frozen target roster")

    rows, availability = [], []
    for region in REGIONS:
        frame = hourly[region].copy()
        times = pd.to_datetime(frame.time, utc=True)
        frame = frame.loc[times.ge("2026-07-01") & times.lt("2026-10-01")
                          & frame.valid.astype(bool)].copy()
        frame["time"] = pd.to_datetime(frame.time, utc=True)
        frame["month"] = frame.time.dt.strftime("%Y-%m-01")
        caprow = capacities.loc[region] if region in capacities.index else pd.Series(dtype=float)
        fine_caps = {fuel: float(caprow.get(fuel, 0)) for fuel in
                     ("coal", "gas", "oil", "nuclear", "hydro", "wind", "solar", "other")}
        group_caps = {group: sum(fine_caps[fuel] for fuel in fuels)
                      for group, fuels in v90.GROUPS.items()}

        for month_key in MONTHS:
            group = frame.loc[frame.month.eq(month_key)]
            hours = len(group)
            month = pd.Timestamp(month_key, tz="UTC")
            expected = month.days_in_month * 24
            coverage = hours / expected
            margins = {fuel: sum((F(str(float(x))) for x in group[f"generation_{fuel}"]), F(0))
                       for fuel in v90.FUELS}
            energy_ok = hours > 0 and all(
                sum(float(margins[fuel]) for fuel in members) <= hours * group_caps[g] + 1e-6
                for g, members in v90.GROUPS.items())
            eligible = coverage >= 0.80
            capacity_available = eligible and energy_ok
            availability.append({"region": region, "month": month_key, "valid_hours": hours,
                                 "expected_hours": expected, "coverage": coverage,
                                 "coverage_eligible": bool(eligible),
                                 "group_energy_feasible": bool(energy_ok),
                                 "capacity_method_available": bool(capacity_available),
                                 "capacity_vintage": 2025})
            if not eligible:
                continue

            for tau in TAUS:
                # Recalculate K counts for this threshold; keep each factor vector fixed.
                observed_k = {}
                mean_anchor = {}
                generation = group[[f"generation_{fuel}" for fuel in v90.FUELS]].to_numpy(float)
                total = generation.sum(axis=1)
                fuel_margins = generation.sum(axis=0)
                positive = total > 0
                for name, ef in factor_vectors.items():
                    emissions = generation @ pd.Series(ef).reindex(v90.FUELS).to_numpy(float)
                    observed_k[name] = int(sum(F(str(float(c))) > F(tau) * F(str(float(g)))
                                               for c, g in zip(emissions[positive], total[positive])))
                    mean_ci = float(sum(fuel_margins[i] * ef[fuel] for i, fuel in enumerate(v90.FUELS))
                                    / fuel_margins.sum()) if fuel_margins.sum() > 0 else 0.0
                    mean_anchor[name] = float(mean_ci > tau)

                anchor_base_lower = min(mean_anchor.values())
                anchor_base_upper = max(mean_anchor.values())
                radius = radii[tau]
                anchor_lower_fraction = max(0.0, anchor_base_lower - radius)
                anchor_upper_fraction = min(1.0, anchor_base_upper + radius)
                anchor_lower = max(0, min(hours, math.ceil(anchor_lower_fraction * hours - 1e-12)))
                anchor_upper = max(0, min(hours, math.floor(anchor_upper_fraction * hours + 1e-12)))
                factor_k_lower, factor_k_upper = min(observed_k.values()), max(observed_k.values())
                anchor_covered = anchor_lower <= factor_k_lower and anchor_upper >= factor_k_upper

                # Monthly-only robust K hull under the three factor vertices.
                monthly_sets = {}
                for name, ef in factor_vectors.items():
                    old_ef = dict(v90.EF)
                    try:
                        v90.EF.clear(); v90.EF.update(ef)
                        monthly_sets[name] = v90.v66_set(margins, tau, hours)
                    finally:
                        v90.EF.clear(); v90.EF.update(old_ef)
                monthly_lower = min(min(s) for s in monthly_sets.values())
                monthly_upper = max(max(s) for s in monthly_sets.values())
                monthly_covered = monthly_lower <= factor_k_lower and monthly_upper >= factor_k_upper

                capacity_lower = capacity_upper = None
                capacity_covered = None
                if capacity_available:
                    capacity_sets = {}
                    for name, ef in factor_vectors.items():
                        old_ef = dict(v90.EF)
                        try:
                            v90.EF.clear(); v90.EF.update(ef)
                            capacity_sets[name] = sharp_group_set(margins, group_caps, tau, hours)
                        finally:
                            v90.EF.clear(); v90.EF.update(old_ef)
                    capacity_lower = min(min(s) for s in capacity_sets.values())
                    capacity_upper = max(max(s) for s in capacity_sets.values())
                    capacity_covered = capacity_lower <= factor_k_lower and capacity_upper >= factor_k_upper

                actual_nominal = observed_k["nominal"]
                row = {"region": region, "month": month_key, "tau": tau, "hours": hours,
                       "coverage": coverage, "capacity_available": bool(capacity_available),
                       "actual_nominal_K": actual_nominal, "factor_box_K_lower": factor_k_lower,
                       "factor_box_K_upper": factor_k_upper,
                       "anchor_robust_lower": anchor_lower, "anchor_robust_upper": anchor_upper,
                       "anchor_robust_width": anchor_upper - anchor_lower,
                       "anchor_factor_box_covered": bool(anchor_covered),
                       "monthly_robust_lower": monthly_lower, "monthly_robust_upper": monthly_upper,
                       "monthly_robust_width": monthly_upper - monthly_lower,
                       "monthly_factor_box_covered": bool(monthly_covered),
                       "capacity_robust_lower": capacity_lower, "capacity_robust_upper": capacity_upper,
                       "capacity_robust_width": None if capacity_lower is None else capacity_upper - capacity_lower,
                       "capacity_factor_box_covered": capacity_covered}
                for rate in RATES:
                    req, pos, neg, abstain = interval_decision(anchor_lower, anchor_upper, hours, rate)
                    row[f"required_{rate:g}"] = req
                    row[f"anchor_positive_{rate:g}"] = pos
                    row[f"anchor_negative_{rate:g}"] = neg
                    row[f"anchor_abstain_{rate:g}"] = abstain
                    _, pos, neg, abstain = interval_decision(monthly_lower, monthly_upper, hours, rate)
                    row[f"monthly_positive_{rate:g}"] = pos
                    row[f"monthly_negative_{rate:g}"] = neg
                    row[f"monthly_abstain_{rate:g}"] = abstain
                    if capacity_available:
                        _, pos, neg, abstain = interval_decision(capacity_lower, capacity_upper, hours, rate)
                        row[f"capacity_positive_{rate:g}"] = pos
                        row[f"capacity_negative_{rate:g}"] = neg
                        row[f"capacity_abstain_{rate:g}"] = abstain
                    else:
                        row[f"capacity_positive_{rate:g}"] = None
                        row[f"capacity_negative_{rate:g}"] = None
                        row[f"capacity_abstain_{rate:g}"] = None
                rows.append(row)
        print(f"processed target region {region}", flush=True)

    detail = pd.DataFrame(rows)
    avail = pd.DataFrame(availability)
    if detail.empty:
        raise ValueError("No eligible 2026Q3 region-months")
    OUT.mkdir(parents=True, exist_ok=False)
    detail.to_csv(OUT / "target_rows.csv", index=False)
    avail.to_csv(OUT / "availability.csv", index=False)
    macro_rows = []
    for tau, group in detail.groupby("tau"):
        macro = {"tau": int(tau), "eligible_region_months": int(group.shape[0]),
                 "regions": int(group.region.nunique()), "anchor_width": float(group.anchor_robust_width.mean()),
                 "anchor_factor_box_coverage": float(group.anchor_factor_box_covered.mean()),
                 "monthly_width": float(group.monthly_robust_width.mean()),
                 "monthly_factor_box_coverage": float(group.monthly_factor_box_covered.mean()),
                 "capacity_region_months": int(group.capacity_available.sum()),
                 "capacity_width": float(group.loc[group.capacity_available, "capacity_robust_width"].mean()),
                 "capacity_factor_box_coverage": float(group.loc[group.capacity_available,
                                                                  "capacity_factor_box_covered"].mean())}
        for rate in RATES:
            for method in ("anchor", "monthly", "capacity"):
                for state in ("positive", "negative", "abstain"):
                    column = f"{method}_{state}_{rate:g}"
                    macro[f"{column}_rate"] = float(group.loc[group[column].notna(), column].mean())
        macro_rows.append(macro)
    macro = pd.DataFrame(macro_rows)
    macro.to_csv(OUT / "future_time_frontier.csv", index=False)
    payload = {"status": "FROZEN_2026Q3_FUTURE_TIME_PROXY_CONFIRMATION",
               "regions": list(REGIONS), "months": list(MONTHS),
               "candidate_slots": len(REGIONS) * len(MONTHS),
               "available_slots_by_coverage": int(avail.coverage_eligible.sum()),
               "capacity_available_slots": int(avail.capacity_method_available.sum()),
               "unavailable_slots": int((~avail.coverage_eligible).sum()),
               "rows": len(detail), "frontier": macro.to_dict("records"),
               "factor_vectors": factor_vectors, "frozen_anchor_radii": radii,
               "source_file_url": "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2026_Jul_Dec.csv",
               "source_file_last_modified_utc": "2026-10-27 14:42:55 GMT",
               "source_file_sha256": sha(NATIVE), "source_file_bytes": NATIVE.stat().st_size,
               "capacity_source_sha256": capacity_sha, "hourly_source_sha256": hourly_hashes,
               "source_audits": audits, "v112_summary_sha256": sha(V112_SUMMARY),
               "v112_protocol_sha256": sha(V112_PROTOCOL), "protocol_sha256": sha(PROTOCOL),
               "script_sha256": sha(Path(__file__)),
               "limitations": ["Only three future months and five previously seen regions.",
                               "Target-month totals make evaluation retrospective nowcasting, not month-ahead forecasting.",
                               "Fixed-factor EIA proxy labels and U.S. EIA capacity are not Chinese real CIF data.",
                               "Worst-region anchor calibration is empirical, not a formal distribution-free guarantee."]}
    (OUT / "summary.json").write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False,
                                                   allow_nan=False) + "\n")
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
