"""V118: broad cohort region-LOO robustness diagnostic."""
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

REGIONS = ()
MONTHS = ("2026-07-01", "2026-08-01", "2026-09-01")
TAUS = (600, 900)
RATES = (0.01, 0.05, 0.10)
DELTA = 0.10
OUT = ROOT / "results/v118-region-loo-broad-cohort-diagnostic-r1"
PROTOCOL = ROOT / "docs/V118_REGION_LOO_BROAD_COHORT_PROTOCOL.md"
NATIVE = ROOT / "data/native-eia-2026/EIA930_BALANCE_2026_Jul_Dec.csv"
V112_SUMMARY = ROOT / "results/v112-worst-region-robust-anchor-r1/summary.json"
CALIBRATION_ROWS = ROOT / "results/v111-factor-robust-monthly-anchor-r1/source_calibration_rows.csv"
V112_PROTOCOL = ROOT / "docs/V112_WORST_REGION_ROBUST_ANCHOR_PROTOCOL.md"
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


def factors_box():
    nominal = dict(v90.EF)
    low = {fuel: ef * (1 - DELTA) if fuel in ("coal", "gas", "oil", "other") else ef
           for fuel, ef in nominal.items()}
    high = {fuel: ef * (1 + DELTA) if fuel in ("coal", "gas", "oil", "other") else ef
            for fuel, ef in nominal.items()}
    return {"low": low, "nominal": nominal, "high": high}


def decide(lower, upper, hours, rate):
    req = int((F(str(rate)) * hours).__ceil__())
    positive, negative = lower >= req, upper < req
    return req, positive, negative, not (positive or negative)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {OUT}")
    census = json.loads(CENSUS.read_text())
    global REGIONS
    REGIONS = tuple(sorted(region for region, periods in census["periods"].items()
                           if float(periods["2026"]["complete_proxy_coverage"]) >= 0.90))
    h1_coverage = {region: float(census["periods"][region]["2026"]["complete_proxy_coverage"])
                   for region in REGIONS}
    previous_paths = list(data.default_native_paths())
    data.default_native_paths = lambda: previous_paths + [NATIVE]
    data.STOP = pd.Timestamp("2026-10-01T00:00:00Z")
    data.ALL_REGIONS = tuple(dict.fromkeys(data.ALL_REGIONS + REGIONS))
    v90.REGIONS = REGIONS
    v90.source_data.ALL_REGIONS = data.ALL_REGIONS
    hourly, _, audits, hourly_hashes = data.prepare_regions()
    capacity, capacity_hash = v90.capacities(2025)
    ef_vectors = factors_box()
    anchor_summary = json.loads(V112_SUMMARY.read_text())
    calibration = pd.read_csv(CALIBRATION_ROWS)
    alpha = 0.10
    calibration_regions = sorted(calibration.region.unique())
    calibration_radii = {}
    for target_region in REGIONS:
        heldout = calibration.loc[~calibration.region.eq(target_region)]
        per_tau = {}
        for tau, group in heldout.groupby("tau"):
            scores = group.groupby("region").worst_case_absolute_error.max().sort_values().to_numpy()
            n = len(scores)
            rank = math.ceil((n + 1) * (1 - alpha))
            if rank > n:
                raise ValueError(f"no finite conformal radius for {target_region}, tau={tau}")
            per_tau[int(tau)] = float(scores[rank - 1])
        calibration_radii[target_region] = per_tau
        expected_n = 24 if target_region in calibration_regions else 25
        if len(heldout.region.unique()) != expected_n:
            raise ValueError(f"unexpected LOO calibration size for {target_region}")
    if sorted(anchor_summary["target_regions"]) != sorted(("CPLE", "FPC", "IPCO", "PNM", "SOCO")):
        raise ValueError("unexpected V112 reference calibration setup")

    rows, availability = [], []
    for region in REGIONS:
        frame = hourly[region].copy()
        times = pd.to_datetime(frame.time, utc=True)
        frame = frame.loc[times.ge("2026-07-01") & times.lt("2026-10-01")
                          & frame.valid.astype(bool)].copy()
        frame["time"] = pd.to_datetime(frame.time, utc=True)
        frame["month"] = frame.time.dt.strftime("%Y-%m-01")
        caprow = capacity.loc[region] if region in capacity.index else pd.Series(dtype=float)
        detailed_caps = {fuel: float(caprow.get(fuel, 0)) for fuel in
                         ("coal", "gas", "oil", "nuclear", "hydro", "wind", "solar", "other")}
        group_caps = {name: sum(detailed_caps[fuel] for fuel in members)
                      for name, members in v90.GROUPS.items()}

        for month_key in MONTHS:
            group = frame.loc[frame.month.eq(month_key)]
            hours = len(group)
            month = pd.Timestamp(month_key, tz="UTC")
            expected = month.days_in_month * 24
            coverage = hours / expected
            margins = {fuel: sum((F(str(float(x))) for x in group[f"generation_{fuel}"]), F(0))
                       for fuel in v90.FUELS}
            energy_ok = hours > 0 and all(
                sum(float(margins[fuel]) for fuel in members) <= hours * group_caps[name] + 1e-6
                for name, members in v90.GROUPS.items())
            eligible = coverage >= 0.80
            cap_available = eligible and energy_ok
            availability.append({"region": region, "month": month_key, "valid_hours": hours,
                                 "expected_hours": expected, "coverage": coverage,
                                 "coverage_eligible": bool(eligible), "group_energy_feasible": bool(energy_ok),
                                 "capacity_available": bool(cap_available)})
            if not eligible:
                continue
            generation = group[[f"generation_{fuel}" for fuel in v90.FUELS]].to_numpy(float)
            total = generation.sum(axis=1)
            fuel_margins = generation.sum(axis=0)

            for tau in TAUS:
                observed_k, anchor_means, monthly_sets = {}, {}, {}
                capacity_sets = {}
                for name, ef in ef_vectors.items():
                    ef_array = pd.Series(ef).reindex(v90.FUELS).to_numpy(float)
                    emissions = generation @ ef_array
                    valid = total > 0
                    observed_k[name] = int(sum(F(str(float(c))) > F(tau) * F(str(float(g)))
                                               for c, g in zip(emissions[valid], total[valid])))
                    mean_ci = float(fuel_margins @ ef_array / fuel_margins.sum()) if fuel_margins.sum() > 0 else 0.0
                    anchor_means[name] = float(mean_ci > tau)
                    old_ef = dict(v90.EF)
                    try:
                        v90.EF.clear(); v90.EF.update(ef)
                        monthly_sets[name] = v90.v66_set(margins, tau, hours)
                        if cap_available:
                            capacity_sets[name] = sharp_group_set(margins, group_caps, tau, hours)
                    finally:
                        v90.EF.clear(); v90.EF.update(old_ef)

                factor_lower, factor_upper = min(observed_k.values()), max(observed_k.values())
                anchor_base_lower, anchor_base_upper = min(anchor_means.values()), max(anchor_means.values())
                anchor_radius = calibration_radii[region][tau]
                anchor_lower_fraction = max(0.0, anchor_base_lower - anchor_radius)
                anchor_upper_fraction = min(1.0, anchor_base_upper + anchor_radius)
                anchor_lower = max(0, min(hours, math.ceil(anchor_lower_fraction * hours - 1e-12)))
                anchor_upper = max(0, min(hours, math.floor(anchor_upper_fraction * hours + 1e-12)))
                monthly_lower = min(min(values) for values in monthly_sets.values())
                monthly_upper = max(max(values) for values in monthly_sets.values())
                monthly_covered = monthly_lower <= factor_lower and monthly_upper >= factor_upper
                anchor_covered = anchor_lower <= factor_lower and anchor_upper >= factor_upper

                if cap_available:
                    capacity_lower = min(min(values) for values in capacity_sets.values())
                    capacity_upper = max(max(values) for values in capacity_sets.values())
                    capacity_covered = capacity_lower <= factor_lower and capacity_upper >= factor_upper
                    intersection_lower = max(anchor_lower, capacity_lower)
                    intersection_upper = min(anchor_upper, capacity_upper)
                    intersection_nonempty = intersection_lower <= intersection_upper
                    intersection_covered = (intersection_nonempty and intersection_lower <= factor_lower
                                            and intersection_upper >= factor_upper)
                else:
                    capacity_lower = capacity_upper = intersection_lower = intersection_upper = None
                    capacity_covered = intersection_covered = intersection_nonempty = None

                row = {"region": region, "month": month_key, "tau": tau, "hours": hours,
                       "coverage": coverage, "capacity_available": bool(cap_available),
                       "group_conformal_radius": calibration_radii[region][tau],
                       "calibration_region_count": 24 if region in calibration_regions else 25,
                       "actual_nominal_K": observed_k["nominal"],
                       "factor_box_K_lower": factor_lower, "factor_box_K_upper": factor_upper,
                       "anchor_lower": anchor_lower, "anchor_upper": anchor_upper,
                       "anchor_width": anchor_upper - anchor_lower,
                       "anchor_factor_box_covered": bool(anchor_covered),
                       "monthly_lower": monthly_lower, "monthly_upper": monthly_upper,
                       "monthly_width": monthly_upper - monthly_lower,
                       "monthly_factor_box_covered": bool(monthly_covered),
                       "capacity_lower": capacity_lower, "capacity_upper": capacity_upper,
                       "capacity_width": None if capacity_lower is None else capacity_upper - capacity_lower,
                       "capacity_factor_box_covered": capacity_covered,
                       "intersection_lower": intersection_lower, "intersection_upper": intersection_upper,
                       "intersection_width": None if not intersection_nonempty else intersection_upper - intersection_lower,
                       "intersection_nonempty": intersection_nonempty,
                       "intersection_factor_box_covered": intersection_covered}
                for rate in RATES:
                    req, p, n, a = decide(anchor_lower, anchor_upper, hours, rate)
                    row[f"required_{rate:g}"] = req
                    row[f"anchor_positive_{rate:g}"] = p; row[f"anchor_negative_{rate:g}"] = n
                    row[f"anchor_abstain_{rate:g}"] = a
                    _, p, n, a = decide(monthly_lower, monthly_upper, hours, rate)
                    row[f"monthly_positive_{rate:g}"] = p; row[f"monthly_negative_{rate:g}"] = n
                    row[f"monthly_abstain_{rate:g}"] = a
                    if cap_available and intersection_nonempty:
                        _, p, n, a = decide(capacity_lower, capacity_upper, hours, rate)
                        row[f"capacity_positive_{rate:g}"] = p; row[f"capacity_negative_{rate:g}"] = n
                        row[f"capacity_abstain_{rate:g}"] = a
                        _, p, n, a = decide(intersection_lower, intersection_upper, hours, rate)
                        row[f"intersection_positive_{rate:g}"] = p
                        row[f"intersection_negative_{rate:g}"] = n
                        row[f"intersection_abstain_{rate:g}"] = a
                    else:
                        for method in ("capacity", "intersection"):
                            row[f"{method}_positive_{rate:g}"] = None
                            row[f"{method}_negative_{rate:g}"] = None
                            row[f"{method}_abstain_{rate:g}"] = None
                rows.append(row)
        print(f"processed region {region}", flush=True)

    detail = pd.DataFrame(rows)
    avail = pd.DataFrame(availability)
    OUT.mkdir(parents=True, exist_ok=False)
    detail.to_csv(OUT / "target_rows.csv", index=False)
    avail.to_csv(OUT / "availability.csv", index=False)
    metrics = ["anchor_width", "monthly_width", "capacity_width", "intersection_width",
               "anchor_factor_box_covered", "monthly_factor_box_covered", "capacity_factor_box_covered",
               "intersection_factor_box_covered"]
    for rate in RATES:
        metrics.extend(f"{method}_{state}_{rate:g}" for method in ("anchor", "monthly", "capacity", "intersection")
                       for state in ("positive", "negative", "abstain"))
    region = detail.groupby(["region", "tau"])[metrics].mean().reset_index()
    macro = region.groupby("tau")[metrics].mean().reset_index()
    region.to_csv(OUT / "per_region_frontier.csv", index=False)
    macro.to_csv(OUT / "region_equal_frontier.csv", index=False)
    payload = {"status": "EXPLORATORY_BROAD_COHORT_REGION_LOO_ROBUSTNESS_DIAGNOSTIC",
               "regions": list(REGIONS), "months": list(MONTHS), "candidate_slots": len(REGIONS) * len(MONTHS),
               "h1_quality_coverage_selection": h1_coverage,
               "scored_region_months": int(detail[["region", "month"]].drop_duplicates().shape[0]),
               "scored_threshold_rows": len(detail),
               "coverage_ineligible_slots": int((~avail.coverage_eligible).sum()),
               "capacity_available_slots": int(avail.capacity_available.sum()),
               "intersection_nonempty_all": bool(detail.loc[detail.capacity_available,
                                                           "intersection_nonempty"].all()),
               "factor_box_coverage": {method: bool(detail.loc[detail[f"{method}_factor_box_covered"].notna(),
                                                                  f"{method}_factor_box_covered"].all())
                                       for method in ("anchor", "monthly", "capacity", "intersection")},
               "frontier": macro.to_dict("records"),
               "group_conformal_radii_by_target_region": calibration_radii,
               "calibration_source_regions": calibration_regions,
               "group_conformal_alpha": alpha,
               "calibration_rows_sha256": sha(CALIBRATION_ROWS),
               "v112_reference_radii": {int(t): float(v) for t, v in anchor_summary["max_region_q90_radius_by_tau"].items()},
               "source_file_url": "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2026_Jul_Dec.csv",
               "source_file_last_modified_utc": "2026-10-27 14:42:55 GMT",
               "source_file_sha256": sha(NATIVE), "hourly_source_sha256": hourly_hashes,
               "capacity_source_sha256": capacity_hash, "source_audits": audits,
               "selection_census_sha256": sha(CENSUS), "v112_summary_sha256": sha(V112_SUMMARY),
               "v112_protocol_sha256": sha(V112_PROTOCOL), "protocol_sha256": sha(PROTOCOL),
               "script_sha256": sha(Path(__file__)),
               "limitations": ["Broad-cohort post-hoc diagnostic; V113/V115 exposed part of the Q3 region labels and V116 was designed after V115.",
                               "Split-conformal marginal validity depends on exchangeable region-level calibration and target units after the quality-based cohort selection.",
                               "Single observed future quarter; monthly rows within a region are dependent.",
                               "Prior target groups and other regions share the same quarter; this is a broad diagnostic, not blind confirmation.",
                               "Empirical anchor is not a hard physical constraint or formal coverage guarantee.",
                               "Fixed-factor EIA proxy and U.S. capacity do not validate China real CIF or month-ahead forecasting."]}
    (OUT / "summary.json").write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False,
                                                   allow_nan=False) + "\n")
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
