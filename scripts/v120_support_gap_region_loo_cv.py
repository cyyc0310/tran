"""V120: outer region-LOO CV of a source-support-gap fallback rule."""
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

OUT = ROOT / "results/v120-support-gap-region-loo-cv-r2"
PROTOCOL = ROOT / "docs/V120_SUPPORT_GAP_REGION_LOO_CV_PROTOCOL.md"
CALIBRATION_ROWS = ROOT / "results/v111-factor-robust-monthly-anchor-r1/source_calibration_rows.csv"
NATIVE = ROOT / "data/native-eia-2026/EIA930_BALANCE_2026_Jul_Dec.csv"
TAUS = (600, 900)
SUPPORT_TAILS = (0.05, 0.10, 0.20, 0.30)
CAL_ALPHA = 0.10
DELTA = 0.10


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ef_box():
    nominal = dict(v90.EF)
    low = {fuel: value * (1 - DELTA) if fuel in ("coal", "gas", "oil", "other") else value
           for fuel, value in nominal.items()}
    high = {fuel: value * (1 + DELTA) if fuel in ("coal", "gas", "oil", "other") else value
            for fuel, value in nominal.items()}
    return {"low": low, "nominal": nominal, "high": high}


def radius_for(calibration: pd.DataFrame, excluded_region: str, tau: int) -> tuple[float, int, int]:
    train = calibration.loc[~calibration.region.eq(excluded_region) & calibration.tau.eq(tau)]
    scores = train.groupby("region").worst_case_absolute_error.max().sort_values().to_numpy()
    n = len(scores)
    rank = math.ceil((n + 1) * (1 - CAL_ALPHA))
    if rank > n:
        raise ValueError("non-finite anchor radius")
    return float(scores[rank - 1]), n, rank


def endpoints(anchor_low: float, anchor_high: float, radius: float, hours: int) -> tuple[int, int]:
    lower_fraction = max(0.0, anchor_low - radius)
    upper_fraction = min(1.0, anchor_high + radius)
    lo = max(0, min(hours, math.ceil(lower_fraction * hours - 1e-12)))
    hi = max(0, min(hours, math.floor(upper_fraction * hours + 1e-12)))
    return lo, hi


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {OUT}")
    calibration = pd.read_csv(CALIBRATION_ROWS)
    calibration["month"] = pd.to_datetime(calibration.month, utc=True)
    regions = tuple(sorted(calibration.region.unique()))
    v90.REGIONS = regions
    v90.source_data.ALL_REGIONS = tuple(dict.fromkeys(v90.source_data.ALL_REGIONS + regions))
    previous_paths = list(data.default_native_paths())
    data.default_native_paths = lambda: previous_paths + [NATIVE]
    data.STOP = pd.Timestamp("2026-01-01T00:00:00Z")
    data.ALL_REGIONS = tuple(dict.fromkeys(data.ALL_REGIONS + regions))
    hourly, _, audits, hourly_hashes = data.prepare_regions()
    capacities, capacity_hashes = {}, {}
    for year in (2023, 2024):
        capacities[year], capacity_hashes[str(year)] = v90.capacities(year)
    ef_vectors = ef_box()
    source_rows = []

    for region in regions:
        frame = hourly[region].copy()
        frame["time"] = pd.to_datetime(frame.time, utc=True)
        frame["month"] = frame.time.dt.strftime("%Y-%m-01")
        cal_region = calibration.loc[calibration.region.eq(region)]
        for month in sorted(cal_region.month.unique()):
            month_key = pd.Timestamp(month).strftime("%Y-%m-01")
            group = frame.loc[frame.month.eq(month_key) & frame.valid.astype(bool)]
            hours = len(group)
            expected = pd.Period(month_key, freq="M").days_in_month * 24
            if hours / expected < 0.80:
                continue
            year = int(month_key[:4])
            cap_year = year - 1
            if cap_year not in capacities:
                continue
            caprow = capacities[cap_year].loc[region] if region in capacities[cap_year].index else pd.Series(dtype=float)
            detailed_caps = {fuel: float(caprow.get(fuel, 0)) for fuel in
                             ("coal", "gas", "oil", "nuclear", "hydro", "wind", "solar", "other")}
            group_caps = {name: sum(detailed_caps[fuel] for fuel in members)
                          for name, members in v90.GROUPS.items()}
            margins = {fuel: sum((F(str(float(x))) for x in group[f"generation_{fuel}"]), F(0))
                       for fuel in v90.FUELS}
            generation = group[[f"generation_{fuel}" for fuel in v90.FUELS]].to_numpy(float)
            exact_generation = [[F(str(float(value))) for value in row] for row in generation]
            factor_box_low, factor_box_high = {}, {}
            cap_sets_by_tau = {}
            for tau in TAUS:
                actual_counts = []
                capacity_sets = []
                old_ef = dict(v90.EF)
                try:
                    for ef in ef_vectors.values():
                        exact_ef = [F(str(float(ef[fuel]))) for fuel in v90.FUELS]
                        actual_k = 0
                        for hourly_fuels in exact_generation:
                            hourly_total = sum(hourly_fuels, F(0))
                            if hourly_total <= 0:
                                continue
                            hourly_carbon = sum((hourly_fuels[index] * exact_ef[index]
                                                 for index in range(len(v90.FUELS))), F(0))
                            actual_k += hourly_carbon > tau * hourly_total
                        actual_counts.append(actual_k)
                        v90.EF.clear(); v90.EF.update(ef)
                        candidate = sharp_group_set(margins, group_caps, tau, hours)
                        if candidate is None:
                            capacity_sets = []
                            break
                        capacity_sets.append(candidate)
                finally:
                    v90.EF.clear(); v90.EF.update(old_ef)
                if not capacity_sets:
                    cap_sets_by_tau[tau] = None
                    continue
                cap_lower = min(min(values) for values in capacity_sets)
                cap_upper = max(max(values) for values in capacity_sets)
                factor_box_low[tau], factor_box_high[tau] = min(actual_counts), max(actual_counts)
                cap_sets_by_tau[tau] = (cap_lower, cap_upper)

            for tau in TAUS:
                if cap_sets_by_tau[tau] is None:
                    continue
                calrow = cal_region.loc[cal_region.month.eq(pd.Timestamp(month)) & cal_region.tau.eq(tau)]
                if len(calrow) != 1:
                    continue
                calrow = calrow.iloc[0]
                radius, ncal, rank = radius_for(calibration, region, tau)
                anchor_lo, anchor_hi = endpoints(float(calrow.anchor_lower_fraction),
                                                 float(calrow.anchor_upper_fraction), radius, hours)
                cap_lo, cap_hi = cap_sets_by_tau[tau]
                inter_lo, inter_hi = max(anchor_lo, cap_lo), min(anchor_hi, cap_hi)
                if inter_lo > inter_hi:
                    inter_lo, inter_hi = cap_lo, cap_hi
                gap = max(0.0, cap_hi / hours - float(calrow.anchor_upper_fraction))
                source_rows.append({"region": region, "month": month_key, "tau": tau, "hours": hours,
                                    "calibration_radius": radius, "calibration_regions": ncal,
                                    "calibration_rank": rank, "anchor_base_upper_fraction": float(calrow.anchor_upper_fraction),
                                    "capacity_lower": cap_lo, "capacity_upper": cap_hi,
                                    "anchor_lower": anchor_lo, "anchor_upper": anchor_hi,
                                    "intersection_lower": inter_lo, "intersection_upper": inter_hi,
                                    "support_gap": gap, "factor_box_K_lower": factor_box_low[tau],
                                    "factor_box_K_upper": factor_box_high[tau]})
        print(f"processed held-out region {region}", flush=True)

    source = pd.DataFrame(source_rows)
    if source.empty:
        raise RuntimeError("no feasible source region-month rows")
    # Each outer fold's support boundary excludes its held-out region entirely.
    for tail in SUPPORT_TAILS:
        outputs = []
        for region, target in source.groupby("region"):
            for tau, group in target.groupby("tau"):
                train = source.loc[(source.region != region) & (source.tau == tau)]
                region_maxima = train.groupby("region").support_gap.max().sort_values().to_numpy()
                n = len(region_maxima)
                rank = math.ceil((n + 1) * (1 - tail))
                if rank > n:
                    threshold = float("inf")
                else:
                    threshold = float(region_maxima[rank - 1])
                for _, row in group.iterrows():
                    fallback = row.support_gap > threshold
                    lo = row.capacity_lower if fallback else row.intersection_lower
                    hi = row.capacity_upper if fallback else row.intersection_upper
                    covered = lo <= row.factor_box_K_lower and hi >= row.factor_box_K_upper
                    outputs.append({"support_tail": tail, "region": region, "month": row.month,
                                    "tau": tau, "hours": row.hours, "support_gap": row.support_gap,
                                    "support_threshold": threshold, "fallback": bool(fallback),
                                    "fallback_lower": lo, "fallback_upper": hi,
                                    "fallback_width": hi - lo, "fallback_covered": bool(covered),
                                    "capacity_width": row.capacity_upper - row.capacity_lower,
                                    "intersection_width": row.intersection_upper - row.intersection_lower,
                                    "intersection_covered": (row.intersection_lower <= row.factor_box_K_lower
                                                              and row.intersection_upper >= row.factor_box_K_upper)})
        tau_rows = pd.DataFrame(outputs)
        if tail == SUPPORT_TAILS[0]:
            all_outputs = tau_rows.copy()
        else:
            all_outputs = pd.concat([all_outputs, tau_rows], ignore_index=True)

    # Summarize all predeclared support quantiles with pooled and region-equal metrics.
    frontier = []
    for (tail, tau), group in all_outputs.groupby(["support_tail", "tau"]):
        per_region = group.groupby("region").agg(
            width=("fallback_width", "mean"), coverage=("fallback_covered", "mean"),
            fallback_rate=("fallback", "mean"), capacity_width=("capacity_width", "mean"),
            intersection_width=("intersection_width", "mean"),
            intersection_coverage=("intersection_covered", "mean"))
        frontier.append({"support_tail": float(tail), "tau": int(tau), "rows": int(len(group)),
                         "regions": int(group.region.nunique()), "macro_width": float(per_region.width.mean()),
                         "macro_coverage": float(per_region.coverage.mean()),
                         "pooled_coverage": float(group.fallback_covered.mean()),
                         "macro_fallback_rate": float(per_region.fallback_rate.mean()),
                         "macro_capacity_width": float(per_region.capacity_width.mean()),
                         "macro_intersection_width": float(per_region.intersection_width.mean()),
                         "macro_intersection_coverage": float(per_region.intersection_coverage.mean())})
    OUT.mkdir(parents=True, exist_ok=False)
    source.to_csv(OUT / "source_outer_fold_features.csv", index=False)
    all_outputs.to_csv(OUT / "outer_fold_fallback_rows.csv", index=False)
    payload = {"status": "REGION_LOO_SUPPORT_GAP_FALLBACK_CV",
               "regions": list(regions), "source_region_months": int(source[["region", "month"]].drop_duplicates().shape[0]),
               "capacity_feasible_rows": int(len(source)), "support_tails": list(SUPPORT_TAILS),
               "frontier": frontier, "calibration_rows_sha256": sha(CALIBRATION_ROWS),
               "capacity_source_hashes": capacity_hashes, "hourly_source_hashes": hourly_hashes,
               "protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__)),
               "source_audits": audits,
               "limitations": ["Outer region-LOO CV is development evidence on the source pool, not independent confirmation.",
                               "Threshold quantiles only control feature support frequency; they are not coverage guarantees.",
                               "The target remains an EIA fixed-factor proxy and lagged U.S. capacity model."]}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(pd.DataFrame(frontier).to_string(index=False))


if __name__ == "__main__":
    main()
