"""V119: source-only support-gap fallback diagnostic on the opened V118 cohort."""
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

OUT = ROOT / "results/v119-support-gap-fallback-diagnostic-r1"
PROTOCOL = ROOT / "docs/V119_SUPPORT_GAP_FALLBACK_PROTOCOL.md"
CALIBRATION_ROWS = ROOT / "results/v111-factor-robust-monthly-anchor-r1/source_calibration_rows.csv"
V118_ROWS = ROOT / "results/v118-region-loo-broad-cohort-diagnostic-r1/target_rows.csv"
V118_AVAIL = ROOT / "results/v118-region-loo-broad-cohort-diagnostic-r1/availability.csv"
NATIVE = ROOT / "data/native-eia-2026/EIA930_BALANCE_2026_Jul_Dec.csv"
TAUS = (600, 900)
DELTA = 0.10
ALPHA = 0.10


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ef_box():
    nominal = dict(v90.EF)
    low = {fuel: value * (1 - DELTA) if fuel in ("coal", "gas", "oil", "other") else value
           for fuel, value in nominal.items()}
    high = {fuel: value * (1 + DELTA) if fuel in ("coal", "gas", "oil", "other") else value
            for fuel, value in nominal.items()}
    return {"low": low, "nominal": nominal, "high": high}


def loo_radius(calibration: pd.DataFrame, heldout_region: str, tau: int) -> tuple[float, int, int]:
    other = calibration.loc[~calibration.region.eq(heldout_region) & calibration.tau.eq(tau)]
    scores = other.groupby("region").worst_case_absolute_error.max().sort_values().to_numpy()
    n = len(scores)
    rank = math.ceil((n + 1) * (1 - ALPHA))
    if rank > n:
        raise ValueError(f"non-finite split-conformal radius for {heldout_region}, tau={tau}")
    return float(scores[rank - 1]), n, rank


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
    capacities, cap_hashes = {}, {}
    for year in (2023, 2024):
        capacities[year], cap_hashes[str(year)] = v90.capacities(year)
    vectors = ef_box()
    rows = []
    for region in regions:
        frame = hourly[region].copy()
        frame["time"] = pd.to_datetime(frame.time, utc=True)
        frame["month"] = frame.time.dt.strftime("%Y-%m-01")
        region_cal = calibration.loc[calibration.region.eq(region)]
        for _, calrow in region_cal.iterrows():
            tau = int(calrow.tau)
            month_key = calrow.month.strftime("%Y-%m-01")
            group = frame.loc[frame.month.eq(month_key) & frame.valid.astype(bool)]
            hours = len(group)
            if hours < 1:
                continue
            year = int(month_key[:4])
            cap_year = year - 1
            if cap_year not in capacities:
                continue
            caprow = capacities[cap_year].loc[region] if region in capacities[cap_year].index else pd.Series(dtype=float)
            fine_caps = {fuel: float(caprow.get(fuel, 0)) for fuel in
                         ("coal", "gas", "oil", "nuclear", "hydro", "wind", "solar", "other")}
            group_caps = {name: sum(fine_caps[fuel] for fuel in members)
                          for name, members in v90.GROUPS.items()}
            margins = {fuel: sum((F(str(float(value))) for value in group[f"generation_{fuel}"]), F(0))
                       for fuel in v90.FUELS}
            upper_candidates = []
            old_ef = dict(v90.EF)
            try:
                for ef in vectors.values():
                    v90.EF.clear(); v90.EF.update(ef)
                    feasible_set = sharp_group_set(margins, group_caps, tau, hours)
                    if feasible_set is not None:
                        upper_candidates.append(max(feasible_set))
            finally:
                v90.EF.clear(); v90.EF.update(old_ef)
            if not upper_candidates:
                continue
            cap_upper = max(upper_candidates)
            radius, ncal, rank = loo_radius(calibration, region, tau)
            base_upper = float(calrow.anchor_upper_fraction)
            anchor_upper = min(hours, math.floor(min(1.0, base_upper + radius) * hours + 1e-12))
            gap = max(0.0, cap_upper / hours - anchor_upper / hours)
            rows.append({"region": region, "month": month_key, "tau": tau, "hours": hours,
                         "capacity_upper_box": cap_upper, "anchor_base_upper_fraction": base_upper,
                         "anchor_upper": anchor_upper, "loo_radius": radius,
                         "loo_calibration_regions": ncal, "loo_rank": rank,
                         "normalized_support_gap": gap,
                         "worst_case_absolute_error": float(calrow.worst_case_absolute_error)})
        print(f"processed source region {region}", flush=True)

    source_features = pd.DataFrame(rows)
    if source_features.empty:
        raise RuntimeError("no source support-gap features computed")
    boundaries = source_features.groupby("tau").normalized_support_gap.max().to_dict()
    target = pd.read_csv(V118_ROWS)
    availability = pd.read_csv(V118_AVAIL)
    target = target.loc[target.capacity_available].copy()
    target["normalized_support_gap"] = (target.capacity_upper - target.anchor_upper).clip(lower=0) / target.hours
    target["support_boundary"] = target.tau.map(boundaries)
    target["fallback_to_capacity"] = target.normalized_support_gap > target.support_boundary
    target["fallback_lower"] = target.capacity_lower.where(target.fallback_to_capacity, target.intersection_lower)
    target["fallback_upper"] = target.capacity_upper.where(target.fallback_to_capacity, target.intersection_upper)
    target["fallback_width"] = target.fallback_upper - target.fallback_lower
    target["fallback_factor_box_covered"] = (
        (target.fallback_lower <= target.factor_box_K_lower)
        & (target.fallback_upper >= target.factor_box_K_upper))
    metrics = []
    for tau, group in target.groupby("tau"):
        by_region = group.groupby("region").agg(
            width=("fallback_width", "mean"),
            covered=("fallback_factor_box_covered", "mean"),
            fallback_rate=("fallback_to_capacity", "mean"),
            intersection_width=("intersection_width", "mean"),
            capacity_width=("capacity_width", "mean"),
            intersection_covered=("intersection_factor_box_covered", lambda s: s.eq(True).mean()),
        )
        metrics.append({"tau": int(tau), "eligible_rows": int(len(group)),
                        "regions": int(group.region.nunique()),
                        "macro_fallback_width": float(by_region.width.mean()),
                        "macro_intersection_width": float(by_region.intersection_width.mean()),
                        "macro_capacity_width": float(by_region.capacity_width.mean()),
                        "macro_fallback_coverage": float(by_region.covered.mean()),
                        "macro_intersection_coverage": float(by_region.intersection_covered.mean()),
                        "macro_fallback_rate": float(by_region.fallback_rate.mean()),
                        "pooled_fallback_coverage": float(group.fallback_factor_box_covered.mean()),
                        "support_boundary": float(boundaries[int(tau)])})

    OUT.mkdir(parents=True, exist_ok=False)
    source_features.to_csv(OUT / "source_support_features.csv", index=False)
    target.to_csv(OUT / "target_fallback_rows.csv", index=False)
    payload = {"status": "EXPLORATORY_SOURCE_SUPPORT_GAP_FALLBACK_DIAGNOSTIC",
               "source_regions": list(regions), "source_feature_rows": len(source_features),
               "target_regions": sorted(target.region.unique()),
               "target_region_months": int(target[["region", "month"]].drop_duplicates().shape[0]),
               "source_only_support_boundaries": {int(k): float(v) for k, v in boundaries.items()},
               "frontier": metrics,
               "calibration_rows_sha256": sha(CALIBRATION_ROWS), "v118_rows_sha256": sha(V118_ROWS),
               "v118_availability_sha256": sha(V118_AVAIL), "protocol_sha256": sha(PROTOCOL),
               "script_sha256": sha(Path(__file__)), "capacity_source_hashes": cap_hashes,
               "hourly_source_hashes": hourly_hashes, "source_audits": audits,
               "limitations": ["Threshold is source-feature support maximum, not a coverage theorem.",
                               "V118 target labels and region cohort were already opened.",
                               "Source and target regions may not be exchangeable after the completeness-based selection.",
                               "Target outcomes are fixed-factor EIA proxy endpoints, not measured Chinese CIF."]}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__ == "__main__":
    main()
