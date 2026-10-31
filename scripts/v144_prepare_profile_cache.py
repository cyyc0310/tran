"""V144: canonical-factor cache for the already examined monthly cohort."""
from __future__ import annotations

import json
import math
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import pandas as pd

import v120_support_gap_region_loo_cv as v120
import v139_physical_reserve_calibration as v139
import v90_coarse_capacity_identification as v90
from v141_prepare_expanded_sources import exact_count, monthly_hull
from v141_source_diversity_ablation import original_budget_frame

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V144_CANONICAL_PROFILE_RESPONSE_PROTOCOL.md"
ORIGINAL = ROOT / "results/v140-monthly-conditional-risk-r1/monthly_feature_panel.csv"
ADDITIONS = ROOT / "results/v141-source-expansion-input-r1/additional_monthly_panel.csv"
OUT = ROOT / "results/v144-canonical-profile-cache-r1"
FACTORS = {"low": (900, 360, 720, 0, 0, 0, 0, 360),
           "nominal": (1000, 400, 800, 0, 0, 0, 0, 400),
           "high": (1100, 440, 880, 0, 0, 0, 0, 440)}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    protocol_hash = v139.sha(PROTOCOL)
    old = original_budget_frame(pd.read_csv(ORIGINAL))
    additional = pd.read_csv(ADDITIONS)
    old = pd.concat([old, additional.loc[additional.eligible_donor]], ignore_index=True, sort=False)
    keys = ["region", "month", "tau"]
    if old.duplicated(keys).any() or len(old) != 1872:
        raise ValueError("unexpected frozen monthly cohort")
    cohort = old[["region", "month", "hours"]].drop_duplicates().sort_values(["region", "month"])
    regions = tuple(sorted(cohort.region.unique()))
    data = v90.source_data
    data.ALL_REGIONS = regions
    data.START, data.STOP = pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")
    hourly, _, audits, raw_hashes = data.prepare_regions()
    legacy = v120.ef_box()
    factor_maps = {name: dict(zip(v90.FUELS, values)) for name, values in FACTORS.items()}
    profiles, monthly_rows, differences, arrays, margin_strings = [], [], [], [], []
    offset = 0
    for region in regions:
        frame = hourly[region].copy()
        frame["month"] = pd.to_datetime(frame.time, utc=True).dt.strftime("%Y-%m-01")
        for item in cohort.loc[cohort.region.eq(region)].itertuples(index=False):
            group = frame.loc[frame.month.eq(item.month) & frame.valid.astype(bool)]
            generation = group[[f"generation_{fuel}" for fuel in v90.FUELS]].to_numpy(float)
            if len(group) != item.hours or not np.isfinite(generation).all() or (generation < 0).any():
                raise ValueError("raw profile does not match archived hours")
            margins = {fuel: sum((F(str(float(value))) for value in generation[:, j]), F(0))
                       for j, fuel in enumerate(v90.FUELS)}
            total = sum(margins.values(), F(0))
            profile_id = len(profiles)
            profiles.append({"profile_id": profile_id, "region": region, "month": item.month,
                             "hours": item.hours, "offset": offset, "stop": offset + item.hours})
            arrays.append(generation)
            margin_strings.append([str(margins[fuel]) for fuel in v90.FUELS])
            offset += item.hours
            month_number = int(item.month[5:7])
            common = {"profile_id": profile_id, "region": region, "month": item.month, "hours": item.hours,
                      "log_total_mwh": math.log1p(float(total)), "month_sin": math.sin(2 * math.pi * month_number / 12),
                      "month_cos": math.cos(2 * math.pi * month_number / 12)}
            common.update({f"fuelshare_{fuel}": float(margins[fuel] / total) for fuel in v90.FUELS})
            for tau in (600, 900):
                archived = old.loc[old.region.eq(region) & old.month.eq(item.month) & old.tau.eq(tau)]
                if len(archived) != 1:
                    raise ValueError("missing archived label row")
                archived = archived.iloc[0]
                for fuel in v90.FUELS:
                    if abs(common[f"fuelshare_{fuel}"] - archived[f"fuelshare_{fuel}"]) > 1e-12:
                        raise ValueError("archived monthly generation shares changed")
                counts, hulls, anchors = {}, {}, {}
                for name in ("low", "high"):
                    old_count, _ = exact_count(generation, legacy[name], tau)
                    expected = int(archived.factor_box_K_lower if name == "low" else archived.factor_box_K_upper)
                    if old_count != expected:
                        raise ValueError(f"legacy count not reproduced: {region}/{item.month}/{tau}/{name}")
                    counts[name], _ = exact_count(generation, factor_maps[name], tau)
                    hulls[name] = monthly_hull(margins, factor_maps[name], tau, item.hours)
                    excess = sum((margins[fuel] * (factor_maps[name][fuel] - tau) for fuel in v90.FUELS), F(0))
                    anchors[name] = int(excess > 0)
                    differences.append({"region": region, "month": item.month, "tau": tau, "factor": name,
                                        "legacy_count": old_count, "canonical_count": counts[name],
                                        "delta": counts[name] - old_count})
                nominal, _ = exact_count(generation, factor_maps["nominal"], tau)
                physical_lower, physical_upper = hulls["low"][0], hulls["high"][1]
                if not physical_lower <= counts["low"] <= nominal <= counts["high"] <= physical_upper:
                    raise ValueError("canonical factor/order/physical containment failure")
                monthly_rows.append({**common, "tau": tau, "factor_box_K_lower": counts["low"],
                                     "factor_box_K_upper": counts["high"], "nominal_K": nominal,
                                     "anchor_lower_fraction": anchors["low"], "anchor_upper_fraction": anchors["high"],
                                     "physical_lower": physical_lower, "physical_upper": physical_upper,
                                     "physical_lower_fraction": physical_lower / item.hours,
                                     "physical_upper_fraction": physical_upper / item.hours,
                                     "physical_width_fraction": (physical_upper - physical_lower) / item.hours})
        print(f"canonical profile {region}", flush=True)
    panel = pd.DataFrame(monthly_rows)
    comparison = pd.DataFrame(differences)
    if len(profiles) != 936 or len(panel) != 1872 or protocol_hash != v139.sha(PROTOCOL):
        raise ValueError("frozen denominator/protocol failure")
    OUT.mkdir(parents=True)
    pd.DataFrame(profiles).to_csv(OUT / "profile_index.csv", index=False)
    panel.to_csv(OUT / "canonical_monthly_panel.csv", index=False)
    comparison.to_csv(OUT / "factor_representation_comparison.csv", index=False)
    np.savez_compressed(OUT / "profiles.npz", generation=np.concatenate(arrays, axis=0),
                        margin_fractions=np.asarray(margin_strings, dtype=str), fuels=np.asarray(v90.FUELS, dtype=str))
    payload = {"status": "CANONICAL_EXACT_FACTOR_PROFILE_CACHE_OPENED_COHORT", "profiles": len(profiles),
               "monthly_rows": len(panel), "regions": len(regions), "observed_hour_rows": offset,
               "legacy_endpoint_counts_reproduced": len(comparison),
               "changed_canonical_endpoint_counts": int(comparison.delta.ne(0).sum()),
               "maximum_absolute_count_change": int(comparison.delta.abs().max()), "canonical_factor_values": FACTORS,
               "legacy_factor_values": legacy, "raw_source_hashes": raw_hashes, "source_audits": audits,
               "input_hashes": {str(p.relative_to(ROOT)): v139.sha(p) for p in (ORIGINAL, ADDITIONS, PROTOCOL)},
               "script_sha256": v139.sha(Path(__file__)), "profiles_sha256": v139.sha(OUT / "profiles.npz"),
               "panel_sha256": v139.sha(OUT / "canonical_monthly_panel.csv"),
               "limitations": ["Literal factors are an assumed proxy contract, not Chinese empirical uncertainty.",
                               "No new region identity or future evaluation period is opened.",
                               "Archived floating-factor outputs are preserved; numerical label changes are not method gains."]}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(comparison.loc[comparison.delta.ne(0)].to_string(index=False))
    print(f"profiles: {len(profiles)}; changed endpoint counts: {comparison.delta.ne(0).sum()}")


if __name__ == "__main__":
    main()
