"""V125: post-hoc sensitivity to October-matched source calibration."""
from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction as F
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v125-october-matched-calibration-diagnostic-r1"
PROTOCOL = ROOT / "docs/V125_OCTOBER_MATCHED_CALIBRATION_DIAGNOSTIC_PROTOCOL.md"
CAL = ROOT / "results/v111-factor-robust-monthly-anchor-r1/source_calibration_rows.csv"
TARGET = ROOT / "results/v122-october-mtd-frozen-rule-confirmation-r1/target_rows.csv"
ALPHA = 0.10
RATES = (0.01, 0.05, 0.10)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calibrate(cal: pd.DataFrame, region: str, tau: int) -> tuple[float, int, int]:
    train = cal.loc[(cal.tau == tau) & (cal.month.dt.month == 10) & (cal.region != region)]
    scores = train.groupby("region").worst_case_absolute_error.max().sort_values().to_numpy()
    n = len(scores); rank = math.ceil((n + 1) * (1 - ALPHA))
    if rank > n:
        raise ValueError(f"non-finite matched calibration for {region}, tau={tau}")
    return float(scores[rank - 1]), n, rank


def endpoints(base_low: int, base_high: int, radius: float, hours: int):
    lo = max(0, min(hours, math.ceil(max(0.0, base_low - radius) * hours - 1e-12)))
    hi = max(0, min(hours, math.floor(min(1.0, base_high + radius) * hours + 1e-12)))
    return lo, hi


def decide(lo, hi, hours, rate):
    req = int((F(str(rate)) * hours).__ceil__())
    pos, neg = lo >= req, hi < req
    return pos, neg, not (pos or neg)


def main():
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite: {OUT}")
    cal = pd.read_csv(CAL)
    cal["month"] = pd.to_datetime(cal.month, utc=True)
    target = pd.read_csv(TARGET)
    rows = []
    for _, src in target.iterrows():
        row = src.to_dict()
        region, tau, hours = str(src.region), int(src.tau), int(src.hours)
        radius, ncal, rank = calibrate(cal, region, tau)
        old_radius = float(src.calibration_radius)
        # The V122 anchor base endpoints are binary (the monthly mean is either above or below tau).
        base_low = int(int(src.anchor_lower) > 0)
        base_high = int(int(src.anchor_upper) == hours)
        old_lo, old_hi = endpoints(base_low, base_high, old_radius, hours)
        if old_lo != int(src.anchor_lower) or old_hi != int(src.anchor_upper):
            raise ValueError(f"failed to reconstruct original anchor: {region}, tau={tau}")
        lo, hi = endpoints(base_low, base_high, radius, hours)
        row.update({"matched_radius": radius, "matched_calibration_regions": ncal,
                    "matched_rank": rank, "base_lower_inferred": base_low,
                    "base_upper_inferred": base_high,
                    "matched_anchor_lower": lo, "matched_anchor_upper": hi,
                    "matched_anchor_width": hi - lo,
                    "matched_anchor_covered": lo <= src.factor_box_K_lower and hi >= src.factor_box_K_upper})
        if bool(src.capacity_available):
            cap_lo, cap_hi = int(src.capacity_lower), int(src.capacity_upper)
            inter_lo, inter_hi = max(lo, cap_lo), min(hi, cap_hi)
            if inter_lo > inter_hi:
                inter_lo, inter_hi = cap_lo, cap_hi
            gap = float(src.support_gap)
            fallback = tau == 900 and gap > float(src.support_boundary)
            if tau == 600 or fallback:
                frozen_lo, frozen_hi, policy = cap_lo, cap_hi, "capacity_only_tau600" if tau == 600 else "capacity_fallback"
            else:
                frozen_lo, frozen_hi, policy = inter_lo, inter_hi, "matched_anchor_capacity_intersection"
            row.update({"matched_intersection_lower": inter_lo, "matched_intersection_upper": inter_hi,
                        "matched_intersection_width": inter_hi - inter_lo,
                        "matched_intersection_covered": inter_lo <= src.factor_box_K_lower and inter_hi >= src.factor_box_K_upper,
                        "matched_fallback": fallback, "matched_policy": policy,
                        "matched_frozen_lower": frozen_lo, "matched_frozen_upper": frozen_hi,
                        "matched_frozen_width": frozen_hi - frozen_lo,
                        "matched_frozen_covered": frozen_lo <= src.factor_box_K_lower and frozen_hi >= src.factor_box_K_upper})
        else:
            for field in ("matched_intersection_lower", "matched_intersection_upper", "matched_intersection_width",
                          "matched_intersection_covered", "matched_fallback", "matched_frozen_lower",
                          "matched_frozen_upper", "matched_frozen_width", "matched_frozen_covered"):
                row[field] = None
            row["matched_policy"] = "unavailable_capacity"
        for rate in RATES:
            for method, low_name, high_name in (
                ("matched_anchor", "matched_anchor_lower", "matched_anchor_upper"),
                ("matched_intersection", "matched_intersection_lower", "matched_intersection_upper"),
                ("matched_frozen", "matched_frozen_lower", "matched_frozen_upper")):
                lower, upper = row[low_name], row[high_name]
                if lower is None or pd.isna(lower):
                    for state in ("positive", "negative", "abstain"):
                        row[f"{method}_{state}_{rate:g}"] = None
                else:
                    p, n, a = decide(int(lower), int(upper), hours, rate)
                    row[f"{method}_positive_{rate:g}"] = p
                    row[f"{method}_negative_{rate:g}"] = n
                    row[f"{method}_abstain_{rate:g}"] = a
        rows.append(row)
    detail = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=False)
    detail.to_csv(OUT / "target_rows.csv", index=False)
    frontier = []
    for tau, g in detail.groupby("tau"):
        eligible = g[g.capacity_available.eq(True)]
        item = {"tau": int(tau), "scored_regions": int(g.region.nunique()),
                "capacity_feasible_regions": int(eligible.region.nunique()),
                "matched_radius_mean": float(g.matched_radius.mean()),
                "old_radius_mean": float(g.calibration_radius.mean())}
        for method, width, covered in (
            ("old_anchor", "anchor_width", "anchor_factor_box_covered"),
            ("matched_anchor", "matched_anchor_width", "matched_anchor_covered"),
            ("old_intersection", "intersection_width", "intersection_factor_box_covered"),
            ("matched_intersection", "matched_intersection_width", "matched_intersection_covered"),
            ("old_frozen", "frozen_width", "frozen_factor_box_covered"),
            ("matched_frozen", "matched_frozen_width", "matched_frozen_covered"),
            ("capacity", "capacity_width", "capacity_factor_box_covered")):
            subset = g if method in ("old_anchor", "matched_anchor") else eligible
            item[f"{method}_width"] = float(subset[width].mean())
            item[f"{method}_coverage"] = float(subset[covered].eq(True).mean())
        item["matched_fallback_count"] = int(eligible.matched_fallback.eq(True).sum())
        for rate in RATES:
            for state in ("positive", "negative", "abstain"):
                item[f"matched_frozen_{state}_{rate:g}"] = float(eligible[f"matched_frozen_{state}_{rate:g}"].mean())
        frontier.append(item)
    payload = {"status": "POSTHOC_OCTOBER_MATCHED_CALIBRATION_DIAGNOSTIC",
               "target_rows": len(detail), "calibration_rows_sha256": sha(CAL),
               "target_rows_sha256": sha(TARGET), "protocol_sha256": sha(PROTOCOL),
               "script_sha256": sha(Path(__file__)), "frontier": frontier,
               "limitations": ["V122 October target outcomes were already observed.",
                               "Calibration uses full historical Octobers while target is partial October.",
                               "This is a sensitivity analysis, not independent validation."]}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(pd.DataFrame(frontier).to_string(index=False))


if __name__ == "__main__":
    main()
