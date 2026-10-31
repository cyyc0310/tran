"""V128: conservatively bound monthly-anchor score over a continuous EF box."""
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

DELTA = .20
TAU = 900
ALPHA = .10
SUPPORT_Q = .20
CAL_ROSTER = ROOT / "results/v111-factor-robust-monthly-anchor-r1/source_calibration_rows.csv"
TARGET_ROSTER = ROOT / "results/v122-october-mtd-frozen-rule-confirmation-r1/availability.csv"
CENSUS = ROOT / "results/eia-ba-quality-census-v52-r1/quality_census.json"
NATIVE = ROOT / "data/native-eia-2026/EIA930_BALANCE_2026_Jul_Dec.csv"
V127_ROWS = ROOT / "results/v127-independent-ef20-corner-stress-r1/target_rows.csv"
V127_GAPS = ROOT / "results/v127-independent-ef20-corner-stress-r1/source_support_gaps.csv"
PROTOCOL = ROOT / "docs/V128_CONTINUOUS_EF_BOX_SCORE_BOUND_PROTOCOL.md"
OUT = ROOT / "results/v128-continuous-ef-box-score-bound-r1"
FOSSIL = ("coal", "gas", "oil", "other")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ef_extreme(multiplier: float) -> dict[str, float]:
    return {fuel: float(ef * (multiplier if fuel in FOSSIL else 1))
            for fuel, ef in v90.EF.items()}


def monthly_mean_and_k(frame: pd.DataFrame, factors: dict[str, float]) -> tuple[F, int, int]:
    margins = {fuel: sum((F(str(float(x))) for x in frame[f"generation_{fuel}"]), F(0))
               for fuel in v90.FUELS}
    total_margin = sum(margins.values(), F(0))
    if total_margin <= 0:
        raise ValueError("nonpositive monthly generation")
    ef = {fuel: F(str(float(factors[fuel]))) for fuel in v90.FUELS}
    mean = sum((margins[fuel] * ef[fuel] for fuel in v90.FUELS), F(0)) / total_margin
    k = 0
    for row in frame.itertuples(index=False):
        values = row._asdict()
        generation = {fuel: F(str(float(values[f"generation_{fuel}"]))) for fuel in v90.FUELS}
        total = sum(generation.values(), F(0))
        if total > 0:
            carbon = sum((generation[fuel] * ef[fuel] for fuel in v90.FUELS), F(0))
            k += carbon > F(TAU) * total
    return mean, k, len(frame)


def region_radius(score_by_region: dict[str, F], excluded: str) -> tuple[F, int, int]:
    scores = sorted(score for region, score in score_by_region.items() if region != excluded)
    n = len(scores)
    rank = math.ceil((n + 1) * (1 - ALPHA))
    if rank > n:
        raise ValueError(f"nonfinite conformal rank for target {excluded}: {rank=}, {n=}")
    return scores[rank - 1], n, rank


def interval(base_lo: int, base_hi: int, radius: F, hours: int) -> tuple[int, int]:
    low = max(0, min(hours, math.ceil(float(F(base_lo) - radius) * hours - 1e-12)))
    high = max(0, min(hours, math.floor(float(F(base_hi) + radius) * hours + 1e-12)))
    return low, high


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    protocol_hash = sha(PROTOCOL)
    calibration = pd.read_csv(CAL_ROSTER)
    calibration["month"] = pd.to_datetime(calibration.month, utc=True)
    source_regions = tuple(sorted(calibration.region.unique()))
    candidates = set(pd.read_csv(TARGET_ROSTER).region.unique())
    census = json.loads(CENSUS.read_text())
    high_h1 = [region for region, period in census["periods"].items()
               if float(period["2026"]["complete_proxy_coverage"]) >= .90]
    targets = tuple(sorted(set(high_h1 + ["NWMT"])))
    if set(targets) != candidates:
        raise ValueError("V122 target roster changed")
    regions = tuple(sorted(set(source_regions + targets)))
    v90.REGIONS = regions
    v90.source_data.ALL_REGIONS = tuple(dict.fromkeys(v90.source_data.ALL_REGIONS + regions))
    old_paths = list(data.default_native_paths())
    data.default_native_paths = lambda: old_paths + [NATIVE]
    data.STOP = pd.Timestamp("2026-11-01T00:00:00Z")
    data.ALL_REGIONS = tuple(dict.fromkeys(data.ALL_REGIONS + regions))
    hourly, _, audits, input_hashes = data.prepare_regions()
    low_ef, high_ef = ef_extreme(1 - DELTA), ef_extreme(1 + DELTA)

    source_rows = []
    score_by_region = {}
    for region in source_regions:
        frame = hourly[region].copy()
        frame["time"] = pd.to_datetime(frame.time, utc=True)
        frame["month"] = frame.time.dt.strftime("%Y-%m-01")
        month_scores = []
        months = calibration.loc[calibration.region.eq(region), "month"].drop_duplicates()
        for month in months:
            key = month.strftime("%Y-%m-01")
            group = frame.loc[frame.month.eq(key) & frame.valid.astype(bool)]
            hours = len(group)
            expected = pd.Period(key, freq="M").days_in_month * 24
            if hours / expected < .80:
                continue
            mean_low, k_low, _ = monthly_mean_and_k(group, low_ef)
            mean_high, k_high, _ = monthly_mean_and_k(group, high_ef)
            anchor_zero_possible = mean_low <= TAU
            anchor_one_possible = mean_high > TAU
            bounds = []
            if anchor_zero_possible:
                bounds.append(F(k_high, hours))
            if anchor_one_possible:
                bounds.append(1 - F(k_low, hours))
            if not bounds:
                raise ArithmeticError("EF box cannot produce an anchor state")
            score_upper = max(bounds)
            corner_score = max(abs(F(k_low, hours) - F(int(mean_low > TAU))),
                               abs(F(k_high, hours) - F(int(mean_high > TAU))))
            month_scores.append(score_upper)
            source_rows.append({"region": region, "month": key, "hours": hours,
                                "mean_low": float(mean_low), "mean_high": float(mean_high),
                                "K_low": k_low, "K_high": k_high,
                                "anchor_zero_possible": anchor_zero_possible,
                                "anchor_one_possible": anchor_one_possible,
                                "corner_score": float(corner_score),
                                "continuous_box_score_upper": float(score_upper)})
        if not month_scores:
            raise ValueError(f"no calibration months for {region}")
        score_by_region[region] = max(month_scores)
        print(f"source {region}", flush=True)

    v127 = pd.read_csv(V127_ROWS)
    v127 = v127[v127.tau.eq(TAU)].copy()
    gaps = pd.read_csv(V127_GAPS)
    gaps = gaps[gaps.tau.eq(TAU)].copy()
    target_rows = []
    for region in targets:
        radius, n_cal, rank = region_radius(score_by_region, region)
        base = v127.loc[v127.region.eq(region)]
        if len(base) == 0:
            # V127 omits the four regions below the frozen 80% valid-hour rule.
            continue
        if len(base) != 1:
            raise ValueError(f"expected at most one V127 target row for {region}")
        prior = base.iloc[0]
        if not bool(prior.capacity_available):
            continue
        frame = hourly[region].copy()
        frame["time"] = pd.to_datetime(frame.time, utc=True)
        frame = frame.loc[frame.time.ge("2026-10-01") & frame.time.lt("2026-11-01")
                          & frame.valid.astype(bool)]
        mean_low, _, hours = monthly_mean_and_k(frame, low_ef)
        mean_high, k_high, _ = monthly_mean_and_k(frame, high_ef)
        _, k_low, _ = monthly_mean_and_k(frame, low_ef)
        base_low, base_high = int(mean_low > TAU), int(mean_high > TAU)
        anchor_low, anchor_high = interval(base_low, base_high, radius, hours)
        cap_low, cap_high = int(prior.capacity_lower), int(prior.capacity_upper)
        inter_low, inter_high = max(anchor_low, cap_low), min(anchor_high, cap_high)
        if inter_low > inter_high:
            inter_low, inter_high = cap_low, cap_high
        fallback = float(prior.support_gap) > float(prior.support_boundary)
        frozen_low, frozen_high = ((cap_low, cap_high) if fallback else (inter_low, inter_high))
        coverage = lambda lo, hi: lo <= k_low and hi >= k_high
        target_rows.append({"region": region, "hours": hours, "radius": float(radius),
                            "calibration_regions": n_cal, "calibration_rank": rank,
                            "anchor_base_lower": base_low, "anchor_base_upper": base_high,
                            "anchor_lower": anchor_low, "anchor_upper": anchor_high,
                            "anchor_width": anchor_high-anchor_low,
                            "factor_K_lower": k_low, "factor_K_upper": k_high,
                            "capacity_lower": cap_low, "capacity_upper": cap_high,
                            "capacity_width": cap_high-cap_low,
                            "capacity_covered": coverage(cap_low, cap_high),
                            "intersection_lower": inter_low, "intersection_upper": inter_high,
                            "intersection_width": inter_high-inter_low,
                            "intersection_covered": coverage(inter_low, inter_high),
                            "support_gap": float(prior.support_gap),
                            "support_boundary": float(prior.support_boundary),
                            "fallback": fallback, "frozen_lower": frozen_low,
                            "frozen_upper": frozen_high, "frozen_width": frozen_high-frozen_low,
                            "frozen_covered": coverage(frozen_low, frozen_high)})
        print(f"target {region}", flush=True)

    target = pd.DataFrame(target_rows)
    frontier = []
    for method in ("capacity", "intersection", "frozen"):
        frontier.append({"tau": TAU, "method": method,
                         "regions": len(target),
                         "endpoint_coverage": float(target[f"{method}_covered"].mean()),
                         "mean_width": float(target[f"{method}_width"].mean()),
                         "fallback_count": int(target.fallback.sum()) if method == "frozen" else 0})
    OUT.mkdir(parents=True)
    source = pd.DataFrame(source_rows)
    source.to_csv(OUT / "source_box_score_bounds.csv", index=False)
    target.to_csv(OUT / "target_rows.csv", index=False)
    summary = {"status": "CONTINUOUS_EF_BOX_CONSERVATIVE_SCORE_BOUND_POSTHOC_OCTOBER",
               "delta": DELTA, "tau": TAU, "source_regions": list(source_regions),
               "target_regions": list(targets), "target_candidate_count": len(targets),
               "target_availability_eligible": 40, "target_capacity_feasible": len(target),
               "source_score_rows": len(source),
               "region_radius_summary": {region: float(value) for region, value in score_by_region.items()},
               "frontier": frontier, "source_audits": audits,
               "input_hashes": {"calibration_roster": sha(CAL_ROSTER), "target_roster": sha(TARGET_ROSTER),
                                "native": sha(NATIVE), "v127_target_rows": sha(V127_ROWS),
                                "v127_support_gaps": sha(V127_GAPS), "protocol": protocol_hash,
                                "script": sha(Path(__file__))},
               "hourly_source_hashes": input_hashes,
               "limitations": ["Posthoc October target labels; no independent time/spatial confirmation.",
                               "The score is a conservative continuous-box upper bound and may overstate the true worst-case score.",
                               "Its region-level calibration still requires time/region exchangeability for a probabilistic coverage claim.",
                               "EIA fixed-factor proxy and capacity analogue; not China hourly validation."]}
    if sha(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed during run")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(pd.DataFrame(frontier).to_string(index=False))


if __name__ == "__main__":
    main()
