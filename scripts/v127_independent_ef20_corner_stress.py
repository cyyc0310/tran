"""V127: recompute tau=900 calibration under 16 independent EF-box corners."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction as F
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v54_unseen_local_ef_confirmation as data  # noqa: E402
import v90_coarse_capacity_identification as v90  # noqa: E402
from scripts.v92_sharp_coarse_capacity_identification import sharp_group_set  # noqa: E402

_WORKER_CALIBRATION_ROSTER = None
_WORKER_HOURLY = None
_WORKER_CAPACITIES = None
_WORKER_VECTORS = None

DELTA = 0.20
TAUS = (900,)
RATES = (0.01, 0.05, 0.10)
CAL_ROSTER = ROOT / "results/v111-factor-robust-monthly-anchor-r1/source_calibration_rows.csv"
TARGET_ROSTER = ROOT / "results/v122-october-mtd-frozen-rule-confirmation-r1/availability.csv"
CENSUS = ROOT / "results/eia-ba-quality-census-v52-r1/quality_census.json"
NATIVE = ROOT / "data/native-eia-2026/EIA930_BALANCE_2026_Jul_Dec.csv"
PROTOCOL = ROOT / "docs/V127_INDEPENDENT_EF20_CORNER_PROTOCOL.md"
OUT = ROOT / "results/v127-independent-ef20-corner-stress-r1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ef_box():
    nominal = dict(v90.EF)
    fossil = ("coal", "gas", "oil", "other")
    vectors = {"nominal": nominal}
    for corner in itertools.product((0, 1), repeat=len(fossil)):
        label = "".join("H" if bit else "L" for bit in corner)
        vector = dict(nominal)
        for fuel, bit in zip(fossil, corner):
            vector[fuel] = nominal[fuel] * (1 + DELTA if bit else 1 - DELTA)
        vectors[label] = vector
    return vectors


def capacities_by_group(caprow):
    fine = {f: F(str(float(caprow.get(f, 0)))) for f in
            ("coal", "gas", "oil", "nuclear", "hydro", "wind", "solar", "other")}
    return {g: sum((fine[f] for f in fuels), F(0)) for g, fuels in v90.GROUPS.items()}


def q_radius(scores_by_region, excluded, tau, alpha=.10):
    vals = sorted(value for region, value in scores_by_region.items() if region != excluded)
    n = len(vals); rank = math.ceil((n + 1) * (1 - alpha))
    if rank > n:
        raise ValueError(f"non-finite radius, n={n}, {excluded=}, {tau=}")
    return vals[rank - 1], n, rank


def q_support(gap_by_region, excluded, tau, tail=.20):
    vals = sorted(value for region, value in gap_by_region.items() if region != excluded)
    n = len(vals); rank = math.ceil((n + 1) * (1 - tail))
    if rank > n:
        raise ValueError(f"non-finite support boundary, n={n}, {excluded=}, {tau=}")
    return vals[rank - 1], n, rank


def int_interval(base_lo, base_hi, radius, hours):
    lo = max(0, min(hours, math.ceil(max(0.0, base_lo - radius) * hours - 1e-12)))
    hi = max(0, min(hours, math.floor(min(1.0, base_hi + radius) * hours + 1e-12)))
    return lo, hi


def decide(lo, hi, hours, rate):
    req = int((F(str(rate)) * hours).__ceil__())
    return lo >= req, hi < req, not (lo >= req or hi < req)


def source_region_worker(region):
    """Build one independent leave-region-out source fold (fork-inherited inputs)."""
    calibration_roster = _WORKER_CALIBRATION_ROSTER
    hourly = _WORKER_HOURLY
    capacity_vintages = _WORKER_CAPACITIES
    vectors = _WORKER_VECTORS
    frame = hourly[region].copy()
    frame["time"] = pd.to_datetime(frame.time, utc=True)
    frame["month"] = frame.time.dt.strftime("%Y-%m-01")
    score_rows, gap_rows = [], []
    cal_months = calibration_roster.loc[calibration_roster.region.eq(region), "month"].drop_duplicates()
    for month in cal_months:
        key = month.strftime("%Y-%m-01")
        group = frame.loc[frame.month.eq(key) & frame.valid.astype(bool)]
        hours = len(group); expected = pd.Period(key, freq="M").days_in_month * 24
        if hours / expected < .80:
            continue
        year = int(key[:4]); vintage = year - 1
        caprow = capacity_vintages[vintage].loc[region] if region in capacity_vintages[vintage].index else pd.Series(dtype=float)
        caps = capacities_by_group(caprow)
        margins = {fuel: sum((F(str(float(x))) for x in group[f"generation_{fuel}"]), F(0))
                   for fuel in v90.FUELS}
        exact_rows = []
        for _, hr in group.iterrows():
            exact_rows.append({fuel: F(str(float(hr[f"generation_{fuel}"]))) for fuel in v90.FUELS})
        for tau in TAUS:
            anchor_bases, observed_fracs, capacity_sets = [], [], []
            for ef in vectors.values():
                exact_ef = {fuel: F(str(float(ef[fuel]))) for fuel in v90.FUELS}
                total_margin = sum(margins.values(), F(0))
                mean = sum((margins[f] * exact_ef[f] for f in v90.FUELS), F(0)) / total_margin if total_margin > 0 else F(0)
                anchor_bases.append(int(mean > tau))
                k = 0
                for gens in exact_rows:
                    total = sum(gens.values(), F(0))
                    if total > 0:
                        co2 = sum((gens[f] * exact_ef[f] for f in v90.FUELS), F(0))
                        k += co2 > F(tau) * total
                observed_fracs.append(F(k, hours))
                old_ef = dict(v90.EF)
                try:
                    v90.EF.clear(); v90.EF.update(ef)
                    capacity_sets.append(sharp_group_set(margins, caps, tau, hours))
                finally:
                    v90.EF.clear(); v90.EF.update(old_ef)
            base_lo, base_hi = min(anchor_bases), max(anchor_bases)
            score = max(max(F(base_lo) - actual, F(0), actual - F(base_hi))
                        for actual in observed_fracs)
            score_rows.append({"region": region, "month": key, "tau": tau,
                               "worst_case_absolute_error": float(score),
                               "anchor_base_lower": base_lo, "anchor_base_upper": base_hi})
            if capacity_sets and all(x is not None for x in capacity_sets):
                cap_hi = max(max(x) for x in capacity_sets)
                gap = max(F(0), F(cap_hi, hours) - F(base_hi))
                gap_rows.append({"region": region, "month": key, "tau": tau,
                                 "support_gap": float(gap), "capacity_upper": cap_hi,
                                 "hours": hours})
    return region, score_rows, gap_rows


def main():
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    protocol_hash = sha(PROTOCOL)
    calibration_roster = pd.read_csv(CAL_ROSTER)
    calibration_roster["month"] = pd.to_datetime(calibration_roster.month, utc=True)
    source_regions = tuple(sorted(calibration_roster.region.unique()))
    frozen_candidates = set(pd.read_csv(TARGET_ROSTER).region.unique())
    census = json.loads(CENSUS.read_text())
    high_h1 = [r for r,p in census["periods"].items() if float(p["2026"]["complete_proxy_coverage"]) >= .90]
    target_regions = tuple(sorted(set(high_h1 + ["NWMT"])))
    if set(target_regions) != frozen_candidates:
        raise ValueError("V122 target cohort no longer matches its frozen candidate roster")
    all_regions = tuple(sorted(set(source_regions + target_regions)))
    v90.REGIONS = all_regions
    v90.source_data.ALL_REGIONS = tuple(dict.fromkeys(v90.source_data.ALL_REGIONS + all_regions))
    old_paths = list(data.default_native_paths())
    data.default_native_paths = lambda: old_paths + [NATIVE]
    data.STOP = pd.Timestamp("2026-11-01T00:00:00Z")
    data.ALL_REGIONS = tuple(dict.fromkeys(data.ALL_REGIONS + all_regions))
    hourly, _, audits, hourly_hashes = data.prepare_regions()
    capacity_vintages, capacity_hashes = {}, {}
    for year in (2023, 2024, 2025):
        capacity_vintages[year], capacity_hashes[str(year)] = v90.capacities(year)
    vectors = ef_box()

    # Fork workers after inputs are prepared, so hourly tables are shared copy-on-write.
    global _WORKER_CALIBRATION_ROSTER, _WORKER_HOURLY, _WORKER_CAPACITIES, _WORKER_VECTORS
    _WORKER_CALIBRATION_ROSTER = calibration_roster
    _WORKER_HOURLY = hourly
    _WORKER_CAPACITIES = capacity_vintages
    _WORKER_VECTORS = vectors
    source_score_rows, source_gap_rows = [], []
    worker_count = min(4, len(source_regions), os.cpu_count() or 1)
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as pool:
        futures = {pool.submit(source_region_worker, region): region for region in source_regions}
        for future in as_completed(futures):
            region, scores, gaps = future.result()
            source_score_rows.extend(scores)
            source_gap_rows.extend(gaps)
            print(f"source {region} complete ({len(scores)} score rows)", flush=True)

    source_scores = pd.DataFrame(source_score_rows)
    source_gaps = pd.DataFrame(source_gap_rows)
    score_by_region = {tau: source_scores[source_scores.tau.eq(tau)].groupby("region").worst_case_absolute_error.max().to_dict()
                       for tau in TAUS}
    gap_by_region = {tau: source_gaps[source_gaps.tau.eq(tau)].groupby("region").support_gap.max().to_dict()
                     for tau in TAUS}
    ef_rows, availability = [], []
    capacity_target, capacity_hash = capacity_vintages[2025], capacity_hashes["2025"]
    for region in target_regions:
        radii = {tau: q_radius(score_by_region[tau], region, tau)[0] for tau in TAUS}
        boundaries = {tau: q_support(gap_by_region[tau], region, tau)[0] for tau in TAUS}
        frame = hourly[region].copy(); frame["time"] = pd.to_datetime(frame.time, utc=True)
        frame = frame.loc[frame.time.ge("2026-10-01") & frame.time.lt("2026-11-01")
                          & frame.valid.astype(bool)].copy()
        hours = len(frame); expected = 744; coverage = hours / expected
        caprow = capacity_target.loc[region] if region in capacity_target.index else pd.Series(dtype=float)
        caps = capacities_by_group(caprow)
        margins = {fuel: sum((F(str(float(x))) for x in frame[f"generation_{fuel}"]), F(0))
                   for fuel in v90.FUELS}
        energy_ok = hours > 0 and all(sum(margins[f] for f in fuels) <= F(hours) * caps[g]
                                      for g,fuels in v90.GROUPS.items())
        cap_available = coverage >= .80 and energy_ok
        availability.append({"region": region, "hours": hours, "expected_hours": expected,
                             "coverage": coverage, "eligible": coverage >= .80,
                             "capacity_feasible": bool(cap_available)})
        if coverage < .80:
            continue
        exact_rows = []
        for _, hr in frame.iterrows():
            exact_rows.append({fuel: F(str(float(hr[f"generation_{fuel}"]))) for fuel in v90.FUELS})
        for tau in TAUS:
            anchor_bases, actual_counts, monthly_sets, capacity_sets = [], [], [], []
            for ef in vectors.values():
                exact_ef = {fuel: F(str(float(ef[fuel]))) for fuel in v90.FUELS}
                total_margin = sum(margins.values(), F(0))
                mean = sum((margins[f] * exact_ef[f] for f in v90.FUELS), F(0)) / total_margin if total_margin else F(0)
                anchor_bases.append(int(mean > tau))
                k = 0
                for gens in exact_rows:
                    total = sum(gens.values(), F(0))
                    if total > 0:
                        co2 = sum((gens[f] * exact_ef[f] for f in v90.FUELS), F(0))
                        k += co2 > F(tau) * total
                actual_counts.append(k)
                old_ef = dict(v90.EF)
                try:
                    v90.EF.clear(); v90.EF.update(ef)
                    monthly_sets.append(v90.v66_set(margins, tau, hours))
                    if cap_available:
                        capacity_sets.append(sharp_group_set(margins, caps, tau, hours))
                finally:
                    v90.EF.clear(); v90.EF.update(old_ef)
            base_lo, base_hi = min(anchor_bases), max(anchor_bases)
            anchor_lo, anchor_hi = int_interval(base_lo, base_hi, radii[tau], hours)
            k_lo, k_hi = min(actual_counts), max(actual_counts)
            month_lo, month_hi = min(min(x) for x in monthly_sets), max(max(x) for x in monthly_sets)
            nominal_k = 0
            for gens in exact_rows:
                total = sum(gens.values(), F(0))
                if total > 0:
                    co2 = sum((gens[f] * F(str(float(v90.EF[f]))) for f in v90.FUELS), F(0))
                    nominal_k += co2 > F(tau) * total
            row = {"region": region, "month": "2026-10-01", "tau": tau, "hours": hours,
                   "coverage": coverage, "capacity_available": bool(cap_available),
                   "radius": radii[tau], "anchor_lower": anchor_lo, "anchor_upper": anchor_hi,
                   "anchor_width": anchor_hi-anchor_lo,
                   "anchor_covered": anchor_lo <= k_lo and anchor_hi >= k_hi,
                   "monthly_lower": month_lo, "monthly_upper": month_hi,
                   "factor_K_lower": k_lo, "factor_K_upper": k_hi,
                   "actual_nominal_K": nominal_k}
            if cap_available:
                cap_lo, cap_hi = min(min(x) for x in capacity_sets), max(max(x) for x in capacity_sets)
                inter_lo, inter_hi = max(anchor_lo, cap_lo), min(anchor_hi, cap_hi)
                if inter_lo > inter_hi: inter_lo, inter_hi = cap_lo, cap_hi
                gap = max(0.0, cap_hi / hours - base_hi)
                fallback = tau == 900 and gap > boundaries[tau]
                if tau == 600 or fallback: frozen_lo, frozen_hi = cap_lo, cap_hi
                else: frozen_lo, frozen_hi = inter_lo, inter_hi
                row.update({"capacity_lower": cap_lo, "capacity_upper": cap_hi,
                            "capacity_width": cap_hi-cap_lo,
                            "capacity_covered": cap_lo <= k_lo and cap_hi >= k_hi,
                            "intersection_lower": inter_lo, "intersection_upper": inter_hi,
                            "intersection_width": inter_hi-inter_lo,
                            "intersection_covered": inter_lo <= k_lo and inter_hi >= k_hi,
                            "support_gap": gap, "support_boundary": boundaries[tau],
                            "fallback": fallback, "frozen_lower": frozen_lo,
                            "frozen_upper": frozen_hi, "frozen_width": frozen_hi-frozen_lo,
                            "frozen_covered": frozen_lo <= k_lo and frozen_hi >= k_hi})
            else:
                for c in ("capacity_lower","capacity_upper","capacity_width","capacity_covered",
                          "intersection_lower","intersection_upper","intersection_width","intersection_covered",
                          "support_gap","support_boundary","fallback","frozen_lower","frozen_upper",
                          "frozen_width","frozen_covered"):
                    row[c] = None
            ef_rows.append(row)
        print(f"target {region}", flush=True)

    target = pd.DataFrame(ef_rows); av = pd.DataFrame(availability)
    OUT.mkdir(parents=True, exist_ok=False)
    source_scores.to_csv(OUT / "source_calibration_scores.csv", index=False)
    source_gaps.to_csv(OUT / "source_support_gaps.csv", index=False)
    target.to_csv(OUT / "target_rows.csv", index=False)
    av.to_csv(OUT / "availability.csv", index=False)
    frontier = []
    for tau,g in target.groupby("tau"):
        eligible = g[g.capacity_available.eq(True)]
        record = {"tau": int(tau), "candidate_regions": len(target_regions),
                  "eligible_regions": int(g.region.nunique()),
                  "capacity_feasible_regions": int(eligible.region.nunique()),
                  "nonzero_factor_box_upper": int((eligible.factor_K_upper>0).sum())}
        for method, width, covered in (("anchor","anchor_width","anchor_covered"),
                                       ("capacity","capacity_width","capacity_covered"),
                                       ("intersection","intersection_width","intersection_covered"),
                                       ("frozen","frozen_width","frozen_covered")):
            subset = g if method == "anchor" else eligible
            record[f"{method}_mean_width"] = float(subset[width].mean())
            record[f"{method}_coverage"] = float(subset[covered].eq(True).mean())
        record["fallback_count"] = int(eligible.fallback.eq(True).sum())
        frontier.append(record)
    payload = {"status":"EF20_INDEPENDENT_CORNER_STRESS_POSTHOC_OCTOBER",
               "delta":DELTA,"source_regions":list(source_regions),"target_regions":list(target_regions),
               "scenario_count_including_nominal":len(vectors),"factors_varied_independently":["coal","gas","oil","other"],
               "source_score_rows":len(source_scores),"source_gap_rows":len(source_gaps),
               "availability":av.to_dict("records"),"frontier":frontier,
               "input_hashes":{"calibration_roster":sha(CAL_ROSTER),"target_roster":sha(TARGET_ROSTER),
                               "native":sha(NATIVE),"protocol":sha(PROTOCOL),"script":sha(Path(__file__))},
               "capacity_hashes":capacity_hashes,"hourly_source_hashes":hourly_hashes,"source_audits":audits,
               "protocol_sha256":protocol_hash,
               "limitations":["Post-hoc robustness test on V122 October labels.",
                              "±20% is a sensitivity box, not an empirically justified Chinese factor interval.",
                              "One partial month and U.S. fixed-factor proxies only."]}
    if sha(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed during run")
    (OUT/"summary.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print(pd.DataFrame(frontier).to_string(index=False))


if __name__ == "__main__":
    main()
