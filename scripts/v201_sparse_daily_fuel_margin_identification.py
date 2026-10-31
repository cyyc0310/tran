#!/usr/bin/env python3
"""V201 sparse daily fuel-margin value for monthly hourly tail-count sets."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v54_unseen_local_ef_confirmation as source_data

OUT = ROOT / "results/v201-sparse-daily-fuel-margin-r1"
PROTOCOL = ROOT / "docs/V201_SPARSE_DAILY_FUEL_MARGIN_IDENTIFICATION_PROTOCOL.md"
CENSUS = ROOT / "results/eia-ba-quality-census-v52-r1/quality_census.json"
REGIONS = ("AVA", "BANC", "BPAT", "CHPD", "DOPD", "EPE", "FMPP", "FPL", "GCPD", "GRID", "GVL", "IID", "JEA", "LDWP", "MISO", "NWMT", "NYIS", "PSEI", "SC", "SCEG", "SCL", "SEC", "TEC", "TIDC", "TPWR")
START = pd.Timestamp("2024-07-01", tz="UTC")
STOP = pd.Timestamp("2026-01-01", tz="UTC")
EF = (1000, 400, 800, 400, 0, 0, 0, 0)
TAUS = (100, 300)
TARGETS = (0.01, 0.05, 0.10)
K_VALUES = (1, 3, 7, 14)
RANDOM_REPLICATES = 50
SEED = 201


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def support_for(fuels: tuple[F, ...]) -> F:
    return F(max((ef for value, ef in zip(fuels, EF) if value > 0), default=0))


def fuel_summary(fuels: tuple[F, ...]) -> tuple[F, F, F]:
    g = sum(fuels, F(0))
    h = sum((amount * ef for amount, ef in zip(fuels, EF)), F(0))
    return h, g, support_for(fuels)


def count_bounds(n: int, fuels: tuple[F, ...], tau: int) -> tuple[int, int]:
    """Exact V87 H/G/M interval; all feasible integer counts are contiguous."""
    if n < 1:
        raise ValueError("count block needs at least one hour")
    h, g, support = fuel_summary(fuels)
    if g <= 0:
        raise ValueError("positive generation required")
    if support <= tau or h == 0:
        return 0, 0
    if h == support * g:
        return n, n
    if h > F(tau) * g:
        return 1, n
    return 0, n - 1


def evenly_spaced(days: int, k: int) -> tuple[int, ...]:
    selected = tuple(int(math.floor((j + 0.5) * days / k)) for j in range(k))
    if len(set(selected)) != k or min(selected) < 0 or max(selected) >= days:
        raise AssertionError((days, k, selected))
    return selected


def metric(bounds: tuple[int, int], actual: int, required: int, n: int) -> dict[str, float | int]:
    lo, hi = bounds
    if not (lo <= actual <= hi):
        raise ValueError("actual K outside identified set")
    positive = lo >= required
    negative = hi < required
    return {
        "lower_K": lo, "upper_K": hi, "span_K": hi - lo,
        "span_fraction": (hi - lo) / n,
        "positive_certificate": int(positive),
        "negative_certificate": int(negative),
        "abstain": int(not positive and not negative),
        "actual_K": actual,
    }


def selected_identification(day_fuels, total_fuels, selected, tau):
    selected_set = set(selected)
    bounds = [count_bounds(24, day_fuels[i], tau) for i in sorted(selected_set)]
    remaining = tuple(total_fuels[j] - sum((day_fuels[i][j] for i in selected_set), F(0))
                       for j in range(len(EF)))
    if any(value < 0 for value in remaining):
        raise ValueError("sampled day margins exceed monthly totals")
    residual_hours = 24 * (len(day_fuels) - len(selected_set))
    if residual_hours:
        bounds.append(count_bounds(residual_hours, remaining, tau))
    return sum(x[0] for x in bounds), sum(x[1] for x in bounds)


def month_records(hourly):
    fuel_cols = [f"generation_{fuel}" for fuel in source_data.FUELS]
    ef_map = dict(zip(("coal", "gas", "oil", "other", "nuclear", "hydro", "wind", "solar"), EF))
    if set(source_data.FUELS) != set(ef_map):
        raise ValueError(f"unexpected fuel roster/order: {source_data.FUELS}")
    rows = []
    quality = []
    for region in REGIONS:
        frame = hourly[region].copy()
        times = pd.to_datetime(frame.time, utc=True)
        frame = frame.loc[times.ge(START) & times.lt(STOP) & frame.valid.astype(bool)].copy()
        frame["time"] = pd.to_datetime(frame.time, utc=True)
        frame["day"] = frame.time.dt.floor("D")
        frame["month"] = pd.to_datetime(frame.time.dt.strftime("%Y-%m-01"), utc=True)
        for month, mg in frame.groupby("month", sort=True):
            valid_days = []
            for day, dg in mg.groupby("day", sort=True):
                dg = dg.sort_values("time")
                if len(dg) == 24 and list(dg.time) == list(pd.date_range(day, periods=24, freq="h")):
                    daily_fuels = [F(0) for _ in EF]
                    actual_by_tau = {tau: 0 for tau in TAUS}
                    for _, row in dg.iterrows():
                        vals = {fuel: F(str(float(row[f"generation_{fuel}"])))
                                for fuel in source_data.FUELS}
                        if any(v < 0 for v in vals.values()) or sum(vals.values(), F(0)) <= 0:
                            raise ValueError("invalid hourly generation")
                        ordered = tuple(vals[f] for f in ("coal", "gas", "oil", "other", "nuclear", "hydro", "wind", "solar"))
                        g = sum(ordered, F(0))
                        h = sum((amount * ef for amount, ef in zip(ordered, EF)), F(0))
                        for tau in TAUS:
                            actual_by_tau[tau] += int(h > F(tau) * g)
                        daily_fuels = [a + b for a, b in zip(daily_fuels, ordered)]
                    valid_days.append((day, tuple(daily_fuels), actual_by_tau))
            quality.append({"region": region, "month": month.isoformat(),
                            "complete_days": len(valid_days), "calendar_days": month.days_in_month,
                            "coverage": len(valid_days) / month.days_in_month})
            if not valid_days:
                continue
            day_fuels = [x[1] for x in valid_days]
            total_fuels = tuple(sum((d[j] for d in day_fuels), F(0)) for j in range(len(EF)))
            actual = {tau: sum(x[2][tau] for x in valid_days) for tau in TAUS}
            n = 24 * len(valid_days)
            monthly_sets = {tau: count_bounds(n, total_fuels, tau) for tau in TAUS}
            full_sets = {tau: (sum(count_bounds(24, d, tau)[0] for d in day_fuels),
                              sum(count_bounds(24, d, tau)[1] for d in day_fuels)) for tau in TAUS}
            for tau in TAUS:
                if not (monthly_sets[tau][0] <= actual[tau] <= monthly_sets[tau][1]) or not (full_sets[tau][0] <= actual[tau] <= full_sets[tau][1]):
                    raise ValueError(f"actual containment failure {region} {month} tau={tau}")
                if not (monthly_sets[tau][0] <= full_sets[tau][0] <= full_sets[tau][1] <= monthly_sets[tau][1]):
                    raise ValueError(f"full daily set not nested in monthly set {region} {month} tau={tau}")
            rows.append({"region": region, "month": month, "day_fuels": day_fuels,
                         "total_fuels": total_fuels, "actual": actual, "n": n,
                         "monthly_sets": monthly_sets, "full_sets": full_sets,
                         "complete_days": len(valid_days)})
    return rows, quality


def sample_schedules(record, k, rng):
    d = record["complete_days"]
    out = {"evenly_spaced": [evenly_spaced(d, k)], "first_days": [tuple(range(k))]}
    out["random"] = [tuple(sorted(rng.choice(d, size=k, replace=False).tolist()))
                     for _ in range(RANDOM_REPLICATES)]
    return out


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    census = json.loads(CENSUS.read_text())
    if tuple(census["quality_qualified_regions_alphabetical"]) != REGIONS:
        raise ValueError("region census mismatch")
    source_data.ALL_REGIONS = tuple(dict.fromkeys(source_data.ALL_REGIONS + REGIONS))
    hourly, monthly, audits, input_hashes = source_data.prepare_regions()
    records, quality = month_records(hourly)
    summaries = []
    rng = np.random.default_rng(SEED)

    for record in records:
        region, month = record["region"], record["month"]
        d, n = record["complete_days"], record["n"]
        context = {"region": region, "month": month.isoformat(), "n_hours": n,
                   "complete_days": d, "monthly_fields": len(EF)}
        for tau in TAUS:
            baseline = record["monthly_sets"][tau]
            for target in TARGETS:
                required = math.ceil(target * n)
                m = metric(baseline, record["actual"][tau], required, n)
                summaries.append({**context, "tau": tau, "target_frequency": target,
                                  "strategy": "monthly_only", "sample_days": 0, "daily_fields": 0,
                                  "replicates": 1, **m})
                full = record["full_sets"][tau]
                fm = metric(full, record["actual"][tau], required, n)
                summaries.append({**context, "tau": tau, "target_frequency": target,
                                  "strategy": "all_daily", "sample_days": d, "daily_fields": 8*d,
                                  "replicates": 1, **fm})
            for k in K_VALUES:
                if k >= d:
                    continue
                schedules = sample_schedules(record, k, rng)
                for strategy, draw_list in schedules.items():
                    draw_metrics = []
                    for selected in draw_list:
                        for tau in TAUS:
                            possible = selected_identification(record["day_fuels"], record["total_fuels"], selected, tau)
                            if not (record["monthly_sets"][tau][0] <= possible[0] <= possible[1] <= record["monthly_sets"][tau][1]):
                                raise ValueError(f"sparse set exceeds monthly set {region} {month} {strategy} k={k}")
                            if not (possible[0] <= record["full_sets"][tau][0] <= record["full_sets"][tau][1] <= possible[1]):
                                raise ValueError(f"sparse set fails to contain full-day set {region} {month} {strategy} k={k}")
                            if not (possible[0] <= record["actual"][tau] <= possible[1]):
                                raise ValueError(f"sparse set misses actual K {region} {month} {strategy} k={k}")
                            for target in TARGETS:
                                required = math.ceil(target * n)
                                draw_metrics.append({"tau": tau, "target_frequency": target,
                                                     **metric(possible, record["actual"][tau], required, n)})
                    draw_frame = pd.DataFrame(draw_metrics)
                    for (tau, target), group in draw_frame.groupby(["tau", "target_frequency"]):
                        if strategy == "random":
                            outrow = {
                                "span_K": group.span_K.mean(), "span_fraction": group.span_fraction.mean(),
                                "span_fraction_p10": group.span_fraction.quantile(.10),
                                "span_fraction_p50": group.span_fraction.quantile(.50),
                                "span_fraction_p90": group.span_fraction.quantile(.90),
                                "positive_certificate": group.positive_certificate.mean(),
                                "negative_certificate": group.negative_certificate.mean(),
                                "abstain": group.abstain.mean(),
                                "lower_K": group.lower_K.mean(), "upper_K": group.upper_K.mean(),
                            }
                        else:
                            outrow = {name: float(group[name].iloc[0]) for name in
                                      ("span_K", "span_fraction", "positive_certificate", "negative_certificate",
                                       "abstain", "lower_K", "upper_K")}
                        summaries.append({**context, "tau": int(tau), "target_frequency": float(target),
                                          "strategy": strategy, "sample_days": k, "daily_fields": 8*k,
                                          "replicates": len(draw_list), **outrow})

    result = pd.DataFrame(summaries)
    OUT.mkdir(parents=True, exist_ok=False)
    result.to_csv(OUT / "region_month_frontier.csv", index=False, float_format="%.10g")
    pd.DataFrame(quality).to_csv(OUT / "quality_audit.csv", index=False, float_format="%.10g")
    # Equal months within each region, then equal weight across regions.
    month_macro = result.groupby(["region", "tau", "target_frequency", "strategy", "sample_days", "daily_fields"], as_index=False).mean(numeric_only=True)
    macro = month_macro.groupby(["tau", "target_frequency", "strategy", "sample_days", "daily_fields"], as_index=False).mean(numeric_only=True)
    macro.to_csv(OUT / "macro_frontier.csv", index=False, float_format="%.10g")
    by_region = month_macro.groupby(["region", "tau", "target_frequency", "strategy", "sample_days", "daily_fields"], as_index=False).mean(numeric_only=True)
    by_region.to_csv(OUT / "region_frontier.csv", index=False, float_format="%.10g")
    payload = {
        "status": "SPARSE_DAILY_FUEL_MARGIN_PARTIAL_IDENTIFICATION_NOT_FORECAST",
        "regions": list(REGIONS), "complete_region_months": len(records),
        "period_start": START.isoformat(), "period_stop_exclusive": STOP.isoformat(),
        "tau": list(TAUS), "target_frequencies": list(TARGETS), "sample_day_budgets": list(K_VALUES),
        "random_replicates": RANDOM_REPLICATES, "daily_fuel_order": ["coal", "gas", "oil", "other", "nuclear", "hydro", "wind", "solar"],
        "emission_factors_g_per_kwh": list(EF), "quality_audit": quality,
        "input_sha256": input_hashes, "census_sha256": sha(CENSUS),
        "protocol_sha256": sha(PROTOCOL), "runner_sha256": sha(Path(__file__)),
        "source_audits": audits,
        "limitations": [
            "Monthly fuel margins are oracle aggregates over the same complete hourly days, not the official release vintage.",
            "All source regions and hourly proxy labels were previously examined; this is development evidence.",
            "Fixed-EF EIA generation proxy is not Chinese measured hourly CIF or forecast coverage.",
            "Daily report value assumes all eight daily fuel totals can be released and reconciled; acquisition cost is not measured.",
            "Identified sets permit arbitrary within-day hourly fuel allocation and impose no capacity, ramp, or exchange constraints."
        ],
        "macro_preview": macro.to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(macro.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
