"""V279: test adaptive hourly channel budgets under fuel-specific EF support."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

import v224_eia_sparse_daily_maxrisk as dayrisk
import v226_frozen_four_field_confirmation as confirm
import v228_factor_aware_minimum_channels as factor_span

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V279_ASYMMETRIC_EF_SENSOR_FRONTIER_PROTOCOL.md"
EF_SUPPORT = ROOT / "results/v269-asymmetric-empirical-ef-support-r2/cems_asymmetric_support_intervals.csv"
OUT = ROOT / "results/v279-asymmetric-ef-sensor-frontier-r1"
REPEATS = 100
N_LOCAL_DAYS = 12
BOOT = 20_000
FUELS = confirm.FUELS
GROUPS = factor_span.GROUPS
GROUP_MASKS = {name: np.isin(np.arange(len(FUELS)), ids) for name, ids in GROUPS.items()}
ALL_EF = np.array([1000., 400., 800., 400., 0., 0., 0., 0.])
CHANNEL_METHODS = ("point_EF_minimum", "fixed_four", "support_minimum", "full8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def factor_support(quarter: int, table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    q = table[table.quarter == quarter].set_index("fuel_group")
    low, high = ALL_EF.copy(), ALL_EF.copy()
    for fuel_idx, label in ((0, "coal"), (1, "gas"), (2, "oil")):
        low[fuel_idx] = float(q.loc[label, "support_low_g_kwh"])
        high[fuel_idx] = float(q.loc[label, "support_high_g_kwh"])
    return low, high


def select_support_channels(margins: np.ndarray) -> list[str]:
    fields = [name for name, idx in (("coal", 0), ("gas", 1), ("oil", 2))
              if margins[idx] > 1e-10]
    # The project EF for EIA's aggregated other-fossil category is fixed at
    # 400 in this stress test; thermal recovers its hourly amount by residual.
    if margins[3] > 1e-10:
        fields.insert(0, "thermal")
    return fields


def channel_observation(fuel_hour: np.ndarray, name: str) -> float:
    if name == "full8":
        raise ValueError("full8 uses the direct endpoint path")
    return float(fuel_hour[GROUP_MASKS[name]].sum())


def lp_hour_bounds(fuel_hour: np.ndarray, margins: np.ndarray,
                   channels: list[str], ef_low: np.ndarray,
                   ef_high: np.ndarray) -> tuple[float, float]:
    total = float(fuel_hour.sum())
    if total <= 0:
        raise ValueError("hourly generation must be positive")
    monthly_total = float(margins.sum())
    lower = np.maximum(0., margins - (monthly_total - total))
    upper = np.minimum(margins, total)
    rows, values = [np.ones(len(FUELS))], [total]
    for channel in channels:
        rows.append(GROUP_MASKS[channel].astype(float))
        values.append(channel_observation(fuel_hour, channel))
    Aeq, beq = np.asarray(rows), np.asarray(values)
    lo = linprog(ef_low, A_eq=Aeq, b_eq=beq,
                 bounds=list(zip(lower, upper)), method="highs")
    hi = linprog(-ef_high, A_eq=Aeq, b_eq=beq,
                 bounds=list(zip(lower, upper)), method="highs")
    if not lo.success or not hi.success:
        raise RuntimeError(f"LP failed: low={lo.message}; high={hi.message}")
    return float(lo.fun / total), float(-hi.fun / total)


def method_channels(method: str, d: dict) -> list[str] | None:
    margins = np.asarray(d["fuel_totals"], float)
    if method == "point_EF_minimum":
        return factor_span.factor_span_fields(d)[0]
    if method == "fixed_four":
        return ["thermal", "coal", "gas"]
    if method == "support_minimum":
        return select_support_channels(margins)
    return None


def robust_hour_ranges(d: dict, method: str, ef_low: np.ndarray,
                       ef_high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    margins = np.asarray(d["fuel_totals"], float)
    fuel = np.asarray(d["fuel_profile"], float) * margins[None, :]
    if method == "full8":
        denominator = np.maximum(fuel.sum(axis=1), 1e-12)
        return fuel @ ef_low / denominator, fuel @ ef_high / denominator
    channels = method_channels(method, d)
    bounds = [lp_hour_bounds(row, margins, channels, ef_low, ef_high) for row in fuel]
    arr = np.asarray(bounds, float)
    return arr[:, 0], arr[:, 1]


def daily_scores(d: dict, lo: np.ndarray, hi: np.ndarray) -> dict[int, float]:
    ids = dayrisk.day_ids(pd.DatetimeIndex(d["time"]))
    center = float(d["monthly_y"])
    score = np.maximum(np.abs(lo - center), np.abs(hi - center))
    return {int(day): float(np.max(score[ids == day])) for day in np.unique(ids)}


def nominal_source_scores(d: dict) -> np.ndarray:
    ids = dayrisk.day_ids(pd.DatetimeIndex(d["time"]))
    score = np.abs(np.asarray(d["cif"], float) - float(d["monthly_y"]))
    return np.asarray([np.max(score[ids == day]) for day in np.unique(ids)], float)


def score_test(d: dict, lo: np.ndarray, hi: np.ndarray, radius: float,
               test_mask: np.ndarray) -> tuple[float, float, float, float, float]:
    center = float(d["monthly_y"])
    y = np.asarray(d["cif"], float)[test_mask]
    lower, upper = center - radius, center + radius
    nominal_inside = (y >= lower) & (y <= upper)
    misses = np.maximum(lower - y, 0.) + np.maximum(y - upper, 0.)
    width = 2 * radius
    winkler = width + (2 / dayrisk.ALPHA) * misses
    lo_test, hi_test = lo[test_mask], hi[test_mask]
    set_inside = (lo_test >= lower) & (hi_test <= upper)
    ids = dayrisk.day_ids(pd.DatetimeIndex(d["time"]))[test_mask]
    uniq = np.unique(ids)
    day_nominal = np.mean([nominal_inside[ids == day].all() for day in uniq])
    day_set = np.mean([set_inside[ids == day].all() for day in uniq])
    return (float(nominal_inside.mean()), float(width), float(winkler.mean()),
            float(day_nominal), float(day_set))


def score_source_baseline(d: dict, radius: float, test_mask: np.ndarray) -> tuple[float, float, float, float]:
    center = float(d["monthly_y"])
    y = np.asarray(d["cif"], float)[test_mask]
    lower, upper = center - radius, center + radius
    inside = (y >= lower) & (y <= upper)
    misses = np.maximum(lower - y, 0.) + np.maximum(y - upper, 0.)
    ids = dayrisk.day_ids(pd.DatetimeIndex(d["time"]))[test_mask]
    daycov = np.mean([inside[ids == day].all() for day in np.unique(ids)])
    return float(inside.mean()), float(2 * radius), float((2 * radius + (2 / dayrisk.ALPHA) * misses).mean()), float(daycov)


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    protocol_hash = sha256(PROTOCOL)
    support = pd.read_csv(EF_SUPPORT)
    native, schemas, input_hashes, helpers = confirm.load_native()
    series = confirm.build_series(native, schemas, helpers)
    dev = set(confirm.DEV_REGIONS)
    eligible, donor_map = [], {}
    for key, d in sorted(series.items()):
        region, year, month = key
        days = np.unique(dayrisk.day_ids(pd.DatetimeIndex(d["time"])))
        if year != 2026 or month not in (7, 8, 9) or region in dev or len(days) < 24:
            continue
        donors = [r for r in sorted({k[0] for k in series})
                  if r != region and (r, 2025, month) in series
                  and len(np.unique(dayrisk.day_ids(pd.DatetimeIndex(series[(r, 2025, month)]["time"])))) >= 24]
        if len(donors) >= 10:
            eligible.append(key)
            donor_map[key] = donors
    if len(eligible) < 30 or len({k[0] for k in eligible}) < 10:
        raise RuntimeError("V226 cohort completeness gate failed")

    prepared, field_rows = {}, []
    for key in eligible:
        region, year, month = key
        d = series[key]
        quarter = (month - 1) // 3 + 1
        ef_low, ef_high = factor_support(quarter, support)
        method_map = {}
        for method in CHANNEL_METHODS:
            lo, hi = robust_hour_ranges(d, method, ef_low, ef_high)
            method_map[method] = (lo, hi, daily_scores(d, lo, hi))
        # The target nominal proxy must be inside every per-hour support range.
        nominal = np.asarray(d["cif"], float)
        for method, (lo, hi, _) in method_map.items():
            if np.any(lo - 1e-7 > nominal) or np.any(hi + 1e-7 < nominal):
                raise RuntimeError(f"nominal proxy not contained: {key}/{method}")
        source = np.concatenate([nominal_source_scores(series[(donor, 2025, month)])
                                 for donor in donor_map[key]])
        prepared[key] = {"data": d, "methods": method_map, "source": source,
                         "ef_low": ef_low, "ef_high": ef_high}
        for method in CHANNEL_METHODS:
            channels = method_channels(method, d)
            field_rows.append({"region": region, "year": year, "month": month,
                "method": method, "hourly_numeric_fields_including_total": 8 if method == "full8" else 1 + len(channels),
                "channels": "full8" if channels is None else "+".join(channels),
                "quarter": quarter, "ef_low": ef_low.tolist(), "ef_high": ef_high.tolist()})

    rng = np.random.default_rng(279)
    records = []
    for key in eligible:
        p = prepared[key]
        d = p["data"]
        method_days = p["methods"]["full8"][2]
        days = np.asarray(sorted(method_days), dtype=np.int64)
        dayids = dayrisk.day_ids(pd.DatetimeIndex(d["time"]))
        for rep in range(REPEATS):
            sampled = rng.choice(days, size=N_LOCAL_DAYS, replace=False)
            test_days = np.setdiff1d(days, sampled)
            test_mask = np.isin(dayids, test_days)
            for method in CHANNEL_METHODS:
                lo, hi, scores = p["methods"][method]
                q = dayrisk.radius(np.asarray([scores[int(day)] for day in sampled]))
                vals = score_test(d, lo, hi, q, test_mask)
                records.append({"region": key[0], "month": key[2], "repeat": rep,
                    "method": method, "hour_coverage": vals[0], "width": vals[1],
                    "winkler90": vals[2], "simultaneous_day_coverage": vals[3],
                    "robust_day_set_containment": vals[4], "n_days": len(days)})
            q_source = dayrisk.radius(p["source"])
            vals = score_source_baseline(d, q_source, test_mask)
            records.append({"region": key[0], "month": key[2], "repeat": rep,
                "method": "source_loo_nominal", "hour_coverage": vals[0],
                "width": vals[1], "winkler90": vals[2],
                "simultaneous_day_coverage": vals[3],
                "robust_day_set_containment": np.nan, "n_days": len(days)})

    reps = pd.DataFrame(records)
    rm = reps.groupby(["region", "month", "method"], as_index=False).agg(
        hour_coverage=("hour_coverage", "mean"), width=("width", "mean"),
        winkler90=("winkler90", "mean"),
        simultaneous_day_coverage=("simultaneous_day_coverage", "mean"),
        robust_day_set_containment=("robust_day_set_containment", "mean"),
        n_repeats=("repeat", "count"))
    region = rm.groupby(["region", "method"], as_index=False).agg(
        hour_coverage=("hour_coverage", "mean"), width=("width", "mean"),
        winkler90=("winkler90", "mean"),
        simultaneous_day_coverage=("simultaneous_day_coverage", "mean"),
        robust_day_set_containment=("robust_day_set_containment", "mean"),
        n_months=("month", "count"))
    summary = region.groupby("method", as_index=False).agg(
        macro_hour_coverage=("hour_coverage", "mean"), macro_width=("width", "mean"),
        macro_winkler90=("winkler90", "mean"),
        macro_simultaneous_day_coverage=("simultaneous_day_coverage", "mean"),
        macro_robust_day_containment=("robust_day_set_containment", "mean"),
        worst_ba_robust_day_containment=("robust_day_set_containment", "min"),
        n_regions=("region", "nunique"))
    pivot = region.pivot(index="region", columns="method")
    boot_rng = np.random.default_rng(280)
    comparisons = []
    for method, baseline in (("support_minimum", "point_EF_minimum"),
                             ("support_minimum", "fixed_four"),
                             ("support_minimum", "full8"),
                             ("support_minimum", "source_loo_nominal")):
        metrics = ("width", "winkler90", "simultaneous_day_coverage")
        if baseline != "source_loo_nominal":
            metrics += ("robust_day_set_containment",)
        for metric in metrics:
            delta = (pivot[metric][method] - pivot[metric][baseline]).to_numpy(float)
            sims = np.mean(boot_rng.choice(delta, size=(BOOT, len(delta)), replace=True), axis=1)
            comparisons.append({"method": method, "baseline": baseline, "metric": metric,
                "delta_method_minus_baseline": float(delta.mean()),
                "ci95_low": float(np.quantile(sims, .025)),
                "ci95_high": float(np.quantile(sims, .975)), "n_regions": len(delta)})

    OUT.mkdir(parents=True)
    reps.to_csv(OUT / "replicate_metrics.csv", index=False)
    rm.to_csv(OUT / "region_month_metrics.csv", index=False)
    region.to_csv(OUT / "region_metrics.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUT / "paired_ba_bootstrap.csv", index=False)
    pd.DataFrame(field_rows).to_csv(OUT / "field_budget.csv", index=False)
    metadata = {"protocol_sha256": protocol_hash, "script_sha256": sha256(Path(__file__)),
        "ef_support_sha256": sha256(EF_SUPPORT), "input_hashes": input_hashes,
        "target_regions": sorted(region.region.unique()),
        "target_region_count": int(region.region.nunique()),
        "target_ba_months": int(rm[["region", "month"]].drop_duplicates().shape[0]),
        "local_random_days": N_LOCAL_DAYS, "repeats": REPEATS,
        "method_fields": {m: int(round(field_budget)) for m, field_budget in
            pd.DataFrame(field_rows).groupby("method").hourly_numeric_fields_including_total.mean().items()},
        "label_caveat": "EIA generation-derived fixed-factor proxy; empirical CEMS support is cross-system, not target confidence set",
        "status": "fuel-specific EF support sensor frontier on opened V226 cohort; not blind confirmation"}
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    if sha256(PROTOCOL) != protocol_hash:
        raise RuntimeError("frozen protocol changed during run")
    print(summary.to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))
    print(f"BA-months={metadata['target_ba_months']} BAs={metadata['target_region_count']} out={OUT}")


if __name__ == "__main__":
    main()
