"""V228: choose local aggregate fields by factor-span identifiability."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import v224_eia_sparse_daily_maxrisk as dayrisk
import v225_partial_snapshot_sharp_risk_bounds as partial
import v226_frozen_four_field_confirmation as confirm

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V228_FACTOR_AWARE_MINIMUM_CHANNEL_PROTOCOL.md"
OUT = ROOT / "results/v228-factor-aware-minimum-channels-r1"
CHANNEL_ORDER = ("thermal", "coal", "oil", "gas", "other")
GROUPS = {
    "thermal": np.array([0, 1, 2, 3]),
    "coal": np.array([0]),
    "gas": np.array([1]),
    "oil": np.array([2]),
    "other": np.array([3]),
}
N_LOCAL_DAYS = 12
REPEATS = 100
BOOT = 20_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def factor_span_fields(d: dict) -> tuple[list[str], list[float], float]:
    totals = np.asarray(d["fuel_totals"], float)
    active = np.flatnonzero(totals > 1e-10)
    ef_active = partial.EF[active]
    for size in range(1, len(CHANNEL_ORDER) + 1):
        for fields in itertools.combinations(CHANNEL_ORDER, size):
            matrix = np.stack([np.isin(active, GROUPS[name]).astype(float)
                               for name in fields], axis=0)
            coefficients, *_ = np.linalg.lstsq(matrix.T, ef_active, rcond=None)
            residual = float(np.max(np.abs(matrix.T @ coefficients - ef_active)))
            if residual <= 1e-9:
                return list(fields), coefficients.tolist(), residual
    raise RuntimeError("no candidate hourly channel subset spans active EF vector")


def reconstruct_cif(d: dict, fields: list[str], coefficients: list[float]) -> np.ndarray:
    totals = np.asarray(d["fuel_totals"], float)
    fuel = np.asarray(d["fuel_profile"], float) * totals[None, :]
    measured = []
    for name in fields:
        measured.append(fuel[:, GROUPS[name]].sum(axis=1))
    emissions = np.column_stack(measured) @ np.asarray(coefficients, float)
    generation = fuel.sum(axis=1)
    return emissions / np.maximum(generation, 1e-12)


def scores_by_day(d: dict, scores: np.ndarray) -> dict[int, float]:
    times = pd.DatetimeIndex(d["time"])
    dayids = dayrisk.day_ids(times)
    return {int(day): float(np.max(scores[dayids == day])) for day in np.unique(dayids)}


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    if not PROTOCOL.exists():
        raise FileNotFoundError(PROTOCOL)
    protocol_hash = sha256(PROTOCOL)
    native, schemas, input_hashes, helpers = confirm.load_native()
    series = confirm.build_series(native, schemas, helpers)
    dev = set(confirm.DEV_REGIONS)
    eligible = []
    donor_map = {}
    for key, d in sorted(series.items()):
        region, year, month = key
        if year != 2026 or month not in (7, 8, 9) or region in dev:
            continue
        if len(np.unique(dayrisk.day_ids(pd.DatetimeIndex(d["time"])))) < 24:
            continue
        donors = [r for r in sorted({k[0] for k in series})
                  if r != region and (r, 2025, month) in series
                  and len(np.unique(dayrisk.day_ids(pd.DatetimeIndex(series[(r, 2025, month)]["time"])))) >= 24]
        if len(donors) >= 10:
            eligible.append(key)
            donor_map[key] = donors
    if len({k[0] for k in eligible}) < 10 or len(eligible) < 30:
        raise RuntimeError(f"frozen cohort gate failed: {len(eligible)} BA-months, {len({k[0] for k in eligible})} BAs")

    prepared = {}
    selection_rows = []
    for key in eligible:
        d = series[key]
        fields, coefficients, residual = factor_span_fields(d)
        reconstructed = reconstruct_cif(d, fields, coefficients)
        max_error = float(np.max(np.abs(reconstructed - np.asarray(d["cif"], float))))
        if max_error > 1e-8:
            raise RuntimeError(f"factor-span identity failed for {key}: {max_error}")
        times = pd.DatetimeIndex(d["time"])
        target_day_ids = np.unique(dayrisk.day_ids(times))
        full_scores = scores_by_day(d, np.abs(np.asarray(d["cif"]) - float(d["monthly_y"])))
        span_scores = scores_by_day(d, np.abs(reconstructed - float(d["monthly_y"])))
        legacy_day, _, _ = partial.build_day_scores(d, "total_thermal_coal_gas")
        three_day, _, _ = partial.build_day_scores(d, "total_thermal_coal")
        if max(abs(span_scores[k] - full_scores[k]) for k in full_scores) > 1e-8:
            raise RuntimeError(f"factor-span day score differs from exact full-8 score for {key}")
        prepared[key] = {"data": d, "span": span_scores, "full": full_scores,
                         "legacy": dict(zip(legacy_day.index.astype(int), legacy_day.to_numpy(float))),
                         "three": dict(zip(three_day.index.astype(int), three_day.to_numpy(float))),
                         "source": np.concatenate([
                             np.array(list(scores_by_day(series[(r, 2025, key[2])], np.abs(
                                 np.asarray(series[(r, 2025, key[2])]["cif"]) -
                                 float(series[(r, 2025, key[2])]["monthly_y"]))).values()))
                             for r in donor_map[key]])}
        selection_rows.append({"region": key[0], "year": key[1], "month": key[2],
            "active_fuels": [confirm.FUELS[i] for i in np.flatnonzero(np.asarray(d["fuel_totals"]) > 1e-10)],
            "selected_channels": fields, "numeric_fields_including_total": len(fields) + 1,
            "span_coefficients": coefficients, "max_factor_span_residual": residual,
            "max_hourly_cif_reconstruction_error": max_error})

    rng = np.random.default_rng(228)
    records = []
    methods = ("factor_span_minimum", "legacy_total_thermal_coal_gas",
               "three_field_total_thermal_coal", "full8_local", "source_loo")
    for key in eligible:
        prepared_month = prepared[key]
        d = prepared_month["data"]
        day_ids = np.array(sorted(prepared_month["span"]), dtype=np.int64)
        d_day_ids = dayrisk.day_ids(pd.DatetimeIndex(d["time"]))
        target_cif = np.asarray(d["cif"], float)
        center = float(d["monthly_y"])
        score_map = {
            "factor_span_minimum": prepared_month["span"],
            "legacy_total_thermal_coal_gas": prepared_month["legacy"],
            "three_field_total_thermal_coal": prepared_month["three"],
            "full8_local": prepared_month["full"],
        }
        q_source = dayrisk.radius(prepared_month["source"])
        for rep in range(REPEATS):
            selected = rng.choice(day_ids, size=N_LOCAL_DAYS, replace=False)
            test_days = np.setdiff1d(day_ids, selected)
            test_mask = np.isin(d_day_ids, test_days)
            y = target_cif[test_mask]
            test_day = d_day_ids[test_mask]
            for method in methods:
                if method == "source_loo":
                    q = q_source
                else:
                    selected_scores = np.array([score_map[method][int(day)] for day in selected])
                    q = dayrisk.radius(selected_scores)
                if not np.isfinite(q):
                    hcov, width, winkler, dcov = 1., np.inf, np.inf, 1.
                else:
                    lo, hi = center - q, center + q
                    inside = (y >= lo) & (y <= hi)
                    miss = np.maximum(lo-y, 0.) + np.maximum(y-hi, 0.)
                    hcov = float(inside.mean())
                    width = float(hi-lo)
                    winkler = float((width + (2/dayrisk.ALPHA)*miss).mean())
                    dcov = float(np.mean([inside[test_day == day].all() for day in np.unique(test_day)]))
                records.append({"region": key[0], "month": key[2], "repeat": rep,
                    "method": method, "hour_coverage": hcov, "width": width,
                    "winkler90": winkler, "simultaneous_day_coverage": dcov,
                    "n_days": len(day_ids), "n_donors": len(donor_map[key])})

    reps = pd.DataFrame(records)
    region_month = reps.groupby(["region", "month", "method"], as_index=False).agg(
        hour_coverage=("hour_coverage", "mean"), width=("width", "mean"),
        winkler90=("winkler90", "mean"), simultaneous_day_coverage=("simultaneous_day_coverage", "mean"),
        n_repeats=("repeat", "count"))
    by_region = region_month.groupby(["region", "method"], as_index=False).agg(
        hour_coverage=("hour_coverage", "mean"), width=("width", "mean"),
        winkler90=("winkler90", "mean"), simultaneous_day_coverage=("simultaneous_day_coverage", "mean"),
        n_months=("month", "count"))
    summary = by_region.groupby("method", as_index=False).agg(
        macro_hour_coverage=("hour_coverage", "mean"), macro_width=("width", "mean"),
        macro_winkler90=("winkler90", "mean"),
        macro_simultaneous_day_coverage=("simultaneous_day_coverage", "mean"),
        n_regions=("region", "nunique"))
    pivot = by_region.pivot(index="region", columns="method")
    boot_rng = np.random.default_rng(229)
    comparisons = []
    for method, baseline in (("factor_span_minimum", "legacy_total_thermal_coal_gas"),
                             ("factor_span_minimum", "full8_local"),
                             ("factor_span_minimum", "source_loo")):
        for metric in ("hour_coverage", "width", "winkler90", "simultaneous_day_coverage"):
            delta = (pivot[metric][method] - pivot[metric][baseline]).to_numpy(float)
            sims = np.mean(boot_rng.choice(delta, size=(BOOT, len(delta)), replace=True), axis=1)
            comparisons.append({"method": method, "baseline": baseline, "metric": metric,
                "delta_method_minus_baseline": float(delta.mean()),
                "ci95_low": float(np.quantile(sims, .025)),
                "ci95_high": float(np.quantile(sims, .975)), "n_regions": len(delta)})
    OUT.mkdir(parents=True)
    reps.to_csv(OUT / "replicate_metrics.csv", index=False)
    region_month.to_csv(OUT / "region_month_metrics.csv", index=False)
    by_region.to_csv(OUT / "region_metrics.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUT / "paired_ba_bootstrap.csv", index=False)
    pd.DataFrame(selection_rows).to_json(OUT / "channel_selection.json", orient="records", indent=2)
    metadata = {"protocol_sha256": protocol_hash, "input_hashes": input_hashes,
        "target_regions": sorted(by_region.region.unique()),
        "target_region_count": int(by_region.region.nunique()),
        "target_ba_months": int(region_month[["region", "month"]].drop_duplicates().shape[0]),
        "target_window": "2026-07 through 2026-09", "local_random_days": N_LOCAL_DAYS,
        "repeats": REPEATS, "methods": list(methods),
        "factor_span_tolerance": 1e-9, "hourly_reconstruction_tolerance": 1e-8,
        "label_caveat": "EIA generation-derived fixed-factor proxy, not metered emissions",
        "status": "method-specific pre-specified reanalysis of V226 outer cohort; not blind"}
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if sha256(PROTOCOL) != protocol_hash:
        raise RuntimeError("frozen protocol changed during run")
    print("field-count histogram:")
    print(pd.Series([x["numeric_fields_including_total"] for x in selection_rows]).value_counts().sort_index().to_string())
    print(summary.to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))
    print(f"protocol_sha256={protocol_hash} BA_months={metadata['target_ba_months']} output={OUT}")


if __name__ == "__main__":
    main()
