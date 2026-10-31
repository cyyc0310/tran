"""V229: sensitivity of factor-span scores to bounded data/EF uncertainty."""
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
import v228_factor_aware_minimum_channels as factor_span

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V229_FACTOR_AND_MEASUREMENT_UNCERTAINTY_PROTOCOL.md"
OUT = ROOT / "results/v229-factor-measurement-uncertainty-r1"
MEASUREMENT_TOLS = (0., .005, .01, .05)
EF_TOLS = (0., .10)
N_LOCAL_DAYS = 12
REPEATS = 100
BOOT = 20_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def day_score_map(d: dict, values: np.ndarray) -> dict[int, float]:
    day_ids = dayrisk.day_ids(pd.DatetimeIndex(d["time"]))
    return {int(day): float(np.max(values[day_ids == day])) for day in np.unique(day_ids)}


def robust_bounds(d: dict, channels: list[str], coefficients: list[float],
                  measurement_tol: float, ef_tol: float) -> tuple[np.ndarray, np.ndarray]:
    fuel = np.asarray(d["fuel_profile"], float) * np.asarray(d["fuel_totals"], float)[None, :]
    margins = np.asarray(d["fuel_totals"], float)
    group_values = np.column_stack([fuel[:, factor_span.GROUPS[name]].sum(axis=1)
                                    for name in channels])
    coeff = np.asarray(coefficients, float)
    n_hat = group_values @ coeff
    g = fuel.sum(axis=1)
    g_lo = g * (1. - measurement_tol)
    g_hi = g * (1. + measurement_tol)
    delta_measure = measurement_tol * (np.abs(group_values) @ np.abs(coeff))
    # Fuel-factor uncertainty is bounded by the largest possible hourly support
    # under monthly margins and the uncertain generation denominator.
    delta_ef = np.zeros(len(fuel), float)
    for j, ef in enumerate(partial.EF):
        if ef == 0.:
            continue
        delta_ef += ef_tol * ef * np.minimum(margins[j], g_hi)
    delta = delta_measure + delta_ef
    n_lo = np.maximum(0., n_hat - delta)
    n_hi = n_hat + delta
    return n_lo / np.maximum(g_hi, 1e-12), n_hi / np.maximum(g_lo, 1e-12)


def source_scores(series: dict, donors: list[str], month: int) -> np.ndarray:
    chunks = []
    for region in donors:
        d = series[(region, 2025, month)]
        vals = np.abs(np.asarray(d["cif"], float) - float(d["monthly_y"]))
        chunks.extend(day_score_map(d, vals).values())
    return np.asarray(chunks, float)


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    protocol_hash = sha256(PROTOCOL)
    native, schemas, input_hashes, helpers = confirm.load_native()
    series = confirm.build_series(native, schemas, helpers)
    dev = set(confirm.DEV_REGIONS)
    eligible, donor_map = [], {}
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
        raise RuntimeError("frozen cohort eligibility failed")

    prepared = {}
    zero_max = 0.
    selection = []
    for key in eligible:
        d = series[key]
        channels, coefficients, span_residual = factor_span.factor_span_fields(d)
        nominal = factor_span.reconstruct_cif(d, channels, coefficients)
        exact = np.asarray(d["cif"], float)
        zero_max = max(zero_max, float(np.max(np.abs(nominal-exact))))
        if zero_max > 1e-8:
            raise RuntimeError(f"zero-uncertainty identity failed at {key}: {zero_max}")
        center = float(d["monthly_y"])
        scores = {}
        for mtol, etol in itertools.product(MEASUREMENT_TOLS, EF_TOLS):
            lo, hi = robust_bounds(d, channels, coefficients, mtol, etol)
            point_score = np.maximum(np.abs(lo-center), np.abs(hi-center))
            scores[(mtol, etol)] = day_score_map(d, point_score)
        full_scores = day_score_map(d, np.abs(exact-center))
        legacy, _, _ = partial.build_day_scores(d, "total_thermal_coal_gas")
        prepared[key] = {"data": d, "channels": channels, "coefficients": coefficients,
                         "scores": scores, "full": full_scores,
                         "legacy": dict(zip(legacy.index.astype(int), legacy.to_numpy(float))),
                         "source": source_scores(series, donor_map[key], key[2])}
        if max(abs(scores[(0.,0.)][day]-full_scores[day]) for day in full_scores) > 1e-8:
            raise RuntimeError(f"zero-uncertainty day score mismatch at {key}")
        selection.append({"region": key[0], "month": key[2], "channels": channels,
                          "coefficients": coefficients, "factor_span_residual": span_residual,
                          "n_numeric_fields_including_total": len(channels)+1})

    methods = [f"robust_m{m:.3f}_ef{e:.2f}" for m,e in itertools.product(MEASUREMENT_TOLS,EF_TOLS)]
    methods += ["exact_factor_span", "legacy_four_field", "full8_local", "source_loo"]
    rng = np.random.default_rng(229)
    records = []
    for key in eligible:
        p = prepared[key]
        d = p["data"]
        days = np.array(sorted(p["full"]), dtype=np.int64)
        d_day_ids = dayrisk.day_ids(pd.DatetimeIndex(d["time"]))
        y = np.asarray(d["cif"], float)
        center = float(d["monthly_y"])
        source_q = dayrisk.radius(p["source"])
        for rep in range(REPEATS):
            selected = rng.choice(days, size=N_LOCAL_DAYS, replace=False)
            test_days = np.setdiff1d(days, selected)
            mask = np.isin(d_day_ids, test_days)
            ytest = y[mask]
            tdays = d_day_ids[mask]
            all_q = {}
            for mtol, etol in itertools.product(MEASUREMENT_TOLS, EF_TOLS):
                m = f"robust_m{mtol:.3f}_ef{etol:.2f}"
                q = dayrisk.radius(np.array([p["scores"][(mtol,etol)][int(day)] for day in selected]))
                all_q[m] = q
            all_q["exact_factor_span"] = dayrisk.radius(np.array([p["full"][int(day)] for day in selected]))
            all_q["legacy_four_field"] = dayrisk.radius(np.array([p["legacy"][int(day)] for day in selected]))
            all_q["full8_local"] = dayrisk.radius(np.array([p["full"][int(day)] for day in selected]))
            all_q["source_loo"] = source_q
            for method, q in all_q.items():
                if not np.isfinite(q):
                    metrics = (1., np.inf, np.inf, 1.)
                else:
                    lo, hi = center-q, center+q
                    inside = (ytest >= lo) & (ytest <= hi)
                    miss = np.maximum(lo-ytest,0.)+np.maximum(ytest-hi,0.)
                    width = hi-lo
                    winkler = width+(2/dayrisk.ALPHA)*miss
                    dcov = np.mean([inside[tdays==day].all() for day in np.unique(tdays)])
                    metrics = (float(inside.mean()),float(width),float(winkler.mean()),float(dcov))
                records.append({"region":key[0],"month":key[2],"repeat":rep,"method":method,
                    "hour_coverage":metrics[0],"width":metrics[1],"winkler90":metrics[2],
                    "simultaneous_day_coverage":metrics[3]})

    reps = pd.DataFrame(records)
    rm = reps.groupby(["region","month","method"],as_index=False).agg(
        hour_coverage=("hour_coverage","mean"),width=("width","mean"),
        winkler90=("winkler90","mean"),simultaneous_day_coverage=("simultaneous_day_coverage","mean"),
        n_repeats=("repeat","count"))
    reg = rm.groupby(["region","method"],as_index=False).agg(
        hour_coverage=("hour_coverage","mean"),width=("width","mean"),
        winkler90=("winkler90","mean"),simultaneous_day_coverage=("simultaneous_day_coverage","mean"),
        n_months=("month","count"))
    summary = reg.groupby("method",as_index=False).agg(
        macro_hour_coverage=("hour_coverage","mean"),macro_width=("width","mean"),
        macro_winkler90=("winkler90","mean"),
        macro_simultaneous_day_coverage=("simultaneous_day_coverage","mean"),
        worst_ba_day_coverage=("simultaneous_day_coverage","min"),
        n_regions=("region","nunique"))
    wide = reg.pivot(index="region",columns="method")
    boot_rng=np.random.default_rng(230)
    comps=[]
    robust_methods=[m for m in methods if m.startswith("robust_")]
    for method in robust_methods:
        for baseline in ("legacy_four_field","full8_local","source_loo"):
            for metric in ("width","winkler90","simultaneous_day_coverage"):
                delta=(wide[metric][method]-wide[metric][baseline]).to_numpy(float)
                sims=np.mean(boot_rng.choice(delta,size=(BOOT,len(delta)),replace=True),axis=1)
                comps.append({"method":method,"baseline":baseline,"metric":metric,
                    "delta_method_minus_baseline":float(delta.mean()),
                    "ci95_low":float(np.quantile(sims,.025)),"ci95_high":float(np.quantile(sims,.975)),
                    "n_regions":len(delta)})
    OUT.mkdir(parents=True)
    reps.to_csv(OUT/"replicate_metrics.csv",index=False)
    rm.to_csv(OUT/"region_month_metrics.csv",index=False)
    reg.to_csv(OUT/"region_metrics.csv",index=False)
    summary.to_csv(OUT/"summary.csv",index=False)
    pd.DataFrame(comps).to_csv(OUT/"paired_ba_bootstrap.csv",index=False)
    pd.DataFrame(selection).to_json(OUT/"channel_selection.json",orient="records",indent=2)
    meta={"protocol_sha256":protocol_hash,"input_hashes":input_hashes,
        "target_ba_months":int(rm[["region","month"]].drop_duplicates().shape[0]),
        "target_regions":sorted(reg.region.unique()),"measurement_tolerances":MEASUREMENT_TOLS,
        "ef_tolerances":EF_TOLS,"local_calibration_days":N_LOCAL_DAYS,"repeats":REPEATS,
        "zero_uncertainty_max_reconstruction_error":zero_max,
        "ef_uncertainty_note":"independent ±10% stress on positive EFs only; not empirical China ranges",
        "measurement_note":"bounded relative errors on total and selected group fields; sensitivity scenario",
        "label_caveat":"EIA fixed-factor generation proxy, not metered emissions",
        "status":"sensitivity analysis on already opened V226 cohort; not blind"}
    (OUT/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    if sha256(PROTOCOL)!=protocol_hash:
        raise RuntimeError("protocol changed during execution")
    print(summary.to_string(index=False))
    print(f"zero_uncertainty_error={zero_max:.3g}; BA_months={meta['target_ba_months']}; output={OUT}")


if __name__ == "__main__":
    main()
