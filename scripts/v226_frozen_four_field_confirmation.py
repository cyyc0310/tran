"""V226 frozen confirmation on unseen EIA BAs and future 2026H2 months."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import v224_eia_sparse_daily_maxrisk as dayrisk
import v225_partial_snapshot_sharp_risk_bounds as partial

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v226-frozen-four-field-confirmation-r1"
PROTOCOL = ROOT / "docs/V226_FOUR_FIELD_PARTIAL_RISK_FROZEN_PROTOCOL.md"
SOURCE_FILES = (
    ROOT / "data/native-eia-2025/EIA930_BALANCE_2025_Jan_Jun.csv",
    ROOT / "data/native-eia-2025/EIA930_BALANCE_2025_Jul_Dec.csv",
    ROOT / "data/native-eia-2026/EIA930_BALANCE_2026_Jul_Dec.csv",
)
DEV_REGIONS = (
    "AVA", "BANC", "BPAT", "CHPD", "DOPD", "EPE", "FMPP", "FPL", "GCPD",
    "GRID", "GVL", "IID", "JEA", "LDWP", "MISO", "NWMT", "NYIS", "PSEI",
    "SC", "SCEG", "SCL", "SEC", "TEC", "TIDC", "TPWR",
)
FUELS = ("coal", "gas", "oil", "other", "nuclear", "hydro", "wind", "solar")
EF = np.array([1000., 400., 800., 400., 0., 0., 0., 0.])
METHODS = ("four_field_12day", "three_field_12day", "full8_12day", "source_loo")
ALPHA = .10
REPEATS = 100
BOOT = 20_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_score(d: dict) -> np.ndarray:
    times = pd.DatetimeIndex(d["time"])
    day_id = dayrisk.day_ids(times)
    scores = np.abs(np.asarray(d["cif"], float) - float(d["monthly_y"]))
    return pd.DataFrame({"day": day_id, "score": scores}).groupby("day", sort=True).score.max().to_numpy(float)


def build_series(native: pd.DataFrame, schemas: dict, eia_helpers) -> dict:
    rows = {}
    for region, frame in native.groupby("region", sort=True):
        frame = frame.sort_values("valid_at").copy()
        frame.index = pd.DatetimeIndex(frame.valid_at)
        hourly, _ = eia_helpers._proxy_rows(frame, region, schemas)
        valid = hourly.loc[hourly.valid].copy()
        valid_times = pd.DatetimeIndex(valid.time)
        mats = np.stack([valid[f"generation_{fuel}"].to_numpy(float) for fuel in FUELS], axis=1)
        # Only complete UTC days form the finite target-day population. The
        # month margins and center are computed over the same complete days.
        did = dayrisk.day_ids(valid_times)
        counts = pd.Series(did).value_counts()
        complete_ids = counts.index[counts.eq(24)]
        complete = np.isin(did, complete_ids)
        valid, mats, valid_times, did = valid.loc[complete].copy(), mats[complete], valid_times[complete], did[complete]
        if not len(valid):
            continue
        months = valid_times.to_period("M")
        for month_period in sorted(months.unique()):
            sel = np.asarray(months == month_period)
            month_fuels = mats[sel]
            month_times = valid_times[sel]
            month_days = np.unique(did[sel])
            if len(month_days) < 1:
                continue
            totals = month_fuels.sum(axis=0)
            monthly_total = totals.sum()
            if monthly_total <= 0:
                continue
            cif = month_fuels @ EF / np.maximum(month_fuels.sum(axis=1), 1e-12)
            profile = month_fuels / np.maximum(totals[None, :], 1e-12)
            rows[(str(region), int(month_period.year), int(month_period.month))] = {
                "time": month_times, "cif": cif, "fuel_profile": profile,
                "fuel_totals": totals, "monthly_y": float((totals @ EF) / monthly_total),
            }
    return rows


def load_native():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    from transcif.data.eia930 import normalize_balance
    import v54_unseen_local_ef_confirmation as eia_helpers

    schemas, frames = {}, []
    input_hashes = {}
    for snapshot, path in enumerate(SOURCE_FILES):
        if not path.exists():
            raise FileNotFoundError(path)
        input_hashes[str(path)] = sha256(path)
        raw = pd.read_csv(path, low_memory=False)
        native = normalize_balance(raw)
        native["source_snapshot"] = snapshot
        schemas[str(snapshot)] = {"fuel_columns": [c for c in native if c.startswith("generation_")]}
        frames.append(native)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.loc[
        combined.valid_at.ge(pd.Timestamp("2025-01-01T00:00:00Z"))
        & combined.valid_at.lt(pd.Timestamp("2026-10-01T00:00:00Z"))
    ].copy()
    # Files are disjoint by date; reject overlap instead of silently picking a row.
    if combined.duplicated(["region", "valid_at"]).any():
        raise ValueError("duplicate BA/hour across native source files")
    eia_helpers.schemas = schemas
    return combined, schemas, input_hashes, eia_helpers


def evaluate(d, test_days: np.ndarray, q: float):
    times = pd.DatetimeIndex(d["time"])
    day_id = dayrisk.day_ids(times)
    mask = np.isin(day_id, test_days)
    y = np.asarray(d["cif"], float)[mask]
    if not np.isfinite(q):
        return (1., np.inf, np.inf, 1.)
    center = float(d["monthly_y"])
    lo, hi = center-q, center+q
    inside = (y >= lo) & (y <= hi)
    miss = np.maximum(lo-y, 0.) + np.maximum(y-hi, 0.)
    winkler = (hi-lo) + (2/ALPHA)*miss
    _, inverse = np.unique(day_id[mask], return_inverse=True)
    by_day = np.ones(inverse.max()+1, dtype=bool)
    np.logical_and.at(by_day, inverse, inside)
    return float(inside.mean()), float(hi-lo), float(winkler.mean()), float(by_day.mean())


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    protocol_hash = sha256(PROTOCOL)
    native, schemas, input_hashes, eia_helpers = load_native()
    if sha256(PROTOCOL) != protocol_hash:
        raise RuntimeError("frozen protocol changed during run")
    series = build_series(native, schemas, eia_helpers)
    dev = set(DEV_REGIONS)
    target_keys = sorted(k for k, d in series.items()
                         if k[1] == 2026 and k[2] in (7, 8, 9)
                         and k[0] not in dev
                         and len(np.unique(dayrisk.day_ids(pd.DatetimeIndex(d["time"])))) >= 24)
    eligible = []
    source_index = {}
    for key in target_keys:
        region, _, month = key
        donors = [r for r in sorted({k[0] for k in series})
                  if r != region and (r, 2025, month) in series
                  and len(np.unique(dayrisk.day_ids(pd.DatetimeIndex(series[(r, 2025, month)]["time"])))) >= 24]
        if len(donors) >= 10:
            eligible.append(key)
            source_index[key] = donors
    if len({k[0] for k in eligible}) < 10 or len(eligible) < 30:
        raise RuntimeError(f"frozen eligibility gate failed: {len(set(k[0] for k in eligible))} BAs, {len(eligible)} BA-months")

    rng = np.random.default_rng(226)
    records = []
    for key in eligible:
        region, year, month = key
        d = series[key]
        fields, _, _ = partial.build_day_scores(d, "total_thermal_coal_gas")
        scores3, _, _ = partial.build_day_scores(d, "total_thermal_coal")
        # Full eight-category exact score, with the same day population.
        dscore = source_score(d)
        days = fields.index.to_numpy(np.int64)
        source_pool = np.concatenate([source_score(series[(r, 2025, month)]) for r in source_index[key]])
        q_source = dayrisk.radius(source_pool)
        for rep in range(REPEATS):
            selected = rng.choice(len(days), size=12, replace=False)
            test_days = np.delete(days, selected)
            qs = {
                "four_field_12day": dayrisk.radius(fields.iloc[selected].to_numpy(float)),
                "three_field_12day": dayrisk.radius(scores3.iloc[selected].to_numpy(float)),
                "full8_12day": dayrisk.radius(dscore[selected]),
                "source_loo": q_source,
            }
            for method, q in qs.items():
                hcov, width, winkler, daycov = evaluate(d, test_days, q)
                records.append({"region": region, "month": month, "repeat": rep,
                                "method": method, "hour_coverage": hcov,
                                "width": width, "winkler": winkler,
                                "simultaneous_day_coverage": daycov,
                                "n_days": len(days), "n_donors": len(source_index[key])})

    OUT.mkdir(parents=True)
    reps = pd.DataFrame(records)
    reps.to_csv(OUT / "replicate_metrics.csv", index=False)
    cm = reps.groupby(["region", "month", "method"], as_index=False).agg(
        hour_coverage=("hour_coverage", "mean"), width=("width", "mean"),
        winkler=("winkler", "mean"), simultaneous_day_coverage=("simultaneous_day_coverage", "mean"),
        n_repeats=("repeat", "count"))
    cm.to_csv(OUT / "region_month_metrics.csv", index=False)
    region_metrics = cm.groupby(["region", "method"], as_index=False).agg(
        hour_coverage=("hour_coverage", "mean"), width=("width", "mean"),
        winkler=("winkler", "mean"), simultaneous_day_coverage=("simultaneous_day_coverage", "mean"),
        n_months=("month", "count"))
    region_metrics.to_csv(OUT / "region_metrics.csv", index=False)
    summary = region_metrics.groupby("method", as_index=False).agg(
        macro_hour_coverage=("hour_coverage", "mean"), macro_width=("width", "mean"),
        macro_winkler=("winkler", "mean"), macro_simultaneous_day_coverage=("simultaneous_day_coverage", "mean"),
        n_regions=("region", "nunique"))
    summary.to_csv(OUT / "summary.csv", index=False)
    rng = np.random.default_rng(227)
    comparisons = []
    piv = region_metrics.pivot(index="region", columns="method")
    for method in ("four_field_12day", "three_field_12day"):
        for baseline in ("source_loo", "full8_12day"):
            for metric in ("hour_coverage", "width", "winkler", "simultaneous_day_coverage"):
                delta = (piv[metric][method]-piv[metric][baseline]).to_numpy(float)
                sims = np.mean(rng.choice(delta, size=(BOOT, len(delta)), replace=True), axis=1)
                comparisons.append({"method": method, "baseline": baseline, "metric": metric,
                                    "delta_method_minus_baseline": float(delta.mean()),
                                    "ci95_low": float(np.quantile(sims, .025)),
                                    "ci95_high": float(np.quantile(sims, .975)), "regions": len(delta)})
    pd.DataFrame(comparisons).to_csv(OUT / "paired_region_bootstrap.csv", index=False)
    metadata = {"protocol_sha256": protocol_hash, "input_hashes": input_hashes,
                "source_year": 2025, "target_months": ["2026-07", "2026-08", "2026-09"],
                "development_roster_excluded": list(DEV_REGIONS),
                "target_regions": sorted(region_metrics.region.unique()),
                "target_region_count": int(region_metrics.region.nunique()),
                "target_region_months": int(cm[["region", "month"]].drop_duplicates().shape[0]),
                "target_day_population": "complete UTC days passing native EIA source-quality screen",
                "monthly_margin_population": "same complete UTC days as target sampling/evaluation population",
                "fields_primary": ["hourly total", "hourly thermal total", "hourly coal", "hourly gas"],
                "local_random_days": 12, "repeats": REPEATS,
                "source_donor_counts": {f"{r}-{m:02d}": len(ds) for (r,y,m), ds in source_index.items()},
                "proxy_caveat": "EIA hourly fuel generation and fixed EF; not metered emissions"}
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print("cohort", metadata["target_region_count"], "BAs", metadata["target_region_months"], "BA-months")
    print(summary.to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))
    print(f"protocol_sha256={protocol_hash} output={OUT}")


if __name__ == "__main__":
    main()
