"""V227: frozen external temporal replication on NESO GB generation data."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/neso-gb/df_fuel_ckan.csv"
PROTOCOL = ROOT / "docs/V227_NESO_GB_EXTERNAL_TEMPORAL_PROTOCOL.md"
OUT = ROOT / "results/v227-neso-gb-partial-risk-r1"
FUELS = ("coal", "gas", "oil", "other", "nuclear", "hydro", "wind", "solar")
EF = np.array([1000., 400., 800., 400., 0., 0., 0., 0.])
TARGET_YEARS = (2023, 2024)
N_LOCAL_DAYS = 12
REPEATS = 100
ALPHA = .10
BOOT = 20_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def finite_rank_radius(scores: np.ndarray) -> float:
    scores = np.asarray(scores, float)
    rank = math.ceil((len(scores) + 1) * (1 - ALPHA))
    if rank > len(scores):
        return float("inf")
    return float(np.partition(scores, rank - 1)[rank - 1])


def allocate_extreme(indices: np.ndarray, amount: float, lower: np.ndarray,
                     upper: np.ndarray, maximize: bool) -> float:
    indices = np.asarray(indices, int)
    lo = lower[indices]
    hi = upper[indices]
    residual = amount - float(lo.sum())
    if residual < -1e-6 or residual > float((hi - lo).sum()) + 1e-5:
        raise ValueError("hourly thermal total conflicts with monthly fuel margins")
    result = lo.copy()
    order = indices[np.argsort(EF[indices])]
    if maximize:
        order = order[::-1]
    for fuel in order:
        j = int(np.where(indices == fuel)[0][0])
        add = min(residual, float(hi[j] - lo[j]))
        result[j] += add
        residual -= add
    return float(result @ EF[indices])


def sharp_four_field_bounds(fuel: np.ndarray, totals: np.ndarray):
    hourly_total = fuel.sum(axis=1)
    lower = np.maximum(0., totals[None, :] - (totals.sum() - hourly_total)[:, None])
    upper = np.minimum(totals[None, :], hourly_total[:, None])
    out_lo = np.empty(len(fuel))
    out_hi = np.empty(len(fuel))
    for i, total in enumerate(hourly_total):
        coal, gas = fuel[i, 0], fuel[i, 1]
        thermal = fuel[i, :4].sum()
        remainder = thermal - coal - gas
        remaining_fuels = np.array([2, 3])
        lo = coal * EF[0] + gas * EF[1] + allocate_extreme(
            remaining_fuels, remainder, lower[i], upper[i], maximize=False)
        hi = coal * EF[0] + gas * EF[1] + allocate_extreme(
            remaining_fuels, remainder, lower[i], upper[i], maximize=True)
        out_lo[i] = lo / max(total, 1e-12)
        out_hi[i] = hi / max(total, 1e-12)
    return out_lo, out_hi


def load_hourly() -> pd.DataFrame:
    raw = pd.read_csv(DATA, usecols=["DATETIME", "GAS", "COAL", "NUCLEAR", "WIND",
        "WIND_EMB", "HYDRO", "BIOMASS", "OTHER", "SOLAR", "GENERATION"])
    raw["time"] = pd.to_datetime(raw.DATETIME, errors="raise")
    # Use only domestic generation categories; IMPORTS/STORAGE are outside scope.
    raw["coal"] = raw.COAL
    raw["gas"] = raw.GAS
    raw["oil"] = 0.0
    raw["other"] = raw.BIOMASS + raw.OTHER
    raw["nuclear"] = raw.NUCLEAR
    raw["hydro"] = raw.HYDRO
    raw["wind"] = raw.WIND + raw.WIND_EMB
    raw["solar"] = raw.SOLAR
    raw["hour"] = raw.time.dt.floor("h")
    fuel_cols = list(FUELS)
    grouped = raw.groupby("hour", sort=True)[fuel_cols].mean()
    nrows = raw.groupby("hour", sort=True).size()
    grouped = grouped.loc[nrows.eq(2)].copy()
    grouped["day"] = grouped.index.strftime("%Y-%m-%d")
    per_day = grouped.groupby("day", sort=True).size()
    complete_days = set(per_day.index[per_day.eq(24)])
    grouped = grouped.loc[grouped.day.isin(complete_days)]
    if not grouped.index.is_unique or grouped.isna().any().any():
        raise ValueError("hour aggregation is not unique/complete")
    return grouped


def make_months(hourly: pd.DataFrame) -> dict[tuple[int, int], dict]:
    out = {}
    periods = hourly.index.to_period("M")
    for period in periods.unique():
        frame = hourly.loc[periods == period].copy()
        day_counts = frame.groupby("day").size()
        frame = frame.loc[frame.day.isin(day_counts.index[day_counts.eq(24)])]
        if frame.day.nunique() < 20:
            continue
        fuel = frame[list(FUELS)].to_numpy(float)
        totals = fuel.sum(axis=0)
        generation = fuel.sum(axis=1)
        if np.any(fuel < -1e-8) or np.any(generation <= 0):
            raise ValueError("invalid fuel generation")
        cif = fuel @ EF / generation
        anchor = float(totals @ EF / totals.sum())
        lower, upper = sharp_four_field_bounds(fuel, totals)
        day_ids = frame.day.to_numpy()
        records = {d: np.flatnonzero(day_ids == d) for d in sorted(set(day_ids))}
        out[(period.year, period.month)] = {
            "fuel": fuel, "totals": totals, "cif": cif, "anchor": anchor,
            "lower": lower, "upper": upper, "day_ids": day_ids,
            "day_rows": records,
        }
    return out


def day_scores(d: dict, kind: str) -> dict[str, float]:
    if kind == "full8":
        scores = np.abs(d["cif"] - d["anchor"])
    elif kind == "four_field":
        scores = np.maximum(np.abs(d["lower"] - d["anchor"]),
                            np.abs(d["upper"] - d["anchor"]))
    else:
        raise ValueError(kind)
    result = {}
    for day, ix in d["day_rows"].items():
        result[day] = float(np.max(scores[ix]))
    return result


def eval_radius(d: dict, days: list[str], radius: float) -> tuple[float, float, float, float]:
    mask = np.isin(d["day_ids"], days)
    y = d["cif"][mask]
    lo, hi = d["anchor"] - radius, d["anchor"] + radius
    inside = (y >= lo) & (y <= hi)
    miss = np.maximum(lo-y, 0.) + np.maximum(y-hi, 0.)
    score = (hi-lo) + (2/ALPHA) * miss
    test_day_ids = d["day_ids"][mask]
    day_inside = [bool(np.all(inside[test_day_ids == day])) for day in np.unique(test_day_ids)]
    return float(inside.mean()), float(hi-lo), float(score.mean()), float(np.mean(day_inside))


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    if not DATA.exists() or not PROTOCOL.exists():
        raise FileNotFoundError(DATA if not DATA.exists() else PROTOCOL)
    protocol_hash = sha256(PROTOCOL)
    input_hash = sha256(DATA)
    hourly = load_hourly()
    months = make_months(hourly)
    targets = []
    for (year, month), d in sorted(months.items()):
        if year not in TARGET_YEARS or (year == 2024 and month > 3):
            continue
        if len(d["day_rows"]) >= 24 and len(d["day_rows"]) > N_LOCAL_DAYS:
            donors = [months[(yy, month)] for yy in sorted(
                {yy for yy, mm in months if mm == month and yy < year})]
            if len(donors) >= 5:
                targets.append((year, month, d, donors))
    if len(targets) < 12:
        raise RuntimeError(f"frozen target gate failed: only {len(targets)} eligible months")

    rng = np.random.default_rng(227)
    records = []
    for year, month, d, donors in targets:
        days = sorted(d["day_rows"])
        source_score = []
        for donor in donors:
            source_score.extend(day_scores(donor, "full8").values())
        q_source = finite_rank_radius(np.asarray(source_score))
        scores_full = day_scores(d, "full8")
        scores_four = day_scores(d, "four_field")
        for rep in range(REPEATS):
            selected = sorted(rng.choice(days, N_LOCAL_DAYS, replace=False).tolist())
            test = [x for x in days if x not in set(selected)]
            radii = {
                "source_only": q_source,
                "full8_local": finite_rank_radius(np.array([scores_full[x] for x in selected])),
                "four_field_local": finite_rank_radius(np.array([scores_four[x] for x in selected])),
            }
            for method, radius in radii.items():
                hcov, width, winkler, dcov = eval_radius(d, test, radius)
                records.append({"year": year, "month": month, "repeat": rep,
                                "method": method, "hour_coverage": hcov,
                                "width": width, "winkler90": winkler,
                                "simultaneous_day_coverage": dcov,
                                "target_days": len(days), "source_years": len(donors)})
    reps = pd.DataFrame(records)
    month_metrics = reps.groupby(["year", "month", "method"], as_index=False).agg(
        hour_coverage=("hour_coverage", "mean"), width=("width", "mean"),
        winkler90=("winkler90", "mean"),
        simultaneous_day_coverage=("simultaneous_day_coverage", "mean"),
        n_repeats=("repeat", "count"))
    summary = month_metrics.groupby("method", as_index=False).agg(
        macro_hour_coverage=("hour_coverage", "mean"), macro_width=("width", "mean"),
        macro_winkler90=("winkler90", "mean"),
        macro_simultaneous_day_coverage=("simultaneous_day_coverage", "mean"),
        n_target_months=("month", "count"))
    wide = month_metrics.pivot(index=["year", "month"], columns="method")
    rng_boot = np.random.default_rng(228)
    comparisons = []
    for method in ("four_field_local", "full8_local"):
        for baseline in ("source_only", "full8_local"):
            if method == baseline:
                continue
            for metric in ("hour_coverage", "width", "winkler90", "simultaneous_day_coverage"):
                delta = (wide[metric][method] - wide[metric][baseline]).to_numpy(float)
                sims = np.mean(rng_boot.choice(delta, (BOOT, len(delta)), replace=True), axis=1)
                comparisons.append({"method": method, "baseline": baseline, "metric": metric,
                    "delta_method_minus_baseline": float(delta.mean()),
                    "ci95_low": float(np.quantile(sims, .025)),
                    "ci95_high": float(np.quantile(sims, .975)),
                    "n_months": len(delta)})
    OUT.mkdir(parents=True)
    reps.to_csv(OUT / "replicate_metrics.csv", index=False)
    month_metrics.to_csv(OUT / "month_metrics.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUT / "paired_month_bootstrap.csv", index=False)
    metadata = {
        "protocol_sha256": protocol_hash, "input_sha256": input_hash,
        "input_file_bytes": DATA.stat().st_size,
        "target_months": [f"{y:04d}-{m:02d}" for y, m, _, _ in targets],
        "target_month_count": len(targets), "first_hour": str(hourly.index.min()),
        "last_hour": str(hourly.index.max()), "local_calibration_days": N_LOCAL_DAYS,
        "repeats": REPEATS, "methods": ["source_only", "full8_local", "four_field_local"],
        "fuel_order": list(FUELS), "fixed_ef_gco2_kwh": EF.tolist(),
        "estimand": "domestic-generation fixed-EF proxy CIF; imports/storage excluded",
        "label_caveat": "constructed from NESO generation mix and fixed factors, not independently metered emissions",
        "source_vintage_caveat": "NESO cleaned/revised download has no historical publication-vintage archive",
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if sha256(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed during run")
    print(summary.to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))
    print(f"target_months={len(targets)} protocol_sha256={protocol_hash} input_sha256={input_hash}")
    print(f"output={OUT}")


if __name__ == "__main__":
    main()
