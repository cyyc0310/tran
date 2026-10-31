"""V222: country-held-out, monthly-margin daily CIF reconstruction benchmark.

The labels are mechanically derived from Carbon Monitor Power daily fuel
generation estimates using fixed fuel emission factors. This is an external
secondary-data stress test, not metered CIF validation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = Path("/tmp/carbonmonitor_power_global.csv")
OUT = ROOT / "results/v222-carbon-monitor-country-coldstart-r4"
FUELS = ("Coal", "Gas", "Oil", "Other sources", "Nuclear", "Hydroelectricity", "Wind", "Solar")
# Match the project's fixed-factor convention; values are gCO2/kWh.
EF = np.array([1000., 400., 800., 400., 0., 0., 0., 0.])
TRAIN_YEARS = tuple(range(2019, 2024))
TEST_YEARS = (2024, 2025)
K = 10
BOOT = 20_000


def bray_curtis(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a-b).sum() / max((a+b).sum(), 1e-12))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load():
    raw = pd.read_csv(INPUT, usecols=["country", "date", "sector", "value"])
    # The published file uses DD/MM/YYYY; explicit parsing avoids locale drift.
    raw["date"] = pd.to_datetime(raw["date"], format="%d/%m/%Y", errors="coerce")
    raw = raw.dropna(subset=["date"])
    raw = raw.loc[raw.sector.isin(FUELS)]
    wide = raw.pivot_table(index=["country", "date"], columns="sector", values="value", aggfunc="mean")
    wide = wide.reindex(columns=FUELS)
    wide = wide.dropna(axis=0, how="any")
    wide = wide.loc[(wide >= 0).all(axis=1) & (wide.sum(axis=1) > 0)].copy()
    wide.columns.name = None
    wide["year"] = wide.index.get_level_values("date").year
    wide["month"] = wide.index.get_level_values("date").month
    wide["weekday"] = wide.index.get_level_values("date").dayofweek
    wide["y"] = wide[list(FUELS)].to_numpy() @ EF / wide[list(FUELS)].sum(axis=1).to_numpy()
    return wide.reset_index(), raw


def source_templates(data: pd.DataFrame, month: int):
    src = data.loc[
        data.year.isin(TRAIN_YEARS)
        & data.month.eq(month)
    ]
    templates = {}
    for country, d in src.groupby("country", sort=True):
        # Equal-weight each training year, then use within-month weekday means.
        annual = []
        for year, dy in d.groupby("year"):
            totals = dy[list(FUELS)].sum().to_numpy(float)
            if totals.sum() > 0:
                annual.append(totals / totals.sum())
        if len(annual) < 3:
            continue
        mix = np.mean(annual, axis=0)
        # Fuel-specific day-of-week profiles. Each column sums to one over a
        # canonical 7-day week; mapping to the actual target month is normalized later.
        grouped = d.groupby("weekday")[list(FUELS)].mean().reindex(range(7), fill_value=0.)
        prof = np.maximum(grouped.to_numpy(float), 0.)
        sums = prof.sum(axis=0)
        prof = np.divide(prof, sums[None, :], out=np.zeros_like(prof), where=sums[None, :] > 0)
        templates[country] = (mix, prof)
    return templates


def seasonal_signatures(data: pd.DataFrame):
    hist = data.loc[data.year.isin(TRAIN_YEARS)]
    signatures = {}
    for country, d in hist.groupby("country", sort=True):
        yearly = []
        for (year, month), dm in d.groupby(["year", "month"]):
            totals = dm[list(FUELS)].sum().to_numpy(float)
            if totals.sum() > 0:
                yearly.append((int(month), totals / totals.sum()))
        month_values = []
        for month in range(1, 13):
            vals = [mix for m, mix in yearly if m == month]
            month_values.append(np.mean(vals, axis=0) if vals else np.full(len(FUELS), np.nan))
        signatures[country] = np.stack(month_values)
    return signatures


def predict(target: pd.DataFrame, templates: dict, mode: str, signatures: dict):
    yfuel = target[list(FUELS)].to_numpy(float)
    totals = yfuel.sum(axis=0)
    mix = totals / max(totals.sum(), 1e-12)
    if mode == "constant":
        return np.full(len(target), float(mix @ EF))

    target_country = target.country.iloc[0]
    if mode == "fuel_shape_season_k10":
        target_sig = signatures.get(target_country)
        def distance(country, donor_mix):
            donor_sig = signatures.get(country)
            if target_sig is None or donor_sig is None:
                return bray_curtis(mix, donor_mix)
            good = np.isfinite(target_sig).all(axis=1) & np.isfinite(donor_sig).all(axis=1)
            return float(np.mean([bray_curtis(target_sig[m], donor_sig[m]) for m in np.flatnonzero(good)]))
    else:
        distance = lambda country, donor_mix: bray_curtis(mix, donor_mix)
    ranked = sorted(((distance(c, tmix), c, prof) for c, (tmix, prof) in templates.items()
                     if c != target_country), key=lambda x: (x[0], x[1]))
    if not ranked:
        return np.full(len(target), float(mix @ EF))
    if mode in ("fuel_shape_k10", "fuel_shape_season_k10"):
        selected = ranked[:min(K, len(ranked))]
        profile = np.mean(np.stack([item[2] for item in selected]), axis=0)
        dayprof = profile[target.weekday.to_numpy(int), :]
        den = dayprof.sum(axis=0)
        share = np.divide(dayprof, den[None, :], out=np.zeros_like(dayprof), where=den[None, :] > 0)
        # A source fuel with no observed temporal support receives uniform allocation.
        no_support = den <= 0
        if no_support.any():
            share[:, no_support] = 1. / len(target)
        forecast_fuel = share * totals[None, :]
        return forecast_fuel @ EF / np.maximum(forecast_fuel.sum(axis=1), 1e-12)
    if mode == "cif_shape_k10":
        selected = ranked[:min(K, len(ranked))]
        # Convert each donor's daily fuel-share profile to an illustrative CIF
        # weekday curve, then rebase exactly to the target monthly CIF mean.
        curves = []
        for _, country, prof in selected:
            donor_mix = templates[country][0]
            donor_weekday_fuels = prof * donor_mix[None, :]
            q = donor_weekday_fuels / np.maximum(donor_weekday_fuels.sum(axis=1, keepdims=True), 1e-12)
            curves.append(q @ EF)
        curve = np.mean(curves, axis=0)
        target_mean = float(mix @ EF)
        curve = curve - curve.mean() + target_mean
        return curve[target.weekday.to_numpy(int)]
    raise ValueError(mode)


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    if not INPUT.exists():
        raise FileNotFoundError(f"download Carbon Monitor Power CSV to {INPUT}")
    OUT.mkdir(parents=True)
    data, raw = load()
    rows = []
    countries = sorted(data.loc[data.year.isin(TEST_YEARS), "country"].unique())
    template_cache = {month: source_templates(data, month) for month in range(1, 13)}
    signatures = seasonal_signatures(data)
    for country in countries:
        for (year, month), target in data.loc[
            data.country.eq(country) & data.year.isin(TEST_YEARS)
        ].groupby(["year", "month"], sort=True):
            # Keep only complete calendar months and all eight fuel categories.
            days = pd.Period(f"{year}-{month:02d}").days_in_month
            if len(target) != days or target.date.nunique() != days:
                continue
            target = target.sort_values("date").reset_index(drop=True)
            if not np.isfinite(target[list(FUELS)].to_numpy(float)).all():
                continue
            templates = template_cache[int(month)]
            if len(templates) < 10:
                continue
            actual = target.y.to_numpy(float)
            preds = {m: predict(target, templates, m, signatures) for m in
                     ("constant", "cif_shape_k10", "fuel_shape_k10", "fuel_shape_season_k10")}
            for method, pred in preds.items():
                for day, (obs, est) in enumerate(zip(actual, pred, strict=True)):
                    rows.append({"country": country, "year": int(year), "month": int(month),
                                 "day": day + 1, "method": method, "actual": obs,
                                 "pred": float(est), "abs_error": abs(float(obs-est)),
                                 "n_source_countries": len(templates)})
    hourly = pd.DataFrame(rows)
    if hourly.empty:
        raise RuntimeError("no complete target country-months survived eligibility")
    hourly.to_csv(OUT / "daily_predictions.csv", index=False)
    cm = hourly.groupby(["country", "year", "month", "method"], as_index=False).agg(
        mae=("abs_error", "mean"), bias=("pred", "mean"), actual_mean=("actual", "mean"),
        n_days=("day", "count"), source_countries=("n_source_countries", "first"))
    cm["mean_bias"] = cm.bias - cm.actual_mean
    cm.to_csv(OUT / "country_month_metrics.csv", index=False)
    country_metrics = cm.groupby(["country", "method"], as_index=False).agg(mae=("mae", "mean"),
        mean_bias=("mean_bias", "mean"), n_months=("month", "count"))
    country_metrics.to_csv(OUT / "country_metrics.csv", index=False)
    overall = country_metrics.groupby("method").agg(macro_country_mae=("mae", "mean"),
        macro_country_bias=("mean_bias", "mean"), n_countries=("country", "nunique")).reset_index()
    overall.to_csv(OUT / "summary.csv", index=False)

    # Paired country-cluster bootstrap for MAE deltas; countries are the
    # independent transfer domains, not daily rows.
    pivot = country_metrics.pivot(index="country", columns="method", values="mae").dropna()
    rng = np.random.default_rng(222)
    comparisons = []
    for candidate in ("cif_shape_k10", "fuel_shape_k10", "fuel_shape_season_k10"):
        delta = (pivot[candidate] - pivot["constant"]).to_numpy(float)
        sims = np.mean(rng.choice(delta, size=(BOOT, len(delta)), replace=True), axis=1)
        comparisons.append({"candidate": candidate, "baseline": "constant", "delta_macro_mae": float(delta.mean()),
                            "ci95_low": float(np.quantile(sims, .025)), "ci95_high": float(np.quantile(sims, .975)),
                            "countries": len(delta), "bootstrap_replicates": BOOT})
    delta = (pivot["fuel_shape_k10"] - pivot["cif_shape_k10"]).to_numpy(float)
    sims = np.mean(rng.choice(delta, size=(BOOT, len(delta)), replace=True), axis=1)
    comparisons.append({"candidate": "fuel_shape_k10", "baseline": "cif_shape_k10", "delta_macro_mae": float(delta.mean()),
                        "ci95_low": float(np.quantile(sims, .025)), "ci95_high": float(np.quantile(sims, .975)),
                        "countries": len(delta), "bootstrap_replicates": BOOT})
    delta = (pivot["fuel_shape_season_k10"] - pivot["fuel_shape_k10"]).to_numpy(float)
    sims = np.mean(rng.choice(delta, size=(BOOT, len(delta)), replace=True), axis=1)
    comparisons.append({"candidate": "fuel_shape_season_k10", "baseline": "fuel_shape_k10", "delta_macro_mae": float(delta.mean()),
                        "ci95_low": float(np.quantile(sims, .025)), "ci95_high": float(np.quantile(sims, .975)),
                        "countries": len(delta), "bootstrap_replicates": BOOT})
    pd.DataFrame(comparisons).to_csv(OUT / "paired_country_bootstrap.csv", index=False)
    metadata = {"input": str(INPUT), "sha256": sha256(INPUT), "bytes": INPUT.stat().st_size,
        "source_url": "https://datas.carbonmonitor.org/API/downloadFullDataset.php?source=energy_global",
        "fuels": list(FUELS), "fixed_EF_gCO2_per_kWh": EF.tolist(), "training_years": list(TRAIN_YEARS),
        "test_years": list(TEST_YEARS), "held_out_country_history_excluded": True,
        "monthly_margins": "target test-month totals by fuel, available post-month; reconstruction only",
        "complete_months": int(cm[["country", "year", "month"]].drop_duplicates().shape[0]),
        "countries": sorted(country_metrics.country.unique()), "raw_source_rows": int(len(raw)),
        "eligible_rows": int(len(data)), "methods": ["constant", "cif_shape_k10", "fuel_shape_k10", "fuel_shape_season_k10"],
        "note": "secondary estimated daily generation labels; no direct metered CIF validation"}
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(overall.to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))
    print(f"country-months={metadata['complete_months']} countries={len(metadata['countries'])} output={OUT}")


if __name__ == "__main__":
    main()
