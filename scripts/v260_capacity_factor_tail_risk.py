#!/usr/bin/env python3
"""Leave-BA-out screen of monthly fossil capacity-utilization features for daily tail risk."""

from __future__ import annotations

import calendar
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "results/v254-zero-local-hourly-calibration"
FUEL_CLASSES = {
    "coal": {"ANT", "BIT", "LIG", "RC", "SGC", "SUB", "WC"},
    "gas": {"NG"},
    "oil": {"DFO", "JF", "KER", "PC", "PG", "RFO", "SGP", "WO"},
    "other": {"AB", "BFG", "BLQ", "GEO", "H2", "LFG", "MSW", "OBG", "OBL", "OBS", "OG", "OTH", "PUR", "SLW", "TDF", "WDL", "WDS", "WH"},
}
CEMS_TO_CAPACITY_CLASS = {
    "Coal": "coal", "Coal Refuse": "coal",
    "Natural Gas": "gas", "Pipeline Natural Gas": "gas",
    "Diesel Oil": "oil", "Other Oil": "oil", "Petroleum Coke": "oil", "Residual Oil": "oil",
    "Other Gas": "other", "Process Gas": "other", "Wood": "other",
}
ALPHAS = [0.1, 1.0, 10.0, 100.0]
SEED = 2602026


def capacity_by_ba(path: Path, year: int) -> tuple[pd.DataFrame, dict]:
    with zipfile.ZipFile(path) as z:
        plant_name = next(n for n in z.namelist() if f"Plant_Y{year}" in n and n.endswith(".xlsx"))
        gen_name = next(n for n in z.namelist() if f"Generator_Y{year}" in n and n.endswith(".xlsx"))
        plant = pd.read_excel(io.BytesIO(z.read(plant_name)), header=1,
                              usecols=["Plant Code", "Balancing Authority Code"])
        gen = pd.read_excel(io.BytesIO(z.read(gen_name)), sheet_name="Operable", header=1,
                            usecols=["Plant Code", "Status", "Energy Source 1", "Summer Capacity (MW)"])
    plant["plant"] = pd.to_numeric(plant["Plant Code"], errors="coerce").astype("Int64")
    plant["ba"] = plant["Balancing Authority Code"].astype("string").str.strip().str.upper()
    plant = plant.dropna(subset=["plant", "ba"])
    counts = plant.groupby("plant", observed=True).ba.nunique()
    unique = plant[plant.plant.isin(counts[counts == 1].index)].drop_duplicates("plant")
    plant_map = unique.set_index("plant").ba
    gen["plant"] = pd.to_numeric(gen["Plant Code"], errors="coerce").astype("Int64")
    gen["ba"] = gen.plant.map(plant_map)
    gen["code"] = gen["Energy Source 1"].astype("string").str.strip().str.upper()
    gen["capacity_class"] = gen.code.map({code: group for group, codes in FUEL_CLASSES.items() for code in codes})
    gen["capacity_mw"] = pd.to_numeric(gen["Summer Capacity (MW)"], errors="coerce")
    gen["status"] = gen.Status.astype("string").str.strip().str.upper()
    selected = gen[(gen.status == "OP") & gen.capacity_class.notna() & gen.capacity_mw.gt(0) & gen.ba.notna()].copy()
    cap = selected.pivot_table(index="ba", columns="capacity_class", values="capacity_mw", aggfunc="sum", fill_value=0)
    for group in FUEL_CLASSES:
        if group not in cap:
            cap[group] = 0.0
    cap = cap[list(FUEL_CLASSES)].sort_index()
    cap["fossil_total"] = cap[list(FUEL_CLASSES)].sum(axis=1)
    audit = {"year": year, "plant_rows_unique_ba": int(len(unique)), "unique_ba_count": int(unique.ba.nunique()),
             "generator_rows": int(len(gen)), "selected_op_fossil_generators": int(len(selected)),
             "ba_with_positive_mapped_capacity": int(len(cap)), "selected_summer_capacity_mw": float(cap.fossil_total.sum()),
             "capacity_source": str(path.relative_to(ROOT))}
    return cap, audit


def make_features(monthly: pd.DataFrame, cap: pd.DataFrame, year: int) -> pd.DataFrame:
    d = monthly.copy()
    d["month"] = pd.to_numeric(d.month, errors="coerce").astype(int)
    d["mwh"] = pd.to_numeric(d.mwh, errors="coerce")
    d["mmbtu"] = pd.to_numeric(d.mmbtu, errors="coerce")
    total = d.groupby(["ba", "month"], observed=True).mwh.sum().rename("total_mwh")
    d = d.join(total, on=["ba", "month"])
    d["share"] = d.mwh / d.total_mwh
    d["heat_rate"] = d.mmbtu / d.mwh
    shares = d.pivot_table(index=["ba", "month"], columns="fuel", values="share", aggfunc="sum", fill_value=0)
    shares.columns = [f"share_{x}" for x in shares.columns]
    hr = d.pivot_table(index=["ba", "month"], columns="fuel", values="heat_rate", aggfunc="first")
    hr.columns = [f"hr_{x}" for x in hr.columns]
    # Heat-rate features are annual BA-fuel ratios, not the target month's heat input.
    annual = d.groupby(["ba", "fuel"], observed=True).agg(mwh=("mwh", "sum"), mmbtu=("mmbtu", "sum"))
    annual["annual_hr"] = annual.mmbtu / annual.mwh
    annual_hr = annual.annual_hr.unstack("fuel")
    annual_hr.columns = [f"annual_hr_{x}" for x in annual_hr.columns]
    idx = shares.index.union(hr.index).union(total.index)
    base = pd.DataFrame(index=idx).join(shares).join(annual_hr, on="ba")
    base["total_mwh"] = total
    # Broad class monthly capacity factors use same-year installed summer capacity.
    d["capacity_class"] = d.fuel.map(CEMS_TO_CAPACITY_CLASS)
    broad = d[d.capacity_class.notna()].groupby(["ba", "month", "capacity_class"], observed=True).mwh.sum().unstack("capacity_class")
    cap = cap.reindex(base.index.get_level_values("ba").unique())
    for group in FUEL_CLASSES:
        if group not in broad:
            broad[group] = 0.0
    rows = []
    for (ba, month), _ in base.iterrows():
        hours = calendar.monthrange(year, int(month))[1] * 24
        c = cap.loc[ba] if ba in cap.index else pd.Series(dtype=float)
        b = broad.loc[(ba, month)] if (ba, month) in broad.index else pd.Series(dtype=float)
        row = {}
        for group in FUEL_CLASSES:
            denom = float(c.get(group, 0.0)) * hours
            row[f"cf_{group}"] = float(b.get(group, 0.0)) / denom if denom > 0 else np.nan
        row["log_fossil_capacity"] = np.log(float(c.get("fossil_total", 0.0))) if c.get("fossil_total", 0.0) > 0 else np.nan
        rows.append(row)
    cf = pd.DataFrame(rows, index=base.index)
    base = base.join(cf)
    month_dummies = pd.get_dummies(base.index.get_level_values("month"), prefix="month", dtype=float)
    month_dummies.index = base.index
    base = base.join(month_dummies)
    return base.replace([np.inf, -np.inf], np.nan).sort_index()


def fit_predict(train_x, train_y, train_groups, test_x, tune=True):
    if tune:
        splitter = GroupKFold(n_splits=5)
        scores = {}
        for alpha in ALPHAS:
            errors = []
            for tr, va in splitter.split(train_x, train_y, train_groups):
                keep = train_x.iloc[tr].columns[train_x.iloc[tr].notna().any(axis=0)]
                model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), Ridge(alpha=alpha))
                model.fit(train_x.iloc[tr][keep], train_y.iloc[tr])
                errors.extend(np.abs(train_y.iloc[va] - model.predict(train_x.iloc[va][keep])))
            scores[alpha] = float(np.mean(errors))
        alpha = min(scores, key=scores.get)
    else:
        alpha = 10.0
    keep = train_x.columns[train_x.notna().any(axis=0)]
    model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), Ridge(alpha=alpha))
    model.fit(train_x[keep], train_y)
    return model.predict(test_x[keep]), float(alpha)


def main():
    out = ROOT / "results/v260-capacity-factor-tail-risk"
    out.mkdir(parents=True, exist_ok=True)
    cap24, audit24 = capacity_by_ba(ROOT / "data/research-v87/eia8602024.zip", 2024)
    cap25, audit25 = capacity_by_ba(ROOT / "data/research-v87/eia8602025.zip", 2025)
    monthly = pd.read_csv(SOURCE_DIR / "source_2024_monthly.csv")
    scores = pd.read_csv(SOURCE_DIR / "source_2024_daily_risk_scores.csv")
    features = make_features(monthly, cap24, 2024)
    target = scores.copy()
    target["month"] = pd.to_numeric(target.month, errors="coerce").astype(int)
    # Empirical 90th percentile of daily max absolute residual in each source BA-month.
    q = target.groupby(["ba", "month"], observed=True).score.quantile(.90, interpolation="higher").rename("risk_q90")
    data = features.join(q, how="inner").dropna(subset=["risk_q90"]).copy()
    data["log_q90"] = np.log(data.risk_q90.clip(lower=1e-6))
    all_x = data.drop(columns=["risk_q90", "log_q90"])
    base_cols = [c for c in all_x if c.startswith(("share_", "annual_hr_", "month_"))]
    aug_cols = base_cols + [c for c in all_x if c.startswith(("cf_", "log_fossil_capacity"))]
    y = data.log_q90
    groups = data.index.get_level_values("ba").to_numpy()
    bas = sorted(set(groups))
    rows = []
    for held in bas:
        train = groups != held
        test = ~train
        for model_name, cols in (("month_mix_annual_heat", base_cols), ("plus_capacity_utilization", aug_cols)):
            pred, alpha = fit_predict(all_x.loc[train, cols], y.loc[train], groups[train], all_x.loc[test, cols])
            for (ba, month), yy, pp in zip(data.index[test], y.loc[test].to_numpy(), pred):
                rows.append({"heldout_ba": ba, "month": month, "model": model_name,
                             "observed_log_q90": float(yy), "predicted_log_q90": float(pp),
                             "abs_log_error": float(abs(yy-pp)), "alpha": alpha,
                             "n_train_bas": int(len(set(groups[train]))), "feature_count": len(cols)})
    pred = pd.DataFrame(rows)
    pred.to_csv(out / "leave_ba_out_predictions.csv", index=False)
    piv = pred.pivot(index=["heldout_ba", "month"], columns="model", values="abs_log_error").reset_index()
    piv["base_minus_capacity"] = piv["month_mix_annual_heat"] - piv["plus_capacity_utilization"]
    piv.to_csv(out / "paired_abs_log_error.csv", index=False)
    ba = piv.groupby("heldout_ba", observed=True).base_minus_capacity.mean()
    rng = np.random.default_rng(SEED)
    draws = ba.to_numpy()[rng.integers(0, len(ba), size=(20_000, len(ba)))].mean(axis=1)
    base_mae = float(pred[pred.model == "month_mix_annual_heat"].abs_log_error.mean())
    aug_mae = float(pred[pred.model == "plus_capacity_utilization"].abs_log_error.mean())
    improvement = base_mae - aug_mae
    rel = improvement / base_mae
    positive_bas = int((ba > 0).sum())
    ci_low, ci_high = np.quantile(draws, [.025, .975])
    gate = bool(rel >= .10 and ci_low > 0 and positive_bas / len(ba) >= .60)
    summary = pd.DataFrame([
        {"model": "month_mix_annual_heat", "heldout_bas": len(ba), "ba_months": len(piv), "log_mae": base_mae},
        {"model": "plus_capacity_utilization", "heldout_bas": len(ba), "ba_months": len(piv), "log_mae": aug_mae},
    ])
    summary.to_csv(out / "summary.csv", index=False)
    ci = {"absolute_log_mae_improvement": improvement, "relative_improvement": rel,
          "ba_cluster_bootstrap_95_ci": [float(ci_low), float(ci_high)],
          "bas_improved": positive_bas, "bas_total": int(len(ba)),
          "gate_passed": gate, "gate": "relative >=10%, BA bootstrap lower CI >0, >=60% BAs improve"}
    (out / "gate.json").write_text(json.dumps(ci, indent=2) + "\n")
    pd.DataFrame({"ba": ba.index, "base_minus_aug_abs_log_error": ba.values}).to_csv(out / "by_ba_gain.csv", index=False)
    meta = {"protocol": "nested alpha-tuned 5-fold GroupKFold inside strict leave-BA-out CV, 2024 source risk q90",
            "source_risk_ba_months": int(len(data)), "source_bas": int(len(ba)),
            "capacity_audits": [audit24, audit25], "base_features": base_cols,
            "augmented_features": aug_cols, "target_2025_not_evaluated": not gate,
            "target_evaluated_only_if_predeclared_source_gate_passes": True,
            "estimand": "CEMS daily maximum absolute residual risk for fossil-fleet gross-load intensity; not BA whole-grid CIF"}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(json.dumps(ci, indent=2))
    print("capacity coverage source BAs", len(set(groups) & set(cap24.index)), "/", len(set(groups)))


if __name__ == "__main__":
    main()
