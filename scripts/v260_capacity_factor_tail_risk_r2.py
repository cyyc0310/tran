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
             "selected_unambiguous_coal_gas_oil_capacity_mw": float(cap[["coal", "gas", "oil"]].sum(axis=1).sum()),
             "capacity_source": str(path.relative_to(ROOT))}
    return cap, audit


def make_features(monthly: pd.DataFrame, cap: pd.DataFrame, year: int,
                  annual_reference: pd.DataFrame | None = None) -> pd.DataFrame:
    d = monthly.copy()
    d["month"] = pd.to_numeric(d.month, errors="coerce").astype(int)
    d["mwh"] = pd.to_numeric(d.mwh, errors="coerce")
    total = d.groupby(["ba", "month"], observed=True).mwh.sum().rename("total_mwh")
    d = d.join(total, on=["ba", "month"])
    d["share"] = d.mwh / d.total_mwh
    shares = d.pivot_table(index=["ba", "month"], columns="fuel", values="share", aggfunc="sum", fill_value=0)
    shares.columns = [f"share_{x}" for x in shares.columns]
    # Heat-rate features come from the preceding/annual reference, never the target month row.
    annual_data = d if annual_reference is None else annual_reference.copy()
    annual_data["mwh"] = pd.to_numeric(annual_data.mwh, errors="coerce")
    annual_data["mmbtu"] = pd.to_numeric(annual_data.mmbtu, errors="coerce")
    annual = annual_data.groupby(["ba", "fuel"], observed=True).agg(mwh=("mwh", "sum"), mmbtu=("mmbtu", "sum"))
    annual["annual_hr"] = annual.mmbtu / annual.mwh
    annual_hr = annual.annual_hr.unstack("fuel")
    annual_hr.columns = [f"annual_hr_{x}" for x in annual_hr.columns]
    idx = shares.index.union(total.index)
    base = pd.DataFrame(index=idx).join(shares).join(annual_hr, on="ba")
    base["total_mwh"] = total
    # Broad class monthly capacity factors use same-year installed summer capacity.
    d["capacity_class"] = d.fuel.map(CEMS_TO_CAPACITY_CLASS)
    broad = d[d.capacity_class.notna()].groupby(["ba", "month", "capacity_class"], observed=True).mwh.sum().unstack("capacity_class")
    mapped_share = d[d.capacity_class.notna()].groupby(["ba", "month"], observed=True).mwh.sum() / total
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
        mapped_cap = float(c.get("coal", 0.0) + c.get("gas", 0.0) + c.get("oil", 0.0))
        row["log_mapped_fossil_capacity"] = np.log(mapped_cap) if mapped_cap > 0 else np.nan
        row["mapped_generation_share"] = float(mapped_share.get((ba, month), 0.0))
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
    out = ROOT / "results/v260-capacity-factor-tail-risk-r2"
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
    aug_cols = base_cols + [c for c in all_x if c in {"cf_coal", "cf_gas", "cf_oil", "log_mapped_fossil_capacity", "mapped_generation_share"}]
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
    meta = {"protocol": "crosswalk-audited r2; nested alpha-tuned 5-fold GroupKFold inside strict leave-BA-out CV, 2024 source risk q90",
            "source_risk_ba_months": int(len(data)), "source_bas": int(len(ba)),
            "capacity_audits": [audit24, audit25], "base_features": base_cols,
            "augmented_features": aug_cols,
            "cems_to_eia_capacity_crosswalk": CEMS_TO_CAPACITY_CLASS,
            "ambiguous_cems_fuel_categories_excluded_from_capacity_factor": ["Other Gas", "Process Gas", "Wood"],
            "target_2025_not_evaluated": not gate,
            "target_evaluated_only_if_predeclared_source_gate_passes": True,
            "estimand": "CEMS daily maximum absolute residual risk for fossil-fleet gross-load intensity; not BA whole-grid CIF"}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(json.dumps(ci, indent=2))
    print("capacity coverage source BAs", len(set(groups) & set(cap24.index)), "/", len(set(groups)))

    if gate:
        target_monthly = pd.read_csv(ROOT / "results/v253-annual-heat-hourly-interval/target_2025_monthly.csv")
        target_monthly = target_monthly[target_monthly.ba.isin(["FMPP", "GVL", "IID", "JEA", "PSCO", "SC", "SEC", "TAL", "TEC", "TIDC"])].copy()
        target_features = make_features(target_monthly, cap25, 2025, annual_reference=monthly)
        target_x = target_features.reindex(columns=aug_cols)
        for col in target_x:
            if col.startswith("share_") or col.startswith("month_"):
                target_x[col] = target_x[col].fillna(0.0)
        final_log_pred, final_alpha = fit_predict(all_x[aug_cols], y, groups, target_x, tune=True)
        target_risk = pd.DataFrame({"ba": target_x.index.get_level_values("ba"),
                                    "month": target_x.index.get_level_values("month"),
                                    "predicted_log_q90": final_log_pred})
        # Calibrate the predicted log-risk with the 90th percentile of strict LOBA residuals.
        oof = pred[pred.model == "plus_capacity_utilization"].copy()
        oof["residual"] = oof.observed_log_q90 - oof.predicted_log_q90
        correction = float(oof.residual.quantile(.90, interpolation="higher"))
        target_risk["predicted_q90"] = np.exp(target_risk.predicted_log_q90)
        target_risk["source_residual_calibrated_q90"] = np.exp(target_risk.predicted_log_q90 + correction)
        target_risk["source_log_residual_q90_correction"] = correction
        target_risk["final_alpha"] = final_alpha

        target_hourly = pd.read_csv(ROOT / "results/v253-annual-heat-hourly-interval/target_2025_hourly.csv")
        target_hourly = target_hourly[target_hourly.ba.isin(target_monthly.ba.unique())].copy()
        target_hourly["day"] = pd.to_datetime(target_hourly.day).dt.strftime("%Y-%m-%d")
        target_hourly["cif"] = target_hourly.tons / target_hourly.mwh * 907.18474
        target_preds = pd.read_csv(ROOT / "results/v252-annual-local-heat-rate/target_2025_ba_month_predictions.csv")
        target_preds = target_preds[target_preds.ba.isin(target_monthly.ba.unique())].copy()
        target_preds["hourly_mwh"] = target_hourly.groupby(["ba", "month"], observed=True).mwh.sum().reindex(
            pd.MultiIndex.from_frame(target_preds[["ba", "month"]])).to_numpy()
        target_preds["center"] = target_preds.annual_heat_rate_tons / target_preds.hourly_mwh * 907.18474
        center_lookup = target_preds.set_index(["ba", "month"]).center.to_dict()
        daily_records = []
        for (ba_key, month_key, day), row in target_hourly.groupby(["ba", "month", "day"], observed=True):
            if len(row) != 24 or set(row.hour.astype(int)) != set(range(24)):
                continue
            center = float(center_lookup[(ba_key, month_key)])
            score = float(np.max(np.abs(row.cif.to_numpy(dtype=float) - center)))
            daily_records.append({"ba": ba_key, "month": month_key, "day": day, "score": score,
                                  "hourly_cif": row.cif.to_numpy(dtype=float), "center": center})
        daily = pd.DataFrame(daily_records)
        risk_lookup = target_risk.set_index(["ba", "month"])
        v254 = pd.read_csv(SOURCE_DIR / "zero_local_interval_repeats.csv")
        v254 = v254[v254.method == "source_2024_zero_local"].groupby(["ba", "month"], observed=True).q90.mean()
        metrics = []
        for (ba_key, month_key), g in daily.groupby(["ba", "month"], observed=True):
            if (ba_key, month_key) not in risk_lookup.index:
                continue
            risk_row = risk_lookup.loc[(ba_key, month_key)]
            yhours = np.concatenate(g.hourly_cif.to_list())
            center = float(g.center.iloc[0])
            qs = {"capacity_risk_model_uncalibrated": float(risk_row.predicted_q90),
                  "capacity_risk_model_source_calibrated": float(risk_row.source_residual_calibrated_q90)}
            if (ba_key, month_key) in v254.index:
                qs["V254_global_source_q90"] = float(v254.loc[(ba_key, month_key)])
            for method, qhat in qs.items():
                res = yhours - center
                outside = np.abs(res) > qhat
                winkler = np.full(len(res), 2*qhat)
                winkler[res < -qhat] += 20*(-qhat-res[res < -qhat])
                winkler[res > qhat] += 20*(res[res > qhat]-qhat)
                daycov = float((g.score.to_numpy(dtype=float) <= qhat).mean())
                metrics.append({"ba": ba_key, "month": month_key, "method": method,
                                "simultaneous_day_coverage": daycov,
                                "hour_coverage": float((~outside).mean()), "width": 2*qhat,
                                "winkler90": float(winkler.mean()), "q90": qhat,
                                "test_days": len(g)})
        target_eval = pd.DataFrame(metrics)
        target_eval.to_csv(out / "target_2025_interval_evaluation.csv", index=False)
        by_ba = target_eval.groupby(["method", "ba"], observed=True).agg(
            simultaneous_day_coverage=("simultaneous_day_coverage", "mean"),
            hour_coverage=("hour_coverage", "mean"), width=("width", "mean"),
            winkler90=("winkler90", "mean")).reset_index()
        by_ba.to_csv(out / "target_2025_by_ba.csv", index=False)
        target_summary = by_ba.groupby("method", observed=True).agg(
            bas=("ba", "nunique"), simultaneous_day_coverage=("simultaneous_day_coverage", "mean"),
            hour_coverage=("hour_coverage", "mean"), width=("width", "mean"),
            winkler90=("winkler90", "mean")).reset_index()
        target_summary.to_csv(out / "target_2025_summary.csv", index=False)
        target_risk.to_csv(out / "target_2025_predicted_risk.csv", index=False)
        meta.update({"target_2025_evaluated_after_source_gate": True,
                     "target_risk_log_residual_correction": correction,
                     "target_final_model_alpha": final_alpha,
                     "target_scored_ba_months": int(target_eval[["ba", "month"]].drop_duplicates().shape[0])})
        (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        print("\nTarget 2025 interval evaluation:")
        print(target_summary.to_string(index=False))


if __name__ == "__main__":
    main()
