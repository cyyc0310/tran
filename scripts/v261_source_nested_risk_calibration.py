#!/usr/bin/env python3
"""Select risk-residual correction only by nested 2024 leave-BA validation."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v260_capacity_factor_tail_risk_r2 as v260

PS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
SEED = 2612026
TARGETS = ["FMPP", "GVL", "IID", "JEA", "PSCO", "SC", "SEC", "TAL", "TEC", "TIDC"]
TONS_TO_G_KWH = 907.18474


def usable(x: pd.DataFrame, train_index: np.ndarray) -> list[str]:
    return list(x.columns[x.iloc[train_index].notna().any(axis=0)])


def fit_model(x: pd.DataFrame, y: pd.Series, train_idx: np.ndarray, test_idx: np.ndarray, alpha: float):
    cols = usable(x, train_idx)
    model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), Ridge(alpha=alpha))
    model.fit(x.iloc[train_idx][cols], y.iloc[train_idx])
    return model.predict(x.iloc[test_idx][cols])


def tune_alpha(x: pd.DataFrame, y: pd.Series, groups: np.ndarray) -> float:
    split = GroupKFold(n_splits=5)
    losses = {}
    for alpha in v260.ALPHAS:
        err = []
        for tr, va in split.split(x, y, groups):
            pred = fit_model(x, y, tr, va, alpha)
            err.extend(np.abs(y.iloc[va].to_numpy() - pred))
        losses[alpha] = float(np.mean(err))
    return float(min(losses, key=losses.get))


def source_nested_cv(x: pd.DataFrame, y: pd.Series, scores: pd.DataFrame):
    bas = x.index.get_level_values("ba").to_numpy()
    outer = GroupKFold(n_splits=5)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(outer.split(x, y, bas), start=1):
        xtr, ytr, gtr = x.iloc[train_idx], y.iloc[train_idx], bas[train_idx]
        alpha = tune_alpha(xtr, ytr, gtr)
        inner = GroupKFold(n_splits=5)
        residuals = np.full(len(train_idx), np.nan)
        for itr, iva in inner.split(xtr, ytr, gtr):
            pred_inner = fit_model(xtr, ytr, itr, iva, alpha)
            residuals[iva] = ytr.iloc[iva].to_numpy() - pred_inner
        if not np.isfinite(residuals).all():
            raise RuntimeError("Incomplete source-training OOF residuals")
        pred_outer = fit_model(x, y, train_idx, test_idx, alpha)
        held_rows = x.index[test_idx]
        for local_j, ix in enumerate(held_rows):
            ba, month = ix
            day_scores = scores[(scores.ba == ba) & (scores.month == month)].score.to_numpy(dtype=float)
            if len(day_scores) < 20:
                continue
            for p in PS:
                delta = float(np.quantile(residuals, p, method="higher"))
                qhat = float(np.exp(pred_outer[local_j] + delta))
                rows.append({"fold": fold, "ba": ba, "month": int(month), "p": p,
                             "observed_log_q90": float(y.loc[ix]),
                             "predicted_log_q90": float(pred_outer[local_j]),
                             "delta": delta, "qhat": qhat,
                             "daily_coverage": float(np.mean(day_scores <= qhat)),
                             "width": 2*qhat, "n_train_bas": len(set(gtr)), "alpha": alpha})
    return pd.DataFrame(rows)


def summarize_source(rows: pd.DataFrame) -> pd.DataFrame:
    by_ba_month = rows.groupby(["p", "ba", "month"], observed=True).agg(
        daily_coverage=("daily_coverage", "mean"), width=("width", "mean"), qhat=("qhat", "mean")
    ).reset_index()
    by_ba_month.to_csv(ROOT / "results/v261-source-nested-risk-calibration/source_by_ba_month.csv", index=False)
    by_ba = by_ba_month.groupby(["p", "ba"], observed=True).agg(
        daily_coverage=("daily_coverage", "mean"), width=("width", "mean")
    ).reset_index()
    rng = np.random.default_rng(SEED)
    out = []
    for p, g in by_ba_month.groupby("p", observed=True):
        b = by_ba[by_ba.p == p].sort_values("ba")
        cov = b.daily_coverage.to_numpy(dtype=float)
        wid = b.width.to_numpy(dtype=float)
        ci_cov = cov[rng.integers(0, len(cov), size=(20_000, len(cov)))].mean(axis=1)
        ci_w = wid[rng.integers(0, len(wid), size=(20_000, len(wid)))].mean(axis=1)
        out.append({"p": p, "heldout_bas": b.ba.nunique(), "ba_months": g.shape[0],
                    "mean_day_coverage": float(cov.mean()),
                    "coverage_ci_low": float(np.quantile(ci_cov, .025)),
                    "coverage_ci_high": float(np.quantile(ci_cov, .975)),
                    "share_ba_month_coverage_ge85": float((g.daily_coverage >= .85).mean()),
                    "mean_width": float(wid.mean()),
                    "width_ci_low": float(np.quantile(ci_w, .025)),
                    "width_ci_high": float(np.quantile(ci_w, .975)),
                    "bas_at_or_above_90": int((cov >= .90).sum())})
    return pd.DataFrame(out).sort_values("p")


def target_application(x_source, y_source, x_target, p_selected, scores_source):
    groups = x_source.index.get_level_values("ba").to_numpy()
    alpha = tune_alpha(x_source, y_source, groups)
    # Target point predictions use every source BA; calibration uses strict LOBA OOF residuals.
    target_idx = np.arange(len(x_target))
    pred = fit_model(pd.concat([x_source, x_target]), pd.concat([y_source, pd.Series(index=x_target.index, dtype=float)]),
                     np.arange(len(x_source)), np.arange(len(x_source), len(x_source)+len(x_target)), alpha)
    # Rebuild source-only LOBA residuals at the same selected alpha for transparent calibration.
    loo = GroupKFold(n_splits=5)
    residuals = np.full(len(x_source), np.nan)
    for tr, va in loo.split(x_source, y_source, groups):
        residuals[va] = y_source.iloc[va].to_numpy() - fit_model(x_source, y_source, tr, va, alpha)
    if not np.isfinite(residuals).all():
        raise RuntimeError("Incomplete final source OOF residuals")
    delta = float(np.quantile(residuals, p_selected, method="higher"))
    qhat = np.exp(pred + delta)
    target_risk = pd.DataFrame({"ba": x_target.index.get_level_values("ba"),
                                "month": x_target.index.get_level_values("month"),
                                "predicted_log_q90": pred, "qhat": qhat,
                                "selected_source_residual_quantile": p_selected,
                                "source_residual_delta": delta, "alpha": alpha})
    return target_risk, delta, alpha


def evaluate_target(risk: pd.DataFrame):
    hourly = pd.read_csv(ROOT / "results/v253-annual-heat-hourly-interval/target_2025_hourly.csv")
    hourly = hourly[hourly.ba.isin(TARGETS)].copy()
    hourly["day"] = pd.to_datetime(hourly.day).dt.strftime("%Y-%m-%d")
    hourly["cif"] = hourly.tons / hourly.mwh * TONS_TO_G_KWH
    pred = pd.read_csv(ROOT / "results/v252-annual-local-heat-rate/target_2025_ba_month_predictions.csv")
    pred = pred[pred.ba.isin(TARGETS)].copy()
    mh = hourly.groupby(["ba", "month"], observed=True).mwh.sum().rename("hourly_mwh")
    pred = pred.merge(mh, on=["ba", "month"], validate="one_to_one")
    pred["center"] = pred.annual_heat_rate_tons / pred.hourly_mwh * TONS_TO_G_KWH
    centers = pred.set_index(["ba", "month"]).center.to_dict()
    q_lookup = risk.set_index(["ba", "month"]).qhat.to_dict()
    source_q = pd.read_csv(ROOT / "results/v254-zero-local-hourly-calibration/zero_local_interval_repeats.csv")
    source_q = source_q[source_q.method == "source_2024_zero_local"].groupby(["ba", "month"], observed=True).q90.mean().to_dict()
    rows = []
    for (ba, month, day), g in hourly.groupby(["ba", "month", "day"], observed=True):
        if len(g) != 24 or set(g.hour.astype(int)) != set(range(24)):
            continue
        center = float(centers[(ba, month)])
        residual = g.cif.to_numpy(dtype=float) - center
        for method, q in (("capacity_nested_source_calibration", float(q_lookup[(ba, month)])),
                          ("V254_global_source_q90", float(source_q[(ba, month)]))):
            lo, hi = -q, q
            outside = (residual < lo) | (residual > hi)
            loss = np.full(len(residual), 2*q)
            loss[residual < lo] += 20*(lo-residual[residual < lo])
            loss[residual > hi] += 20*(residual[residual > hi]-hi)
            rows.append({"ba": ba, "month": month, "day": day, "method": method,
                         "day_coverage": float(not outside.any()), "hour_coverage": float((~outside).mean()),
                         "width": 2*q, "winkler90": float(loss.mean()), "q90": q})
    daily = pd.DataFrame(rows)
    daily.to_csv(ROOT / "results/v261-source-nested-risk-calibration/target_daily_scores.csv", index=False)
    bm = daily.groupby(["method", "ba", "month"], observed=True).agg(
        day_coverage=("day_coverage", "mean"), hour_coverage=("hour_coverage", "mean"),
        width=("width", "mean"), winkler90=("winkler90", "mean")).reset_index()
    by_ba = bm.groupby(["method", "ba"], observed=True).agg(
        simultaneous_day_coverage=("day_coverage", "mean"), hour_coverage=("hour_coverage", "mean"),
        width=("width", "mean"), winkler90=("winkler90", "mean")).reset_index()
    by_ba.to_csv(ROOT / "results/v261-source-nested-risk-calibration/target_by_ba.csv", index=False)
    summary = by_ba.groupby("method", observed=True).agg(
        bas=("ba", "nunique"), simultaneous_day_coverage=("simultaneous_day_coverage", "mean"),
        hour_coverage=("hour_coverage", "mean"), width=("width", "mean"),
        winkler90=("winkler90", "mean")).reset_index()
    summary.to_csv(ROOT / "results/v261-source-nested-risk-calibration/target_summary.csv", index=False)
    return summary


def main():
    out = ROOT / "results/v261-source-nested-risk-calibration"
    out.mkdir(parents=True, exist_ok=True)
    source_m = pd.read_csv(ROOT / "results/v254-zero-local-hourly-calibration/source_2024_monthly.csv")
    source_scores = pd.read_csv(ROOT / "results/v254-zero-local-hourly-calibration/source_2024_daily_risk_scores.csv")
    cap24, cap_audit = v260.capacity_by_ba(ROOT / "data/research-v87/eia8602024.zip", 2024)
    xall = v260.make_features(source_m, cap24, 2024)
    q = source_scores.groupby(["ba", "month"], observed=True).score.quantile(.90, interpolation="higher").rename("risk_q90")
    data = xall.join(q, how="inner").dropna(subset=["risk_q90"]).sort_index()
    base_cols = [c for c in data if c.startswith(("share_", "annual_hr_", "month_"))]
    aug_cols = base_cols + [c for c in data if c in {"cf_coal", "cf_gas", "cf_oil", "log_mapped_fossil_capacity", "mapped_generation_share"}]
    x, y = data[aug_cols], np.log(data.risk_q90.clip(lower=1e-6))
    cv_rows = source_nested_cv(x, y, source_scores)
    cv_rows.to_csv(out / "nested_source_cv_predictions.csv", index=False)
    summary = summarize_source(cv_rows)
    summary["passes_coverage_gate"] = (summary.mean_day_coverage >= .90) & (summary.share_ba_month_coverage_ge85 >= .80)
    passing = summary[summary.passes_coverage_gate]
    if len(passing):
        selected_p = float(passing.sort_values(["mean_width", "p"]).iloc[0].p)
        selection = "minimum_width_among_passers"
    else:
        selected_p = float(summary.sort_values(["mean_day_coverage", "p"], ascending=[False, True]).iloc[0].p)
        selection = "highest_mean_coverage_no_candidate_passed"
    summary["selected_p"] = summary.p == selected_p
    summary.to_csv(out / "nested_source_cv_summary.csv", index=False)
    (out / "selection.json").write_text(json.dumps({"selected_p": selected_p, "selection_rule": selection,
        "source_gate_passed": bool(len(passing)), "gate": "BA-equal mean day coverage >=90%, >=80% BA-month coverage >=85%",
        "candidate_ps": PS, "outer_folds": 5, "outer_unit": "BA", "source_bas": int(data.index.get_level_values("ba").nunique()),
        "source_ba_months": int(len(data)), "capacity_audit": cap_audit}, indent=2) + "\n")
    print("Nested source CV:")
    print(summary.to_string(index=False))
    if not len(passing):
        print("No source-calibrated level passed; target evaluation skipped by protocol.")
        return

    cap25, _ = v260.capacity_by_ba(ROOT / "data/research-v87/eia8602025.zip", 2025)
    target_m = pd.read_csv(ROOT / "results/v253-annual-heat-hourly-interval/target_2025_monthly.csv")
    target_m = target_m[target_m.ba.isin(TARGETS)].copy()
    txall = v260.make_features(target_m, cap25, 2025, annual_reference=source_m)
    tx = txall.reindex(columns=aug_cols)
    for col in tx:
        if col.startswith("share_") or col.startswith("month_"):
            tx[col] = tx[col].fillna(0.0)
    target_risk, delta, alpha = target_application(x, y, tx, selected_p, source_scores)
    target_risk.to_csv(out / "target_predicted_risk.csv", index=False)
    target_summary = evaluate_target(target_risk)
    print("\nTarget evaluation (labels previously opened):")
    print(target_summary.to_string(index=False))
    meta = {"protocol": "V261 nested source-only risk correction selection; capacity features from corrected V260 r2",
            "selected_p": selected_p, "selection_rule": selection,
            "target_final_source_residual_delta": delta, "target_final_model_alpha": alpha,
            "target_labels_previously_opened": True, "target_ba_months": int(target_risk.shape[0]),
            "estimand": "EPA CEMS measured fossil-fleet gross-load intensity, not BA whole-grid CIF",
            "target_center": "2024 BA annual heat rate with 2025 monthly gross MWh"}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
