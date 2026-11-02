#!/usr/bin/env python3
"""Select an hourly expert increment by source-BA pseudo-targets (development only)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import run_us_to_gb_2025_official_generation as base

ROOT = base.ROOT
PROTOCOL = ROOT / "docs/CROSS_COUNTRY_GENERATION_PSEUDOTARGET_EXPERTS_PROTOCOL_20261102.md"
TAUS = (np.inf, .1, .25, .5, 1., 2., 5.)
ALPHAS = (0., .25, .5, .75, 1.)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(raw: dict[str, pd.DataFrame], labels: dict[str, pd.DataFrame]):
    prepared, responses = {}, {}
    cols = None
    for ba in base.BAS:
        xframe = base.monthly_features(raw[ba]).dropna(subset=list(base.COARSE)).copy()
        x, cols = base.features(xframe)
        y = labels[ba].set_index("time").reindex(xframe.time).y.to_numpy(float)
        keep = np.isfinite(y)
        prepared[ba] = (xframe.loc[keep].reset_index(drop=True),
                        x.loc[keep].reset_index(drop=True), y[keep])
    return prepared, cols


def collect(prepared, regions, years):
    frames, matrices, targets, ids = [], [], [], []
    for ba in regions:
        frame, x, y = prepared[ba]
        mask = frame.clock.dt.year.isin(years).to_numpy()
        if mask.any():
            frames.append(frame.loc[mask]); matrices.append(x.loc[mask]); targets.append(y[mask])
            ids.extend([ba] * int(mask.sum()))
    if not frames:
        raise ValueError(f"empty training cohort: {regions=}, {years=}")
    frame = pd.concat(frames, ignore_index=True)
    x = pd.concat(matrices, ignore_index=True)
    y = np.concatenate(targets)
    counts = pd.Series(ids).value_counts()
    w = np.asarray([1. / counts[ba] for ba in ids], dtype=float)
    w *= len(w) / w.sum()
    return frame, x, y, w


def fit_hgb(x, y, w, columns):
    model = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15,
        min_samples_leaf=100, l2_regularization=10., random_state=412, early_stopping=False)
    model.fit(x[list(columns)], y, sample_weight=w)
    return model


def prototypes(prepared, regions, years):
    result = {}
    for ba in regions:
        frame, _, _ = prepared[ba]
        subset = frame.loc[frame.clock.dt.year.isin(years)]
        grouped = subset.groupby(subset.clock.dt.month)[list(base.COARSE)].mean()
        overall = grouped.mean().to_numpy(float)
        values = np.tile(overall, (12, 1))
        for month, row in grouped.iterrows():
            values[int(month)-1] = row.to_numpy(float)
        result[ba] = values
    return result


def expert_weights(query, experts, proto, tau):
    if np.isinf(tau):
        return np.full((len(query), len(experts)), 1. / len(experts))
    obs = query[list(base.COARSE)].to_numpy(float)
    months = query.clock.dt.month.to_numpy(int)-1
    exp = np.stack([proto[name][months] for name in experts], axis=1)
    spread = np.stack([np.std(np.stack([proto[name][m] for name in experts]), axis=0)
                       for m in months])
    spread = np.where(spread > 1e-5, spread, 1.)
    d = np.sqrt(np.mean(((exp-obs[:, None, :])/spread[:, None, :])**2, axis=2))
    logits = -d/tau
    logits -= logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    return w / w.sum(axis=1, keepdims=True)


def predict_experts(query, x, experts, models, columns, proto, tau):
    matrix = np.column_stack([models[b].predict(x[list(columns)]) for b in experts])
    w = expert_weights(query, experts, proto, tau)
    return np.sum(matrix*w, axis=1), w


def shape_mae(clock, y, p):
    d = pd.DataFrame({"day": pd.to_datetime(clock).dt.strftime("%Y-%m-%d"), "y": y, "p": p})
    c = d[["y", "p"]] - d.groupby("day")[["y", "p"]].transform("mean")
    return float(np.abs(c.y-c.p).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-labels", type=Path, default=(ROOT / "docs/evidence/neso-national-intensity-2025-r2/neso_national_intensity_history.csv"))
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    raw, labels, source_hashes = base.load_sources()
    gb_raw, target_audit = base.load_gb_inputs()
    prepared, cols = prepare(raw, labels)
    feature_cols = list(cols)
    proto = prototypes(prepared, list(base.BAS), (2022,))

    # Nested LOBA pseudo-target selection: pseudo region is held out; each expert
    # excludes both the pseudo target and its own BA from 2022 fitting.
    selections = []
    for held in base.BAS:
        q, x, y = prepared[held]
        mask = q.clock.dt.year.eq(2023).to_numpy() & q.clock.dt.month.le(6).to_numpy()
        qv, xv, yv = q.loc[mask].reset_index(drop=True), x.loc[mask].reset_index(drop=True), y[mask]
        others = [b for b in base.BAS if b != held]
        f, xx, yy, ww = collect(prepared, others, (2022,))
        pooled = fit_hgb(xx, yy, ww, feature_cols)
        pp = pooled.predict(xv[feature_cols])
        models = {}
        for expert in others:
            train_regions = [b for b in others if b != expert]
            _, xe, ye, we = collect(prepared, train_regions, (2022,))
            models[expert] = fit_hgb(xe, ye, we, feature_cols)
        cache = {tau: predict_experts(qv, xv, others, models, feature_cols, proto, tau)
                 for tau in TAUS}
        for alpha in ALPHAS:
            for tau in TAUS:
                pe, weights = cache[tau]
                pred = pp if alpha == 0 else (1-alpha)*pp + alpha*pe
                selections.append({"pseudo_target": held, "alpha": alpha,
                    "tau": "inf" if np.isinf(tau) else tau,
                    "mae": float(np.mean(np.abs(yv-pred))),
                    "shape_mae": shape_mae(qv.clock, yv, pred),
                    "pooled_mae": float(np.mean(np.abs(yv-pp))),
                    "expert_mae": float(np.mean(np.abs(yv-pe))),
                    "mean_max_expert_weight": float(weights.max(axis=1).mean()), "hours": len(yv)})
        print(f"pseudo-target {held}: {len(yv)} hours", flush=True)
    cv = pd.DataFrame(selections)
    cv["tau_key"] = cv.tau.astype(str)
    summary = cv.groupby(["alpha", "tau_key"], sort=False).mae.mean().sort_values(kind="mergesort")
    selected_alpha, selected_tau = summary.index[0]
    selected_tau = np.inf if selected_tau == "inf" else float(selected_tau)
    cv.to_csv(out/"source_pseudotarget_validation.csv", index=False)
    (out/"selection.json").write_text(json.dumps({"alpha": float(selected_alpha),
        "tau": "inf" if np.isinf(selected_tau) else float(selected_tau),
        "macro_mae": {f"alpha={a}|tau={t}": float(v) for (a,t),v in summary.items()},
        "tie_rule": "stable grid order; alpha=0 first"}, indent=2))

    # Final source model, leave-one-BA expert bank, source calibration and GB forecast.
    train_f, train_x, train_y, train_w = collect(prepared, list(base.BAS), (2022, 2023))
    pooled = fit_hgb(train_x, train_y, train_w, feature_cols)
    experts = {}
    for excluded in base.BAS:
        _, ex, ey, ew = collect(prepared, [b for b in base.BAS if b != excluded], (2022, 2023))
        experts[excluded] = fit_hgb(ex, ey, ew, feature_cols)
    cal_preds = {m: {b: [] for b in base.BAS} for m in base.METHODS}
    cal_truth = {b: [] for b in base.BAS}
    for ba in base.BAS:
        q, x, y = prepared[ba]
        mask = q.clock.dt.year.eq(2024).to_numpy()
        qc, xc, yc = q.loc[mask].reset_index(drop=True), x.loc[mask].reset_index(drop=True), y[mask]
        predicted = {"anchor700": 700*qc.thermal.to_numpy(),
                     "mix_hgb": pooled.predict(xc[feature_cols])}
        pe, _ = predict_experts(qc, xc, list(base.BAS), experts, feature_cols, proto, selected_tau)
        predicted["pseudo_expert"] = pe
        predicted["pseudo_backoff"] = (1-float(selected_alpha))*predicted["mix_hgb"] + float(selected_alpha)*pe
        for method, value in predicted.items():
            cal_preds.setdefault(method, {})[ba] = (yc-value).tolist()
        cal_truth[ba] = yc.tolist()
    method_names = ["anchor700", "mix_hgb", "pseudo_expert", "pseudo_backoff"]
    calibration = {}
    for method in method_names:
        by_ba = [np.asarray(cal_preds[method][ba]) for ba in base.BAS]
        bias = float(np.mean([r.mean() for r in by_ba]))
        values, weights = [], []
        for resid in by_ba:
            absr = np.abs(resid-bias)
            values.extend(absr.tolist()); weights.extend([1/len(base.BAS)/len(absr)]*len(absr))
        order = np.argsort(values); cumulative = np.cumsum(np.asarray(weights)[order])
        radius = float(np.asarray(values)[order[np.searchsorted(cumulative,.9,side="left")]])
        calibration[method] = {"bias": bias, "radius90": radius}

    gb_features = base.monthly_features(gb_raw)
    target = gb_features.loc[gb_features.clock.dt.year.eq(2025)].dropna(subset=list(base.COARSE)).copy().reset_index(drop=True)
    if len(target) != 8760: raise ValueError(f"target GB support mismatch: {len(target)}")
    xt, _ = base.features(target)
    preds = {"anchor700": 700*target.thermal.to_numpy(),
             "mix_hgb": pooled.predict(xt[feature_cols])}
    pe, w = predict_experts(target, xt, list(base.BAS), experts, feature_cols, proto, selected_tau)
    preds["pseudo_expert"] = pe
    preds["pseudo_backoff"] = (1-float(selected_alpha))*preds["mix_hgb"] + float(selected_alpha)*pe
    prediction = pd.DataFrame({"time": target.time, "clock": target.clock,
                               **{k:v for k,v in preds.items()}})
    path = out/"sealed_predictions.csv.gz"; prediction.to_csv(path,index=False,compression="gzip")
    hashes = {str(Path(__file__).resolve()): sha(Path(__file__)), str(PROTOCOL.resolve()): sha(PROTOCOL), **source_hashes,
              str(base.TARGET_FUELS): sha(base.TARGET_FUELS)}
    seal = {"sha256": sha(path), "rows": len(prediction), "input_and_code_hashes": hashes,
            "selected_alpha": float(selected_alpha), "selected_tau": "inf" if np.isinf(selected_tau) else float(selected_tau),
            "target_labels_read_before_seal": False, "target_labels_previously_viewed_in_project": True,
            "target_inputs": "exact t-2 CN5 shares and local calendar; revised NESO archive"}
    (out/"prediction_seal.json").write_text(json.dumps(seal,indent=2))
    (out/"source_calibration.json").write_text(json.dumps(calibration,indent=2))
    (out/"status.json").write_text(json.dumps({"status":"DEVELOPMENT_ONLY_US_BA_PSEUDOTARGET_TO_GB",
        "target_year":2025,"target_country_previously_viewed":True,
        "source_train":[2022,2023],"pseudo_target_validation":"2023H1",
        "source_calibration":2024,"target_input_delay_months":2,
        "target_label_first_read_after_prediction_seal":True},indent=2))

    # Ensure inputs did not move during fitting, then open the already-used target label.
    for filename, digest in hashes.items():
        if sha(Path(filename)) != digest: raise RuntimeError(f"input changed during run: {filename}")
    label_path = args.target_labels
    label = pd.read_csv(label_path, parse_dates=["from"])
    label["time"] = pd.to_datetime(label["from"], utc=True).dt.floor("h")
    label["y"] = pd.to_numeric(label.actual_gco2_per_kwh, errors="coerce")
    hourly = label.groupby("time").y.agg(["mean","count"]).reset_index()
    hourly = hourly.loc[hourly["count"].eq(2)&hourly["mean"].notna(),["time","mean"]].rename(columns={"mean":"y"})
    scored = prediction.merge(hourly,on="time",how="inner",validate="one_to_one")
    rows=[]
    for method in method_names:
        err = scored.y.to_numpy()-(scored[method].to_numpy()+calibration[method]["bias"])
        radius = calibration[method]["radius90"]
        rows.append({"method":method,"hours":len(scored),"mae":float(np.abs(err).mean()),
            "bias":float(err.mean()),"coverage90":float((np.abs(err)<=radius).mean()),
            "width90":2*radius,"winkler90":float(np.mean(2*radius+20*np.maximum(np.abs(err)-radius,0)))})
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
    (out/"target_scoring_hash.json").write_text(json.dumps({str(label_path.resolve()):sha(label_path)},indent=2))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__": main()
