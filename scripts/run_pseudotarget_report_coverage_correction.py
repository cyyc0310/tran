"""Learn a cross-fitted EIA-923-to-complete-mix correction and test GB transfer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import helmert
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import audit_eia923_ba_mix_alignment as eia_audit
import run_eia923_pseudotarget_transfer as eia_parser
import run_gb_et54_official_input_transfer as gb_shared
from transcif.research.cross_country import (
    COARSE, FUELS, METHODS, EF, fit_models, labels, metrics, monthly_inputs,
    predict_models,
)

ROOT = Path(__file__).resolve().parents[1]
BAS = ("BPAT", "CISO", "ERCO", "FPL", "ISNE", "MISO", "NYIS")
YEARS_TRAIN = (2022, 2023)
ALPHAS = (0.1, 1.0, 10.0, 100.0)
ILR_BASIS = helmert(len(COARSE), full=False)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close_shares(values: np.ndarray, floor: float = 1e-4) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.maximum(values, floor)
    return values / values.sum(axis=-1, keepdims=True)


def ilr(values: np.ndarray) -> np.ndarray:
    p = close_shares(values)
    logp = np.log(p)
    return logp @ ILR_BASIS.T


def inv_ilr(values: np.ndarray) -> np.ndarray:
    logp = np.asarray(values, dtype=float) @ ILR_BASIS
    logp -= logp.max(axis=-1, keepdims=True)
    exp = np.exp(logp)
    return exp / exp.sum(axis=-1, keepdims=True)


def fit_mapping(train: pd.DataFrame, alpha: float):
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    x = ilr(train[[f"reported_{c}" for c in COARSE]].to_numpy())
    y = ilr(train[[f"complete_{c}" for c in COARSE]].to_numpy())
    # Balance BA contributions when some official hourly months are incomplete.
    counts = train.groupby("ba").ba.transform("size").to_numpy(float)
    weights = 1.0 / counts
    weights *= len(weights) / weights.sum()
    model.fit(x, y, ridge__sample_weight=weights)
    return model


def predict_mapping(model, frame: pd.DataFrame) -> np.ndarray:
    x = ilr(frame[[f"reported_{c}" for c in COARSE]].to_numpy())
    return inv_ilr(model.predict(x))


def share_mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.abs(np.asarray(y) - np.asarray(p)).sum(axis=1).mean())


def composition_pairs() -> pd.DataFrame:
    report = eia_parser.parse_monthly((2022, 2023, 2024), BAS)
    report["year"] = report.observed_month.str[:4].astype(int)
    report["month"] = report.observed_month.str[5:7].astype(int)
    reported_cols = list(COARSE)
    # parser has thermal plus the remaining four coarse shares.
    report = report[["ba", "observed_month", "year", "month", *reported_cols]].rename(
        columns={c: f"reported_{c}" for c in COARSE})

    full = pd.concat([eia_audit.parse_eia930(
        ROOT / "data/eia930-hourly-co2-r1/feature_inputs", BAS, year)
        for year in (2022, 2023, 2024)], ignore_index=True)
    full["year"] = full.month.str[:4].astype(int)
    full["month_num"] = full.month.str[5:7].astype(int)
    totals = np.column_stack([
        sum(full[f"eia930_{name}_mwh"] for name in ("coal", "gas", "oil", "other")),
        full["eia930_hydro_mwh"], full["eia930_nuclear_mwh"],
        full["eia930_wind_mwh"], full["eia930_solar_mwh"],
    ]).astype(float)
    valid = np.isfinite(totals).all(axis=1) & (totals >= 0).all(axis=1) & (totals.sum(axis=1) > 0)
    full_shares = totals / totals.sum(axis=1, keepdims=True)
    complete = full[["ba", "month", "year", "month_num"]].copy()
    for j, name in enumerate(COARSE):
        complete[f"complete_{name}"] = np.where(valid, full_shares[:, j], np.nan)
    complete = complete.rename(columns={"month": "observed_month"})
    pairs = report.merge(complete, on=["ba", "observed_month", "year"], how="inner")
    pairs = pairs.dropna(subset=[*(f"complete_{c}" for c in COARSE),
                                 *(f"reported_{c}" for c in COARSE)]).copy()
    pairs = pairs.loc[pairs[[*(f"complete_{c}" for c in COARSE),
                             *(f"reported_{c}" for c in COARSE)]].ge(0).all(axis=1)]
    pairs = pairs.loc[pairs[[*(f"complete_{c}" for c in COARSE),
                             *(f"reported_{c}" for c in COARSE)]].sum(axis=1).gt(0)]
    for prefix in ("reported", "complete"):
        cols = [f"{prefix}_{c}" for c in COARSE]
        pairs[cols] = pairs[cols].div(pairs[cols].sum(axis=1), axis=0)
    return pairs.reset_index(drop=True)


def cv_select_alpha(pairs: pd.DataFrame, bas: tuple[str, ...]) -> tuple[str, pd.DataFrame]:
    train = pairs.loc[pairs.ba.isin(bas) & pairs.year.isin(YEARS_TRAIN)].copy()
    records = []
    for alpha in ALPHAS:
        fold_scores = []
        for held_ba in bas:
            valid = train.loc[train.ba.eq(held_ba)]
            fit = train.loc[~train.ba.eq(held_ba)]
            if fit.empty or valid.empty:
                continue
            pred = predict_mapping(fit_mapping(fit, alpha), valid)
            truth = valid[[f"complete_{c}" for c in COARSE]].to_numpy(float)
            fold_scores.append({"held_ba": held_ba, "alpha": alpha,
                                "l1": share_mae(truth, pred)})
        score = float(np.mean([row["l1"] for row in fold_scores]))
        records.append({"candidate": f"ridge_alpha_{alpha:g}", "alpha": alpha,
                        "cv_ba_equal_l1": score,
                        "folds": len(fold_scores)})
    identity_scores = []
    for held_ba in bas:
        valid = train.loc[train.ba.eq(held_ba)]
        if len(valid):
            identity_scores.append({"held_ba": held_ba, "alpha": "identity",
                                    "l1": share_mae(
                                        valid[[f"complete_{c}" for c in COARSE]].to_numpy(float),
                                        valid[[f"reported_{c}" for c in COARSE]].to_numpy(float))})
    identity_score = float(np.mean([row["l1"] for row in identity_scores]))
    records.append({"candidate": "identity", "alpha": "identity",
                    "cv_ba_equal_l1": identity_score,
                    "folds": len(identity_scores)})
    table = pd.DataFrame(records).sort_values("cv_ba_equal_l1", kind="mergesort")
    best = table.iloc[0]
    if best.candidate != "identity" and float(best.cv_ba_equal_l1) >= identity_score:
        return "identity", table
    return str(best.alpha) if best.candidate != "identity" else "identity", table


def fitted_mapping(train: pd.DataFrame, alpha_choice: str):
    if alpha_choice == "identity":
        return None
    return fit_mapping(train, float(alpha_choice))


def apply_mapping(model, frame: pd.DataFrame) -> np.ndarray:
    if model is None:
        return close_shares(frame[[f"reported_{c}" for c in COARSE]].to_numpy())
    return predict_mapping(model, frame)


def outer_pseudotarget_cv(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions, selections = [], []
    for held_ba in BAS:
        train_bas = tuple(ba for ba in BAS if ba != held_ba)
        fit_pool = pairs.loc[pairs.ba.isin(train_bas) & pairs.year.isin(YEARS_TRAIN)].copy()
        choice, cv_table = cv_select_alpha(pairs, train_bas)
        test = pairs.loc[pairs.ba.eq(held_ba) & pairs.year.eq(2024)].copy()
        model = fitted_mapping(fit_pool, choice)
        pred = apply_mapping(model, test)
        truth = test[[f"complete_{c}" for c in COARSE]].to_numpy(float)
        raw = test[[f"reported_{c}" for c in COARSE]].to_numpy(float)
        for i, row in enumerate(test.itertuples(index=False)):
            record = {"ba": held_ba, "observed_month": row.observed_month,
                      "chosen_alpha": choice,
                      "raw_l1": float(np.abs(raw[i] - truth[i]).sum()),
                      "corrected_l1": float(np.abs(pred[i] - truth[i]).sum())}
            record.update({f"truth_{c}": truth[i, j] for j, c in enumerate(COARSE)})
            record.update({f"reported_{c}": raw[i, j] for j, c in enumerate(COARSE)})
            record.update({f"corrected_{c}": pred[i, j] for j, c in enumerate(COARSE)})
            predictions.append(record)
        selections.append({"outer_ba": held_ba, "selected_alpha": choice,
                           "train_pairs": len(fit_pool), "test_pairs": len(test),
                           "inner_cv": cv_table.to_dict("records")})
    return pd.DataFrame(predictions), pd.DataFrame(selections)


def source_hour_models():
    raw = gb_shared.load_us_source()
    prepared = {ba: monthly_inputs(frame) for ba, frame in raw.items()}
    usable = {ba: frame.dropna(subset=list(COARSE)).copy() for ba, frame in prepared.items()}
    train_parts = [usable[ba].loc[usable[ba].year.eq(2022)] for ba in raw]
    train = pd.concat(train_parts)
    fuel = np.concatenate([raw[ba].loc[part.index, list(FUELS)].to_numpy(float)
                           for ba, part in zip(raw, train_parts)])
    models = fit_models(train, fuel)
    residuals = {method: [] for method in METHODS}
    for ba in raw:
        cal = usable[ba].loc[usable[ba].year.eq(2023)]
        pred = predict_models(models, cal)
        y = labels(raw[ba].loc[cal.index])
        for method in METHODS:
            residuals[method].append(y - pred[method])
    calibration = {}
    for method, values in residuals.items():
        bias = float(np.mean([v.mean() for v in values]))
        sigma = float(np.sqrt(np.mean([np.mean((v - bias) ** 2) for v in values])))
        calibration[method] = (bias, max(sigma, 1.))
    return models, calibration, raw


def eia930_full_monthly_from_gb(path: Path) -> pd.DataFrame:
    use = ["DATETIME", "GAS", "COAL", "NUCLEAR", "WIND", "WIND_EMB",
           "HYDRO", "BIOMASS", "OTHER", "SOLAR"]
    raw = pd.read_csv(path, usecols=use)
    raw["stamp"] = pd.to_datetime(raw.DATETIME, errors="raise")
    raw = raw.loc[raw.stamp.dt.year.between(2022, 2024)].copy()
    raw["thermal"] = raw.GAS + raw.COAL + raw.BIOMASS + raw.OTHER
    raw["wind"] = raw.WIND + raw.WIND_EMB
    raw["hydro"], raw["nuclear"], raw["solar"] = raw.HYDRO, raw.NUCLEAR, raw.SOLAR
    raw["period"] = raw.stamp.dt.to_period("M")
    hourly = raw.groupby(["period", raw.stamp.dt.floor("h")])[list(COARSE)].mean()
    monthly = hourly.groupby(level=0)[list(COARSE)].sum(min_count=1)
    return monthly.div(monthly.sum(axis=1), axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--et54", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    pairs = composition_pairs()
    outer, selections = outer_pseudotarget_cv(pairs)
    outer.to_csv(args.output / "us_2024_loco_monthly_predictions.csv", index=False)
    selections.to_json(args.output / "us_outer_fold_selections.json", orient="records", indent=2)
    outer_summary = outer.groupby("ba")[ ["raw_l1", "corrected_l1"] ].mean().reset_index()
    outer_summary.to_csv(args.output / "us_2024_loco_country_metrics.csv", index=False)

    final_alpha, final_cv = cv_select_alpha(pairs, BAS)
    final_cv.to_csv(args.output / "us_inner_loco_alpha_selection.csv", index=False)
    final_train = pairs.loc[pairs.year.isin(YEARS_TRAIN)].copy()
    mapping = fitted_mapping(final_train, final_alpha)
    (args.output / "mapping_selection.json").write_text(json.dumps({
        "selected_alpha": final_alpha, "training_years": list(YEARS_TRAIN),
        "training_bas": list(BAS), "training_pairs": len(final_train),
        "outer_2024_pairs": len(outer),
    }, indent=2) + "\n")

    # Apply the frozen map to GB's current-revised official MPP reports.
    et54_all = gb_shared.official_et54_shares(args.et54)
    target_months = pd.period_range("2023-11", "2024-10", freq="M")
    et54_inputs = et54_all.loc[target_months].copy()
    et54_pair = et54_inputs.reset_index(names="observed_month")
    et54_pair["observed_month"] = et54_pair.observed_month.astype(str)
    for group in COARSE:
        et54_pair[f"reported_{group}"] = et54_pair[group]
    corrected = apply_mapping(mapping, et54_pair)
    corrected_inputs = pd.DataFrame(corrected, columns=COARSE, index=target_months)
    corrected_inputs.to_csv(args.output / "gb_corrected_tminus2_shares.csv")
    et54_inputs.to_csv(args.output / "gb_raw_et54_tminus2_shares.csv")

    models, calibration, source_raw = source_hour_models()
    neso_path = ROOT / "data/neso-gb/df_fuel_ckan.csv"
    target_times = gb_shared.calendar_target(neso_path)
    raw_features = gb_shared.attach_input(target_times, et54_inputs)
    corrected_features = gb_shared.attach_input(target_times, corrected_inputs)
    frames = {"ET54_raw": raw_features, "ET54_corrected": corrected_features}
    prediction_seals = []
    predictions = {}
    for arm, target in frames.items():
        point = predict_models(models, target)
        predictions[arm] = point
        file = args.output / f"GB_2024_{arm}_predictions.csv"
        pd.DataFrame({"time": target.time.astype(str), **point}).to_csv(file, index=False)
        prediction_seals.append({"path": file.name, "sha256": sha(file), "rows": len(target)})
    (args.output / "gb_prediction_seal.json").write_text(json.dumps(prediction_seals, indent=2) + "\n")

    # Only after prediction seals exist do we read the held-out GB hourly fuel response.
    label_columns = ["DATETIME", "GAS", "COAL", "NUCLEAR", "WIND", "WIND_EMB",
                     "HYDRO", "BIOMASS", "OTHER", "SOLAR"]
    gb_label = pd.read_csv(neso_path, usecols=label_columns)
    gb_label["stamp"] = pd.to_datetime(gb_label.DATETIME, errors="raise")
    gb_label = gb_label.loc[gb_label.stamp.dt.year.eq(2024)].copy()
    gb_label["coal"], gb_label["gas"], gb_label["oil"] = gb_label.COAL, gb_label.GAS, 0.
    gb_label["other"] = gb_label.BIOMASS + gb_label.OTHER
    gb_label["nuclear"], gb_label["hydro"] = gb_label.NUCLEAR, gb_label.HYDRO
    gb_label["wind"], gb_label["solar"] = gb_label.WIND + gb_label.WIND_EMB, gb_label.SOLAR
    gb_label["hour"] = gb_label.stamp.dt.floor("h")
    gb_hourly = gb_label.groupby("hour")[list(FUELS)].mean()
    y_gb = gb_hourly.to_numpy(float) @ EF / gb_hourly.to_numpy(float).sum(axis=1)
    gb_month_oracle = eia930_full_monthly_from_gb(neso_path)
    composition_rows = []
    for period in target_months:
        target_share = gb_month_oracle.loc[period].to_numpy(float)
        raw_share = et54_inputs.loc[period].to_numpy(float)
        corrected_share = corrected_inputs.loc[period].to_numpy(float)
        composition_rows.append({"observed_month": str(period),
            "raw_l1": float(np.abs(raw_share-target_share).sum()),
            "corrected_l1": float(np.abs(corrected_share-target_share).sum()),
            "raw_tv": float(np.abs(raw_share-target_share).sum()/2),
            "corrected_tv": float(np.abs(corrected_share-target_share).sum()/2)})
    pd.DataFrame(composition_rows).to_csv(args.output / "gb_tminus2_composition_metrics.csv", index=False)

    gb_metrics = []
    for arm, target in frames.items():
        for method in METHODS:
            gb_metrics.append({"input_arm": arm, "method": method,
                **metrics(target, y_gb, predictions[arm][method], *calibration[method])})
    # Post-hoc full-composition oracle isolates the hourly-response ceiling.
    oracle_inputs = gb_shared.attach_input(target_times, gb_month_oracle.loc[target_months])
    oracle_predictions = predict_models(models, oracle_inputs)
    for method in METHODS:
        gb_metrics.append({"input_arm": "NESO_full_composition_oracle", "method": method,
            **metrics(oracle_inputs, y_gb, oracle_predictions[method], *calibration[method])})
    pd.DataFrame(gb_metrics).to_csv(args.output / "gb_hourly_transfer_metrics.csv", index=False)
    (args.output / "postscore_target_label_sha256.json").write_text(json.dumps({
        "path": str(neso_path), "sha256": sha(neso_path), "read_after_gb_prediction_seal": True,
        "label_columns": label_columns,
    }, indent=2) + "\n")

    input_paths = [args.et54,
        ROOT / "docs/PSEUDOTARGET_REPORTED_MIX_CORRECTION_PROTOCOL_20261101.md",
        ROOT / "scripts/run_pseudotarget_report_coverage_correction.py",
        ROOT / "scripts/run_eia923_pseudotarget_transfer.py",
        ROOT / "scripts/audit_eia923_ba_mix_alignment.py",
        ROOT / "scripts/run_gb_et54_official_input_transfer.py"]
    input_paths += [eia_parser.eia_archive(year) for year in (2022, 2023, 2024)]
    input_paths += [ROOT / f"data/eia930-hourly-co2-r1/feature_inputs/{ba}_generation_2022_2024.csv.gz"
                    for ba in BAS]
    input_paths += [ROOT / f"results/research/data/{ba}_hourly.csv" for ba in BAS]
    input_paths += [args.output / "gb_raw_et54_tminus2_shares.csv",
                    args.output / "gb_corrected_tminus2_shares.csv"]
    input_hashes = {str(path): sha(path) for path in input_paths}
    (args.output / "input_code_hashes.json").write_text(json.dumps(input_hashes, indent=2) + "\n")

    summary = {
        "status": "DEVELOPMENT_PSEUDOTARGET_REPORTED_MIX_CORRECTION",
        "selected_alpha": final_alpha,
        "outer_pseudotarget_ba_count": int(outer.ba.nunique()),
        "outer_2024_ba_month_pairs": int(len(outer)),
        "outer_mean_raw_l1": float(outer.raw_l1.mean()),
        "outer_mean_corrected_l1": float(outer.corrected_l1.mean()),
        "gb_target_hours": int(len(y_gb)),
        "gb_result_is_development": True,
        "eia923_vintage": "2022-2024 current annual final archives; lag simulation, not as-issued",
        "et54_vintage": "2026-10-29 current revised workbook; MPP-only monthly coverage",
        "gb_prediction_seals": prediction_seals,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("\nGB hourly comparison:")
    print(pd.DataFrame(gb_metrics).query("method == 'coarse_recomposition'").to_string(index=False))


if __name__ == "__main__":
    main()
