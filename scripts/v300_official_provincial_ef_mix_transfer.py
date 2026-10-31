"""V300: test lagged monthly fuel mix as an annual province-CO2 level anchor."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V300_OFFICIAL_PROVINCIAL_EF_MIX_TRANSFER_PROTOCOL.md"
FACTORS = ROOT / "data/china/mee_provincial_grid_co2_factors_2021_2023.csv"
EMBER = ROOT / "results/china-monthly-ember-audit-r1/ember_china_province_monthly.csv"
OUT = ROOT / "results/v300-official-provincial-ef-mix-transfer-r1"
SOURCES = ("Coal", "Gas", "Other fossil", "Bioenergy", "Hydro", "Nuclear", "Wind", "Solar")
COARSE = ("Thermal", "Hydro", "Nuclear", "Wind", "Solar")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def monthly_mix(path: Path) -> dict[tuple[str, int], dict[str, float]]:
    raw = pd.read_csv(path)
    raw = raw.loc[raw["Province code"].notna() & raw["Province"].ne("China (total)")].copy()
    raw["year"] = pd.to_datetime(raw["Date"]).dt.year
    src = raw.pivot_table(index=["Province", "year", "Date"], columns="Electricity source",
                          values="Generation (TWh)", aggfunc="first")
    required = set(SOURCES) | {"Total generation"}
    missing = required - set(src.columns)
    if missing or src[list(required)].isna().any().any():
        raise ValueError(f"Ember source/month coverage is incomplete: {sorted(missing)}")
    max_gap = (src[list(SOURCES)].sum(axis=1) - src["Total generation"]).abs().max()
    if max_gap > 0.004:
        raise ValueError(f"atomic sources do not reconcile to total: max gap {max_gap}")
    annual = src.groupby(level=["Province", "year"]).sum(numeric_only=True)
    out: dict[tuple[str, int], dict[str, float]] = {}
    for (province, year), row in annual.iterrows():
        total = float(row["Total generation"])
        if total <= 0:
            continue
        out[(str(province), int(year))] = {s: float(row[s]) / total for s in SOURCES}
    return out


def aggregate_coarse(features: dict[str, float]) -> np.ndarray:
    return np.array([
        features["Coal"] + features["Gas"] + features["Other fossil"] + features["Bioenergy"],
        features["Hydro"], features["Nuclear"], features["Wind"], features["Solar"],
    ], dtype=float)


def metrics(errors: np.ndarray) -> dict[str, float]:
    return {"mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "mean_signed_error": float(np.mean(errors)),
            "p90_absolute_error": float(np.quantile(np.abs(errors), 0.9))}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {OUT}")
    factors = pd.read_csv(FACTORS).set_index("province")
    mix = monthly_mix(EMBER)
    target = factors["ef_2023_kgco2_kwh"].astype(float)
    provinces = sorted(target.index)

    # Frozen supervised rows: 2021 target <- 2020 mix; 2022 target <- 2021 mix.
    train_rows = []
    for year, mix_year, ef_col in ((2021, 2020, "ef_2021_kgco2_kwh"),
                                   (2022, 2021, "ef_2022_kgco2_kwh")):
        for province in provinces:
            key = (province, mix_year)
            if key not in mix:
                raise ValueError(f"missing lagged Ember mix {key}")
            train_rows.append({"province": province, "target_year": year,
                               "mix_year": mix_year, "y": float(factors.loc[province, ef_col]),
                               **mix[key]})
    train = pd.DataFrame(train_rows)

    preds = []
    for province in provinces:
        held = train.loc[train.province.ne(province)]
        y_train = held.y.to_numpy(float)
        global_mean = float(y_train.mean())
        x8 = held[list(SOURCES)].to_numpy(float)
        beta8, _ = nnls(x8, y_train)
        x5 = np.vstack([aggregate_coarse(row) for row in held[list(SOURCES)].to_dict("records")])
        beta5, _ = nnls(x5, y_train)
        x_target8 = np.array([mix[(province, 2022)][s] for s in SOURCES], dtype=float)
        x_target5 = aggregate_coarse(mix[(province, 2022)])
        actual = float(target.loc[province])
        preds.append({"province": province, "actual_2023": actual,
                      "global_mean_pred": global_mean,
                      "full8_mix_pred": float(x_target8 @ beta8),
                      "coarse5_mix_pred": float(x_target5 @ beta5),
                      "local_2022_factor_reference": float(factors.loc[province, "ef_2022_kgco2_kwh"]),
                      "full8_beta": json.dumps(dict(zip(SOURCES, beta8.tolist())), ensure_ascii=False),
                      "coarse5_beta": json.dumps(dict(zip(COARSE, beta5.tolist())), ensure_ascii=False)})
    result = pd.DataFrame(preds)
    methods = {"global_mean": "global_mean_pred", "full8_mix": "full8_mix_pred",
               "coarse5_mix": "coarse5_mix_pred",
               "local_2022_delayed_reference": "local_2022_factor_reference"}
    score_rows = []
    for name, col in methods.items():
        m = metrics(result[col].to_numpy(float) - result.actual_2023.to_numpy(float))
        score_rows.append({"method": name, **m})
    scores = pd.DataFrame(score_rows)
    base_mae = float(scores.loc[scores.method.eq("global_mean"), "mae"].iloc[0])
    scores["mae_reduction_vs_global_pct"] = 100 * (base_mae - scores.mae) / base_mae

    drift = factors.reset_index()[["province", "ef_2021_kgco2_kwh", "ef_2022_kgco2_kwh", "ef_2023_kgco2_kwh"]].copy()
    drift["delta_2022_minus_2021"] = drift.ef_2022_kgco2_kwh - drift.ef_2021_kgco2_kwh
    drift["delta_2023_minus_2022"] = drift.ef_2023_kgco2_kwh - drift.ef_2022_kgco2_kwh
    drift["abs_delta_2023_minus_2022"] = drift.delta_2023_minus_2022.abs()

    OUT.mkdir(parents=True)
    result.to_csv(OUT / "province_predictions_2023.csv", index=False)
    scores.to_csv(OUT / "model_scores.csv", index=False)
    drift.to_csv(OUT / "official_factor_drift.csv", index=False)
    summary = {
        "status": "OFFICIAL_ANNUAL_CO2_FACTOR_COLD_START_TRANSFER_RETROSPECTIVE_MIX_VINTAGE_UNVERIFIED",
        "n_provinces": len(provinces), "training_rows_per_fold": len(train) - 2,
        "training_targets": [2021, 2022], "test_target_year": 2023,
        "feature_mix_years": {"target_2021": 2020, "target_2022": 2021, "target_2023": 2022},
        "models": scores.to_dict("records"),
        "median_abs_2023_factor_drift_kgco2_kwh": float(drift.abs_delta_2023_minus_2022.median()),
        "max_abs_2023_factor_drift_kgco2_kwh": float(drift.abs_delta_2023_minus_2022.max()),
        "protocol_sha256": sha(PROTOCOL), "factor_table_sha256": sha(FACTORS),
        "ember_input_sha256": sha(EMBER), "script_sha256": sha(Path(__file__)),
        "limitations": [
            "Official target is a province-average electricity CO2 factor for consumption accounting, not hourly generation CIF.",
            "Only three official annual target vintages; the 2023 year is the sole forward target after 2021-2022 training.",
            "Ember monthly historical source mix is retrospective and has no verified as-of vintage in this repository.",
            "No prediction of hourly shape, hourly coverage, or threshold exceedance is evaluated.",
            "LOPO over 30 provinces is a spatial cold-start check, not an independent external-country validation.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(scores.to_string(index=False))
    print("drift", drift[["delta_2022_minus_2021", "delta_2023_minus_2022"]].describe().to_string())


if __name__ == "__main__":
    main()
