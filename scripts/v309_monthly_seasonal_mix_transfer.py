"""V309: nested leave-province-out test of lagged monthly composition seasonality."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V309_MONTHLY_SEASONAL_MIX_TRANSFER_PROTOCOL.md"
FACTORS = ROOT / "data/china/mee_provincial_grid_co2_factors_2021_2023.csv"
EMBER = ROOT / "results/v294-china-ember-lineage-r4/ember_china_monthly_generation.csv"
OUT = ROOT / "results/v309-monthly-seasonal-mix-transfer-r1"
SOURCE_COLUMNS = {
    "Thermal": ("Coal", "Gas", "Other fossil", "Bioenergy"),
    "Hydro": ("Hydro",),
    "Nuclear": ("Nuclear",),
    "Wind": ("Wind",),
    "Solar": ("Solar",),
}
ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
BOOTSTRAP_REPS = 10_000
SEED = 309_2026


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def monthly_features() -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    raw = pd.read_csv(EMBER)
    raw = raw.loc[raw["Province code"].notna() & raw["Province"].ne("China (total)")].copy()
    raw["Date"] = pd.to_datetime(raw["Date"], errors="raise")
    raw["year"] = raw["Date"].dt.year
    raw["month"] = raw["Date"].dt.month
    p = raw.pivot_table(index=["Province", "year", "month"], columns="Electricity source",
                        values="Generation (TWh)", aggfunc="first")
    required = set(SOURCE_COLUMNS) - {"Thermal"}
    required |= {s for source_tuple in SOURCE_COLUMNS.values() for s in source_tuple}
    required.add("Total generation")
    missing = required - set(p.columns)
    if missing:
        raise ValueError(f"missing Ember generation sources: {sorted(missing)}")
    out = {}
    for (province, year), block in p.groupby(level=["Province", "year"], sort=True):
        block = block.droplevel(["Province", "year"]).sort_index()
        if list(block.index.astype(int)) != list(range(1, 13)):
            continue
        total = block["Total generation"].to_numpy(float)
        if np.any(~np.isfinite(total)) or np.any(total <= 0):
            continue
        monthly_generation = np.column_stack([
            block[list(sources)].sum(axis=1).to_numpy(float)
            for sources in SOURCE_COLUMNS.values()
        ])
        if np.any(monthly_generation < 0):
            raise ValueError("negative generation encountered")
        monthly_shares = monthly_generation / total[:, None]
        annual_generation = monthly_generation.sum(axis=0)
        annual_total = float(total.sum())
        annual_shares = annual_generation / annual_total
        out[(str(province), int(year))] = (annual_shares, monthly_shares, total)
    return out


def feature_vector(parts: tuple[np.ndarray, np.ndarray, np.ndarray], harmonics: int) -> np.ndarray:
    annual, monthly, _ = parts
    fields = [annual]
    centered = monthly - monthly.mean(axis=0, keepdims=True)
    months = np.arange(1, 13)
    for harmonic in range(1, harmonics + 1):
        angle = 2.0 * np.pi * harmonic * months / 12.0
        fields.append((2.0 / 12.0) * (centered * np.sin(angle)[:, None]).sum(axis=0))
        fields.append((2.0 / 12.0) * (centered * np.cos(angle)[:, None]).sum(axis=0))
    return np.concatenate(fields)


def predict_ridge(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, alpha: float) -> np.ndarray:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (train_x - mean) / scale
    z_test = (test_x - mean) / scale
    ymean = float(train_y.mean())
    yc = train_y - ymean
    beta = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ yc)
    return z_test @ beta + ymean


def choose_alpha(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[float, dict[float, float]]:
    scores: dict[float, float] = {}
    for alpha in ALPHAS:
        actuals, predictions = [], []
        for province in np.unique(groups):
            validation = groups == province
            training = ~validation
            pred = predict_ridge(x[training], y[training], x[validation], alpha)
            predictions.extend(pred.tolist())
            actuals.extend(y[validation].tolist())
        scores[alpha] = float(np.mean(np.abs(np.asarray(predictions) - np.asarray(actuals))))
    best_score = min(scores.values())
    best = max(alpha for alpha, score in scores.items() if abs(score - best_score) <= 1e-12)
    return best, scores


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - y
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias": float(np.mean(error)),
        "p90_absolute_error": float(np.quantile(np.abs(error), 0.9)),
    }


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {OUT}")
    factors = pd.read_csv(FACTORS).set_index("province")
    mix = monthly_features()
    results = []
    for target_year in (2022, 2023):
        ef_col = f"ef_{target_year}_kgco2_kwh"
        train_target_years = list(range(2021, target_year))
        train_rows = []
        for year in train_target_years:
            mix_year = year - 1
            col = f"ef_{year}_kgco2_kwh"
            for province in factors.index:
                key = (province, mix_year)
                if key not in mix:
                    raise ValueError(f"missing lagged monthly generation {key}")
                annual, monthly, total = mix[key]
                train_rows.append({
                    "province": province, "target_year": year,
                    "y": float(factors.loc[province, col]),
                    "annual": annual, "monthly": monthly,
                })
        train = pd.DataFrame(train_rows)
        y_train_all = train.y.to_numpy(float)
        group_all = train.province.to_numpy(str)
        annual_x_all = np.vstack(train.annual.to_list())
        feature_matrices = {0: annual_x_all}
        for harmonics in (1, 2):
            feature_matrices[harmonics] = np.vstack([
                feature_vector(mix[(str(row.province), int(row.target_year) - 1)], harmonics)
                for row in train.itertuples(index=False)
            ])
        for held_province in factors.index:
            held = group_all == held_province
            tr = ~held
            actual = float(factors.loc[held_province, ef_col])
            target_mix = mix[(held_province, target_year - 1)]
            target_annual = target_mix[0][None, :]
            preds = {
                "global_mean": float(y_train_all[tr].mean()),
                "annual_nnls": float(target_annual[0] @ nnls(annual_x_all[tr], y_train_all[tr])[0]),
            }
            selected = {}
            for harmonics, name in ((0, "annual_ridge"), (1, "annual_h1_ridge"), (2, "annual_h2_ridge")):
                x = feature_matrices[harmonics]
                alpha, inner_scores = choose_alpha(x[tr], y_train_all[tr], group_all[tr])
                target_x = target_annual if harmonics == 0 else feature_vector(target_mix, harmonics)[None, :]
                preds[name] = float(predict_ridge(x[tr], y_train_all[tr], target_x, alpha)[0])
                selected[name] = {"alpha": alpha, "inner_lopo_mae": inner_scores[alpha]}
            row = {"target_year": target_year, "province": held_province,
                   "actual": actual, **preds,
                   "selected_hyperparameters": json.dumps(selected, ensure_ascii=False)}
            results.append(row)

    detail = pd.DataFrame(results)
    score_rows = []
    for year, block in detail.groupby("target_year"):
        for method in ("global_mean", "annual_nnls", "annual_ridge", "annual_h1_ridge", "annual_h2_ridge"):
            score_rows.append({"target_year": int(year), "method": method,
                               **metrics(block.actual.to_numpy(float), block[method].to_numpy(float))})
    scores = pd.DataFrame(score_rows)
    rng = np.random.default_rng(SEED)
    boot_rows = []
    held2023 = detail.loc[detail.target_year.eq(2023)].set_index("province")
    for method in ("annual_h1_ridge", "annual_h2_ridge"):
        delta = np.abs(held2023[method].to_numpy() - held2023.actual.to_numpy()) - np.abs(
            held2023.annual_nnls.to_numpy() - held2023.actual.to_numpy())
        draws = rng.integers(0, len(delta), size=(BOOTSTRAP_REPS, len(delta)))
        boot_mae_diff = delta[draws].mean(axis=1)
        boot_rows.append({"method": method, "mean_mae_difference_vs_annual_nnls": float(delta.mean()),
                          "bootstrap_95_low": float(np.quantile(boot_mae_diff, 0.025)),
                          "bootstrap_95_high": float(np.quantile(boot_mae_diff, 0.975)),
                          "n_provinces": len(delta)})

    OUT.mkdir(parents=True)
    detail.to_csv(OUT / "province_predictions.csv", index=False)
    scores.to_csv(OUT / "model_scores.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(OUT / "paired_province_bootstrap_2023.csv", index=False)
    summary = {
        "status": "POSTHOC_LAGGED_SEASONAL_MIX_TRANSFER_NOT_ASOF_VALIDATION",
        "target_years": [2022, 2023],
        "train_target_years_by_test": {"2022": [2021], "2023": [2021, 2022]},
        "n_provinces_per_year": int(factors.shape[0]),
        "methods": scores.to_dict("records"),
        "paired_province_bootstrap_2023": boot_rows,
        "input_vintage": "Current retrospective Ember revision; historical as-of release dates not verified.",
        "limitations": [
            "MEE outcomes are annual consumption-side factors, not local generation-side monthly/hourly CIF.",
            "Ember source decomposition includes estimation and revision and is not the user's raw monthly table.",
            "2022 and 2023 are only two temporal holdouts; the labels were already opened in V300/V301.",
            "Seasonality is predictive association only; no claim that it causally captures imports or dispatch.",
            "Bootstrap resamples provinces and does not estimate future-year uncertainty.",
        ],
        "protocol_sha256": sha(PROTOCOL), "factor_sha256": sha(FACTORS),
        "ember_sha256": sha(EMBER), "script_sha256": sha(Path(__file__)),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(scores.to_string(index=False))
    print(pd.DataFrame(boot_rows).to_string(index=False))


if __name__ == "__main__":
    main()
