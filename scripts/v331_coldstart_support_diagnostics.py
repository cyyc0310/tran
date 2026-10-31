"""V331: test training-only support scores as predictors of ridge benefit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

import v300_official_provincial_ef_mix_transfer as v300
import v327_nbs_mix_transfer as v327
import v330_spatial_nested_ridge_transfer as v330


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V331_COLDSTART_SUPPORT_DIAGNOSTIC_PROTOCOL.md"
FACTORS = ROOT / "data/china/mee_provincial_grid_co2_factors_2021_2023.csv"
MONTHLY = ROOT / "results/v320-oreo-official-monthly-generation-audit-r1/cn_prov_monthly_generation_by_tech.csv"
EMBER = ROOT / "results/china-monthly-ember-audit-r1/ember_china_province_monthly.csv"
OUT = ROOT / "results/v331-coldstart-support-diagnostics-r1"
REGIONS = v330.REGIONS
GROUPS = v330.GROUPS


def hellinger(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 0, None); q = np.clip(q, 0, None)
    p = p / p.sum(); q = q / q.sum()
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


def bootstrap_assoc(frame: pd.DataFrame, score: str, outcome: str, seed: int) -> dict[str, object]:
    value = float(spearmanr(frame[score], frame[outcome]).statistic)
    region_indices = [np.flatnonzero(frame.held_region.to_numpy() == r) for r in frame.held_region.unique()]
    x_rank = rankdata(frame[score].to_numpy(float))
    y_rank = rankdata(frame[outcome].to_numpy(float))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(5000):
        chosen = rng.integers(0, len(region_indices), size=len(region_indices))
        sample_idx = np.concatenate([region_indices[i] for i in chosen])
        boots.append(float(np.corrcoef(x_rank[sample_idx], y_rank[sample_idx])[0, 1]))
    return {"spearman_rho": value,
            "region_cluster_bootstrap_95pct": np.quantile(boots, [.025, .975]).tolist(),
            "positive_ridge_gain_auc": float(roc_auc_score(frame.ridge_better.astype(int), frame[score]))}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite frozen results: {OUT}")
    factors = pd.read_csv(FACTORS).set_index("province")
    provinces = sorted(factors.index.astype(str))
    assignment = {p: r for r, members in REGIONS.items() for p in members}
    nbs_mix, _ = v327.nbs_annual_mix(MONTHLY, provinces)
    ember_mix = v300.monthly_mix(EMBER)
    train_rows = []
    for target_year in (2021, 2022):
        feature_year = target_year - 1
        for province in provinces:
            nbs = nbs_mix[(province, feature_year)]
            ember = v300.aggregate_coarse(ember_mix[(province, feature_year)])
            train_rows.append({"province": province, "region": assignment[province], "target_year": target_year,
                               "actual": float(factors.loc[province, f"ef_{target_year}_kgco2_kwh"]),
                               **{f"nbs_{g}": float(nbs[g]) for g in GROUPS},
                               **{f"ember_{g}": float(v) for g, v in zip(GROUPS, ember)}})
    train_all = pd.DataFrame(train_rows)
    audit = []
    for held_region, members in REGIONS.items():
        for target_year in (2022, 2023):
            feature_year = target_year - 1
            train = train_all.loc[train_all.target_year.lt(target_year) & train_all.region.ne(held_region)]
            nbs_x_train = train[[f"nbs_{g}" for g in GROUPS]].to_numpy(float)
            nbs_beta = v330.fit_ridge_nnls(nbs_x_train, train.actual.to_numpy(float), 0.0)
            ridge_lambda, _ = v330.choose_lambda(train, target_year)
            ridge_beta = v330.fit_ridge_nnls(nbs_x_train, train.actual.to_numpy(float), ridge_lambda)
            ember_x_train = train[[f"ember_{g}" for g in GROUPS]].to_numpy(float)
            ember_beta, _ = v330.nnls(ember_x_train, train.actual.to_numpy(float))
            for province in members:
                nbs = np.array([nbs_mix[(province, feature_year)][g] for g in GROUPS], float)
                ember_x = np.array(v300.aggregate_coarse(ember_mix[(province, feature_year)]), float)
                pred_nnls, pred_ridge, pred_ember = float(nbs @ nbs_beta), float(nbs @ ridge_beta), float(ember_x @ ember_beta)
                train_mix = nbs_x_train
                distances = [hellinger(nbs, q) for q in train_mix]
                actual = float(factors.loc[province, f"ef_{target_year}_kgco2_kwh"])
                err_nnls, err_ridge = abs(pred_nnls - actual), abs(pred_ridge - actual)
                audit.append({"province": province, "held_region": held_region, "target_year": target_year,
                              "actual": actual, "nbs_nnls_pred": pred_nnls, "nbs_ridge_pred": pred_ridge,
                              "ember_pred": pred_ember, "abs_error_nnls": err_nnls, "abs_error_ridge": err_ridge,
                              "ridge_gain": err_nnls - err_ridge, "ridge_better": err_ridge < err_nnls,
                              "nearest_training_hellinger": min(distances),
                              "regularization_adjustment": abs(pred_ridge - pred_nnls),
                              "nbs_ember_disagreement": abs(pred_nnls - pred_ember),
                              "selected_lambda": ridge_lambda})
    frame = pd.DataFrame(audit)
    scores = ("nearest_training_hellinger", "regularization_adjustment", "nbs_ember_disagreement")
    outcomes = ("abs_error_nnls", "ridge_gain")
    results = {score: {outcome: bootstrap_assoc(frame, score, outcome, 331 + i * 10 + j)
                       for j, outcome in enumerate(outcomes)} for i, score in enumerate(scores)}
    region = frame.groupby("held_region").agg(
        n_cases=("ridge_gain", "size"), nnls_mae=("abs_error_nnls", "mean"), ridge_mae=("abs_error_ridge", "mean"),
        mean_nearest_distance=("nearest_training_hellinger", "mean"), mean_ridge_gain=("ridge_gain", "mean"),
        mean_adjustment=("regularization_adjustment", "mean"), mean_disagreement=("nbs_ember_disagreement", "mean")).reset_index()
    OUT.mkdir(parents=True)
    frame.to_csv(OUT / "case_diagnostics.csv", index=False)
    region.to_csv(OUT / "region_diagnostics.csv", index=False)
    summary = {"status": "TRAINING_ONLY_SUPPORT_DIAGNOSTICS_FOR_ANNUAL_MEE_TRANSFER",
               "n_case_rows": int(len(frame)), "n_regions": int(frame.held_region.nunique()),
               "ridge_gain_mean": float(frame.ridge_gain.mean()), "ridge_improves_fraction": float(frame.ridge_better.mean()),
               "diagnostic_associations": results,
               "interpretation_guardrail": "Exploratory correlations over 60 province-year rows and seven region clusters; no province gate is validated.",
               "sha256": {str(path.relative_to(ROOT)): __import__("hashlib").sha256(path.read_bytes()).hexdigest()
                          for path in (FACTORS, MONTHLY, EMBER, PROTOCOL, Path(__file__))}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(region.to_string(index=False))


if __name__ == "__main__":
    main()
