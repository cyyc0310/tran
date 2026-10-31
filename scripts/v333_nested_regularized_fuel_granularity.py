"""V333: compare nested ridge-NNLS across coarse and detailed fuel groups."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

import v300_official_provincial_ef_mix_transfer as v300
import v327_nbs_mix_transfer as v327
import v329_china_region_block_coldstart as v329
import v330_spatial_nested_ridge_transfer as v330


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V333_NESTED_REGULARIZED_FUEL_GRANULARITY_PROTOCOL.md"
FACTORS = ROOT / "data/china/mee_provincial_grid_co2_factors_2021_2023.csv"
MONTHLY = ROOT / "results/v320-oreo-official-monthly-generation-audit-r1/cn_prov_monthly_generation_by_tech.csv"
EMBER = ROOT / "results/china-monthly-ember-audit-r1/ember_china_province_monthly.csv"
OUT = ROOT / "results/v333-nested-regularized-fuel-granularity-r1"
FULL8 = tuple(v300.SOURCES)
COARSE5 = tuple(v300.COARSE)
LAMBDAS = tuple(v330.LAMBDAS)
REGIONS = v329.REGIONS


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(frame: pd.DataFrame, col: str) -> dict[str, float]:
    e = frame[col].to_numpy(float) - frame.actual.to_numpy(float)
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "mean_signed_error": float(np.mean(e)), "p90_absolute_error": float(np.quantile(np.abs(e), .9))}


def choose_lambda(train: pd.DataFrame, target_year: int, columns: tuple[str, ...]) -> tuple[float, dict[str, float]]:
    val_year = 2021 if target_year == 2022 else 2022
    valid = train.loc[train.target_year.eq(val_year)]
    fit_pool = train.loc[train.target_year.lt(val_year)]
    if val_year == 2021:
        fit_pool = valid
    errors = {lam: [] for lam in LAMBDAS}
    for region in sorted(train.region.unique()):
        v = valid.loc[valid.region.eq(region)]
        f = fit_pool.loc[fit_pool.region.ne(region)]
        if f.empty or v.empty:
            continue
        x, y = f[list(columns)].to_numpy(float), f.actual.to_numpy(float)
        xv, yv = v[list(columns)].to_numpy(float), v.actual.to_numpy(float)
        for lam in LAMBDAS:
            beta = v330.fit_ridge_nnls(x, y, lam)
            errors[lam].extend(np.abs(yv - xv @ beta).tolist())
    loss = {f"{lam:g}": float(np.mean(e)) for lam, e in errors.items() if e}
    if not loss:
        raise RuntimeError("No inner validation rows")
    best = min(loss.values())
    selected = min(lam for lam in LAMBDAS if abs(loss[f"{lam:g}"] - best) <= 1e-12)
    return selected, loss


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite frozen results: {OUT}")
    factors = pd.read_csv(FACTORS).set_index("province")
    provinces = sorted(factors.index.astype(str))
    assignment = {p: r for r, members in REGIONS.items() for p in members}
    if set(assignment) != set(provinces):
        raise ValueError("Region partition mismatch")
    mix = v300.monthly_mix(EMBER)
    nbs_mix, _ = v327.nbs_annual_mix(MONTHLY, provinces)
    rows = []
    for target_year in (2021, 2022):
        fy = target_year - 1
        for p in provinces:
            x = mix[(p, fy)]
            c = v300.aggregate_coarse(x)
            rows.append({"province": p, "region": assignment[p], "target_year": target_year,
                         "actual": float(factors.loc[p, f"ef_{target_year}_kgco2_kwh"]),
                         **{f"f8_{k}": float(x[k]) for k in FULL8},
                         **{f"c5_{k}": float(v) for k, v in zip(COARSE5, c)},
                         **{f"nbs_{k}": float(nbs_mix[(p, fy)][k]) for k in v327.GROUPS}})
    all_rows = pd.DataFrame(rows)
    preds, tunes = [], []
    models = {"full8_nnls": (tuple(f"f8_{s}" for s in FULL8), 0.0),
              "coarse5_nnls": (tuple(f"c5_{s}" for s in COARSE5), 0.0),
              "full8_ridge": (tuple(f"f8_{s}" for s in FULL8), None),
              "coarse5_ridge": (tuple(f"c5_{s}" for s in COARSE5), None)}
    for held_region, members in REGIONS.items():
        for target_year in (2022, 2023):
            fy = target_year - 1
            train = all_rows.loc[all_rows.target_year.lt(target_year) & all_rows.region.ne(held_region)]
            model_betas = {}
            for model, (cols, fixed_lam) in models.items():
                if fixed_lam is None:
                    lam, losses = choose_lambda(train, target_year, cols)
                    tunes.append({"held_region": held_region, "target_year": target_year, "model": model,
                                  "selected_lambda": lam, "inner_mae_by_lambda": json.dumps(losses)})
                else:
                    lam = fixed_lam
                xfit, yfit = train[list(cols)].to_numpy(float), train.actual.to_numpy(float)
                beta = v330.fit_ridge_nnls(xfit, yfit, lam)
                model_betas[model] = (cols, beta)
            # NBS is retained as a distinct observed-mix context baseline.
            ncols = tuple(f"nbs_{s}" for s in v327.GROUPS)
            nfit = train[list(ncols)].to_numpy(float)
            nbeta = nnls(nfit, train.actual.to_numpy(float))[0]
            for p in members:
                x = mix[(p, fy)]
                row = {"province": p, "held_region": held_region, "target_year": target_year,
                       "actual": float(factors.loc[p, f"ef_{target_year}_kgco2_kwh"]),
                       "outside_mean_pred": float(train.actual.mean()),
                       "nbs5_pred": float(np.array([nbs_mix[(p, fy)][s] for s in v327.GROUPS]) @ nbeta)}
                for model, (cols, beta) in model_betas.items():
                    if model.startswith("full8"):
                        xt = np.array([x[s] for s in FULL8], float)
                    else:
                        xt = v300.aggregate_coarse(x)
                    row[f"{model}_pred"] = float(xt @ beta)
                    row[f"{model}_beta"] = json.dumps(dict(zip(cols, beta.tolist())))
                preds.append(row)
    pred = pd.DataFrame(preds)
    methods = {"outside_mean": "outside_mean_pred", "NBS5_NNLS": "nbs5_pred",
               **{m: f"{m}_pred" for m in models}}
    score_rows, region_rows = [], []
    for label, sub in [("pooled", pred), ("2022", pred.loc[pred.target_year.eq(2022)]),
                       ("2023", pred.loc[pred.target_year.eq(2023)])]:
        for name, col in methods.items():
            score_rows.append({"evaluation": label, "weighting": "province_weighted", "method": name, **score(sub, col)})
            rs = [score(g, col) for _, g in sub.groupby("held_region")]
            score_rows.append({"evaluation": label, "weighting": "region_equal", "method": name,
                               **{k: float(np.mean([x[k] for x in rs])) for k in rs[0]}})
    for (region, year), g in pred.groupby(["held_region", "target_year"]):
        for name, col in methods.items():
            region_rows.append({"held_region": region, "target_year": int(year), "method": name, **score(g, col)})
    region_err = pred.groupby("held_region").apply(
        lambda g: pd.Series({m: float(np.mean(np.abs(g[c] - g.actual))) for m, c in methods.items()}),
        include_groups=False)
    rng = np.random.default_rng(333)
    comparisons = {}
    for comparator in ("coarse5_nnls", "coarse5_ridge", "NBS5_NNLS"):
        d = (region_err["full8_ridge"] - region_err[comparator]).to_numpy(float)
        b = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(50000)])
        comparisons[f"full8_ridge_minus_{comparator}_region_mae"] = {
            "difference": float(d.mean()), "region_cluster_bootstrap_95pct": np.quantile(b, [.025, .975]).tolist(),
            "regions_full8_ridge_better": int((d < 0).sum()), "per_region": dict(zip(region_err.index, d.tolist()))}
    OUT.mkdir(parents=True)
    pred.to_csv(OUT / "province_predictions.csv", index=False)
    pd.DataFrame(score_rows).to_csv(OUT / "model_scores.csv", index=False)
    pd.DataFrame(region_rows).to_csv(OUT / "region_year_scores.csv", index=False)
    pd.DataFrame(tunes).to_csv(OUT / "tuning.csv", index=False)
    summary = {"status": "NESTED_REGULARIZED_FUEL_GRANULARITY_REGION_COLDSTART", "lambda_candidates": list(LAMBDAS),
               "full8_ridge_comparisons": comparisons, "models": score_rows,
               "limitations": ["Annual MEE consumption-side targets, not hourly CIF.",
                               "Only seven correlated regions and two target years.",
                               "Ember is estimated retrospective mix; vintage availability unverified.",
                               "Regularization is standard and does not establish novelty by itself."]}
    summary["sha256"] = {str(path.relative_to(ROOT)): sha(path) for path in (FACTORS, MONTHLY, EMBER, PROTOCOL, Path(__file__))}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(pd.DataFrame(score_rows).to_string(index=False))
    print(json.dumps(comparisons, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
