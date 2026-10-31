"""V332: compare detailed vs coarse source mix under whole-region holdout."""

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


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V332_FUEL_GRANULARITY_REGION_COLDSTART_PROTOCOL.md"
FACTORS = ROOT / "data/china/mee_provincial_grid_co2_factors_2021_2023.csv"
MONTHLY = ROOT / "results/v320-oreo-official-monthly-generation-audit-r1/cn_prov_monthly_generation_by_tech.csv"
EMBER = ROOT / "results/china-monthly-ember-audit-r1/ember_china_province_monthly.csv"
OUT = ROOT / "results/v332-fuel-granularity-region-coldstart-r1"
FULL8 = tuple(v300.SOURCES)
COARSE5 = tuple(v300.COARSE)
REGIONS = v329.REGIONS


def score(frame: pd.DataFrame, pred_col: str) -> dict[str, float]:
    e = frame[pred_col].to_numpy(float) - frame.actual.to_numpy(float)
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "mean_signed_error": float(np.mean(e)), "p90_absolute_error": float(np.quantile(np.abs(e), .9))}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite frozen results: {OUT}")
    factors = pd.read_csv(FACTORS).set_index("province")
    provinces = sorted(factors.index.astype(str))
    assignment = {p: r for r, members in REGIONS.items() for p in members}
    if set(assignment) != set(provinces):
        raise ValueError("V329 regions do not partition the MEE factor cohort")
    ember_mix = v300.monthly_mix(EMBER)
    nbs_mix, _ = v327.nbs_annual_mix(MONTHLY, provinces)
    rows = []
    for target_year in (2021, 2022):
        fy = target_year - 1
        for province in provinces:
            full = ember_mix[(province, fy)]
            coarse = v300.aggregate_coarse(full)
            nbs = nbs_mix[(province, fy)]
            rows.append({"province": province, "region": assignment[province], "target_year": target_year,
                         "actual": float(factors.loc[province, f"ef_{target_year}_kgco2_kwh"]),
                         **{f"full8_{s}": float(full[s]) for s in FULL8},
                         **{f"coarse5_{s}": float(x) for s, x in zip(COARSE5, coarse)},
                         **{f"nbs5_{s}": float(nbs[s]) for s in v327.GROUPS}})
    train_all = pd.DataFrame(rows)
    predictions = []
    for held_region, members in REGIONS.items():
        for target_year in (2022, 2023):
            fy = target_year - 1
            fit = train_all.loc[train_all.target_year.lt(target_year) & train_all.region.ne(held_region)]
            if fit.region.eq(held_region).any():
                raise AssertionError("held-region label leakage")
            y = fit.actual.to_numpy(float)
            x8 = fit[[f"full8_{s}" for s in FULL8]].to_numpy(float)
            x5 = fit[[f"coarse5_{s}" for s in COARSE5]].to_numpy(float)
            xn = fit[[f"nbs5_{s}" for s in v327.GROUPS]].to_numpy(float)
            b8, _ = nnls(x8, y); b5, _ = nnls(x5, y); bn, _ = nnls(xn, y)
            for province in members:
                f = ember_mix[(province, fy)]
                x8t = np.array([f[s] for s in FULL8], float)
                x5t = v300.aggregate_coarse(f)
                xn_t = np.array([nbs_mix[(province, fy)][s] for s in v327.GROUPS], float)
                predictions.append({"province": province, "held_region": held_region, "target_year": target_year,
                                    "actual": float(factors.loc[province, f"ef_{target_year}_kgco2_kwh"]),
                                    "outside_mean_pred": float(y.mean()), "ember_full8_pred": float(x8t @ b8),
                                    "ember_coarse5_pred": float(x5t @ b5), "nbs5_pred": float(xn_t @ bn),
                                    "full8_beta": json.dumps(dict(zip(FULL8, b8.tolist()))),
                                    "coarse5_beta": json.dumps(dict(zip(COARSE5, b5.tolist())))} )
    pred = pd.DataFrame(predictions)
    methods = {"outside_mean": "outside_mean_pred", "Ember_full8": "ember_full8_pred",
               "Ember_coarse5": "ember_coarse5_pred", "NBS5": "nbs5_pred"}
    scores = []
    for label, sub in [("pooled", pred), ("2022", pred.loc[pred.target_year.eq(2022)]),
                       ("2023", pred.loc[pred.target_year.eq(2023)])]:
        for method, col in methods.items():
            scores.append({"evaluation": label, "weighting": "province_weighted", "method": method, **score(sub, col)})
            reg_scores = [score(g, col) for _, g in sub.groupby("held_region")]
            scores.append({"evaluation": label, "weighting": "region_equal", "method": method,
                           **{k: float(np.mean([m[k] for m in reg_scores])) for k in reg_scores[0]}})
    region_rows = []
    for (region, year), group in pred.groupby(["held_region", "target_year"]):
        for method, col in methods.items():
            region_rows.append({"held_region": region, "target_year": int(year), "method": method,
                                "n_test_provinces": int(group.province.nunique()), **score(group, col)})
    paired = pred.assign(full_abs=(pred.ember_full8_pred - pred.actual).abs(),
                         coarse_abs=(pred.ember_coarse5_pred - pred.actual).abs()).groupby("held_region")[["full_abs", "coarse_abs"]].mean()
    delta = (paired.full_abs - paired.coarse_abs).to_numpy(float)
    rng = np.random.default_rng(332)
    bootstrap = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(50000)])
    comparison = {"region_equal_full8_minus_coarse5_mae": float(delta.mean()),
                  "region_cluster_bootstrap_95pct": np.quantile(bootstrap, [.025, .975]).tolist(),
                  "regions_full8_better": int((delta < 0).sum()),
                  "per_region_difference": dict(zip(paired.index, delta.tolist()))}
    OUT.mkdir(parents=True)
    pred.to_csv(OUT / "province_predictions.csv", index=False)
    pd.DataFrame(scores).to_csv(OUT / "model_scores.csv", index=False)
    pd.DataFrame(region_rows).to_csv(OUT / "region_year_scores.csv", index=False)
    summary = {"status": "FUEL_GRANULARITY_WHOLE_REGION_COLDSTART_ANNUAL_MEE_TRANSFER", "n_regions": len(REGIONS),
               "n_provinces": len(provinces), "full8_groups": list(FULL8), "coarse5_groups": list(COARSE5),
               "full8_vs_coarse5": comparison, "models": scores,
               "limitations": ["MEE targets are annual consumption-side factors, not hourly CIF.",
                               "Only two forward target years and seven correlated region clusters.",
                               "Ember mix is a retrospective estimated snapshot without verified as-of vintage.",
                               "NBS monthly data covers above-designated-size generation and is used only as context baseline."]}
    summary["sha256"] = {str(path.relative_to(ROOT)): sha(path) for path in (FACTORS, MONTHLY, EMBER, PROTOCOL, Path(__file__))}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(pd.DataFrame(scores).to_string(index=False))
    print(json.dumps({"full8_vs_coarse5": comparison}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
