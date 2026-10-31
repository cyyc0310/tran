"""V301: temporal replication of V300 on the 2022 official province factors."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

import v300_official_provincial_ef_mix_transfer as v300

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V301_MIX_TRANSFER_2022_TEMPORAL_REPLICATION_PROTOCOL.md"
FACTORS = ROOT / "data/china/mee_provincial_grid_co2_factors_2021_2023.csv"
EMBER = ROOT / "results/china-monthly-ember-audit-r1/ember_china_province_monthly.csv"
OUT = ROOT / "results/v301-mix-transfer-2022-temporal-replication-r1"


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {OUT}")
    factors = pd.read_csv(FACTORS).set_index("province")
    mix = v300.monthly_mix(EMBER)
    rows = []
    for province in sorted(factors.index):
        train = factors.drop(index=province)
        x8 = np.vstack([[mix[(p, 2020)][s] for s in v300.SOURCES] for p in train.index])
        y = train.ef_2021_kgco2_kwh.to_numpy(float)
        b8, _ = nnls(x8, y)
        x5 = np.vstack([v300.aggregate_coarse(mix[(p, 2020)]) for p in train.index])
        b5, _ = nnls(x5, y)
        target8 = np.array([mix[(province, 2021)][s] for s in v300.SOURCES])
        target5 = v300.aggregate_coarse(mix[(province, 2021)])
        rows.append({"province": province,
                     "actual_2022": float(factors.loc[province, "ef_2022_kgco2_kwh"]),
                     "global_mean_pred": float(y.mean()),
                     "full8_mix_pred": float(target8 @ b8),
                     "coarse5_mix_pred": float(target5 @ b5),
                     "local_2021_factor_reference": float(factors.loc[province, "ef_2021_kgco2_kwh"]),
                     "full8_beta": json.dumps(dict(zip(v300.SOURCES, b8.tolist())), ensure_ascii=False),
                     "coarse5_beta": json.dumps(dict(zip(v300.COARSE, b5.tolist())), ensure_ascii=False)})
    pred = pd.DataFrame(rows)
    methods = {"global_mean": "global_mean_pred", "full8_mix": "full8_mix_pred",
               "coarse5_mix": "coarse5_mix_pred",
               "local_2021_delayed_reference": "local_2021_factor_reference"}
    score_rows = []
    for name, col in methods.items():
        e = pred[col].to_numpy(float) - pred.actual_2022.to_numpy(float)
        score_rows.append({"method": name, **v300.metrics(e)})
    scores = pd.DataFrame(score_rows)
    base = float(scores.loc[scores.method.eq("global_mean"), "mae"].iloc[0])
    scores["mae_reduction_vs_global_pct"] = 100 * (base - scores.mae) / base
    OUT.mkdir(parents=True)
    pred.to_csv(OUT / "province_predictions_2022.csv", index=False)
    scores.to_csv(OUT / "model_scores.csv", index=False)
    summary = {"status": "SAME_SOURCE_TEMPORAL_REPLICATION_RETROSPECTIVE",
               "n_provinces": len(pred), "training_target_year": 2021,
               "training_mix_year": 2020, "test_target_year": 2022,
               "test_mix_year": 2021, "models": scores.to_dict("records"),
               "protocol_sha256": v300.sha(PROTOCOL), "factor_table_sha256": v300.sha(FACTORS),
               "ember_input_sha256": v300.sha(EMBER), "script_sha256": v300.sha(Path(__file__)),
               "limitations": ["Same official target series and same Ember retrospective input family as V300.",
                               "Official annual target is a consumption-accounting grid factor, not hourly generation CIF.",
                               "No Ember release vintage; lagged mix timing is not verified historical as-of availability.",
                               "No hourly shape, hourly coverage, or exceedance prediction is evaluated."]}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(scores.to_string(index=False))


if __name__ == "__main__":
    main()
