#!/usr/bin/env python3
"""Post-hoc exploratory level/shape decomposition on the V410 scenario labels.

All S2-S5 results have already been inspected in V410. This is discovery only,
not a confirmatory experiment or an independent holdout.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v410_china_simulated_cif_scenario_stress as v410

OUT = ROOT / "results" / "v411-level-shape-scenario-exploratory-r1"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    provinces, y_s1 = v410.read_scenario(v410.DATA_DIR / "CEF data for Scenario S1.xlsx")
    shares, _ = v410.load_lagged_nbs(provinces)
    scenario_y = {}
    for scenario in v410.TARGET_SCENARIOS:
        ps, vals = v410.read_scenario(v410.DATA_DIR / f"CEF data for Scenario {scenario}.xlsx")
        if ps != provinces:
            raise ValueError("Province order mismatch")
        scenario_y[scenario] = vals
    months = v410.month_index()
    anomaly = {p: {} for p in provinces}
    for p in provinces:
        for m in v410.TARGET_MONTHS:
            y = y_s1[p][months == m]
            anomaly[p][m] = y - y.mean()

    rows, tails = [], []
    nprov = len(provinces)
    for target in provinces:
        sources = [p for p in provinces if p != target]
        q_by_month = {}
        pred_target = {}
        for m in v410.TARGET_MONTHS:
            anchor_t = v410.THERMAL_EF * shares[target][m][0]
            pred_target[m] = anchor_t + np.mean([anomaly[q][m] for q in sources], axis=0)
            scores = []
            for calibration_region in sources:
                donors = [q for q in sources if q != calibration_region]
                shape_q = np.mean([anomaly[q][m] for q in donors], axis=0)
                yq = y_s1[calibration_region][months == m]
                anchor_q = v410.THERMAL_EF * shares[calibration_region][m][0]
                scores.append(float(np.max(np.abs(yq - (anchor_q + shape_q)))))
            q_by_month[m] = v410.quantile_order_stat(scores, v410.INTERVAL_LEVEL)

        for scenario in v410.TARGET_SCENARIOS:
            for m in v410.TARGET_MONTHS:
                y = scenario_y[scenario][target][months == m]
                pred = pred_target[m]
                q = q_by_month[m]
                lower, upper = pred - q, pred + q
                contains_hourly = bool(np.all((y >= lower - 1e-12) & (y <= upper + 1e-12)))
                rows.append({"scenario": scenario, "province": target, "month": m,
                             "point_mae": float(np.mean(np.abs(y - pred))),
                             "point_rmse": float(np.sqrt(np.mean((y-pred)**2))),
                             "interval_width": 2*q, "simultaneous_month_coverage": contains_hourly,
                             "hour_fraction_covered": float(np.mean((y >= lower - 1e-12) & (y <= upper + 1e-12)))})
                n = len(y)
                for tau in v410.TAUS:
                    ka = int(np.count_nonzero(y > tau))
                    klo, khi = int(np.count_nonzero(lower > tau)), int(np.count_nonzero(upper > tau))
                    for rate in v410.RATES:
                        need = math.ceil(rate*n)
                        decision = "positive" if klo >= need else ("negative" if khi < need else "abstain")
                        tails.append({"scenario": scenario, "province": target, "month": m,
                                      "tau_gco2_kwh": tau, "event_rate": rate,
                                      "k_actual": ka, "k_lower": klo, "k_upper": khi,
                                      "k_width": khi-klo, "k_contains_actual": klo <= ka <= khi,
                                      "simultaneous_month_coverage": contains_hourly,
                                      "decision": decision})

    df, td = pd.DataFrame(rows), pd.DataFrame(tails)
    df.to_csv(OUT / "region_month_metrics.csv", index=False)
    td.to_csv(OUT / "tail_count_intervals.csv", index=False)
    summary=[]
    for scenario, g in df.groupby("scenario"):
        by_p=g.groupby("province")
        summary.append({"scenario":scenario,"method":"local_thermal_anchor_plus_source_shape_anomaly",
                        "region_equal_point_mae":float(by_p.point_mae.mean().mean()),
                        "region_equal_point_rmse":float(by_p.point_rmse.mean().mean()),
                        "region_equal_interval_width":float(by_p.interval_width.mean().mean()),
                        "region_equal_simultaneous_month_coverage":float(by_p.simultaneous_month_coverage.mean().mean()),
                        "worst_province_month_coverage":float(by_p.simultaneous_month_coverage.mean().min()),
                        "n_provinces":int(g.province.nunique()),"n_province_months":int(len(g))})
    sdf=pd.DataFrame(summary)
    sdf.to_csv(OUT / "summary_by_scenario.csv", index=False)
    ts=[]
    for (sc,tau,rate),g in td.groupby(["scenario","tau_gco2_kwh","event_rate"]):
        by_p=g.groupby("province")
        counts=g.decision.value_counts(normalize=True)
        ts.append({"scenario":sc,"tau_gco2_kwh":tau,"event_rate":rate,
                   "region_equal_k_width":float(by_p.k_width.mean().mean()),
                   "region_equal_k_containment":float(by_p.k_contains_actual.mean().mean()),
                   "positive_rate":float(counts.get("positive",0)),"negative_rate":float(counts.get("negative",0)),
                   "abstention_rate":float(counts.get("abstain",0)),
                   "simultaneous_month_coverage":float(by_p.simultaneous_month_coverage.mean().mean())})
    tdf=pd.DataFrame(ts)
    tdf.to_csv(OUT / "tail_frontier.csv", index=False)
    (OUT / "status.json").write_text(json.dumps({"status":"exploratory_posthoc",
        "hypothesis":"transfer source hourly anomalies around each source's monthly level, anchored at target local thermal share",
        "uses_v410_development_s1_for_source_shapes_and_calibration":True,
        "s2_s5_outcomes_already_inspected_before_this_method_was_added":True,
        "n_provinces":nprov,"n_scenarios":4,"n_region_months_per_scenario":len(provinces)*len(v410.TARGET_MONTHS),
        "data_are_simulated_not_measured_hourly_cif":True},ensure_ascii=False,indent=2)+"\n")
    print(sdf.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\nTail τ=700 frontier:")
    print(tdf[(tdf.tau_gco2_kwh==700)&(tdf.event_rate==0.01)].to_string(index=False,float_format=lambda x:f"{x:.4f}"))


if __name__ == "__main__":
    main()
