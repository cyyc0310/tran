"""V343: sensitivity of hourly tail-risk hulls to thermal-EF support width."""
from __future__ import annotations

import json
from fractions import Fraction as F
from pathlib import Path

import pandas as pd

from v342_china_sharp_group_hull import GROUPS, ROOT, V341, V341_SCRIPT, V339_IMPL, sharp_group_set, sha

PROTOCOL = ROOT / "docs/V343_THERMAL_EF_SUPPORT_VALUE_PROTOCOL.md"
V342_SCRIPT = ROOT / "scripts/v342_china_sharp_group_hull.py"
MONTHLY_INPUT = ROOT / "results/v320-oreo-official-monthly-generation-audit-r1/cn_prov_monthly_generation_by_tech.csv"
CAPACITY_INPUT = ROOT / "data/china/cec_province_installed_capacity_2022_2023.csv"
OUT = ROOT / "results/v343-thermal-ef-support-value-r1"
EF_GRID = (400, 500, 600, 700, 800, 900, 1000)
BANDS = ((400, 1000), (500, 900), (600, 800), (700, 700))
TAU = 300


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    source = pd.read_csv(V341 / "monthly_margin_capacity_audit.csv")
    source = source.loc[source.included_in_complete_case.astype(bool)].copy()
    if len(source) != 330 or source.duplicated(["province", "period"]).any():
        raise ValueError("V341 source cohort mismatch")
    curves = []
    sets: dict[tuple[str, str, str, int], tuple[int, ...]] = {}
    for rec in source.to_dict("records"):
        province, period = rec["province"], rec["period"]
        hours = int(rec["hours"])
        margins = {g: F(str(rec[f"{g}_mwh"])) for g in GROUPS}
        cap_arms = {
            "monthly_only": {g: margins[g] for g in GROUPS},
            "monthly_plus_lagged_capacity": {g: F(str(rec[f"2022_{g}_cap_mw"])) for g in GROUPS},
            "monthly_plus_2023_capacity": {g: F(str(rec[f"2023_{g}_cap_mw"])) for g in GROUPS},
        }
        for arm, caps in cap_arms.items():
            for ef_value in EF_GRID:
                factors = {g: 0 for g in GROUPS}
                factors["thermal"] = ef_value
                feasible = sharp_group_set(margins, caps, factors, TAU, hours)
                if feasible is None:
                    raise ValueError(f"capacity infeasible: {province} {period} {arm} EF={ef_value}")
                sets[(province, period, arm, ef_value)] = feasible
                curves.append({"province": province, "period": period, "hours": hours,
                               "arm": arm, "thermal_ef_g_kwh": ef_value,
                               "K_lower": min(feasible), "K_upper": max(feasible),
                               "K_hull_width_hours": max(feasible) - min(feasible),
                               "feasible_K_count": len(feasible),
                               "fixed_factor_set_gap_count": max(feasible) - min(feasible) + 1 - len(feasible)})
    curve_df = pd.DataFrame(curves)
    if len(curve_df) != 330 * 3 * len(EF_GRID):
        raise ValueError(f"curve row count mismatch: {len(curve_df)}")
    if curve_df.duplicated(["province", "period", "arm", "thermal_ef_g_kwh"]).any():
        raise ValueError("duplicate EF sensitivity keys")
    expected_keys = {(r.province, r.period, arm, ef)
                     for r in source[["province", "period"]].itertuples(index=False)
                     for arm in ("monthly_only", "monthly_plus_lagged_capacity", "monthly_plus_2023_capacity")
                     for ef in EF_GRID}
    actual_keys = set(curve_df[["province", "period", "arm", "thermal_ef_g_kwh"]]
                      .itertuples(index=False, name=None))
    if actual_keys != expected_keys:
        raise ValueError(f"expected/observed sensitivity keys differ: missing={len(expected_keys-actual_keys)}, "
                         f"extra={len(actual_keys-expected_keys)}")
    for (province, period, arm), frame in curve_df.groupby(["province", "period", "arm"]):
        ordered = frame.sort_values("thermal_ef_g_kwh")
        if not ordered.K_lower.is_monotonic_increasing or not ordered.K_upper.is_monotonic_increasing:
            raise ValueError(f"K hull not monotone in thermal EF: {province} {period} {arm}")
    bands = []
    for rec in source.to_dict("records"):
        province, period = rec["province"], rec["period"]
        for arm in ("monthly_only", "monthly_plus_lagged_capacity", "monthly_plus_2023_capacity"):
            for lo_ef, hi_ef in BANDS:
                low_set = sets[(province, period, arm, lo_ef)]
                high_set = sets[(province, period, arm, hi_ef)]
                lo, hi = min(low_set), max(high_set)
                bands.append({"province": province, "period": period, "arm": arm,
                              "thermal_ef_low": lo_ef, "thermal_ef_high": hi_ef,
                              "support_width_g_kwh": hi_ef - lo_ef,
                              "K_hull_lower": lo, "K_hull_upper": hi,
                              "K_hull_width_hours": hi - lo})
    band_df = pd.DataFrame(bands)
    metrics = []
    for (arm, low, high), frame in band_df.groupby(["arm", "thermal_ef_low", "thermal_ef_high"]):
        metrics.append({"arm": arm, "thermal_ef_low": low, "thermal_ef_high": high,
                        "support_width_g_kwh": high - low, "province_periods": len(frame),
                        "mean_K_hull_width_hours": float(frame.K_hull_width_hours.mean()),
                        "median_K_hull_width_hours": float(frame.K_hull_width_hours.median()),
                        "mean_hull_reduction_vs_400_1000_hours": float(
                            band_df.loc[(band_df.arm.eq(arm)) & band_df.thermal_ef_low.eq(400) & band_df.thermal_ef_high.eq(1000),
                                        "K_hull_width_hours"].mean() - frame.K_hull_width_hours.mean())})
    OUT.mkdir(parents=True)
    curve_df.to_csv(OUT / "fixed_factor_hull_sensitivity.csv", index=False)
    band_df.to_csv(OUT / "factor_support_band_hulls.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUT / "support_band_frontier.csv", index=False)
    payload = {"status": "FACTOR_SUPPORT_VALUE_SENSITIVITY_NO_HOURLY_LABELS",
               "n_provinces": int(curve_df.province.nunique()), "n_province_periods": int(source.shape[0]),
               "n_fixed_factor_rows": len(curve_df), "expected_fixed_factor_rows": 330 * 3 * len(EF_GRID),
               "n_support_band_rows": len(band_df), "expected_support_band_rows": 330 * 3 * len(BANDS),
               "tau_g_kwh": TAU, "thermal_ef_grid_g_kwh": EF_GRID,
               "support_bands": BANDS, "frontier": metrics,
               "limitations": ["EF bands are sensitivity scenarios, not observed uncertainty intervals or data available from the NBS five-technology table.",
                               "Point-factor results are an oracle diagnostic, not a deployable forecast.",
                               "K is an exact set only under fixed-factor endpoint assumptions, monthly margins, group capacity bounds and positive output in both blocks.",
                               "No hourly China labels exist in this experiment; no empirical coverage or forecasting performance is measured."]}
    summary_paths = (V341 / "monthly_margin_capacity_audit.csv", V341 / "summary.json",
                     MONTHLY_INPUT, CAPACITY_INPUT, V341_SCRIPT, V342_SCRIPT, V339_IMPL, PROTOCOL, Path(__file__))
    payload["sha256"] = {str(p.relative_to(ROOT)): sha(p) for p in summary_paths}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(pd.DataFrame(metrics).to_string(index=False))
    print(f"fixed-factor rows={len(curve_df)}; support-band rows={len(band_df)}")


if __name__ == "__main__":
    main()
