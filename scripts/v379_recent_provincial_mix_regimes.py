"""V379: map recent provincial mix regimes and local-vs-national threshold gates."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/v320-oreo-official-monthly-generation-audit-r1/cn_prov_monthly_generation_by_tech.csv"
OUT = ROOT / "results/v379-recent-provincial-mix-regimes-r1"
LOW_EF = 400.0
HIGH_EF = 1000.0
THRESHOLDS = (100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0)


def certificate(share: pd.Series, threshold: float) -> pd.Series:
    return (LOW_EF * share <= threshold) & (threshold < HIGH_EF * share)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT)
    data = data.loc[data["technology"].isin(["total", "thermal", "hydro", "nuclear", "wind", "solar"])].copy()
    data["year"] = data["year"].astype(int)
    data["month"] = data["month"].astype(int)
    # Use fully observed single-month periods only; February is a Jan-Feb cumulative report bin.
    period_mask = ((data["year"].eq(2025) & data["month"].between(3, 12)) |
                   (data["year"].eq(2026) & data["month"].between(3, 9)))
    data = data.loc[period_mask]
    panel = data.pivot_table(
        index=["province", "year", "month"], columns="technology", values="generation_gwh", aggfunc="sum"
    ).reset_index()
    required = ["total", "thermal", "hydro", "nuclear", "wind", "solar"]
    panel = panel.dropna(subset=required).copy()
    if panel.groupby(["year", "month"])["province"].nunique().ne(31).any():
        raise ValueError("Expected 31 provinces in each selected report month")
    panel["thermal_share"] = panel["thermal"] / panel["total"]
    panel["hydro_share"] = panel["hydro"] / panel["total"]
    panel["nuclear_share"] = panel["nuclear"] / panel["total"]
    panel["wind_share"] = panel["wind"] / panel["total"]
    panel["solar_share"] = panel["solar"] / panel["total"]
    national = panel.groupby(["year", "month"], as_index=False).agg(
        national_thermal_gwh=("thermal", "sum"), national_total_gwh=("total", "sum")
    )
    national["national_thermal_share"] = national["national_thermal_gwh"] / national["national_total_gwh"]
    panel = panel.merge(national[["year", "month", "national_thermal_share"]], on=["year", "month"], validate="many_to_one")

    gate_rows = []
    for threshold in THRESHOLDS:
        panel["local_certificate"] = certificate(panel["thermal_share"], threshold)
        panel["national_certificate"] = certificate(panel["national_thermal_share"], threshold)
        panel["local_resolves_national"] = panel["national_certificate"] & ~panel["local_certificate"]
        panel["local_adds_vs_national"] = panel["local_certificate"] & ~panel["national_certificate"]
        gate_rows.append({
            "threshold_g_kwh": threshold,
            "province_periods": int(len(panel)),
            "national_certificate_periods": int(panel.groupby(["year", "month"])['national_certificate'].first().sum()),
            "local_certificate_province_periods": int(certificate(panel["thermal_share"], threshold).sum()),
            "national_ambiguity_cases_resolved_by_local_mix": int(panel["local_resolves_national"].sum()),
            "local_ambiguity_cases_added_vs_national": int(panel["local_adds_vs_national"].sum()),
        })
    pd.DataFrame(gate_rows).to_csv(OUT / "threshold_gate_summary.csv", index=False)

    panel["local_certificate_tau300"] = certificate(panel["thermal_share"], 300.0)
    panel["national_certificate_tau300"] = certificate(panel["national_thermal_share"], 300.0)
    panel["local_resolves_national_tau300"] = panel["national_certificate_tau300"] & ~panel["local_certificate_tau300"]
    panel["local_adds_vs_national_tau300"] = panel["local_certificate_tau300"] & ~panel["national_certificate_tau300"]
    panel.to_csv(OUT / "province_period_regimes.csv", index=False)
    province = panel.groupby("province", as_index=False).agg(
        periods=("month", "size"),
        mean_thermal_share=("thermal_share", "mean"),
        mean_hydro_share=("hydro_share", "mean"),
        mean_nuclear_share=("nuclear_share", "mean"),
        mean_wind_share=("wind_share", "mean"),
        mean_solar_share=("solar_share", "mean"),
        local_certificate_rate_tau300=("local_certificate_tau300", "mean"),
        resolved_count_tau300=("local_resolves_national_tau300", "sum"),
        added_count_tau300=("local_adds_vs_national_tau300", "sum"),
    )
    province["leading_nonthermal_class"] = province[["mean_hydro_share", "mean_nuclear_share", "mean_wind_share", "mean_solar_share"]].idxmax(axis=1).str.replace("mean_", "", regex=False).str.replace("_share", "", regex=False)
    province.to_csv(OUT / "province_regime_summary.csv", index=False)

    latest = panel.loc[panel["year"].eq(2026) & panel["month"].eq(9)].copy()
    latest.sort_values("hydro_share", ascending=False).head(8).to_csv(OUT / "latest_hydro_leaders.csv", index=False)
    latest.sort_values("wind_share", ascending=False).head(8).to_csv(OUT / "latest_wind_leaders.csv", index=False)
    latest.sort_values("nuclear_share", ascending=False).head(8).to_csv(OUT / "latest_nuclear_leaders.csv", index=False)
    latest.sort_values("solar_share", ascending=False).head(8).to_csv(OUT / "latest_solar_leaders.csv", index=False)

    national_certificate_periods = int(national.groupby(["year", "month"]).apply(
        lambda g: bool(certificate(g["national_thermal_share"].iloc[:1], 300.0).iloc[0]), include_groups=False
    ).sum())
    resolved = panel.loc[panel["local_resolves_national_tau300"]]
    summary = {
        "experiment": "Recent 2025-2026 provincial mix regime map and local-vs-national threshold information",
        "input": str(INPUT.relative_to(ROOT)),
        "input_scope": "V320 OREO-transcribed NBS above-size monthly panel; Feb cumulative bins excluded",
        "n_single_month_periods": int(panel.groupby(["year", "month"]).ngroups),
        "periods": [f"{int(y)}-{int(m):02d}" for y, m in sorted(panel[["year", "month"]].drop_duplicates().itertuples(index=False, name=None))],
        "n_province_periods": int(len(panel)),
        "tau300_national_certificate_report_periods": national_certificate_periods,
        "tau300_national_certified_province_periods": int(panel["national_certificate_tau300"].sum()),
        "tau300_local_resolves_cases": int(panel["local_resolves_national_tau300"].sum()),
        "tau300_local_adds_cases": int(panel["local_adds_vs_national_tau300"].sum()),
        "provinces_with_resolved_cases": sorted(resolved["province"].unique().tolist()),
        "mean_thermal_share_range_across_provinces": [float(province["mean_thermal_share"].min()), float(province["mean_thermal_share"].max())],
        "main_result_scope": "Structural mix-support certificate only; no local hourly label, EF calibration, sharp hourly K comparison, or empirical coverage.",
        "next_direct_export_candidates": {
            "hydro": latest.sort_values("hydro_share", ascending=False).head(3)["province"].tolist(),
            "wind": latest.sort_values("wind_share", ascending=False).head(3)["province"].tolist(),
            "nuclear": latest.sort_values("nuclear_share", ascending=False).head(3)["province"].tolist(),
            "solar": latest.sort_values("solar_share", ascending=False).head(3)["province"].tolist(),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
