"""V366: quantify what province-level monthly generation mix adds to risk identification."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/v364-china-nbs-threshold-crossing-certificate-r1/province_period_certificates.csv"
OUT = ROOT / "results/v366-provincial-information-gain-r1"
LOW_EF = 400.0
HIGH_EF = 1000.0
THRESHOLD = 300.0


def crossing(share: pd.Series) -> pd.Series:
    """Strict event C > tau: low endpoint may equal tau, high endpoint must exceed it."""
    return (LOW_EF * share <= THRESHOLD) & (HIGH_EF * share > THRESHOLD)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT)
    data = data.loc[
        data["support"].eq("base_400_1000")
        & data["threshold_g_kwh"].eq(THRESHOLD)
    ].copy()

    national = (
        data.groupby(["year", "reported_month_bin"], as_index=False)
        .apply(
            lambda x: pd.Series(
                {
                    "national_thermal_share": x["reported_thermal_gwh"].sum()
                    / x["reported_total_gwh"].sum()
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    result = data.merge(national, on=["year", "reported_month_bin"], validate="many_to_one")
    result["provincial_full_range_certificate"] = crossing(result["thermal_share"])
    result["national_mix_full_range_certificate"] = crossing(result["national_thermal_share"])
    result["provincial_resolves_national_ambiguity"] = (
        result["national_mix_full_range_certificate"]
        & ~result["provincial_full_range_certificate"]
    )
    result.to_csv(OUT / "province_period_comparison.csv", index=False)

    yearly_rows = []
    for year, subset in result.groupby("year"):
        unresolved = subset.loc[subset["provincial_full_range_certificate"]]
        yearly_rows.append(
            {
                "year": int(year),
                "province_periods": int(len(subset)),
                "national_mix_full_range": int(subset["national_mix_full_range_certificate"].sum()),
                "provincial_mix_full_range": int(subset["provincial_full_range_certificate"].sum()),
                "resolved_by_provincial_mix": int(subset["provincial_resolves_national_ambiguity"].sum()),
                "provinces_with_provincial_certificate": int(unresolved["province"].nunique()),
            }
        )
    yearly = pd.DataFrame(yearly_rows)
    yearly.to_csv(OUT / "yearly_comparison.csv", index=False)

    summary = {
        "experiment": "V366 province-level versus pooled national monthly thermal-share information",
        "input": str(INPUT.relative_to(ROOT)),
        "input_scope": "NBS-derived above-designated-size provincial monthly generation, same source and periods for both arms",
        "thermal_ef_support_g_kwh": [LOW_EF, HIGH_EF],
        "strict_tail_threshold_g_kwh": THRESHOLD,
        "correct_full_range_condition": "low_EF * thermal_share <= threshold < high_EF * thermal_share",
        "n_province_periods": int(len(result)),
        "national_mix_full_range_province_periods": int(result["national_mix_full_range_certificate"].sum()),
        "province_mix_full_range_province_periods": int(result["provincial_full_range_certificate"].sum()),
        "province_mix_resolves_national_full_range_cases": int(result["provincial_resolves_national_ambiguity"].sum()),
        "province_periods_resolved_fraction_of_national_ambiguous": float(
            result["provincial_resolves_national_ambiguity"].sum()
            / max(result["national_mix_full_range_certificate"].sum(), 1)
        ),
        "yearly": yearly.to_dict(orient="records"),
        "interpretation": "Information-identification comparison only. The EF interval is a scenario, not locally validated. It does not establish actual CIF or hourly interval coverage. Provincial data have thermal/hydro/nuclear/wind/solar classes, not thermal fuel split.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
