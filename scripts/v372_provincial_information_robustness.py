"""V372: robustness of provincial-vs-national monthly thermal-share certificates."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/v364-china-nbs-threshold-crossing-certificate-r1/province_period_certificates.csv"
OUT = ROOT / "results/v372-provincial-information-robustness-r1"
SUPPORTS = {
    "base_400_1000": (400.0, 1000.0),
    "wide_350_1100": (350.0, 1100.0),
    "narrow_700_900": (700.0, 900.0),
    "zero_floor_0_1100": (0.0, 1100.0),
}
THRESHOLDS = (100.0, 300.0, 500.0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def certificate(share: pd.Series, low: float, high: float, threshold: float) -> pd.Series:
    """V364 sufficient certificate for a full [0,H] feasible K range."""
    return (low * share <= threshold) & (threshold < high * share)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT)
    base = data.loc[
        data["support"].eq("base_400_1000") & data["threshold_g_kwh"].eq(300)
    ].copy()
    if base["province"].nunique() != 31 or base.groupby(["year", "reported_month_bin"]).size().nunique() != 1:
        raise ValueError("Expected balanced 31-province panel for each year-period bin")

    pooled = (
        base.groupby(["year", "reported_month_bin"], as_index=False)
        .agg(thermal_gwh=("reported_thermal_gwh", "sum"), total_gwh=("reported_total_gwh", "sum"))
    )
    pooled["national_thermal_share"] = pooled["thermal_gwh"] / pooled["total_gwh"]
    panel = base[[
        "province", "year", "reported_month_bin", "period_hours", "thermal_share"
    ]].merge(
        pooled[["year", "reported_month_bin", "national_thermal_share"]],
        on=["year", "reported_month_bin"], validate="many_to_one",
    )

    detailed_rows: list[dict[str, object]] = []
    yearly_rows: list[dict[str, object]] = []
    for support, (low, high) in SUPPORTS.items():
        for threshold in THRESHOLDS:
            local_cert = certificate(panel["thermal_share"], low, high, threshold)
            pooled_cert = certificate(panel["national_thermal_share"], low, high, threshold)
            frame = panel.assign(local_cert=local_cert, pooled_cert=pooled_cert)
            resolved = frame["pooled_cert"] & ~frame["local_cert"]
            newly_ambiguous = frame["local_cert"] & ~frame["pooled_cert"]
            detailed_rows.append({
                "support": support,
                "low_ef_g_kwh": low,
                "high_ef_g_kwh": high,
                "threshold_g_kwh": threshold,
                "province_periods": int(len(frame)),
                "national_full_range_certificates": int(frame["pooled_cert"].sum()),
                "local_full_range_certificates": int(frame["local_cert"].sum()),
                "national_certificates_exited_under_local_mix": int(resolved.sum()),
                "local_certificates_not_seen_under_national_mix": int(newly_ambiguous.sum()),
                "fraction_national_certificates_exited": float(resolved.sum() / max(frame["pooled_cert"].sum(), 1)),
                "provinces_with_any_local_certificate": int(frame.loc[frame["local_cert"], "province"].nunique()),
            })
            for year, group in frame.groupby("year"):
                yearly_rows.append({
                    "support": support,
                    "low_ef_g_kwh": low,
                    "high_ef_g_kwh": high,
                    "threshold_g_kwh": threshold,
                    "year": int(year),
                    "national_full_range_certificates": int(group["pooled_cert"].sum()),
                    "local_full_range_certificates": int(group["local_cert"].sum()),
                    "national_certificates_exited_under_local_mix": int((group["pooled_cert"] & ~group["local_cert"]).sum()),
                    "provinces_with_any_local_certificate": int(group.loc[group["local_cert"], "province"].nunique()),
                })

    detailed = pd.DataFrame(detailed_rows)
    yearly = pd.DataFrame(yearly_rows)
    detailed.to_csv(OUT / "support_threshold_comparison.csv", index=False)
    yearly.to_csv(OUT / "yearly_comparison.csv", index=False)
    summary = {
        "experiment": "V372 provincial versus pooled national monthly thermal-share certificate robustness",
        "input": str(INPUT.relative_to(ROOT)),
        "input_source_scope": "NBS-derived OREO transcription; above-designated-size scope; not raw portal response",
        "n_province_periods": int(len(panel)),
        "supports": {key: list(value) for key, value in SUPPORTS.items()},
        "thresholds_g_kwh": list(THRESHOLDS),
        "sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (INPUT, ROOT / "docs/V372_PROVINCIAL_INFORMATION_ROBUSTNESS_PROTOCOL.md", Path(__file__))
        },
        "combinations_with_local_certificate_reduction": int((detailed["local_full_range_certificates"] < detailed["national_full_range_certificates"]).sum()),
        "combinations_total": int(len(detailed)),
        "combination_table": detailed.to_dict(orient="records"),
        "interpretation": "Finite-panel structural certificate sensitivity only; local removal of this witness is not proof of narrow true K bounds, predictive accuracy, or interval coverage.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
