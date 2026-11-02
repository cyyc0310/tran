"""Audit the public NBS input snapshot against the frozen China forecast origin."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from transcif.research.public_inference import nbs_months, verify_input_seal


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["total_100m_kwh", "thermal_100m_kwh", "hydro_100m_kwh",
          "nuclear_100m_kwh", "wind_100m_kwh", "solar_100m_kwh"]
EXPORTS = {"thermal": "thermal.csv", "wind": "wind.csv", "solar": "solar.csv"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_current_value(path: Path, label: str) -> tuple[str, float]:
    """Read the first (latest-month) 当期值 from the user's NBS tab/comma export."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    header = lines[2].split("\t")
    month_match = re.search(r"(\d{4})年(\d{1,2})月", header[1])
    if not month_match:
        raise ValueError(f"cannot identify latest month in {path}")
    row = next((line for line in lines[3:] if line.startswith(label)), None)
    if row is None:
        raise ValueError(f"missing {label!r} in {path}")
    parts = row.split("\t")
    if "当期值" not in parts[0]:
        raise ValueError(f"expected 当期值 row in {path}")
    value = next((part.strip() for part in parts[1].split(",") if part.strip()), None)
    if value is None:
        raise ValueError(f"missing latest current value in {path}")
    return f"{month_match.group(1)}-{int(month_match.group(2)):02d}", float(value)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--panel", type=Path, default=ROOT / "results/v394-nbs-all-province-long-panel-r1/provincial_mix_and_closure.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/v394-nbs-all-province-long-panel-r1/manifest.json")
    parser.add_argument("--seal", type=Path, default=ROOT / "docs/evidence/shanghai-public-inference-r2/nbs-public-input-seal-r1.json")
    parser.add_argument("--exports", type=Path, default=ROOT / "data/nbs_user_exports/shanghai")
    parser.add_argument("--accepted-months", type=Path, default=ROOT / "docs/evidence/china-province-public-inference-r2/accepted_months.csv")
    parser.add_argument("--origin", default="2026-11-01T00:00:00+08:00")
    parser.add_argument("--output", type=Path, default=ROOT / "results/china-nbs-prospective-origin-audit-r1")
    args = parser.parse_args()

    origin = pd.Timestamp(args.origin)
    if origin.tzinfo is None:
        raise ValueError("origin must include a timezone")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    seal = json.loads(args.seal.read_text(encoding="utf-8"))
    available = verify_input_seal(seal, args.panel.read_bytes(), args.manifest.read_bytes())
    retrieved = pd.Timestamp(manifest["retrieved_at_utc"])
    if retrieved.tzinfo is None or available > origin or retrieved > origin:
        raise ValueError("source retrieval and sealed availability must precede forecast origin")

    panel = pd.read_csv(args.panel)
    snapshot = panel.loc[panel.period_code.astype(str).eq("202609MM")].copy()
    if len(snapshot) != 31 or snapshot.province.nunique() != 31:
        raise ValueError("2026-09 snapshot must contain one row for each of 31 provinces")
    if not snapshot[FIELDS].notna().all().all():
        raise ValueError("2026-09 snapshot has missing generation values")
    if (snapshot[FIELDS] < 0).any().any():
        raise ValueError("generation fields cannot be negative")
    snapshot["closure_residual_100m_kwh"] = snapshot["total_100m_kwh"] - snapshot[[
        "thermal_100m_kwh", "hydro_100m_kwh", "nuclear_100m_kwh", "wind_100m_kwh", "solar_100m_kwh"
    ]].sum(axis=1)
    if (snapshot.closure_residual_100m_kwh.abs() > 0.30000001).any():
        raise ValueError("2026-09 technology totals do not close within the documented tolerance")

    # Reapply the production input contract, including exact t-2 month selection.
    snapshot_at = available
    selected_rows = []
    for province in sorted(panel.province.dropna().unique()):
        months, audit = nbs_months(panel, province, origin, snapshot_at)
        if audit["latest_allowed_month"] != "2026-09" or str(months.index.max()) != "2026-09":
            raise ValueError(f"{province}: production input contract did not select 2026-09")
        selected_rows.append({"province": province, "latest_allowed_month": audit["latest_allowed_month"],
                              "eligible_months": audit["eligible_months"],
                              "recent_six_months_complete": all(
                                  pd.Period(f"2026-{month:02d}", freq="M") in months.index
                                  for month in range(4, 10))})
    provinces = pd.DataFrame(selected_rows)

    shanghai = snapshot.loc[snapshot.province.eq("Shanghai")].iloc[0]
    export_rows = []
    for technology, filename in EXPORTS.items():
        export_month, export_value = export_current_value(args.exports / filename, {
            "thermal": "火力发电量", "wind": "风力发电量", "solar": "太阳能发电量"
        }[technology])
        panel_value = float(shanghai[f"{technology}_100m_kwh"])
        export_rows.append({"technology": technology, "export_month": export_month,
                            "export_value_100m_kwh": export_value,
                            "panel_month": "2026-09", "panel_value_100m_kwh": panel_value,
                            "matches": export_month == "2026-09" and abs(export_value-panel_value) < 1e-10})
    export_match = pd.DataFrame(export_rows)
    if not export_match.matches.all() or float(shanghai.hydro_100m_kwh) != 0 or float(shanghai.nuclear_100m_kwh) != 0:
        raise ValueError("Shanghai user export comparison or stated hydro/nuclear zero check failed")

    accepted = pd.read_csv(args.accepted_months)
    accepted_shanghai = accepted.loc[accepted.province.eq("Shanghai")]
    accepted_sep = accepted_shanghai.loc[accepted_shanghai.period.astype(str).eq("2026-09")]
    if len(accepted_sep) != 1:
        raise ValueError("existing inference artifact must contain one accepted Shanghai September row")

    args.output.mkdir(parents=True, exist_ok=False)
    snapshot.sort_values("province").to_csv(args.output / "province_snapshot.csv", index=False)
    provinces.to_csv(args.output / "province_origin_contract.csv", index=False)
    export_match.to_csv(args.output / "shanghai_user_export_match.csv", index=False)
    retrieved_gap = (origin.tz_convert("UTC") - retrieved.tz_convert("UTC")).total_seconds() / 3600
    sealed_gap = (origin.tz_convert("UTC") - available.tz_convert("UTC")).total_seconds() / 3600
    summary = {
        "audit": "prospective public NBS input availability and schema audit",
        "origin": origin.isoformat(), "latest_allowed_period": "2026-09",
        "source": manifest.get("source"), "source_entry_url": manifest.get("entry_url"),
        "official_nbs_release_date": "2026-10-19",
        "official_release_url": "https://www.stats.gov.cn/zwfwck/sjfb/202610/t20261019_1965613.html",
        "retrieved_at_utc": retrieved.isoformat(), "retrieval_lead_hours": retrieved_gap,
        "sealed_input_available_at_utc": available.isoformat(), "seal_lead_hours": sealed_gap,
        "retrieval_and_seal_precede_origin": retrieved < origin and available < origin,
        "panel_sha256": digest(args.panel), "manifest_sha256": digest(args.manifest),
        "seal_sha256": digest(args.seal), "province_count": int(len(snapshot)),
        "all_six_values_complete": bool(snapshot[FIELDS].notna().all().all()),
        "max_abs_closure_residual_100m_kwh": float(snapshot.closure_residual_100m_kwh.abs().max()),
        "provinces_with_complete_apr_sep_window": int(provinces.recent_six_months_complete.sum()),
        "production_contract_latest_period_all_31": bool(provinces.latest_allowed_month.eq("2026-09").all()),
        "shanghai_september": {name: float(shanghai[f"{name}_100m_kwh"]) for name in [
            "total", "thermal", "hydro", "nuclear", "wind", "solar"]},
        "shanghai_hydro_nuclear_are_reported_zero": float(shanghai.hydro_100m_kwh) == 0 and float(shanghai.nuclear_100m_kwh) == 0,
        "shanghai_user_exports_match": bool(export_match.matches.all()),
        "target_hourly_labels_read": False, "target_model_fitting": False,
        "historical_as_issued_vintage_certified": False,
        "scope": "NBS above-designated-size industrial generation; thermal is aggregate, not coal/gas split",
        "interpretation": "confirms prospective availability and input execution only; does not establish forecast accuracy or China interval coverage",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    inputs = [args.panel, args.manifest, args.seal, args.accepted_months, *[args.exports / name for name in EXPORTS.values()]]
    provenance = {str(path.resolve()): digest(path) for path in inputs}
    provenance[str(Path(__file__).resolve())] = digest(Path(__file__).resolve())
    (args.output / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ["retrieval_lead_hours", "seal_lead_hours", "province_count",
          "provinces_with_complete_apr_sep_window", "max_abs_closure_residual_100m_kwh",
          "shanghai_user_exports_match"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
