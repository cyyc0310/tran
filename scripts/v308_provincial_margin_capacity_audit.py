"""V308: compare provincial send/receipt totals with regional transfer margins."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROVINCE_RECEIPTS = ROOT / "data/china/cec_province_crossprov_receipts_2023.csv"
REGION_MARGINS = ROOT / "data/china/mee_regional_factor_trade_margins_2023.csv"
OUT = ROOT / "results/v308-provincial-margin-capacity-audit-r1"

# Province assignment for the seven MEE regional-grid factors used in V304.
REGIONS = {
    "North": ["Beijing", "Tianjin", "Hebei", "Shanxi", "Inner Mongolia", "Shandong"],
    "Northeast": ["Liaoning", "Jilin", "Heilongjiang"],
    "East": ["Shanghai", "Jiangsu", "Zhejiang", "Anhui", "Fujian"],
    "Central": ["Henan", "Hubei", "Hunan", "Jiangxi"],
    "Northwest": ["Shaanxi", "Gansu", "Qinghai", "Ningxia", "Xinjiang"],
    "South": ["Guangdong", "Guangxi", "Yunnan", "Guizhou", "Hainan"],
    "Southwest": ["Chongqing", "Sichuan", "Tibet"],
}

# Table 3 of Quan et al. (2025), unit 10,000 kWh; rows marked “/” are zero.
PROVINCE_SENDS = {
    "Inner Mongolia": 28325180, "Sichuan": 19314233, "Yunnan": 17964440,
    "Shanxi": 12785500, "Xinjiang": 12303595, "Shaanxi": 9824895,
    "Anhui": 9123216, "Ningxia": 9077121, "Gansu": 8927043,
    "Hubei": 8604628, "Guizhou": 6219885, "Jilin": 4010727,
    "Jiangsu": 3149781, "Qinghai": 2583723, "Heilongjiang": 2475438,
    "Fujian": 2013132, "Zhejiang": 1639141, "Hunan": 1211877,
    "Guangxi": 969151, "Chongqing": 905643, "Shanghai": 425586,
    "Henan": 396159, "Jiangxi": 352876, "Hebei": 297287,
    "Tibet": 256973, "Guangdong": 58750, "Hainan": 32782,
    "Liaoning": 31289, "Shandong": 13773,
}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {OUT}")
    province = pd.read_csv(PROVINCE_RECEIPTS)
    # The V303 transcription file covers 30 rows; Quan et al. Table 3 also
    # reports Tibet (送出 256,973 / 受入 217,422, both in 10,000 kWh).
    province = pd.concat([province, pd.DataFrame([{
        "province": "Tibet", "province_zh": "西藏", "consumption_100m_kwh": pd.NA,
        "receipt_10k_kwh": 217422,
    }])], ignore_index=True)
    region = pd.read_csv(REGION_MARGINS).set_index("region")
    if province.province.duplicated().any():
        raise ValueError("duplicate province rows")
    if set(PROVINCE_SENDS) - set(province.province):
        raise ValueError("sendout list contains unrecognized province")
    province["send_10k_kwh"] = province.province.map(PROVINCE_SENDS).fillna(0).astype(int)
    if set(sum(REGIONS.values(), [])) != set(province.province):
        raise ValueError("region mapping must partition the province table")

    rows = []
    for name, members in REGIONS.items():
        subset = province[province.province.isin(members)]
        provincial_send = int(subset.send_10k_kwh.sum())
        provincial_receipt = int(subset.receipt_10k_kwh.sum())
        regional_send = int(region.loc[name, "outgoing_10k_kwh"])
        regional_receipt = int(region.loc[name, "incoming_10k_kwh"])
        rows.append({
            "region": name,
            "province_crossprov_send_total_10k_kwh": provincial_send,
            "regional_crossregion_send_margin_10k_kwh": regional_send,
            "send_margin_to_provincial_total_ratio": regional_send / provincial_send if provincial_send else None,
            "province_crossprov_receipt_total_10k_kwh": provincial_receipt,
            "regional_crossregion_receipt_margin_10k_kwh": regional_receipt,
            "receipt_margin_to_provincial_total_ratio": regional_receipt / provincial_receipt if provincial_receipt else None,
            "regional_send_upper_bound_redundant": regional_send <= provincial_send,
            "regional_receipt_upper_bound_redundant": regional_receipt <= provincial_receipt,
        })
    result = pd.DataFrame(rows)
    OUT.mkdir(parents=True)
    result.to_csv(OUT / "regional_vs_provincial_margins.csv", index=False)
    summary = {
        "status": "AGGREGATED_PROVINCE_CAPACITY_AUDIT_NOT_PROVINCE_OD_RECONSTRUCTION",
        "province_send_total_10k_kwh": int(province.send_10k_kwh.sum()),
        "province_receipt_total_10k_kwh": int(province.receipt_10k_kwh.sum()),
        "regional_crossregion_total_10k_kwh": int(region.outgoing_10k_kwh.sum()),
        "province_send_receipt_gap_10k_kwh": int(province.send_10k_kwh.sum() - province.receipt_10k_kwh.sum()),
        "all_region_send_capacity_bounds_redundant": bool(result.regional_send_upper_bound_redundant.all()),
        "all_region_receipt_capacity_bounds_redundant": bool(result.regional_receipt_upper_bound_redundant.all()),
        "limitations": [
            "Province margins include a broader cross-province exchange scope than the regional cross-region table.",
            "No province-to-province origin-destination pairing is reported; region membership follows the seven-factor grid grouping.",
            "This audit tests only aggregate capacity ceilings; it does not show bilateral data have no value.",
            "Values are manually transcribed from the paper's Table 3 and retain its rounding/coverage.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(result.to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
