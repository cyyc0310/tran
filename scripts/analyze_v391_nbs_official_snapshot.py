#!/usr/bin/env python3
"""Cross-check V391 first-party NBS data and recompute coarse threshold states."""
from __future__ import annotations

import csv
import argparse
import json
import re
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v391-nbs-official-provincial-snapshot-r1"
SNAPSHOT = OUT / "official_current_values.csv"
MIX = OUT / "provincial_mix_and_closure.csv"
BASELINE = ROOT / "results/v320-oreo-official-monthly-generation-audit-r1/cn_prov_monthly_generation_by_tech.csv"
USER_FILES = {
    "thermal": Path.home() / "Desktop/火电-分省月度数据.csv",
    "wind": Path.home() / "Desktop/风电-分省月度数据.csv",
    "solar": Path.home() / "Desktop/太阳能-分省月度数据.csv",
}
PROVINCES = ("Shanghai", "Sichuan", "Jilin", "Fujian", "Qinghai")
TECHS = ("total", "thermal", "hydro", "nuclear", "wind", "solar")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        return
    fields = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_user_monthly_files() -> list[dict]:
    rows_out = []
    for tech, path in USER_FILES.items():
        rows = list(csv.reader(path.open(encoding="utf-8-sig", newline="")))
        header = [x.strip().strip("\t") for x in rows[2]]
        for row in rows[3:]:
            if not row or "当期值" not in row[0]:
                continue
            for label, value in zip(header[1:], row[1:]):
                match = re.fullmatch(r"(\d{4})年(\d{1,2})月", label.strip())
                if not match or not value.strip():
                    continue
                rows_out.append({
                    "province": "Shanghai",
                    "technology": tech,
                    "year": int(match.group(1)),
                    "month": int(match.group(2)),
                    "user_value_gwh": float(value.strip()) * 100.0,
                    "user_file": path.name,
                })
    return rows_out


def main(input_dir: Path = OUT, output_dir: Path | None = None) -> None:
    global OUT, SNAPSHOT, MIX
    OUT = output_dir or input_dir
    SNAPSHOT = input_dir / "official_current_values.csv"
    MIX = input_dir / "provincial_mix_and_closure.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    official = read_csv(SNAPSHOT)
    baseline = read_csv(BASELINE)
    official_values = {}
    for row in official:
        key = (row["province"], int(row["year"]), int(row["month"]), row["technology"])
        value = float(row["value_100m_kwh"]) * 100.0 if row["value_100m_kwh"] else None
        official_values[key] = value

    baseline_values = defaultdict(list)
    for row in baseline:
        key = (row["province"], int(row["year"]), int(row["month"]), row["technology"].lower())
        baseline_values[key].append(float(row["generation_gwh"]))

    source_comparison = []
    for key, value in official_values.items():
        province, year, month, tech = key
        candidates = baseline_values.get(key, [])
        base = candidates[0] if len(candidates) == 1 else None
        rounded_base = (
            float((Decimal(str(base)) / Decimal(10)).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * Decimal(10))
            if base is not None else None
        )
        source_comparison.append({
            "province": province, "year": year, "month": month, "technology": tech,
            "official_value_gwh": "" if value is None else value,
            "oreo_value_gwh": "" if base is None else base,
            "difference_gwh": "" if value is None or base is None else value - base,
            "oreo_rounded_to_nbs_10_gwh_step": "" if rounded_base is None else rounded_base,
            "match_at_nbs_10_gwh_display_step": int(value is not None and rounded_base is not None and abs(value - rounded_base) < 1e-7),
            "within_half_publication_step_gwh": int(value is not None and base is not None and abs(value - base) <= 5.0 + 1e-9),
            "comparison_status": "both_missing" if value is None and base is None else
                "official_current_missing" if value is None else
                "oreo_missing" if base is None else "matched" if abs(value - base) < 1e-8 else "different",
        })
    write_csv(OUT / "official_vs_oreo_crosscheck.csv", source_comparison)

    official_nonmissing = [row for row in source_comparison if row["official_value_gwh"] != ""]
    valid_pairs = [row for row in official_nonmissing if row["oreo_value_gwh"] != ""]
    diffs = [abs(float(row["difference_gwh"])) for row in valid_pairs]
    differences_by_technology = defaultdict(list)
    for row in valid_pairs:
        differences_by_technology[row["technology"]].append(float(row["difference_gwh"]))

    user_rows = parse_user_monthly_files()
    user_comparison = []
    for row in user_rows:
        key = (row["province"], row["year"], row["month"], row["technology"])
        official_value = official_values.get(key)
        user_comparison.append({
            **row,
            "official_value_gwh": "" if official_value is None else official_value,
            "difference_gwh": "" if official_value is None else row["user_value_gwh"] - official_value,
            "match": int(official_value is not None and abs(row["user_value_gwh"] - official_value) < 1e-8),
        })
    write_csv(OUT / "user_shanghai_exports_crosscheck.csv", user_comparison)

    mix = read_csv(MIX)
    marginal_rows = [
        row for row in mix
        if row["thermal_100m_kwh"] and row["total_100m_kwh"] and float(row["total_100m_kwh"]) > 0
    ]
    five_rows = [row for row in mix if row["has_all_current_values"] == "1"]
    baseline_by_month = defaultdict(lambda: defaultdict(float))
    for row in baseline:
        if row["technology"].lower() in ("total", "thermal"):
            baseline_by_month[(int(row["year"]), int(row["month"]))][row["technology"].lower()] += float(row["generation_gwh"])

    states = []
    breakpoints = []
    for row in marginal_rows:
        province = row["province"]
        year, month = int(row["year"]), int(row["month"])
        thermal = float(row["thermal_100m_kwh"])
        total = float(row["total_100m_kwh"])
        local_share = thermal / total
        listed = float(row["listed_five_tech_sum_100m_kwh"]) if row["listed_five_tech_sum_100m_kwh"] else None
        five_share = thermal / listed if listed and listed > 0 else None
        national_thermal = baseline_by_month[(year, month)]["thermal"]
        national_total = baseline_by_month[(year, month)]["total"]
        if national_total <= 0:
            continue
        national_share = national_thermal / national_total
        local_certificate = 400 * local_share <= 300 < 1000 * local_share
        national_certificate = 400 * national_share <= 300 < 1000 * national_share
        five_certificate = 400 * five_share <= 300 < 1000 * five_share if five_share is not None else None
        states.append({
            "province": province, "year": year, "month": month,
            "local_thermal_share": local_share,
            "local_share_denominator": "official_total_generation",
            "five_technology_thermal_share": "" if five_share is None else five_share,
            "five_technology_tau300_certificate": "" if five_certificate is None else int(five_certificate),
            "national_pooled_thermal_share": national_share,
            "share_gap_percentage_points": 100 * (local_share - national_share),
            "local_tau300_cross_threshold_certificate": int(local_certificate),
            "national_tau300_cross_threshold_certificate": int(national_certificate),
            "status_disagreement": int(local_certificate != national_certificate),
            "local_thermal_support_lower_g_kwh": 400 * local_share,
        })
        breakpoints.extend((400 * local_share, 1000 * local_share, 400 * national_share, 1000 * national_share))
    write_csv(OUT / "threshold_states_tau300.csv", states)

    by_year_rows = []
    for year in sorted({row["year"] for row in states}):
        subset = [row for row in states if row["year"] == year]
        by_year_rows.append({
            "year": year,
            "province_months": len(subset),
            "local_certificate_count": sum(row["local_tau300_cross_threshold_certificate"] for row in subset),
            "national_certificate_count": sum(row["national_tau300_cross_threshold_certificate"] for row in subset),
            "local_national_disagreement_count": sum(row["status_disagreement"] for row in subset),
        })
    write_csv(OUT / "threshold_state_by_year.csv", by_year_rows)

    lower = max(value for value in breakpoints if value < 300)
    upper = min(value for value in breakpoints if value > 300)
    by_province = {}
    for province in PROVINCES:
        subset = [row for row in states if row["province"] == province]
        shares = [row["local_thermal_share"] for row in subset]
        by_province[province] = {
            "complete_months": len(subset),
            "thermal_share_min": min(shares),
            "thermal_share_max": max(shares),
            "thermal_share_mean": sum(shares) / len(shares),
            "local_certificate_months": sum(row["local_tau300_cross_threshold_certificate"] for row in subset),
            "national_certificate_months": sum(row["national_tau300_cross_threshold_certificate"] for row in subset),
            "local_national_disagreements": sum(row["status_disagreement"] for row in subset),
        }
    closure = [abs(float(row["total_minus_five_tech_100m_kwh"])) for row in five_rows]
    closure_rel = [
        abs(float(row["total_minus_five_tech_100m_kwh"]) / float(row["total_100m_kwh"]))
        for row in five_rows
    ]
    denominator_pairs = [row for row in states if row["five_technology_tau300_certificate"] != ""]
    nonzero_by_technology = {
        tech: sum(abs(float(row["difference_gwh"])) > 1e-8 for row in valid_pairs if row["technology"] == tech)
        for tech in sorted({row["technology"] for row in valid_pairs})
    }
    summary = {
        "official_snapshot_rows": len(official),
        "official_nonmissing_rows": sum(value is not None for value in official_values.values()),
        "official_missing_current_rows": sum(value is None for value in official_values.values()),
        "thermal_and_total_complete_province_months": len(states),
        "five_technology_complete_province_months": len(five_rows),
        "official_vs_oreo_valid_pairs": len(valid_pairs),
        "official_vs_oreo_nonzero_differences": sum(abs(float(row["difference_gwh"])) > 1e-8 for row in valid_pairs),
        "official_vs_oreo_max_abs_difference_gwh": max(diffs) if diffs else None,
        "official_vs_oreo_within_half_nbs_display_step_5_gwh": sum(row["within_half_publication_step_gwh"] for row in valid_pairs),
        "official_vs_oreo_match_after_rounding_to_10_gwh_nbs_display_step": sum(row["match_at_nbs_10_gwh_display_step"] for row in valid_pairs),
        "official_vs_oreo_valid_pairs_by_technology": {tech: len(values) for tech, values in differences_by_technology.items()},
        "official_vs_oreo_nonzero_differences_by_technology": nonzero_by_technology,
        "shanghai_user_export_values": len(user_comparison),
        "shanghai_user_export_exact_matches": sum(row["match"] for row in user_comparison),
        "shanghai_user_export_max_abs_difference_gwh": max(abs(float(row["difference_gwh"])) for row in user_comparison),
        "max_abs_five_technology_closure_100m_kwh": max(closure),
        "max_abs_closure_as_fraction_of_total": max(closure_rel),
        "tau_g_kwh": 300,
        "local_certificate_count": sum(row["local_tau300_cross_threshold_certificate"] for row in states),
        "national_certificate_count": sum(row["national_tau300_cross_threshold_certificate"] for row in states),
        "local_national_disagreement_count": sum(row["status_disagreement"] for row in states),
        "five_technology_denominator_comparison_months": len(denominator_pairs),
        "five_tech_vs_total_denominator_state_disagreements": sum(
            row["five_technology_tau300_certificate"] != row["local_tau300_cross_threshold_certificate"]
            for row in denominator_pairs
        ),
        "maximal_tau_interval_around_300_with_unchanged_local_and_national_states_g_kwh": [lower, upper],
        "by_province": by_province,
        "interpretation_limit": "This replicates a coarse mean-support threshold certificate on a current official NBS snapshot. Primary thermal shares use the official reported total denominator; a complete-five-technology denominator is reported as sensitivity. It is not an hourly CIF forecast, hourly K sharp set, probability coverage result, or independent source validation of the OREO transcription. Historical values are not as-published vintages.",
    }
    (OUT / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=OUT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
