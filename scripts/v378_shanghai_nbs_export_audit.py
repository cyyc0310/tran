"""Audit user-exported Shanghai NBS monthly generation tables against V320."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/nbs_user_exports/shanghai"
BASELINE = ROOT / "results/v320-oreo-official-monthly-generation-audit-r1/cn_prov_monthly_generation_by_tech.csv"
OUT = ROOT / "results/v378-shanghai-nbs-export-audit-r1"
FILES = {"thermal": "thermal.csv", "solar": "solar.csv", "wind": "wind.csv"}
TECH_LABELS = {
    "thermal": "火力发电量",
    "solar": "太阳能发电量",
    "wind": "风力发电量",
}


def clean(value: str) -> str:
    return value.strip().strip("\t").strip()


def parse_month(label: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"(\d{4})年(\d{1,2})月", label.strip())
    return (int(match.group(1)), int(match.group(2))) if match else None


def parse_export(technology: str, path: Path) -> tuple[list[dict], list[dict]]:
    rows = list(csv.reader(path.open(encoding="utf-8-sig", newline="")))
    header = [clean(cell) for cell in rows[2]]
    observations: list[dict] = []
    cumulative: list[dict] = []
    for row in rows[3:]:
        if not row:
            continue
        name = clean(row[0])
        if not name.startswith(TECH_LABELS[technology]):
            continue
        field = "current" if "当期值" in name else "cumulative" if "累计值" in name else None
        if field is None:
            continue
        values = [clean(cell) for cell in row[1:]]
        for label, raw_value in zip(header[1:], values):
            period = parse_month(label)
            if period is None or not raw_value:
                continue
            year, month = period
            record = {
                "province": "Shanghai",
                "technology": technology,
                "year": year,
                "month": month,
                "value_100m_kwh": float(raw_value),
                "unit": "100 million kWh",
                "observation_kind": field,
                "source_file": path.name,
            }
            (observations if field == "current" else cumulative).append(record)
    return observations, cumulative


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    observations: list[dict] = []
    cumulative: list[dict] = []
    source_hashes = {}
    for technology, filename in FILES.items():
        path = RAW / filename
        source_hashes[filename] = sha256(path)
        current, ytd = parse_export(technology, path)
        observations.extend(current)
        cumulative.extend(ytd)

    obs = pd.DataFrame(observations)
    cum = pd.DataFrame(cumulative)
    obs.to_csv(OUT / "reported_monthly_values.csv", index=False)
    cum.to_csv(OUT / "reported_cumulative_values.csv", index=False)

    wide = obs.pivot(index=["year", "month"], columns="technology", values="value_100m_kwh").reset_index()
    for technology in ("thermal", "solar", "wind"):
        if technology not in wide:
            wide[technology] = pd.NA
    wide["hydro"] = 0.0
    wide["nuclear"] = 0.0
    wide["listed_technology_sum_100m_kwh"] = wide[["thermal", "hydro", "nuclear", "wind", "solar"]].sum(axis=1, min_count=5)
    wide["thermal_share_of_listed_five_tech"] = wide["thermal"] / wide["listed_technology_sum_100m_kwh"]
    wide["reported_period_semantics"] = "single-month current value"

    pooled = pd.read_csv(BASELINE)
    pooled = pooled.loc[pooled["technology"].isin(["thermal", "total"])].copy()
    pooled["year"] = pooled["year"].astype(int)
    pooled["month"] = pooled["month"].astype(int)
    pooled = pooled.pivot_table(
        index=["year", "month"], columns=["province", "technology"], values="generation_gwh", aggfunc="sum"
    )
    comparison_rows = []
    for row in wide.itertuples(index=False):
        period = (int(row.year), int(row.month))
        data_row = pooled.loc[period]
        national_thermal = float(data_row.xs("thermal", level=1).sum())
        national_total = float(data_row.xs("total", level=1).sum())
        national_share = national_thermal / national_total
        local_share = float(row.thermal_share_of_listed_five_tech)
        comparison_rows.append(
            {
                "year": period[0],
                "month": period[1],
                "national_thermal_share_v320": national_share,
                "shanghai_thermal_share_user_export": local_share,
                "local_minus_national_share": local_share - national_share,
                "national_mean_cif_support_low_g_kwh": 400.0 * national_share,
                "shanghai_mean_cif_support_low_g_kwh": 400.0 * local_share,
                "national_cross_threshold_certificate": 400.0 * national_share <= 300.0 < 1000.0 * national_share,
                "shanghai_cross_threshold_certificate": 400.0 * local_share <= 300.0 < 1000.0 * local_share,
                "shanghai_mean_support_entirely_above_300": 400.0 * local_share > 300.0,
            }
        )
    local_vs_national = pd.DataFrame(comparison_rows)
    local_vs_national.to_csv(OUT / "local_vs_national_mix_certificate.csv", index=False)
    wide.to_csv(OUT / "monthly_mix_summary.csv", index=False)

    base = pd.read_csv(BASELINE)
    base = base.loc[base["province"].eq("Shanghai")].copy()
    base["year"] = base["year"].astype(int)
    base["month"] = base["month"].astype(int)
    base["technology"] = base["technology"].str.lower()
    base = base.pivot(index=["year", "month"], columns="technology", values="generation_gwh").reset_index()
    long_base = base.melt(id_vars=["year", "month"], var_name="technology", value_name="baseline_gwh")
    crosscheck = obs.merge(long_base, on=["year", "month", "technology"], how="left", validate="one_to_one")
    crosscheck["export_gwh"] = crosscheck["value_100m_kwh"] * 100.0
    crosscheck["difference_gwh"] = crosscheck["export_gwh"] - crosscheck["baseline_gwh"]
    crosscheck["match_within_0_01_gwh"] = crosscheck["difference_gwh"].abs().le(0.01)
    crosscheck.to_csv(OUT / "v320_crosscheck.csv", index=False)

    base_wide = base.set_index(["year", "month"])
    closure_rows = []
    for _, row in wide.iterrows():
        key = (int(row["year"]), int(row["month"]))
        reference = base_wide.loc[key]
        category_names = ["thermal", "hydro", "nuclear", "wind", "solar"]
        category_sum = sum(float(reference[name]) for name in category_names)
        total = float(reference["total"])
        closure_rows.append(
            {
                "year": key[0],
                "month": key[1],
                "v320_technology_sum_gwh": category_sum,
                "v320_total_gwh": total,
                "total_minus_five_tech_gwh": total - category_sum,
            }
        )
    closure = pd.DataFrame(closure_rows)
    closure.to_csv(OUT / "five_technology_closure.csv", index=False)

    # Compare consecutive cumulative values with the reported single-month values.
    cum_audit_rows = []
    for technology, group in cum.groupby("technology"):
        group = group.sort_values(["year", "month"])
        for _, row in group.iterrows():
            prev = group.loc[(group["year"].eq(row["year"])) & (group["month"].eq(row["month"] - 1))]
            if prev.empty:
                continue
            current = obs.loc[
                obs["technology"].eq(technology)
                & obs["year"].eq(row["year"])
                & obs["month"].eq(row["month"]),
                "value_100m_kwh",
            ]
            if current.empty:
                continue
            increment = row["value_100m_kwh"] - prev.iloc[0]["value_100m_kwh"]
            cum_audit_rows.append(
                {
                    "technology": technology,
                    "year": int(row["year"]),
                    "month": int(row["month"]),
                    "cumulative_increment_100m_kwh": increment,
                    "reported_current_100m_kwh": float(current.iloc[0]),
                    "increment_minus_current_100m_kwh": increment - float(current.iloc[0]),
                }
            )
    cum_audit = pd.DataFrame(cum_audit_rows)
    cum_audit.to_csv(OUT / "cumulative_increment_audit.csv", index=False)

    missing_current = []
    for technology, filename in FILES.items():
        path = RAW / filename
        rows = list(csv.reader(path.open(encoding="utf-8-sig", newline="")))
        header = [clean(cell) for cell in rows[2]]
        for row in rows[3:]:
            if row and "当期值" in clean(row[0]):
                values = [clean(cell) for cell in row[1:]]
                missing_current.extend(
                    {"technology": technology, "year": period[0], "month": period[1]}
                    for label, value in zip(header[1:], values)
                    if (period := parse_month(label)) and not value
                )
    pd.DataFrame(missing_current).to_csv(OUT / "missing_current_values.csv", index=False)

    mean_share = float(wide["thermal_share_of_listed_five_tech"].mean())
    # For each period, the full-range certificate holds for thresholds tau in
    # [400*s, 1000*s). Intersect these intervals across all paired periods.
    robust_national_low = float(local_vs_national["national_mean_cif_support_low_g_kwh"].max())
    robust_national_high = float(
        (1000.0 * local_vs_national["national_thermal_share_v320"]).min()
    )
    robust_shanghai_low = float(local_vs_national["shanghai_mean_cif_support_low_g_kwh"].min())
    robust_shanghai_high = float(
        (1000.0 * local_vs_national["shanghai_thermal_share_user_export"]).max()
    )
    threshold_frontier = {
        "national_all_periods_certificate_tau_interval_g_kwh": [robust_national_low, robust_national_high],
        "shanghai_all_periods_no_certificate_below_tau_g_kwh": [0.0, robust_shanghai_low],
        "common_tau_interval_national_certificate_shanghai_no_certificate_g_kwh": [
            robust_national_low,
            min(robust_national_high, robust_shanghai_low),
        ],
    }
    (OUT / "threshold_frontier.json").write_text(
        json.dumps(threshold_frontier, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "experiment": "V378 direct NBS Shanghai export audit and consistency check",
        "province_assignment": "Shanghai, based on the user's statement; the CSV files themselves do not contain a province-name field",
        "source": "User-provided exports from NBS fsMonthData provincial monthly data portal",
        "unit_conversion": "1 unit of 100 million kWh = 100 GWh",
        "hydro_nuclear": "Set to zero based on the user's explicit statement; no separate hydro or nuclear export files were supplied",
        "single_month_observations": int(len(obs)),
        "single_month_periods": int(len(wide)),
        "single_month_period_labels": [f"{int(r.year)}-{int(r.month):02d}" for r in wide.itertuples()],
        "2026_february_semantics": "Current-value fields are blank. The available February cumulative value is January-February combined and is retained as a cumulative anchor, not a February monthly observation.",
        "mean_thermal_share_of_five_tech_sum": mean_share,
        "min_thermal_share": float(wide["thermal_share_of_listed_five_tech"].min()),
        "max_thermal_share": float(wide["thermal_share_of_listed_five_tech"].max()),
        "national_thermal_share_min_max": [
            float(local_vs_national["national_thermal_share_v320"].min()),
            float(local_vs_national["national_thermal_share_v320"].max()),
        ],
        "same_period_national_full_range_certificates": int(local_vs_national["national_cross_threshold_certificate"].sum()),
        "same_period_shanghai_full_range_certificates": int(local_vs_national["shanghai_cross_threshold_certificate"].sum()),
        "same_period_shanghai_mean_support_entirely_above_300": int(local_vs_national["shanghai_mean_support_entirely_above_300"].sum()),
        "same_periods_compared": int(len(local_vs_national)),
        "threshold_frontier_across_all_11_periods": threshold_frontier,
        "v320_overlap_rows": int(len(crosscheck)),
        "v320_overlap_exact_matches_within_0_01_gwh": int(crosscheck["match_within_0_01_gwh"].sum()),
        "v320_overlap_max_abs_difference_gwh": float(crosscheck["difference_gwh"].abs().max()),
        "v320_total_vs_five_tech_max_abs_gap_gwh": float(closure["total_minus_five_tech_gwh"].abs().max()),
        "cumulative_increment_rows_checked": int(len(cum_audit)),
        "cumulative_increment_max_abs_difference_100m_kwh": float(cum_audit["increment_minus_current_100m_kwh"].abs().max()) if len(cum_audit) else None,
        "source_sha256": source_hashes,
        "interpretation": "These files add source-verifiable recent Shanghai coarse generation composition, but no hourly CIF labels, thermal fuel split, imports, or operational dispatch. They support an input audit and information-budget analysis, not direct hourly forecast validation.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
