#!/usr/bin/env python3
"""Source-only coarse fuel-mix prediction of target monthly daily-risk scales."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from openpyxl import load_workbook


BAS = ("BANC", "BPAT", "CISO", "ERCO", "FPL", "MISO", "NYIS")
TRAIN_YEARS = range(2019, 2024)
CALIBRATION_YEAR = 2024
DEFAULT_TEST_YEAR = 2025
ALPHA = 0.10
RIDGE_ALPHA = 10.0
LB_TO_G = 453.59237

CATEGORIES = {
    "coal": ("NG: COL",),
    "gas": ("NG: NG",),
    "oil": ("NG: OIL",),
    "nuclear": ("NG: NUC",),
    "hydro": ("NG: WAT", "NG: PS"),
    "wind": ("NG: WND", "NG: WNB"),
    "solar": ("NG: SUN", "NG: SNB"),
    "other": ("NG: GEO", "NG: OES", "NG: UES", "NG: OTH", "NG: UNK", "NG: BAT"),
}
ALL_FUEL_COLUMNS = tuple(column for cols in CATEGORIES.values() for column in cols)

# Observation fields: local-year, local-month, ordinal, weekday, hour-slot,
# generation-CI, consumption-CI, positive-generation, consumed-electricity,
# net-interchange, category fuel outputs, then the three BA fuel factors.
FACTOR_COLUMNS = ("CO2 Factor: COL", "CO2 Factor: NG", "CO2 Factor: OIL")


def parse_date(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).date()
        except ValueError:
            return None
    return None


def finite_number(value: object) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def read_ba(path: Path, max_year: int = 2026) -> list[tuple]:
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb["Published Hourly Data"]
    iterator = ws.iter_rows(values_only=True)
    headers = next(iterator)
    ix = {str(value): i for i, value in enumerate(headers) if value is not None}
    required = (
        "UTC time", "Local date", "Hour", "Positive Generation", "Consumed Electricity",
        "CO2 Emissions Intensity for Generated Electricity",
        "CO2 Emissions Intensity for Consumed Electricity", "Total interchange",
        *ALL_FUEL_COLUMNS,
        *FACTOR_COLUMNS,
    )
    missing = [name for name in required if name not in ix]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
    data = []
    for row in iterator:
        utc = row[ix["UTC time"]]
        local_day = parse_date(row[ix["Local date"]])
        if not isinstance(utc, datetime) or local_day is None or not (2019 <= local_day.year <= max_year):
            continue
        hour = finite_number(row[ix["Hour"]])
        if hour is None:
            continue
        values = [finite_number(row[ix[name]]) for name in ALL_FUEL_COLUMNS]
        factors = [finite_number(row[ix[name]]) for name in FACTOR_COLUMNS]
        data.append(
            (
                local_day.year,
                local_day.month,
                local_day.toordinal(),
                local_day.weekday(),
                (int(hour) - 1) % 24,
                finite_number(row[ix["CO2 Emissions Intensity for Generated Electricity"]]),
                finite_number(row[ix["CO2 Emissions Intensity for Consumed Electricity"]]),
                finite_number(row[ix["Positive Generation"]]),
                finite_number(row[ix["Consumed Electricity"]]),
                finite_number(row[ix["Total interchange"]]),
                *values,
                *factors,
            )
        )
    wb.close()
    return data


def category_month_features(rows: list[tuple], year: int, month: int, variant: str) -> tuple[list[float], float] | None:
    with_import = "net_import" in variant
    with_factors = "factors" in variant
    idx_gen = 7
    idx_ti = 9
    idx_fuel = 10
    positive_generation = 0.0
    sums = {name: 0.0 for name in CATEGORIES}
    reported_fuel = 0.0
    total_interchange = 0.0
    consumed = 0.0
    valid_interchange = True
    factor_sums = [0.0, 0.0, 0.0]
    factor_counts = [0, 0, 0]
    fuel_idx = {column: idx_fuel + i for i, column in enumerate(ALL_FUEL_COLUMNS)}
    factor_idx = idx_fuel + len(ALL_FUEL_COLUMNS)
    for row in rows:
        if row[0] != year or row[1] != month:
            continue
        gen = row[idx_gen]
        if gen is not None and gen > 0:
            positive_generation += gen
        for category, columns in CATEGORIES.items():
            for column in columns:
                value = row[fuel_idx[column]]
                if value is not None:
                    contribution = max(0.0, value)
                    sums[category] += contribution
                    reported_fuel += contribution
        for i in range(len(FACTOR_COLUMNS)):
            value = row[factor_idx + i]
            if value is not None:
                factor_sums[i] += value
                factor_counts[i] += 1
        ti = row[idx_ti]
        if ti is None:
            valid_interchange = False
        else:
            total_interchange += ti
        if row[8] is not None and row[8] > 0:
            consumed += row[8]
    if positive_generation <= 0 or reported_fuel <= 0:
        return None
    coverage = reported_fuel / positive_generation
    # EIA's positive fuel outputs should reconcile to positive generation. Do
    # not turn missing fuel-source values into a false composition signature.
    if not 0.95 <= coverage <= 1.05:
        return None
    total = sum(sums.values())
    shares = [sums[name] / total for name in CATEGORIES]
    features = shares + [math.log(positive_generation)]
    if with_factors:
        if any(count == 0 for count in factor_counts):
            return None
        features += [factor_sums[i] / factor_counts[i] for i in range(len(FACTOR_COLUMNS))]
    if with_import:
        if not valid_interchange or consumed <= 0:
            return None
        features.append(max(0.0, -total_interchange) / consumed)
    return features, coverage


def weighted_center(rows: list[tuple], year: int, month: int, scope: str) -> float | None:
    ci_idx, weight_idx = (5, 7) if scope == "generated" else (6, 8)
    numerator = denominator = 0.0
    for row in rows:
        if row[0] != year or row[1] != month:
            continue
        ci, weight = row[ci_idx], row[weight_idx]
        if ci is not None and weight is not None and weight > 0:
            numerator += ci * LB_TO_G * weight
            denominator += weight
    return numerator / denominator if denominator > 0 else None


def daily_month_scores(rows: list[tuple], year: int, month: int, center: float, scope: str) -> list[tuple[int, float]]:
    ci_idx = 5 if scope == "generated" else 6
    by_day: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        if row[0] != year or row[1] != month:
            continue
        ci = row[ci_idx]
        if ci is not None:
            by_day[row[2]].append((row[4], abs(ci * LB_TO_G - center)))
    result = []
    for ordinal, values in by_day.items():
        hours = [item[0] for item in values]
        if len(values) == 24 and len(set(hours)) == 24:
            result.append((ordinal, max(item[1] for item in values)))
    return result


def monthly_table(rows_by_ba: dict[str, list[tuple]], scope: str) -> tuple[dict[tuple[str, int, int], dict], dict[tuple[str, int, int], list[tuple[int, float]]]]:
    info: dict[tuple[str, int, int], dict] = {}
    days: dict[tuple[str, int, int], list[tuple[int, float]]] = {}
    for ba, rows in rows_by_ba.items():
        candidates = sorted({(row[0], row[1]) for row in rows})
        for year, month in candidates:
            center = weighted_center(rows, year, month, scope)
            if center is None:
                continue
            daily = daily_month_scores(rows, year, month, center, scope)
            if len(daily) < 20:
                continue
            risk = float(np.quantile([score for _, score in daily], 0.90, method="higher"))
            info[(ba, year, month)] = {"center": center, "risk": risk, "days": len(daily)}
            days[(ba, year, month)] = daily
    return info, days


def fit_ridge(train_x: list[list[float]], train_y: list[float], alpha: float = RIDGE_ALPHA) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(train_x, dtype=float)
    y = np.asarray(train_y, dtype=float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z = (x - mean) / scale
    design = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return mean, scale, coef


def predict_ridge(model: tuple[np.ndarray, np.ndarray, np.ndarray], features: list[float]) -> float:
    mean, scale, coef = model
    z = (np.asarray(features, dtype=float) - mean) / scale
    log_risk = float(coef[0] + z @ coef[1:])
    return max(1.0, math.exp(min(math.log(1000.0), max(math.log(1.0), log_risk))))


def quantile_higher(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("empty calibration sample")
    return float(np.quantile(np.asarray(values, dtype=float), probability, method="higher"))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("generated", "consumed"), default="generated")
    parser.add_argument("--include-factor-ablation", action="store_true")
    parser.add_argument("--selective-gate", action="store_true")
    parser.add_argument("--test-year", type=int, default=DEFAULT_TEST_YEAR)
    parser.add_argument("--fuel-mix-only", action="store_true")
    args = parser.parse_args()
    scope = args.scope
    test_year = args.test_year
    root = Path(__file__).resolve().parents[1]
    input_dir = root / "results" / "v272-eia-monitor"
    out_name = (
        "v278-selective-2026-confirmation" if args.selective_gate and test_year == 2026
        else "v277-selective-risk-abstention" if args.selective_gate
        else "v276-local-factor-risk-ablation" if args.include_factor_ablation
        else "v275-coarse-mix-conditional-risk"
    )
    out_dir = root / "results" / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_ba = {ba: read_ba(input_dir / f"{ba}.xlsx", max_year=test_year) for ba in BAS}

    feature_variants = ["fuel_mix"] if scope == "generated" or args.fuel_mix_only else ["fuel_mix", "fuel_mix_plus_net_import"]
    if args.include_factor_ablation:
        feature_variants += ["fuel_mix_plus_factors"]
        if scope == "consumed":
            feature_variants += ["fuel_mix_plus_factors_and_net_import"]
    tables = {}
    days_by_variant = {}
    for variant in feature_variants:
        monthly_info, daily_scores = monthly_table(rows_by_ba, scope)
        features = {}
        quality = {}
        for ba, rows in rows_by_ba.items():
            years_months = sorted({(row[0], row[1]) for row in rows})
            for year, month in years_months:
                result = category_month_features(rows, year, month, variant)
                if result:
                    features[(ba, year, month)] = result[0]
                    quality[(ba, year, month)] = result[1]
        tables[variant] = (monthly_info, features, quality)
        days_by_variant[variant] = daily_scores

    per_ba_rows = []
    macro_rows = []
    selective_rows = []
    monthly_predictions = []
    calibration_metadata = {}
    for variant in feature_variants:
        info, features, quality = tables[variant]
        daily_scores = days_by_variant[variant]
        # Keep one dictionary of monthly risk targets/features for source fit.
        valid_keys = [key for key in info if key in features]
        target_predictions: dict[str, dict[tuple[int, int], float]] = defaultdict(dict)
        target_radii: dict[str, dict[str, dict[tuple[int, int], float]]] = defaultdict(lambda: defaultdict(dict))
        variant_calibration = {}

        # For each held source BA, fit on 2019-2023 from other BAs and obtain
        # out-of-region 2024 risk predictions. Its 2024 hourly scores calibrate
        # a multiplicative scale, never used as training labels.
        calibration_predictions: dict[str, dict[tuple[str, int, int], float]] = defaultdict(dict)
        for held_ba in BAS:
            train_keys = [
                key for key in valid_keys
                if key[0] != held_ba and key[1] in TRAIN_YEARS
            ]
            if len(train_keys) < 50:
                continue
            model = fit_ridge(
                [features[key] for key in train_keys],
                [math.log(info[key]["risk"]) for key in train_keys],
            )
            for key in valid_keys:
                ba, year, month = key
                if ba == held_ba and year == CALIBRATION_YEAR:
                    calibration_predictions[variant][key] = predict_ridge(model, features[key])

        for target_ba in BAS:
            train_keys = [
                key for key in valid_keys
                if key[0] != target_ba and key[1] in TRAIN_YEARS
            ]
            if len(train_keys) < 50:
                continue
            model = fit_ridge(
                [features[key] for key in train_keys],
                [math.log(info[key]["risk"]) for key in train_keys],
            )
            ratios: list[float] = []
            baseline_calibration: list[float] = []
            for (cal_ba, year, month), predicted_risk in calibration_predictions[variant].items():
                if cal_ba == target_ba:
                    continue
                for _, score in daily_scores.get((cal_ba, year, month), []):
                    ratios.append(score / predicted_risk)
                    baseline_calibration.append(score)
            if not ratios:
                continue
            multiplier = quantile_higher(ratios, 1 - ALPHA)
            baseline_radius = quantile_higher(baseline_calibration, 1 - ALPHA)
            variant_calibration[target_ba] = {
                "multiplier": multiplier,
                "baseline_radius": baseline_radius,
                "calibration_days": len(ratios),
            }

            if args.selective_gate:
                accepted_source_days: list[tuple[float, float]] = []
                source_months = 0
                source_total_months = 0
                source_eligible_days = 0
                for (cal_ba, year, month), predicted_risk in calibration_predictions[variant].items():
                    if cal_ba != target_ba:
                        continue
                    source_total_months += 1
                    source_eligible_days += len(daily_scores.get((cal_ba, year, month), []))
                    radius = predicted_risk * multiplier
                    accepted = radius < baseline_radius
                    if accepted:
                        source_months += 1
                        accepted_source_days.extend(
                            (score, radius)
                            for _, score in daily_scores.get((cal_ba, year, month), [])
                        )
                n_source = len(accepted_source_days)
                selective_rows.append(
                    {
                            "scope": scope,
                            "feature_variant": variant,
                            "evaluation": "source_2024_crossfit",
                            "ba": target_ba,
                            "accepted_months": source_months,
                            "eligible_months": source_total_months,
                            "accepted_days": n_source,
                            "eligible_days": source_eligible_days,
                            "accepted_day_fraction": n_source / source_eligible_days if source_eligible_days else 0.0,
                            "daily_coverage": sum(score <= radius for score, radius in accepted_source_days) / n_source if n_source else None,
                            "mean_interval_width_g_per_kwh": sum(2 * radius for _, radius in accepted_source_days) / n_source if n_source else None,
                            "daily_winkler90": sum(2 * radius + (2 / ALPHA) * max(0.0, score - radius) for score, radius in accepted_source_days) / n_source if n_source else None,
                        }
                )

            for key in valid_keys:
                ba, year, month = key
                if ba != target_ba or year != test_year:
                    continue
                predicted_risk = predict_ridge(model, features[key])
                target_predictions[target_ba][(year, month)] = predicted_risk
                target_radii[target_ba]["risk_model"][(year, month)] = predicted_risk * multiplier
                target_radii[target_ba]["pooled_source"][(year, month)] = baseline_radius
                monthly_predictions.append(
                    {
                        "scope": scope,
                        "feature_variant": variant,
                        "ba": target_ba,
                        "year": year,
                        "month": month,
                        "monthly_risk_label_g_per_kwh": info[key]["risk"],
                        "predicted_risk_g_per_kwh": predicted_risk,
                        "calibrated_risk_radius_g_per_kwh": predicted_risk * multiplier,
                        "pooled_source_radius_g_per_kwh": baseline_radius,
                        "fuel_margin_coverage_ratio": quality[key],
                    }
                )

            for method in ("pooled_source", "risk_model"):
                daily_all: list[tuple[float, float]] = []
                for (year, month), radius in target_radii[target_ba][method].items():
                    for _, score in daily_scores.get((target_ba, year, month), []):
                        daily_all.append((score, radius))
                if not daily_all:
                    continue
                covered = sum(score <= radius for score, radius in daily_all)
                width = sum(2 * radius for _, radius in daily_all) / len(daily_all)
                winkler = sum(2 * radius + (2 / ALPHA) * max(0.0, score - radius) for score, radius in daily_all) / len(daily_all)
                row = {
                    "scope": scope,
                    "feature_variant": variant,
                    "ba": target_ba,
                    "method": method,
                    "complete_test_days": len(daily_all),
                    "daily_simultaneous_coverage": covered / len(daily_all),
                    "mean_daily_interval_width_g_per_kwh": width,
                    "daily_winkler90": winkler,
                }
                per_ba_rows.append(row)

            if args.selective_gate:
                accepted_days: list[tuple[float, float]] = []
                accepted_months = 0
                eligible_months = len(target_radii[target_ba]["risk_model"])
                for (year, month), radius in target_radii[target_ba]["risk_model"].items():
                    baseline = target_radii[target_ba]["pooled_source"][(year, month)]
                    if radius < baseline:
                        accepted_months += 1
                        accepted_days.extend(
                            (score, radius)
                            for _, score in daily_scores.get((target_ba, year, month), [])
                        )
                n_accepted = len(accepted_days)
                eligible_days = sum(
                    len(daily_scores.get((target_ba, year, month), []))
                    for year, month in target_radii[target_ba]["risk_model"]
                )
                selective_rows.append(
                    {
                            "scope": scope,
                            "feature_variant": variant,
                            "evaluation": f"test_{test_year}",
                            "ba": target_ba,
                            "accepted_months": accepted_months,
                            "eligible_months": eligible_months,
                            "accepted_days": n_accepted,
                            "eligible_days": eligible_days,
                            "accepted_day_fraction": n_accepted / eligible_days if eligible_days else 0.0,
                            "daily_coverage": sum(score <= radius for score, radius in accepted_days) / n_accepted if n_accepted else None,
                            "mean_interval_width_g_per_kwh": sum(2 * radius for _, radius in accepted_days) / n_accepted if n_accepted else None,
                            "daily_winkler90": sum(2 * radius + (2 / ALPHA) * max(0.0, score - radius) for score, radius in accepted_days) / n_accepted if n_accepted else None,
                        }
                )

        for method in ("pooled_source", "risk_model"):
            subset = [row for row in per_ba_rows if row["scope"] == scope and row["feature_variant"] == variant and row["method"] == method]
            if subset:
                macro_rows.append(
                    {
                        "scope": scope,
                        "feature_variant": variant,
                        "method": method,
                        "target_ba_count": len(subset),
                        "macro_daily_coverage": sum(r["daily_simultaneous_coverage"] for r in subset) / len(subset),
                        "macro_mean_daily_interval_width_g_per_kwh": sum(r["mean_daily_interval_width_g_per_kwh"] for r in subset) / len(subset),
                        "macro_daily_winkler90": sum(r["daily_winkler90"] for r in subset) / len(subset),
                        "ba_count_at_or_above_90pct": sum(r["daily_simultaneous_coverage"] >= 0.90 for r in subset),
                        "worst_ba_daily_coverage": min(r["daily_simultaneous_coverage"] for r in subset),
                    }
                )
        calibration_metadata[variant] = variant_calibration

    write_csv(out_dir / f"{scope}_by_ba_metrics.csv", per_ba_rows)
    write_csv(out_dir / f"{scope}_macro_metrics.csv", macro_rows)
    write_csv(out_dir / f"{scope}_monthly_risk_predictions.csv", monthly_predictions)
    if args.selective_gate:
        write_csv(out_dir / f"{scope}_selective_by_ba.csv", selective_rows)
        summaries = []
        for evaluation in ("source_2024_crossfit", f"test_{test_year}"):
            subset = [row for row in selective_rows if row["evaluation"] == evaluation]
            if subset:
                accepted = [row for row in subset if row["accepted_days"] > 0]
                accepted_days = sum(row["accepted_days"] for row in accepted)
                eligible_days = sum(row["eligible_days"] for row in subset)
                summaries.append(
                    {
                        "scope": scope,
                        "feature_variant": subset[0]["feature_variant"],
                        "evaluation": evaluation,
                        "ba_count": len(subset),
                        "ba_count_with_accepted_days": len(accepted),
                        "accepted_days": accepted_days,
                        "eligible_days": eligible_days,
                        "pooled_accept_fraction": accepted_days / eligible_days if eligible_days else None,
                        "accepted_only_pooled_daily_coverage": sum(row["accepted_days"] * float(row["daily_coverage"]) for row in accepted) / accepted_days if accepted_days else None,
                        "macro_ba_daily_coverage": sum(float(row["daily_coverage"]) for row in accepted) / len(accepted) if accepted else None,
                        "macro_ba_accept_fraction": sum(float(row["accepted_day_fraction"]) for row in subset) / len(subset),
                        "accepted_only_pooled_interval_width_g_per_kwh": sum(row["accepted_days"] * float(row["mean_interval_width_g_per_kwh"]) for row in accepted) / accepted_days if accepted_days else None,
                        "accepted_only_pooled_winkler90": sum(row["accepted_days"] * float(row["daily_winkler90"]) for row in accepted) / accepted_days if accepted_days else None,
                        "worst_ba_daily_coverage": min(float(row["daily_coverage"]) for row in accepted) if accepted else None,
                    }
                )
        write_csv(out_dir / f"{scope}_selective_summary.csv", summaries)
    metadata_path = out_dir / f"{scope}_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "protocol": "V278_SELECTIVE_2026_CONFIRMATION_PROTOCOL.md" if args.selective_gate and test_year == 2026 else "V277_SELECTIVE_RISK_ABSTENTION_PROTOCOL.md" if args.selective_gate else "V276_LOCAL_FACTOR_LEVEL_RISK_ABLATION_PROTOCOL.md" if args.include_factor_ablation else "V275_COARSE_MIX_CONDITIONAL_RISK_PROTOCOL.md",
                "scope": scope,
                "target_bas": list(BAS),
                "training_years": list(TRAIN_YEARS),
                "calibration_year": CALIBRATION_YEAR,
                "test_year": test_year,
                "ridge_alpha": RIDGE_ALPHA,
                "daily_alpha": ALPHA,
                "feature_variants": feature_variants,
                "calibration_by_target": calibration_metadata,
                "interpretation": "EIA estimated label; target monthly CIF center is oracle; risk features are monthly fuel margins; not China or CEMS validation",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for row in macro_rows:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
