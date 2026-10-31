#!/usr/bin/env python3
"""Retrospective shape-transfer benchmark on EIA's published BA hourly CIF estimates.

This is a shape-only study: each target BA-month is centered using its own
generation-weighted monthly mean from the target labels. It is not a cold-start
monthly-mean forecast and must not be described as one.
"""

from __future__ import annotations

import csv
import argparse
import json
import math
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

from openpyxl import load_workbook


BA_CODES = ("BANC", "BPAT", "CISO", "ERCO", "FPL", "MISO", "NYIS")
WORKBOOK_SHEET = "Published Hourly Data"
CI_COLUMN = "CO2 Emissions Intensity for Generated Electricity"
GEN_COLUMN = "Positive Generation"
DATE_COLUMN = "Local date"
HOUR_COLUMN = "Hour"
YEARS = set(range(2019, 2026))
TRAIN_YEARS = set(range(2019, 2024))
CALIBRATION_YEAR = 2024
TEST_YEAR = 2025
LB_PER_KWH_TO_G_PER_KWH = 453.59237
ALPHA = 0.10

# Compact observation tuple:
# (year, month, local_day_ordinal, weekday, hour_slot_0_to_23, g_per_kWh, positive_MWh)
Observation = tuple[int, int, int, int, int, float, float]


def parse_day(value: object) -> date | None:
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


def load_ba(path: Path, ba: str) -> tuple[list[Observation], dict[str, object]]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    if WORKBOOK_SHEET not in workbook.sheetnames:
        raise ValueError(f"{path}: missing sheet {WORKBOOK_SHEET}")
    sheet = workbook[WORKBOOK_SHEET]
    rows = sheet.iter_rows(values_only=True)
    headers = next(rows)
    indexes = {str(value): index for index, value in enumerate(headers) if value is not None}
    required = ("UTC time", DATE_COLUMN, HOUR_COLUMN, GEN_COLUMN, CI_COLUMN)
    absent = [column for column in required if column not in indexes]
    if absent:
        raise ValueError(f"{path}: missing fields {absent}")

    observations: list[Observation] = []
    year_quality: dict[int, dict[str, int]] = defaultdict(lambda: {"rows": 0, "valid": 0})
    for row in rows:
        utc = row[indexes["UTC time"]]
        if not isinstance(utc, datetime):
            continue
        local_day = parse_day(row[indexes[DATE_COLUMN]])
        if local_day is None or local_day.year not in YEARS:
            continue
        # Train/test periods and calendar-feature alignment use the BA's local
        # date, not UTC. UTC-year splits would leak/misclassify boundary hours.
        local_year = local_day.year
        stats = year_quality[local_year]
        stats["rows"] += 1
        hour = row[indexes[HOUR_COLUMN]]
        generation = row[indexes[GEN_COLUMN]]
        intensity = row[indexes[CI_COLUMN]]
        if (
            local_day is None
            or not isinstance(hour, (int, float))
            or not isinstance(generation, (int, float))
            or not isinstance(intensity, (int, float))
            or not math.isfinite(float(generation))
            or not math.isfinite(float(intensity))
            or generation <= 0
        ):
            continue
        slot = (int(hour) - 1) % 24
        observations.append(
            (
                local_year,
                utc.month,
                local_day.toordinal(),
                local_day.weekday(),
                slot,
                float(intensity) * LB_PER_KWH_TO_G_PER_KWH,
                float(generation),
            )
        )
        stats["valid"] += 1
    workbook.close()
    metadata = {
        "ba": ba,
        "file": path.name,
        "bytes": path.stat().st_size,
        "years": {str(year): values for year, values in sorted(year_quality.items())},
    }
    return observations, metadata


def monthly_centers(observations: Iterable[Observation]) -> dict[tuple[int, int], float]:
    totals: dict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for year, month, _, _, _, intensity, generation in observations:
        key = (year, month)
        totals[key][0] += intensity * generation
        totals[key][1] += generation
    return {key: values[0] / values[1] for key, values in totals.items() if values[1] > 0}


def build_profile(
    observations_by_ba: dict[str, list[Observation]],
    target_ba: str,
    profile_kind: str,
) -> dict[tuple[int, ...], float]:
    """Equal-BA average of donor within-month intensity residual profiles."""
    per_ba: dict[str, dict[tuple[int, ...], list[float]]] = {}
    for ba, observations in observations_by_ba.items():
        if ba == target_ba:
            continue
        centers = monthly_centers(
            row for row in observations if row[0] in TRAIN_YEARS
        )
        sums: dict[tuple[int, ...], list[float]] = defaultdict(lambda: [0.0, 0.0])
        for year, month, _, weekday, hour, intensity, _ in observations:
            if year not in TRAIN_YEARS:
                continue
            center = centers.get((year, month))
            if center is None:
                continue
            if profile_kind == "hour_of_day":
                key = (month, hour)
            elif profile_kind == "hour_of_week":
                key = (month, weekday, hour)
            else:
                raise ValueError(f"unknown profile kind: {profile_kind}")
            sums[key][0] += intensity - center
            sums[key][1] += 1
        per_ba[ba] = {key: value[0] / value[1] for key, value in sums.items() if value[1]}

    keys = set().union(*(values.keys() for values in per_ba.values())) if per_ba else set()
    profile: dict[tuple[int, ...], float] = {}
    for key in keys:
        values = [by_key[key] for by_key in per_ba.values() if key in by_key]
        if values:
            profile[key] = sum(values) / len(values)
    return profile


def predict_residual(
    profile: dict[tuple[int, ...], float],
    profile_kind: str,
    month: int,
    weekday: int,
    hour: int,
) -> float:
    if profile_kind == "monthly_anchor":
        return 0.0
    if profile_kind == "hour_of_day":
        return profile.get((month, hour), 0.0)
    exact = profile.get((month, weekday, hour))
    if exact is not None:
        return exact
    return profile.get((month, hour), 0.0)


def quantile_higher(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot calibrate on an empty score set")
    ordered = sorted(values)
    rank = math.ceil((len(ordered) + 1) * probability)
    return ordered[min(len(ordered) - 1, max(0, rank - 1))]


def daily_scores(
    observations: list[Observation],
    ba: str,
    year: int,
    center_by_month: dict[tuple[int, int], float],
    profile: dict[tuple[int, ...], float],
    model: str,
) -> list[tuple[float, int]]:
    by_day: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for row_year, month, day_ordinal, weekday, hour, intensity, _ in observations:
        if row_year != year:
            continue
        center = center_by_month.get((year, month))
        if center is None:
            continue
        prediction = center + predict_residual(profile, model, month, weekday, hour)
        by_day[day_ordinal].append((hour, abs(intensity - prediction)))
    scores: list[tuple[float, int]] = []
    for ordinal, hourly_errors in by_day.items():
        hours = [hour for hour, _ in hourly_errors]
        # Exclude DST transition days and incomplete/missing-label days.
        if len(hourly_errors) == 24 and len(set(hours)) == 24:
            scores.append((max(error for _, error in hourly_errors), ordinal))
    return scores


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    global CI_COLUMN, GEN_COLUMN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("generated", "consumed"), default="generated")
    args = parser.parse_args()
    if args.scope == "consumed":
        CI_COLUMN = "CO2 Emissions Intensity for Consumed Electricity"
        GEN_COLUMN = "Consumed Electricity"

    root = Path(__file__).resolve().parents[1]
    input_dir = root / "results" / "v272-eia-monitor"
    output_name = (
        "v273-eia-generated-cif-shape-transfer"
        if args.scope == "generated"
        else "v274-eia-consumed-cif-shape-transfer"
    )
    output_dir = root / "results" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    observations_by_ba: dict[str, list[Observation]] = {}
    source_metadata: list[dict[str, object]] = []
    for ba in BA_CODES:
        path = input_dir / f"{ba}.xlsx"
        if not path.exists():
            raise FileNotFoundError(path)
        observations, metadata = load_ba(path, ba)
        observations_by_ba[ba] = observations
        source_metadata.append(metadata)

    models = ("monthly_anchor", "hour_of_day", "hour_of_week")
    calibration_scores: dict[str, dict[str, list[float]]] = {
        model: {ba: [] for ba in BA_CODES} for model in models
    }
    calibration_rows: list[dict[str, object]] = []
    for target_ba in BA_CODES:
        target_observations = observations_by_ba[target_ba]
        centers = monthly_centers(target_observations)
        for model in models:
            profile = (
                {} if model == "monthly_anchor"
                else build_profile(observations_by_ba, target_ba, model)
            )
            scores = daily_scores(
                target_observations, target_ba, CALIBRATION_YEAR, centers, profile, model
            )
            for score, ordinal in scores:
                calibration_scores[model][target_ba].append(score)
                calibration_rows.append(
                    {
                        "model": model,
                        "pseudo_target_ba": target_ba,
                        "local_date": date.fromordinal(ordinal).isoformat(),
                        "max_abs_hourly_error_g_per_kwh": score,
                    }
                )

    radii_by_target: dict[str, dict[str, float]] = {}
    for target_ba in BA_CODES:
        radii_by_target[target_ba] = {}
        for model in models:
            source_scores = [
                score
                for calibration_ba in BA_CODES
                if calibration_ba != target_ba
                for score in calibration_scores[model][calibration_ba]
            ]
            radii_by_target[target_ba][model] = quantile_higher(source_scores, 1 - ALPHA)
    metrics_by_ba: list[dict[str, object]] = []
    metrics_all: list[dict[str, object]] = []
    for target_ba in BA_CODES:
        target_observations = observations_by_ba[target_ba]
        centers = monthly_centers(target_observations)
        for model in models:
            profile = (
                {} if model == "monthly_anchor"
                else build_profile(observations_by_ba, target_ba, model)
            )
            q = radii_by_target[target_ba][model]
            n = 0
            abs_errors: list[float] = []
            squared_errors: list[float] = []
            covered_hours = 0
            daily = daily_scores(target_observations, target_ba, TEST_YEAR, centers, profile, model)
            daily_covered = sum(score <= q for score, _ in daily)
            winkler_scores = [
                2 * q + (2 / ALPHA) * max(0.0, score - q)
                for score, _ in daily
            ]
            for year, month, _, weekday, hour, intensity, _ in target_observations:
                if year != TEST_YEAR:
                    continue
                center = centers.get((year, month))
                if center is None:
                    continue
                prediction = center + predict_residual(profile, model, month, weekday, hour)
                error = intensity - prediction
                n += 1
                abs_errors.append(abs(error))
                squared_errors.append(error * error)
                covered_hours += abs(error) <= q
            row = {
                "ba": target_ba,
                "model": model,
                "test_year": TEST_YEAR,
                "oracle_month_anchor": True,
                "hourly_n": n,
                "hourly_mae_g_per_kwh": sum(abs_errors) / n if n else None,
                "hourly_rmse_g_per_kwh": math.sqrt(sum(squared_errors) / n) if n else None,
                "calibration_half_width_g_per_kwh": q,
                "hourly_coverage": covered_hours / n if n else None,
                "hourly_interval_width_g_per_kwh": 2 * q,
                "complete_test_days": len(daily),
                "daily_simultaneous_coverage": daily_covered / len(daily) if daily else None,
                "daily_winkler90": sum(winkler_scores) / len(winkler_scores) if daily else None,
            }
            metrics_by_ba.append(row)

    for model in models:
        subset = [row for row in metrics_by_ba if row["model"] == model]
        metrics_all.append(
            {
                "model": model,
                "test_year": TEST_YEAR,
                "target_ba_count": len(subset),
                "macro_hourly_mae_g_per_kwh": sum(float(r["hourly_mae_g_per_kwh"]) for r in subset) / len(subset),
                "macro_hourly_rmse_g_per_kwh": sum(float(r["hourly_rmse_g_per_kwh"]) for r in subset) / len(subset),
                "macro_calibration_half_width_g_per_kwh": sum(float(r["calibration_half_width_g_per_kwh"]) for r in subset) / len(subset),
                "macro_hourly_interval_width_g_per_kwh": sum(float(r["hourly_interval_width_g_per_kwh"]) for r in subset) / len(subset),
                "macro_hourly_coverage": sum(float(r["hourly_coverage"]) for r in subset) / len(subset),
                "macro_daily_simultaneous_coverage": sum(float(r["daily_simultaneous_coverage"]) for r in subset) / len(subset),
                "macro_daily_winkler90": sum(float(r["daily_winkler90"]) for r in subset) / len(subset),
                "macro_complete_test_days": sum(int(r["complete_test_days"]) for r in subset),
            }
        )

    coverage_rows: list[dict[str, object]] = []
    for ba, observations in observations_by_ba.items():
        for year in sorted(YEARS):
            rows = [row for row in observations if row[0] == year]
            coverage_rows.append(
                {
                    "ba": ba,
                    "year": year,
                    "valid_hour_rows": len(rows),
                    "calendar_hour_expectation": 8784 if year % 4 == 0 else 8760,
                    "unique_local_days": len({row[2] for row in rows}),
                }
            )

    write_csv(output_dir / "target_ba_metrics.csv", metrics_by_ba)
    write_csv(output_dir / "macro_metrics.csv", metrics_all)
    write_csv(output_dir / "calibration_day_scores.csv", calibration_rows)
    write_csv(output_dir / "data_coverage.csv", coverage_rows)
    metadata = {
        "protocol": (
            "V273_EIA_GENERATED_CIF_SHAPE_TRANSFER_PROTOCOL.md"
            if args.scope == "generated"
            else "V274_EIA_CONSUMED_CIF_SHAPE_TRANSFER_PROTOCOL.md"
        ),
        "script": Path(__file__).name,
        "source_files": source_metadata,
        "target_bas": list(BA_CODES),
        "training_years": sorted(TRAIN_YEARS),
        "calibration_year": CALIBRATION_YEAR,
        "test_year": TEST_YEAR,
        "label": CI_COLUMN,
        "label_unit_conversion": "lbs/kWh * 453.59237 = g/kWh",
        "scope": args.scope,
        "monthly_anchor": f"target-year hourly EIA {args.scope} CIF, weighted by its matching electricity denominator; oracle shape-only evaluation",
        "daily_conformal_alpha": ALPHA,
        "calibration_half_widths_by_target_g_per_kwh": radii_by_target,
        "daily_calibration_scores_by_ba": {
            model: {ba: len(scores) for ba, scores in by_ba.items()}
            for model, by_ba in calibration_scores.items()
        },
        "interpretation": "retrospective EIA estimated-label shape transfer; not direct CEMS and not China validation",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for row in metrics_all:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
