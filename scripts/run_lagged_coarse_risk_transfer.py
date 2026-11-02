#!/usr/bin/env python3
"""Test if China-compatible lagged monthly mix predicts hourly tail risk."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v275_coarse_mix_conditional_risk as base  # noqa: E402

INPUT = ROOT / "results/v272-eia-monitor"
OUTPUT = ROOT / "results/lagged-coarse-risk-transfer-r1"
TRAIN_YEARS = range(2019, 2024)
CAL_YEAR = 2024
TEST_YEAR = 2025
ALPHA = 0.10


def shift_month(year: int, month: int, offset: int) -> tuple[int, int]:
    serial = year * 12 + month - 1 + offset
    return serial // 12, serial % 12 + 1


def cn5_features(rows: list[tuple], year: int, month: int, with_import: bool = False) -> list[float] | None:
    variant = "fuel_mix_plus_net_import" if with_import else "fuel_mix"
    result = base.category_month_features(rows, year, month, variant)
    if result is None:
        return None
    features, _ = result
    coal, gas, oil, nuclear, hydro, wind, solar, other = features[:8]
    thermal = coal + gas + oil + other
    cn5 = [thermal, hydro, nuclear, wind, solar]
    out = cn5 + [features[8]]  # five shares plus log total generation
    if with_import:
        out.append(features[9])
    return out


def higher(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("empty calibration scores")
    return float(np.quantile(np.asarray(values, dtype=float), probability, method="higher"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(days: list[float], radius: float) -> dict[str, float | int]:
    covered = sum(value <= radius for value in days)
    width = 2 * radius
    winkler = np.mean([width + (2 / ALPHA) * max(0.0, value - radius) for value in days])
    return {
        "complete_days": len(days),
        "daily_simultaneous_coverage": covered / len(days),
        "mean_interval_width_g_per_kwh": width,
        "winkler90": float(winkler),
    }


def feature_offsets(variant: str) -> tuple[int, ...]:
    if variant == "current_cn5_oracle":
        return (0,)
    if variant.startswith("lag2to7_"):
        return (-2, -3, -4, -5, -6, -7)
    return (-2,)


def run_scope(
    rows_by_ba: dict[str, list[tuple]],
    scope: str,
    variant_names: list[str] | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    info, daily = base.monthly_table(rows_by_ba, scope)
    custom_paired = variant_names is not None
    if variant_names is None:
        variants = ["lag2_cn5", "current_cn5_oracle"]
        if scope == "consumed":
            variants.insert(1, "lag2_cn5_plus_net_import")
    else:
        variants = variant_names

    features_by_variant: dict[str, dict[tuple[str, int, int], list[float]]] = {}
    for variant in variants:
        include_import = variant == "lag2_cn5_plus_net_import"
        include_import = include_import or variant == "lag2to7_cn5_plus_net_import"
        offsets = feature_offsets(variant)
        feature_map = {}
        for ba, rows in rows_by_ba.items():
            for year, month in sorted({(r[0], r[1]) for r in rows}):
                vector = []
                for lag in offsets:
                    source_year, source_month = shift_month(year, month, lag)
                    part = cn5_features(rows, source_year, source_month, include_import)
                    if part is None:
                        vector = []
                        break
                    vector.extend(part)
                key = (ba, year, month)
                if vector:
                    feature_map[key] = vector
        features_by_variant[variant] = feature_map

    # Net interchange is occasionally missing. Compare the consumption flow
    # increment only on common support so that its cohort is paired with M.
    if custom_paired:
        common = set.intersection(*(set(features_by_variant[name]) for name in variants))
        for variant in variants:
            features_by_variant[variant] = {k: v for k, v in features_by_variant[variant].items() if k in common}
    elif scope == "consumed":
        matched_pairs = (
            ("lag2_cn5", "lag2_cn5_plus_net_import"),
            ("lag2to7_cn5", "lag2to7_cn5_plus_net_import"),
        )
        for plain, flow in matched_pairs:
            if plain not in features_by_variant or flow not in features_by_variant:
                continue
            common = set(features_by_variant[plain]) & set(features_by_variant[flow])
            for variant in (plain, flow):
                features_by_variant[variant] = {k: v for k, v in features_by_variant[variant].items() if k in common}

    by_ba: list[dict] = []
    monthly: list[dict] = []
    macro: list[dict] = []
    for variant, feature_map in features_by_variant.items():
        valid_keys = sorted(set(info).intersection(feature_map))
        calibration_pred: dict[tuple[str, int, int], float] = {}
        for held_ba in base.BAS:
            train = [k for k in valid_keys if k[0] != held_ba and k[1] in TRAIN_YEARS]
            if len(train) < 50:
                continue
            model = base.fit_ridge([feature_map[k] for k in train], [math.log(info[k]["risk"]) for k in train])
            for key in valid_keys:
                if key[0] == held_ba and key[1] == CAL_YEAR:
                    calibration_pred[key] = base.predict_ridge(model, feature_map[key])

        for target_ba in base.BAS:
            train = [k for k in valid_keys if k[0] != target_ba and k[1] in TRAIN_YEARS]
            if len(train) < 50:
                continue
            model = base.fit_ridge([feature_map[k] for k in train], [math.log(info[k]["risk"]) for k in train])
            ratios: list[float] = []
            pooled_scores: list[float] = []
            for key, prediction in calibration_pred.items():
                if key[0] == target_ba:
                    continue
                scores = [v for _, v in daily.get(key, [])]
                ratios.extend(score / prediction for score in scores)
                pooled_scores.extend(scores)
            if not ratios:
                continue
            multiplier = higher(ratios, 1 - ALPHA)
            radius = higher(pooled_scores, 1 - ALPHA)

            target_keys = [k for k in valid_keys if k[0] == target_ba and k[1] == TEST_YEAR]
            paired_scores: dict[str, list[tuple[float, float]]] = defaultdict(list)
            for key in target_keys:
                predicted = base.predict_ridge(model, feature_map[key])
                calibrated = predicted * multiplier
                scores = [v for _, v in daily.get(key, [])]
                periods = [shift_month(key[1], key[2], lag) for lag in feature_offsets(variant)]
                period_field = (
                    {"feature_period_end": f"{periods[0][0]:04d}-{periods[0][1]:02d}"}
                    if len(periods) == 1
                    else {"feature_periods": ";".join(f"{y:04d}-{m:02d}" for y, m in periods)}
                )
                monthly.append({
                    "scope": scope,
                    "feature_variant": variant,
                    "ba": target_ba,
                    "target_year": key[1],
                    "target_month": key[2],
                    **period_field,
                    "daily_risk_label_g_per_kwh": info[key]["risk"],
                    "predicted_risk_g_per_kwh": predicted,
                    "calibrated_radius_g_per_kwh": calibrated,
                    "pooled_source_radius_g_per_kwh": radius,
                    "source_calibration_multiplier": multiplier,
                    "complete_days": len(scores),
                })
                # Store paired daily scores with their month-specific radius.
                for _, score in daily.get(key, []):
                    paired_scores["conditional"].append((score, calibrated))
                    paired_scores["pooled_source"].append((score, radius))

            method_rows = []
            for method in ("conditional", "pooled_source"):
                pairs = paired_scores[method]
                if not pairs:
                    continue
                scores = [p[0] for p in pairs]
                radii = [p[1] for p in pairs]
                covered = sum(score <= r for score, r in pairs)
                width = float(np.mean([2 * r for r in radii]))
                winkler = float(np.mean([2 * r + (2 / ALPHA) * max(0.0, score - r) for score, r in pairs]))
                method_rows.append({
                    "scope": scope,
                    "feature_variant": variant,
                    "ba": target_ba,
                    "method": method,
                    "complete_days": len(pairs),
                    "daily_simultaneous_coverage": covered / len(pairs),
                    "mean_interval_width_g_per_kwh": width,
                    "winkler90": winkler,
                    "monthly_risk_mae_g_per_kwh": float(np.mean([
                        abs(float(row["predicted_risk_g_per_kwh"] if method == "conditional" else row["pooled_source_radius_g_per_kwh"]) - float(row["daily_risk_label_g_per_kwh"]))
                        for row in monthly if row["scope"] == scope and row["feature_variant"] == variant and row["ba"] == target_ba
                    ])),
                })
            by_ba.extend(method_rows)

        for method in ("conditional", "pooled_source"):
            subset = [r for r in by_ba if r["scope"] == scope and r["feature_variant"] == variant and r["method"] == method]
            if subset:
                macro.append({
                    "scope": scope,
                    "feature_variant": variant,
                    "method": method,
                    "ba_count": len(subset),
                    "macro_daily_coverage": float(np.mean([r["daily_simultaneous_coverage"] for r in subset])),
                    "macro_interval_width_g_per_kwh": float(np.mean([r["mean_interval_width_g_per_kwh"] for r in subset])),
                    "macro_winkler90": float(np.mean([r["winkler90"] for r in subset])),
                    "ba_count_at_or_above_90pct": sum(r["daily_simultaneous_coverage"] >= 0.90 for r in subset),
                    "worst_ba_daily_coverage": min(r["daily_simultaneous_coverage"] for r in subset),
                    "macro_monthly_risk_mae_g_per_kwh": float(np.mean([r["monthly_risk_mae_g_per_kwh"] for r in subset])),
                })
    return by_ba, macro, monthly


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows_by_ba = {ba: base.read_ba(INPUT / f"{ba}.xlsx", max_year=TEST_YEAR) for ba in base.BAS}
    generated_ba, generated_macro, generated_monthly = run_scope(rows_by_ba, "generated")
    consumed_ba, consumed_macro, consumed_monthly = run_scope(rows_by_ba, "consumed")
    base.write_csv(OUTPUT / "generated_by_ba_metrics.csv", generated_ba)
    base.write_csv(OUTPUT / "generated_macro_metrics.csv", generated_macro)
    base.write_csv(OUTPUT / "generated_monthly_risk_predictions.csv", generated_monthly)
    base.write_csv(OUTPUT / "consumed_by_ba_metrics.csv", consumed_ba)
    base.write_csv(OUTPUT / "consumed_macro_metrics.csv", consumed_macro)
    base.write_csv(OUTPUT / "consumed_monthly_risk_predictions.csv", consumed_monthly)
    metadata = {
        "protocol": "docs/LAGGED_COARSE_RISK_TRANSFER_PROTOCOL_20261101.md",
        "input_directory": str(INPUT.relative_to(ROOT)),
        "target_ba": list(base.BAS),
        "train_years": [min(TRAIN_YEARS), max(TRAIN_YEARS)],
        "calibration_year": CAL_YEAR,
        "test_year": TEST_YEAR,
        "lag_months": 2,
        "target_center": "oracle target-month generation- or consumption-weighted mean; evaluates residual risk only",
        "fuel_mapping": "coal+gas+oil+other=thermal; hydro+pumped storage=hydro; nuclear; wind; solar",
        "flow_feature": "consumption track only; lagged net imports / consumed electricity; extra information budget, not asserted public for China",
        "contemporaneous_branch": "same-month five-group composition; oracle timing upper bound only, not deployable",
        "source_vintage": "final EIA BA workbook histories; lag-simulated, not historical as-issued vintages",
        "input_sha256": {f"{ba}.xlsx": sha256(INPUT / f"{ba}.xlsx") for ba in base.BAS},
    }
    (OUTPUT / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "generated_macro": generated_macro, "consumed_macro": consumed_macro}, indent=2))


if __name__ == "__main__":
    main()
