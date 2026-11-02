"""Prediction-free schema and coverage preflight for Poland fuel-response confirmation."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import run_energy_charts_nine_country_loco as nine
from transcif.research import multicountry as model

DATA = ROOT / "data/energy-charts-v2-panel-r1"
MAPPING = {
    "coal": ("fossil_hard_coal", "fossil_brown_coal_lignite"),
    "gas": ("fossil_gas",), "oil": ("fossil_oil",),
    # Poland's response does not expose a separate waste series; keep the
    # proxy boundary explicit instead of imputing waste into another bucket.
    "other": ("fossil_coal_derived_gas", "biomass", "others"),
    "nuclear": (), "hydro": ("hydro_run_of_river", "hydro_water_reservoir"),
    "wind": ("wind_onshore",), "solar": ("solar",),
    "other_renewable": ("other_renewables",),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest_path = DATA / "acquisition_manifest.json"
    raw_path = DATA / "pl_2022_2024.json"
    manifest = json.loads(manifest_path.read_text())
    expected_sha = next(r["sha256"] for r in manifest["countries"] if r["country"] == "pl")
    if sha(raw_path) != expected_sha:
        raise ValueError("PL raw data does not match the acquisition manifest")
    payload = json.loads(raw_path.read_text())
    rows = [r for r in payload["data"] if r["timestamp"][:4] in {"2023", "2024"}]
    stamps = pd.DatetimeIndex([pd.Timestamp(r["timestamp"]).tz_convert("UTC") for r in rows])
    if stamps.has_duplicates or not stamps.is_monotonic_increasing:
        raise ValueError("PL timestamps must be unique and ordered")
    if not np.all(np.diff(stamps) == pd.Timedelta(hours=1)):
        raise ValueError("PL source panel is not continuous hourly data")

    values = pd.DataFrame([r["values"] for r in rows])
    present = set(values.columns)
    required = {key for keys in MAPPING.values() for key in keys}
    if missing := required - present:
        raise ValueError(f"mapped PL production categories absent: {sorted(missing)}")
    fuel = pd.DataFrame(index=values.index)
    nulls = {}
    for name in model.FINE:
        keys = MAPPING[name]
        if not keys:
            fuel[name] = 0.
            nulls[name] = 0
            continue
        block = values[list(keys)].apply(pd.to_numeric, errors="coerce")
        nulls[name] = int(block.isna().sum().sum())
        fuel[name] = block.sum(axis=1, min_count=len(keys))
    complete = fuel[list(model.FINE)].notna().all(axis=1)
    finite = np.isfinite(fuel[list(model.FINE)].to_numpy(float)).all(axis=1)
    nonnegative = (fuel[list(model.FINE)].to_numpy(float) >= 0).all(axis=1)
    stamps_local = stamps.tz_convert(payload["timezone"]).tz_localize(None)
    valid = complete.to_numpy() & finite & nonnegative
    frame = fuel.loc[valid].copy()
    frame.index = stamps_local[valid]
    frame["time"] = stamps[valid]
    frame["clock"] = frame.index
    frame["country"], frame["region"] = "PL", "PL"
    periods = frame.clock.dt.to_period("M")
    frame["expected_hours"] = periods.map({
        p: nine.base.expected_local_hours(p, payload["timezone"]) for p in periods.unique()
    }).to_numpy()
    frame = frame.loc[frame[list(model.FINE)].sum(axis=1).gt(0)].reset_index(drop=True)

    # Label completeness only; no emissions factors, model predictions or error scores.
    target_rows = frame.loc[frame.clock.dt.year.eq(2024)]
    target_period = target_rows.clock.dt.to_period("M")
    target_counts = target_rows.groupby(target_period).size()
    target_expected = target_rows.groupby(target_period).expected_hours.first()
    label_coverage = target_counts.div(target_expected)
    all_2024 = pd.period_range("2024-01", "2024-12", freq="M")
    label_coverage = label_coverage.reindex(all_2024, fill_value=0.)

    months = model.public_months(frame)
    features = model.public_features(months, frame.clock)
    features["year"] = frame.clock.dt.year.to_numpy()
    features["raw_row"] = np.arange(len(frame))
    usable = features.dropna(subset=list(model.HISTORY))
    usable_2024 = usable.loc[usable.year.eq(2024)]
    input_hours_by_month = usable_2024.groupby(usable_2024.clock.dt.to_period("M")).size()
    label_by_month = label_coverage.reindex(input_hours_by_month.index)
    target_months = sorted(str(p) for p in input_hours_by_month.index
                           if label_by_month.loc[p] >= .99)

    result = {
        "status": "PREDICTION_FREE_SCHEMA_AND_COVERAGE_PREFLIGHT",
        "country": "PL", "timezone": payload["timezone"],
        "unit": payload.get("unit"), "reported_resolution": payload.get("resolution"),
        "raw_path": str(raw_path.relative_to(ROOT)), "raw_sha256": expected_sha,
        "manifest_sha256": sha(manifest_path), "rows_2023_2024": len(rows),
        "mapped_production_types": MAPPING,
        "mapped_category_null_cells": nulls,
        "unused_source_categories": sorted(present - required),
        "unresolved_source_components": ["waste is not separately exposed; omitted from proxy unless folded into API `others`"],
        "retained_complete_nonnegative_rows": len(frame),
        "2024_label_coverage_by_month": {str(p): float(v) for p, v in label_coverage.items()},
        "2024_cn5_feature_hours_by_month": {str(p): int(v) for p, v in input_hours_by_month.items()},
        "eligible_target_months": target_months,
        "nuclear_zero_basis": "Eurostat reports Poland did not produce nuclear electricity in 2024",
        "scored_target_metrics_read": False,
    }
    if len(target_months) >= 4 and target_months == ["2024-09", "2024-10", "2024-11", "2024-12"]:
        result["status"] = "PREFLIGHT_ELIGIBLE_RESTRICTED_WINDOW"
    else:
        result["status"] = "PREFLIGHT_FAILED_NO_FIXED_CONTIGUOUS_WINDOW"
    out = ROOT / "results/cn5-fuel-resolved-pl-preflight-r2"
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    (out / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
