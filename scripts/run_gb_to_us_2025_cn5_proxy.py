#!/usr/bin/env python3
"""GB-to-US 2025 transfer under a lagged CN5 budget and common EF proxy."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from transcif.data.eia930 import normalize_balance  # noqa: E402
from transcif.research import cross_country as cc  # noqa: E402
from transcif.research.eia_proxy import _proxy_rows  # noqa: E402
from audit_eia923_ba_mix_alignment import parse_eia923  # noqa: E402

PROTOCOL = ROOT / "docs/GB_TO_US_2025_CN5_PROXY_SUPPORTED_PROTOCOL_20261102.md"
TARGET_BAS = ("BPAT", "ERCO", "FPL", "MISO", "NYIS")
EIA923_ARCHIVES = {
    year: ROOT / f"docs/evidence/eia923-ba-mapping-r1/f923_{year}.zip"
    for year in (2023, 2024, 2025)
}
EIA930_FILES = (
    ROOT / "data/native-eia-2025/EIA930_BALANCE_2025_Jan_Jun.csv",
    ROOT / "data/native-eia-2025/EIA930_BALANCE_2025_Jul_Dec.csv",
)
OUTPUT = ROOT / "results/gb-to-us-2025-cn5-proxy-r4"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_models() -> tuple[dict, dict, dict, dict]:
    raw, audit = cc.load_inputs(ROOT)
    gb = raw["GB"]
    features = cc.monthly_inputs(gb)
    features["raw_row"] = features.index
    usable = features.dropna(subset=list(cc.COARSE)).copy()
    train = usable.loc[usable.year.eq(2022)].copy()
    source_fuel = gb.loc[train.raw_row, list(cc.FUELS)].to_numpy(float)
    if len(train) != len(source_fuel) or train.empty:
        raise ValueError("GB 2022 source rows and fixed-EF fuel labels do not align")
    models = cc.fit_models(train, source_fuel)

    calibration_rows = usable.loc[usable.year.eq(2023)].copy()
    if calibration_rows.empty:
        raise ValueError("GB 2023 source calibration rows are missing")
    calibration_prediction = cc.predict_models(models, calibration_rows)
    calibration_truth = cc.labels(gb.loc[calibration_rows.raw_row, list(cc.FUELS)])
    calibration = {}
    for method in cc.METHODS:
        residual = calibration_truth - calibration_prediction[method]
        bias = float(np.mean(residual))
        sigma = max(float(np.sqrt(np.mean((residual - bias) ** 2))), 1.0)
        calibration[method] = {
            "bias_gco2_per_kwh": bias,
            "sigma_gco2_per_kwh": sigma,
            "calibration_rows": int(len(residual)),
            "calibration_period": "GB 2023",
        }
    return models, calibration, audit, {
        "training_rows": int(len(train)),
        "training_period": "GB 2022",
        "training_region": "GB",
    }


def target_month_features() -> tuple[pd.DataFrame, dict]:
    parts = [parse_eia923(path, TARGET_BAS, year)
             for year, path in EIA923_ARCHIVES.items()]
    monthly = pd.concat(parts, ignore_index=True)
    records = []
    for row in monthly.to_dict("records"):
        observed = pd.Period(row["month"], freq="M")
        target_period = observed + 2
        total = float(row["eia923_total_mwh"])
        if not np.isfinite(total) or total <= 0:
            continue
        fuels = {
            name: float(row[f"eia923_{name}_mwh"])
            for name in cc.FUELS
        }
        thermal = sum(fuels[name] for name in ("coal", "gas", "oil", "other"))
        group_values = {
            "thermal": thermal,
            **{name: fuels[name] for name in ("hydro", "nuclear", "wind", "solar")},
        }
        denominator = float(sum(group_values.values()))
        if not np.isfinite(denominator) or denominator <= 0:
            continue
        records.append({
            "ba": row["ba"],
            "target_period": str(target_period),
            "observed_month": str(observed),
            **{name: value / denominator for name, value in group_values.items()},
            "eia923_total_mwh": denominator,
        })
    mix = pd.DataFrame(records)
    expected_observed = {
        str(period - 2)
        for period in pd.period_range("2025-01", "2025-12", freq="M")
    }
    target_mix = mix.loc[mix.target_period.str.startswith("2025-")].copy()
    support = target_mix.groupby("ba").observed_month.agg(
        rows="size", observed_months=lambda values: sorted(values.tolist())
    )
    if set(target_mix.ba.unique()) != set(TARGET_BAS):
        raise ValueError("EIA-923 target mix does not contain the frozen five-BA roster")
    for ba in TARGET_BAS:
        observed = set(target_mix.loc[target_mix.ba.eq(ba), "observed_month"])
        if observed != expected_observed:
            raise ValueError(f"EIA-923 t-2 month support is incomplete for {ba}: {sorted(observed)}")
    return mix, {
        "archive_sha256": {str(path.resolve()): sha(path) for path in EIA923_ARCHIVES.values()},
        "target_ba_month_rows": int(len(target_mix)),
        "target_observed_months": sorted(expected_observed),
        "target_features": "EIA-923 final; exact t-2 five-group shares",
        "target_periods": ["2025-01", "2025-12"],
        "input_months_exactly_t_minus_2": True,
    }


def target_inputs(monthly: pd.DataFrame) -> pd.DataFrame:
    utc = pd.date_range("2025-01-01 00:00", "2026-01-01 00:00", freq="h",
                        inclusive="left", tz="UTC")
    rows = []
    for ba in TARGET_BAS:
        target = pd.DataFrame({"time": utc})
        target["ba"] = ba
        local = target.time.dt.tz_convert(cc.ZONES[ba]).dt.tz_localize(None)
        target["clock"] = local
        target["period"] = local.dt.to_period("M").astype(str)
        mix = monthly.loc[monthly.ba.eq(ba)].drop(columns="ba").rename(
            columns={"target_period": "period"}
        )
        target = target.merge(mix, on="period", how="left", validate="many_to_one")
        rows.append(target)
    frame = pd.concat(rows, ignore_index=True)
    if frame[list(cc.COARSE)].isna().any().any():
        missing = frame.loc[frame[list(cc.COARSE)].isna().any(axis=1), ["ba", "period"]].drop_duplicates()
        raise ValueError(f"missing exact EIA-923 t-2 features: {missing.to_dict('records')}")
    frame["country"] = "US"
    frame["region"] = frame.ba
    frame["year"] = frame.clock.dt.year
    frame["observed_month"] = frame.observed_month.astype(str)
    for name, values, period in [("hour", frame.clock.dt.hour, 24),
                                 ("dow", frame.clock.dt.dayofweek, 7),
                                 ("month", frame.clock.dt.month, 12)]:
        frame[f"{name}_sin"] = np.sin(2 * np.pi * values / period)
        frame[f"{name}_cos"] = np.cos(2 * np.pi * values / period)
    for fuel in cc.COARSE:
        for suffix in ("sin", "cos"):
            frame[f"{fuel}_hour_{suffix}"] = frame[fuel] * frame[f"hour_{suffix}"]
    return frame


def load_target_proxy() -> tuple[pd.DataFrame, dict, dict]:
    normalized, schemas, source_hashes = [], {}, {}
    for snapshot, path in enumerate(EIA930_FILES):
        raw = pd.read_csv(path)
        raw = raw.loc[raw["Balancing Authority"].isin(TARGET_BAS)].copy()
        frame = normalize_balance(raw)
        frame["source_snapshot"] = snapshot
        schemas[str(snapshot)] = {
            "fuel_columns": [name for name in frame if name.startswith("generation_")]
        }
        normalized.append(frame)
        source_hashes[str(path.resolve())] = sha(path)
    rows = pd.concat(normalized, ignore_index=True).sort_values(["region", "valid_at"])
    if rows.duplicated(["region", "valid_at"]).any():
        raise ValueError("duplicate BA/hour response rows across EIA snapshots")
    response, audits = {}, {}
    for ba in TARGET_BAS:
        part = rows.loc[rows.region.eq(ba)].copy()
        if part.empty:
            raise ValueError(f"target EIA-930 response is missing {ba}")
        part.index = pd.DatetimeIndex(part.valid_at)
        hourly, audit = _proxy_rows(part, ba, schemas)
        hourly = hourly.loc[hourly.valid.astype(bool)].copy()
        hourly["time"] = pd.to_datetime(hourly.time, utc=True)
        hourly = hourly.rename(columns={f"generation_{fuel}": fuel for fuel in cc.FUELS})
        hourly["ba"] = ba
        expected = pd.date_range("2025-01-01", "2026-01-01", freq="h",
                                  inclusive="left", tz="UTC")
        target_rows = hourly.loc[hourly.time.dt.year.eq(2025)].copy()
        coverage = len(target_rows) / len(expected)
        if coverage < 0.80:
            raise ValueError(f"frozen target support gate failed for {ba}: {coverage:.4f}")
        response[ba] = target_rows
        audits[ba] = {
            **audit,
            "target_2025_expected_hours": len(expected),
            "target_2025_valid_hours": int(len(target_rows)),
            "target_2025_valid_fraction": float(coverage),
        }
    return pd.concat(response.values(), ignore_index=True), audits, source_hashes


def monthly_alignment_audit(inputs: pd.DataFrame, response: pd.DataFrame) -> pd.DataFrame:
    target = response.copy()
    target["period"] = ""
    for ba, indexes in target.groupby("ba").groups.items():
        local = pd.to_datetime(target.loc[indexes, "time"], utc=True).dt.tz_convert(cc.ZONES[ba])
        target.loc[indexes, "period"] = local.dt.strftime("%Y-%m").to_numpy()
    target["thermal"] = target[["coal", "gas", "oil", "other"]].sum(axis=1)
    records = []
    for (ba, period), label_month in target.groupby(["ba", "period"], sort=True):
        input_rows = inputs.loc[inputs.ba.eq(ba) & inputs.period.eq(period)]
        if input_rows.empty:
            continue
        input_row = input_rows.iloc[0]
        response_energy = label_month[list(cc.FUELS)].sum(axis=0)
        group_energy = {
            "thermal": float(response_energy[["coal", "gas", "oil", "other"]].sum()),
            **{name: float(response_energy[name]) for name in ("hydro", "nuclear", "wind", "solar")},
        }
        response_total = sum(group_energy.values())
        input_total = float(input_row.eia923_total_mwh)
        response_share = {name: value / response_total for name, value in group_energy.items()}
        l1 = sum(abs(response_share[name] - float(input_row[name])) for name in cc.COARSE)
        records.append({
            "ba": ba,
            "target_month": period,
            "input_observed_month": input_row.observed_month,
            "eia923_input_total_mwh": input_total,
            "eia930_proxy_total_mwh": response_total,
            "eia923_to_eia930_total_ratio": input_total / response_total if response_total else np.nan,
            "five_group_share_l1": float(l1),
            **{f"input_{name}_share": float(input_row[name]) for name in cc.COARSE},
            **{f"response_{name}_share": response_share[name] for name in cc.COARSE},
        })
    return pd.DataFrame(records)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    models, calibration, source_audit, source_model_audit = source_models()
    target_monthly, target_month_audit = target_month_features()
    target = target_inputs(target_monthly)
    predictions = cc.predict_models(models, target)
    pred = pd.DataFrame({
        "time": target.time.astype(str),
        "clock": target.clock.astype(str),
        "ba": target.ba,
        "period": target.period,
        "observed_month": target.observed_month,
        **{name: target[name] for name in cc.COARSE},
        **predictions,
    })
    for method in cc.METHODS:
        pred[f"{method}_bias"] = calibration[method]["bias_gco2_per_kwh"]
        pred[f"{method}_sigma"] = calibration[method]["sigma_gco2_per_kwh"]
    pred_file = OUTPUT / "sealed_predictions.csv.gz"
    pred.to_csv(pred_file, index=False, compression="gzip")
    calibration_file = OUTPUT / "source_calibration.json"
    calibration_file.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")

    source_paths = {item["path"] for item in source_audit.values()}
    source_hashes = {str((ROOT / name).resolve()): sha(ROOT / name)
                     for name in sorted(source_paths)}
    code_paths = [Path(__file__).resolve(), PROTOCOL, Path(cc.__file__).resolve(),
                  Path(normalize_balance.__code__.co_filename).resolve(),
                  Path(_proxy_rows.__code__.co_filename).resolve(),
                  Path(parse_eia923.__code__.co_filename).resolve()]
    code_hashes = {str(path.resolve()): sha(path) for path in code_paths}
    target_input_hashes = {
        **target_month_audit["archive_sha256"],
        **{str(path.resolve()): sha(path) for path in EIA930_FILES},
    }
    prediction_seal = {
        "status": "SEALED_GB_TO_US_2025_CN5_COMMON_EF_DEVELOPMENT",
        "prediction_sha256": sha(pred_file),
        "prediction_rows": int(len(pred)),
        "source_calibration_sha256": sha(calibration_file),
        "protocol_sha256": sha(PROTOCOL),
        "target_response_values_opened_before_seal": False,
        "target_response_previously_opened_elsewhere_in_project": True,
        "target_bas": list(TARGET_BAS),
        "target_ba_count": len(TARGET_BAS),
        "target_hours_per_ba": 8760,
        "target_input_contract": "EIA-923 final five-group shares, exact t-2 plus BA local calendar",
        "source_model": source_model_audit,
        "source_inputs_sha256": source_hashes,
        "target_files_sha256": target_input_hashes,
        "code_sha256": code_hashes,
        "target_month_input_audit": target_month_audit,
    }
    seal_file = OUTPUT / "prediction_seal.json"
    seal_file.write_text(json.dumps(prediction_seal, indent=2) + "\n", encoding="utf-8")
    if sha(pred_file) != prediction_seal["prediction_sha256"]:
        raise RuntimeError("prediction seal verification failed")
    if sha(PROTOCOL) != prediction_seal["protocol_sha256"]:
        raise RuntimeError("frozen protocol changed during prediction run")

    # Only after the prediction and calibration seal is verified, open the 2025 target response.
    response, response_audits, response_hashes = load_target_proxy()
    sealed = pd.read_csv(pred_file, compression="gzip")
    sealed["time"] = pd.to_datetime(sealed.time, utc=True, errors="raise")
    sealed["clock"] = pd.to_datetime(sealed.clock, errors="raise")
    response["time"] = pd.to_datetime(response.time, utc=True, errors="raise")
    response_for_join = response.rename(
        columns={fuel: f"response_{fuel}" for fuel in cc.FUELS}
    )
    response_columns = [f"response_{fuel}" for fuel in cc.FUELS]
    joined = sealed.merge(response_for_join[["ba", "time", *response_columns]],
                          on=["ba", "time"], how="left", validate="one_to_one", indicator=True)
    joined = joined.loc[joined._merge.eq("both")].copy()
    target_fuel = joined[response_columns].rename(
        columns={f"response_{fuel}": fuel for fuel in cc.FUELS}
    )
    joined["y_fixed_ef_proxy"] = cc.labels(target_fuel)
    align = monthly_alignment_audit(target, response)
    align.to_csv(OUTPUT / "monthly_input_response_alignment.csv", index=False)
    support = (joined.groupby("ba").size() / 8760).rename("valid_fraction")
    if set(support.index) != set(TARGET_BAS) or (support < .80).any():
        raise ValueError("sealed target support gate failed; do not interpret model metrics")

    region_rows, monthly_rows = [], []
    for ba, group in joined.groupby("ba", sort=True):
        inputs = pd.DataFrame({"clock": pd.to_datetime(group.clock)})
        y = group.y_fixed_ef_proxy.to_numpy(float)
        for method in cc.METHODS:
            metric = cc.metrics(inputs, y, group[method].to_numpy(float),
                                float(group[f"{method}_bias"].iloc[0]),
                                float(group[f"{method}_sigma"].iloc[0]))
            region_rows.append({"ba": ba, "method": method,
                                "valid_hours": len(group), "valid_fraction": len(group)/8760,
                                **metric})
            for month, month_group in group.groupby(group.clock.dt.month, sort=True):
                month_inputs = pd.DataFrame({"clock": pd.to_datetime(month_group.clock)})
                month_metrics = cc.metrics(
                    month_inputs,
                    month_group.y_fixed_ef_proxy.to_numpy(float),
                    month_group[method].to_numpy(float),
                    float(month_group[f"{method}_bias"].iloc[0]),
                    float(month_group[f"{method}_sigma"].iloc[0]),
                )
                monthly_rows.append({"ba": ba, "month": int(month), "method": method,
                                     "valid_hours": len(month_group), **month_metrics})
    regions = pd.DataFrame(region_rows)
    regions.to_csv(OUTPUT / "region_metrics.csv", index=False)
    monthly = pd.DataFrame(monthly_rows)
    monthly.to_csv(OUTPUT / "monthly_metrics.csv", index=False)
    metric_fields = ["mae", "shape_mae", "crps_gaussian", "coverage90", "width90", "interval_score90"]
    macro = regions.groupby("method")[metric_fields].mean().reset_index()
    macro.insert(1, "target_ba_count", len(TARGET_BAS))
    macro.insert(2, "macro_hours_per_ba", regions.groupby("method").hours.mean().values)
    macro.to_csv(OUTPUT / "summary_metrics.csv", index=False)
    response_audit_file = OUTPUT / "target_response_audit.json"
    response_audit_file.write_text(json.dumps({
        "target_response": "EIA-930 common fixed-EF generation proxy",
        "target_response_files_sha256": response_hashes,
        "target_response_previously_opened_elsewhere_in_project": True,
        "ba_audits": response_audits,
        "scored_hours_by_ba": {ba: int(value) for ba, value in support.items()},
        "monthly_input_response_scope": {
            "ba_month_rows": len(align),
            "median_eia923_to_eia930_total_ratio": float(align.eia923_to_eia930_total_ratio.median()),
            "median_five_group_share_l1": float(align.five_group_share_l1.median()),
        },
    }, indent=2) + "\n", encoding="utf-8")
    outputs = ["source_calibration.json", "sealed_predictions.csv.gz", "prediction_seal.json",
               "monthly_input_response_alignment.csv", "region_metrics.csv", "monthly_metrics.csv",
               "summary_metrics.csv", "target_response_audit.json"]
    score_seal = {
        "status": "SCORED_AFTER_PREDICTION_SEAL",
        "prediction_sha256": prediction_seal["prediction_sha256"],
        "protocol_sha256": prediction_seal["protocol_sha256"],
        "target_response_files_sha256": response_hashes,
        "scoring_code_sha256": sha(Path(__file__).resolve()),
        "output_sha256": {name: sha(OUTPUT / name) for name in outputs},
    }
    (OUTPUT / "score_seal.json").write_text(json.dumps(score_seal, indent=2) + "\n", encoding="utf-8")
    print(macro.to_string(index=False), flush=True)
    print(json.dumps({"prediction_rows": len(pred), "joined_target_hours": len(joined),
                      "minimum_ba_coverage": float(support.min()),
                      "median_input_response_mix_l1": float(align.five_group_share_l1.median()),
                      "score_seal_sha256": sha(OUTPUT / "score_seal.json")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
