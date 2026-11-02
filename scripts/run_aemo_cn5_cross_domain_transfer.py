"""Frozen US/GB-to-AEMO CN5 transfer development experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import preflight_aemo_cn5_fuel_monthly_support as preflight  # noqa: E402
from transcif.research.cross_country import (  # noqa: E402
    COARSE, FUELS, METHODS, fit_models, labels, load_inputs, metrics,
    monthly_inputs, predict_models,
)
from transcif.data.aemo import sha256  # noqa: E402

PROTOCOL = ROOT / "docs/AEMO_CN5_CROSS_DOMAIN_TRANSFER_PROTOCOL_20261102.md"
DEFAULT_STAGING = Path("/Users/cyyc0310/Desktop/tran/data/staging/aemo-native-2023")
DEFAULT_PREFLIGHT = ROOT / "results/aemo-cn5-fuel-preflight-r1"
DEFAULT_PROXY = ROOT / "results/aemo-cdeii-validation-r3/hourly_scada_ef_proxy_2023.csv"
SOURCE_CONFIGS = ("US-only", "GB-only", "US+GB")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_aemo_inputs(staging: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Rebuild target five-group monthly totals without reading any target label."""
    metadata_path, snapshots = preflight.validation._snapshot_index(staging)
    details = preflight.validation._read_staged(staging, snapshots[("DETAIL", "202312")])
    allocations = preflight.validation._read_staged(staging, snapshots[("ALLOC", "202312")])
    pieces, provenance = [], {str(metadata_path.resolve()): sha256(metadata_path)}
    for month in sorted(m for table, m in snapshots if table == "SCADA"):
        gensets = preflight.validation._read_staged(staging, snapshots[("GENUNITS", month)])
        hourly, _ = preflight.month_record(staging, month, snapshots, details, allocations, gensets)
        pieces.append(hourly)
        for key in (("SCADA", month), ("GENUNITS", month)):
            item = snapshots[key]
            provenance[str((staging / item["output"]).resolve())] = item["output_sha256"]
    for key in (("DETAIL", "202312"), ("ALLOC", "202312")):
        item = snapshots[key]
        provenance[str((staging / item["output"]).resolve())] = item["output_sha256"]

    raw = pd.concat(pieces, ignore_index=True)
    raw["clock"] = pd.to_datetime(raw["date"]) + pd.to_timedelta(raw["hour"], unit="h")
    raw["time"] = raw["clock"]
    raw["country"] = "AEMO"
    raw["expected_hours"] = raw["clock"].dt.days_in_month * 24
    # monthly_inputs expects the same eight fuel fields as source data. These
    # five known generation groups fill the corresponding columns; unmodelled
    # fossil and renewable details have already been combined in thermal.
    raw["coal"] = raw["thermal"]
    raw["gas"] = 0.
    raw["oil"] = 0.
    raw["other"] = 0.
    raw["nuclear"] = raw["nuclear"]
    raw["hydro"] = raw["hydro"]
    raw["wind"] = raw["wind"]
    raw["solar"] = raw["solar"]
    raw["region"] = raw["region"].astype(str)
    raw = raw.sort_values(["region", "clock"]).reset_index(drop=True)
    return raw, provenance


def fit_and_calibrate(frames: dict[str, pd.DataFrame], prepared: dict[str, pd.DataFrame],
                      source: str) -> tuple[dict, dict[str, tuple[float, float]], dict]:
    eligible_sources = [r for r, frame in frames.items()
                        if (source == "US+GB" or frame.country.iloc[0] == source)]
    train_rows, train_fuel = [], []
    for region in eligible_sources:
        inp = prepared[region]
        subset = inp.loc[inp.year.eq(2022)]
        if subset.empty:
            continue
        train_rows.append(subset)
        raw_subset = frames[region].loc[subset.raw_row]
        train_fuel.append(raw_subset[list(FUELS)].to_numpy(float))
    train = pd.concat(train_rows, ignore_index=True)
    fuel = np.concatenate(train_fuel)
    models = fit_models(train, fuel)

    residuals = {method: [] for method in METHODS}
    calibration_regions = []
    for region in eligible_sources:
        inp = prepared[region]
        cal = inp.loc[inp.year.eq(2023)]
        if cal.empty:
            continue
        prediction = predict_models(models, cal)
        y = labels(frames[region].loc[cal.raw_row])
        calibration_regions.append(region)
        for method in METHODS:
            residuals[method].append(y - prediction[method])
    calibration = {}
    for method, parts in residuals.items():
        if not parts:
            raise ValueError(f"no source calibration residuals for {source}/{method}")
        bias = float(np.mean([values.mean() for values in parts]))
        sigma = float(np.sqrt(np.mean([np.mean((values - bias) ** 2) for values in parts])))
        calibration[method] = (bias, max(sigma, 1.))
    audit = {"training_regions": eligible_sources, "calibration_regions": calibration_regions,
             "training_hours": int(len(train)), "train_year": 2022, "calibration_year": 2023}
    return models, calibration, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--proxy", type=Path, default=DEFAULT_PROXY)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    # Target labels are intentionally not opened anywhere above this line.
    raw_source, source_audit = load_inputs(ROOT)
    source_prepared = {region: monthly_inputs(frame) for region, frame in raw_source.items()}
    source_usable = {}
    for region, frame in source_prepared.items():
        usable = frame.dropna(subset=list(COARSE)).copy()
        usable["raw_row"] = usable.index
        source_usable[region] = usable
    aemo_raw, aemo_provenance = make_aemo_inputs(args.staging)
    target_prepared = {region: monthly_inputs(frame) for region, frame in
                       ((region, part.copy()) for region, part in aemo_raw.groupby("region", sort=True))}
    target_usable = {}
    for region, frame in target_prepared.items():
        target_usable[region] = frame.dropna(subset=list(COARSE)).copy()

    support = pd.read_csv(args.preflight / "region_support.csv").set_index("region")
    primary_regions = sorted(support.index[support.eligible_fraction.ge(.95)].tolist())
    if primary_regions != ["NSW1", "QLD1", "TAS1"]:
        raise ValueError(f"frozen target support gate changed: {primary_regions}")
    target_regions = sorted(target_usable)
    all_predictions = []
    source_audits, calibration_audits = {}, {}
    for source in SOURCE_CONFIGS:
        source_label = {"US-only": "US", "GB-only": "GB", "US+GB": "US+GB"}[source]
        models, calibration, audit = fit_and_calibrate(raw_source, source_usable, source_label)
        source_audits[source] = audit
        calibration_audits[source] = {method: {"bias": value[0], "sigma": value[1]}
                                      for method, value in calibration.items()}
        for region in target_regions:
            target = target_usable[region]
            prediction = predict_models(models, target)
            pred = pd.DataFrame({"time_local": target.clock.astype(str),
                                 "observed_month": target.observed_month,
                                 "region": region, "source": source,
                                 "primary_region": region in primary_regions,
                                 "eligible_cn5_hour": aemo_raw.loc[target.index, "eligible_cn5_hour"].to_numpy(),
                                 **{f"input_{fuel}": target[fuel].to_numpy() for fuel in COARSE},
                                 **prediction})
            for method, (bias, sigma) in calibration.items():
                pred[f"{method}_bias"] = bias
                pred[f"{method}_sigma"] = sigma
            all_predictions.append(pred)

    prediction_frame = pd.concat(all_predictions, ignore_index=True)
    # Restrict the frozen experiment's scored period. The monthly feature fields
    # themselves are strictly t-2 and the target proxy remains unopened.
    prediction_frame["month"] = pd.to_datetime(prediction_frame.time_local).dt.strftime("%Y%m")
    prediction_frame = prediction_frame.loc[prediction_frame.month.between("202303", "202312")].copy()
    prediction_path = out / "predictions.csv"
    prediction_frame.to_csv(prediction_path, index=False)
    seal = {"prediction_file": prediction_path.name, "prediction_sha256": digest(prediction_path),
            "protocol_sha256": digest(PROTOCOL),
            "runner_sha256": digest(Path(__file__)),
            "method_module_sha256": digest(ROOT / "src/transcif/research/cross_country.py"),
            "aemo_input_sha256": aemo_provenance,
            "source_input_sha256": {v["path"]: digest(ROOT / v["path"])
                                    for v in source_audit.values()},
            "primary_regions": primary_regions, "all_candidate_regions": target_regions,
            "source_models": source_audits, "source_calibration": calibration_audits,
            "target_proxy_read_before_seal": False}
    (out / "prediction_seal.json").write_text(json.dumps(seal, indent=2) + "\n")

    # Verify the sealed forecast before opening the outcome proxy.
    if digest(prediction_path) != seal["prediction_sha256"]:
        raise ValueError("prediction seal mismatch")
    proxy = pd.read_csv(args.proxy)
    proxy["clock"] = (pd.to_datetime(proxy.time_utc, utc=True)
                      .dt.tz_convert("Etc/GMT-10").dt.tz_localize(None))
    proxy["month"] = proxy.clock.dt.strftime("%Y%m")
    proxy = proxy.loc[proxy.month.between("202303", "202312")].copy()
    labels_by_hour = proxy[["clock", "region", "proxy_cdeii_tco2e_mwh", "hourly_label_eligible"]].copy()
    scored = prediction_frame.merge(labels_by_hour, left_on=["time_local", "region"],
                                    right_on=[labels_by_hour.clock.astype(str), "region"],
                                    how="left", validate="many_to_one")
    if scored.proxy_cdeii_tco2e_mwh.isna().any():
        raise ValueError("missing AEMO outcome rows after seal")
    scored["y_g_kwh"] = scored.proxy_cdeii_tco2e_mwh * 1000.
    score_rows = []
    for (source, region), group in scored.groupby(["source", "region"], sort=True):
        if region not in primary_regions:
            continue
        mask = group.eligible_cn5_hour.astype(bool) & group.hourly_label_eligible.astype(bool)
        eligible = group.loc[mask].copy()
        if eligible.empty:
            continue
        inputs = pd.DataFrame({"clock": pd.to_datetime(eligible.time_local)})
        y = eligible.y_g_kwh.to_numpy(float)
        for method in METHODS:
            values = metrics(inputs, y, eligible[method].to_numpy(float),
                             float(eligible[f"{method}_bias"].iloc[0]),
                             float(eligible[f"{method}_sigma"].iloc[0]))
            score_rows.append({"source": source, "region": region,
                               "scored_hours": len(eligible), "method": method, **values})
    region_metrics = pd.DataFrame(score_rows)
    if region_metrics.empty:
        raise ValueError("no primary-region eligible hours to score")
    region_metrics.to_csv(out / "region_metrics.csv", index=False)
    summary = region_metrics.groupby(["source", "method"]).mean(numeric_only=True).reset_index()
    summary.to_csv(out / "primary_region_mean_metrics.csv", index=False)
    (out / "scoring_manifest.json").write_text(json.dumps({
        "target_proxy_path": str(args.proxy.resolve()),
        "target_proxy_sha256": digest(args.proxy),
        "prediction_sha256_verified": seal["prediction_sha256"],
        "region_metrics_sha256": digest(out / "region_metrics.csv"),
        "scored_regions": primary_regions,
        "score_months": "2023-03..2023-12",
        "outcome_opened_after_prediction_seal": True,
    }, indent=2) + "\n")
    (out / "status.json").write_text(json.dumps({
        "status": "DEVELOPMENT_ONLY_RETROSPECTIVE_AEMO_HOURLY_PROXY",
        "primary_regions": primary_regions, "support_failed_diagnostic_regions": [
            region for region in target_regions if region not in primary_regions],
        "score_months": "2023-03..2023-12", "lag_months": 2,
        "label": "retrospective DUID EF CO2e hourly proxy; not official hourly CIF",
        "consumption_track": False, "unseen_external_confirmation": False,
        "source_targets": "fixed-factor domestic generation proxies",
        "target_features_reconstructed_from_retrospective_SCADA": True,
        "target_proxy_opened_after_prediction_seal": True,
    }, indent=2) + "\n")
    print(region_metrics.groupby(["source", "method"])[
        ["scored_hours", "mae", "shape_mae", "crps_gaussian", "coverage90", "width90",
         "interval_score90"]].mean().round(4).to_string())
    print(f"wrote AEMO transfer development result: {out}", flush=True)


if __name__ == "__main__":
    main()
