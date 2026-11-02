"""Score sealed US/GB-to-AEMO forecasts under source-matched fixed fuel EFs."""
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
import validate_aemo_daily_proxy as validation  # noqa: E402
from transcif.data.aemo import sha256  # noqa: E402
from transcif.data.aemo_registry import registry_at  # noqa: E402
from transcif.research.cross_country import METHODS, metrics  # noqa: E402

PROTOCOL = ROOT / "docs/AEMO_CN5_COMMON_EF_CROSS_DOMAIN_PROTOCOL_20261102.md"
ORIGINAL_PROTOCOL = ROOT / "docs/AEMO_CN5_CROSS_DOMAIN_TRANSFER_PROTOCOL_20261102.md"
PREDICTIONS = ROOT / "results/aemo-cn5-cross-domain-transfer-r2/predictions.csv"
PREDICTION_SEAL = ROOT / "results/aemo-cn5-cross-domain-transfer-r2/prediction_seal.json"
PREFLIGHT = ROOT / "results/aemo-cn5-fuel-preflight-r1/region_support.csv"
DEFAULT_STAGING = Path("/Users/cyyc0310/Desktop/tran/data/staging/aemo-native-2023")
DEFAULT_PROXY = ROOT / "results/aemo-cdeii-validation-r3/hourly_scada_ef_proxy_2023.csv"
EXPECTED_PREDICTION_SHA = "bcc11f64680df9c122655f5d8c1ffe337baf6f9a22cd10fa85f9efae2bd9bbb1"
PRIMARY_REGIONS = ("NSW1", "QLD1", "TAS1")
EF = {"coal": 1000.0, "gas": 400.0, "oil": 800.0, "other": 400.0,
      "nuclear": 0.0, "hydro": 0.0, "wind": 0.0, "solar": 0.0}

FUEL_GROUP = {
    "black coal": "coal", "brown coal": "coal",
    "natural gas (pipeline)": "gas", "coal seam methane": "gas",
    "coal mine waste gas": "gas",
    "diesel oil": "oil", "kerosene - non aviation": "oil",
    "landfill biogas methane": "other", "bagasse": "other",
    "biomass and industrial materials": "other", "other biofuels": "other",
    "primary solid biomass fuels": "other", "other solid fossil fuels": "other",
    "hydro": "hydro", "nuclear": "nuclear", "wind": "wind", "solar": "solar",
}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_prediction_and_inputs(staging: Path) -> tuple[dict, dict, pd.DataFrame]:
    seal = json.loads(PREDICTION_SEAL.read_text())
    if seal["prediction_sha256"] != EXPECTED_PREDICTION_SHA or _hash(PREDICTIONS) != EXPECTED_PREDICTION_SHA:
        raise ValueError("original sealed prediction hash mismatch")
    if _hash(ORIGINAL_PROTOCOL) != seal["protocol_sha256"]:
        raise ValueError("original AEMO prediction protocol no longer matches its seal")
    prediction = pd.read_csv(PREDICTIONS)
    if prediction.duplicated(["time_local", "region", "source"]).any():
        raise ValueError("duplicate sealed prediction keys")
    metadata, snapshots = validation._snapshot_index(staging)
    current_inputs = {str(metadata.resolve()): sha256(metadata)}
    for key, item in snapshots.items():
        if key[0] not in ("SCADA", "GENUNITS", "DETAIL", "ALLOC"):
            continue
        path = staging / item["output"]
        current_inputs[str(path.resolve())] = sha256(path)
    for path, expected in seal["aemo_input_sha256"].items():
        if current_inputs.get(path) != expected:
            raise ValueError(f"AEMO input differs from prediction seal: {path}")
    return seal, snapshots, prediction


def _aemo_fuel_labels(staging: Path, snapshots: dict) -> pd.DataFrame:
    """Build hourly fixed-EF labels from registered fuel categories and SCADA."""
    detail = validation._read_staged(staging, snapshots[("DETAIL", "202312")])
    allocation = validation._read_staged(staging, snapshots[("ALLOC", "202312")])
    pieces = []
    for month in sorted(m for table, m in snapshots if table == "SCADA"):
        gensets = validation._read_staged(staging, snapshots[("GENUNITS", month)])
        instant = pd.Timestamp(f"{month[:4]}-{month[4:]}-15T12:00:00+10:00")
        records = registry_at(sorted(detail.DUID.unique()), instant, detail, allocation, gensets)["records"]
        region_by_duid, group_by_duid = {}, {}
        for unit in records:
            if unit["dispatch_type_candidates"] != ["GENERATOR"]:
                continue
            regions = unit["region_candidates"]
            if len(regions) != 1 or regions[0] not in validation.REGIONS:
                continue
            fuel_names = {str(value).strip().lower() for value in unit["fuel_candidates"] if str(value).strip()}
            mapped = {FUEL_GROUP[name] for name in fuel_names if name in FUEL_GROUP}
            # An ambiguous/mixed registration is omitted and counted against support.
            region_by_duid[unit["duid"]] = regions[0]
            if len(fuel_names) == 1 and len(mapped) == 1:
                group_by_duid[unit["duid"]] = next(iter(mapped))

        item = snapshots[("SCADA", month)]
        scada_path = staging / item["output"]
        if sha256(scada_path) != item["output_sha256"]:
            raise ValueError(f"SCADA hash mismatch for {month}")
        monthly_chunks = []
        for chunk in pd.read_csv(scada_path,
                                 usecols=["dispatch_interval_timestamp_utc", "duid", "scada_mw"],
                                 dtype={"duid": str}, chunksize=500_000):
            mapped_region = chunk.duid.map(region_by_duid)
            keep = mapped_region.notna()
            if not keep.any():
                continue
            selected = chunk.loc[keep].reset_index(drop=True)
            local = pd.to_datetime(selected.dispatch_interval_timestamp_utc, utc=True)
            local = local.dt.tz_convert("Etc/GMT-10").dt.tz_localize(None)
            within_month = local.dt.strftime("%Y%m").eq(month).to_numpy()
            if not within_month.any():
                continue
            selected = selected.loc[within_month].reset_index(drop=True)
            local = local.loc[within_month].reset_index(drop=True).dt.floor("h")
            duid = selected.duid
            mw = pd.to_numeric(selected.scada_mw, errors="coerce").fillna(0.).clip(lower=0.)
            category = duid.map(group_by_duid).fillna("unmapped")
            part = pd.DataFrame({"time_local": local.to_numpy(),
                                 "region": selected.duid.map(region_by_duid).to_numpy(),
                                 "fuel": category.to_numpy(), "mw": mw.to_numpy()})
            monthly_chunks.append(part.groupby(["time_local", "region", "fuel"],
                                               as_index=False).mw.sum())
        if not monthly_chunks:
            raise ValueError(f"no SCADA fuel labels for {month}")
        pieces.append(pd.concat(monthly_chunks, ignore_index=True).groupby(
            ["time_local", "region", "fuel"], as_index=False).mw.sum())

    detail_rows = pd.concat(pieces, ignore_index=True)
    pivot = detail_rows.pivot_table(index=["time_local", "region"], columns="fuel",
                                    values="mw", aggfunc="sum", fill_value=0.)
    for name in EF:
        if name not in pivot:
            pivot[name] = 0.
    mapped_names = list(EF)
    mapped_total = pivot[mapped_names].sum(axis=1)
    all_total = pivot.sum(axis=1)
    support = mapped_total.div(all_total.replace(0., np.nan))
    generation = pivot[mapped_names].to_numpy(float)
    emissions_proxy = generation @ np.array([EF[name] for name in mapped_names])
    y = emissions_proxy / mapped_total.to_numpy(float)
    result = pd.DataFrame({"time_local": pivot.index.get_level_values("time_local"),
                           "region": pivot.index.get_level_values("region"),
                           **{f"generation_{name}_mw": pivot[name].to_numpy(float)
                              for name in mapped_names},
                           "fixed_ef_numerator": emissions_proxy,
                           "fixed_ef_cif_g_kwh": y,
                           "mapped_fuel_mw": mapped_total.to_numpy(float),
                           "all_registered_positive_mw": all_total.to_numpy(float),
                           "fixed_ef_fuel_support": support.to_numpy(float)})
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=["fixed_ef_cif_g_kwh"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--proxy", type=Path, default=DEFAULT_PROXY)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    prediction_seal, snapshots, prediction = verify_prediction_and_inputs(args.staging)
    prediction["time_local"] = pd.to_datetime(prediction.time_local, errors="raise")
    proxy_hash = sha256(args.proxy)
    startup = {"protocol_sha256": _hash(PROTOCOL), "runner_sha256": _hash(Path(__file__)),
               "prediction_sha256": prediction_seal["prediction_sha256"],
               "prediction_seal_sha256": _hash(PREDICTION_SEAL),
               "original_protocol_sha256": prediction_seal["protocol_sha256"],
               "original_runner_sha256": prediction_seal["runner_sha256"],
               "source_module_sha256": prediction_seal["method_module_sha256"],
               "aemo_input_sha256": prediction_seal["aemo_input_sha256"],
               "native_proxy_sha256": proxy_hash}
    (out / "startup_hashes.json").write_text(json.dumps(startup, indent=2) + "\n")

    labels = _aemo_fuel_labels(args.staging, snapshots)
    label_path = out / "target_fixed_ef_labels.csv"
    labels.to_csv(label_path, index=False)
    label_seal = {"labels_sha256": _hash(label_path), "label_rows": len(labels),
                  "ef_g_per_kwh": EF, "target_label_opened_after_prediction_verification": True}
    (out / "target_label_seal.json").write_text(json.dumps(label_seal, indent=2) + "\n")

    proxy = pd.read_csv(args.proxy)
    proxy["time_local"] = (pd.to_datetime(proxy.time_utc, utc=True)
                           .dt.tz_convert("Etc/GMT-10").dt.tz_localize(None).dt.floor("h"))
    proxy = proxy[["time_local", "region", "proxy_cdeii_tco2e_mwh", "hourly_label_eligible"]]
    scoring = prediction.merge(labels, on=["time_local", "region"], how="inner", validate="many_to_one")
    scoring = scoring.merge(proxy, on=["time_local", "region"], how="inner", validate="many_to_one")
    scoring["native_proxy_g_kwh"] = scoring.proxy_cdeii_tco2e_mwh * 1000.
    scoring["month_key"] = pd.to_datetime(scoring.time_local).dt.strftime("%Y%m")
    scoring = scoring.loc[scoring.month_key.between("202303", "202312")
                          & scoring.region.isin(PRIMARY_REGIONS)
                          & scoring.eligible_cn5_hour.astype(bool)
                          & scoring.hourly_label_eligible.astype(bool)
                          & scoring.fixed_ef_fuel_support.ge(.99)].copy()
    if set(scoring.region.unique()) != set(PRIMARY_REGIONS):
        raise ValueError("paired supported sample does not include all frozen primary regions")
    if scoring.duplicated(["time_local", "region", "source"]).any():
        raise ValueError("duplicated scoring keys")
    scoring.to_csv(out / "paired_scoring_rows.csv", index=False)

    score_rows = []
    for (source, region), group in scoring.groupby(["source", "region"], sort=True):
        metrics_input = pd.DataFrame({"clock": pd.to_datetime(group.time_local)})
        for label_name, y_name in [("common_fixed_ef", "fixed_ef_cif_g_kwh"),
                                   ("native_cdeii_proxy", "native_proxy_g_kwh")]:
            y = group[y_name].to_numpy(float)
            for method in METHODS:
                values = metrics(metrics_input, y, group[method].to_numpy(float),
                                 float(group[f"{method}_bias"].iloc[0]),
                                 float(group[f"{method}_sigma"].iloc[0]))
                score_rows.append({"source": source, "region": region, "label": label_name,
                                   "scored_hours": len(group), "method": method, **values})
    region_metrics = pd.DataFrame(score_rows)
    region_metrics.to_csv(out / "region_metrics.csv", index=False)
    macro = region_metrics.groupby(["source", "label", "method"]).mean(numeric_only=True).reset_index()
    macro.to_csv(out / "macro_metrics.csv", index=False)

    summary = {"status": "DEVELOPMENT_ONLY_LABEL_DEFINITION_SENSITIVITY",
               "prediction_reused": str(PREDICTIONS), "prediction_sha256": startup["prediction_sha256"],
               "prediction_seal_verified": True, "score_months": "2023-03..2023-12",
               "primary_regions": list(PRIMARY_REGIONS), "scoring_rows_per_source_region": {
                   f"{source}/{region}": int(len(group))
                   for (source, region), group in scoring.groupby(["source", "region"])},
               "hour_support_min": float(scoring.fixed_ef_fuel_support.min()),
               "common_fixed_ef": EF, "native_proxy_sha256": proxy_hash,
               "target_label_kind": "reconstructed standardized fixed-factor generation proxy",
               "official_hourly_cif": False, "consumption_track": False,
               "target_labels_read_before_prediction_verification": False,
               "limitations": ["AEMO remains a single retrospective development system",
                               "SCADA-derived target monthly inputs and labels are not as-issued",
                               "fixed EF labels are constructed proxies, not measured emissions",
                               "same predictions and source calibration are reused for both target label variants"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(macro.round(4).to_string(index=False), flush=True)
    print(f"wrote AEMO fixed-EF label sensitivity: {out}", flush=True)


if __name__ == "__main__":
    main()
