"""Compare actual EIA-923 BA/month mix with matched EIA-930 oracle mix.

This is an alignment/information-value diagnostic, not a prospective replay:
the 2026 August EIA-923 release was issued after the evaluated Jan-Aug months.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import platform

import numpy as np
import pandas as pd
import scipy
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import audit_live_eia923_ba_vintage as vintage
import run_actual_issue_cross_country as source_model
from transcif.data.eia930 import normalize_balance
from transcif.research.cross_country import COARSE, FUELS, ZONES, labels, metrics, predict_models
from transcif.research.eia_proxy import _proxy_rows

OUT = ROOT / "results/eia923-actual-ba-information-value-r1"
EIA923 = ROOT / "docs/evidence/eia923-live-monthly-2026-08-r1/f923_2026.zip"
ALIGNMENT = ROOT / "docs/evidence/eia923-live-monthly-2026-08-r1/results/ba_month_alignment.csv"
PROTOCOL = ROOT / "docs/EIA923_ACTUAL_BA_INFORMATION_VALUE_PROTOCOL_20261101.md"
BASES = tuple(vintage.DEFAULT_BAS)
MONTHS = range(1, 9)
NATIVE = [ROOT / "data/native-eia-2025/EIA930_BALANCE_2025_Jan_Jun.csv",
          ROOT / "data/native-eia-2025/EIA930_BALANCE_2025_Jul_Dec.csv",
          ROOT / "data/native-eia-2026/EIA930_BALANCE_2026_Jan_Jun.csv",
          ROOT / "data/native-eia-2026/EIA930_BALANCE_2026_Jul_Dec.csv"]
SOURCE_INPUTS = [ROOT / f"results/research/data/{region}_hourly.csv" for region in ZONES]
SOURCE_INPUTS += [ROOT / "data/neso-gb/df_fuel_ckan.csv"]
CODE_INPUTS = [Path(__file__).resolve(), Path(vintage.__file__).resolve(),
               Path(source_model.__file__).resolve(), PROTOCOL,
               ROOT / "src/transcif/data/eia930.py",
               ROOT / "src/transcif/research/cross_country.py",
               ROOT / "src/transcif/research/eia_proxy.py"]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def to_coarse(record: pd.Series, prefix: str) -> dict[str, float]:
    values = {fuel: float(record[f"{prefix}_{fuel}_mwh"]) for fuel in FUELS}
    values["thermal"] = sum(values[f] for f in ("coal", "gas", "oil", "other"))
    total = sum(values[f] for f in FUELS)
    if not np.isfinite(total) or total <= 0 or min(values.values()) < 0:
        raise ValueError(f"invalid fuel vector in {prefix}: {values}")
    shares = {name: values[name] / total for name in COARSE}
    if abs(sum(shares.values()) - 1.) > 1e-10:
        raise ValueError(f"five-group shares fail closure: {shares}")
    return shares


def feature_rows(hourly: pd.DataFrame, shares: dict[str, float], ba: str) -> pd.DataFrame:
    frame = hourly[["time"]].copy()
    utc = pd.to_datetime(frame.time, utc=True, errors="raise")
    local = utc.dt.tz_convert(ZONES[ba]).dt.tz_localize(None)
    frame["clock"] = local
    frame["year"] = local.dt.year
    for name, values, cycle in (("hour", local.dt.hour, 24),
                                ("dow", local.dt.dayofweek, 7),
                                ("month", local.dt.month, 12)):
        frame[f"{name}_sin"] = np.sin(2 * np.pi * values / cycle)
        frame[f"{name}_cos"] = np.cos(2 * np.pi * values / cycle)
    for category in COARSE:
        frame[category] = shares[category]
        for suffix in ("sin", "cos"):
            frame[f"{category}_hour_{suffix}"] = frame[category] * frame[f"hour_{suffix}"]
    return frame


def load_hourly(paths: list[Path], bas: tuple[str, ...]) -> pd.DataFrame:
    parts, schemas = [], {}
    for snapshot, path in enumerate(paths):
        native = pd.read_csv(path)
        native = native.loc[native["Balancing Authority"].isin(bas)].copy()
        normalized = normalize_balance(native)
        normalized["source_snapshot"] = snapshot
        schemas[str(snapshot)] = {"fuel_columns": [c for c in normalized
                                                   if c.startswith("generation_")]}
        parts.append(normalized)
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.loc[(combined.valid_at >= "2026-01-01")
                            & (combined.valid_at < "2026-09-01")].copy()
    outputs = []
    # Proxy quality is indexed by timestamp, so process each BA separately;
    # timestamps naturally repeat across different balancing authorities.
    for ba, region_rows in combined.groupby("region", sort=True):
        region_rows = region_rows.copy()
        region_rows.index = pd.DatetimeIndex(region_rows.valid_at)
        region_rows = region_rows.sort_index()
        hourly, _ = _proxy_rows(region_rows, ba, schemas)
        outputs.append(hourly.loc[hourly.valid.astype(bool)].copy())
    result = pd.concat(outputs, ignore_index=True)
    result["time"] = pd.to_datetime(result.time, utc=True, errors="raise")
    result["month"] = result.time.dt.strftime("%Y-%m")
    return result


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    if not all(path.is_file() for path in [EIA923, ALIGNMENT, PROTOCOL, *NATIVE]):
        raise FileNotFoundError("one or more frozen diagnostic inputs are missing")
    if not all(path.is_file() for path in [*SOURCE_INPUTS, *CODE_INPUTS]):
        raise FileNotFoundError("one or more source-model provenance inputs are missing")

    audit = pd.read_csv(ALIGNMENT).sort_values(["ba", "month"]).reset_index(drop=True)
    monthly, _ = vintage.parse_monthly_generation(EIA923, 2026, MONTHS, BASES)
    hourly_summary = vintage.parse_hourly(NATIVE, 2026, MONTHS, BASES)
    recomputed = (monthly.merge(hourly_summary, on=["ba", "month"], validate="one_to_one")
                  .sort_values(["ba", "month"]).reset_index(drop=True))
    for fuel in FUELS:
        recomputed[f"eia923_share_{fuel}"] = (
            recomputed[f"eia923_{fuel}_mwh"] / recomputed.eia923_total_mwh)
        recomputed[f"eia930_share_{fuel}"] = (
            recomputed[f"eia930_{fuel}_mwh"] / recomputed.eia930_total_mwh)
    recomputed["total_ratio_eia923_to_eia930"] = (
        recomputed.eia923_total_mwh / recomputed.eia930_total_mwh)
    recomputed["mix_share_l1"] = sum(
        (recomputed[f"eia923_share_{fuel}"] - recomputed[f"eia930_share_{fuel}"]).abs()
        for fuel in FUELS)
    comparison = recomputed.merge(audit, on=["ba", "month"], suffixes=("_current", "_prior"),
                                  validate="one_to_one")
    changed_fields = []
    changed_rows = np.zeros(len(comparison), dtype=bool)
    compared_fields = ["eia923_complete", "eia930_complete", "eia923_total_mwh",
                       "eia930_total_mwh", "eia930_native_hours", "eia930_valid_hours",
                       "eia930_expected_hours", "total_ratio_eia923_to_eia930", "mix_share_l1"]
    compared_fields.extend(f"eia923_{fuel}_mwh" for fuel in FUELS)
    compared_fields.extend(f"eia930_{fuel}_mwh" for fuel in FUELS)
    for base in compared_fields:
        left, right = comparison[f"{base}_current"], comparison[f"{base}_prior"]
        if base.endswith("complete"):
            equal = left.astype(bool).to_numpy() == right.astype(bool).to_numpy()
        else:
            equal = np.isclose(left.to_numpy(float), right.to_numpy(float),
                               rtol=1e-10, atol=1e-8, equal_nan=True)
        if not equal.all():
            changed_fields.append(base)
            changed_rows |= ~equal

    eligible = recomputed.loc[recomputed.eia923_complete & recomputed.eia930_complete
                              & recomputed.eia923_total_mwh.gt(0)
                              & recomputed.eia930_total_mwh.gt(0)].copy()
    if len(eligible) != 20 or eligible.groupby("ba").size().to_dict() != {
            "BPAT": 6, "CISO": 1, "ERCO": 2, "FPL": 7, "NYIS": 4}:
        raise ValueError("current raw inputs no longer match the frozen 20-cell comparison queue")
    eligible["eia923_hydro_mwh"] = pd.to_numeric(eligible.eia923_hydro_mwh, errors="raise")
    if (eligible[[f"eia923_{fuel}_mwh" for fuel in FUELS]].to_numpy() < 0).any():
        raise ValueError("paired EIA-923 inputs contain negative fuel values")

    hourly = load_hourly(NATIVE, BASES)
    # Keep exactly the intersection certified by the independent BA-month audit.
    keys = set(zip(eligible.ba, eligible.month))
    hourly = hourly.loc[[key in keys for key in zip(hourly.region, hourly.month)]].copy()
    counts = hourly.groupby(["region", "month"]).size()
    expected = {(r.ba, r.month): int(r.eia930_expected_hours)
                for r in eligible.itertuples(index=False)}
    if any(int(counts.get(key, 0)) != expected[key] for key in keys):
        raise ValueError("hourly scoring rows do not match the independently audited complete months")

    # Source-only training/calibration and frozen candidate methods.
    _, _, models, calibration = source_model.fit_source()
    target_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    for record in eligible.itertuples(index=False):
        key = (record.ba, record.month)
        block = hourly.loc[hourly.region.eq(key[0]) & hourly.month.eq(key[1])].copy()
        observed_shares = to_coarse(pd.Series(record._asdict()), "eia923")
        oracle_shares = to_coarse(pd.Series(record._asdict()), "eia930")
        fuel_rows = block.rename(columns={f"generation_{fuel}": fuel for fuel in FUELS})
        y = labels(fuel_rows)
        for arm, shares in (("eia923_actual_release_mix", observed_shares),
                            ("eia930_same_month_oracle_mix", oracle_shares)):
            features = feature_rows(block, shares, key[0])
            pred = predict_models(models, features)
            for method, point in pred.items():
                bias, sigma = calibration[method]
                score = metrics(features, y, point, bias, sigma)
                target_rows.append({"ba": key[0], "month": key[1], "arm": arm,
                                    "method": method, **score,
                                    "eia923_to_eia930_total_ratio": float(record.total_ratio_eia923_to_eia930),
                                    "five_group_share_l1": float(record.mix_share_l1)})
                prediction_rows.append(pd.DataFrame({
                    "ba": key[0], "month": key[1], "arm": arm, "method": method,
                    "time_utc": block.time.astype(str),
                    "local_time": features.clock.astype(str), "y_fixed_ef_proxy": y,
                    "prediction_raw": point, "prediction_bias_corrected": point + bias,
                    "thermal": features.thermal, "hydro": features.hydro,
                    "nuclear": features.nuclear, "wind": features.wind, "solar": features.solar,
                }))

    detail = pd.DataFrame(target_rows)
    # Equal weight per BA-month prevents longer months dominating the macro comparison.
    macro = (detail.groupby(["arm", "method"], as_index=False)
             .agg(ba_months=("ba", "size"), hours=("hours", "sum"),
                  macro_mae=("mae", "mean"), macro_shape_mae=("shape_mae", "mean"),
                  macro_crps=("crps_gaussian", "mean"), macro_coverage90=("coverage90", "mean"),
                  mean_width90=("width90", "mean"), macro_interval_score90=("interval_score90", "mean")))
    pooled = []
    pred_frame = pd.concat(prediction_rows, ignore_index=True)
    for (arm, method), group in pred_frame.groupby(["arm", "method"], sort=True):
        bias, sigma = calibration[method]
        features = pd.DataFrame({"clock": pd.to_datetime(group.local_time)})
        score = metrics(features, group.y_fixed_ef_proxy.to_numpy(float),
                         group.prediction_raw.to_numpy(float), bias, sigma)
        daily = pd.DataFrame({"ba": group.ba.to_numpy(),
                              "day": features.clock.dt.strftime("%Y-%m-%d"),
                              "y": group.y_fixed_ef_proxy.to_numpy(float),
                              "p": group.prediction_bias_corrected.to_numpy(float)})
        centered = daily[["y", "p"]] - daily.groupby(["ba", "day"])[["y", "p"]].transform("mean")
        score["shape_mae"] = float(np.abs(centered.y - centered.p).mean())
        pooled.append({"arm": arm, "method": method, **score})

    OUT.mkdir(parents=True)
    detail.to_csv(OUT / "ba_month_method_metrics.csv", index=False)
    macro.to_csv(OUT / "macro_metrics.csv", index=False)
    pd.DataFrame(pooled).to_csv(OUT / "pooled_metrics.csv", index=False)
    pred_frame.to_csv(OUT / "hourly_predictions_and_labels.csv.gz", index=False, compression="gzip")
    provenance = {
        "status": "DEVELOPMENT_MATCHED_BA_MONTH_INFORMATION_VALUE_DIAGNOSTIC",
        "paired_ba_months": int(len(eligible)), "target_bas": sorted(eligible.ba.unique().tolist()),
        "target_months": sorted(eligible.month.unique().tolist()),
        "paired_cells_by_ba": {str(k): int(v) for k, v in eligible.groupby("ba").size().items()},
        "monthly_mix_alignment": {
            "median_total_ratio_eia923_to_eia930": float(
                (eligible.eia923_total_mwh / eligible.eia930_total_mwh).median()),
            "median_five_group_share_l1": float(eligible.mix_share_l1.median()),
            "mean_five_group_share_l1": float(eligible.mix_share_l1.mean()),
            "p90_five_group_share_l1": float(eligible.mix_share_l1.quantile(.9)),
        },
        "prior_saved_alignment_changed_ba_months": int(changed_rows.sum()),
        "prior_saved_alignment_changed_fields": changed_fields,
        "prior_alignment_source": str(ALIGNMENT.relative_to(ROOT)),
        "eia923_release": "2026 August monthly release, announced 2026-10-23",
        "release_after_evaluated_months": True,
        "eia923_input_arm": "actual published EIA-923 BA/month Netgen fuel mix; not as-issued prospective input",
        "oracle_arm": "same BA/month EIA-930 full-month fuel mix; target-period oracle and not deployable",
        "target_label": "current EIA-930 hourly fuel mix with fixed factors; not measured CIF",
        "training": "GB/NESO source model, 2022 fit and 2023 source-only residual calibration",
        "metrics_weighting": "macro table gives equal weight to each BA-month; pooled table gives equal weight to each valid hour",
        "runtime_versions": {"python": platform.python_version(), "pandas": pd.__version__,
                             "numpy": np.__version__, "scipy": scipy.__version__,
                             "scikit_learn": sklearn.__version__},
        "input_hashes": {str(path.relative_to(ROOT)): sha(path)
                         for path in [EIA923, ALIGNMENT, *NATIVE, *SOURCE_INPUTS]},
        "independent_audit_recomputed_from_raw_release_and_hourly_files": True,
        "code_hashes": {str(path.relative_to(ROOT)): sha(path) for path in CODE_INPUTS},
        "limitations": ["20 BA-months from five BAs only", "EIA-923 current release postdates evaluated months",
                        "EIA-923 covers reported plants rather than certified full BA generation",
                        "EIA-930 target label and oracle mix are current revised snapshots",
                        "source-calibrated intervals have no target-coverage guarantee"],
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(macro.to_string(index=False))
    print("\n", json.dumps({k: provenance[k] for k in
                          ("status", "paired_ba_months", "target_bas", "target_months", "limitations")}, indent=2))


if __name__ == "__main__":
    main()
