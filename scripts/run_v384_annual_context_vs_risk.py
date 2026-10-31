"""V384 source-only pilot: annual fuel context vs same-budget risk baselines."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import annual_context_v26 as annual
from scripts import monthly_transfer_v25 as monthly
from scripts import run_local_sketch_v17 as local
from scripts import run_monthly_transfer_v25 as v25
from scripts.native_center_v18 import scaled_paths
from transcif.research.weighted_metrics import weighted_quantile

OUT = ROOT / "results/v384r2-annual-context-vs-risk"
V25_OUT = ROOT / "results/monthly-transfer-v25"
ANNUAL = ROOT / "results/coarse-fields-v24/annual_bounds.csv"
PROTOCOL = ROOT / "docs/V384_ANNUAL_CONTEXT_VS_CONDITIONAL_RISK_PROTOCOL.md"
SOURCES = monthly.SOURCES
DAYS_2024 = pd.date_range("2024-01-01", "2024-12-31", tz="UTC")
DAYS_2025 = v25.DAYS
FACTORS = v25.FACTORS
ALPHA = 0.10
MODEL_CODE = (
    "scripts/run_v384_annual_context_vs_risk.py",
    "scripts/annual_context_v26.py", "scripts/monthly_transfer_v25.py",
    "scripts/run_monthly_transfer_v25.py", "scripts/run_local_sketch_v17.py",
    "scripts/run_information_value_v10.py", "scripts/native_center_v18.py",
    "src/transcif/research/fuel_transport_v7.py",
    "src/transcif/research/weighted_metrics.py",
    "docs/V384_ANNUAL_CONTEXT_VS_CONDITIONAL_RISK_PROTOCOL.md",
)
STAGES = ("prepare", "predict", "score")
ANN_ARMS = annual.ARMS
ENSEMBLE_ARMS = (*ANN_ARMS, "MG-recompose_conditional", "M8-recompose_conditional")


def sha(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy(order="K") for key in archive.files}


def environment() -> dict:
    versions = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "executable": str(Path(sys.executable).resolve()),
            "platform": sys.platform, "packages": versions,
            "thread_env": {key: os.environ.get(key) for key in
                           ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}


def v25_parent() -> tuple[dict, dict, dict]:
    freeze = read_json(V25_OUT / "freeze.json")
    seal = read_json(V25_OUT / "prediction_seal.json")
    completed = read_json(V25_OUT / "source_completed.json")
    if (freeze.get("version") != 25 or seal.get("complete_grid") is not True
            or completed.get("target_values_opened") is not False
            or seal.get("freeze_sha256") != sha(V25_OUT / "freeze.json")
            or completed.get("freeze_sha256") != sha(V25_OUT / "freeze.json")
            or completed.get("prediction_seal_sha256") != sha(V25_OUT / "prediction_seal.json")):
        raise ValueError("Verified V25 parent and complete prediction seal required")
    if tuple(seal.get("entries", [{}])[i].get("region") for i in range(len(SOURCES))) != SOURCES:
        raise ValueError("V25 seven-region prediction registry differs")
    for entry in seal["entries"]:
        if sha(entry["path"]) != entry["sha256"] or sha(entry["audit_path"]) != entry["audit_sha256"]:
            raise ValueError("V25 sealed prediction or audit hash mismatch")
    return freeze, seal, completed


def expected_contract() -> dict:
    parent, seal, completed = v25_parent()
    source = Path(parent["source"])
    inputs = {
        str(V25_OUT / "freeze.json"): sha(V25_OUT / "freeze.json"),
        str(V25_OUT / "prediction_seal.json"): sha(V25_OUT / "prediction_seal.json"),
        str(V25_OUT / "source_completed.json"): sha(V25_OUT / "source_completed.json"),
        str(ANNUAL): sha(ANNUAL), str(source / "monthly.csv"): sha(source / "monthly.csv"),
        str(PROTOCOL): sha(PROTOCOL),
    }
    for name in ("annual_bounds.csv", "freeze.json", "completed.json"):
        path = ROOT / "results/coarse-fields-v24" / name
        inputs[str(path)] = sha(path)
    for entry in seal["entries"]:
        inputs[entry["path"]] = entry["sha256"]
        inputs[entry["audit_path"]] = entry["audit_sha256"]
    for region in SOURCES:
        path = source / f"{region}_hourly.csv"
        inputs[str(path)] = sha(path)
    code = {str(ROOT / name): sha(ROOT / name) for name in MODEL_CODE}
    return {
        "version": 384, "sources": list(SOURCES), "target_days": [str(DAYS_2025[0]), str(DAYS_2025[-1])],
        "training_year": 2024, "training_months": [1, 6], "calibration_months": [7, 12],
        "factors": list(FACTORS), "annual_source": str(ANNUAL), "monthly_fuel_budget": "MG",
        "annual_fuel_groups": list(annual.FUELS), "annual_pos_groups": [annual.FUELS[i] for i in annual.POS],
        "risk_models": ["AY-CQR90", "AY-HET90"], "seed": 17023, "alpha": ALPHA,
        "code_hashes": code, "input_hashes": inputs, "environment": environment(),
        "v25_source_completed_sha256": sha(V25_OUT / "source_completed.json"),
        "v25_parent_target_values_already_open": True,
        "new_target_values_opened": False, "unseen_values_opened": False,
        "china_data_used_for_model_fit": False, "independent_confirmation": False,
    }


def check_freeze() -> dict:
    frozen = read_json(OUT / "freeze.json")
    expected = expected_contract()
    if set(frozen) != set(expected) | {"created_utc"} or any(frozen[key] != value for key, value in expected.items()):
        raise ValueError("V384 frozen input/code/contract closure changed")
    for path, digest in frozen["input_hashes"].items():
        if sha(path) != digest:
            raise ValueError("V384 input bytes changed: " + path)
    for path, digest in frozen["code_hashes"].items():
        if sha(path) != digest:
            raise ValueError("V384 producer bytes changed: " + path)
    return frozen


def parse_monthly_asof(path: Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Keep only source-region rows released by cutoff before parsing fuel values."""
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("region") not in SOURCES:
                continue
            period_end = pd.Timestamp(row["period_end"])
            available = pd.Timestamp(row["available_at"])
            if period_end.tzinfo is None or available.tzinfo is None:
                raise ValueError("Explicit timezone required for monthly rows")
            if period_end <= cutoff and available <= cutoff:
                rows.append({key: row[key] for key in
                             ("region", "period_start", "period_end", "available_at", *monthly.M8_COLUMNS)})
    frame = pd.DataFrame(rows)
    for key in ("period_start", "period_end", "available_at"):
        frame[key] = pd.to_datetime(frame[key], utc=True, errors="raise")
    if not frame.empty:
        frame[list(monthly.M8_COLUMNS)] = frame[list(monthly.M8_COLUMNS)].astype(float)
    return frame


def hourly_truth(path: Path, start: str, stop: str) -> dict[int, np.ndarray]:
    days = local.source_days(path, start, stop)
    result = {}
    for day, generation in days:
        shares = generation / generation.sum(axis=1, keepdims=True)
        cif = shares @ monthly.EF
        result[pd.Timestamp(day).value] = cif
    return result


def annual_feature(context: annual.AnnualContext) -> np.ndarray:
    if context.status != "available":
        return np.r_[np.zeros(4), 0.]
    return np.r_[np.asarray(context.ratio, float), 1.]


def day_feature(mix_draws: np.ndarray, day: pd.Timestamp, annual_rows: list[dict[str, str]], region: str) -> np.ndarray:
    mix_draws = np.asarray(mix_draws, float)
    if mix_draws.ndim != 2 or mix_draws.shape[1] != 5 or not np.isfinite(mix_draws).all():
        raise ValueError("MG daily forecast particles must be finite (samples,5)")
    mix_mean, mix_sd = mix_draws.mean(0), mix_draws.std(0)
    context = annual.select_annual_context(annual_rows, region, day)
    doy = day.dayofyear
    phase = 2 * np.pi * (doy - 1) / 365.25
    return np.r_[mix_mean, mix_sd, annual_feature(context), np.sin(phase), np.cos(phase)]


def hourly_design(day_features: np.ndarray) -> np.ndarray:
    x = np.asarray(day_features, float)
    if x.ndim != 2 or x.shape[1] != 17 or not np.isfinite(x).all():
        raise ValueError("Expected daily (N,17) MG/annual/calendar features")
    n = len(x)
    hour = np.tile(np.arange(24, dtype=float), n)
    h1, h2 = 2 * np.pi * hour / 24, 4 * np.pi * hour / 24
    repeated = np.repeat(x, 24, axis=0)
    return np.column_stack((repeated, np.sin(h1), np.cos(h1), np.sin(h2), np.cos(h2)))


def region_weights(regions: np.ndarray) -> np.ndarray:
    regions = np.asarray(regions)
    unique = np.unique(regions)
    if not len(unique):
        raise ValueError("At least one source region is required")
    counts = {region: np.count_nonzero(regions == region) for region in unique}
    return np.asarray([1 / (len(unique) * counts[region]) for region in regions], float)


def weighted_rank(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    return float(weighted_quantile(np.asarray(values, float), np.asarray(weights, float), q))


def train_risk_models(train_data: dict[str, dict], calibration_data: dict[str, dict], *, outer: str) -> dict:
    train_regions = [region for region in SOURCES if region != outer]
    x_parts, y_parts, region_parts, dates_parts = [], [], [], []
    cal_x_parts, cal_y_parts, cal_region_parts = [], [], []
    for region in train_regions:
        train = train_data[region]
        cal = calibration_data[region]
        x_parts.append(train["x"]); y_parts.append(train["y"])
        region_parts.append(np.repeat(region, len(train["y"])))
        dates_parts.append(train["dates"])
        cal_x_parts.append(cal["x"]); cal_y_parts.append(cal["y"])
        cal_region_parts.append(np.repeat(region, len(cal["y"])))
    x = np.concatenate(x_parts); y = np.concatenate(y_parts)
    regions = np.concatenate(region_parts); sample_weight = region_weights(regions)
    cx = np.concatenate(cal_x_parts); cy = np.concatenate(cal_y_parts)
    cregions = np.concatenate(cal_region_parts); cal_weight = region_weights(cregions)

    quantile_models = []
    for q in (.05, .95):
        model = GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=80,
            max_depth=2, min_samples_leaf=100, learning_rate=.05, random_state=17023)
        model.fit(x, y, sample_weight=sample_weight)
        quantile_models.append(model)
    cal_ends = np.sort(np.stack([model.predict(cx) for model in quantile_models]), axis=0)
    cqr_scores = np.maximum(cal_ends[0] - cy, cy - cal_ends[1])
    cqr_offset = weighted_rank(cqr_scores, cal_weight, .90)

    et_kwargs = dict(n_estimators=100, min_samples_leaf=50, max_features=.7,
                     n_jobs=-1, random_state=17023)
    mean_model = ExtraTreesRegressor(**et_kwargs).fit(x, y, sample_weight=sample_weight)
    oof_abs = np.full(len(y), np.nan)
    unique = list(dict.fromkeys(regions.tolist()))
    start = 0
    row_regions = regions
    for held in unique:
        keep_train = row_regions != held
        held_mask = row_regions == held
        if not held_mask.any() or not keep_train.any():
            raise ValueError("Region-crossfit residual support is incomplete")
        oof_model = ExtraTreesRegressor(**et_kwargs).fit(x[keep_train], y[keep_train],
            sample_weight=region_weights(row_regions[keep_train]))
        oof_abs[held_mask] = np.abs(y[held_mask] - oof_model.predict(x[held_mask]))
        start += 1
    if not np.isfinite(oof_abs).all():
        raise ValueError("Every training row requires a leave-one-region-out residual")
    scale_model = ExtraTreesRegressor(**{**et_kwargs, "random_state": 17024}).fit(
        x, oof_abs, sample_weight=sample_weight)
    cal_mean = mean_model.predict(cx)
    cal_scale = np.maximum(scale_model.predict(cx), 1e-3)
    hetero_score = np.abs(cy - cal_mean) / cal_scale
    hetero_multiplier = weighted_rank(hetero_score, cal_weight, .90)
    return dict(quantile_models=quantile_models, cqr_offset=cqr_offset,
        mean_model=mean_model, scale_model=scale_model, hetero_multiplier=hetero_multiplier,
        audit=dict(outer=outer, train_regions=train_regions, fit_year=2024,
            fit_period="Jan-Jun", calibration_period="Jul-Dec", fit_rows=int(len(y)),
            calibration_rows=int(len(cy)), equal_region_fit=True, equal_region_calibration=True,
            cqr_calibration_score="max(lower-y,y-upper)", cqr_offset=float(cqr_offset),
            heteroscedastic_scale="ExtraTrees on leave-one-source-region-out absolute residuals",
            heteroscedastic_multiplier=float(hetero_multiplier),
            target_hourly_labels_used=False, unseen_hourly_labels_used=False))


def train_data_2024(source: Path, monthly_frame: pd.DataFrame, annual_rows: list[dict[str, str]]) -> tuple[dict, dict]:
    visible = monthly.group_monthly(monthly_frame, "MG")
    truth_by_region, features_by_region = {}, {}
    for region in SOURCES:
        law = monthly.coarse_law(visible, region, DAYS_2024, budget="MG", samples=128, seed=17023)
        feats = np.stack([day_feature(law["draws"][i], day, annual_rows, region)
                          for i, day in enumerate(DAYS_2024)])
        truth = hourly_truth(source / f"{region}_hourly.csv", "2024-01-01", "2025-01-01")
        truth_by_region[region] = truth
        features_by_region[region] = feats
    fit_data, calibration_data = {}, {}
    for region in SOURCES:
        truth = truth_by_region[region]
        x_rows, y_rows, date_rows = [], [], []
        for i, day in enumerate(DAYS_2024):
            if day.value not in truth:
                continue
            x_rows.append(hourly_design(features_by_region[region][i:i+1]))
            y_rows.append(truth[day.value]); date_rows.extend([day] * 24)
        x = np.concatenate(x_rows); y = np.concatenate(y_rows); dates = pd.DatetimeIndex(date_rows)
        fit = dates.month <= 6; cal = dates.month >= 7
        if not fit.any() or not cal.any():
            raise ValueError("Each source requires 2024 proper and calibration hours")
        fit_data[region] = dict(x=x[fit], y=y[fit], dates=dates[fit])
        calibration_data[region] = dict(x=x[cal], y=y[cal], dates=dates[cal])
    return fit_data, calibration_data


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError("Preserve any existing or partial V384 run")
    contract = expected_contract()
    OUT.mkdir(parents=True)
    write_json(OUT / "freeze.json", {"created_utc": datetime.now(timezone.utc).isoformat(), **contract})
    print("Prepared V384", len(contract["input_hashes"]), "input hashes", flush=True)


def prediction_seal() -> dict:
    seal = read_json(OUT / "prediction_seal.json")
    if (seal.get("freeze_sha256") != sha(OUT / "freeze.json")
            or seal.get("complete_grid") is not True or seal.get("regions") != list(SOURCES)
            or seal.get("target_hourly_labels_used") is not False
            or len(seal.get("entries", [])) != len(SOURCES)):
        raise ValueError("Complete seven-region prediction seal required")
    for entry in seal["entries"]:
        if sha(entry["path"]) != entry["sha256"] or sha(entry["audit_path"]) != entry["audit_sha256"]:
            raise ValueError("V384 prediction artifact hash mismatch")
    return seal


def predict() -> None:
    frozen = check_freeze()
    if (OUT / "predictions").exists() or (OUT / "prediction_seal.json").exists():
        raise FileExistsError("Preserve partial V384 prediction stage")
    parent, pseal, _ = v25_parent()
    source = Path(parent["source"])
    monthly_frame = parse_monthly_asof(source / "monthly.csv", pd.Timestamp("2024-12-31T23:59:59Z"))
    annual_rows = pd.read_csv(ANNUAL, dtype=str, keep_default_na=False).to_dict("records")
    # 2024 source features and labels are training-only; cached once before leave-one-region fitting.
    fit_data, calibration_data = train_data_2024(source, monthly_frame, annual_rows)
    by_region = {entry["region"]: entry for entry in pseal["entries"]}
    bank_paths = parent["banks_by_outer"]
    prediction_dir = OUT / "predictions"
    prediction_dir.mkdir()
    entries = []
    for region in SOURCES:
        archived = load_npz(by_region[region]["path"])
        original_query = v25.load(parent["query_paths"][region])
        bank_files = bank_paths[region]
        bank = monthly.union_bank(v25.load(bank_files[0]), v25.load(bank_files[1]), original_query, outer=region)
        mg_index = monthly.ARMS.index("MG-recompose_conditional")
        baseline = {key: archived[key][mg_index] for key in
                    ("raw", "fine_draws", "donor_ids", "fallback_ids")}
        predicted = annual.predict(bank, baseline, archived["MG_draws"], DAYS_2025,
                                   annual_rows, region=region)
        risk = train_risk_models(fit_data, calibration_data, outer=region)
        # Use target coarse particles and legal annual context, never target hourly truth.
        day_features = np.stack([day_feature(archived["MG_draws"][i], day, annual_rows, region)
                                  for i, day in enumerate(DAYS_2025)])
        design = hourly_design(day_features)
        cqr_ends = np.sort(np.stack([model.predict(design) for model in risk["quantile_models"]]), axis=0)
        cqr_lower = cqr_ends[0] - risk["cqr_offset"]
        cqr_upper = cqr_ends[1] + risk["cqr_offset"]
        cross = cqr_lower > cqr_upper
        midpoint = (cqr_lower + cqr_upper) / 2
        cqr_lower = np.where(cross, midpoint, cqr_lower)
        cqr_upper = np.where(cross, midpoint, cqr_upper)
        hetero_point = risk["mean_model"].predict(design)
        hetero_scale = np.maximum(risk["scale_model"].predict(design), 1e-3)
        hetero_lower = hetero_point - risk["hetero_multiplier"] * hetero_scale
        hetero_upper = hetero_point + risk["hetero_multiplier"] * hetero_scale
        arrays = {
            "origins": DAYS_2025.asi8, "annual_arms": np.asarray(ANN_ARMS),
            "annual_raw": np.stack([predicted["arms"][arm]["raw"] for arm in ANN_ARMS]),
            "annual_ratios": np.stack([predicted["arms"][arm]["ratios"] for arm in ANN_ARMS]),
            "annual_unsupported": np.stack([predicted["arms"][arm]["unsupported_positive_energy_fraction"] for arm in ANN_ARMS]),
            "mg_raw": archived["raw"][monthly.ARMS.index("MG-recompose_conditional")],
            "m8_raw": archived["raw"][monthly.ARMS.index("M8-recompose_conditional")],
            "cqr_lower": np.clip(cqr_lower.reshape(len(DAYS_2025), 24), 0, 1000),
            "cqr_upper": np.clip(cqr_upper.reshape(len(DAYS_2025), 24), 0, 1000),
            "hetero_lower": np.clip(hetero_lower.reshape(len(DAYS_2025), 24), 0, 1000),
            "hetero_upper": np.clip(hetero_upper.reshape(len(DAYS_2025), 24), 0, 1000),
            "hetero_point": np.clip(hetero_point.reshape(len(DAYS_2025), 24), 0, 1000),
            "annual_target_year": predicted["target_year"],
            "target_context_missing": predicted["target_context_unavailable"],
        }
        if any(not np.isfinite(value).all() for value in arrays.values() if value.dtype.kind in "f"):
            raise ValueError("Nonfinite V384 prediction artifact")
        path = prediction_dir / f"{region}.npz"
        with path.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        audit_path = prediction_dir / f"{region}.json"
        write_json(audit_path, {"region": region, "training_regions": [r for r in SOURCES if r != region],
            "source_fit_year": 2024, "target_year": 2025, "annual_prediction_audit": predicted["audit"],
            "risk_model_audit": risk["audit"], "cqr_crossed_hour_fraction": float(cross.mean()),
            "target_hourly_labels_used": False, "unseen_hourly_labels_used": False,
            "feature_names": [*(f"mg_mean_{f}" for f in ("POS", "nuclear", "hydro", "wind", "solar")),
                *(f"mg_sd_{f}" for f in ("POS", "nuclear", "hydro", "wind", "solar")),
                *(f"annual_{f}" for f in ("coal", "gas", "oil", "other")), "annual_available",
                "day_sin", "day_cos", "hour_sin", "hour_cos", "hour2_sin", "hour2_cos"]})
        entries.append({"region": region, "path": str(path), "sha256": sha(path),
                        "audit_path": str(audit_path), "audit_sha256": sha(audit_path)})
        print("Predicted and sealed candidate artifacts for", region, flush=True)
    check_freeze()
    write_json(OUT / "prediction_seal.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_sha256": sha(OUT / "freeze.json"), "regions": list(SOURCES), "entries": entries,
        "complete_grid": True, "target_hourly_labels_used": False, "unseen_hourly_labels_used": False})


def interval_metrics(lower: np.ndarray, upper: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    lower, upper, truth = map(lambda value: np.asarray(value, float), (lower, upper, truth))
    if lower.shape != (24,) or upper.shape != (24,) or truth.shape != (24,) or np.any(lower > upper):
        raise ValueError("Invalid 24-hour interval arrays")
    width = upper - lower
    below, above = np.maximum(lower - truth, 0), np.maximum(truth - upper, 0)
    return {
        "coverage90": float(np.mean((truth >= lower) & (truth <= upper))),
        "width90": float(width.mean()),
        "interval_score90": float(np.mean(width + 20 * (below + above))),
        "lower_tail_miss": float(np.mean(truth < lower)),
        "upper_tail_miss": float(np.mean(truth > upper)),
        "midpoint_mae": float(np.mean(np.abs((lower + upper) / 2 - truth))),
    }


def score() -> None:
    if (OUT / "completed.json").exists():
        raise FileExistsError("Preserve completed V384 score stage")
    frozen = check_freeze(); seal = prediction_seal()
    parent, _, _ = v25_parent()
    source = Path(parent["source"])
    ensemble_rows, interval_rows = [], []
    for entry in seal["entries"]:
        region = entry["region"]
        archive = load_npz(entry["path"])
        truth = hourly_truth(source / f"{region}_hourly.csv", "2025-01-01", "2025-07-01")
        for index, day in enumerate(DAYS_2025):
            if day.value not in truth:
                continue
            y = truth[day.value]
            candidates = {arm: archive["annual_raw"][i, index] for i, arm in enumerate(ANN_ARMS)}
            candidates["MG-recompose_conditional"] = archive["mg_raw"][index]
            candidates["M8-recompose_conditional"] = archive["m8_raw"][index]
            for arm, raw in candidates.items():
                for factor in FACTORS:
                    point, paths, _ = scaled_paths(raw[None, ...], factor)
                    metrics = local.score_arrays(point, paths, y[None, :])
                    ensemble_rows.append({"region": region, "origin": day.isoformat(), "method": arm,
                        "factor": factor, **{key: float(value[0]) for key, value in metrics.items()}})
            for method, lower_key, upper_key in (
                ("AY-CQR90", "cqr_lower", "cqr_upper"), ("AY-HET90", "hetero_lower", "hetero_upper")):
                metrics = interval_metrics(archive[lower_key][index], archive[upper_key][index], y)
                interval_rows.append({"region": region, "origin": day.isoformat(), "method": method, **metrics})
        print("Scored sealed prediction archive for", region, flush=True)

    ensemble = pd.DataFrame(ensemble_rows)
    intervals = pd.DataFrame(interval_rows)
    if ensemble.empty or intervals.empty or ensemble.duplicated(["region", "origin", "method", "factor"]).any():
        raise ValueError("Missing or duplicated V384 score rows")
    e_region = ensemble.groupby(["region", "method", "factor"], as_index=False).mean(numeric_only=True)
    e_macro = e_region.groupby(["method", "factor"], as_index=False).mean(numeric_only=True)
    worst = e_region.groupby(["method", "factor"])["empirical_coverage90"].min().rename("worst_region_coverage90")
    e_macro = e_macro.merge(worst, on=["method", "factor"])
    e_macro["qualified"] = e_macro.empirical_coverage90.between(.90, .95) & e_macro.worst_region_coverage90.ge(.85)
    e_macro["violation"] = ((.90-e_macro.empirical_coverage90).clip(lower=0)
        + (e_macro.empirical_coverage90-.95).clip(lower=0) + (.85-e_macro.worst_region_coverage90).clip(lower=0))
    choices = []
    for method, group in e_macro.groupby("method"):
        passed = group[group.qualified]
        selected = (passed.sort_values(["empirical_interval_score90", "factor"]) if not passed.empty
                    else group.sort_values(["violation", "empirical_interval_score90", "factor"])).iloc[0]
        choices.append({"method": method, "selected_factor": float(selected.factor),
            "qualified": bool(selected.qualified), "crps": float(selected.crps),
            "coverage90": float(selected.empirical_coverage90), "worst_region_coverage90": float(selected.worst_region_coverage90),
            "width90": float(selected.empirical_width90), "interval_score90": float(selected.empirical_interval_score90),
            "selection_source": "opened 2025 development cohort; not blind"})
    i_region = intervals.groupby(["region", "method"], as_index=False).mean(numeric_only=True)
    i_macro = i_region.groupby("method", as_index=False).mean(numeric_only=True)
    i_worst = i_region.groupby("method")["coverage90"].min().rename("worst_region_coverage90")
    i_macro = i_macro.merge(i_worst, on="method")
    i_macro["qualified"] = i_macro.coverage90.between(.90, .95) & i_macro.worst_region_coverage90.ge(.85)
    for row in i_macro.to_dict("records"):
        choices.append({"method": row["method"], "selected_factor": 1., "qualified": bool(row["qualified"]),
            "crps": None, "coverage90": float(row["coverage90"]),
            "worst_region_coverage90": float(row["worst_region_coverage90"]),
            "width90": float(row["width90"]), "interval_score90": float(row["interval_score90"]),
            "selection_source": "source-trained calibration; no target tuning"})
    OUT.mkdir(exist_ok=True)
    ensemble.to_csv(OUT / "ensemble_day_metrics.csv", index=False, mode="x")
    intervals.to_csv(OUT / "interval_day_metrics.csv", index=False, mode="x")
    e_region.to_csv(OUT / "ensemble_region_metrics.csv", index=False, mode="x")
    e_macro.to_csv(OUT / "ensemble_macro_metrics.csv", index=False, mode="x")
    i_region.to_csv(OUT / "interval_region_metrics.csv", index=False, mode="x")
    i_macro.to_csv(OUT / "interval_macro_metrics.csv", index=False, mode="x")
    write_json(OUT / "method_choices.json", {"choices": choices,
        "interpretation": "Development selection on previously opened 2025 source cohort; not an independent confirmation."})
    check_freeze(); prediction_seal()
    outputs = ["ensemble_day_metrics.csv", "interval_day_metrics.csv", "ensemble_region_metrics.csv",
        "ensemble_macro_metrics.csv", "interval_region_metrics.csv", "interval_macro_metrics.csv", "method_choices.json"]
    write_json(OUT / "completed.json", {"version": 384, "created_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_sha256": sha(OUT / "freeze.json"), "prediction_seal_sha256": sha(OUT / "prediction_seal.json"),
        "output_hashes": {name: sha(OUT / name) for name in outputs},
        "target_values_opened": True, "unseen_values_opened": False,
        "independent_confirmation": False, "algorithm_breakthrough": False,
        "annual_fields_verified_as_china_public_source": False})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=STAGES)
    stage = parser.parse_args().stage
    {"prepare": prepare, "predict": predict, "score": score}[stage]()


if __name__ == "__main__":
    main()
