"""Source-region cross-fit risk-radius diagnostic for bidirectional M and M+F centers."""
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

import run_cross_country_fine_fuel_bidirectional as main
from transcif.research import cross_country as cc

OUT = ROOT / "results/cross-country-fine-fuel-risk-r1"
PROTOCOL = ROOT / "docs/CROSS_COUNTRY_FINE_FUEL_RISK_CALIBRATION_PROTOCOL_20261101.md"
PREDICTIONS = ROOT / "results/cross-country-fine-fuel-bidirectional-r1/hourly_predictions.csv.gz"
PRIOR_PROVENANCE = ROOT / "results/cross-country-fine-fuel-bidirectional-r1/provenance.json"
METHODS = ("M5_coarse_recomposition", "MplusF_fuel_profile")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score_interval(y: np.ndarray, point: np.ndarray, bias: float,
                   radius: float) -> dict[str, float]:
    error = y - (point + bias)
    covered = np.abs(error) <= radius
    interval_score = (2 * radius + 20 * np.maximum(np.abs(error) - radius, 0)).mean()
    return {"hours": int(len(y)), "mae": float(np.abs(error).mean()),
            "coverage90": float(covered.mean()), "width90": float(2 * radius),
            "interval_score90": float(interval_score), "bias": bias,
            "radius90": radius}


def main_run() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    if not all(path.is_file() for path in [PROTOCOL, PREDICTIONS, PRIOR_PROVENANCE,
                                           *main.INPUTS, *main.CODE_INPUTS]):
        raise FileNotFoundError("frozen protocol, prior output, code, or inputs are missing")

    raw, load_audit = cc.load_inputs(ROOT)
    source_raw = pd.concat([raw[r].assign(_origin_region=r)
                            for r in raw if r != "GB"], ignore_index=True)
    source_raw["region"] = source_raw._origin_region.astype(str)
    features = main.combined_features(source_raw)
    year = pd.to_datetime(features.clock).dt.year
    valid = main.supported(features)
    residual_rows: list[dict] = []
    folds = []
    for heldout in main.US_REGIONS:
        train_mask = valid & year.eq(2022).to_numpy() & source_raw.region.ne(heldout).to_numpy()
        test_mask = valid & year.eq(2023).to_numpy() & source_raw.region.eq(heldout).to_numpy()
        if train_mask.sum() < 1000 or test_mask.sum() < 1000:
            raise ValueError(f"incomplete source cross-fit fold: {heldout}")
        train = features.loc[train_mask].copy()
        train_fuels = source_raw.loc[train_mask, list(cc.FUELS)].to_numpy(float)
        models = cc.fit_models(train, train_fuels)
        profile, support = main.source_fine_profiles(source_raw, train_mask)
        heldout_features = features.loc[test_mask].copy()
        y = cc.labels(source_raw.loc[test_mask])
        points = main.predict_all(models, profile, heldout_features)
        for method in METHODS:
            residual = y - points[method]
            residual_rows.extend({"heldout_region": heldout, "method": method,
                                  "time": str(time), "residual": float(value)}
                                 for time, value in zip(heldout_features.time, residual))
        folds.append({"heldout_region": heldout, "train_hours": int(train_mask.sum()),
                      "calibration_hours": int(test_mask.sum()),
                      "source_profile_support": support})

    residuals = pd.DataFrame(residual_rows)
    risk = {}
    for method, group in residuals.groupby("method", sort=True):
        values = group.residual.to_numpy(float)
        bias = float(values.mean())
        radius = float(np.quantile(np.abs(values - bias), .90, method="higher"))
        risk[method] = {"bias": bias, "radius90": radius,
                        "hours": int(len(values)),
                        "heldout_regions": int(group.heldout_region.nunique())}

    target = pd.read_csv(PREDICTIONS)
    target = target.loc[target.direction.eq("US_to_GB")
                        & target.method.isin(METHODS)
                        & target.month.isin(main.COMMON_MONTHS)].copy()
    if target.empty:
        raise ValueError("frozen US-to-GB target predictions are empty")
    results = []
    for method, group in target.groupby("method", sort=True):
        spec = risk[method]
        score = score_interval(group.label.to_numpy(float), group.prediction_raw.to_numpy(float),
                               spec["bias"], spec["radius90"])
        results.append({"method": method, "calibration": "US_leave_BA_out_2023_abs_q90",
                        **score})
        prior_method = method
        prior = target.loc[target.method.eq(prior_method)]
        prior_spec = None
        # Recover the original source-only Gaussian radius from the sealed run record.
        for direction in json.loads(PRIOR_PROVENANCE.read_text())["direction_metadata"]:
            if direction["direction"] == "US_to_GB":
                prior_spec = direction["calibration"][method]
                break
        if prior_spec is None:
            raise ValueError(f"missing prior source calibration for {method}")
        old = score_interval(prior.label.to_numpy(float), prior.prediction_raw.to_numpy(float),
                             float(prior_spec["bias"]), 1.6448536269514722 * float(prior_spec["sigma"]))
        results.append({"method": method, "calibration": "source_internal_Gaussian_2023",
                        **old})

    OUT.mkdir(parents=True)
    residuals.to_csv(OUT / "source_lobo_residuals.csv.gz", index=False, compression="gzip")
    pd.DataFrame(results).to_csv(OUT / "target_gb_interval_metrics.csv", index=False)
    summary = {
        "status": "POSTHOC_SOURCE_CROSSFIT_RISK_DIAGNOSTIC_NOT_COVERAGE_GUARANTEE",
        "source_target_direction": "US_to_GB",
        "source_train_year": 2022, "source_leave_ba_out_calibration_year": 2023,
        "target_scored_from_frozen_output": "GB 2024-02/04/06, 2160 hours",
        "risk_calibration": risk, "folds": folds,
        "target_interval_metrics": results, "load_audit": load_audit,
        "input_hashes": {str(path.relative_to(ROOT)): sha(path) for path in main.INPUTS},
        "code_hashes": {str(path.relative_to(ROOT)): sha(path)
                        for path in [Path(__file__).resolve(), *main.CODE_INPUTS]},
        "prior_prediction_hash": sha(PREDICTIONS),
        "prior_provenance_hash": sha(PRIOR_PROVENANCE),
        "limitations": ["2024 GB target response was previously viewed",
                        "one target country and three months only",
                        "cross-BA residual exchangeability is not established",
                        "hourly dependence means marginal row quantile is not a region-level guarantee",
                        "GB-to-US has only one GB source region and cannot estimate cross-region source risk"],
    }
    (OUT / "provenance.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(pd.DataFrame(results).to_string(index=False))
    print("source_crossfit_radii", json.dumps(risk, sort_keys=True))


if __name__ == "__main__":
    main_run()
