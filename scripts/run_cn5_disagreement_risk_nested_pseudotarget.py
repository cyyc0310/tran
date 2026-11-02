#!/usr/bin/env python3
"""Strict double-held-out test for CN5 disagreement-conditioned risk bands."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import run_cn5_disagreement_risk_pseudotarget as base  # noqa: E402
import run_energy_charts_nine_country_cn5_loco as nine  # noqa: E402
import run_cn5_fuel_resolved_response as response  # noqa: E402
from transcif.research import multicountry_cn5 as model  # noqa: E402

COUNTRIES = nine.COUNTRIES
PROTOCOL = ROOT / "docs/CN5_DISAGREEMENT_RISK_NESTED_PROTOCOL_20261101.md"
SIGNALS = ("factor_ratio_disagreement", "three_model_spread")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pseudo_prediction(prepared: dict, target: str, train_countries: list[str]) -> pd.DataFrame:
    fitted, fuel_head = base.source_fit(prepared, train_countries)
    raw, features = prepared[target]
    query = features.loc[features.year.eq(2023) & features.clock.dt.month.gt(6)].copy()
    fuel = nine.fuel_for(raw, query)
    truth = fuel @ model.FACTORS / fuel.sum(axis=1)
    predictions = model.predict(fitted, query)
    predictions["fuelshare_history"] = response.predict_fuelshare(fuel_head, fitted, query)
    ratio = predictions["ratio_recomposition"]
    factor = predictions["factor_history"]
    fuelshare = predictions["fuelshare_history"]
    return pd.DataFrame({"country": target.upper(), "truth": truth, "ratio": ratio,
        "factor_ratio_disagreement": np.abs(factor-ratio),
        "three_model_spread": np.maximum.reduce([ratio, factor, fuelshare])-
                             np.minimum.reduce([ratio, factor, fuelshare]),
        "time": raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy()})


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    input_paths = [PROTOCOL, base.PROTOCOL]
    for country in COUNTRIES:
        folder, manifest, _ = nine.six_group.data_locations(country)
        input_paths.extend([manifest, folder / f"{country}_2022_2024.json"])
        if country in nine.six_group.OLD_COUNTRIES:
            input_paths.append(folder / "audit.json")
    code_paths = [Path(__file__).resolve(), Path(base.__file__).resolve(),
        Path(response.__file__).resolve(), Path(nine.__file__).resolve(),
        Path(nine.six_group.__file__).resolve(), Path(nine.six_group.base.__file__).resolve(),
        Path(model.__file__).resolve(), ROOT / "src/transcif/research/cross_country.py",
        ROOT / "src/transcif/research/monthly_adaptation.py"]
    input_hashes = {str(p.resolve()): sha(p) for p in sorted(set(input_paths))}
    code_hashes = {str(p.resolve()): sha(p) for p in code_paths}

    prepared, audits = {}, {}
    for country in COUNTRIES:
        raw, features, audits[country] = nine.prepare_country(country)
        prepared[country] = (raw, features)

    metric_rows, prediction_frames, fold_audit = [], [], []
    for held in COUNTRIES:
        held_sources = [c for c in COUNTRIES if c != held]
        test = pseudo_prediction(prepared, held, held_sources)
        calibration_parts = []
        for pseudo in held_sources:
            nested_sources = [c for c in held_sources if c != pseudo]
            if held in nested_sources or pseudo in nested_sources or len(nested_sources) != 7:
                raise RuntimeError("double-held-out source boundary violated")
            calibration_parts.append(pseudo_prediction(prepared, pseudo, nested_sources))
        calibration = pd.concat(calibration_parts, ignore_index=True)
        global_radius = float(model_weighted_q90(calibration))
        metric_rows.append({"heldout_country": held.upper(), "method": "global_radius",
                            **base.metrics(test, np.full(len(test), global_radius))})
        prediction = test.copy()
        prediction["global_radius"] = global_radius
        for signal in SIGNALS:
            radii, details = base.calibrate(calibration, test, signal)
            method = f"binned_{signal}"
            metric_rows.append({"heldout_country": held.upper(), "method": method,
                                **base.metrics(test, radii)})
            prediction[method] = radii
            fold_audit.append({"outer_target": held.upper(), "method": method,
                "outer_target_excluded_from_all_calibration_models": True,
                "calibration_countries": [c.upper() for c in held_sources],
                "training_countries_per_calibration_model": 7, "details": details})
        prediction_frames.append(prediction)
        fold_audit.append({"outer_target": held.upper(), "method": "global_radius",
            "outer_target_excluded_from_all_calibration_models": True,
            "calibration_countries": [c.upper() for c in held_sources],
            "training_countries_per_calibration_model": 7,
            "global_radius": global_radius})
        print(f"double-held-out outer pseudo target {held.upper()} complete", flush=True)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out / "heldout_pseudotarget_metrics.csv", index=False)
    macro = metrics.groupby("method", as_index=False).agg(
        macro_mae=("mae", "mean"), macro_coverage90=("coverage90", "mean"),
        macro_width90=("width90", "mean"), macro_interval_score90=("interval_score90", "mean"),
        minimum_country_coverage90=("coverage90", "min"))
    macro.to_csv(out / "macro_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        out / "double_heldout_pseudotarget_predictions.csv.gz", index=False, compression="gzip")
    (out / "fold_audit.json").write_text(json.dumps(fold_audit, ensure_ascii=False,
        indent=2, default=float)+"\n", encoding="utf-8")
    audit = {"status": "STRICT_DOUBLE_HELDOUT_SOURCE_ONLY_CN5_RISK_TEST",
        "countries": [c.upper() for c in COUNTRIES], "outer_train_year": 2022,
        "pseudo_target_period": "2023H2", "risk_calibration": "double leave-country-out",
        "target_inputs": "strict CN5 t-2 through t-7 monthly shares and calendar",
        "label": "fixed-factor generation-side proxy; not measured CIF",
        "outer_pseudo_target_hourly_labels_used_by_any_calibrator_model": False,
        "calibration_training_countries_per_pseudotarget": 7,
        "conditional_risk_bins": base.BINS,
        "prediction_sha256": sha(out / "double_heldout_pseudotarget_predictions.csv.gz"),
        "input_sha256": input_hashes, "code_sha256": code_hashes,
        "country_audits": audits}
    (out / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n",
                                    encoding="utf-8")
    print(macro.to_string(index=False), flush=True)
    if any(sha(Path(p)) != digest for p, digest in input_hashes.items()):
        raise RuntimeError("input changed during experiment")
    if any(sha(Path(p)) != digest for p, digest in code_hashes.items()):
        raise RuntimeError("code changed during experiment")


def model_weighted_q90(frame: pd.DataFrame) -> float:
    residual = np.abs(frame.truth.to_numpy(float)-frame.ratio.to_numpy(float))
    weights = base.equal_country_weights(frame)
    return float(base.weighted_quantile(residual, weights, .9))


if __name__ == "__main__":
    main()
