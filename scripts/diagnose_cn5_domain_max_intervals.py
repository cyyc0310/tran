"""Diagnostic cost of country-maximum residual calibration on a seen FR target."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import run_cn5_fuel_resolved_fr_confirmation as confirmation  # noqa: E402
import run_cn5_fuel_resolved_response as candidate  # noqa: E402
import run_energy_charts_nine_country_cn5_loco as nine  # noqa: E402
from transcif.research import multicountry_cn5 as model  # noqa: E402
from transcif.research.monthly_adaptation import weighted_quantile  # noqa: E402

PROTOCOL = ROOT / "docs/CN5_DOMAIN_MAX_INTERVAL_DIAGNOSTIC_PROTOCOL_20261102.md"
FR_RESULT = ROOT / "results/cn5-fuel-resolved-fr-confirmation-r1"
METHODS = ("ratio_recomposition", "factor_history", "fuelshare_history")
ALPHA = .10


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_maxima(prepared: dict, sources: list[str]) -> tuple[dict, list[dict]]:
    residuals: dict[str, list[np.ndarray]] = {method: [] for method in METHODS}
    maxima: list[dict] = []
    for pseudo_target in sources:
        training_sources = [code for code in sources if code != pseudo_target]
        base, fuelshare = confirmation.fit_source_models(prepared, training_sources)
        _, features, _ = prepared[pseudo_target]
        query = features.loc[
            features.year.eq(2023) & features.clock.dt.month.ge(7)
        ].copy()
        if query.empty:
            raise ValueError(f"missing H2 pseudo-target rows: {pseudo_target}")
        truth = confirmation.truth_for(prepared[pseudo_target][0], query)
        predictions = confirmation.point_predictions(base, fuelshare, query)
        for method in METHODS:
            absolute = np.abs(truth - predictions[method])
            if not np.isfinite(absolute).all():
                raise ValueError(f"nonfinite residuals: {pseudo_target}/{method}")
            residuals[method].append(absolute)
            maxima.append({"country": pseudo_target.upper(), "method": method,
                           "hours": len(absolute), "maximum_absolute_residual": float(absolute.max()),
                           "q90_absolute_residual": float(np.quantile(absolute, .9))})
    radii = {method: float(max(row["maximum_absolute_residual"] for row in maxima
                              if row["method"] == method))
             for method in METHODS}
    pooled = {}
    for method in METHODS:
        values = np.concatenate(residuals[method])
        weights = np.concatenate([
            model.weights(prepared[country][1].loc[
                prepared[country][1].year.eq(2023)
                & prepared[country][1].clock.dt.month.ge(7)
            ]) for country in sources
        ])
        pooled[method] = float(weighted_quantile(values, weights, 1 - ALPHA))
    return {"domain_max": radii, "hour_pooled_q90": pooled}, maxima


def interval_score(y: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    lower, upper = center - radius, center + radius
    width = upper - lower
    return width + (2 / ALPHA) * np.maximum(lower - y, 0.) + (2 / ALPHA) * np.maximum(y - upper, 0.)


def score(y: np.ndarray, center: np.ndarray, radius: float) -> dict:
    lower, upper = center - radius, center + radius
    return {"hours": int(len(y)), "mae": float(np.mean(np.abs(y - center))),
            "coverage90_hourly": float(np.mean((y >= lower) & (y <= upper))),
            "full_width": float(2 * radius),
            "winkler90_hourly_mean": float(np.mean(interval_score(y, center, radius))),
            "whole_domain_all_hours_covered": bool(np.all((y >= lower) & (y <= upper)))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    sources = list(nine.COUNTRIES)
    prepared = {country: nine.prepare_country(country) for country in sources}
    prepared[confirmation.TARGET] = confirmation.prepare_fr()
    target_raw, target_features, _ = prepared[confirmation.TARGET]
    query = target_features.loc[
        target_features.year.eq(2024) & target_features.clock.dt.month.between(3, 12)
    ].copy()
    if len(query) != 7344 or set(query.clock.dt.month) != set(range(3, 13)):
        raise ValueError("FR diagnostic cohort differs from frozen 2024-03:12 cohort")
    truth = confirmation.truth_for(target_raw, query)
    stamps = target_raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy()

    prediction_path = FR_RESULT / "FR_heldout_predictions.csv.gz"
    prediction_seal_path = FR_RESULT / "prediction_seal.json"
    prediction_seal = json.loads(prediction_seal_path.read_text(encoding="utf-8"))
    if sha(prediction_path) != prediction_seal["prediction_sha256"]:
        raise ValueError("frozen FR prediction SHA mismatch")
    sealed = pd.read_csv(prediction_path)
    if not np.array_equal(sealed.time.to_numpy(), stamps):
        raise ValueError("frozen FR prediction timestamps do not align")

    radii, maxima = source_maxima(prepared, sources)
    source_rows = []
    metrics_rows = []
    for method in METHODS:
        center = sealed[method].to_numpy(float)
        source_rows.append({"method": method,
                            "domain_max_radius": radii["domain_max"][method],
                            "hour_pooled_q90_radius": radii["hour_pooled_q90"][method]})
        for kind in ("hour_pooled_q90", "domain_max"):
            result = score(truth, center, radii[kind][method])
            result.update({"method": method, "calibration": kind,
                           "radius": radii[kind][method]})
            metrics_rows.append(result)
    maxima_frame = pd.DataFrame(maxima)
    radius_frame = pd.DataFrame(source_rows)
    metrics_frame = pd.DataFrame(metrics_rows)
    maxima_frame.to_csv(output / "source_country_maxima.csv", index=False)
    radius_frame.to_csv(output / "calibration_radii.csv", index=False)
    metrics_frame.to_csv(output / "fr_interval_metrics.csv", index=False)

    input_paths = [PROTOCOL, prediction_path, prediction_seal_path,
                   FR_RESULT / "startup_hashes.json", FR_RESULT / "preflight_audit.json"]
    for country in sources:
        folder, manifest_path, _ = nine.six_group.data_locations(country)
        input_paths.extend([manifest_path, folder / f"{country}_2022_2024.json"])
        if country in nine.six_group.OLD_COUNTRIES:
            input_paths.append(folder / "audit.json")
    code_paths = [Path(__file__).resolve(), Path(confirmation.__file__).resolve(),
                  Path(candidate.__file__).resolve(), Path(nine.__file__).resolve(),
                  Path(nine.six_group.base.__file__).resolve(), Path(model.__file__).resolve()]
    summary = {"analysis": "country-maximum absolute residual interval diagnostic",
               "source_domains": [country.upper() for country in sources],
               "target": "FR 2024-03:12; previously scored; not blind confirmation",
               "target_hours": int(len(query)), "methods": list(METHODS), "alpha": ALPHA,
               "source_group_count": len(sources),
               "domain_max_rule": "maximum over each source pseudo-target's H2 maximum absolute residual",
               "limitations": [
                   "This is a post-hoc method diagnostic on an already opened target.",
                   "The maximum radius is not a validated distribution-free target-domain guarantee.",
                   "Country-level exchangeability, one target domain and label mapping differences remain unresolved.",
                   "Source pseudo-target models train on eight domains while the sealed FR centers train on nine."
               ],
               "prediction_sha256": sha(prediction_path),
               "input_sha256": {str(path.resolve()): sha(path) for path in sorted(set(input_paths))},
               "code_sha256": {str(path.resolve()): sha(path) for path in sorted(set(code_paths))}}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
    print(radius_frame.to_string(index=False))
    print(metrics_frame.to_string(index=False))


if __name__ == "__main__":
    main()
