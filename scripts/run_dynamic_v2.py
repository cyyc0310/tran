"""Frozen exploratory 2023 leave-region-out validation; never uses 2024 labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from transcif.research.dynamic_v2 import (EF, METHODS, daily_profiles, fit_model, read_year)
from transcif.research.eia_proxy import REGIONS
from transcif.research.weighted_metrics import PreparedDistribution


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shifted_target_history(monthly, target, extra_days):
    """Delay only held-out-region information; never advance release metadata."""
    if extra_days < 0:
        raise ValueError("Release stress must not reveal an earlier snapshot")
    result = monthly.copy(deep=True)
    available = pd.to_datetime(result.available_at, utc=True)
    available.loc[result.region == target] += pd.Timedelta(days=extra_days)
    result["available_at"] = available
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("results/research-prefix/data"))
    parser.add_argument("--output", type=Path, default=Path("results/dynamic-v2-20261023"))
    parser.add_argument("--regions", nargs="+", default=list(REGIONS))
    parser.add_argument("--extra-release-days", type=int, default=0)
    args = parser.parse_args()
    if args.extra_release_days < 0:
        raise ValueError("Release stress cannot expose earlier prefix information")
    if args.output.exists():
        raise FileExistsError("Immutable run: choose a fresh output path")
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    protocol = {
        "status": "exploratory source validation; not untouched confirmation",
        "outer_validation_year": 2023, "source_train_year": 2022,
        "sources": "all other six regions; no held-out region hourly training",
        "calibration": "fit source January–June 2022, residuals source July–December 2022; same 128 day indices for every method",
        "variants": list(METHODS), "ridge_penalty": 30, "ef": EF.tolist(),
        "transition": "source month-lag coefficient correlation; tau clipped [14,365] days; global median control",
        "calendar": "source UTC month x 24-hour daily profile; no target hourly inputs",
        "quarterly": "three contiguous complete months, MWh-weighted shares, max monthly release; annual unavailable in 2022 training, not fitted",
        "release": "source training/calibration month-end +30 days; target month-end +30+extra days; actual revision vintages unavailable",
        "target_extra_release_days": args.extra_release_days,
        "input_intervention": "target information latency stress; source model unchanged; exploratory, not a theorem",
        "residuals": "retain source temporal bias; shared sampling indices; clip all samples to [0,1000]",
        "hashes": {name: sha(root / name) for name in (
            "src/transcif/research/dynamic_v2.py", "scripts/run_dynamic_v2.py", "docs/DYNAMIC_V2_PROTOCOL.md",
            "src/transcif/research/eia_proxy.py", "src/transcif/research/references.py",
            "src/transcif/research/mechanisms.py", "src/transcif/research/weighted_metrics.py")},
        "data_hashes": {path.name: sha(path) for path in sorted(args.data.glob("*.csv"))},
        "python": platform.python_version(), "seed": 13017,
        "no_tuning": "No changes after 2023 metrics; historical 2024 development results excluded",
    }
    (args.output / "run_dynamic_v2_frozen.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    monthly = pd.read_csv(args.data / "monthly.csv")
    monthly = monthly.loc[monthly.period_start.str.startswith(("2022-", "2023-"))].copy()
    frames = {region: read_year(args.data / f"{region}_hourly.csv", 2022) for region in REGIONS}
    origins = pd.date_range("2023-01-01", "2023-12-31", tz="UTC")
    rows, metadata = [], {}
    for target in args.regions:
        if target not in REGIONS:
            raise ValueError(target)
        sources = {name: frame for name, frame in frames.items() if name != target}
        calibration_model = fit_model(sources, monthly, "2022-07-01T00:00:00Z")
        errors = {method: [] for method in METHODS}
        for region, frame in sources.items():
            days, truth = daily_profiles(frame)
            take = days >= pd.Timestamp("2022-07-01", tz="UTC")
            days, truth = days[take], truth[take]
            predictions = calibration_model.predict(days, region, monthly)
            for method in METHODS:
                errors[method].extend(truth - predictions[method])
        size = len(errors[METHODS[0]])
        indices = np.random.default_rng(13017).choice(size, 128, replace=size < 128)
        banks = {method: np.asarray(values)[indices] for method, values in errors.items()}
        model = fit_model(sources, monthly)
        predictions = model.predict(origins, target, shifted_target_history(monthly, target, args.extra_release_days))
        folder = args.output / target
        folder.mkdir()
        np.savez_compressed(folder / "predictions.npz", **predictions,
                            **{f"residual_{key}": value for key, value in banks.items()},
                            origins=origins.asi8)
        seal = {"target": target, "source_regions": sorted(sources),
                "predictions_sha256": sha(folder / "predictions.npz"),
                "target_validation_year_hourly_read_before_seal": False,
                "target_extra_release_days": args.extra_release_days,
                "calibration_day_count": size, "calibration_indices": indices.tolist(),
                "tau": {key: value.tolist() for key, value in model.tau.items()},
                "methods_without_enough_support": [key for key in METHODS[3:] if key not in model.fits]}
        (folder / "prediction_seal.json").write_text(json.dumps(seal, indent=2))
        # Target validation hourly labels first opened only after forecasts seal.
        target_frame = read_year(args.data / f"{target}_hourly.csv", 2023)
        days, truth = daily_profiles(target_frame)
        for day, actual in zip(days, truth):
            i = origins.get_indexer([day])[0]
            if i < 0:
                continue
            for method in METHODS:
                point = predictions[method][i]
                ensemble = np.clip(point + banks[method], 0, EF.max())
                metric = PreparedDistribution(ensemble, np.ones(128)).score(actual)
                metric.update(region=target, origin=str(day), method=method,
                              raw_point_mae=float(np.abs(point - actual).mean()),
                              raw_shape_mae=float(np.abs(point - point.mean() - actual + actual.mean()).mean()),
                              raw_level_mae=float(abs(point.mean() - actual.mean())))
                rows.append(metric)
        metadata[target] = {"scored_days": len(days), "calibration_days": size}
        print(f"{target}: sealed then scored {len(days)} validation days", flush=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(args.output / "day_metrics.csv", index=False)
    metrics = ["crps", "coverage90", "width90", "mae", "shape_mae", "level_mae",
               "raw_point_mae", "raw_shape_mae", "raw_level_mae"]
    regional = scores.groupby(["region", "method"])[metrics].mean().reset_index()
    regional.to_csv(args.output / "region_metrics.csv", index=False)
    macro = regional.groupby("method")[metrics].mean().sort_values("crps")
    macro.to_csv(args.output / "macro_metrics.csv")
    summary = {"folds": metadata, "macro": macro.reset_index().to_dict("records"),
               "physical_bounds": [0, EF.max()], "target_year_used": 2023,
               "limitations": ["fixed-EF generation proxy, not actual CO2", "synthetic release/vintages",
                               "2023 is exploratory validation", "no Chinese or untouched-year evaluation",
                               "different bases alone are not novelty; learned scale transition is heuristic",
                               "monthly generation-weighted mix is not arithmetic hourly CIF mean",
                               "source residual calibration does not guarantee target coverage"]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    mismatches = [name for name, digest in protocol["hashes"].items() if sha(root / name) != digest]
    data_mismatches = [name for name, digest in protocol["data_hashes"].items() if sha(args.data / name) != digest]
    (args.output / "postrun_verification.json").write_text(json.dumps({
        "source_or_protocol_hash_mismatches": mismatches, "data_hash_mismatches": data_mismatches,
        "all_prediction_seals_match": all(sha(args.output / region / "predictions.npz") == json.loads(
            (args.output / region / "prediction_seal.json").read_text())["predictions_sha256"] for region in args.regions),
    }, indent=2))
    if mismatches or data_mismatches:
        raise RuntimeError("Frozen source/data changed during run")
    print(macro.to_string(), flush=True)


if __name__ == "__main__":
    main()
