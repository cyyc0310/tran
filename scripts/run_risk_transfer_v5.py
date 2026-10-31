"""Immutable V5 coarse-risk transfer, calibrated only on inspected source 2023.

The point models are byte-for-byte V4 round2 artifacts trained on four 2022
source domains. V5 recalibrates all methods on seven source domains in 2023;
this is temporal separation and does not claim an independent-domain split.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import pickle
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SOURCES = ("BPAT", "CISO", "ERCO", "FPL", "ISNE", "MISO", "NYIS")
HOLDOUTS = ("SCL", "WACM", "SEC", "SCEG")
POINTS = ("support_gp_medium", "harmonic_ridge", "softmax_moe")
DELAYS = (0, 30, 90)
VARIANT = "risk_v5"
PROTOCOL = "docs/RISK_TRANSFER_V5_PROTOCOL.md"
CONFIGS = {
    "seven_raw": (None, "source_residual"),
    "seven_component110": (None, "component_transport_110"),
    "adaptive_component110_beta0p5": (.5, "component_transport_110"),
    "adaptive_component110_beta1": (1., "component_transport_110"),
    "adaptive_component110_beta2": (2., "component_transport_110"),
}
BASE_DEPENDENCIES = tuple(f"src/transcif/research/{name}.py" for name in (
    "transfer_v4", "structural_v3", "dynamic_v2", "eia_proxy", "references", "mechanisms"))


def load_helpers():
    """Adapt V4 utilities in memory; no V4 source/protocol file is modified."""
    spec = importlib.util.spec_from_file_location("risk_v5_v4_helpers", ROOT / "scripts/run_transfer_v4.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    helpers.PROTOCOL = PROTOCOL
    helpers.CAL_REGIONS = SOURCES
    helpers.HOLDOUT_REGIONS = HOLDOUTS
    select = helpers.select_config
    helpers.select_config = lambda macro, regional=None, **kwargs: select(macro, regional, regions=SOURCES)
    return helpers


V4 = load_helpers()


def calibration_parameters():
    return {name: {"beta": beta, "mode": mode} for name, (beta, mode) in CONFIGS.items()}


def validate_grid(config):
    """The point/lag/beta grid is model state, including its ordering and names."""
    expected = {"variants": [VARIANT], "methods": list(POINTS), "delays": list(DELAYS),
                "calibration_modes": list(CONFIGS), "calibration_parameters": calibration_parameters(),
                "calibration_regions": list(SOURCES), "calibration_year": 2023,
                "selection_regions": list(SOURCES), "selection_year": 2024,
                "fit_regions": list(V4.FIT_REGIONS), "fit_year": 2022,
                "external_regions": list(HOLDOUTS), "external_year": 2025,
                "projection_bounds": [0., float(V4.EF.max())]}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Current V5 grid differs from the frozen source configuration")
    records = config.get("records", ())
    if [record.get("delay") for record in records] != list(DELAYS):
        raise ValueError("Frozen V5 fit records do not match the fixed delay grid")
    for record in records:
        delay = record["delay"]
        if (record.get("variant") != VARIANT or record.get("path") != f"fit/risk_v5-{delay}.pkl"
                or record.get("base_path") != f"base/round2-{delay}.pkl"):
            raise ValueError("Frozen V5 fit record paths or variants changed")


def verify_base(source, root=ROOT):
    """Require original point-code closure and exact upstream fitted bytes."""
    source, root = Path(source).resolve(), Path(root).resolve()
    manifest = json.loads((source / "prefit_freeze.json").read_text())
    seal = V4._verify_artifacts(source, "fit_seal.json")
    config = json.loads((source / "run_config.json").read_text())
    if config.get("fit_year") != 2022 or tuple(config.get("fit_regions", ())) != V4.FIT_REGIONS:
        raise ValueError("V5 point baseline must be the four-domain 2022 fit")
    if manifest.get("environment") != V4.runtime_environment():
        raise ValueError("Original point-model environment differs")
    dependencies = {}
    for name in BASE_DEPENDENCIES:
        digest = manifest["code_hashes"].get(str(Path(manifest["root"]) / name))
        if not digest or V4.sha(root / name) != digest or V4.sha(source / "snapshot" / name) != digest:
            raise ValueError("Original point-model dependency changed: " + name)
        dependencies[name] = digest
    records = sorted((r for r in config["records"] if r["variant"] == "round2"), key=lambda r: r["delay"])
    if tuple(r["delay"] for r in records) != DELAYS:
        raise ValueError("Original round2 artifacts must contain all three fixed delays")
    return manifest, seal, records, dependencies


def base_points(model, origins, region, monthly):
    from transcif.research import risk_transfer_v5 as risk
    if hasattr(risk, "predict_base"):
        result = risk.predict_base(model, origins, region, monthly)
    else:
        result = model.predict(origins, region, monthly)
    if not set(POINTS) <= set(result):
        raise ValueError("Original baseline lacks a declared V5 point method")
    return {method: result[method] for method in POINTS}


def transform(origins, region, monthly, prediction, delay, calibration):
    from transcif.research.risk_transfer_v5 import risk_scales
    beta, _ = CONFIGS[calibration]
    result = {key: np.asarray(value).copy() for key, value in prediction.items()}
    if beta is not None:
        result = risk_scales(origins, region, monthly, result, extra_delay=delay, beta=beta)
    V4._validate_prediction(result, len(origins))
    if not np.array_equal(result["point"], prediction["point"]):
        raise ValueError("Risk calibration may not alter the frozen point forecast")
    return result


def project_distribution(ensemble, lower, upper):
    """Apply the same public fixed-EF support to every probabilistic method."""
    maximum = float(V4.EF.max())
    endpoints = np.stack((lower, upper), axis=1)
    audit = {
        "preprojection_ensemble_below_zero": np.mean(ensemble < 0, axis=(1, 2)),
        "preprojection_ensemble_above_max": np.mean(ensemble > maximum, axis=(1, 2)),
        "preprojection_interval_endpoints_below_zero": np.mean(endpoints < 0, axis=(1, 2)),
        "preprojection_interval_endpoints_above_max": np.mean(endpoints > maximum, axis=(1, 2)),
    }
    return np.clip(ensemble, 0, maximum), np.clip(lower, 0, maximum), np.clip(upper, 0, maximum), audit


def _fit_seal(output, config):
    validate_grid(config)
    V4.write_json(output / "run_config.json", config)
    artifacts = {record[key]: V4.sha(output / record[key]) for record in config["records"] for key in ("path", "base_path")}
    artifacts["run_config.json"] = V4.sha(output / "run_config.json")
    V4.verify_frozen_inputs(output)
    V4.write_json(output / "fit_seal.json", {"prefit_sha256": V4.sha(output / "prefit_freeze.json"), "artifacts": artifacts})


def _selection(path):
    selected = V4.load_source_selection(path)
    if selected.get("status") != "passed" or tuple(selected.get("selection_regions", ())) != SOURCES:
        raise ValueError("V5 external evaluation requires a passed seven-source 2024 selection")
    source = Path(path).resolve().parent.parent
    config = json.loads((source / "run_config.json").read_text())
    validate_grid(config)
    if config.get("variants") != [VARIANT] or tuple(config.get("calibration_regions", ())) != SOURCES:
        raise ValueError("Selection belongs to a different study or calibration split")
    return source, config, selected


def fit_phase(data, baseline, output, *, external_data=None, selection=None, n_samples=128, root=ROOT):
    from transcif.research.calibration_v4 import SourceCalibrator
    from transcif.research.component_calibration_v4 import ComponentCalibrator
    data, baseline, output, root = map(lambda p: Path(p).resolve(), (data, baseline, output, root))
    _, base_seal, base_records, dependencies = verify_base(baseline, root)
    extra = [root / "scripts/run_risk_transfer_v5.py", baseline / "prefit_freeze.json",
             baseline / "fit_seal.json", baseline / "run_config.json"]
    extra.extend(baseline / record["path"] for record in base_records)
    extra.extend(baseline / "snapshot" / name for name in BASE_DEPENDENCIES)
    selected, selected_source, selected_config = None, None, None
    if external_data is not None:
        if selection is None:
            raise ValueError("External mode requires a source-only selection")
        selected_source, selected_config, selected = _selection(selection)
        selected_seal = V4._verify_artifacts(selected_source, "fit_seal.json")
        selected_manifest = json.loads((selected_source / "prefit_freeze.json").read_text())
        runner_relative = "scripts/run_risk_transfer_v5.py"
        selected_runner = selected_manifest["code_hashes"].get(str(Path(selected_manifest["root"]) / runner_relative))
        if selected_runner != V4.sha(root / runner_relative):
            raise ValueError("V5 runner changed since source selection")
        if selected_config["base_code_dependency_hashes"] != dependencies or selected_config["base_fit_seal_sha256"] != V4.sha(baseline / "fit_seal.json"):
            raise ValueError("Source selection original point-model provenance differs")
        for selected_record, original_record in zip(selected_config["records"], base_records):
            original_path = baseline / original_record["path"]
            if selected_record["original_base_path"] != str(original_path) or selected_record["original_base_sha256"] != base_seal["artifacts"][original_record["path"]]:
                raise ValueError("Source-selected point-model fit record differs")
        extra.extend([Path(selection).resolve(), *(selected_source / name for name in (
            "prefit_freeze.json", "fit_seal.json", "run_config.json", "all_predictions_sealed.json", "evaluation_plan.json",
            "scores/macro_metrics.csv", "scores/region_metrics.csv"))])
        extra.extend(selected_source / name for name in selected_seal["artifacts"])
    frozen = V4.freeze_inputs(root, data, output, extra_files=extra, external_data=external_data)
    if external_data is not None:
        V4.verify_external_audit(frozen)
        source_manifest = json.loads((selected_source / "prefit_freeze.json").read_text())
        if V4.model_closure_hashes(source_manifest) != V4.model_closure_hashes(frozen) or source_manifest["environment"] != frozen["environment"]:
            raise ValueError("V5 risk/calibration code or environment changed since source selection")
        if selected_config["n_samples"] != n_samples:
            raise ValueError("Source-selected ensemble size must remain fixed")
        for record in selected_config["records"]:
            for key in ("path", "base_path"):
                destination = output / record[key]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(selected_source / record[key], destination)
                if V4.sha(destination) != selected_seal["artifacts"][record[key]]:
                    raise ValueError("Source-fitted bytes changed while copying")
        config = {**selected_config, "source_selection": selected,
            "source_provenance": {"source_run": str(selected_source), "source_fit_seal_sha256": V4.sha(selected_source / "fit_seal.json"),
                "source_fit_artifact_hashes": selected_seal["artifacts"], "fitted_bytes_identical": True}}
        _fit_seal(output, config)
        return config
    monthly = V4.monthly_inputs(frozen)
    origins = V4._origins(2023)
    truths = {region: V4.daily_profiles(V4.read_year(data / f"{region}_hourly.csv", 2023)) for region in SOURCES}
    (output / "base").mkdir()
    (output / "fit").mkdir()
    records = []
    for original in base_records:
        delay = original["delay"]
        V4.verify_frozen_inputs(output)
        base_path = f"base/round2-{delay}.pkl"
        shutil.copyfile(baseline / original["path"], output / base_path)
        if V4.sha(output / base_path) != base_seal["artifacts"][original["path"]]:
            raise ValueError("Original point-model bytes changed while copying")
        with (output / base_path).open("rb") as handle:
            model = pickle.load(handle)["model"]
        banks = {(method, calibration): [] for method in POINTS for calibration in CONFIGS}
        for region in SOURCES:
            predictions = base_points(model, origins, region, monthly)
            days, truth = truths[region]
            ix = origins.get_indexer(days)
            if not len(ix) or np.any(ix < 0):
                raise ValueError("Missing source 2023 calibration days")
            for method in POINTS:
                for calibration in CONFIGS:
                    p = transform(origins, region, monthly, predictions[method], delay, calibration)
                    banks[method, calibration].append({"point": p["point"][ix], "truth": truth,
                        "scales": p["scale"][ix], "level_scales": p["level_scale"][ix], "shape_scales": p["shape_scale"][ix],
                        "regions": np.repeat(region, len(ix)), "ages": V4.age_bins(p["age"][ix]),
                        "months": days.month.to_numpy(), "years": days.year.to_numpy()})
        calibrators = {}
        for key, parts in banks.items():
            values = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
            if CONFIGS[key[1]][1] == "source_residual":
                values = {name: value for name, value in values.items() if name not in ("level_scales", "shape_scales")}
                calibrator = SourceCalibrator(source_regions=SOURCES, alpha=.1)
            else:
                calibrator = ComponentCalibrator(source_regions=SOURCES, alpha=.1)
            calibrators[key] = calibrator.fit(**values)
        path = f"fit/risk_v5-{delay}.pkl"
        with (output / path).open("xb") as handle:
            pickle.dump({"calibrators": calibrators}, handle)
        records.append({"variant": VARIANT, "delay": delay, "path": path, "base_path": base_path,
                        "original_base_path": str(baseline / original["path"]), "original_base_sha256": V4.sha(output / base_path)})
    config = {"study_version": 5, "variants": [VARIANT], "methods": list(POINTS), "delays": list(DELAYS),
        "calibration_modes": list(CONFIGS), "calibration_parameters": calibration_parameters(),
        "fit_regions": list(V4.FIT_REGIONS), "fit_year": 2022, "calibration_regions": list(SOURCES), "calibration_year": 2023,
        "calibration_split": "temporal separation; fit/calibration domains overlap; no independent-domain calibration claim",
        "selection_regions": list(SOURCES), "selection_year": 2024, "external_year": 2025, "external_regions": list(HOLDOUTS),
        "records": records, "n_samples": n_samples, "ensemble_dtype": "float32", "ensemble_seed": 17023,
        "projection_bounds": [0., float(V4.EF.max())], "projection_stage": "before_global_seal; raw point remains unchanged",
        "base_fit_seal_sha256": V4.sha(baseline / "fit_seal.json"), "base_code_dependency_hashes": dependencies,
        "source_selection": None, "v4_external_feedback": "six V4 domains are development feedback only; never used for V5 fitting, calibration or quantitative selection"}
    _fit_seal(output, config)
    return config


def predict_seal_phase(output, *, study="development", end=None, selection=None):
    output = Path(output).resolve()
    frozen = V4.verify_frozen_inputs(output)
    V4._verify_artifacts(output, "fit_seal.json")
    config = json.loads((output / "run_config.json").read_text())
    validate_grid(config)
    if study == "development":
        targets, year, selected = SOURCES, 2024, None
    elif study == "holdout":
        V4.verify_external_audit(frozen)
        selected = config.get("source_selection")
        if not selected or not config.get("source_provenance") or end is not None:
            raise ValueError("Holdout requires a frozen source selection and complete fixed 2025 year")
        if selection is not None and _selection(selection)[2] != selected:
            raise ValueError("External selection differs from its frozen source fit")
        targets, year = HOLDOUTS, 2025
    else:
        raise ValueError("Unknown study")
    origins = V4._origins(year, end)
    monthly = V4.monthly_inputs(frozen)
    expected = [(region, delay, VARIANT, method, calibration) for region in targets for delay in DELAYS for method in POINTS for calibration in CONFIGS]
    V4.write_json(output / "evaluation_plan.json", {"study": study, "year": year, "targets": list(targets),
        "origins": [str(day) for day in origins], "selection": selected,
        "label_data": frozen["external_data"] if study == "holdout" else frozen["data"], "expected": [list(key) for key in expected]})
    folder = output / "predictions"
    folder.mkdir()
    entries = []
    for record in config["records"]:
        V4.verify_frozen_inputs(output)
        V4._verify_artifacts(output, "fit_seal.json")
        with (output / record["base_path"]).open("rb") as handle:
            model = pickle.load(handle)["model"]
        with (output / record["path"]).open("rb") as handle:
            calibrators = pickle.load(handle)["calibrators"]
        for region in targets:
            predictions = base_points(model, origins, region, monthly)
            for method in POINTS:
                for name, (_, mode) in CONFIGS.items():
                    p = transform(origins, region, monthly, predictions[method], record["delay"], name)
                    options = {"ages": V4.age_bins(p["age"])}
                    if mode != "source_residual":
                        options.update(level_scale=p["level_scale"], shape_scale=p["shape_scale"])
                    calibrator = calibrators[method, name]
                    lower, upper = calibrator.intervals(p["point"], p["scale"], mode=mode, **options)
                    ensemble = np.asarray(calibrator.ensemble(p["point"], p["scale"], mode=mode,
                        **options, n_samples=config["n_samples"], seed=config["ensemble_seed"]), dtype=config["ensemble_dtype"])
                    if ensemble.shape != (len(origins), config["n_samples"], 24) or not np.isfinite(ensemble).all():
                        raise ValueError("Invalid V5 ensemble")
                    lower, upper = np.asarray(lower), np.asarray(upper)
                    if lower.shape != (len(origins), 24) or upper.shape != lower.shape or not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower > upper):
                        raise ValueError("Invalid V5 analytic interval")
                    ensemble, lower, upper, boundary = project_distribution(ensemble, lower, upper)
                    path = folder / f"{region}-{record['delay']}-{method}-{name}.npz"
                    with path.open("xb") as handle:
                        np.savez_compressed(handle, origins=origins.as_unit("ns").asi8,
                            **{key: p[key] for key in ("point", "scale", "age", "level_scale", "shape_scale")},
                            ensemble=ensemble, analytic_lower=lower, analytic_upper=upper, **boundary)
                    entries.append({"region": region, "delay": record["delay"], "variant": VARIANT,
                        "method": method, "calibration": name, "path": str(path.relative_to(output)),
                        "preprojection_boundary_mass": {key: float(value.mean()) for key, value in boundary.items()}})
    boundary_rows = [{**{key: entry[key] for key in ("region", "delay", "variant", "method", "calibration")},
                      **entry["preprojection_boundary_mass"]} for entry in entries]
    pd.DataFrame(boundary_rows).to_csv(output / "preprojection_boundary_audit.csv", index=False)
    V4.seal_predictions(output, expected, entries)
    V4.write_json(output / "evaluation_commit.json", {"plan_sha256": V4.sha(output / "evaluation_plan.json"),
        "global_seal_sha256": V4.sha(output / "all_predictions_sealed.json")})
    return entries


def score_phase(output):
    return V4.score_phase(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("fit", "predict_seal", "score", "all"), default="all")
    parser.add_argument("--data", type=Path, default=Path("results/heldout-2026h1-data"))
    parser.add_argument("--baseline", type=Path, default=Path("results/transfer-v4-round2-source2024"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--external-data", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--study", choices=("development", "holdout"), default="development")
    parser.add_argument("--end")
    parser.add_argument("--n-samples", type=int, default=128)
    args = parser.parse_args()
    if args.n_samples < 2:
        parser.error("At least two samples required")
    if args.phase in ("fit", "all"):
        if args.study == "holdout" and (args.external_data is None or args.selection is None):
            parser.error("Holdout fit requires --external-data and --selection")
        if args.study == "development" and args.external_data is not None:
            parser.error("External data may not enter source development fitting")
        fit_phase(args.data, args.baseline, args.output, external_data=args.external_data,
                  selection=args.selection, n_samples=args.n_samples)
    if args.phase in ("predict_seal", "all"):
        predict_seal_phase(args.output, study=args.study, end=args.end, selection=args.selection)
    if args.phase in ("score", "all"):
        print(score_phase(args.output).to_string(index=False))


if __name__ == "__main__":
    main()
