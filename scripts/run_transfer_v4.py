"""Immutable source-fit / calibration / selection / external evaluation runner.

Hashing input bytes is permitted before sealing; parsing external hourly outcomes
is only permitted through LabelAccess after every prediction artifact is sealed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from io import StringIO
import json
import importlib.metadata
import platform
from pathlib import Path
import pickle
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from transcif.research.dynamic_v2 import daily_profiles, EF
from transcif.research.weighted_metrics import PreparedDistribution

FIT_REGIONS = ("BPAT", "ERCO", "MISO", "NYIS")
CAL_REGIONS = ("CISO", "FPL", "ISNE")
DELAYS = (0, 30, 90)
HOLDOUT_REGIONS = ("SWPP", "AVA", "BANC", "JEA", "SC", "NWMT")
AGE_BIN_UPPER_BOUNDS = (60., 120., 240.)
AGE_BIN_LABELS = ("le60", "61_120", "121_240", "gt240")
PROTOCOL = "docs/TRANSFER_V4_PROTOCOL.md"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def read_year(path, year):
    """Filter raw timestamp text before pandas parses any numeric label."""
    with Path(path).open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        time_column = header.index("time")
        selected = [row for row in reader if row[time_column].startswith(f"{year}-")]
    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerows(selected)
    buffer.seek(0)
    frame = pd.read_csv(buffer)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    return frame


def runtime_environment():
    versions = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "packages": versions}


def model_closure_hashes(manifest):
    source = Path(manifest["root"]) / "src" / "transcif"
    return {str(Path(path).relative_to(source)): digest
            for path, digest in manifest["code_hashes"].items() if Path(path).is_relative_to(source)}


def freeze_inputs(root, data, output, extra_files=(), external_data=None):
    """Create a new run and snapshot the complete transitive Python package."""
    root, data, output = Path(root).resolve(), Path(data).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Immutable output: choose a fresh run directory")
    files = sorted({*root.glob("src/transcif/**/*.py"), root / "scripts/run_transfer_v4.py",
                    root / PROTOCOL, *(Path(p).resolve() for p in extra_files)})
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen dependency: " + ", ".join(missing))
    external_data = Path(external_data).resolve() if external_data is not None else None
    directories = (data, external_data) if external_data is not None else (data,)
    csv_files = sorted({p for directory in directories for p in directory.rglob("*.csv")})
    input_files = sorted({*csv_files, *(directory / "audit.json" for directory in directories if (directory / "audit.json").is_file())})
    if not csv_files:
        raise ValueError("No raw CSV inputs to freeze")
    # Hash before copying or fitting. Never refresh this manifest.
    manifest = {"root": str(root), "data": str(data),
                "code_hashes": {str(p): sha(p) for p in files},
                "external_data": str(external_data) if external_data is not None else None,
                "environment": runtime_environment(),
                "data_hashes": {str(p): sha(p) for p in input_files}}
    output.mkdir(parents=True)
    for index, path in enumerate(files):
        relative = path.relative_to(root) if path.is_relative_to(root) else Path("external") / str(index) / path.name
        destination = output / "snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    manifest["snapshot_hashes"] = {str(p.relative_to(output)): sha(p)
                                   for p in sorted((output / "snapshot").rglob("*")) if p.is_file()}
    write_json(output / "prefit_freeze.json", manifest)
    verify_frozen_inputs(output, manifest)
    return manifest


def verify_frozen_inputs(output, manifest=None):
    output = Path(output)
    if manifest is None:
        manifest = json.loads((output / "prefit_freeze.json").read_text())
    paths = {**manifest["code_hashes"], **manifest["data_hashes"],
             **{str(output / p): h for p, h in manifest["snapshot_hashes"].items()}}
    for path, digest in paths.items():
        if not Path(path).is_file() or sha(path) != digest:
            raise ValueError("Frozen input changed before label access: " + str(path))
    return manifest


def _verify_artifacts(output, manifest_name):
    output = Path(output)
    manifest = json.loads((output / manifest_name).read_text())
    if sha(output / "prefit_freeze.json") != manifest["prefit_sha256"]:
        raise ValueError("Prefit manifest changed")
    for name, digest in manifest["artifacts"].items():
        if not (output / name).is_file() or sha(output / name) != digest:
            raise ValueError("Sealed artifact changed: " + name)
    return manifest


def seal_predictions(output, expected, entries):
    """Commit a global seal only if the full predeclared grid is present."""
    output = Path(output)
    verify_frozen_inputs(output)
    fit_seal = _verify_artifacts(output, "fit_seal.json")
    names = [entry["path"] for entry in entries]
    keys = [tuple(entry[k] for k in ("region", "delay", "variant", "method", "calibration")) for entry in entries]
    if len(set(names)) != len(names) or len(set(keys)) != len(keys) or set(keys) != set(expected):
        raise ValueError("Cannot globally seal an incomplete or duplicate prediction grid")
    artifacts = {name: sha(output / name) for name in sorted(names)}
    write_json(output / "all_predictions_sealed.json", {
        "target_labels_read": False, "prefit_sha256": sha(output / "prefit_freeze.json"),
        "fit_seal_sha256": sha(output / "fit_seal.json"), "artifacts": artifacts,
        "entries": entries, "expected": [list(key) for key in expected],
        "fit_artifacts": fit_seal["artifacts"],
        "plan_sha256": sha(output / "evaluation_plan.json") if (output / "evaluation_plan.json").exists() else None})


class LabelAccess:
    """Only a verified global seal grants external target-year numeric access."""

    def __init__(self, output):
        self.output = Path(output)
        verify_frozen_inputs(self.output)
        self.seal = _verify_artifacts(self.output, "all_predictions_sealed.json")
        if self.seal.get("plan_sha256") is not None and sha(self.output / "evaluation_plan.json") != self.seal["plan_sha256"]:
            raise ValueError("Sealed evaluation plan changed")
        if self.seal.get("target_labels_read") is not False:
            raise ValueError("Missing pre-label global seal")
        if sha(self.output / "fit_seal.json") != self.seal["fit_seal_sha256"]:
            raise ValueError("Fit seal changed")
        _verify_artifacts(self.output, "fit_seal.json")
        expected = {tuple(key) for key in self.seal["expected"]}
        actual = {tuple(e[k] for k in ("region", "delay", "variant", "method", "calibration")) for e in self.seal["entries"]}
        if expected != actual or len(actual) != len(self.seal["entries"]):
            raise ValueError("Global seal grid mismatch")
        self.global_digest = sha(self.output / "all_predictions_sealed.json")

    def read(self, path, year):
        verify_frozen_inputs(self.output)
        if sha(self.output / "all_predictions_sealed.json") != self.global_digest:
            raise ValueError("Global seal changed before target labels")
        _verify_artifacts(self.output, "all_predictions_sealed.json")
        return read_year(path, year)


def _origins(year, end=None):
    end = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp(f"{year}-12-31", tz="UTC")
    start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    if end.year != year or end < start:
        raise ValueError("End date must lie within its fixed evaluation year")
    return pd.date_range(start, end, freq="D")


def _validate_prediction(prediction, n):
    required = {"point": (n, 24), "scale": (n, 24), "age": (n,),
                "level_scale": (n,), "shape_scale": (n, 24)}
    for key, shape in required.items():
        value = np.asarray(prediction[key])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError("Invalid model prediction: " + key)
    if np.any(prediction["scale"] <= 0):
        raise ValueError("Predictive scales must be positive")


def age_bins(ages):
    ages = np.asarray(ages, dtype=float)
    if ages.ndim != 1 or not np.isfinite(ages).all() or np.any(ages < 0):
        raise ValueError("Calibration ages must be finite nonnegative day ages")
    return np.asarray(AGE_BIN_LABELS)[np.searchsorted(AGE_BIN_UPPER_BOUNDS, ages, side="left")]


def monthly_inputs(manifest):
    directories = [Path(manifest["data"])]
    if manifest.get("external_data"):
        directories.append(Path(manifest["external_data"]))
    combined = pd.concat([pd.read_csv(directory / "monthly.csv") for directory in directories], ignore_index=True)
    keys = [key for key in ("region", "period_start", "period_end") if key in combined]
    if len(keys) == 3 and combined.duplicated(keys).any():
        raise ValueError("Ambiguous overlapping monthly histories across input directories")
    return combined


def verify_external_audit(manifest):
    if not manifest.get("external_data"):
        raise ValueError("Holdout requires separately frozen --external-data")
    directory = Path(manifest["external_data"])
    audit_path = directory / "audit.json"
    if str(audit_path) not in manifest["data_hashes"]:
        raise ValueError("External data audit was not frozen before fitting")
    audit = json.loads(audit_path.read_text())
    if tuple(audit.get("external_regions", ())) != HOLDOUT_REGIONS or audit.get("final_external_year") != 2025:
        raise ValueError("External data audit differs from the predeclared holdout")
    for key in ("cif_scores_inspected", "cif_summaries_computed", "target_hourly_used_for_fit_or_calibration"):
        if audit.get(key) is not False:
            raise ValueError("External audit lacks clean pre-evaluation provenance: " + key)
    if not audit.get("protocol_sha256_before") or audit.get("protocol_sha256_before") != audit.get("protocol_sha256_after"):
        raise ValueError("External protocol changed during preparation")
    for name, digest in audit.get("derived_hashes", {}).items():
        if manifest["data_hashes"].get(str(directory / name)) != digest:
            raise ValueError("External derived input differs from its audit: " + name)
    required = {"monthly.csv", *(f"{region}_hourly.csv" for region in HOLDOUT_REGIONS)}
    if not required <= set(audit.get("derived_hashes", {})):
        raise ValueError("External audit omits required data files")
    return audit


def reuse_source_fit(source_output, output, manifest, selection, variants, delays, n_samples):
    """Use the exact source-selected model/calibration bytes externally."""
    source_output, output = Path(source_output), Path(output)
    source_manifest = json.loads((source_output / "prefit_freeze.json").read_text())
    if source_manifest.get("environment") != manifest["environment"]:
        raise ValueError("Source selection environment is missing or differs from this run")
    if model_closure_hashes(source_manifest) != model_closure_hashes(manifest):
        raise ValueError("Model/calibration code closure differs from source selection")
    source_seal = _verify_artifacts(source_output, "fit_seal.json")
    if selection.get("fit_seal_sha256") != sha(source_output / "fit_seal.json"):
        raise ValueError("Selection is not bound to this source fitted model")
    if selection.get("model_closure_hashes") != model_closure_hashes(manifest):
        raise ValueError("Selection model closure does not match the current model")
    source_config = json.loads((source_output / "run_config.json").read_text())
    if source_config["variants"] != list(variants) or source_config["delays"] != list(delays) or source_config["n_samples"] != n_samples:
        raise ValueError("External grid or ensemble size differs from source evaluation")
    if source_config.get("age_bin_upper_bounds_days") != list(AGE_BIN_UPPER_BOUNDS) or source_config.get("age_bin_labels") != list(AGE_BIN_LABELS):
        raise ValueError("External age-bin policy differs from source calibration")
    selected = selection["configuration"]
    for key, allowed in (("variant", source_config["variants"]), ("method", source_config["methods"]), ("calibration", source_config["calibration_modes"])):
        if selected[key] not in allowed:
            raise ValueError("Source-selected configuration is outside the fixed grid")
    (output / "fit").mkdir()
    for record in source_config["records"]:
        source_path, destination = source_output / record["path"], output / record["path"]
        shutil.copyfile(source_path, destination)
        if sha(destination) != source_seal["artifacts"][record["path"]]:
            raise ValueError("Copied source fit artifact differs")
    config = {**source_config, "source_selection": selection,
              "source_provenance": {"source_run": str(source_output.resolve()),
                  "source_fit_seal_sha256": sha(source_output / "fit_seal.json"),
                  "source_fit_artifact_hashes": {r["path"]: source_seal["artifacts"][r["path"]] for r in source_config["records"]},
                  "source_runner_sha256": source_manifest["code_hashes"].get(str(Path(source_manifest["root"]) / "scripts/run_transfer_v4.py")),
                  "current_runner_sha256": sha(Path(manifest["root"]) / "scripts/run_transfer_v4.py"),
                  "model_closure_identical": True, "fitted_bytes_identical": True}}
    write_json(output / "run_config.json", config)
    artifacts = {record["path"]: sha(output / record["path"]) for record in config["records"]}
    artifacts["run_config.json"] = sha(output / "run_config.json")
    verify_frozen_inputs(output, manifest)
    write_json(output / "fit_seal.json", {"prefit_sha256": sha(output / "prefit_freeze.json"), "artifacts": artifacts})
    return config


def fit_phase(data, output, variants, delays=DELAYS, n_samples=128, root=ROOT,
              external_data=None, selection=None):
    from transcif.research.transfer_v4 import fit_transfer
    from transcif.research.calibration_v4 import SourceCalibrator, MODES
    from transcif.research.component_calibration_v4 import ComponentCalibrator, COMPONENT_MODES
    data, output = Path(data).resolve(), Path(output).resolve()
    source_selection = load_source_selection(selection) if selection is not None else None
    extra_files = []
    source_output = None
    if external_data is not None:
        if source_selection is None or source_selection.get("status") != "passed":
            raise ValueError("External evaluation requires a passed fixed source selection")
        source_output = Path(selection).resolve().parent.parent
        extra_files = [Path(selection).resolve(), source_output / "prefit_freeze.json", source_output / "fit_seal.json",
            source_output / "run_config.json", source_output / "evaluation_plan.json", source_output / "all_predictions_sealed.json",
            source_output / "scores/macro_metrics.csv", source_output / "scores/region_metrics.csv"]
        source_config = json.loads((source_output / "run_config.json").read_text())
        extra_files.extend(source_output / record["path"] for record in source_config["records"])
    manifest = freeze_inputs(root, data, output, extra_files=extra_files, external_data=external_data)
    if external_data is not None:
        verify_external_audit(manifest)
        return reuse_source_fit(source_output, output, manifest, source_selection, variants, delays, n_samples)
    frames = {region: read_year(data / f"{region}_hourly.csv", 2022) for region in FIT_REGIONS}
    monthly = monthly_inputs(manifest)
    cal_origins = _origins(2023)
    # Independent calibration outcomes are never passed into fit_transfer.
    calibration_truth = {}
    for region in CAL_REGIONS:
        days, truth = daily_profiles(read_year(data / f"{region}_hourly.csv", 2023))
        calibration_truth[region] = (days, truth)
    fit_dir = output / "fit"
    fit_dir.mkdir()
    records, methods = [], None
    for variant in variants:
        for delay in delays:
            verify_frozen_inputs(output, manifest)
            model = fit_transfer(frames, monthly, extra_delay=delay, variant=variant)
            calibrators, component_calibrators = {}, {}
            banks = {}
            for region in CAL_REGIONS:
                predictions = model.predict(cal_origins, region, monthly)
                current = tuple(sorted(predictions))
                if methods is None:
                    methods = current
                if current != methods:
                    raise ValueError("Method set must be identical in every forecast")
                days, truth = calibration_truth[region]
                positions = cal_origins.get_indexer(days)
                if np.any(positions < 0) or not len(positions):
                    raise ValueError("Missing independent calibration days")
                for method, prediction in predictions.items():
                    _validate_prediction(prediction, len(cal_origins))
                    banks.setdefault(method, []).append({"point": prediction["point"][positions], "truth": truth,
                        "scales": prediction["scale"][positions], "ages": age_bins(prediction["age"][positions]),
                        "level_scales": prediction["level_scale"][positions],
                        "shape_scales": prediction["shape_scale"][positions],
                        "regions": np.repeat(region, len(days)), "months": days.month.to_numpy(),
                        "years": days.year.to_numpy()})
            for method, parts in banks.items():
                combined = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
                base = {key: value for key, value in combined.items() if key not in ("level_scales", "shape_scales")}
                calibrators[method] = SourceCalibrator(source_regions=list(CAL_REGIONS), alpha=.1).fit(**base)
                component_calibrators[method] = ComponentCalibrator(source_regions=list(CAL_REGIONS), alpha=.1).fit(
                    **base, level_scales=combined["level_scales"], shape_scales=combined["shape_scales"])
            path = fit_dir / f"{variant}-{delay}.pkl"
            with path.open("xb") as handle:
                pickle.dump({"model": model, "calibrators": calibrators, "component_calibrators": component_calibrators}, handle)
            records.append({"variant": variant, "delay": delay, "path": str(path.relative_to(output))})
    config = {"fit_regions": FIT_REGIONS, "fit_year": 2022, "calibration_regions": CAL_REGIONS,
              "calibration_year": 2023, "selection_regions": CAL_REGIONS, "selection_year": 2024,
              "external_year": 2025, "variants": list(variants), "delays": list(delays),
              "methods": list(methods or ()), "calibration_modes": list(MODES) + list(COMPONENT_MODES), "n_samples": n_samples,
              "component_modes": list(COMPONENT_MODES), "ensemble_dtype": "float32", "ensemble_seed": 17023,
              "age_bin_upper_bounds_days": AGE_BIN_UPPER_BOUNDS, "age_bin_labels": AGE_BIN_LABELS,
              "selection_rule": "equal-region/equal-delay macro coverage90 absolute gap <= .05 and minimum region-delay coverage >= .85, then minimum CRPS; empirical ensemble intervals",
              "records": records}
    write_json(output / "run_config.json", config)
    verify_frozen_inputs(output, manifest)
    artifacts = {record["path"]: sha(output / record["path"]) for record in records}
    artifacts["run_config.json"] = sha(output / "run_config.json")
    write_json(output / "fit_seal.json", {"prefit_sha256": sha(output / "prefit_freeze.json"), "artifacts": artifacts})
    return config


def predict_seal_phase(output, study="development", targets=(), selection=None, end=None):
    output = Path(output).resolve()
    frozen = verify_frozen_inputs(output)
    _verify_artifacts(output, "fit_seal.json")
    config = json.loads((output / "run_config.json").read_text())
    if study not in ("development", "holdout"):
        raise ValueError("Unknown study")
    if study == "holdout":
        if not targets:
            targets = HOLDOUT_REGIONS
        if tuple(targets) != HOLDOUT_REGIONS:
            raise ValueError("External targets must match the predeclared six-region order")
        if end is not None:
            raise ValueError("The external holdout is fixed to the complete 2025 calendar year")
        verify_external_audit(frozen)
        selected = load_source_selection(selection) if selection else config.get("source_selection", {})
        if not config.get("source_provenance") or selected != config.get("source_selection"):
            raise ValueError("Holdout must use the selection bound before its fit freeze")
        if selected.get("status") != "passed" or selected.get("selection_year") != 2024 or tuple(selected.get("selection_regions", ())) != CAL_REGIONS:
            raise ValueError("Holdout requires a passed, fixed source-2024 selection")
    else:
        targets, selected = CAL_REGIONS, None
    year = 2024 if study == "development" else 2025
    origins = _origins(year, end)
    prediction_dir = output / "predictions"
    prediction_dir.mkdir()
    monthly = monthly_inputs(frozen)
    expected = [(region, delay, variant, method, mode) for region in targets
                for delay in config["delays"] for variant in config["variants"]
                for method in config["methods"] for mode in config["calibration_modes"]]
    entries = []
    write_json(output / "evaluation_plan.json", {"study": study, "year": year, "targets": list(targets),
        "origins": [str(day) for day in origins], "selection": selected,
        "label_data": frozen.get("external_data") if study == "holdout" else frozen["data"],
        "expected": [list(key) for key in expected]})
    # No target hourly file is opened anywhere in this phase.
    for record in config["records"]:
        verify_frozen_inputs(output)
        _verify_artifacts(output, "fit_seal.json")
        with (output / record["path"]).open("rb") as handle:
            fitted = pickle.load(handle)
        for region in targets:
            predictions = fitted["model"].predict(origins, region, monthly)
            if sorted(predictions) != config["methods"]:
                raise ValueError("Prediction method grid changed")
            for method, prediction in predictions.items():
                _validate_prediction(prediction, len(origins))
                for mode in config["calibration_modes"]:
                    component = mode in config["component_modes"]
                    calibrator = fitted["component_calibrators" if component else "calibrators"][method]
                    options = {"ages": age_bins(prediction["age"])}
                    if component:
                        options.update(level_scale=prediction["level_scale"], shape_scale=prediction["shape_scale"])
                    lower, upper = calibrator.intervals(prediction["point"], prediction["scale"], mode=mode, **options)
                    ensemble = calibrator.ensemble(prediction["point"], prediction["scale"], mode=mode,
                        **options, n_samples=config["n_samples"], seed=config["ensemble_seed"])
                    ensemble = np.asarray(ensemble, dtype=config["ensemble_dtype"])
                    if ensemble.shape != (len(origins), config["n_samples"], 24) or not np.isfinite(ensemble).all():
                        raise ValueError("Invalid calibrated ensemble")
                    lower, upper = np.asarray(lower), np.asarray(upper)
                    if lower.shape != (len(origins), 24) or upper.shape != lower.shape or not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower > upper):
                        raise ValueError("Invalid analytic intervals")
                    name = f"{region}-{record['delay']}-{record['variant']}-{method}-{mode}.npz"
                    path = prediction_dir / name
                    with path.open("xb") as handle:
                        np.savez_compressed(handle, origins=origins.as_unit("ns").asi8, **prediction,
                            ensemble=ensemble, analytic_lower=lower, analytic_upper=upper)
                    entries.append({"region": region, "delay": record["delay"], "variant": record["variant"],
                                    "method": method, "calibration": mode, "path": str(path.relative_to(output))})
    seal_predictions(output, expected, entries)
    # Bind the plan to the global seal by freezing its digest in a separate commit.
    write_json(output / "evaluation_commit.json", {"plan_sha256": sha(output / "evaluation_plan.json"),
        "global_seal_sha256": sha(output / "all_predictions_sealed.json")})
    return entries


def interval_metrics(lower, upper, truth, alpha=.1):
    lower, upper, truth = map(np.asarray, (lower, upper, truth))
    score = upper - lower + 2 / alpha * np.maximum(lower - truth, 0) + 2 / alpha * np.maximum(truth - upper, 0)
    return {"coverage90": float(np.mean((truth >= lower) & (truth <= upper))),
            "width90": float(np.mean(upper - lower)), "interval_score90": float(np.mean(score)),
            "lower_tail_miss": float(np.mean(truth < lower)), "upper_tail_miss": float(np.mean(truth > upper)),
            "path_coverage90": float(np.all((truth >= lower) & (truth <= upper)))}


def score_day(point, ensemble, lower, upper, truth):
    point, ensemble, truth = map(np.asarray, (point, ensemble, truth))
    distribution = PreparedDistribution(ensemble, np.ones(len(ensemble)))
    result = distribution.score(truth)
    result.update(raw_point_mae=float(np.abs(point - truth).mean()),
        raw_shape_mae=float(np.abs((point - point.mean()) - (truth - truth.mean())).mean()),
        raw_level_mae=float(abs(point.mean() - truth.mean())),
        raw_point_below_zero=float(np.mean(point < 0)), raw_point_above_max=float(np.mean(point > EF.max())),
        ensemble_below_zero=float(np.mean(ensemble < 0)), ensemble_above_max=float(np.mean(ensemble > EF.max())))
    result.update({"empirical_" + k: v for k, v in interval_metrics(distribution.lower, distribution.upper, truth).items()})
    result.update({"analytic_" + k: v for k, v in interval_metrics(lower, upper, truth).items()})
    return result


def load_source_selection(path):
    """Check a completed development selection against its sealed source scores."""
    path = Path(path).resolve()
    selected = json.loads(path.read_text())
    for name, key in (("macro_metrics.csv", "macro_metrics_sha256"),
                      ("region_metrics.csv", "region_metrics_sha256")):
        if sha(path.parent / name) != selected.get(key):
            raise ValueError("Source selection scores changed")
    source_output = path.parent.parent
    source_fit_seal = _verify_artifacts(source_output, "fit_seal.json")
    if selected.get("fit_seal_sha256") != sha(source_output / "fit_seal.json"):
        raise ValueError("Selection fit seal differs")
    source_manifest = json.loads((source_output / "prefit_freeze.json").read_text())
    if selected.get("model_closure_hashes") != model_closure_hashes(source_manifest) or selected.get("environment") != source_manifest.get("environment"):
        raise ValueError("Selection lacks the frozen model/environment binding")
    if selected.get("fit_artifact_hashes") != source_fit_seal["artifacts"]:
        raise ValueError("Selection fitted artifact binding differs")
    for name, digest in source_manifest["snapshot_hashes"].items():
        if sha(source_output / name) != digest:
            raise ValueError("Source-selected code snapshot changed: " + name)
    if sha(source_output / "all_predictions_sealed.json") != selected.get("prediction_global_seal_sha256"):
        raise ValueError("Source selection prediction seal changed")
    plan = json.loads((source_output / "evaluation_plan.json").read_text())
    source_global = json.loads((source_output / "all_predictions_sealed.json").read_text())
    if sha(source_output / "evaluation_plan.json") != source_global.get("plan_sha256"):
        raise ValueError("Source selection evaluation plan changed")
    if plan.get("study") != "development" or plan.get("year") != 2024 or tuple(plan.get("targets", ())) != CAL_REGIONS:
        raise ValueError("Selection must derive exclusively from source-region 2024 outcomes")
    recomputed = select_config(pd.read_csv(path.parent / "macro_metrics.csv"), pd.read_csv(path.parent / "region_metrics.csv"))
    if any(selected.get(key) != recomputed.get(key) for key in ("status", "configuration")):
        raise ValueError("Source selection does not match predeclared rule")
    return selected


def select_config(macro, regional=None, regions=CAL_REGIONS):
    """A failed coverage gate has no winner; target scores never enter selection."""
    keys = ["variant", "method", "calibration"]
    numeric = [column for column in macro.select_dtypes(include="number").columns if column != "delay"]
    candidates = macro.groupby(keys, sort=True)[numeric].mean().reset_index()
    if regional is None:
        raise ValueError("Regional scores are required for the worst-region coverage gate")
    worst = regional.groupby(keys)["empirical_coverage90"].min().rename("worst_region_delay_coverage90").reset_index()
    candidates = candidates.merge(worst, on=keys, validate="one_to_one")
    candidates["coverage_gap"] = np.abs(candidates["empirical_coverage90"] - .9)
    passing = candidates.loc[(candidates.coverage_gap <= .05 + 1e-12) & (candidates.worst_region_delay_coverage90 >= .85 - 1e-12)]
    base = {"selection_year": 2024, "selection_regions": list(regions), "coverage_gate": .05,
            "metric": "equal-region/equal-delay macro empirical_coverage90 and CRPS",
            "worst_region_delay_coverage_floor": .85, "candidate_count": len(candidates)}
    if passing.empty:
        closest = candidates.assign(gate_violation=np.maximum(candidates.coverage_gap - .05, 0) + np.maximum(.85 - candidates.worst_region_delay_coverage90, 0)).sort_values(["gate_violation", "crps", *keys], kind="stable").iloc[0]
        return {**base, "status": "failed", "reason": "No candidate passed the predeclared coverage gate",
                "best_effort_diagnostic_only": {key: str(closest[key]) for key in keys}}
    winner = passing.sort_values(["crps", "variant", "method", "calibration"], kind="stable").iloc[0]
    return {**base, "status": "passed", "configuration": {key: str(winner[key]) for key in keys},
        "source_metrics": {key: float(winner[key]) for key in ("crps", "empirical_coverage90", "empirical_width90", "empirical_interval_score90", "coverage_gap", "worst_region_delay_coverage90")}}


def score_phase(output):
    output = Path(output).resolve()
    if (output / "scores").exists():
        raise FileExistsError("Scores are immutable; use a new run")
    access = LabelAccess(output)
    commit = json.loads((output / "evaluation_commit.json").read_text())
    if sha(output / "evaluation_plan.json") != commit["plan_sha256"] or sha(output / "all_predictions_sealed.json") != commit["global_seal_sha256"]:
        raise ValueError("Evaluation plan or global seal changed")
    plan = json.loads((output / "evaluation_plan.json").read_text())
    frozen = verify_frozen_inputs(output)
    score_dir = output / "scores"
    score_dir.mkdir()
    # Open one region's labels only after the entire grid has passed global verification.
    summaries, counts, columns = [], [], None
    with (score_dir / "day_metrics.csv").open("x", newline="") as handle:
        writer = None
        for region in plan["targets"]:
            days, truths = daily_profiles(access.read(Path(plan["label_data"]) / f"{region}_hourly.csv", plan["year"]))
            by_day = {day.value: truth for day, truth in zip(days, truths)}
            for entry in access.seal["entries"]:
                if entry["region"] != region:
                    continue
                with np.load(output / entry["path"], allow_pickle=False) as stored:
                    archive = {key: stored[key] for key in ("origins", "point", "ensemble", "analytic_lower", "analytic_upper")}
                    metrics = []
                    for i, origin in enumerate(archive["origins"]):
                        truth = by_day.get(int(origin))
                        if truth is None:
                            continue
                        metric = score_day(archive["point"][i], archive["ensemble"][i], archive["analytic_lower"][i], archive["analytic_upper"][i], truth)
                        row = {key: entry[key] for key in ("region", "delay", "variant", "method", "calibration")}
                        row.update(origin=str(pd.Timestamp(int(origin), tz="UTC")), **metric)
                        if writer is None:
                            writer = csv.DictWriter(handle, fieldnames=list(row))
                            writer.writeheader()
                            columns = list(metric)
                        writer.writerow(row)
                        metrics.append(metric)
                if not metrics:
                    raise ValueError("No matching target outcomes for " + entry["path"])
                summary = {key: entry[key] for key in ("region", "delay", "variant", "method", "calibration")}
                summary.update(pd.DataFrame(metrics).mean().to_dict())
                summaries.append(summary)
                counts.append({**{key: entry[key] for key in ("region", "delay", "variant", "method", "calibration")}, "days": len(metrics)})
    regional = pd.DataFrame(summaries)
    if regional.empty:
        raise ValueError("No predictions to score")
    regional.to_csv(score_dir / "region_metrics.csv", index=False)
    pd.DataFrame(counts).to_csv(score_dir / "day_counts.csv", index=False)
    macro = regional.groupby(["delay", "variant", "method", "calibration"], sort=True)[columns].mean().reset_index()
    macro.to_csv(score_dir / "macro_metrics.csv", index=False)
    if plan["study"] == "development":
        selection = select_config(macro, regional)
        selection.update(model_closure_hashes=model_closure_hashes(frozen), environment=frozen["environment"],
            fit_seal_sha256=sha(output / "fit_seal.json"), fit_artifact_hashes=_verify_artifacts(output, "fit_seal.json")["artifacts"],
            prediction_global_seal_sha256=sha(output / "all_predictions_sealed.json"),
            region_metrics_sha256=sha(score_dir / "region_metrics.csv"), macro_metrics_sha256=sha(score_dir / "macro_metrics.csv"))
        write_json(score_dir / "selected_config.json", selection)
    else:
        fixed = plan["selection"]["configuration"]
        primary = macro.loc[np.logical_and.reduce([macro[key] == value for key, value in fixed.items()])]
        if len(primary) != len(set(macro.delay)):
            raise ValueError("Fixed source-selected configuration absent from holdout grid")
        primary.to_csv(score_dir / "fixed_primary_metrics.csv", index=False)
    verify_frozen_inputs(output)
    _verify_artifacts(output, "all_predictions_sealed.json")
    write_json(score_dir / "postrun_verification.json", {"all_frozen_hashes_match": True,
        "global_seal_sha256": access.global_digest, "target_label_access_after_global_seal": True,
        "score_hashes": {p.name: sha(p) for p in sorted(score_dir.glob("*.csv"))}})
    return macro


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("fit", "predict_seal", "score", "all"), default="all")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--external-data", type=Path)
    parser.add_argument("--study", choices=("development", "holdout"), default="development")
    parser.add_argument("--targets", nargs="+", default=[])
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--delays", nargs="+", type=int, default=list(DELAYS))
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--end")
    args = parser.parse_args()
    if args.phase in ("fit", "all"):
        if args.data is None:
            parser.error("--data required for fit")
        from transcif.research.transfer_v4 import VARIANTS
        variants = tuple(args.variants or (VARIANTS[-1],))
        if not variants or len(set(variants)) != len(variants) or not set(variants) <= set(VARIANTS):
            parser.error("Variants must be unique members of the fixed grid")
        if not args.delays or len(set(args.delays)) != len(args.delays) or not set(args.delays) <= set(DELAYS) or args.n_samples < 2:
            parser.error("Unique fixed-grid delays and at least two ensemble samples required")
        if args.study == "holdout" and (args.external_data is None or args.selection is None):
            parser.error("Holdout fit requires --external-data and --selection")
        if args.study == "development" and args.external_data is not None:
            parser.error("--external-data is reserved for the fixed holdout")
        fit_phase(args.data, args.output, variants, tuple(args.delays), args.n_samples,
                  external_data=args.external_data, selection=args.selection)
    if args.phase in ("predict_seal", "all"):
        predict_seal_phase(args.output, args.study, tuple(args.targets), args.selection, args.end)
    if args.phase in ("score", "all"):
        print(score_phase(args.output).to_string(index=False))


if __name__ == "__main__":
    main()
