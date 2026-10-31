"""Frozen V10 source development and seen-target information-value retest.

Prediction phases use released monthly data, lagged total-generation telemetry,
and optionally a separately frozen historical weather forecast proxy. Hourly
evaluation labels are parsed only after every stage prediction is sealed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from scripts.information_features_v10 import build_features, SOURCES, TARGETS
from scripts.information_models_v10 import InfoModel, SEED
from scripts.collect_information_weather_v10 import (BASE as WEATHER_API, MODEL as WEATHER_MODEL,
                                                     VARIABLES as WEATHER_VARIABLES,
                                                     UNITS as WEATHER_UNITS,
                                                     PUBLICATION_MARGIN_HOURS)
from transcif.research.dynamic_v2 import EF, read_year, daily_profiles

PROTOCOL = ROOT / "docs/INFORMATION_VALUE_V10_PROTOCOL.md"
V9_ROOT = ROOT / "results/coarse-dynamics-v9-rerun"
BUDGETS_G = ("B", "R", "S", "RS")
MODELS = ("ridge", "extra_trees")
LAGS = (7, 30)
FACTORS = (.75, 1., 1.25, 1.5, 2.)
N_SAMPLES = 128
METRICS = ("crps", "empirical_coverage90", "empirical_width90",
           "empirical_interval_score90", "empirical_lower_tail_miss",
           "empirical_upper_tail_miss", "empirical_path_coverage90",
           "point_level_mae", "point_shape_mae")
DAY_KEYS = ("stage", "region", "origin", "lag", "budget", "model", "factor", "semester")
REGION_KEYS = ("stage", "region", "lag", "budget", "model", "factor", "semester")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value: Any) -> None:
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def read(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def environment() -> dict[str, Any]:
    return {"python": sys.version, "executable": sys.executable,
            "packages": {name: importlib.metadata.version(name) for name in
                         ("numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl")}}


def _v9_background_paths() -> dict[str, str]:
    choices = read(V9_ROOT / "source_choices.json")["selections"]["joint"]
    expected = {"v9_standard_dynamic": (choices["standard_dynamic"]["configuration"],
                                        "harmonic_ridge--recompose", 1.25),
                "v9_simple_anchor": (choices["simple_anchor"]["configuration"],
                                     "harmonic_ridge--additive", 1.5)}
    for config, name, factor in expected.values():
        if config != {"round": 1, "name": name, "factor": factor}:
            raise ValueError("V9 background source choice changed")
    paths = {}
    for stage, regions in (("source", SOURCES), ("target", TARGETS)):
        seal = read(V9_ROOT / f"{stage}_prediction_seal.json")
        for region in regions:
            for label, (config, _, _) in expected.items():
                found = [entry for entry in seal["entries"]
                         if entry["region"] == region and int(entry["delay"]) == 0
                         and int(entry["round"]) == config["round"] and entry["name"] == config["name"]]
                if len(found) != 1:
                    raise ValueError("V9 background archive missing or duplicated")
                path = V9_ROOT / found[0]["path"]
                if sha(path) != found[0]["sha256"]:
                    raise ValueError("V9 background archive bytes changed")
                paths[f"{stage}/{region}/{label}"] = str(path.resolve())
    return paths


def _code_paths() -> list[Path]:
    return [Path(__file__).resolve(), ROOT / "scripts/information_features_v10.py",
            ROOT / "scripts/information_models_v10.py",
            ROOT / "scripts/collect_information_weather_v10.py",
            ROOT / "scripts/analyze_information_value_v10.py",
            ROOT / "scripts/coarse_dynamics_v9.py",
            PROTOCOL, *sorted((ROOT / "src/transcif/research").glob("*.py"))]


def freeze(output: Path, data: Path, *, weather_csv: Path | None = None,
           points_json: Path | None = None) -> dict[str, Any]:
    """Freeze model code, provenance and label bytes before either evaluation."""
    output, data = Path(output).resolve(), Path(data).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh immutable V10 output directory")
    if points_json is None:
        raise ValueError("A frozen shared points JSON is required for all V10 budgets")
    points_json = Path(points_json).resolve()
    weather_csv = Path(weather_csv).resolve() if weather_csv is not None else None
    weather_audit = weather_csv.parent / "metadata.json" if weather_csv is not None else None
    v9 = read(V9_ROOT / "prefit_freeze.json")
    source, target = Path(v9["source"]), Path(v9["target"])
    inputs = [data / "total_generation.csv", data / "audit.json", points_json,
              source / "monthly.csv", target / "monthly.csv",
              *[source / f"{r}_hourly.csv" for r in SOURCES],
              *[target / f"{r}_hourly.csv" for r in TARGETS]]
    if weather_csv is not None:
        inputs.extend((weather_csv, weather_audit,
                       *sorted((weather_csv.parent / "raw").rglob("*.json"))))
    old = [V9_ROOT / name for name in ("prefit_freeze.json", "source_choices.json",
           "source_prediction_seal.json", "target_prediction_seal.json",
           "source_completed.json", "target_completed.json")]
    old.extend(map(Path, _v9_background_paths().values()))
    code = _code_paths()
    if missing := [str(path) for path in (*inputs, *old, *code) if not path.is_file()]:
        raise FileNotFoundError("Missing frozen V10 dependency: " + ", ".join(missing))
    if read(data / "audit.json").get("publication_vintage_known") is not False:
        raise ValueError("V10 total-generation provenance must disclose unknown publication vintages")
    points = _points(points_json)
    weather = _weather(weather_csv) if weather_csv is not None else None
    if weather is not None and not set(SOURCES + TARGETS) <= set(weather.region):
        raise ValueError("Weather forecast proxy lacks registered regions")
    if weather is not None:
        weather_meta = read(weather_audit)
        if (weather_meta.get("points_sha256") != sha(points_json)
                or weather_meta.get("daily_weather_sha256") != sha(weather_csv)
                or weather_meta.get("target_hourly_labels_read") is not False
                or weather_meta.get("fixed_lead_hours") != 72
                or weather_meta.get("schema_version") != 1
                or weather_meta.get("model") != WEATHER_MODEL
                or weather_meta.get("api") != WEATHER_API
                or weather_meta.get("variable_ids") != WEATHER_VARIABLES
                or weather_meta.get("units") != WEATHER_UNITS
                or weather_meta.get("publication_margin_hours_assumed") != PUBLICATION_MARGIN_HOURS):
            raise ValueError("Weather proxy provenance or point registry mismatch")
        requests = weather_meta.get("raw_requests")
        if not isinstance(requests, list) or len(requests) != 27:
            raise ValueError("Weather raw-request manifest incomplete")
        raw_root = (weather_csv.parent / "raw").resolve()
        raw_paths = [(weather_csv.parent / item["raw_file"]).resolve() for item in requests]
        if (len(set(raw_paths)) != 27
                or any(not path.is_relative_to(raw_root) for path in raw_paths)
                or any(not path.is_file() or sha(path) != item["raw_sha256"]
                       for path, item in zip(raw_paths, requests))):
            raise ValueError("Weather raw response hash manifest mismatch")
    record = {"created_utc": now(), "source": str(source), "target": str(target),
              "data": str(data), "points_json": str(points_json),
              "weather_csv": str(weather_csv) if weather_csv else None,
              "weather_status": "historical_forecast_proxy" if weather_csv else "not_tested",
              "weather_audit": str(weather_audit) if weather_audit else None,
              "proper_start": "2024-01-25" if weather_csv else "2024-01-01",
              "sources": list(SOURCES), "targets": list(TARGETS),
              "budgets": list(_budgets(weather_csv is not None)), "models": list(MODELS),
              "lags": list(LAGS), "factors": list(FACTORS), "samples": N_SAMPLES,
              "seed": SEED, "coarse_extra_delay": 0,
              "source_choice_period": "2025H1 source LODO, telemetry lag 7d",
              "target_scope": "Previously viewed 2026H1 development regions",
              "target_training_performed": False, "target_selection_performed": False,
              "input_hashes": {str(p): sha(p) for p in sorted(set(inputs + old))},
              "code_hashes": {str(p): sha(p) for p in code},
              "v9_background_paths": _v9_background_paths(),
              "environment": environment(), "point_regions": sorted(points)}
    output.mkdir(parents=True)
    (output / "predictions").mkdir()
    write(output / "prefit_freeze.json", record)
    return record


def verify(output: Path) -> dict[str, Any]:
    frozen = read(Path(output) / "prefit_freeze.json")
    for path, digest in {**frozen["input_hashes"], **frozen["code_hashes"]}.items():
        if sha(Path(path)) != digest:
            raise ValueError("Frozen V10 bytes changed: " + path)
    if (frozen["sources"] != list(SOURCES) or frozen["targets"] != list(TARGETS)
            or frozen["budgets"] != list(_budgets(frozen["weather_csv"] is not None))
            or frozen["models"] != list(MODELS) or frozen["lags"] != list(LAGS)
            or frozen["factors"] != list(FACTORS) or frozen["samples"] != N_SAMPLES
            or frozen["seed"] != SEED or frozen["coarse_extra_delay"] != 0
            or frozen["environment"] != environment()):
        raise ValueError("Frozen V10 grid/environment changed")
    return frozen


def _budgets(weather: bool) -> tuple[str, ...]:
    return BUDGETS_G + (("W", "RW", "SW", "RSW") if weather else ())


def _points(path: Path) -> dict[str, tuple[float, float]]:
    raw = read(path)
    if not isinstance(raw, dict) or raw.get("schema_version") != 1 or not isinstance(raw.get("points"), list):
        raise ValueError("Points JSON needs the frozen version-1 point registry")
    registry = {item["region"]: item for item in raw["points"]}
    if len(registry) != len(raw["points"]) or set(registry) != set(SOURCES + TARGETS):
        raise ValueError("Points registry must contain each registered region once")
    result = {}
    for region in SOURCES + TARGETS:
        item = registry[region]
        lat, lon = item.get("latitude"), item.get("longitude")
        if not np.isfinite([lat, lon]).all() or not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("Invalid registered region coordinates")
        result[region] = (float(lat), float(lon))
    return result


def _weather(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    names = [f"weather_{kind}_{h:02d}" for kind in ("temp", "wind", "solar") for h in range(24)]
    needed = {"region", "origin", *names, "weather_complete", "weather_missing_values"}
    if not needed.issubset(table):
        raise ValueError("Weather CSV missing fixed wide forecast schema")
    table["origin"] = pd.to_datetime(table.origin, utc=True, errors="raise")
    if (table.origin.isna().any() or table.duplicated(["region", "origin"]).any()
            or not set(table.region) <= set(SOURCES + TARGETS)):
        raise ValueError("Weather region/origin grid invalid")
    values = table[names].to_numpy(float)
    if np.isinf(values).any():
        raise ValueError("Weather proxy contains infinity")
    return table


def _evaluation_days(stage: str, weather: bool) -> pd.DatetimeIndex:
    if stage not in {"source", "target"}:
        raise ValueError("Unknown V10 stage")
    year = 2025 if stage == "source" else 2026
    return pd.date_range(f"{year}-01-01", f"{year}-07-01", inclusive="left", tz="UTC")


def _source_training_days(source: Path, weather: bool) -> dict[str, dict[str, Any]]:
    """Read only 2024 source CIF labels before evaluation prediction seals."""
    records = {}
    proper_start = pd.Timestamp("2024-01-25" if weather else "2024-01-01", tz="UTC")
    for region in SOURCES:
        days, truth = daily_profiles(read_year(source / f"{region}_hourly.csv", 2024))
        proper = (days >= proper_start) & (days < pd.Timestamp("2024-07-01", tz="UTC"))
        calibration = (days >= pd.Timestamp("2024-07-01", tz="UTC")) & (
            days < pd.Timestamp("2025-01-01", tz="UTC"))
        if not proper.any() or not calibration.any():
            raise ValueError("Source proper/calibration cohort incomplete: " + region)
        records[region] = {"proper": (days[proper], truth[proper]),
                           "calibration": (days[calibration], truth[calibration])}
    return records


def _feature_record(monthly: pd.DataFrame, telemetry: pd.DataFrame,
                    weather: pd.DataFrame | None, points: dict[str, tuple[float, float]],
                    region: str, days: pd.DatetimeIndex, lag: int) -> dict[str, Any]:
    built = build_features(monthly, telemetry, region, days, lag_days=lag)
    for row in built["audit"]["rows"]:
        origin = pd.Timestamp(row["origin"])
        if (pd.Timestamp(row["max_hour_end_allowed"]) != origin - pd.Timedelta(days=lag)
                or (row["max_hour_end_used"] is not None and
                    pd.Timestamp(row["max_hour_end_used"]) > pd.Timestamp(row["max_hour_end_allowed"]))
                or (row["released_available_at"] is not None and
                    pd.Timestamp(row["released_available_at"]) > origin)
                or (row["released_period_end"] is not None and
                    pd.Timestamp(row["released_period_end"]) > origin)):
            raise ValueError("Feature availability exceeds V10 prediction origin")
    base = np.column_stack((built["base"], np.broadcast_to(points[region], (len(days), 2))))
    if not np.isfinite(base[:, :8]).all() or not np.allclose(base[:, :8].sum(axis=1), 1.0):
        raise ValueError("Harmonic baseline must be a finite fuel simplex")
    anchor = base[:, :8] @ EF
    weather_values = None
    if weather is not None:
        names = [f"weather_{kind}_{h:02d}" for kind in ("temp", "wind", "solar") for h in range(24)]
        part = weather.loc[weather.region.eq(region)].set_index("origin")
        if not pd.Index(days).isin(part.index).all():
            raise ValueError("Weather common cohort has missing region-day keys")
        matched = part.loc[days]
        weather_values = matched[[*names, "weather_complete", "weather_missing_values"]].to_numpy(float)
        if np.isinf(weather_values).any():
            raise ValueError("Weather feature matrix contains infinity")
    matrices = {}
    for budget in _budgets(weather is not None):
        sections = [base]
        if "R" in budget:
            sections.append(built["range"])
        if "S" in budget:
            sections.append(built["shape"])
        if "W" in budget:
            sections.append(weather_values)
        matrices[budget] = np.column_stack(sections)
    return {"days": days, "anchor": anchor, "features": matrices,
            "availability": built["audit"]}


def _feature_cache(frozen: dict[str, Any], stage: str,
                   source_training: dict[str, dict[str, Any]]) -> dict[tuple[int, str, str], dict[str, Any]]:
    source, target = Path(frozen["source"]), Path(frozen["target"])
    source_monthly, target_monthly = pd.read_csv(source / "monthly.csv"), pd.read_csv(target / "monthly.csv")
    telemetry = pd.read_csv(Path(frozen["data"]) / "total_generation.csv")
    weather = _weather(Path(frozen["weather_csv"])) if frozen["weather_csv"] else None
    points = _points(Path(frozen["points_json"]))
    cohort_weather = weather is not None
    cache = {}
    for lag in LAGS:
        for region in SOURCES:
            for split in ("proper", "calibration"):
                days, truth = source_training[region][split]
                record = _feature_record(source_monthly, telemetry, weather, points, region, days, lag)
                record["truth"] = truth
                cache[(lag, region, split)] = record
        for region in SOURCES if stage == "source" else TARGETS:
            monthly = source_monthly if stage == "source" else target_monthly
            days = _evaluation_days(stage, cohort_weather)
            cache[(lag, region, "evaluation")] = _feature_record(
                monthly, telemetry, weather, points, region, days, lag)
    return cache


def _fit_source_model(cache: dict, lag: int, budget: str, kind: str,
                      excluded: str | None) -> InfoModel:
    regions = [region for region in SOURCES if region != excluded]
    def gather(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        records = [cache[(lag, region, split)] for region in regions]
        return (np.concatenate([r["features"][budget] for r in records]),
                np.concatenate([r["truth"] for r in records]),
                np.concatenate([np.repeat(region, len(cache[(lag, region, split)]["days"]))
                                for region in regions]),
                np.concatenate([r["anchor"] for r in records]))
    return InfoModel(kind).fit(*gather("proper"), *gather("calibration"),
                               exclude_region=excluded)


def _save_prediction(output: Path, stage: str, region: str, lag: int,
                     budget: str, kind: str, record: dict[str, Any], model: InfoModel) -> dict[str, Any]:
    prediction = model.predict(record["features"][budget], record["anchor"],
                               record["days"], n_samples=N_SAMPLES)
    name = f"{stage}-{region}-lag{lag}-{budget}-{kind}.npz"
    path = output / "predictions" / name
    if path.exists():
        raise FileExistsError(path)
    np.savez_compressed(path, origins=record["days"].as_unit("ns").asi8,
                        point=prediction["point"], ensemble=prediction["ensemble"])
    return {"stage": stage, "region": region, "lag": lag, "budget": budget,
            "model": kind, "path": str(path.relative_to(output)), "sha256": sha(path),
            "rows": int(len(record["days"])), "training_audit": prediction["audit"],
            "feature_max_available_at_le_origin": True}


def _prediction_grid(frozen: dict[str, Any], stage: str) -> set[tuple[str, int, str, str]]:
    regions = SOURCES if stage == "source" else TARGETS
    return {(region, lag, budget, kind) for region in regions for lag in LAGS
            for budget in frozen["budgets"] for kind in MODELS}


def check_entries(output: Path, frozen: dict[str, Any], stage: str,
                  seal: dict[str, Any]) -> None:
    entries = seal.get("entries", [])
    actual = [(e["region"], int(e["lag"]), e["budget"], e["model"]) for e in entries]
    if len(actual) != len(set(actual)) or set(actual) != _prediction_grid(frozen, stage):
        raise ValueError("Incomplete or duplicated V10 prediction grid")
    days = _evaluation_days(stage, frozen["weather_csv"] is not None)
    expected_origins = days.as_unit("ns").asi8
    for entry in entries:
        path = (output / entry["path"]).resolve()
        if not path.is_relative_to((output / "predictions").resolve()) or sha(path) != entry["sha256"]:
            raise ValueError("V10 prediction archive path/hash changed")
        if entry["rows"] != len(days) or entry.get("feature_max_available_at_le_origin") is not True:
            raise ValueError("V10 prediction archive availability/rows changed")
        audit = entry["training_audit"]
        allowed = set(SOURCES) - ({entry["region"]} if stage == "source" else set())
        if (set(audit["proper_regions"]) != allowed or set(audit["calibration_regions"]) != allowed
                or audit["excluded_evaluation_region"] != (entry["region"] if stage == "source" else None)):
            raise ValueError("V10 source proper/calibration region exclusion changed")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"origins", "point", "ensemble"}:
                raise ValueError("V10 archive schema changed")
            if (not np.array_equal(archive["origins"], expected_origins)
                    or archive["point"].shape != (len(days), 24)
                    or archive["ensemble"].shape != (len(days), N_SAMPLES, 24)
                    or not np.isfinite(archive["point"]).all()
                    or not np.isfinite(archive["ensemble"]).all()
                    or np.any((archive["ensemble"] < 0) | (archive["ensemble"] > 1000))
                    or np.any((archive["point"] < 0) | (archive["point"] > 1000))):
                raise ValueError("V10 archive shape/origins/bounds changed")


def score_arrays(point: np.ndarray, ensemble: np.ndarray,
                 truth: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized per-day empirical proper/interval and point diagnostics."""
    point, ensemble, truth = (np.asarray(value, float) for value in (point, ensemble, truth))
    if (point.ndim != 2 or point.shape[1] != 24 or truth.shape != point.shape
            or ensemble.ndim != 3 or ensemble.shape[::2] != point.shape
            or ensemble.shape[1] < 2 or not np.isfinite(point).all()
            or not np.isfinite(ensemble).all() or not np.isfinite(truth).all()):
        raise ValueError("Score arrays need finite days×24 points/truth and days×draws×24 paths")
    draws = ensemble.shape[1]
    ordered = np.sort(ensemble, axis=1)
    coefficients = 2 * np.arange(draws, dtype=float) - draws + 1
    first = np.abs(ensemble - truth[:, None, :]).mean(axis=1)
    pair = np.einsum("s,dsh->dh", coefficients, ordered) / (draws * draws)
    crps = (first - pair).mean(axis=1)
    lower = ordered[:, int(np.ceil(.05 * draws)) - 1, :]
    upper = ordered[:, int(np.ceil(.95 * draws)) - 1, :]
    below, above = truth < lower, truth > upper
    width = upper - lower
    interval = width + 20 * np.where(below, lower - truth, 0.) + 20 * np.where(above, truth - upper, 0.)
    centered_point = point - point.mean(axis=1, keepdims=True)
    centered_truth = truth - truth.mean(axis=1, keepdims=True)
    return {"crps": crps,
            "empirical_coverage90": (~(below | above)).mean(axis=1),
            "empirical_width90": width.mean(axis=1),
            "empirical_interval_score90": interval.mean(axis=1),
            "empirical_lower_tail_miss": below.mean(axis=1),
            "empirical_upper_tail_miss": above.mean(axis=1),
            "empirical_path_coverage90": (~(below | above)).all(axis=1).astype(float),
            "point_level_mae": np.abs(point.mean(axis=1) - truth.mean(axis=1)),
            "point_shape_mae": np.abs(centered_point - centered_truth).mean(axis=1)}


def _scored_rows(stage: str, region: str, lag: int, budget: str, kind: str,
                 factor: float, origins: np.ndarray, point: np.ndarray,
                 ensemble: np.ndarray, truth: np.ndarray) -> list[dict[str, Any]]:
    center = ensemble.mean(axis=1, keepdims=True)
    scaled = np.clip(center + factor * (ensemble - center), 0.0, 1000.0)
    metrics = score_arrays(point, scaled, truth)
    return [{"stage": stage, "region": region,
             "origin": pd.Timestamp(int(origin), tz="UTC").isoformat(),
             "lag": int(lag), "budget": budget, "model": kind, "factor": float(factor),
             "semester": "H1" if pd.Timestamp(int(origin), tz="UTC").month <= 6 else "H2",
             **{name: float(values[i]) for name, values in metrics.items()}}
            for i, origin in enumerate(origins)]


def _truth_lookup(data: Path, stage: str, region: str) -> tuple[np.ndarray, np.ndarray]:
    year = 2025 if stage == "source" else 2026
    days, truths = daily_profiles(read_year(data / f"{region}_hourly.csv", year))
    if not len(days):
        raise ValueError("No valid evaluation CIF days: " + region)
    return days.as_unit("ns").asi8, truths


def _match_truth(origins: np.ndarray, truth_days: np.ndarray,
                 truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = pd.Index(truth_days).get_indexer(origins)
    keep = positions >= 0
    if not keep.any():
        raise ValueError("No matching valid evaluation days")
    return keep, truth[positions[keep]]


def _v9_background_rows(output: Path, frozen: dict[str, Any], stage: str,
                        region: str, truth_days: np.ndarray,
                        truth: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for label, factor in (("v9_standard_dynamic", 1.25), ("v9_simple_anchor", 1.5)):
        path = Path(frozen["v9_background_paths"][f"{stage}/{region}/{label}"])
        if sha(path) != frozen["input_hashes"][str(path)]:
            raise ValueError("V9 historical background archive changed")
        with np.load(path, allow_pickle=False) as archive:
            origins = archive["origins"]
            keep, aligned = _match_truth(origins, truth_days, truth)
            cohort = _evaluation_days(stage, frozen["weather_csv"] is not None)
            in_cohort = np.isin(origins, cohort.as_unit("ns").asi8)
            keep &= in_cohort
            if not keep.any():
                raise ValueError("V9 background has no common V10 cohort days")
            lookup = pd.Index(truth_days).get_indexer(origins[keep])
            aligned = truth[lookup]
            rows.extend(_scored_rows(stage, region, 0, "V9_background", label, factor,
                                     origins[keep], archive["point"][keep],
                                     archive["ensemble"][keep], aligned))
    return rows


def score_phase(output: Path, frozen: dict[str, Any], stage: str,
                seal: dict[str, Any]) -> pd.DataFrame:
    """Parse evaluation CIF only after the complete stage prediction seal."""
    seal_path = output / f"{stage}_prediction_seal.json"
    if not seal_path.is_file() or read(seal_path) != seal:
        raise ValueError("Prediction seal required before reading evaluation labels")
    check_entries(output, frozen, stage, seal)
    folder = output / f"{stage}_scores"
    folder.mkdir()
    rows = []
    entries_by_region = {region: [entry for entry in seal["entries"] if entry["region"] == region]
                         for region in (SOURCES if stage == "source" else TARGETS)}
    data = Path(frozen["source"] if stage == "source" else frozen["target"])
    for region, entries in entries_by_region.items():
        truth_days, truth = _truth_lookup(data, stage, region)
        for entry in entries:
            path = output / entry["path"]
            if sha(path) != entry["sha256"]:
                raise ValueError("Prediction archive changed before evaluation")
            with np.load(path, allow_pickle=False) as archive:
                origins = archive["origins"]
                keep, aligned = _match_truth(origins, truth_days, truth)
                for factor in FACTORS:
                    rows.extend(_scored_rows(stage, region, entry["lag"], entry["budget"],
                                             entry["model"], factor, origins[keep],
                                             archive["point"][keep], archive["ensemble"][keep], aligned))
        rows.extend(_v9_background_rows(output, frozen, stage, region, truth_days, truth))
    day = pd.DataFrame(rows)
    if day.empty or day.duplicated(list(DAY_KEYS)).any() or not np.isfinite(day[list(METRICS)]).all().all():
        raise ValueError("V10 day score grid empty, duplicated, or nonfinite")
    day.to_csv(folder / "day_metrics.csv", index=False)
    grouped = day.groupby(list(REGION_KEYS), sort=True)
    region = grouped[list(METRICS)].mean().reset_index()
    region["days"] = grouped.size().to_numpy()
    region.to_csv(folder / "region_metrics.csv", index=False)
    macro = region.groupby(["stage", "lag", "budget", "model", "factor", "semester"], sort=True)[list(METRICS)].mean().reset_index()
    macro.to_csv(folder / "macro_metrics.csv", index=False)
    write(folder / "score_seal.json", {p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()})
    return region


def selection(region_table: pd.DataFrame, budgets: tuple[str, ...]) -> dict[str, Any]:
    """Choose every budget and one added-information primary from source lag 7 only."""
    source = region_table.loc[(region_table.stage == "source") &
                              (region_table.lag == 7) &
                              (region_table.semester == "H1") &
                              region_table.budget.isin(budgets)]
    expected = {(region, budget, kind, float(factor)) for region in SOURCES
                for budget in budgets for kind in MODELS for factor in FACTORS}
    observed = {(row.region, row.budget, row.model, float(row.factor)) for _, row in source.iterrows()}
    if len(source) != len(expected) or observed != expected:
        raise ValueError("Source H1 lag-7 selection grid incomplete")
    grouped = source.groupby(["budget", "model", "factor"], sort=True)
    ranking = grouped[list(METRICS)].mean().reset_index()
    ranking["worst_coverage"] = grouped.empirical_coverage90.min().to_numpy()
    ranking["violation"] = (np.maximum(.90 - ranking.empirical_coverage90, 0.)
                            + np.maximum(ranking.empirical_coverage90 - .95, 0.)
                            + np.maximum(.85 - ranking.worst_coverage, 0.))
    def choose(frame: pd.DataFrame) -> dict[str, Any]:
        passed = frame.loc[frame.violation <= 1e-12]
        candidates = passed if len(passed) else frame
        ordering = ([] if len(passed) else ["violation"]) + ["crps", "budget", "model", "factor"]
        row = candidates.sort_values(ordering).iloc[0]
        return {"budget": str(row.budget), "model": str(row.model),
                "factor": float(row.factor),
                "status": "passed" if row.violation <= 1e-12 else "failed",
                "source_metrics": {name: float(row[name]) for name in
                                   ("crps", "empirical_coverage90", "empirical_width90",
                                    "empirical_interval_score90", "worst_coverage", "violation")}}
    selections = {budget: choose(ranking.loc[ranking.budget.eq(budget)]) for budget in budgets}
    primary = choose(ranking.loc[ranking.budget.ne("B")])
    return {"selections": selections, "primary": primary}


def check_score_grid(output: Path, frozen: dict[str, Any], stage: str) -> None:
    folder = output / f"{stage}_scores"
    day, region, macro = (pd.read_csv(folder / name) for name in
                          ("day_metrics.csv", "region_metrics.csv", "macro_metrics.csv"))
    if (day.empty or day.duplicated(list(DAY_KEYS)).any() or
            region.duplicated(list(REGION_KEYS)).any() or
            not np.isfinite(day[list(METRICS)]).all().all() or
            not np.isfinite(region[list(METRICS)]).all().all() or
            not np.isfinite(macro[list(METRICS)]).all().all()):
        raise ValueError("V10 score grid duplicated or nonfinite")
    regions = SOURCES if stage == "source" else TARGETS
    expected = {(stage, r, lag, budget, model, float(factor), "H1")
                for r in regions for lag in LAGS for budget in frozen["budgets"]
                for model in MODELS for factor in FACTORS}
    expected |= {(stage, r, 0, "V9_background", model, factor, "H1")
                 for r in regions for model, factor in
                 (("v9_standard_dynamic", 1.25), ("v9_simple_anchor", 1.5))}
    actual = {(row.stage, row.region, int(row.lag), row.budget, row.model,
               float(row.factor), row.semester) for _, row in region.iterrows()}
    if len(region) != len(expected) or actual != expected:
        raise ValueError("V10 region score grid incomplete")
    # Every information arm must be scored on identical valid outcome days.
    for r in regions:
        reference = set(day.loc[day.region.eq(r) & day.lag.eq(7) &
                                day.budget.eq("B") & day.model.eq("ridge") &
                                day.factor.eq(1.), "origin"])
        if not reference:
            raise ValueError("No valid evaluation day for " + r)
        for (_, lag, budget, model, factor), group in day.loc[day.region.eq(r)].groupby(
                ["region", "lag", "budget", "model", "factor"], sort=True):
            if set(group.origin) != reference or len(group) != len(reference):
                raise ValueError("V10 information budgets do not share scoring support")
    grouped = day.groupby(list(REGION_KEYS), sort=True)
    expected_region = grouped[list(METRICS)].mean().reset_index()
    expected_region["days"] = grouped.size().to_numpy()
    keys = list(REGION_KEYS)
    joined = region.merge(expected_region, on=keys, suffixes=("_sealed", "_computed"), validate="one_to_one")
    if (len(joined) != len(region) or not np.allclose(
            joined[[m + "_sealed" for m in (*METRICS, "days")]].to_numpy(float),
            joined[[m + "_computed" for m in (*METRICS, "days")]].to_numpy(float), atol=1e-10)):
        raise ValueError("V10 region means/counts do not reproduce day scores")
    macro_keys = ["stage", "lag", "budget", "model", "factor", "semester"]
    expected_macro = region.groupby(macro_keys, sort=True)[list(METRICS)].mean().reset_index()
    joined = macro.merge(expected_macro, on=macro_keys, suffixes=("_sealed", "_computed"), validate="one_to_one")
    if (len(joined) != len(macro) or not np.allclose(
            joined[[m + "_sealed" for m in METRICS]].to_numpy(float),
            joined[[m + "_computed" for m in METRICS]].to_numpy(float), atol=1e-10)):
        raise ValueError("V10 equal-region macro does not reproduce region scores")


def validate_choices(output: Path, frozen: dict[str, Any]) -> dict[str, Any]:
    choice = read(output / "source_choices.json")
    if (choice.get("target_scores_used") is not False or
            choice.get("source_h2_used_for_selection") is not False or
            choice.get("selection_lag") != 7 or
            choice.get("source_prediction_seal_sha256") != sha(output / "source_prediction_seal.json") or
            choice.get("source_score_seal_sha256") != sha(output / "source_scores/score_seal.json")):
        raise ValueError("V10 source choice information boundary changed")
    actual = selection(pd.read_csv(output / "source_scores/region_metrics.csv"),
                       tuple(frozen["budgets"]))
    if actual != {"selections": choice["selections"], "primary": choice["primary"]}:
        raise ValueError("V10 source choices do not reproduce source H1")
    return choice


def verify_phase(output: Path, stage: str) -> dict[str, Any]:
    if stage not in {"source", "target"}:
        raise ValueError("Unknown V10 phase")
    output = Path(output).resolve()
    frozen = verify(output)
    done = read(output / f"{stage}_completed.json")
    seal_path = output / f"{stage}_prediction_seal.json"
    if (done.get("prefit_freeze_sha256") != sha(output / "prefit_freeze.json") or
            done.get("prediction_seal_sha256") != sha(seal_path)):
        raise ValueError("V10 completion detached from freeze/prediction seal")
    seal = read(seal_path)
    if (seal.get("evaluation_hourly_labels_read_preseal") is not False or
            seal.get("complete_grid") is not True or
            seal.get("weather_status") != frozen["weather_status"] or
            seal.get("telemetry_publication_vintage_known") is not False):
        raise ValueError("V10 prediction information boundary changed")
    check_entries(output, frozen, stage, seal)
    folder = output / f"{stage}_scores"
    if done.get("score_seal_sha256") != sha(folder / "score_seal.json"):
        raise ValueError("V10 score seal detached from completion")
    scores = read(folder / "score_seal.json")
    if set(scores) != {"day_metrics.csv", "region_metrics.csv", "macro_metrics.csv"}:
        raise ValueError("V10 score manifest incomplete")
    for name, digest in scores.items():
        if sha(folder / name) != digest:
            raise ValueError("V10 score bytes changed")
    check_score_grid(output, frozen, stage)
    if (done.get("target_selection_performed") is not False or
            done.get("target_training_performed") is not False or
            pd.Timestamp(seal["created_utc"]) > pd.Timestamp(done["created_utc"])):
        raise ValueError("V10 completion information boundary changed")
    choice = validate_choices(output, frozen)
    if done.get("source_choices_sha256") != sha(output / "source_choices.json"):
        raise ValueError("V10 completion detached from source choice")
    if stage == "target":
        verify_phase(output, "source")
        if (seal.get("source_choices_sha256") != sha(output / "source_choices.json") or
                pd.Timestamp(choice["created_utc"]) > pd.Timestamp(seal["created_utc"])):
            raise ValueError("V10 target prediction detached from prior source choice")
    return seal


def run_phase(output: Path, data: Path, stage: str, *,
              weather_csv: Path | None = None, points_json: Path | None = None) -> None:
    """Run one immutable phase; the first source call may create the prefit freeze."""
    output, data = Path(output).resolve(), Path(data).resolve()
    if stage not in {"source", "target"}:
        raise ValueError("Unknown V10 phase")
    if not output.exists():
        if stage != "source":
            raise FileNotFoundError("Source phase must precede target")
        freeze(output, data, weather_csv=weather_csv, points_json=points_json)
    frozen = verify(output)
    if Path(frozen["data"]) != data:
        raise ValueError("V10 data path differs from prefit freeze")
    if weather_csv is not None and Path(weather_csv).resolve() != Path(frozen["weather_csv"]):
        raise ValueError("Weather path differs from prefit freeze")
    if points_json is not None and Path(points_json).resolve() != Path(frozen["points_json"]):
        raise ValueError("Points path differs from prefit freeze")
    if stage == "target":
        verify_phase(output, "source")
    if any((output / f"{stage}_{suffix}").exists() for suffix in
           ("prediction_seal.json", "scores", "completed.json")):
        raise FileExistsError("V10 stage is immutable")
    training = _source_training_days(Path(frozen["source"]), frozen["weather_csv"] is not None)
    cache = _feature_cache(frozen, stage, training)
    entries = []
    if stage == "source":
        for lag in LAGS:
            for region in SOURCES:
                for budget in frozen["budgets"]:
                    for kind in MODELS:
                        model = _fit_source_model(cache, lag, budget, kind, region)
                        entries.append(_save_prediction(output, stage, region, lag, budget, kind,
                                                        cache[(lag, region, "evaluation")], model))
                print("Predicted V10 source", region, lag, flush=True)
    else:
        for lag in LAGS:
            for budget in frozen["budgets"]:
                for kind in MODELS:
                    model = _fit_source_model(cache, lag, budget, kind, None)
                    for region in TARGETS:
                        entries.append(_save_prediction(output, stage, region, lag, budget, kind,
                                                        cache[(lag, region, "evaluation")], model))
            print("Predicted V10 target lag", lag, flush=True)
    verify(output)
    seal = {"created_utc": now(), "entries": entries, "complete_grid": True,
            "evaluation_hourly_labels_read_preseal": False,
            "source_2024_proper_calibration_labels_only_preseal": True,
            "telemetry_publication_vintage_known": False,
            "weather_status": frozen["weather_status"],
            "weather_issued_public_certified": False,
            "source_choices_sha256": sha(output / "source_choices.json") if stage == "target" else None}
    check_entries(output, frozen, stage, seal)
    write(output / f"{stage}_prediction_seal.json", seal)
    region_table = score_phase(output, frozen, stage, seal)
    if stage == "source":
        chosen = selection(region_table, tuple(frozen["budgets"]))
        write(output / "source_choices.json", {"created_utc": now(), **chosen,
              "selection_lag": 7, "target_scores_used": False,
              "source_h2_used_for_selection": False,
              "source_prediction_seal_sha256": sha(output / "source_prediction_seal.json"),
              "source_score_seal_sha256": sha(output / "source_scores/score_seal.json")})
    write(output / f"{stage}_completed.json", {"created_utc": now(),
          "prefit_freeze_sha256": sha(output / "prefit_freeze.json"),
          "prediction_seal_sha256": sha(output / f"{stage}_prediction_seal.json"),
          "score_seal_sha256": sha(output / f"{stage}_scores/score_seal.json"),
          "source_choices_sha256": sha(output / "source_choices.json"),
          "target_selection_performed": False, "target_training_performed": False})
    verify_phase(output, stage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("source", "target"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "results/information-value-v10-data")
    parser.add_argument("--weather-csv", type=Path)
    parser.add_argument("--points-json", type=Path)
    args = parser.parse_args()
    run_phase(args.output, args.data, args.phase,
              weather_csv=args.weather_csv, points_json=args.points_json)


if __name__ == "__main__":
    main()
