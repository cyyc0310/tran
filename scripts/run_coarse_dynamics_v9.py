"""Frozen V9 source development and seen-target retest for lawful coarse dynamics.

Prediction archives, including the coarse forecasts, are sealed before either
stage reads its hourly outcome files. Target roles are selected on source H1.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    try:
        spec.loader.exec_module(value)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return value


A7 = module("v9_prior_audit", ROOT / "scripts/analyze_fuel_transport_v7.py")
V7 = A7.RUNNER
from transcif.research.dynamic_v2 import EF
COARSE_PATH = ROOT / "scripts/coarse_dynamics_v9.py"
RISK_PATH = ROOT / "scripts/risk_paths_v9.py"
PROTOCOL = ROOT / "docs/COARSE_DYNAMICS_V9_PROTOCOL.md"
sha, write = V7.sha, V7.write
SCORE_FILES = A7.SCORE_FILES
COARSE_SCORE_FILES = {"day_metrics.csv", "monthly_metrics.csv", "macro_metrics.csv", "month_support.csv"}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def risk():
    return module("v9_risk_paths", RISK_PATH)


def coarse():
    return module("v9_coarse_forecast", COARSE_PATH)


def configurations() -> list[dict[str, Any]]:
    values = risk().configs()
    keys = [(int(c["round"]), c["name"]) for c in values]
    if not values or len(keys) != len(set(keys)):
        raise ValueError("V9 configurations are empty or not unique by round/name")
    return values


def _dependencies(upstream, old):
    result = {**old["code"], **old["inputs"]}
    for name in ("prefit_freeze.json", "source_choices.json", "source_prediction_seal.json",
                 "target_prediction_seal.json", "completed.json"):
        path = upstream / name
        result[str(path)] = sha(path)
    for folder in ("fits", "source_scores", "target_scores"):
        for path in sorted((upstream / folder).iterdir()):
            if path.is_file():
                result[str(path)] = sha(path)
    return result


def freeze(output: Path, upstream: Path) -> dict[str, Any]:
    output, upstream = Path(output).resolve(), Path(upstream).resolve()
    if output.exists():
        raise FileExistsError("Use a new V9 output directory")
    prior = A7.verify_run(upstream)
    old = prior["frozen"]
    code = [Path(__file__).resolve(), ROOT / "scripts/analyze_coarse_dynamics_v9.py",
            COARSE_PATH, RISK_PATH, PROTOCOL,
            Path(A7.__file__).resolve()]
    for path in code:
        if not path.is_file():
            raise FileNotFoundError(path)
    record = {"created_utc": now(), "upstream": str(upstream),
              "source": old["source"], "target": old["target"],
              "inputs": old["inputs"], "dependencies": _dependencies(upstream, old),
              "code": {str(p): sha(p) for p in code}, "configs": configurations(),
              "coarse_methods": ["latest", "seasonal", "v7_mixture", "harmonic_ridge",
                                 "seasonal_kalman", "release_bridge"],
              "delays": list(V7.DELAYS), "factors": list(V7.FACTORS),
              "samples": V7.N_SAMPLES, "seed": V7.SEED,
              "environment": V7.environment(),
              "source_choice_period": "2025H1 LODO", "source_stress_period": "2025H2 LODO",
              "target_scope": "Previously viewed 2026H1 regions; development retest only",
              "no_new_target_training": True, "hourly_labels_after_stage_seal": True}
    output.mkdir(parents=True)
    (output / "predictions").mkdir()
    (output / "coarse_predictions").mkdir()
    write(output / "prefit_freeze.json", record)
    return record


def verify(output: Path) -> dict[str, Any]:
    output = Path(output).resolve()
    frozen = read(output / "prefit_freeze.json")
    for path, digest in {**frozen["dependencies"], **frozen["code"]}.items():
        if sha(path) != digest:
            raise ValueError("Frozen bytes changed: " + path)
    if (frozen["configs"] != configurations() or frozen["delays"] != list(V7.DELAYS)
            or frozen["factors"] != list(V7.FACTORS) or frozen["samples"] != V7.N_SAMPLES
            or frozen["seed"] != V7.SEED or frozen["environment"] != V7.environment()
            or frozen["coarse_methods"] != ["latest", "seasonal", "v7_mixture", "harmonic_ridge",
                                            "seasonal_kalman", "release_bridge"]
            or frozen.get("no_new_target_training") is not True):
        raise ValueError("Frozen V9 grid/environment changed")
    return frozen


def _family(config):
    family = str(config.get("family", ""))
    if family not in ("proposed", "standard"):
        raise ValueError("Each V9 config needs proposed/standard family")
    return family


def selection(region_table: pd.DataFrame, grid: list[dict[str, Any]] | None = None) -> tuple[dict[str, Any], pd.DataFrame]:
    """Select fixed roles from equal region/delay 2025H1 means only."""
    grid = configurations() if grid is None else grid
    lookup = {(int(c["round"]), c["name"]): c for c in grid}
    source = region_table[region_table.semester.eq("H1")].copy()
    expected = {(r, d, *k) for r in V7.SOURCES for d in V7.DELAYS for k in lookup}
    observed = {(r.region, int(r.delay), int(r["round"]), r["name"]) for _, r in source.iterrows() if float(r.factor) == 1.}
    if not expected.issubset(observed):
        raise ValueError("Source H1 selection grid incomplete")
    grouped = source.groupby(["round", "name", "mode", "factor"], sort=True)
    ranking = grouped.mean(numeric_only=True).drop(columns=["delay", "days"], errors="ignore").reset_index()
    ranking["worst_coverage"] = grouped.empirical_coverage90.min().to_numpy()
    ranking["family"] = [_family(lookup[(int(row["round"]), row["name"])]) for _, row in ranking.iterrows()]
    ranking["simple_anchor"] = [bool(lookup[(int(row["round"]), row["name"])].get("simple_anchor", False)) for _, row in ranking.iterrows()]
    ranking["standard_dynamic"] = [bool(lookup[(int(row["round"]), row["name"])].get("standard_dynamic", False)) for _, row in ranking.iterrows()]
    ranking["violation"] = (np.maximum(.9 - ranking.empirical_coverage90, 0)
                            + np.maximum(ranking.empirical_coverage90 - .95, 0)
                            + np.maximum(.85 - ranking.worst_coverage, 0))
    def chosen(frame):
        if frame.empty:
            raise ValueError("Missing required source candidate family")
        passed = frame[frame.violation.le(1e-12)]
        keys = ([] if len(passed) else ["violation"]) + ["crps", "round", "name", "factor"]
        row = (passed if len(passed) else frame).sort_values(keys).iloc[0]
        return {"configuration": {"round": int(row["round"]), "name": row["name"],
                                  "factor": float(row.factor)},
                "family": row.family, "status": "passed" if row.violation <= 1e-12 else "failed",
                "metrics": {k: float(row[k]) for k in
                            ("crps", "empirical_coverage90", "empirical_width90",
                             "empirical_interval_score90", "worst_coverage", "violation")}}
    roles = {}
    for scope in [*sorted({int(c["round"]) for c in grid}), "joint"]:
        part = ranking if scope == "joint" else ranking[ranking["round"].eq(scope)]
        proposed = part[part.family.eq("proposed")]
        standard = part[part.family.eq("standard")]
        roles[str(scope)] = {"primary": chosen(proposed), "baseline": chosen(standard),
                             "simple_anchor": chosen(part[part.simple_anchor]),
                             "standard_dynamic": chosen(part[part.standard_dynamic])}
    return roles, ranking


def validate_choices(output: Path, choice: dict[str, Any]) -> None:
    if choice.get("target_scores_used") is not False or choice.get("source_h2_used_for_selection") is not False:
        raise ValueError("Source choice information boundary changed")
    if choice["source_seal_sha256"] != sha(output / "source_prediction_seal.json"):
        raise ValueError("Source prediction seal changed")
    folder = output / "source_scores"
    if set(choice["source_score_hashes"]) != SCORE_FILES | {"ranking.csv", "score_seal.json"}:
        raise ValueError("Incomplete source score manifest")
    for name, digest in choice["source_score_hashes"].items():
        if sha(folder / name) != digest:
            raise ValueError("Source score changed")
    actual, _ = selection(pd.read_csv(folder / "region_metrics.csv"))
    if actual != choice["selections"]:
        raise ValueError("Source choice does not reproduce")


def load_fit(frozen: dict[str, Any], region: str, delay: int, stage: str) -> dict[str, Any]:
    upstream = Path(frozen["upstream"])
    name = f"lodo-{region}-{delay}.pkl" if stage == "source" else f"full-{delay}.pkl"
    path = upstream / "fits" / name
    if sha(path) != frozen["dependencies"][str(path)]:
        raise ValueError("Fitted V7 bank changed")
    with path.open("rb") as handle:
        bundle = pickle.load(handle)
    expected = set(V7.SOURCES) - ({region} if stage == "source" else set())
    if set(bundle["training_regions"]) != expected or set(bundle["bank"].source_regions_) != expected:
        raise ValueError("Wrong LODO/full donor bank")
    return bundle


def _audit(value):
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
    if isinstance(value, dict):
        return {str(k): _audit(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_audit(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _check_archive(path, days, coarse_archive=False):
    with np.load(path, allow_pickle=False) as a:
        if set(a.files) != ({"origins", "mix", "draws"} if coarse_archive else {"origins", "point", "ensemble"}):
            raise ValueError("Prediction archive schema changed")
        if not np.array_equal(a["origins"], pd.DatetimeIndex(days).as_unit("ns").asi8):
            raise ValueError("Prediction origins changed")
        n = len(days)
        expected = ((n, 8), (n, V7.N_SAMPLES, 8)) if coarse_archive else ((n, 24), (n, V7.N_SAMPLES, 24))
        keys = ("mix", "draws") if coarse_archive else ("point", "ensemble")
        for key, shape in zip(keys, expected):
            arr = a[key]
            if arr.shape != shape or not np.isfinite(arr).all():
                raise ValueError("Prediction values invalid")
            if coarse_archive and (np.any(arr < -1e-9) or not np.allclose(arr.sum(-1), 1, atol=1e-7)):
                raise ValueError("Coarse forecast is not a fuel simplex")


def check_entries(output: Path, frozen: dict[str, Any], stage: str, seal: dict[str, Any]) -> None:
    regions = V7.SOURCES if stage == "source" else V7.TARGETS
    grid = {(int(c["round"]), c["name"]): c for c in frozen["configs"]}
    expected = {(r, d, *k) for r in regions for d in V7.DELAYS for k in grid}
    seen, paths = set(), set()
    days = pd.date_range("2025-01-01" if stage == "source" else "2026-01-01",
                         "2026-01-01" if stage == "source" else "2026-07-01",
                         inclusive="left", tz="UTC")
    for entry in seal["entries"]:
        key = entry["region"], int(entry["delay"]), int(entry["round"]), entry["name"]
        path = (output / entry["path"]).resolve()
        if (key in seen or path in paths or key not in expected or not path.is_relative_to(output.resolve())
                or not path.is_relative_to((output / "predictions").resolve())
                or any(entry.get(k) != v for k, v in grid[key[2:]].items())
                or entry.get("interval_only") is not False or sha(path) != entry["sha256"]):
            raise ValueError("Incomplete, changed or aliased hourly prediction grid")
        _check_archive(path, days)
        seen.add(key)
        paths.add(path)
    if seen != expected or seal.get("complete_grid") is not True:
        raise ValueError("Incomplete hourly prediction grid")
    coarse_expected = {(r, d, m) for r in regions for d in V7.DELAYS for m in frozen["coarse_methods"]}
    seen, paths = set(), set()
    for entry in seal["coarse_entries"]:
        key = entry["region"], int(entry["delay"]), entry["method"]
        path = (output / entry["path"]).resolve()
        if (key in seen or path in paths or key not in coarse_expected or not path.is_relative_to((output / "coarse_predictions").resolve())
                or sha(path) != entry["sha256"]):
            raise ValueError("Incomplete or changed coarse forecast grid")
        _check_archive(path, days, coarse_archive=True)
        seen.add(key)
        paths.add(path)
    if seen != coarse_expected or seal.get("coarse_complete_grid") is not True:
        raise ValueError("Incomplete coarse forecast grid")


def check_score_grid(output: Path, frozen: dict[str, Any], stage: str) -> None:
    table = pd.read_csv(output / f"{stage}_scores/region_metrics.csv")
    regions = V7.SOURCES if stage == "source" else V7.TARGETS
    semesters = ("H1", "H2") if stage == "source" else ("H1",)
    expected = {(region, delay, int(config["round"]), config["name"], float(factor), semester)
                for region in regions for delay in V7.DELAYS for config in frozen["configs"]
                for factor in V7.FACTORS for semester in semesters}
    found = [(row.region, int(row.delay), int(row["round"]), row["name"],
              float(row.factor), row.semester) for _, row in table.iterrows()]
    if (len(found) != len(set(found)) or set(found) != expected or table.days.le(0).any()
            or not np.isfinite(table[["crps", "empirical_coverage90", "empirical_width90",
                                      "empirical_interval_score90"]]).all().all()):
        raise ValueError("Incomplete or invalid V9 scoring grid")


def check_coarse_score_grid(output: Path, frozen: dict[str, Any], stage: str) -> None:
    folder = output / f"{stage}_coarse_scores"
    table = pd.read_csv(folder / "day_metrics.csv")
    monthly = pd.read_csv(folder / "monthly_metrics.csv")
    macro = pd.read_csv(folder / "macro_metrics.csv")
    support = pd.read_csv(folder / "month_support.csv")
    regions = V7.SOURCES if stage == "source" else V7.TARGETS
    start, end = ("2025-01-01", "2026-01-01") if stage == "source" else ("2026-01-01", "2026-07-01")
    days = pd.date_range(start, end, inclusive="left", tz="UTC")
    expected_count = len(regions) * len(V7.DELAYS) * len(frozen["coarse_methods"]) * len(days)
    if (len(table) != expected_count or table.duplicated(["region", "delay", "method", "origin"]).any()
            or set(table.region) != set(regions) or set(table.delay) != set(V7.DELAYS)
            or set(table.method) != set(frozen["coarse_methods"])
            or set(pd.to_datetime(table.origin, utc=True)) != set(days)):
        raise ValueError("Incomplete or invalid coarse scoring grid")
    metrics = ["mix_mae", "mix_crps", "mix_energy_score32", "ef_point_mae", "ef_crps"]
    if table.scored.dtype != bool:
        raise ValueError("Coarse day label availability must be Boolean")
    months = pd.period_range(start, periods=12 if stage == "source" else 6, freq="M")
    actual_path = Path(frozen["source"] if stage == "source" else frozen["target"]) / "monthly.csv"
    actual = pd.read_csv(actual_path)
    actual["month"] = pd.to_datetime(actual.period_start, utc=True).dt.strftime("%Y-%m")
    available = {(r, m) for r, m in zip(actual.region, actual.month)
                 if r in regions and m in {str(month) for month in months}}
    if stage == "target" and len(available) != len(regions) * len(months):
        raise ValueError("Target actual coarse month grid incomplete")
    expected_scored = np.array([(r, m) in available for r, m in zip(table.region, table.month)])
    if (not np.array_equal(table.scored.to_numpy(), expected_scored)
            or not np.isfinite(table.loc[table.scored, metrics]).all().all()
            or not table.loc[~table.scored, metrics].isna().all().all()):
        raise ValueError("Coarse scored/missing label mask differs")
    expected_months = {(r, d, method, m) for r, m in available for d in V7.DELAYS
                       for method in frozen["coarse_methods"]}
    monthly_keys = [(r.region, int(r.delay), r.method, r.month) for _, r in monthly.iterrows()]
    if (len(monthly_keys) != len(set(monthly_keys)) or set(monthly_keys) != expected_months
            or not np.isfinite(monthly[metrics]).all().all() or monthly.days.le(0).any()):
        raise ValueError("Incomplete or invalid coarse monthly scoring grid")
    grouped = table.loc[table.scored].groupby(["region", "delay", "method", "month", "semester"], sort=True)
    expected = grouped[metrics].mean().reset_index()
    expected["days"] = grouped.size().to_numpy()
    A7.assert_equal_table(monthly, expected,
                          ["region", "delay", "method", "month", "semester"], [*metrics, "days"])
    cell_means = monthly.groupby(["region", "delay", "method", "semester"], sort=True)[metrics].mean().reset_index()
    expected_macro = cell_means.groupby(["method", "semester"], sort=True)[metrics].mean().reset_index()
    A7.assert_equal_table(macro, expected_macro, ["method", "semester"], metrics)
    first_method = frozen["coarse_methods"][0]
    coverage = table.loc[table.method.eq(first_method)].groupby(
        ["region", "delay", "month", "semester"], sort=True)
    expected_support = coverage.scored.agg([("calendar_days", "size"), ("scored_days", "sum")]).reset_index()
    expected_support["missing_days"] = expected_support.calendar_days - expected_support.scored_days
    expected_support["label_available"] = expected_support.scored_days.gt(0)
    A7.assert_equal_table(support, expected_support, ["region", "delay", "month", "semester"],
                          ["calendar_days", "scored_days", "missing_days", "label_available"])


def verify_phase(output: Path, stage: str) -> dict[str, Any]:
    if stage not in ("source", "target"):
        raise ValueError("Unknown stage")
    output = Path(output).resolve()
    frozen = verify(output)
    done = read(output / f"{stage}_completed.json")
    if done["prefit_freeze_sha256"] != sha(output / "prefit_freeze.json"):
        raise ValueError("Completion/freeze binding changed")
    seal_path = output / f"{stage}_prediction_seal.json"
    if done["prediction_seal_sha256"] != sha(seal_path):
        raise ValueError("Completion/prediction seal binding changed")
    seal = read(seal_path)
    if (seal.get("hourly_labels_read_in_prediction_phase") is not False
            or seal.get("actual_monthly_mix_used_for_scoring_before_seal") is not False
            or seal.get("monthly_forecast_vintages_read_preseal") is not True):
        raise ValueError("Prediction information boundary changed")
    check_entries(output, frozen, stage, seal)
    folder = output / f"{stage}_scores"
    if done["score_seal_sha256"] != sha(folder / "score_seal.json"):
        raise ValueError("Completion/score binding changed")
    hashes = read(folder / "score_seal.json")
    expected_scores = SCORE_FILES | ({"ranking.csv"} if stage == "source" else set())
    if set(hashes) != expected_scores:
        raise ValueError("Incomplete score manifest")
    for name, digest in hashes.items():
        if sha(folder / name) != digest:
            raise ValueError("Score bytes changed")
    check_score_grid(output, frozen, stage)
    coarse_folder = output / f"{stage}_coarse_scores"
    if done["coarse_score_seal_sha256"] != sha(coarse_folder / "score_seal.json"):
        raise ValueError("Completion/coarse score binding changed")
    coarse_hashes = read(coarse_folder / "score_seal.json")
    if set(coarse_hashes) != COARSE_SCORE_FILES:
        raise ValueError("Incomplete coarse score manifest")
    for name, digest in coarse_hashes.items():
        if sha(coarse_folder / name) != digest:
            raise ValueError("Coarse score bytes changed")
    check_coarse_score_grid(output, frozen, stage)
    if pd.Timestamp(seal["created_utc"]) > pd.Timestamp(done["created_utc"]):
        raise ValueError("Completion predates prediction seal")
    if done.get("target_selection_performed") is not False or done.get("target_training_performed") is not False:
        raise ValueError("Completion indicates target selection/training")
    choices = read(output / "source_choices.json")
    validate_choices(output, choices)
    if done.get("source_choices_sha256") != sha(output / "source_choices.json"):
        raise ValueError("Completion/source choice binding changed")
    if stage == "target":
        verify_phase(output, "source")
        validate_choices(output, read(output / "source_choices.json"))
        if (seal.get("source_choices_sha256") != sha(output / "source_choices.json")
                or done.get("source_choices_sha256") != sha(output / "source_choices.json")
                or pd.Timestamp(choices["created_utc"]) > pd.Timestamp(seal["created_utc"])):
            raise ValueError("Target detached from prior source selection")
    return seal


def _save_predictions(output, stage, region, delay, days, result, grid):
    entries = []
    if set(result) != {c["name"] for c in grid}:
        raise ValueError("Risk producer did not return the complete configuration grid")
    for config in grid:
        path = output / "predictions" / f"{stage}-{region}-{delay}-r{config['round']}-{config['name']}.npz"
        if path.exists():
            raise FileExistsError(path)
        prediction = result[config["name"]]
        np.savez_compressed(path, origins=pd.DatetimeIndex(days).as_unit("ns").asi8,
                            point=prediction["point"], ensemble=prediction["ensemble"])
        _check_archive(path, days)
        entries.append({**config, "region": region, "delay": delay,
                        "path": str(path.relative_to(output)), "sha256": sha(path),
                        "interval_only": False, "audit": _audit(prediction.get("audit", {}))})
    return entries


def _save_coarse(output, stage, region, delay, days, forecasts, methods):
    entries = []
    if set(forecasts) != set(methods):
        raise ValueError("Coarse producer did not return the complete method grid")
    for method in methods:
        path = output / "coarse_predictions" / f"{stage}-{region}-{delay}-{method}.npz"
        if path.exists():
            raise FileExistsError(path)
        result = forecasts[method]
        np.savez_compressed(path, origins=pd.DatetimeIndex(days).as_unit("ns").asi8,
                            mix=result["mix"], draws=result["draws"])
        _check_archive(path, days, coarse_archive=True)
        entries.append({"region": region, "delay": delay, "method": method,
                        "path": str(path.relative_to(output)), "sha256": sha(path),
                        "audit": _audit(result.get("audit", {}))})
    return entries


def score_coarse(output: Path, entries: list[dict[str, Any]], monthly: pd.DataFrame, folder: Path) -> None:
    """Read actual composition only after the complete prediction seal exists."""
    if not (output / ("source_prediction_seal.json" if entries[0]["region"] in V7.SOURCES else "target_prediction_seal.json")).exists():
        raise ValueError("Coarse actuals require a prediction seal")
    from transcif.research.eia_proxy import FUELS
    rows = monthly.copy()
    rows["period_start"] = pd.to_datetime(rows.period_start, utc=True)
    lookup = {}
    for _, row in rows.iterrows():
        key = row["region"], row["period_start"]
        if key in lookup:
            raise ValueError("Duplicate actual region-month composition")
        lookup[key] = row[["fuelshare_" + fuel for fuel in FUELS]].to_numpy(dtype=float)
    metrics = []
    for entry in entries:
        path = output / entry["path"]
        if sha(path) != entry["sha256"]:
            raise ValueError("Coarse prediction changed before scoring")
        with np.load(path, allow_pickle=False) as a:
            origins, mix, draws = a["origins"], a["mix"], a["draws"]
        for i, stamp in enumerate(origins):
            day = pd.Timestamp(int(stamp), tz="UTC")
            month = pd.Timestamp(year=day.year, month=day.month, day=1, tz="UTC")
            actual = lookup.get((entry["region"], month))
            scored = actual is not None
            values = {key: np.nan for key in ("mix_mae", "mix_crps", "mix_energy_score32",
                                                "ef_point_mae", "ef_crps")}
            if scored:
                if not np.isfinite(actual).all() or np.any(actual < 0) or not np.isclose(actual.sum(), 1, atol=1e-8):
                    raise ValueError("Invalid actual monthly composition")
                x = draws[i]
                sorted_x = np.sort(x, axis=0)
                coefficient = 2 * np.arange(1, len(x) + 1) - len(x) - 1
                pair_term = np.sum(sorted_x * coefficient[:, None], axis=0) / len(x) ** 2
                crps = (np.abs(x - actual).mean(axis=0) - pair_term).mean()
                joint = x[:min(32, len(x))]
                norm = np.linalg.norm(joint, axis=1)
                squared_distances = np.maximum(norm[:, None] ** 2 + norm[None, :] ** 2 - 2 * (joint @ joint.T), 0)
                energy = np.linalg.norm(joint - actual, axis=1).mean() - .5 * np.sqrt(squared_distances).mean()
                intensity = x @ EF
                actual_intensity = float(actual @ EF)
                sorted_intensity = np.sort(intensity)
                intensity_pair = float(np.dot(sorted_intensity, coefficient) / len(x) ** 2)
                intensity_crps = float(np.abs(intensity - actual_intensity).mean() - intensity_pair)
                values = {"mix_mae": float(np.abs(mix[i] - actual).mean()),
                          "mix_crps": float(crps), "mix_energy_score32": float(energy),
                          "ef_point_mae": float(abs(mix[i] @ EF - actual_intensity)),
                          "ef_crps": intensity_crps}
            metrics.append({"region": entry["region"], "delay": entry["delay"], "method": entry["method"],
                            "origin": str(day), "month": day.strftime("%Y-%m"),
                            "semester": "H1" if day.month <= 6 else "H2",
                            "scored": scored, **values})
    folder.mkdir()
    table = pd.DataFrame(metrics)
    table.to_csv(folder / "day_metrics.csv", index=False)
    score_columns = ["mix_mae", "mix_crps", "mix_energy_score32", "ef_point_mae", "ef_crps"]
    grouped = table.loc[table.scored].groupby(["region", "delay", "method", "month", "semester"], sort=True)
    by_month = grouped[score_columns].mean().reset_index()
    by_month["days"] = grouped.size().to_numpy()
    by_month.to_csv(folder / "monthly_metrics.csv", index=False)
    cell_means = by_month.groupby(["region", "delay", "method", "semester"], sort=True)[score_columns].mean().reset_index()
    cell_means.groupby(["method", "semester"], sort=True)[score_columns].mean().reset_index().to_csv(
        folder / "macro_metrics.csv", index=False)
    first_method = entries[0]["method"]
    coverage = table.loc[table.method.eq(first_method)].groupby(
        ["region", "delay", "month", "semester"], sort=True)
    support = coverage.scored.agg([("calendar_days", "size"), ("scored_days", "sum")]).reset_index()
    support["missing_days"] = support.calendar_days - support.scored_days
    support["label_available"] = support.scored_days.gt(0)
    support.to_csv(folder / "month_support.csv", index=False)
    write(folder / "score_seal.json", {p.name: sha(p) for p in folder.iterdir() if p.is_file()})


def run_phase(output: Path, upstream: Path, stage: str) -> None:
    output, upstream = Path(output).resolve(), Path(upstream).resolve()
    if stage not in ("source", "target"):
        raise ValueError("Unknown stage")
    frozen = freeze(output, upstream) if stage == "source" else verify(output)
    if upstream != Path(frozen["upstream"]):
        raise ValueError("Upstream V7 run changed")
    if stage == "target":
        verify_phase(output, "source")
    if any((output / f"{stage}_{suffix}").exists() for suffix in
           ("prediction_seal.json", "scores", "coarse_scores", "completed.json")):
        raise FileExistsError("Stage is immutable")
    source, target = Path(frozen["source"]), Path(frozen["target"])
    monthly_path = (source if stage == "source" else target) / "monthly.csv"
    forecast_vintages = pd.read_csv(monthly_path)
    days = pd.date_range("2025-01-01" if stage == "source" else "2026-01-01",
                         "2026-01-01" if stage == "source" else "2026-07-01",
                         inclusive="left", tz="UTC")
    regions = V7.SOURCES if stage == "source" else V7.TARGETS
    producer, coarse_producer = risk(), coarse()
    entries, coarse_entries = [], []
    for delay in V7.DELAYS:
        for region in regions:
            bundle = load_fit(frozen, region, delay, stage)
            forecasts = coarse_producer.forecast_monthly(forecast_vintages, region, days, extra_delay=delay,
                                                          n_samples=V7.N_SAMPLES, seed=V7.SEED)
            coarse_entries.extend(_save_coarse(output, stage, region, delay, days, forecasts,
                                               frozen["coarse_methods"]))
            predictions = producer.predict_paths(bundle["bank"], forecasts, days)
            entries.extend(_save_predictions(output, stage, region, delay, days, predictions,
                                             frozen["configs"]))
            print("Predicted", stage, region, delay, flush=True)
    verify(output)
    seal = {"created_utc": now(), "entries": entries, "coarse_entries": coarse_entries,
            "complete_grid": True, "coarse_complete_grid": True,
            "hourly_labels_read_in_prediction_phase": False,
            "actual_monthly_mix_used_for_scoring_before_seal": False,
            "monthly_forecast_vintages_read_preseal": True}
    if stage == "target":
        seal["source_choices_sha256"] = sha(output / "source_choices.json")
    check_entries(output, frozen, stage, seal)
    write(output / f"{stage}_prediction_seal.json", seal)
    V7.score_entries(output, entries, source if stage == "source" else target,
                     2025 if stage == "source" else 2026, output / f"{stage}_scores")
    actual_monthly = pd.read_csv(monthly_path)
    score_coarse(output, coarse_entries, actual_monthly, output / f"{stage}_coarse_scores")
    if stage == "source":
        choices, ranking = selection(pd.read_csv(output / "source_scores/region_metrics.csv"), frozen["configs"])
        ranking.to_csv(output / "source_scores/ranking.csv", index=False)
        write(output / "source_scores/score_seal.json", {p.name: sha(p) for p in
              (output / "source_scores").iterdir() if p.is_file() and p.name != "score_seal.json"})
        write(output / "source_choices.json", {"created_utc": now(), "selections": choices,
            "target_scores_used": False, "source_h2_used_for_selection": False,
            "source_seal_sha256": sha(output / "source_prediction_seal.json"),
            "source_score_hashes": {p.name: sha(p) for p in (output / "source_scores").iterdir() if p.is_file()}})
    write(output / f"{stage}_completed.json", {"created_utc": now(),
        "prefit_freeze_sha256": sha(output / "prefit_freeze.json"),
        "prediction_seal_sha256": sha(output / f"{stage}_prediction_seal.json"),
        "score_seal_sha256": sha(output / f"{stage}_scores/score_seal.json"),
        "coarse_score_seal_sha256": sha(output / f"{stage}_coarse_scores/score_seal.json"),
        "source_choices_sha256": sha(output / "source_choices.json"),
        "target_selection_performed": False, "target_training_performed": False})
    verify_phase(output, stage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("source", "target"))
    parser.add_argument("--upstream", type=Path, default=ROOT / "results/fuel-transport-v7")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_phase(args.output, args.upstream, args.phase)


if __name__ == "__main__":
    main()
