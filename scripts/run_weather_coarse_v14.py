"""Immutable source-first V14 weather policies and coarse-to-hourly risk arms."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import numpy as np
import pandas as pd
from scripts import run_weather_increment_v12 as v12
from transcif.research.coarse_forecast_v7 import COLUMNS

v10 = v12.v10
SOURCES, TARGETS = v12.SOURCES, v12.TARGETS
SEED, N_SAMPLES, FACTORS = v12.SEED, v12.N_SAMPLES, v12.FACTORS
METRICS, DAY_KEYS, REGION_KEYS = v12.METRICS, v12.DAY_KEYS, v12.REGION_KEYS
sha, read, write, now = v12.sha, v12.read, v12.write, v12.now
allowed_sources, _load_training, _coarse = v12.allowed_sources, v12._load_training, v12._coarse
PROTOCOL = ROOT / "docs/WEATHER_COARSE_V14_PROTOCOL.md"
V12_ROOT = ROOT / "results/weather-increment-v12-round2"
V13_REPAIR_ROOT = ROOT / "results/weather-coarse-v13-round1"
POLICIES = ("fixed_donor", "constant_half", "gate_ridge", "gate_forest")
RISK_FAMILIES = ("global_total", "hetero_total", "location_scale", "qrf_0", "qrf_50",
                 "level_independent", "level_rank", "physical_independent", "physical_rank")
CALIBRATION_CUTOFF = pd.Timestamp("2025-01-01", tz="UTC")
CALIBRATION_KEYS = ("calibration_regions", "calibration_origins",
                    "calibration_response_available_at", "calibration_response_period_end")


def calibration_sources(source: Path) -> list[str]:
    """Preflight monthly-only H2 response support before creating a V14 run."""
    table = pd.read_csv(Path(source) / "monthly.csv")
    table = table.loc[table.region.isin(SOURCES)].copy()
    for key in ("period_start", "period_end", "available_at"):
        table[key] = pd.to_datetime(table[key], utc=True, errors="raise")
    keep = (table.period_start.ge(pd.Timestamp("2024-07-01", tz="UTC")) &
            table.period_start.lt(CALIBRATION_CUTOFF) & table.period_end.le(CALIBRATION_CUTOFF) &
            table.available_at.le(CALIBRATION_CUTOFF))
    rows = table.loc[keep]
    if rows.duplicated(["region", "period_start"]).any():
        raise ValueError("Calibration preflight requires one frozen response per source-month")
    values = rows[list(COLUMNS)].to_numpy(float)
    if (not np.isfinite(values).all() or np.any(values < 0) or
            not np.allclose(values.sum(1), 1., rtol=0., atol=1e-8) or
            not rows.period_start.eq(rows.period_start.dt.normalize()).all() or
            not rows.period_start.dt.day.eq(1).all() or
            not rows.period_end.eq(rows.period_start+pd.offsets.MonthBegin(1)).all() or
            rows.available_at.lt(rows.period_end).any()):
        raise ValueError("Invalid released H2 monthly calibration response")
    eligible = [r for r in SOURCES if r in set(rows.region)]
    if len(eligible) < 5:
        raise ValueError("Calibration preflight needs five sources for nested location-scale folds")
    return eligible


def allowed_calibration_sources(frozen: dict, proper_sources: tuple[str, ...]) -> tuple[str, ...]:
    """Keep every approved source response while preserving all proper sources."""
    return tuple(r for r in proper_sources if r in frozen["calibration_sources"])


def configurations(round_number: int) -> list[dict[str, Any]]:
    """Return the predeclared six/22 arms and their primary eligibility."""
    if round_number not in (1, 2):
        raise ValueError("V14 round must be 1 or 2")
    result = [dict(budget="B", model=m, primary_eligible=False)
              for m in ("native_recompose", "simple_anchor")]
    if round_number == 1:
        return result + [dict(budget="W", model=m, primary_eligible=True) for m in POLICIES]
    result += [dict(budget="W", model=m, primary_eligible=m == "selected_donor")
               for m in ("fixed_donor", "selected_donor")]
    return result + [dict(budget=b, model=m, primary_eligible=b == "W")
                     for b in ("B", "W") for m in RISK_FAMILIES]


def _code_paths() -> list[Path]:
    old = read(V12_ROOT / "prefit_freeze.json")["code_hashes"]
    paths = sorted(set([*map(Path, old), Path(__file__).resolve(), PROTOCOL,
                        ROOT / "scripts/weather_coarse_v14.py",
                        ROOT / "scripts/analyze_weather_coarse_v14.py"]))
    if len(old) != 46 or len(paths) != 50:
        raise ValueError("V14 requires the complete inherited 46 plus four new code paths")
    return paths


def _baseline_paths() -> dict[str, str]:
    paths = {}
    for stage, regions in (("source", SOURCES), ("target", TARGETS)):
        seal = read(V12_ROOT / f"{stage}_prediction_seal.json")
        for region in regions:
            for budget, model, old_model in (("B", "native_recompose", "native_recompose"),
                                              ("B", "simple_anchor", "simple_anchor"),
                                              ("W", "fixed_donor", "donor")):
                entries = [e for e in seal["entries"] if (e["region"], e["budget"], e["model"]) ==
                           (region, budget, old_model)]
                if len(entries) != 1:
                    raise ValueError("Missing unique V12 baseline archive")
                path = V12_ROOT / entries[0]["path"]
                if sha(path) != entries[0]["sha256"]:
                    raise ValueError("Frozen V12 baseline changed")
                paths[f"{stage}/{region}/{model}"] = str(path.resolve())
    return paths


def _verify_repair_identity(prior_round: Path) -> list[Path]:
    """Bind V14 R1 to the completed V13 R1 without executing its producer."""
    old, new = V13_REPAIR_ROOT, Path(prior_round)
    names = ("prefit_freeze.json", "source_prediction_seal.json", "source_completed.json", "source_choices.json",
             "source_scores/score_seal.json", "source_scores/day_metrics.csv",
             "source_scores/region_metrics.csv", "source_scores/macro_metrics.csv")
    paths = [old / name for name in names]
    frozen, seal, done = (read(old / name) for name in names[:3])
    if frozen.get("version") != 13 or frozen.get("round") != 1 or seal.get("round") != 1 or (
            seal.get("complete_grid") is not True or seal.get("evaluation_hourly_labels_read_preseal") is not False):
        raise ValueError("V13 repair reference must be a sealed source round 1")
    for key, name in (("prefit_freeze_sha256", "prefit_freeze.json"),
                      ("prediction_seal_sha256", "source_prediction_seal.json"),
                      ("score_seal_sha256", "source_scores/score_seal.json"),
                      ("source_choices_sha256", "source_choices.json")):
        if done.get(key) != sha(old / name):
            raise ValueError("V13 repair reference completion binding changed")
    scores = read(old / "source_scores/score_seal.json")
    if set(scores) != {"day_metrics.csv", "region_metrics.csv", "macro_metrics.csv"}:
        raise ValueError("V13 repair source score manifest incomplete")
    for name, digest in scores.items():
        path = old / "source_scores" / name
        if sha(path) != digest or path.read_bytes() != (new / "source_scores" / name).read_bytes():
            raise ValueError("V14 R1 scores differ from the preserved V13 R1")
    old_choice, new_choice = read(old / "source_choices.json"), read(new / "source_choices.json")
    core = ("selections", "primary", "native", "simple", "baseline", "diagnostic_crps_best", "locked_policy")
    if (old_choice.get("target_scores_used") is not False or
            any(old_choice.get(k) != new_choice.get(k) for k in core) or any(k not in old_choice for k in core)):
        raise ValueError("V14 R1 source choices differ from preserved V13 R1")
    previous, current = seal["entries"], read(new / "source_prediction_seal.json")["entries"]
    expected = _grid("source", {"round": 1})
    def index(entries: list[dict]) -> dict:
        result = {(e["region"], e["budget"], e["model"]): e for e in entries}
        if len(entries) != len(expected) or set(result) != expected:
            raise ValueError("Repair identity requires both complete source R1 grids")
        return result
    old_entries, new_entries = index(previous), index(current)
    for key, entry in old_entries.items():
        old_path, new_path = (old / entry["path"]).resolve(), (new / new_entries[key]["path"]).resolve()
        if (not old_path.is_relative_to((old / "predictions").resolve()) or
                not new_path.is_relative_to((new / "predictions").resolve()) or
                sha(old_path) != entry["sha256"] or sha(new_path) != new_entries[key]["sha256"]):
            raise ValueError("Repair identity archive path/hash changed")
        with np.load(old_path, allow_pickle=False) as a, np.load(new_path, allow_pickle=False) as b:
            if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
                raise ValueError("V14 R1 prediction arrays differ from preserved V13 R1")
        paths.append(old_path)
    return paths


def freeze(output: Path, data: Path, *, round_number: int,
           prior_round: Path | None = None, weather_csv: Path | None = None,
           points_json: Path | None = None) -> dict[str, Any]:
    """Freeze the inherited 46 paths and four V14 paths before source predictions."""
    output, data = Path(output).resolve(), Path(data).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh immutable V14 output directory")
    configurations(round_number)
    if round_number == 2 and prior_round is None:
        raise ValueError("Round 2 requires --prior-round source choices/locks")
    if round_number == 1 and prior_round is not None:
        raise ValueError("Round 1 does not use a prior round")
    old = v12.verify(V12_ROOT)
    if (Path(old["data"]) != data or old["alpha"] != .5 or
            (weather_csv is not None and Path(weather_csv).resolve() != Path(old["weather_csv"])) or
            (points_json is not None and Path(points_json).resolve() != Path(old["points_json"]))):
        raise ValueError("V14 must reuse frozen V12 inputs and fixed tau=.5")
    policy, prior_paths = None, []
    if round_number == 2:
        prior_round = Path(prior_round).resolve()
        verify_phase(prior_round, "source")
        if verify(prior_round)["round"] != 1 or any((prior_round / n).exists() for n in
                ("target_prediction_seal.json", "target_scores", "target_completed.json")):
            raise ValueError("Round 2 source must precede every target prediction/score")
        policy = read(prior_round / "source_choices.json")["locked_policy"]
        prior_paths = [prior_round / n for n in ("prefit_freeze.json", "source_choices.json",
                       "source_prediction_seal.json", "source_completed.json", "source_scores/score_seal.json",
                       "source_scores/day_metrics.csv", "source_scores/region_metrics.csv", "source_scores/macro_metrics.csv")]
        prior_paths += [prior_round / e["path"] for e in read(prior_round / "source_prediction_seal.json")["entries"]]
        prior_paths += _verify_repair_identity(prior_round)
    baselines, frozen = _baseline_paths(), deepcopy(old)
    eligible_calibration = calibration_sources(Path(old["source"]))
    inputs = [V12_ROOT / n for n in ("prefit_freeze.json", "source_choices.json",
              "source_prediction_seal.json", "target_prediction_seal.json", "source_completed.json", "target_completed.json")]
    inputs += [*map(Path, baselines.values()), *prior_paths, Path(sys.executable).resolve()]
    frozen["input_hashes"].update({str(p): sha(p) for p in inputs})
    frozen.update(created_utc=now(), version=14, round=round_number, alpha=.5,
                  weather_policy=policy, prior_round=str(prior_round) if prior_round else None,
                  baseline_paths=baselines, configurations=configurations(round_number),
                  models=sorted({c["model"] for c in configurations(round_number)}),
                  calibration_response_cutoff=CALIBRATION_CUTOFF.isoformat(),
                  calibration_sources=eligible_calibration,
                  code_hashes={str(p): sha(p) for p in _code_paths()},
                  environment=v10.environment(), target_scores_used_for_round_lock=False)
    output.mkdir(parents=True)
    (output / "predictions").mkdir()
    write(output / "prefit_freeze.json", frozen)
    return frozen


def verify(output: Path) -> dict[str, Any]:
    """Verify immutable bytes, the fixed grid, environment and source policy lock."""
    frozen = read(Path(output) / "prefit_freeze.json")
    for path, digest in {**frozen["input_hashes"], **frozen["code_hashes"]}.items():
        if sha(Path(path)) != digest:
            raise ValueError("Frozen V14 bytes changed: " + path)
    if (set(frozen["code_hashes"]) != {str(p) for p in _code_paths()} or
            frozen["version"] != 14 or frozen["round"] not in (1, 2) or
            frozen["configurations"] != configurations(frozen["round"]) or
            frozen["sources"] != list(SOURCES) or frozen["targets"] != list(TARGETS) or
            frozen["samples"] != N_SAMPLES or frozen["seed"] != SEED or
            frozen["factors"] != list(FACTORS) or frozen["proper_start"] != "2024-01-25" or
            frozen["alpha"] != .5 or frozen["environment"] != v10.environment() or
            frozen.get("calibration_response_cutoff") != CALIBRATION_CUTOFF.isoformat() or
            frozen.get("calibration_sources") != calibration_sources(Path(frozen["source"])) or
            frozen.get("target_scores_used_for_round_lock") is not False):
        raise ValueError("V14 frozen grid/environment changed")
    policy = frozen["weather_policy"]
    if frozen["round"] == 1 and policy is not None:
        raise ValueError("Round 1 cannot inherit a weather policy")
    if frozen["round"] == 2 and (policy not in POLICIES or
            policy != read(Path(frozen["prior_round"]) / "source_choices.json")["locked_policy"]):
        raise ValueError("Round 2 detached from round 1 source policy lock")
    return frozen


def _grid(stage: str, frozen: dict) -> set[tuple[str, str, str]]:
    if stage not in ("source", "target"):
        raise ValueError("Unknown V14 phase")
    return {(r, c["budget"], c["model"]) for r in (SOURCES if stage == "source" else TARGETS)
            for c in configurations(frozen["round"])}


def _check_calibration(a: Any, expected: tuple[str, ...]) -> None:
    regions, origins = a["calibration_regions"], a["calibration_origins"]
    if regions.ndim != 1 or regions.dtype.kind != "U" or set(regions) != set(expected):
        raise ValueError("Wrong source calibration regions")
    if any(a[k].shape != regions.shape for k in CALIBRATION_KEYS[1:]):
        raise ValueError("Invalid calibration response cohort shape")
    for key in CALIBRATION_KEYS[1:]:
        if not np.issubdtype(a[key].dtype, np.integer):
            raise ValueError("Calibration dates require integer UTC nanoseconds")
    dates = pd.to_datetime(origins, utc=True)
    ends = (dates.normalize()-pd.to_timedelta(dates.day-1, unit="D")+pd.offsets.MonthBegin(1)).as_unit("ns").asi8
    if (len(set(zip(regions, origins))) != len(origins) or
            np.any(origins < pd.Timestamp("2024-07-01", tz="UTC").value) or
            np.any(origins >= CALIBRATION_CUTOFF.value) or
            np.any(a["calibration_response_available_at"] > CALIBRATION_CUTOFF.value) or
            np.any(a["calibration_response_period_end"] > CALIBRATION_CUTOFF.value) or
            np.any(a["calibration_response_period_end"] <= origins) or
            not np.array_equal(a["calibration_response_period_end"], ends) or
            np.any(a["calibration_response_available_at"] < a["calibration_response_period_end"])):
        raise ValueError("Calibration realized monthly response exceeds lawful cutoff")


def _check_baseline(a: Any, path: Path) -> None:
    with np.load(path, allow_pickle=False) as old:
        keys = ("origins", "point", "ensemble", "raw_ensemble", "donor_indices", "center")
        if any(not np.array_equal(a[k], old[k]) for k in keys):
            raise ValueError("Exact V12 native/simple/fixed weather identity failed")


def _check_factor_archive(a: Any, n: int, model: str) -> None:
    """Bind the new fine/coarse decomposition and coupling to its saved inputs."""
    c = len(a["calibration_regions"])
    shapes = dict(calibration_errors=(c, 24), calibration_fine_errors=(c, 24), calibration_z=(c,),
        calibration_true_mix=(c, 8), calibration_forecast_mix=(c, 8), calibration_qbar=(c, 24, 8),
        query_qbar=(n, 24, 8), query_coarse_mix=(n, 8), query_coarse_draws=(n, N_SAMPLES, 8),
        query_coarse_features=(n, 25), structural_center=(n, 24))
    if any(a[k].shape != shape or not np.isfinite(a[k]).all() for k, shape in shapes.items()):
        raise ValueError("Invalid coarse/fine archive shapes or values")
    for key in ("calibration_true_mix", "calibration_forecast_mix", "query_coarse_mix", "query_coarse_draws"):
        if np.any(a[key] < 0) or not np.allclose(a[key].sum(-1), 1., rtol=0., atol=1e-10):
            raise ValueError("Invalid archived composition simplex")
    if np.any(a["calibration_qbar"] < 0) or np.any(a["query_qbar"] < 0):
        raise ValueError("Negative archived donor profile")
    forecast = v12.recompose_paths(a["calibration_forecast_mix"], a["calibration_qbar"])
    actual = v12.recompose_paths(a["calibration_true_mix"], a["calibration_qbar"])
    if (not np.allclose(a["calibration_errors"]-a["calibration_fine_errors"], actual-forecast, rtol=0., atol=1e-9) or
            not np.allclose(a["calibration_z"], (a["calibration_true_mix"]-a["calibration_forecast_mix"]) @ v12.EF, rtol=0., atol=1e-10) or
            not np.allclose(a["structural_center"], v12.recompose_paths(a["query_coarse_mix"], a["query_qbar"]), rtol=0., atol=1e-10)):
        raise ValueError("Archived coarse/fine factor identity changed")
    if model == "location_scale":
        location = a["location_prediction"]
        if location.shape != (n, 24) or not np.isfinite(location).all() or not np.array_equal(
                a["center"], a["structural_center"]+location):
            raise ValueError("Location-scale center detached from location prediction")
    elif not np.array_equal(a["center"], a["structural_center"]):
        raise ValueError("Risk reference center differs from structural center")
    if model.startswith(("level_", "physical_")):
        ids, independent, rank = (a[k] for k in ("fine_residual_indices", "independent_residual_indices", "calibration_rank_indices"))
        if any(x.shape != (n, N_SAMPLES) or not np.issubdtype(x.dtype, np.integer) or
               np.any((x < 0) | (x >= c)) for x in (ids, independent, rank)):
            raise ValueError("Invalid coupled residual indices")
        if (not np.array_equal(np.sort(rank, axis=1), np.sort(independent, axis=1)) or
                not np.array_equal(ids, rank if model.endswith("rank") else independent)):
            raise ValueError("Rank coupling changed the source residual-day multiset")
        labels = a["calibration_regions"]
        mass = np.array([1/(len(set(labels))*np.sum(labels == r)) for r in labels])
        cdf = np.cumsum(mass)
        cdf[-1] = 1.
        expected = np.searchsorted(cdf, a["residual_uniforms"], side="right")
        if not np.array_equal(independent, expected):
            raise ValueError("Independent pairing detached from equal-region day weights/uniforms")


def _check_profile_audit(audit: dict) -> None:
    error, zero = audit.get("donor_q_mean_max_error"), audit.get("sampled_zero_creation")
    if not isinstance(error, (int, float)) or not np.isfinite(error) or not 0 <= error <= 1e-10 or not isinstance(zero, dict):
        raise ValueError("Missing/invalid measured structural amplitude audit")
    keys = ("supported_original_zero_count_by_fuel", "created_positive_from_zero_count_by_fuel",
            "created_positive_from_zero_fraction_by_fuel", "created_positive_from_zero_q_amount_by_fuel")
    values = [np.asarray(zero[k], dtype=float) for k in keys]
    if any(x.shape != (8,) or not np.isfinite(x).all() or np.any(x < 0) for x in values):
        raise ValueError("Invalid measured zero-creation audit")
    count, created, fraction, _ = values
    if (np.any(created > count) or np.any(count != np.floor(count)) or np.any(created != np.floor(created)) or
            not np.allclose(fraction, np.divide(created, count, out=np.zeros(8), where=count > 0), rtol=0., atol=1e-15)):
        raise ValueError("Zero-creation counts/fractions disagree")


def check_entries(output: Path, frozen: dict, stage: str, seal: dict) -> None:
    """Check the complete archive grid, exact controls and common lawful risk cohort."""
    entries = seal["entries"]
    if len(entries) != len(_grid(stage, frozen)) or {(e["region"], e["budget"], e["model"]) for e in entries} != _grid(stage, frozen):
        raise ValueError("V14 prediction grid incomplete/duplicated")
    days = v10._evaluation_days(stage, True).as_unit("ns").asi8
    shared, cohorts, uniforms = {}, {}, {}
    prediction_root = (Path(output) / "predictions").resolve()
    for entry in entries:
        region, budget, model = entry["region"], entry["budget"], entry["model"]
        expected, audit = allowed_sources(region if stage == "source" else None), entry["audit"]
        calibration_expected = allowed_calibration_sources(frozen, expected) if frozen["round"] == 2 else ()
        if (set(audit["training_regions"]) != set(expected) or set(audit["donor_regions"]) != set(expected) or
                audit.get("target_hourly_used") is not False):
            raise ValueError("Outer region leaked into fit/donor inputs")
        if frozen["round"] == 2 and set(audit["source_calibration_regions"]) != set(calibration_expected):
            raise ValueError("Wrong approved source calibration audit")
        if model == "native_recompose" or model in (*POLICIES, "selected_donor"):
            _check_profile_audit(audit)
        projection = audit.get("sampled_fixed_weather_projection")
        if projection is not None and (not np.isfinite(projection["mean_absolute_projection_change"]) or
                projection["mean_absolute_projection_change"] < 0 or
                not 0 <= projection["fraction_negative_before_projection"] <= 1 or
                not isinstance(projection["sampled_supported_active_cells"], int) or
                projection["sampled_supported_active_cells"] < 0):
            raise ValueError("Invalid sampled fixed-weather projection audit")
        path = (Path(output) / entry["path"]).resolve()
        if not path.is_relative_to(prediction_root) or sha(path) != entry["sha256"]:
            raise ValueError("V14 prediction path/hash changed")
        with np.load(path, allow_pickle=False) as a:
            if (not np.array_equal(a["origins"], days) or a["ensemble"].shape != (len(days), N_SAMPLES, 24) or
                    a["raw_ensemble"].shape != a["ensemble"].shape or
                    any(a[k].shape != (len(days), 24) for k in ("point", "center")) or
                    not all(np.isfinite(a[k]).all() for k in ("raw_ensemble", "point", "center")) or
                    not np.array_equal(a["ensemble"], np.clip(a["raw_ensemble"], 0, 1000)) or
                    not np.array_equal(a["point"], a["ensemble"].mean(1)) or
                    a["donor_indices"].shape != (len(days), N_SAMPLES) or
                    not np.issubdtype(a["donor_indices"].dtype, np.integer) or np.any(a["donor_indices"] < 0)):
                raise ValueError("Invalid V14 prediction archive")
            if region in shared and not np.array_equal(a["donor_indices"], shared[region]):
                raise ValueError("Common structural donor indices differ")
            shared[region] = a["donor_indices"].copy()
            if model in ("native_recompose", "simple_anchor", "fixed_donor"):
                _check_baseline(a, Path(frozen["baseline_paths"][f"{stage}/{region}/{model}"]))
            if model in RISK_FAMILIES:
                _check_calibration(a, calibration_expected)
                _check_factor_archive(a, len(days), model)
                cohort = tuple(a[k].copy() for k in CALIBRATION_KEYS)
                if region in cohorts and any(not np.array_equal(x, y) for x, y in zip(cohort, cohorts[region])):
                    raise ValueError("Risk references do not share the same calibration cohort")
                cohorts[region] = cohort
                if "residual_uniforms" in a:
                    u = a["residual_uniforms"]
                    if u.shape != (len(days), N_SAMPLES) or not np.isfinite(u).all() or np.any((u < 0) | (u >= 1)):
                        raise ValueError("Invalid residual uniforms")
                    if region in uniforms and not np.array_equal(u, uniforms[region]):
                        raise ValueError("Risk reference uniforms differ")
                    uniforms[region] = u.copy()
            if model.startswith("qrf_"):
                v12._check_qrf_archive(a, len(days), calibration_expected)
                if (a["raw_conditional_quantiles"].shape != (len(days), 3, 24) or
                        not np.isfinite(a["raw_conditional_quantiles"]).all() or
                        not np.array_equal(a["quantile_levels"], [.05, .5, .95]) or
                        np.any(np.diff(a["raw_conditional_quantiles"], axis=1) < 0)):
                    raise ValueError("Invalid QRF weighted quantile archive")


def score_phase(output: Path, frozen: dict, stage: str, seal: dict) -> pd.DataFrame:
    """Open evaluation labels only after complete prediction grids are verified."""
    output = Path(output)
    seal_path = output / f"{stage}_prediction_seal.json"
    if not seal_path.is_file() or read(seal_path) != seal:
        raise ValueError("Complete prediction seal required before evaluation labels")
    check_entries(output, frozen, stage, seal)
    if stage == "target":
        _target_prediction_gate(output, frozen, seal)
    rows = []
    for region in SOURCES if stage == "source" else TARGETS:
        truth_days, truth = v10._truth_lookup(Path(frozen[stage]), stage, region)
        for entry in (e for e in seal["entries"] if e["region"] == region):
            with np.load(output / entry["path"], allow_pickle=False) as a:
                keep, aligned = v10._match_truth(a["origins"], truth_days, truth)
                point, ensemble, center = a["point"][keep], a["ensemble"][keep], a["center"][keep]
                for factor in FACTORS:
                    mean = ensemble.mean(1, keepdims=True)
                    scaled = np.clip(mean + factor * (ensemble - mean), 0, 1000)
                    metrics = {**v10.score_arrays(point, scaled, aligned), **v12._extra_metrics(point, scaled, aligned, center)}
                    rows.extend(dict(stage=stage, region=region, origin=pd.Timestamp(int(day), tz="UTC").isoformat(),
                        lag=7, budget=entry["budget"], model=entry["model"], factor=factor, semester="H1",
                        **{k: float(v[i]) for k, v in metrics.items()}) for i, day in enumerate(a["origins"][keep]))
    day = pd.DataFrame(rows)
    if day.empty or day.duplicated(list(DAY_KEYS)).any() or not np.isfinite(day[list(METRICS)]).all().all():
        raise ValueError("Invalid V14 scores")
    folder = output / f"{stage}_scores"
    folder.mkdir()
    day.to_csv(folder / "day_metrics.csv", index=False)
    grouped = day.groupby(list(REGION_KEYS), sort=True)
    region = grouped[list(METRICS)].mean().reset_index()
    region["days"] = grouped.size().to_numpy()
    region.to_csv(folder / "region_metrics.csv", index=False)
    region.groupby(["stage", "lag", "budget", "model", "factor", "semester"])[list(METRICS)].mean().reset_index().to_csv(folder / "macro_metrics.csv", index=False)
    write(folder / "score_seal.json", {p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()})
    return pd.read_csv(folder / "region_metrics.csv")


def selection(table: pd.DataFrame, round_number: int, weather_policy: str | None = None) -> dict[str, Any]:
    """Select only canonical source scores, preserving failed coverage-gate status."""
    table = table.loc[table.stage.eq("source") & table.semester.eq("H1") & table.lag.eq(7)]
    expected = {(r, c["budget"], c["model"], f) for r in SOURCES for c in configurations(round_number) for f in FACTORS}
    actual = {(r.region, r.budget, r.model, float(r.factor)) for _, r in table.iterrows()}
    if len(table) != len(expected) or actual != expected or not np.isfinite(table[list(METRICS)]).all().all():
        raise ValueError("Incomplete/nonfinite source selection grid")
    grouped = table.groupby(["budget", "model", "factor"], sort=True)
    ranks = grouped[list(METRICS)].mean().reset_index()
    ranks["worst_coverage"] = grouped.empirical_coverage90.min().to_numpy()
    ranks["violation"] = (np.maximum(.90-ranks.empirical_coverage90, 0) +
                          np.maximum(ranks.empirical_coverage90-.95, 0) + np.maximum(.85-ranks.worst_coverage, 0))
    def choose(part: pd.DataFrame, *, crps_only: bool = False) -> dict[str, Any]:
        passed = part[part.violation <= 1e-12]
        candidates = part if crps_only or passed.empty else passed
        keys = ([] if crps_only or not passed.empty else ["violation"]) + ["crps", "budget", "model", "factor"]
        row = candidates.sort_values(keys).iloc[0]
        return dict(budget=str(row.budget), model=str(row.model), factor=float(row.factor),
                    status="passed" if row.violation <= 1e-12 else "failed",
                    source_metrics={k: float(row[k]) for k in (*METRICS, "worst_coverage", "violation")})
    selections = {f"{c['budget']}--{c['model']}": choose(ranks[ranks.budget.eq(c["budget"]) & ranks.model.eq(c["model"])])
                  for c in configurations(round_number)}
    eligible = [c["model"] for c in configurations(round_number) if c["primary_eligible"]]
    primary = choose(ranks[ranks.budget.eq("W") & ranks.model.isin(eligible)])
    policy = primary["model"] if round_number == 1 else weather_policy
    if policy not in POLICIES:
        raise ValueError("Round 2 requires a registered source-locked weather policy")
    return dict(selections=selections, primary=primary, locked_policy=policy,
                native=selections["B--native_recompose"], simple=selections["B--simple_anchor"],
                baseline=choose(ranks[ranks.budget.eq("B")]),
                diagnostic_crps_best=choose(ranks[ranks.budget.eq("W")], crps_only=True))


def validate_choices(output: Path, frozen: dict) -> dict[str, Any]:
    output = Path(output)
    choice = read(output / "source_choices.json")
    expected = selection(pd.read_csv(output / "source_scores/region_metrics.csv"), frozen["round"], frozen["weather_policy"])
    if (any(choice.get(k) != v for k, v in expected.items()) or choice.get("target_scores_used") is not False or
            choice.get("source_score_seal_sha256") != sha(output / "source_scores/score_seal.json") or
            choice.get("source_prediction_seal_sha256") != sha(output / "source_prediction_seal.json")):
        raise ValueError("V14 source choice does not reproduce canonical source scores")
    return choice


def check_score_grid(output: Path, frozen: dict, stage: str) -> None:
    folder = Path(output) / f"{stage}_scores"
    day, region, macro = [pd.read_csv(folder / name) for name in ("day_metrics.csv", "region_metrics.csv", "macro_metrics.csv")]
    expected = {(stage, r, 7, b, m, f, "H1") for r, b, m in _grid(stage, frozen) for f in FACTORS}
    if (len(region) != len(expected) or {tuple(row[k] for k in REGION_KEYS) for _, row in region.iterrows()} != expected or
            day.empty or day.duplicated(list(DAY_KEYS)).any() or
            any(not np.isfinite(t[list(METRICS)]).all().all() for t in (day, region, macro))):
        raise ValueError("V14 score grid incomplete/duplicated/nonfinite")
    for r in SOURCES if stage == "source" else TARGETS:
        groups = list(day[day.region.eq(r)].groupby(["budget", "model", "factor"]))
        if len(groups) != len(configurations(frozen["round"])) * len(FACTORS):
            raise ValueError("V14 day score grid incomplete")
        support = set(groups[0][1].origin)
        if not support or any(set(g.origin) != support for _, g in groups):
            raise ValueError("V14 unequal scoring support")
    macro_keys = ["stage", "lag", "budget", "model", "factor", "semester"]
    for saved, keys, calculated in ((region, list(REGION_KEYS), day.groupby(list(REGION_KEYS))[list(METRICS)].mean().reset_index()),
                                    (macro, macro_keys, region.groupby(macro_keys)[list(METRICS)].mean().reset_index())):
        joined = saved.merge(calculated, on=keys, validate="one_to_one", suffixes=("_saved", "_computed"))
        if len(joined) != len(saved) or len(saved) != len(calculated) or not np.allclose(
                joined[[m+"_saved" for m in METRICS]], joined[[m+"_computed" for m in METRICS]], rtol=0., atol=1e-10):
            raise ValueError("V14 score aggregation mismatch")


def verify_phase(output: Path, stage: str) -> dict[str, Any]:
    """Verify complete immutable prediction and score artifacts for one phase."""
    if stage not in ("source", "target"):
        raise ValueError("Unknown V14 phase")
    output = Path(output)
    frozen = verify(output)
    done, seal = read(output / f"{stage}_completed.json"), read(output / f"{stage}_prediction_seal.json")
    for field, path in (("prefit_freeze_sha256", "prefit_freeze.json"),
                        ("prediction_seal_sha256", f"{stage}_prediction_seal.json"),
                        ("score_seal_sha256", f"{stage}_scores/score_seal.json"),
                        ("source_choices_sha256", "source_choices.json")):
        if done.get(field) != sha(output / path):
            raise ValueError("V14 completion detached from " + field)
    if (seal.get("round") != frozen["round"] or seal.get("evaluation_hourly_labels_read_preseal") is not False or
            seal.get("complete_grid") is not True or done.get("target_training_performed") is not False or
            done.get("target_selection_performed") is not False):
        raise ValueError("V14 information boundary changed")
    check_entries(output, frozen, stage, seal)
    folder = output / f"{stage}_scores"
    scoreseal = read(folder / "score_seal.json")
    if set(scoreseal) != {"day_metrics.csv", "region_metrics.csv", "macro_metrics.csv"} or any(
            sha(folder / name) != digest for name, digest in scoreseal.items()):
        raise ValueError("V14 score bytes/manifest changed")
    check_score_grid(output, frozen, stage)
    choice = validate_choices(output, frozen)
    if stage == "target":
        peer = _target_prediction_gate(output, frozen, seal)
        if (done.get("other_round_target_prediction_seal_sha256") != sha(peer / "target_prediction_seal.json") or
                pd.Timestamp(choice["created_utc"]) > pd.Timestamp(seal["created_utc"])):
            raise ValueError("Targets precede both frozen source choices/predictions")
    return seal


def _target_source_gate(output: Path, frozen: dict, prior_round: Path | None) -> Path:
    verify_phase(output, "source")
    peer = Path(frozen["prior_round"]) if frozen["round"] == 2 else (
        Path(prior_round).resolve() if prior_round else output.with_name("weather-coarse-v14-round2"))
    verify_phase(peer, "source")
    other = verify(peer)
    if other["round"] != 3-frozen["round"] or (frozen["round"] == 1 and Path(other["prior_round"]) != output):
        raise ValueError("Both V14 linked source rounds must finish before targets")
    return peer


def _target_prediction_gate(output: Path, frozen: dict, seal: dict) -> Path:
    peer = _target_source_gate(output, frozen, Path(seal["other_round_source"]))
    own_path, peer_path = output / "target_prediction_seal.json", peer / "target_prediction_seal.json"
    if not own_path.is_file() or read(own_path) != seal or not peer_path.is_file():
        raise ValueError("Both target prediction seals required before evaluation labels")
    for folder, manifest, prediction in ((output, frozen, seal), (peer, verify(peer), read(peer_path))):
        if (prediction.get("round") != manifest["round"] or prediction.get("complete_grid") is not True or
                prediction.get("evaluation_hourly_labels_read_preseal") is not False or
                prediction.get("source_choices_sha256") != sha(folder / "source_choices.json") or
                prediction.get("other_round_source_completed_sha256") != sha(
                    (peer if folder == output else output) / "source_completed.json")):
            raise ValueError("Invalid complete target prediction seal")
        if any(pd.Timestamp(read(p / "source_choices.json")["created_utc"]) > pd.Timestamp(prediction["created_utc"])
               for p in (output, peer)):
            raise ValueError("Target predictions precede both source policy choices")
        check_entries(folder, manifest, "target", prediction)
    return peer


def _finish_phase(output: Path, frozen: dict, stage: str, seal: dict, table: pd.DataFrame) -> None:
    if stage == "source":
        write(output / "source_choices.json", dict(created_utc=now(),
            **selection(table, frozen["round"], frozen["weather_policy"]), target_scores_used=False,
            source_h2_used_for_selection=False, source_prediction_seal_sha256=sha(output / "source_prediction_seal.json"),
            source_score_seal_sha256=sha(output / "source_scores/score_seal.json")))
    peer = Path(seal["other_round_source"]) if stage == "target" else None
    write(output / f"{stage}_completed.json", dict(created_utc=now(),
        prefit_freeze_sha256=sha(output / "prefit_freeze.json"), prediction_seal_sha256=sha(output / f"{stage}_prediction_seal.json"),
        score_seal_sha256=sha(output / f"{stage}_scores/score_seal.json"), source_choices_sha256=sha(output / "source_choices.json"),
        other_round_target_prediction_seal_sha256=sha(peer / "target_prediction_seal.json") if peer else None,
        target_training_performed=False, target_selection_performed=False))
    verify_phase(output, stage)


def _save_region(output: Path, frozen: dict, stage: str, region: str, days: pd.DatetimeIndex,
                 results: dict, donors: np.ndarray, residuals: np.ndarray | None, audit: dict) -> list[dict]:
    expected = {(c["budget"], c["model"]) for c in configurations(frozen["round"])}
    if set(results) != expected:
        raise ValueError("Modeling context returned an incomplete/extra V14 region grid")
    entries = []
    for (budget, model), prediction in results.items():
        path = output / "predictions" / f"{stage}-{region}-{budget}-{model}.npz"
        if path.exists():
            raise FileExistsError(path)
        raw = np.asarray(prediction["raw"])
        ensemble = np.clip(raw, 0, 1000)
        arrays = dict(origins=days.as_unit("ns").asi8, point=ensemble.mean(1), ensemble=ensemble,
                      center=prediction["center"], raw_ensemble=raw, donor_indices=donors)
        if residuals is not None:
            arrays["residual_indices"] = residuals
        extras = prediction.get("archive_arrays", {})
        if set(extras) & set(arrays):
            raise ValueError("Model archive arrays overwrite protected prediction keys")
        arrays.update(extras)
        # Support the established V12 QRF result layout as well as archive_arrays.
        for key in ("conditional_residual_indices", "raw_conditional_quantiles", "quantile_levels",
                    "conditional_weights", "calibration_errors", "calibration_regions", "calibration_origins",
                    "calibration_response_available_at", "calibration_response_period_end", "residual_uniforms"):
            if key in prediction:
                arrays[key] = prediction[key]
        if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
            raise ValueError("Prediction archives cannot contain pickled objects")
        np.savez_compressed(path, **arrays)
        entries.append(dict(stage=stage, region=region, lag=7, budget=budget, model=model,
            path=str(path.relative_to(output)), sha256=sha(path), audit={**audit, **prediction.get("audit", {}),
                "donor_q_mean_max_error": prediction.get("amplitude_error"),
                "sampled_zero_creation": prediction.get("sampled_zero_creation"),
                "raw_clip_fraction": float(np.mean(raw != ensemble)),
                "clip_point_shift_mae": float(np.abs(raw.mean(1)-ensemble.mean(1)).mean())}))
    return entries


def run_phase(output: Path, data: Path, stage: str, *, round_number: int,
              prior_round: Path | None = None, weather_csv: Path | None = None,
              points_json: Path | None = None, predict_only: bool = False,
              score_only: bool = False) -> None:
    """Execute source, or explicitly predict/score targets behind both round seals."""
    if stage not in ("source", "target"):
        raise ValueError("Unknown V14 phase")
    if stage == "source" and (predict_only or score_only):
        raise ValueError("Prediction/score mode flags are target-only")
    if stage == "target" and predict_only == score_only:
        raise ValueError("Target requires exactly one of --predict-only or --score-only")
    output, data = Path(output).resolve(), Path(data).resolve()
    if not output.exists():
        if stage != "source":
            raise ValueError("Source phase must precede target")
        freeze(output, data, round_number=round_number, prior_round=prior_round,
               weather_csv=weather_csv, points_json=points_json)
    frozen = verify(output)
    if (frozen["round"] != round_number or Path(frozen["data"]) != data or
            (weather_csv is not None and Path(weather_csv).resolve() != Path(frozen["weather_csv"])) or
            (points_json is not None and Path(points_json).resolve() != Path(frozen["points_json"]))):
        raise ValueError("Input paths/round differ from freeze")
    peer = _target_source_gate(output, frozen, prior_round) if stage == "target" else None
    if score_only:
        if any((output / f"target_{suffix}").exists() for suffix in ("scores", "completed.json")):
            raise FileExistsError("V14 target scores immutable")
        seal = read(output / "target_prediction_seal.json")
        _target_prediction_gate(output, frozen, seal)
        _finish_phase(output, frozen, stage, seal, score_phase(output, frozen, stage, seal))
        return
    if any((output / f"{stage}_{suffix}").exists() for suffix in ("prediction_seal.json", "scores", "completed.json")):
        raise FileExistsError("V14 stage immutable")
    from scripts import weather_coarse_v14 as modeling
    frames, records = _load_training(frozen, stage)
    context = modeling.prepare(frozen, frames, records, stage=stage, round_number=round_number,
                               weather_policy=frozen["weather_policy"])
    queries = ({tuple(sorted(allowed_sources(r))): [r] for r in SOURCES} if stage == "source"
               else {tuple(sorted(SOURCES)): list(TARGETS)})
    entries = []
    for subset, regions in queries.items():
        for region in regions:
            days = records[region, "evaluation"]["days"]
            mixes, draws = _coarse(frozen, stage, region, days)
            results, donors, residuals, audit = context.predict_region(subset, region, mixes, draws,
                stage=stage, round_number=round_number, weather_policy=frozen["weather_policy"])
            entries.extend(_save_region(output, frozen, stage, region, days, results, donors, residuals, audit))
            print("Predicted V14 round", round_number, stage, region, flush=True)
    verify(output)
    seal = dict(created_utc=now(), round=round_number, entries=entries, complete_grid=True,
        evaluation_hourly_labels_read_preseal=False, source_2024_labels_only_preseal=True,
        target_training_performed=False, target_selection_performed=False,
        source_choices_sha256=sha(output / "source_choices.json") if stage == "target" else None,
        shape_fit_audits=context.shape_fit_audits, gate_fit_audits=context.gate_fit_audits,
        other_round_source=str(peer) if peer else None,
        other_round_source_completed_sha256=sha(peer / "source_completed.json") if peer else None)
    check_entries(output, frozen, stage, seal)
    write(output / f"{stage}_prediction_seal.json", seal)
    if not predict_only:
        _finish_phase(output, frozen, stage, seal, score_phase(output, frozen, stage, seal))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--round1", action="store_true")
    group.add_argument("--round2", action="store_true")
    parser.add_argument("--phase", choices=["source", "target"], required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--predict-only", action="store_true")
    mode.add_argument("--score-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-round", type=Path)
    parser.add_argument("--data", type=Path, default=ROOT / "results/information-value-v10-data")
    parser.add_argument("--weather-csv", type=Path)
    parser.add_argument("--points-json", type=Path)
    args = parser.parse_args()
    run_phase(args.output, args.data, args.phase, round_number=1 if args.round1 else 2,
        prior_round=args.prior_round, weather_csv=args.weather_csv, points_json=args.points_json,
        predict_only=args.predict_only, score_only=args.score_only)


if __name__ == "__main__":
    main()
