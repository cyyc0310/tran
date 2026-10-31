"""Source-fixed joint-path, shape, and UTC phase diagnostics for completed V7.

Only completed, fully verified target runs are accepted. These previously viewed
targets remain development retests; this script never selects on target scores.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "scripts/analyze_fuel_transport_v7.py"
_SPEC = importlib.util.spec_from_file_location("v7_path_verified_analysis", HELPER_PATH)
ANALYSIS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ANALYSIS)
ROLES = ("primary", "exact_direct", "exact_independent", "exact_pooled", "simple_anchor", "baseline")
SCOPES = ("1", "2")
PLOT_REGIONS = ("PACE", "AZPS", "PSCO")
FLAT_RMS_THRESHOLD = 1.  # gCO2/kWh, fixed diagnostic threshold; never tuned.
PAIR_I, PAIR_J = np.triu_indices(24, 1)
MEAN_METRICS = ("energy_score", "variogram_p05", "point_shape_rms", "truth_shape_rms",
    "ensemble_mean_shape_rms", "point_shape_mae", "ensemble_mean_shape_mae", "point_level_bias",
    "ensemble_mean_level_bias", "point_shape_cosine", "ensemble_mean_shape_cosine",
    "point_daymin_circular_first", "ensemble_mean_daymin_circular_first", "point_daymin_circular_stable",
    "ensemble_mean_daymin_circular_stable", "point_phase_valid", "ensemble_mean_phase_valid",
    "point_flat", "truth_flat", "ensemble_mean_flat")
CELL = ["scope", "role", "source_status", "round", "name", "factor", "region", "delay"]


def _path(values, name):
    values = np.asarray(values, dtype=float)
    if values.shape != (24,) or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite 24-hour path.")
    return values


def _particles(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 24 or not len(values) or not np.isfinite(values).all():
        raise ValueError("ensemble must be finite nonempty (S,24).")
    return values


def energy_score(ensemble, truth):
    """mean_s ||X_s-y||/sqrt24 - sum_(s<t)||X_s-X_t||/(S² sqrt24).

    Equivalent to subtracting .5/S² times the full ordered-pair sum, including
    zero diagonal self-pairs. All finite-distribution pairs are evaluated exactly.
    """
    ensemble, truth = _particles(ensemble), _path(truth, "truth")
    first = np.sqrt(np.mean((ensemble - truth)**2, axis=1)).mean()
    second = pdist(ensemble, metric="euclidean").sum() / (len(ensemble)**2 * np.sqrt(24.))
    return float(max(first - second, 0.))


def variogram_score(ensemble, truth):
    """Mean over 276 unordered hours of (|y_i-y_j|^.5 - E|X_i-X_j|^.5)^2."""
    ensemble, truth = _particles(ensemble), _path(truth, "truth")
    observed = np.sqrt(np.abs(truth[PAIR_I] - truth[PAIR_J]))
    expected = np.sqrt(np.abs(ensemble[:, PAIR_I] - ensemble[:, PAIR_J])).mean(axis=0)
    return float(np.mean((observed - expected)**2))


def circular_hour_distance(left, right):
    difference = abs(int(left) - int(right))
    if not 0 <= int(left) < 24 or not 0 <= int(right) < 24:
        raise ValueError("UTC hour indices must lie in [0,23].")
    return min(difference, 24 - difference)


def shape_summary(point, truth, flat_threshold=FLAT_RMS_THRESHOLD):
    """Raw UTC argmin is retained; stable phase excludes flat paths/min ties."""
    point, truth = _path(point, "point"), _path(truth, "truth")
    if not np.isfinite(flat_threshold) or flat_threshold < 0:
        raise ValueError("flat_threshold must be finite and nonnegative.")
    point_centered, truth_centered = point - point.mean(), truth - truth.mean()
    point_rms, truth_rms = np.sqrt(np.mean(point_centered**2)), np.sqrt(np.mean(truth_centered**2))
    point_flat, truth_flat = point_rms <= flat_threshold, truth_rms <= flat_threshold
    point_hour, truth_hour = int(np.argmin(point)), int(np.argmin(truth))
    point_ties = int(np.isclose(point, point.min(), atol=1e-8, rtol=1e-10).sum())
    truth_ties = int(np.isclose(truth, truth.min(), atol=1e-8, rtol=1e-10).sum())
    stable = not (point_flat or truth_flat or point_ties > 1 or truth_ties > 1)
    cosine = (float(np.dot(point_centered, truth_centered) / (24 * point_rms * truth_rms))
              if not point_flat and not truth_flat else np.nan)
    distance = circular_hour_distance(point_hour, truth_hour)
    return {"point_shape_rms": float(point_rms), "truth_shape_rms": float(truth_rms),
        "point_shape_mae": float(np.abs(point_centered - truth_centered).mean()),
        "point_level_bias": float(point.mean() - truth.mean()), "point_shape_cosine": cosine,
        "point_daymin_hour": point_hour, "truth_daymin_hour": truth_hour,
        "point_daymin_circular_first": distance, "point_daymin_circular_stable": float(distance) if stable else np.nan,
        "point_phase_valid": stable, "point_flat": bool(point_flat), "truth_flat": bool(truth_flat),
        "point_min_ties": point_ties, "truth_min_ties": truth_ties}


def daily_metrics(point, ensemble, truth):
    """Separate stored raw point from the factor/projected scenario mean."""
    ensemble = _particles(ensemble)
    result = {"energy_score": energy_score(ensemble, truth), "variogram_p05": variogram_score(ensemble, truth),
              **shape_summary(point, truth)}
    mean_result = shape_summary(ensemble.mean(axis=0), truth)
    result.update({key.replace("point_", "ensemble_mean_", 1): value
                   for key, value in mean_result.items() if key.startswith("point_")})
    return result


def scale_scenarios(ensemble, factor):
    """Exactly the producer's spread-factor and [0,1000] projection operator."""
    ensemble = np.asarray(ensemble, dtype=float)
    if ensemble.ndim != 3 or ensemble.shape[-1] != 24 or not np.isfinite(ensemble).all():
        raise ValueError("Daily ensemble batches must be finite (N,S,24).")
    if not np.isfinite(factor) or factor < 0:
        raise ValueError("factor must be finite and nonnegative.")
    mean = ensemble.mean(axis=1, keepdims=True)
    return np.clip(mean + factor * (ensemble - mean), 0., 1000.)


def _mean_profile_metrics(point, ensemble_mean, truth):
    values = shape_summary(point, truth)
    result = {"mean_profile_" + key: value for key, value in values.items()}
    mean_values = shape_summary(ensemble_mean, truth)
    result.update({"mean_profile_" + key.replace("point_", "ensemble_mean_", 1): value
                   for key, value in mean_values.items() if key.startswith("point_")})
    return result


def _expected_days(verified, configurations):
    result = {}
    days = verified["target"]["day"]
    for config in configurations:
        selected = ANALYSIS.subset(days, config)
        for (region, delay), group in selected.groupby(["region", "delay"]):
            key = (region, int(delay), int(config["round"]), config["name"], float(config["factor"]))
            result[key] = set(pd.DatetimeIndex(pd.to_datetime(group.origin, utc=True)).as_unit("ns").asi8)
    return result


def figures(profiles, roles, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for scope in SCOPES:
        for column, label in (("point_mean", "stored raw point"), ("ensemble_mean", "factor/projected scenario mean")):
            fig, axes = plt.subplots(3, 3, figsize=(15, 10), constrained_layout=True)
            for row, region in enumerate(PLOT_REGIONS):
                for col, delay in enumerate(ANALYSIS.RUNNER.DELAYS):
                    ax = axes[row, col]
                    cell = profiles[profiles.scope.eq(scope) & profiles.region.eq(region) & profiles.delay.eq(delay)]
                    if cell.empty:
                        ax.set_visible(False)
                        continue
                    truth = cell[cell.role.eq("primary")].sort_values("hour_utc")
                    ax.plot(truth.hour_utc, truth.truth_mean, color="black", linestyle="--", linewidth=2, label="Truth")
                    for role in ROLES:
                        group = cell[cell.role.eq(role)].sort_values("hour_utc")
                        source_failed = roles[scope][role]["status"] == "failed"
                        name = ANALYSIS.ROLE_LABELS[role] + (" [source failed]" if source_failed else "")
                        ax.plot(group.hour_utc, group[column], color=ANALYSIS.ROLE_COLORS[role], label=name, linewidth=1.4)
                    ax.set(title=f"{region}, delay {delay}d", xlabel="UTC hour", ylabel="gCO2/kWh", xticks=[0, 6, 12, 18, 23])
            handles, labels = next((ax.get_legend_handles_labels() for ax in axes.flat if ax.get_visible()), ([], []))
            fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=8)
            fig.suptitle(f"V7 round {scope}: {label}, source-fixed roles\n"
                         "Viewed development targets; retained-day averages. Flat/tied minima do not identify phase.", fontsize=11)
            for extension in ("png", "pdf"):
                fig.savefig(output / f"round{scope}_{column}_profiles.{extension}", dpi=170)
            plt.close(fig)


def diagnose(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new path-diagnostics output directory.")
    # This helper verifies completed/source choices, both seals, all archives,
    # fits, frozen input/code bytes, and daily/aggregate score consistency. It
    # never parses hourly labels. No partial target run can pass this boundary.
    verified = ANALYSIS.verify_run(run, source_only=False)
    roles = verified["roles"]
    fixed = [(scope, role, roles[scope][role]) for scope in SCOPES for role in ROLES]
    configurations = [record["configuration"] for _, _, record in fixed]
    expected = _expected_days(verified, configurations)
    entries = {(entry["region"], int(entry["delay"]), int(entry["round"]), entry["name"]): entry
               for entry in verified["target_seal"]["entries"]}
    frozen = verified["frozen"]
    target = Path(frozen["target"])
    inputs = {**frozen["code"], **frozen["inputs"]}
    for relative in ("prefit_freeze.json", "completed.json", "source_choices.json", "source_prediction_seal.json",
                     "target_prediction_seal.json", "source_scores/score_seal.json", "target_scores/score_seal.json"):
        inputs[str(run / relative)] = ANALYSIS.sha(run / relative)
    for path in (Path(__file__).resolve(), HELPER_PATH, ANALYSIS.RUNNER_PATH):
        inputs[str(path)] = ANALYSIS.sha(path)
    # Free the large all-grid source/target daily score tables after retaining
    # only fixed-choice day support. Role identities and seals remain in scope.
    del verified["source"], verified["target"]
    output.mkdir(parents=True)
    truths = {}
    for region in ANALYSIS.RUNNER.TARGETS:
        path = target / f"{region}_hourly.csv"
        if str(path) not in frozen["inputs"] or ANALYSIS.sha(path) != frozen["inputs"][str(path)]:
            raise ValueError("Target hourly input is not frozen or changed before label access.")
        days, values = ANALYSIS.RUNNER.daily_profiles(ANALYSIS.RUNNER.read_year(path, 2026))
        if days.has_duplicates:
            raise ValueError("Duplicate target truth day.")
        truths[region] = {int(day.value): value for day, value in zip(days, values)}
    cache, daily_rows, profile_rows, group_rows = {}, [], [], []
    for scope, role, record in fixed:
        config = record["configuration"]
        for region in ANALYSIS.RUNNER.TARGETS:
            for delay in ANALYSIS.RUNNER.DELAYS:
                key = (region, delay, int(config["round"]), config["name"], float(config["factor"]))
                if key not in cache:
                    entry = entries[key[:4]]
                    if entry["interval_only"]:
                        raise ValueError("Path diagnostics cannot fabricate a CQR ensemble.")
                    path = ANALYSIS.member(run, entry["path"])
                    if ANALYSIS.sha(path) != entry["sha256"]:
                        raise ValueError("Selected prediction changed before loading.")
                    inputs[str(path)] = entry["sha256"]
                    with np.load(path, allow_pickle=False) as archive:
                        origins, points, paths = archive["origins"], archive["point"], archive["ensemble"]
                    if (origins.ndim != 1 or len(np.unique(origins)) != len(origins)
                            or points.shape != (len(origins), 24) or paths.shape != (len(origins), ANALYSIS.RUNNER.N_SAMPLES, 24)
                            or not np.isfinite(points).all() or not np.isfinite(paths).all()):
                        raise ValueError("Invalid sealed point/ensemble/origin arrays.")
                    indices = [i for i, origin in enumerate(origins) if int(origin) in truths[region]]
                    if {int(origins[i]) for i in indices} != expected.get(key):
                        raise ValueError("Path diagnostics and sealed scores have different retained days.")
                    if not indices:
                        raise ValueError("No retained days for a fixed source configuration.")
                    projected = scale_scenarios(paths[indices], config["factor"])
                    point, truth = points[indices], np.stack([truths[region][int(origins[i])] for i in indices])
                    rows = [dict(origin=str(pd.Timestamp(int(origins[i]), tz="UTC")),
                                 **daily_metrics(point[j], projected[j], truth[j])) for j, i in enumerate(indices)]
                    average_point, average_truth, average_ensemble = point.mean(0), truth.mean(0), projected.mean(axis=(0, 1))
                    cache[key] = (pd.DataFrame(rows), average_point, average_truth, average_ensemble)
                days, point_profile, truth_profile, ensemble_profile = cache[key]
                context = {"scope": scope, "role": role, "source_status": record["status"], **config, "region": region, "delay": delay}
                daily_rows.append(days.assign(**context))
                group_rows.append({**context, "n_days": len(days), **days[list(MEAN_METRICS)].mean().to_dict(),
                    "point_phase_valid_days": int(days.point_phase_valid.sum()),
                    "ensemble_mean_phase_valid_days": int(days.ensemble_mean_phase_valid.sum()),
                    **_mean_profile_metrics(point_profile, ensemble_profile, truth_profile)})
                profile_rows.extend({**context, "hour_utc": hour, "point_mean": point_profile[hour],
                                     "ensemble_mean": ensemble_profile[hour], "truth_mean": truth_profile[hour]} for hour in range(24))
    daily, regional, profiles = pd.concat(daily_rows, ignore_index=True), pd.DataFrame(group_rows), pd.DataFrame(profile_rows)
    daily.to_csv(output / "day_path_metrics.csv", index=False)
    regional.to_csv(output / "region_delay_path_metrics.csv", index=False)
    profiles.to_csv(output / "mean_profiles.csv", index=False)
    macro_rows = []
    macro_profiles = profiles.groupby(["scope", "role", "hour_utc"])[["point_mean", "ensemble_mean", "truth_mean"]].mean().reset_index()
    macro_profiles.to_csv(output / "macro_mean_profiles.csv", index=False)
    for (scope, role), cells in regional.groupby(["scope", "role"], sort=False):
        profile = macro_profiles[macro_profiles.scope.eq(scope) & macro_profiles.role.eq(role)].sort_values("hour_utc")
        macro_rows.append({"scope": scope, "role": role, "source_status": roles[scope][role]["status"],
            **roles[scope][role]["configuration"], "n_region_delay_cells": len(cells), "n_days_total": int(cells.n_days.sum()),
            **cells[list(MEAN_METRICS)].mean().to_dict(),
            "point_phase_valid_days": int(cells.point_phase_valid_days.sum()),
            "ensemble_mean_phase_valid_days": int(cells.ensemble_mean_phase_valid_days.sum()),
            **_mean_profile_metrics(profile.point_mean.to_numpy(), profile.ensemble_mean.to_numpy(), profile.truth_mean.to_numpy())})
    pd.DataFrame(macro_rows).to_csv(output / "macro_path_metrics.csv", index=False)
    figures(profiles, roles, output)
    (output / "READ_ME.md").write_text(
        "# V7 source-fixed path diagnostics\n\n"
        "These are development retests on previously viewed targets, not independent confirmation. "
        "Only the source-fixed two-round primary, exact direct/independent/pooled controls, simple anchor and baseline are analyzed. "
        "Repeated configuration identities are computed once but all semantic role rows remain. No target re-selection occurs.\n\n"
        "Energy score uses normalized L2 norm ||x||/sqrt(24) and all finite-distribution pairs: "
        "mean_s d(X_s,y) - (0.5/S^2) sum_{s,t} d(X_s,X_t), including zero self-pairs. "
        "Variogram p=0.5 uses equal weight 1/276 on each unordered UTC hour pair i<j: "
        "mean_pairs (|y_i-y_j|^0.5 - mean_s |X_si-X_sj|^0.5)^2.\n\n"
        "Scenarios use the source-fixed multiplier around the original scenario mean then clip to [0,1000], "
        "exactly as producer scoring. Stored raw point and resulting scenario mean are reported separately; "
        "they need not coincide, especially for residual-bank baselines or after clipping. "
        "Shape RMS and MAE center each daily path separately. Signed level bias is predicted minus truth mean.\n\n"
        "All minima use UTC hour and first argmin. Circular distance is min(|h1-h2|,24-|h1-h2|). "
        "Flat/near-flat profiles (centered RMS <=1 gCO2/kWh) and tied minima do not identify a stable phase: "
        "raw first-argmin distances remain descriptive, while stable-phase distances/cosines are missing as appropriate "
        "and valid-day counts/flat/tie flags remain explicit. Cosines compare centered shapes. "
        "Mean-profile diagnostics operate on average retained-day profiles, separately from mean daily metrics.\n\n"
        "Macro daily metrics first average each region/delay cell, then weight cells equally. Macro mean profiles "
        "also weight region/delay profiles equally. NaN stable-phase statistics omit undefined cells; inspect valid-day counts.\n")
    # Recheck every numeric input and provenance binding used by this analysis.
    # All producer archive members were additionally verified before label read.
    for path, digest in inputs.items():
        if ANALYSIS.sha(path) != digest:
            raise ValueError("Analysis input changed during path scoring: " + path)
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "run": str(run),
        "scope": "Previously viewed 2026H1 development retest; no independent confirmation",
        "target_selection_performed": False, "target_training_performed": False,
        "all_prediction_archives_verified_before_label_read": True,
        "fixed_roles": {scope: {role: roles[scope][role] for role in ROLES} for scope in SCOPES},
        "selected_configuration_cells_cached": len(cache), "role_configuration_cells_reported": len(regional),
        "energy_score": "normalized L2 /sqrt24; exact finite S^2 ordered-pair expectation with self-pairs",
        "variogram": {"p": .5, "unordered_pairs": 276, "uniform_pair_weight": 1 / 276},
        "flat_rms_threshold": FLAT_RMS_THRESHOLD, "minimum_tie_atol": 1e-8, "minimum_tie_rtol": 1e-10,
        "point": "stored raw archive point", "ensemble_mean": "source factor then projection; mean can differ from raw point",
        "input_hashes": inputs, "analysis_environment": {"python": sys.version,
            **{name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "matplotlib")}},
        "output_hashes": {path.name: ANALYSIS.sha(path) for path in sorted(output.iterdir()) if path.is_file()}}
    ANALYSIS.write_json(output / "diagnostics_metadata.json", metadata)
    (output / "diagnostics_metadata.sha256").write_text(ANALYSIS.sha(output / "diagnostics_metadata.json") + "\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.run, args.output)
