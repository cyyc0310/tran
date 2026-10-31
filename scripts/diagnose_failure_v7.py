"""Post-hoc V7 failure diagnosis: never fitting, selecting or confirming a model.

Reads already inspected V6 hourly proxies, frozen prediction ensembles and
source fitted artifacts. Oracle replacements deliberately use target truth and
are diagnostic bounds only. Every output is written into a fresh V7 directory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from transcif.data.eia930 import normalize_balance
from transcif.research.dynamic_v2 import EF, daily_profiles
from transcif.research.eia_proxy import FUELS, _bucket
from transcif.research.structural_v3 import features, delayed
from transcif.research.conditional_transfer_v6 import FEATURE_NAMES, information_features

SOURCES = ("BPAT", "CISO", "ERCO", "FPL", "ISNE", "MISO", "NYIS")
TARGETS = ("AZPS", "DUK", "LGEE", "PACE", "PSCO", "SRP")
FOCUS = ("PACE", "AZPS", "PSCO")
DELAYS = (0, 30, 90)
GEN = [f"generation_{fuel}" for fuel in FUELS]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Inputs:
    def __init__(self):
        self.hashes = {}

    def path(self, path):
        path = Path(path).resolve()
        digest = sha(path)
        if str(path) in self.hashes and self.hashes[str(path)] != digest:
            raise RuntimeError("Diagnostic input changed during execution: " + str(path))
        self.hashes[str(path)] = digest
        return path

    def json(self, path):
        return json.loads(self.path(path).read_text())

    def pickle(self, path):
        # These are local trusted, hash-bound research artifacts, not downloads.
        with self.path(path).open("rb") as handle:
            return pickle.load(handle)

    def csv(self, path):
        return pd.read_csv(self.path(path))

    def verify(self):
        if any(sha(path) != digest for path, digest in self.hashes.items()):
            raise RuntimeError("A diagnostic input changed before publication.")


def centered(value):
    value = np.asarray(value, dtype=float)
    return value - value.mean(axis=-1, keepdims=True)


def shape_summary(paths):
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 2 or paths.shape[1] != 24 or not len(paths) or not np.isfinite(paths).all():
        raise ValueError("Finite nonempty daily 24-hour paths required.")
    shape = centered(paths)
    profile = shape.mean(axis=0)
    angle = np.arange(24) * 2 * np.pi / 24
    harmonic = np.array([profile @ np.cos(angle), profile @ np.sin(angle)]) * 2 / 24
    return {"shape_rms": float(np.sqrt(np.mean(shape**2))),
        "mean_day_range": float(np.ptp(paths, axis=1).mean()),
        "mean_profile_range": float(np.ptp(profile)),
        "profile_trough_utc": int(profile.argmin()), "profile_peak_utc": int(profile.argmax()),
        "first_harmonic_amplitude": float(np.linalg.norm(harmonic)),
        "first_harmonic_peak_utc": float(np.arctan2(harmonic[1], harmonic[0]) * 24 / (2 * np.pi) % 24)}


def residual_metrics(point, truth):
    point, truth = np.asarray(point), np.asarray(truth)
    if point.shape != truth.shape or point.ndim != 2 or point.shape[1] != 24:
        raise ValueError("Matching daily point and truth paths required.")
    error = truth - point
    level = error.mean(axis=1)
    shape = centered(error)
    return {"mae": np.abs(error).mean(axis=1), "signed_level_error_truth_minus_point": level,
        "level_mae": np.abs(level), "shape_mae": np.abs(shape).mean(axis=1),
        "point_shape_rms": np.sqrt(np.mean(centered(point)**2, axis=1)),
        "truth_shape_rms": np.sqrt(np.mean(centered(truth)**2, axis=1))}


def ensemble_metrics(samples, truth):
    samples, truth = np.asarray(samples, dtype=float), np.asarray(truth, dtype=float)
    if samples.ndim != 3 or samples.shape[0] != len(truth) or samples.shape[2:] != (24,) or truth.shape[1:] != (24,):
        raise ValueError("Expected samples (days, scenarios, 24) and truth (days,24).")
    if not np.isfinite(samples).all() or not np.isfinite(truth).all() or not samples.shape[1]:
        raise ValueError("Finite nonempty samples and truth required.")
    n = samples.shape[1]
    ordered = np.sort(samples, axis=1)
    coefficient = (2 * np.arange(1, n + 1) - n - 1)[None, :, None]
    crps = (np.abs(samples - truth[:, None]).mean(axis=1)
            - np.sum(coefficient * ordered, axis=1) / n**2).mean(axis=1)
    # Match V6 PreparedDistribution's inverse empirical CDF, rather than numpy's
    # default linearly interpolated quantiles. The scenarios are equally weighted.
    lower = ordered[:, max(0, int(np.ceil(.05 * n)) - 1), :]
    upper = ordered[:, max(0, int(np.ceil(.95 * n)) - 1), :]
    score = upper - lower + 20 * np.maximum(lower - truth, 0) + 20 * np.maximum(truth - upper, 0)
    return {"crps": crps, "coverage90": ((truth >= lower) & (truth <= upper)).mean(axis=1),
        "width90": (upper - lower).mean(axis=1), "interval_score90": score.mean(axis=1),
        "lower_tail_miss": (truth < lower).mean(axis=1), "upper_tail_miss": (truth > upper).mean(axis=1),
        "ensemble_mean_level_bias": (truth - samples.mean(axis=1)).mean(axis=1),
        "ensemble_mean_shape_mae": np.abs(centered(truth) - centered(samples.mean(axis=1))).mean(axis=1)}


def oracle_replace(samples, truth, component):
    """Illegal target-truth oracle; preserves exactly the other component."""
    samples, truth = np.asarray(samples), np.asarray(truth)
    if component == "level":
        return centered(samples) + truth.mean(axis=1)[:, None, None]
    if component == "shape":
        return samples.mean(axis=2, keepdims=True) + centered(truth)[:, None, :]
    if component == "both":
        return np.broadcast_to(truth[:, None], samples.shape).copy()
    raise ValueError("Oracle component must be level, shape or both.")


def hourly_frame(frame):
    frame = frame.copy()
    frame["time"] = pd.to_datetime(frame.time, utc=True)
    if frame.time.duplicated().any() or not frame.time.eq(frame.time.dt.floor("h")).all():
        raise ValueError("Unique aligned UTC hour-start observations required.")
    if not frame.valid.isin((True, False)).all():
        raise ValueError("Explicit boolean native quality mask required.")
    return frame.sort_values("time").reset_index(drop=True)


def monthly_table(frame):
    frame = frame.copy()
    for name in ("period_start", "period_end", "available_at"):
        frame[name] = pd.to_datetime(frame[name], utc=True)
    if frame.duplicated(["region", "period_start"]).any():
        raise ValueError("Unique region/month coarse rows required.")
    return frame


def distribution_summary(frame, region, year, group, coarse):
    part = frame.loc[frame.time.dt.year.eq(year)]
    valid = part.loc[part.valid].copy()
    days, truth = daily_profiles(part)
    if not len(days):
        raise ValueError(f"No complete quality-passing days: {region}/{year}")
    generation = valid[GEN].to_numpy()
    energy = generation.sum(axis=1)
    cif = generation @ EF / energy
    mix = generation.sum(axis=0) / generation.sum()
    shares = generation / energy[:, None]
    solar = shares[:, FUELS.index("solar")]
    wind = shares[:, FUELS.index("wind")]
    solar_profile = pd.Series(solar).groupby(valid.time.dt.hour.to_numpy()).mean().reindex(range(24))
    wind_profile = pd.Series(wind).groupby(valid.time.dt.hour.to_numpy()).mean().reindex(range(24))
    record = {"group": group, "region": region, "year": year, "native_hours_in_prepared_csv": len(part),
        "valid_hours": len(valid), "valid_days": len(days), "uniform_hourly_mean_cif": float(cif.mean()),
        "energy_weighted_cif": float(np.average(cif, weights=energy)),
        "weighted_minus_uniform": float(np.average(cif, weights=energy) - cif.mean()), **shape_summary(truth),
        "cif_solar_fraction_correlation": float(np.corrcoef(cif, solar)[0, 1]) if np.std(solar) > 1e-12 else np.nan,
        "cif_wind_fraction_correlation": float(np.corrcoef(cif, wind)[0, 1]) if np.std(wind) > 1e-12 else np.nan,
        "solar_hour_fraction_peak_utc": int(solar_profile.argmax()),
        **{f"energy_share_{fuel}": float(mix[i]) for i, fuel in enumerate(FUELS)}}
    monthly, profiles = [], []
    for month, rows in valid.groupby(valid.time.dt.month):
        values = rows[GEN].to_numpy(); total = values.sum(axis=1); hourly = values @ EF / total
        released = coarse.loc[coarse.region.eq(region) & coarse.period_start.dt.year.eq(year) & coarse.period_start.dt.month.eq(month)]
        coarse_cif = float(released[[f"fuelshare_{fuel}" for fuel in FUELS]].iloc[0].to_numpy() @ EF) if len(released) else None
        weighted = float(np.average(hourly, weights=total))
        monthly.append({"group": group, "region": region, "year": year, "month": int(month),
            "valid_hours": len(rows), "uniform_cif": float(hourly.mean()), "energy_weighted_cif": weighted,
            "weighted_minus_uniform": weighted - float(hourly.mean()), "released_coarse_cif": coarse_cif,
            "released_minus_retrospective_weighted": None if coarse_cif is None else coarse_cif - weighted,
            "released_observed_hours": int(released.observed_hours.iloc[0]) if len(released) else None,
            **{f"energy_share_{fuel}": float(values[:, i].sum() / values.sum()) for i, fuel in enumerate(FUELS)}})
    for hour, value in enumerate(centered(truth).mean(axis=0)):
        profiles.append({"group": group, "region": region, "year": year, "hour_utc": hour,
                         "mean_centered_truth_cif": float(value),
                         "mean_hourly_solar_fraction": float(solar_profile.iloc[hour]),
                         "mean_hourly_wind_fraction": float(wind_profile.iloc[hour])})
    return record, monthly, profiles, days, truth


def aligned_archive(path, days, inputs):
    with np.load(inputs.path(path), allow_pickle=False) as stored:
        positions = pd.DatetimeIndex(pd.to_datetime(stored["origins"], unit="ns", utc=True)).get_indexer(days)
        if np.any(positions < 0):
            raise ValueError("Frozen predictions omit a quality-passing day.")
        return {name: stored[name][positions] for name in
                ("point", "ensemble", "corrected_point", "level_scale", "shape_scale", "bias")}


def latest_asof(coarse, region, days, delay):
    history = coarse.loc[coarse.region.eq(region)].sort_values("period_end")
    rows = []
    for day in days:
        allowed = history.loc[history.available_at.add(pd.Timedelta(days=delay)).le(day) & history.period_end.le(day)]
        if allowed.empty:
            rows.append({"latest_period_start": None, "latest_coarse_cif": np.nan, "age_days": np.nan})
        else:
            last = allowed.iloc[-1]
            rows.append({"latest_period_start": last.period_start.isoformat(),
                "latest_coarse_cif": float(last[[f"fuelshare_{fuel}" for fuel in FUELS]].to_numpy() @ EF),
                "age_days": (day - last.period_end).total_seconds() / 86400})
    return pd.DataFrame(rows)


def analog_support(model, source_frames, coarse, days, region, truth):
    """Reconstruct the existing 32-neighbor mixture; no fit or new routing rule."""
    bank = model.structural
    x, *_ = features(days, region, delayed(coarse, model.extra_delay), bank.seasonal)
    labels = []
    for source in bank.source_regions:
        training_days, _ = daily_profiles(source_frames[source].loc[source_frames[source].time.dt.year.eq(2022)])
        *_, valid = features(training_days, source, delayed(model.monthly, model.extra_delay), bank.seasonal)
        labels.extend([source] * (int(valid.sum()) if valid.any() else len(training_days)))
    if len(labels) != len(bank.analog_shape):
        raise ValueError("Reconstructed training domains do not match the frozen analog bank.")
    labels = np.asarray(labels)
    norm = np.maximum(bank.analog_x.std(axis=0), .05)
    distance = (((x[:, None] - bank.analog_x[None]) / norm)**2).mean(axis=2)
    k = min(32, len(bank.analog_x)); indices = np.argpartition(distance, k - 1, axis=1)[:, :k]
    selected = np.take_along_axis(distance, indices, axis=1)
    weights = 1 / (selected + .1); weights /= weights.sum(axis=1, keepdims=True)
    shapes = bank.analog_shape[indices]
    mixture = np.sum(weights[:, :, None] * shapes, axis=1)
    per_neighbor_rms = np.sqrt(np.mean(shapes**2, axis=2))
    mixture_rms = np.sqrt(np.mean(mixture**2, axis=1))
    rms = np.sqrt(np.mean(centered(truth)**2, axis=1))
    full_bank_rms = np.sqrt(np.mean(bank.analog_shape**2, axis=1))
    return {"minimum_standardized_neighbor_distance": float(np.min(distance, axis=1).mean()),
        "neighbor_path_rms_weighted_mean": float(np.sum(weights * per_neighbor_rms, axis=1).mean()),
        "neighbor_mean_path_rms": float(mixture_rms.mean()),
        "neighbor_phase_cancellation_ratio": float((mixture_rms / np.maximum(np.sum(weights * per_neighbor_rms, axis=1), 1e-9)).mean()),
        "truth_shape_above_2022_bank_p95_fraction": float(np.mean(rms > np.quantile(full_bank_rms, .95))),
        "truth_shape_above_2022_bank_max_fraction": float(np.mean(rms > full_bank_rms.max())),
        **{f"neighbor_weight_{source}": float(np.sum(weights * (labels[indices] == source), axis=1).mean())
           for source in bank.source_regions}}


def bank_diagnostics(calibrator, archive, truth, multiplier):
    center = archive["corrected_point"]
    level = (truth - center).mean(axis=1) / (multiplier * archive["level_scale"])
    shape = centered(truth - center) / (multiplier * archive["shape_scale"][:, None])
    bank_level, bank_shape = calibrator.level_paths_, calibrator.shape_paths_
    bank_rms, target_rms = np.sqrt(np.mean(bank_shape**2, axis=1)), np.sqrt(np.mean(shape**2, axis=1))
    return {"standardized_level_error_mean": float(level.mean()),
        "level_error_outside_source_bank_range_fraction": float(np.mean((level < bank_level.min()) | (level > bank_level.max()))),
        "level_error_outside_source_bank_p01_p99_fraction": float(np.mean((level < np.quantile(bank_level, .01)) | (level > np.quantile(bank_level, .99)))),
        "standardized_shape_rms_mean": float(target_rms.mean()),
        "shape_rms_above_source_bank_p95_fraction": float(np.mean(target_rms > np.quantile(bank_rms, .95))),
        "shape_rms_above_source_bank_max_fraction": float(np.mean(target_rms > bank_rms.max())),
        "shape_hour_error_outside_source_bank_range_fraction": float(np.mean((shape < bank_shape.min(axis=0)) | (shape > bank_shape.max(axis=0)))),
        "standardized_diagnostic_note": "corrected center and factor-scaled preprojection component scales; source bank is unprojected"}


def native_quality(audit, hourly, inputs):
    # UTC Jan 1 may begin in the prior local-year Jul-Dec native file. Use the
    # actual preparation snapshot audit, never a filename-year approximation.
    snapshots = {snapshot for region in TARGETS for snapshot in audit["regions"][region]["2026"]["source_snapshots"]}
    sources = {record["source_snapshot"]: record["path"] for record in audit["native_sources"]}
    if not snapshots or not snapshots.issubset(sources):
        raise ValueError("Audited 2026 source snapshots are missing from native provenance.")
    paths = [sources[snapshot] for snapshot in sorted(snapshots)]
    frames = []
    for path in paths:
        raw = inputs.csv(path)
        frames.append(normalize_balance(raw.loc[raw["Balancing Authority"].isin(TARGETS)], TARGETS))
    native = pd.concat(frames, ignore_index=True)
    if native.duplicated(["region", "valid_at"]).any():
        raise ValueError("Duplicate native region/hour across diagnostic snapshots.")
    records = []
    for region in TARGETS:
        part = native.loc[native.region.eq(region)].copy()
        retained = pd.DatetimeIndex(hourly[region].loc[hourly[region].time.dt.year.eq(2026) & hourly[region].valid, "time"])
        part = part.loc[part.valid_at.isin(retained)]
        if len(part) != len(retained) or set(part.valid_at) != set(retained):
            raise ValueError("Native retained timestamps must equal the complete prepared valid-hour set.")
        columns = [column for column in part if column.startswith("generation_")]
        modeled = [column for column in columns if _bucket(column) is not None]
        excluded = [column for column in columns if _bucket(column) is None]
        positive = float(part[modeled].clip(lower=0).sum().sum())
        clipped = float(-part[modeled].clip(upper=0).sum().sum())
        signed = part[columns].sum(axis=1)
        residual = (signed - part.net_generation_mw).abs()
        records.append({"region": region, "retained_native_hours": len(part),
            "clipped_negative_generation_mwh": clipped, "clipped_fraction_positive_proxy_energy": clipped / positive,
            "excluded_storage_unknown_signed_mwh": float(part[excluded].sum().sum()),
            "excluded_signed_fraction_positive_proxy_energy": float(part[excluded].sum().sum() / positive),
            "max_abs_signed_native_balance_residual_mw": float(residual.max()),
            "retained_native_balance_failure_hours": int((residual > np.maximum(10, .01 * part.net_generation_mw)).sum())})
    return records


def diagnose(source_run, external_run, base_run, output, *, method="harmonic_ridge", calibration="structured_x1p25"):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("V7 diagnostic outputs require a fresh directory.")
    inputs = Inputs()
    inputs.path(__file__)
    source_run, external_run, base_run = [Path(path).resolve() for path in (source_run, external_run, base_run)]
    source_manifest = inputs.json(source_run / "prefit_freeze.json")
    for path, digest in source_manifest["code_hashes"].items():
        source_path = Path(path)
        if source_path.suffix == ".py" and source_path.is_relative_to(ROOT / "src/transcif"):
            if sha(inputs.path(source_path)) != digest:
                raise ValueError("V6 model dependency changed before diagnostic inference: " + path)
    external_manifest = inputs.json(external_run / "prefit_freeze.json")
    base_config = inputs.json(base_run / "run_config.json")
    source_config = inputs.json(source_run / "run_config.json")
    regional_scores = {str(root): inputs.csv(root / "scores/region_metrics.csv") for root in (source_run, external_run)}
    for root in (source_run, external_run):
        inputs.json(root / "fit_seal.json"); inputs.json(root / "all_predictions_sealed.json")
        inputs.json(root / "scores/postrun_verification.json")
    source_data, target_data = Path(source_manifest["data"]), Path(external_manifest["external_data"])
    target_audit = inputs.json(target_data / "audit.json")
    coarse = monthly_table(pd.concat([inputs.csv(source_data / "monthly.csv"), inputs.csv(target_data / "monthly.csv")], ignore_index=True))
    source_frames = {region: hourly_frame(inputs.csv(source_data / f"{region}_hourly.csv")) for region in SOURCES}
    target_frames = {region: hourly_frame(inputs.csv(target_data / f"{region}_hourly.csv")) for region in TARGETS}
    summaries, monthly, profiles, truths = [], [], [], {}
    categories = [("source", SOURCES, (2024, 2025), source_frames),
                  ("point_fit", tuple(base_config["fit_regions"]), (2022,), source_frames),
                  ("target_history", TARGETS, (2024, 2025), target_frames), ("target", TARGETS, (2026,), target_frames)]
    for group, regions, years, frames in categories:
        for region in regions:
            for year in years:
                record, months, hours, days, truth = distribution_summary(frames[region], region, year, group, coarse)
                summaries.append(record); monthly.extend(months); profiles.extend(hours); truths[group, region, year] = (days, truth)
    daily, oracle, support, banks, feature_rows, training, forecast_shapes = [], [], [], [], [], [], []
    multiplier = source_config["parameters"][calibration]["multiplier"]
    mode = source_config["parameters"][calibration]["mode"]
    for delay in DELAYS:
        base = inputs.pickle(source_run / f"base/round2-{delay}.pkl")["model"]
        if set(base.structural.source_regions) != set(base_config["fit_regions"]):
            raise ValueError("Fitted point domains differ from the original point-run manifest.")
        calibrator = inputs.pickle(source_run / f"fit/conditional_v6-{delay}.pkl")["calibrators"][method, mode]
        bank_shape_rms = np.sqrt(np.mean(calibrator.shape_paths_**2, axis=1))
        training.append({"delay": delay, "point_fit_year": base_config["fit_year"],
            "point_fit_regions_from_run_config": base_config["fit_regions"],
            "point_fit_regions_from_fitted_object": list(base.structural.source_regions),
            "analog_bank_days": len(base.structural.analog_shape), "residual_source_regions": sorted(calibrator.source_regions),
            "proper_bank_audit": calibrator.audit_,
            "point_bank_shape_rms_quantiles": {str(q): float(np.quantile(np.sqrt(np.mean(base.structural.analog_shape**2, axis=1)), q)) for q in (0, .5, .95, .99, 1)},
            "monthly_template_shape_rms": np.sqrt(np.mean(centered(base.structural.template)**2, axis=1)).tolist()})
        for region in sorted(calibrator.source_regions):
            take = calibrator.regions_ == region
            banks.append({"delay": delay, "source_region": region, "bank_days": int(take.sum()),
                "level_path_min": float(calibrator.level_paths_[take].min()), "level_path_max": float(calibrator.level_paths_[take].max()),
                "shape_path_rms_mean": float(bank_shape_rms[take].mean()), "shape_path_rms_p95": float(np.quantile(bank_shape_rms[take], .95)),
                "shape_path_rms_max": float(bank_shape_rms[take].max())})
        contexts = [("source", region, 2025, source_run) for region in SOURCES] + [("target", region, 2026, external_run) for region in TARGETS]
        for group, region, year, run in contexts:
            days, truth = truths[group, region, year]
            archive = aligned_archive(run / "predictions" / f"{region}-{delay}-{method}-{calibration}.npz", days, inputs)
            raw = residual_metrics(archive["point"], truth)
            ensemble_mean = residual_metrics(archive["ensemble"].mean(axis=1), truth)
            base_context = {"group": group, "region": region, "year": year, "delay": delay}
            truth_profile = centered(truth).mean(axis=0)
            for name, value in (("raw_point", archive["point"]), ("ensemble_mean", archive["ensemble"].mean(axis=1))):
                profile = centered(value).mean(axis=0)
                denominator = max(np.linalg.norm(profile) * np.linalg.norm(truth_profile), 1e-12)
                correlation = np.asarray([truth_profile @ np.roll(profile, shift) / denominator for shift in range(24)])
                forecast_shapes.append({**base_context, "prediction": name, **shape_summary(value),
                    "mean_profile_cosine_with_truth": float(correlation[0]),
                    "diagnostic_best_circular_shift_hours": int(correlation.argmax()),
                    "diagnostic_best_circular_profile_cosine": float(correlation.max())})
            table = pd.DataFrame({"origin": days, "month": days.month, **{f"raw_{key}": value for key, value in raw.items()},
                **{f"ensemble_mean_{key}": value for key, value in ensemble_mean.items()},
                **ensemble_metrics(archive["ensemble"], truth), "truth_daily_level": truth.mean(axis=1),
                "point_daily_level": archive["point"].mean(axis=1), "bias_correction": archive["bias"],
                "conditional_level_scale": archive["level_scale"], "conditional_shape_scale": archive["shape_scale"]})
            asof = latest_asof(coarse, region, days, delay)
            table = pd.concat([table, asof], axis=1)
            table["latest_coarse_minus_truth_level"] = table.latest_coarse_cif - table.truth_daily_level
            for key, value in base_context.items(): table[key] = value
            daily.append(table)
            for component in ("none", "level", "shape", "both"):
                samples = archive["ensemble"] if component == "none" else oracle_replace(archive["ensemble"], truth, component)
                result = ensemble_metrics(samples, truth)
                if component == "none":
                    scored = regional_scores[str(run)]
                    scored = scored.loc[scored.region.eq(region) & scored.delay.eq(delay)
                        & scored.method.eq(method) & scored.calibration.eq(calibration)]
                    if len(scored) != 1:
                        raise ValueError("Exactly one corresponding frozen regional score is required.")
                    for ours, original in (("crps", "crps"), ("coverage90", "empirical_coverage90"),
                                           ("width90", "empirical_width90"), ("interval_score90", "empirical_interval_score90")):
                        if not np.isclose(result[ours].mean(), scored.iloc[0][original], rtol=1e-9, atol=1e-9):
                            raise ValueError("V7 failed to reproduce the frozen V6 score: " + original)
                oracle.append({**base_context, "oracle": component, "diagnostic_only_illegal_if_not_none": component != "none",
                    **{key: float(value.mean()) for key, value in result.items()},
                    "outside_proxy_support_fraction": float(np.mean((samples < 0) | (samples > EF.max())))})
            support.append({**base_context, **analog_support(base, source_frames, coarse, days, region, truth),
                **bank_diagnostics(calibrator, archive, truth, multiplier)})
            x, names = information_features(days, region, coarse, delay)
            proper = calibrator.audit_["feature_audit"]["proper"]["ranges"]
            for index, name in enumerate(names):
                feature_rows.append({**base_context, "feature": name, "mean": float(x[:, index].mean()),
                    "min": float(x[:, index].min()), "max": float(x[:, index].max()),
                    "proper_min": proper[name]["min"], "proper_max": proper[name]["max"],
                    "outside_proper_range_fraction": float(np.mean((x[:, index] < proper[name]["min"]) | (x[:, index] > proper[name]["max"])))})
            if group == "target":
                for name, value in (("raw_point", archive["point"]), ("ensemble_mean", archive["ensemble"].mean(axis=1))):
                    for hour, cif in enumerate(centered(value).mean(axis=0)):
                        profiles.append({**base_context, "profile_type": name, "hour_utc": hour, "mean_centered_truth_cif": float(cif)})
        # Source-2024 raw point inference reuses existing bytes; no coefficients
        # or residual banks are refit. Only base predictions are evaluated here.
        for region in SOURCES:
            days, truth = truths["source", region, 2024]
            prediction = base.predict(days, region, coarse)[method]
            raw = residual_metrics(prediction["point"], truth)
            table = pd.DataFrame({"origin": days, "month": days.month,
                **{f"raw_{key}": value for key, value in raw.items()},
                "truth_daily_level": truth.mean(axis=1), "point_daily_level": prediction["point"].mean(axis=1)})
            table = pd.concat([table, latest_asof(coarse, region, days, delay)], axis=1)
            table["latest_coarse_minus_truth_level"] = table.latest_coarse_cif - table.truth_daily_level
            for key, value in {"group": "source", "region": region, "year": 2024, "delay": delay}.items(): table[key] = value
            daily.append(table)
        print(f"Diagnosed fixed source/target paths and source bank: extra delay={delay}", flush=True)
    native = native_quality(target_audit, target_frames, inputs)
    inputs.verify()
    output.mkdir(parents=True)
    daily = pd.concat(daily, ignore_index=True)
    tables = {"composition_quality_shape.csv": pd.DataFrame(summaries), "monthly_aggregation_gap.csv": pd.DataFrame(monthly),
        "mean_utc_profiles.csv": pd.DataFrame(profiles), "daily_residual_and_asof.csv": daily,
        "oracle_component_replacements.csv": pd.DataFrame(oracle), "source_bank_support.csv": pd.DataFrame(support),
        "bank_by_source_region.csv": pd.DataFrame(banks), "feature_support.csv": pd.DataFrame(feature_rows),
        "native_2026_quality.csv": pd.DataFrame(native)}
    tables["forecast_shape_phase.csv"] = pd.DataFrame(forecast_shapes)
    numeric = [column for column in daily.select_dtypes(include="number") if column not in ("year", "delay", "month")]
    tables["residual_and_asof_by_month_delay.csv"] = daily.groupby(["group", "region", "year", "delay", "month"])[numeric].mean().reset_index()
    tables["residual_by_region_delay.csv"] = daily.groupby(["group", "region", "year", "delay"])[numeric].mean().reset_index()
    for name, table in tables.items(): table.to_csv(output / name, index=False)
    (output / "training_and_bank_provenance.json").write_text(json.dumps(training, indent=2, allow_nan=False) + "\n")
    figures(tables, output)
    inputs.verify()
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "post-hoc diagnosis after V6 target scores were inspected",
        "target_hourly_used_for_fitting": False, "model_refit": False, "new_confirmation": False,
        "oracle_replacements_are_legal_methods": False, "oracle_projection": "none, to preserve exact components; support violations reported",
        "method": method, "calibration": calibration, "multiplier": multiplier,
        "original_v6_regional_scores_reproduced": True,
        "frozen_v6_package_code_hashes_rechecked": True,
        "environment": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__},
        "empirical_quantiles": "inverse empirical CDF, same as V6 PreparedDistribution",
        "fixed_ef": EF.tolist(), "fuels": list(FUELS), "time_convention": "native-normalized UTC hour start; phases are UTC, not local time",
        "input_sha256": inputs.hashes, "script_sha256": sha(__file__),
        "output_sha256": {path.name: sha(path) for path in sorted(output.iterdir()) if path.is_file()},
        "limitations": ["Fixed-EF positive reported-generation proxy, not full grid emissions truth.",
            "Source and target quality masks select different day populations; counts are explicit.",
            "Monthly gap compares matched retained hours; exported monthly prefixes may retain different hours.",
            "Oracle replacements use target truth, remove component uncertainty, and are deliberately illegal for forecasting.",
            "2024 raw point results reuse frozen 2022 point fits; no 2024 ensemble validation is claimed.",
            "Bank standardized support is computed before projection; actual ensembles are already projected.",
            "Observed support/range violations and component correlations do not prove a unique causal mechanism."]}
    (output / "diagnosis_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return metadata


def figures(tables, output):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib-cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output / ".cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    profiles = tables["mean_utc_profiles.csv"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, region in zip(axes, FOCUS):
        truth = profiles.loc[profiles.group.eq("target") & profiles.region.eq(region) & profiles.profile_type.isna()]
        point = profiles.loc[profiles.group.eq("target") & profiles.region.eq(region) & profiles.profile_type.eq("raw_point") & profiles.delay.eq(0)]
        ensemble = profiles.loc[profiles.group.eq("target") & profiles.region.eq(region) & profiles.profile_type.eq("ensemble_mean") & profiles.delay.eq(0)]
        for table, color, label in ((truth, "#111111", "Observed truth (diagnostic)"), (point, "#0072B2", "Frozen raw point"),
                                    (ensemble, "#D55E00", "Frozen ensemble mean")):
            ax.plot(table.hour_utc, table.mean_centered_truth_cif, color=color, label=label)
        ax.axhline(0, color="#999999", linewidth=.5)
        ax.set(title=region, xlabel="UTC hour", ylabel="Mean centered fixed-EF proxy CIF")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("V7 post-hoc diagnosis: target shape amplitude and phase; extra delay 0")
    for ext in ("png", "pdf"): fig.savefig(output / f"shape_phase_diagnosis.{ext}", dpi=250, bbox_inches="tight")
    plt.close(fig)
    summary = tables["composition_quality_shape.csv"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    for group, marker, color in (("point_fit", "s", "#6B6B6B"), ("source", "o", "#0072B2"), ("target", "^", "#D55E00")):
        part = summary.loc[summary.group.eq(group)]
        ax.scatter(part.energy_share_solar + part.energy_share_wind, part.shape_rms, marker=marker, color=color, label=group)
        for row in part.itertuples(): ax.annotate(f"{row.region}/{row.year}", (row.energy_share_solar + row.energy_share_wind, row.shape_rms), fontsize=6)
    ax.set(xlabel="Solar + wind fraction of retained generation energy", ylabel="Daily centered CIF RMS", title="Observed domain support; post-hoc descriptive diagnosis")
    ax.legend(frameon=False)
    for ext in ("png", "pdf"): fig.savefig(output / f"composition_shape_support.{ext}", dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=ROOT / "results/conditional-transfer-v6-round1-source2025")
    parser.add_argument("--external-run", type=Path, default=ROOT / "results/conditional-transfer-v6-round1-external2026h1")
    parser.add_argument("--base-run", type=Path, default=ROOT / "results/transfer-v4-round2-source2024")
    parser.add_argument("--output", type=Path, default=ROOT / "results/failure-diagnosis-v7")
    args = parser.parse_args()
    result = diagnose(args.source_run, args.external_run, args.base_run, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "diagnostic_only": True, "inputs": len(result["input_sha256"])}))


if __name__ == "__main__":
    main()
