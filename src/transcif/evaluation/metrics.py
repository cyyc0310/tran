"""Point, empirical-distribution and path diagnostics (no coverage guarantees)."""

import numpy as np


def ensemble_crps(truth, samples):
    """Exact empirical-CDF CRPS, samples (N,S,H), truth (N,H)."""
    samples, truth = np.asarray(samples, float), np.asarray(truth, float)
    if (samples.ndim != 3 or min(samples.shape) < 1 or
            truth.shape != samples.shape[::2] or samples.shape[1] < 2
            or not np.isfinite(samples).all() or not np.isfinite(truth).all()):
        raise ValueError("finite samples (N,S,H), S>=2, truth (N,H) required")
    s = samples.shape[1]
    coeff = (2 * np.arange(1, s + 1) - s - 1)[None, :, None]
    spread = (np.sort(samples, axis=1) * coeff).sum(1) / s**2
    return float((np.abs(samples - truth[:, None]).mean(1) - spread).mean())


def forecast_metrics(truth, samples, point=None):
    truth, samples = np.asarray(truth, float), np.asarray(samples, float)
    crps = ensemble_crps(truth, samples)
    point_definition = "explicit_point" if point is not None else "scenario_mean"
    point = samples.mean(1) if point is None else np.asarray(point, float)
    if point.shape != truth.shape or not np.isfinite(point).all():
        raise ValueError("finite point predictions must match truth")
    lo, hi = np.quantile(samples, [.05, .95], axis=1)
    covered = (truth >= lo) & (truth <= hi)
    norm = np.linalg.norm(samples - truth[:, None], axis=-1).mean()
    # Accumulate paths in bounded chunks, avoiding an N*S*S*H allocation.
    spread = 0.
    for paths in samples:
        for start in range(0, len(paths), 32):
            spread += np.linalg.norm(paths[start:start + 32, None] - paths[None], axis=-1).sum()
    spread /= samples.shape[0] * samples.shape[1]**2
    return {"point_definition": point_definition, "mae": float(np.abs(point - truth).mean()),
            "rmse": float(np.sqrt(np.square(point - truth).mean())), "crps": crps,
            "scenario_mean_mae": float(np.abs(samples.mean(1) - truth).mean()),
            "scenario_median_mae": float(np.abs(np.median(samples, axis=1) - truth).mean()),
            "energy_score": float((norm - spread / 2) / truth.shape[1]**.5),
            "pointwise_90_coverage": float(covered.mean()),
            "path_coverage_of_pointwise_90_band": float(covered.all(-1).mean()),
            "mean_90_width": float((hi - lo).mean())}


def source_validation_metrics(truth, forecast, regions):
    """Pooled and equal-region scores; never use outer-target labels here.

    Temporal CRPS checkpointing uses the pooled score. Source-group selection
    uses region-macro scores. Keeping both explicit preserves those protocols.
    """
    truth, samples = np.asarray(truth), forecast["cif"]
    point = forecast.get("point_cif", samples.mean(1))
    regions = np.asarray(regions)
    if regions.shape != (len(truth),):
        raise ValueError("one source region per validation window required")

    def scores(y, draws, estimate):
        error = y - estimate
        return {"crps": ensemble_crps(y, draws),
                "mae": float(np.abs(error).mean()),
                "rmse": float(np.square(error).mean() ** .5)}

    pooled = scores(truth, samples, point)
    by_region = [scores(truth[regions == name], samples[regions == name], point[regions == name])
                 for name in sorted(set(regions))]
    return pooled, {key: float(np.mean([row[key] for row in by_region])) for key in pooled}
