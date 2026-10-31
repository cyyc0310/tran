"""Source-fitted low-dimensional local generation composition reference.

This is a positive generation model, not a capacity-constrained dispatch model.
Historical monthly composition conditions a static monthly state; daily source
residuals are propagated only to the forecast. No load or weather is required.
"""

from dataclasses import dataclass
import calendar

import numpy as np
import pandas as pd


FUELS = ("coal", "gas", "oil", "nuclear", "hydro", "wind", "solar", "other")
REGION_TIMEZONES = {
    "BPAT": "America/Los_Angeles", "CISO": "America/Los_Angeles",
    "ERCO": "America/Chicago", "FPL": "America/New_York",
    "ISNE": "America/New_York", "MISO": "America/Chicago",
    "NYIS": "America/New_York",
}


def _utc_times(times):
    values = pd.DatetimeIndex(pd.to_datetime(times, utc=True, errors="raise"))
    if values.isna().any() or not (values == values.floor("h")).all():
        raise ValueError("finite hour-aligned UTC timestamps required")
    return values


def local_hours(times, region):
    """Convert UTC timestamps to local clock hours, respecting DST."""
    if region not in REGION_TIMEZONES:
        raise ValueError(f"no reviewed IANA timezone for {region}")
    return _utc_times(times).tz_convert(REGION_TIMEZONES[region]).hour.to_numpy()


def calendar_basis(hours):
    hours = np.asarray(hours, dtype=float)
    if hours.ndim != 1 or not np.isfinite(hours).all() or ((hours < 0) | (hours >= 24)).any():
        raise ValueError("one-dimensional finite local hours in [0, 24) required")
    angle = 2 * np.pi * hours / 24
    return np.stack([np.ones_like(angle), np.sin(angle), np.cos(angle),
                     np.sin(2 * angle), np.cos(2 * angle)], axis=-1)


def _pca(values, rank):
    values = np.asarray(values, dtype=float)
    center = values.mean(axis=0)
    residual = values - center
    _, singular, right = np.linalg.svd(residual, full_matrices=False)
    keep = min(rank, len(singular))
    loadings = np.zeros((rank, values.shape[1]))
    if len(values) > 1:
        loadings[:keep] = right[:keep] * (singular[:keep] / np.sqrt(len(values) - 1))[:, None]
    # Deterministic signs make serialized/source-fitted coefficients reproducible.
    for row in loadings:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    variance = singular**2
    fraction = float(variance[:keep].sum() / variance.sum()) if variance.sum() > 0 else 0.0
    return center.reshape(5, len(FUELS)), loadings.reshape(rank, 5, len(FUELS)), fraction


@dataclass(frozen=True)
class GenerationModel:
    center: np.ndarray
    slow_loadings: np.ndarray
    daily_center: np.ndarray
    daily_loadings: np.ndarray
    metadata: dict

    @property
    def dimension(self):
        return self.slow_loadings.shape[0] + self.daily_loadings.shape[0]

    def coefficients(self, z, include_daily=True):
        z = np.asarray(z, dtype=float)
        if z.ndim != 2 or z.shape[1] != self.dimension or not np.isfinite(z).all():
            raise ValueError(f"finite latent array (N, {self.dimension}) required")
        slow_rank = len(self.slow_loadings)
        coefficients = self.center[None] + np.einsum("nr,rhf->nhf", z[:, :slow_rank], self.slow_loadings)
        if include_daily:
            coefficients = coefficients + self.daily_center[None] + np.einsum(
                "nr,rhf->nhf", z[:, slow_rank:], self.daily_loadings)
        return coefficients

    def generation(self, z, hours, include_daily=True):
        """Positive relative local generation, in a source-normalized MW scale."""
        log_generation = np.einsum("hk,nkf->nhf", calendar_basis(hours),
                                   self.coefficients(z, include_daily))
        return np.exp(np.clip(log_generation, -20, 10))

    def aggregate_mix(self, z):
        """Energy-weighted canonical-day fuel mix, excluding daily log residuals.

        This observation model does not integrate a lognormal daily residual;
        its tolerance accounts approximately for the omitted daily variability.
        """
        totals = self.generation(z, np.arange(24), include_daily=False).sum(axis=1)
        return totals / totals.sum(axis=-1, keepdims=True)

    def cif(self, z, hours, ef):
        ef = np.asarray(ef, dtype=float)
        if ef.shape != (len(FUELS),) or not np.isfinite(ef).all() or (ef < 0).any():
            raise ValueError("eight finite nonnegative fuel emission factors required")
        generation = self.generation(z, hours)
        return (generation * ef).sum(axis=-1) / generation.sum(axis=-1)


def fit(source_frames: dict[str, pd.DataFrame], max_year=2023):
    """Fit monthly and daily coefficient priors using supplied source regions only.

    A month requires at least 80% of its actual local-calendar hours; a day
    requires 80% of its 23/24/25 hours. Source rows after the UTC ``max_year``
    cutoff are excluded, even when they belong to the previous local year.
    """
    if not source_frames or not isinstance(max_year, int):
        raise ValueError("nonempty source frames and an integer max_year required")
    columns = [f"generation_{fuel}" for fuel in FUELS]
    monthly, daily, month_records, used_times = [], [], [], []
    contributing = set()
    for region in sorted(source_frames):
        if region not in REGION_TIMEZONES:
            raise ValueError(f"no reviewed IANA timezone for {region}")
        frame = source_frames[region]
        if not {"time", "valid", *columns}.issubset(frame.columns):
            raise ValueError(f"{region}: time, valid and eight generation columns required")
        times = _utc_times(frame["time"])
        if times.has_duplicates:
            raise ValueError(f"{region}: duplicate source hours")
        if frame["valid"].isna().any() or not frame["valid"].isin([True, False]).all():
            raise ValueError(f"{region}: valid must contain booleans")
        generation = frame[columns].to_numpy(dtype=float)
        finite = np.isfinite(generation).all(axis=1)
        nonnegative = (generation >= 0).all(axis=1)
        usable = frame["valid"].to_numpy(dtype=bool) & finite & nonnegative & (generation.sum(axis=1) > 0)
        local = times.tz_convert(REGION_TIMEZONES[region])
        usable &= times.year <= max_year
        month_keys = local.year.to_numpy() * 100 + local.month.to_numpy()
        for key in sorted(set(month_keys[usable])):
            year, month = divmod(int(key), 100)
            start = pd.Timestamp(year=year, month=month, day=1, tz=REGION_TIMEZONES[region])
            end = start + pd.offsets.MonthBegin(1)
            expected_hours = int((end.tz_convert("UTC") - start.tz_convert("UTC")) / pd.Timedelta(hours=1))
            rows = np.flatnonzero(usable & (month_keys == key))
            if len(rows) < .8 * expected_hours:
                continue
            values = generation[rows]
            scale = values.sum(axis=1).mean()
            log_values = np.log((values + 1e-4 * scale / len(FUELS)) / scale)
            basis = calendar_basis(local.hour.to_numpy()[rows])
            coefficients = np.linalg.lstsq(basis, log_values, rcond=None)[0]
            monthly.append(coefficients.ravel())
            contributing.add(region)
            used_times.extend([times[rows].min(), times[rows].max()])
            month_records.append({"region": region, "month": f"{year:04d}-{month:02d}",
                                  "valid_hours": len(rows), "expected_hours": expected_hours})
            local_days = local.day.to_numpy()[rows]
            for day in range(1, calendar.monthrange(year, month)[1] + 1):
                chosen = local_days == day
                day_start = pd.Timestamp(year=year, month=month, day=day, tz=REGION_TIMEZONES[region])
                day_end = day_start + pd.DateOffset(days=1)
                expected_day_hours = int((day_end.tz_convert("UTC") - day_start.tz_convert("UTC")) / pd.Timedelta(hours=1))
                if chosen.sum() < .8 * expected_day_hours:
                    continue
                day_coefficients = np.linalg.lstsq(basis[chosen], log_values[chosen], rcond=None)[0]
                daily.append((day_coefficients - coefficients).ravel())
    if len(monthly) < 2 or len(daily) < 2:
        raise ValueError("at least two sufficiently complete source months and days required")
    center, slow_loadings, slow_variance = _pca(monthly, 4)
    daily_center, daily_loadings, daily_variance = _pca(daily, 2)
    metadata = {
        "source_ids": sorted(contributing), "provided_source_ids": sorted(source_frames),
        "fit_range": [str(min(used_times)), str(max(used_times))], "max_year": max_year,
        "fit_cutoff_exclusive": f"{max_year + 1:04d}-01-01T00:00:00Z",
        "retained_pca_variance": {"monthly": slow_variance, "daily": daily_variance},
        "number_months": len(monthly), "number_days": len(daily), "month_records": month_records,
        "fuels": list(FUELS), "dimension": 6,
        "assumptions": [
            "Static monthly-state persistence; observation age is not modeled.",
            "Monthly observations use a canonical 24-hour local day and exclude daily log residuals.",
            "Independent standard-normal monthly and daily PCA coordinates; source Gaussian prior.",
            "Each source region-month receives equal PCA weight; daily residuals pooled by day.",
            "Positive normalized local generation, not load, capacity-constrained dispatch, or causal mechanisms.",
            "Log-generation clipped to [-20, 10]; source zero fuels receive a 1e-4/8 total-generation floor.",
        ],
    }
    return GenerationModel(center, slow_loadings, daily_center, daily_loadings, metadata)


def make_evaluator(model, observed_mix, future_hours, ef, observation_sigma=.06):
    """Bind permitted historical composition and calendar to a batched evaluator."""
    observed_mix = np.asarray(observed_mix, dtype=float).copy()
    future_hours = np.asarray(future_hours, dtype=float).copy()
    ef = np.asarray(ef, dtype=float).copy()
    if (observed_mix.shape != (len(FUELS),) or not np.isfinite(observed_mix).all()
            or (observed_mix < 0).any() or not np.isclose(observed_mix.sum(), 1, atol=1e-6)):
        raise ValueError("eight nonnegative observed shares summing to one required")
    if not np.isfinite(observation_sigma) or observation_sigma <= 0:
        raise ValueError("finite positive absolute fuel-share tolerance required")
    calendar_basis(future_hours)
    if ef.shape != (len(FUELS),) or not np.isfinite(ef).all() or (ef < 0).any():
        raise ValueError("eight finite nonnegative fuel emission factors required")

    def evaluate(z):
        predicted = model.aggregate_mix(z)
        residual = (predicted - observed_mix) / observation_sigma
        # Fixed Gaussian tolerance in the redundant eight-share representation.
        log_likelihood = -.5 * np.square(residual).sum(axis=-1)
        return log_likelihood, model.cif(z, future_hours, ef)

    return evaluate
