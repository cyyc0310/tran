"""Delayed daily-composition laws and conditional within-day shape analogs.

Development mechanisms combine standard prequential bootstrap and analog
conditioning. They carry no novelty or cross-region coverage guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

FUELS = ('coal', 'gas', 'oil', 'nuclear', 'hydro', 'wind', 'solar', 'other')
EF = np.array([1000., 400., 800., 0., 0., 0., 0., 400.])
FORECASTERS = ('latest', 'weekday', 'ewma', 'seasonal_kalman')
ARMS = ('additive', 'monthly_q', 'daily_native_ids', 'daily_analog',
        'forecast_analog', 'particle_analog', 'innovation_carbon',
        'innovation_full', 'innovation_permuted')
FloatArray = NDArray[np.float64]


def simplex(values: ArrayLike) -> FloatArray:
    """Euclidean simplex projection with all eight fuel categories allowed."""
    x = np.asarray(values, float)
    if x.shape[-1:] != (8,) or not x.size or not np.isfinite(x).all():
        raise ValueError('Finite eight-fuel vectors required')
    shape = x.shape
    flat = x.reshape(-1, 8)
    ordered = np.sort(flat, axis=1)[:, ::-1]
    candidates = (np.cumsum(ordered, axis=1)-1)/np.arange(1, 9)
    rho = np.sum(ordered > candidates, axis=1)-1
    threshold = candidates[np.arange(len(flat)), rho]
    out = np.maximum(flat-threshold[:, None], 0)
    return (out/out.sum(1, keepdims=True)).reshape(shape)


def carbon_coordinates(value: ArrayLike) -> FloatArray:
    """EF-active contrasts, not a sufficient statistic for hourly ratio paths."""
    x = np.asarray(value, float)
    if x.shape[-1:] != (8,) or not np.isfinite(x).all():
        raise ValueError('Finite eight-fuel vectors required')
    return np.stack([.4*x[..., [0, 1, 2, 7]].sum(-1), .6*x[..., 0], .4*x[..., 2]], -1)


def normalized_shapes(q: ArrayLike) -> FloatArray:
    """Set each fuel's daily mean to one; zero-amplitude fuels fall back flat."""
    x = np.asarray(q, float)
    if x.ndim != 3 or x.shape[1:] != (24, 8) or not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError('Nonnegative donor×24×8 profiles required')
    amplitude = x.mean(1, keepdims=True)
    return np.divide(x, amplitude, out=np.ones_like(x), where=amplitude > 0)


def ratio(mix: ArrayLike, q: ArrayLike) -> FloatArray:
    """Recompose matched particle shares and profiles with an explicit fallback."""
    m, profiles = np.asarray(mix, float), np.asarray(q, float)
    if (m.ndim != 2 or m.shape[1] != 8 or profiles.shape != (len(m), 24, 8)
            or not np.isfinite(m).all() or not np.isfinite(profiles).all()
            or np.any(m < 0) or np.any(profiles < 0) or not np.allclose(m.sum(1), 1)):
        raise ValueError('Matching finite simplex particles and nonnegative profiles required')
    a = m[:, None, :]*profiles
    total = a.sum(-1)
    # Match the frozen ratio operator's normalized-fuel computation.
    proportions = np.divide(a, total[..., None], out=np.zeros_like(a), where=total[..., None] > 0)
    return np.clip(np.where(total > 0, proportions@EF, (m@EF)[:, None]), 0, 1000)


def harmonic_design(days: pd.DatetimeIndex) -> FloatArray:
    """Fixed annual two-harmonic local seasonal reference."""
    phase = 2*np.pi*(days.dayofyear.to_numpy()-1)/365.25
    return np.column_stack([np.ones(len(days)), np.sin(phase), np.cos(phase),
                            np.sin(2*phase), np.cos(2*phase)])


@dataclass
class DailyHistory:
    """One region's coarse response table, with release filtering on every call."""
    table: pd.DataFrame
    lag: int
    _cache: dict[tuple[int, str], tuple[FloatArray, int]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.lag not in (7, 30):
            raise ValueError('Only frozen 7/30-day information delays allowed')
        t = self.table.copy().sort_values('day').reset_index(drop=True)
        if t.empty or t.region.nunique() != 1 or t.day.duplicated().any():
            raise ValueError('A nonempty single-region unique-day table is required')
        self.days = pd.DatetimeIndex(pd.to_datetime(t.day, utc=True, errors='raise'))
        if self.days.hasnans or not self.days.equals(self.days.normalize()):
            raise ValueError('UTC daily start required')
        energy = t[[f'energy_{f}' for f in FUELS]].to_numpy(float)
        if not np.isfinite(energy).all() or np.any(energy < 0) or np.any(energy.sum(1) <= 0):
            raise ValueError('Finite nonnegative positive-total coarse energy required')
        self.mix = energy/energy.sum(1, keepdims=True)
        self.released = self.days+pd.Timedelta(days=1+self.lag)
        self.table = t

    def forecast(self, origin: pd.Timestamp, method: str) -> tuple[FloatArray, int]:
        """Use only labels whose entire period and assumed release precede origin."""
        at = pd.Timestamp(origin)
        if at.tzinfo is None or at != at.normalize() or method not in FORECASTERS:
            raise ValueError('Aware midnight origin and registered forecaster required')
        at = at.tz_convert('UTC')
        key = (at.value, method)
        if key in self._cache:
            value, n = self._cache[key]
            return value.copy(), n
        take = np.flatnonzero((self.released <= at) & (self.days >= at-pd.Timedelta(days=730)))
        if not len(take):
            answer = (np.full(8, 1/8), 0)
        else:
            dates, values = self.days[take], self.mix[take]
            age = (at-dates).total_seconds().to_numpy()/86400
            recent = age <= age.min()+55
            if method == 'latest':
                estimate = values[-1]
            elif method == 'weekday':
                same = recent & (dates.dayofweek == at.dayofweek)
                use = same if same.any() else recent
                weight = np.exp(-(age[use]-age[use].min())/28)
                estimate = np.average(values[use], axis=0, weights=weight)
            elif method == 'ewma' or len(take) < 30:
                use = age <= age.min()+27
                estimate = np.average(values[use], axis=0, weights=np.exp(-(age[use]-age[use].min())/14))
            else:
                x = harmonic_design(dates)
                penalty = np.diag([0., 4., 4., 8., 8.])
                coefficient = np.linalg.solve(x.T@x+penalty, x.T@values)
                residual = values-x@coefficient
                level, variance, last = np.zeros(8), .01, dates[0]
                for date, error in zip(dates, residual):
                    variance += .0001*max((date-last).days, 0)
                    gain = variance/(variance+.0025)
                    level += gain*(error-level)
                    variance *= 1-gain
                    last = date
                estimate = harmonic_design(pd.DatetimeIndex([at]))[0]@coefficient+level
            answer = (simplex(estimate), len(take))
        self._cache[key] = (answer[0].copy(), answer[1])
        return answer[0].copy(), answer[1]

    def errors(self, origin: pd.Timestamp, method: str) -> tuple[FloatArray, pd.DatetimeIndex]:
        """Prequential errors with both pseudo-origin and response-release gates."""
        at = pd.Timestamp(origin)
        eligible = ((self.released <= at) & (self.days >= pd.Timestamp('2024-07-01', tz='UTC')))
        indices = np.flatnonzero(eligible)[-168:]
        if not len(indices):
            return np.empty((0, 8)), self.days[indices]
        prediction = np.stack([self.forecast(self.days[i], method)[0] for i in indices])
        return self.mix[indices]-prediction, self.days[indices]


def sample_indices(weights: ArrayLike, uniforms: ArrayLike) -> NDArray[np.int64]:
    """Stable inverse-CDF categorical sampling with no change of particle count."""
    w, u = np.asarray(weights, float), np.asarray(uniforms, float)
    if w.ndim != 1 or not len(w) or np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError('Nonnegative categorical weights required')
    if np.any(u < 0) or np.any(u >= 1) or not np.isfinite(u).all():
        raise ValueError('Uniforms must lie in [0,1)')
    cdf = np.cumsum(w/w.sum())
    cdf[np.flatnonzero(w > 0)[-1]:] = 1
    return np.searchsorted(cdf, u, side='right')


def daily_law(history: DailyHistory, origin: pd.Timestamp, method: str,
              uniforms: ArrayLike) -> dict[str, Any]:
    """Sample the local released daily error law, not old monthly uncertainty."""
    point, support = history.forecast(origin, method)
    errors, dates = history.errors(origin, method)
    u = np.asarray(uniforms, float)
    if not len(errors):
        raise ValueError('At least one released prequential daily error required')
    age = (origin-dates).total_seconds().to_numpy()/86400
    weight = np.exp(-(age-age.min())/56)
    ids = sample_indices(weight, u)
    drawn = errors[ids]
    draws = simplex(point+drawn)
    return dict(point=point, draws=draws, innovations=draws-point, raw_innovations=drawn,
                local_applied_errors=simplex(point+errors)-point,
                error_indices=ids, error_days=dates.as_unit('ns').asi8,
                error_release=(dates+pd.Timedelta(days=1+history.lag)).as_unit('ns').asi8,
                errors=errors, point_history_count=support)


def analog_weights(mix: ArrayLike, donor_mix: ArrayLike, donor_months: ArrayLike,
                   donor_regions: ArrayLike, origin: pd.Timestamp) -> FloatArray:
    """Ordinary equal-region, composition and calendar conditioned analogs."""
    m = np.asarray(mix, float)
    if m.shape != (8,):
        raise ValueError('One eight-fuel point required')
    return particle_analog_weights(m[None],donor_mix,donor_months,donor_regions,origin)[0]


def particle_analog_weights(mix: ArrayLike, donor_mix: ArrayLike, donor_months: ArrayLike,
                            donor_regions: ArrayLike, origin: pd.Timestamp) -> FloatArray:
    """Ordinary realized-particle composition analog, without innovation inputs."""
    m, dm = np.asarray(mix, float), np.asarray(donor_mix, float)
    labels, months = np.asarray(donor_regions), np.asarray(donor_months)
    if (m.ndim != 2 or m.shape[1] != 8 or not len(m) or dm.ndim != 2 or dm.shape[1] != 8 or not len(dm)
            or labels.shape != (len(dm),) or months.shape != (len(dm),)
            or not np.isfinite(dm).all() or not np.isfinite(m).all() or np.any(dm < 0) or np.any(m < 0)
            or not np.allclose(m.sum(1), 1) or not np.allclose(dm.sum(1), 1)):
        raise ValueError('Finite simplex and aligned donor metadata required')
    _, inverse, counts = np.unique(labels,return_inverse=True,return_counts=True)
    prior = 1/counts[inverse]
    diff = abs(origin.month-months)
    distance = (np.minimum(diff, 12-diff)/6)**2
    hellinger = .5*((np.sqrt(dm)[None]-np.sqrt(m)[:,None])**2).sum(-1)
    log_weight = -distance[None]/.5**2-hellinger/.3**2
    weights = prior[None]*np.exp(log_weight-log_weight.max(1,keepdims=True))
    return weights/weights.sum(1,keepdims=True)


def conditional_ids(base: ArrayLike, donor_errors: ArrayLike, query_errors: ArrayLike,
                    local_errors: ArrayLike, uniforms: ArrayLike,
                    *, carbon: bool = True) -> tuple[NDArray[np.int64], FloatArray]:
    """Refine analog weights by standardized innovation, with 25% base mixture.

    This changes donor marginals; it is not a fixed-marginal copula ablation.
    All scales use proper source errors or already released local errors.
    """
    transform = carbon_coordinates if carbon else lambda x: np.asarray(x, float)
    source, query, local = map(transform, (donor_errors, query_errors, local_errors))
    weights = np.asarray(base, float)
    if (source.ndim != 2 or query.ndim != 2 or local.ndim != 2
            or not len(local) or source.shape[1] != query.shape[1] or source.shape[1] != local.shape[1]
            or weights.shape != (len(source),) or not np.isfinite(weights).all()
            or not all(np.isfinite(x).all() for x in (source, query, local))
            or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError('Aligned proper and released innovation arrays required')
    weights = weights/weights.sum()
    source_mean = np.average(source, axis=0, weights=weights)
    source_sd = np.sqrt(np.maximum(np.average((source-source_mean)**2, axis=0, weights=weights), .005**2))
    local_mean, local_sd = local.mean(0), np.maximum(local.std(0), .005)
    source_z, query_z = (source-source_mean)/source_sd, (query-local_mean)/local_sd
    distance = ((query_z[:, None, :]-source_z[None, :, :])**2).sum(-1)/source.shape[1]
    log_likelihood = -.5*distance
    likelihood = weights[None]*np.exp(log_likelihood-log_likelihood.max(1, keepdims=True))
    likelihood /= likelihood.sum(1, keepdims=True)
    posterior = .25*weights[None]+.75*likelihood
    u = np.asarray(uniforms, float)
    if u.shape != (len(query),):
        raise ValueError('One uniform per query particle required')
    ids = np.array([sample_indices(w, v) for w, v in zip(posterior, u)], dtype=np.int64)
    return ids, 1/(posterior**2).sum(1)
