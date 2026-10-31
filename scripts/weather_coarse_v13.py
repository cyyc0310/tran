"""Source-trained weather gates and coarse/fine risk-factor transfer experiments.

All mechanisms here are experimental combinations of standard methods. No
coverage or originality guarantee is implied. Existing V9--V12 code is reused
without mutation. Only declared source training labels enter this module.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

from scripts import coarse_dynamics_v9 as coarse
from scripts import run_weather_increment_v12 as v12
from scripts.weather_increment_v12 import JointQuantileForest, delta_transport
from scripts.weather_shapes_v11 import ScaleModel, residual_paths
from transcif.research.coarse_forecast_v7 import COLUMNS
from transcif.research.dynamic_v2 import EF, daily_profiles
from transcif.research.fuel_transport_v7 import recompose_paths

SEED, SAMPLES = 17023, 128
GATES = np.array([0., .5, 1.])
CUTOFF = pd.Timestamp("2025-01-01", tz="UTC")
RISK_MODELS = ("global_total", "hetero_total", "location_scale", "qrf_0", "qrf_50",
               "level_independent", "level_rank", "physical_independent", "physical_rank")
COARSE_FEATURE_NAMES = tuple(
    [f"coarse_error_draw_mean_{i}" for i in range(8)]
    + [f"coarse_error_draw_sd_{i}" for i in range(8)]
    + ["coarse_ef_q05", "coarse_ef_q50", "coarse_ef_q95", "coarse_ef_mean",
       "coarse_ef_sd", "released_months", "prequential_months", "horizon_months", "release_age_days"]
)


def region_weights(regions: np.ndarray) -> np.ndarray:
    """Equal total mass per represented source region."""
    labels = np.asarray(regions)
    if labels.ndim != 1 or not len(labels):
        raise ValueError("Nonempty region vector required")
    counts = Counter(labels)
    return np.array([1 / (len(counts) * counts[r]) for r in labels])


def legal_coarse_law(monthly: pd.DataFrame, region: str, days: pd.DatetimeIndex) -> dict[str, Any]:
    """Exact V9 harmonic-only prequential law, omitting unrelated experts.

    Each residual label is a previously completed and released month. Its own
    historical forecast is fit at that month's pseudo origin from only the
    then-released prefix. The law is identical to V9 harmonic_ridge.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(days, utc=True))
    if dates.hasnans or dates.empty:
        raise ValueError("Finite nonempty query dates required")
    part = monthly.loc[monthly.region.eq(region)].copy()
    for column in ("period_start", "period_end", "available_at"):
        part[column] = pd.to_datetime(part[column], utc=True)
    if len(part):
        if part[["period_start", "period_end", "available_at"]].isna().any().any():
            raise ValueError("Monthly timestamps must be finite")
        starts = part.period_start
        if (not starts.eq(starts.dt.normalize()).all() or not starts.dt.day.eq(1).all()
                or not part.period_end.eq(starts+pd.offsets.MonthBegin(1)).all()):
            raise ValueError("Monthly intervals must cover one UTC calendar month")
    part = part.loc[(part.period_end <= dates.max()) & (part.available_at <= dates.max())]
    part = part.sort_values(["period_end", "available_at"]).reset_index(drop=True)
    if part.period_start.duplicated().any():
        raise ValueError("One frozen vintage per region-month required")
    values = part[list(COLUMNS)].to_numpy(float)
    if (not np.isfinite(values).all() or np.any(values < 0)
            or not np.allclose(values.sum(1), 1)):
        raise ValueError("Invalid monthly fuel simplex")
    points, particles, meta = [], [], []
    point_cache: dict[tuple, np.ndarray] = {}
    for origin in dates:
        prefix = part.loc[(part.period_end <= origin) & (part.available_at <= origin)]

        def point_at(history: pd.DataFrame, at: pd.Timestamp) -> np.ndarray:
            key = (tuple(history.index), at.year, at.month)
            if key not in point_cache:
                point_cache[key] = coarse._ridge(history, at)
            return point_cache[key]

        history_errors = []
        for _, label in prefix.tail(60).iterrows():
            pseudo = coarse._pseudo_origin(label, origin.day)
            history = prefix.loc[(prefix.period_end <= pseudo) & (prefix.available_at <= pseudo)]
            history_errors.append(dict(
                period_end=label.period_end, available_at=label.available_at,
                horizon=coarse._horizon(history, pseudo),
                truth=label[list(COLUMNS)].to_numpy(float),
                harmonic_ridge=point_at(history, pseudo),
            ))
        history_errors = history_errors[-24:]
        point = point_at(prefix, origin)
        horizon = coarse._horizon(prefix, origin)
        draw = coarse._draws(point, history_errors, "harmonic_ridge", horizon,
                             SAMPLES, SEED, origin, coarse._support(prefix))
        age = float((origin-prefix.available_at.max()) / pd.Timedelta(days=1)) if len(prefix) else 999.
        points.append(point); particles.append(draw)
        meta.append([len(prefix), len(history_errors), horizon, age])
    return dict(mix=np.stack(points), draws=np.stack(particles), metadata=np.asarray(meta, float),
                audit=dict(method="exact V9 harmonic prequential law", region=region,
                           query_days=len(dates), current_month_truth_used=False,
                           released_label_and_historical_prefix_only=True))


def coarse_features(mix: np.ndarray, draws: np.ndarray, metadata: np.ndarray) -> np.ndarray:
    """Shared features of the legal projected coarse-innovation distribution."""
    mix, draws, metadata = np.asarray(mix), np.asarray(draws), np.asarray(metadata)
    if (mix.ndim != 2 or mix.shape[1] != 8 or draws.shape != (len(mix), SAMPLES, 8)
            or metadata.shape != (len(mix), 4) or not np.isfinite(mix).all()
            or not np.isfinite(draws).all() or not np.isfinite(metadata).all()):
        raise ValueError("Invalid coarse-law shapes or values")
    error = draws-mix[:, None]
    scalar = error @ EF
    return np.column_stack((error.mean(1), error.std(1),
        np.quantile(scalar, [.05, .5, .95], axis=1).T, scalar.mean(1), scalar.std(1), metadata))


def blend_profiles(native: np.ndarray, weather: np.ndarray, gates: np.ndarray) -> np.ndarray:
    """Convex q blending with bit-exact zero/one endpoints."""
    a, b, g = np.asarray(native), np.asarray(weather), np.asarray(gates, float)
    if (a.shape != b.shape or a.ndim < 3 or a.shape[-2:] != (24, 8)
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or np.any(a < 0) or np.any(b < 0)
            or g.shape != (len(a),) or not np.isfinite(g).all()
            or np.any((g < 0) | (g > 1))):
        raise ValueError("Matching q profiles and per-day gates in [0,1] required")
    expanded = g.reshape((len(g),) + (1,)*(a.ndim-1))
    return np.where(expanded == 0, a, np.where(expanded == 1, b, a+expanded*(b-a)))


def profile_audit(native: np.ndarray, updated: np.ndarray, support: np.ndarray) -> dict[str, Any]:
    """Measure sampled fuel-amplitude invariance and newly positive zero cells."""
    active = np.asarray(support, bool) & (native.mean(-2) > 0)
    zero = (native == 0) & active[..., None, :]
    created = zero & (updated > 1e-12)
    axes = tuple(range(native.ndim-1))
    zero_counts, created_counts = zero.sum(axes), created.sum(axes)
    return dict(amplitude_error=float(np.abs(updated.mean(-2)-native.mean(-2)).max()),
        sampled_zero_creation=dict(supported_original_zero_count_by_fuel=zero_counts.tolist(),
            created_positive_from_zero_count_by_fuel=created_counts.tolist(),
            created_positive_from_zero_fraction_by_fuel=np.divide(created_counts, zero_counts,
                out=np.zeros(8), where=zero_counts > 0).tolist(),
            created_positive_from_zero_q_amount_by_fuel=np.where(created, updated, 0.).sum(axes).tolist()))


class Regression:
    """Fixed source-region-weighted regression with fit-only missing handling."""
    def __init__(self, kind: str = "forest") -> None:
        if kind not in ("ridge", "forest"):
            raise ValueError("Unknown regression family")
        self.kind = kind

    def fit(self, x: np.ndarray, y: np.ndarray, regions: np.ndarray) -> Regression:
        """Fit only the rows explicitly supplied by the enclosing source fold."""
        x, y, regions = np.asarray(x, float), np.asarray(y, float), np.asarray(regions)
        if (x.ndim != 2 or not len(x) or len(y) != len(x) or len(regions) != len(x)
                or np.isinf(x).any() or not np.isfinite(y).all()
                or not set(regions) <= set(v12.SOURCES) or len(set(regions)) < 2):
            raise ValueError("Finite source-only regression rows required")
        self.medians_ = np.array([np.median(c[~np.isnan(c)]) if (~np.isnan(c)).any() else 0.
                                  for c in x.T])
        raw = self._impute(x)
        weights = region_weights(regions)
        self.mean_ = np.average(raw, axis=0, weights=weights)
        self.sd_ = np.sqrt(np.average((raw-self.mean_)**2, axis=0, weights=weights))
        self.sd_[self.sd_ < 1e-8] = 1.
        self.model_ = (Ridge(alpha=10.) if self.kind == "ridge" else ExtraTreesRegressor(
            n_estimators=128, min_samples_leaf=20, max_features=.7, random_state=SEED, n_jobs=1))
        self.model_.fit(self._design(x), y, sample_weight=weights*len(x))
        self.audit_ = dict(training_regions=sorted(set(regions)), kind=self.kind,
                           equal_region_weights=True, fit_only_preprocessing=True)
        return self

    def _impute(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        if x.ndim != 2 or x.shape[1] != len(self.medians_) or np.isinf(x).any():
            raise ValueError("Wrong feature width or infinite features")
        missing = np.isnan(x)
        return np.column_stack((np.where(missing, self.medians_, x), missing.astype(float)))

    def _design(self, x: np.ndarray) -> np.ndarray:
        return (self._impute(x)-self.mean_)/self.sd_

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict using frozen source preprocessing."""
        return self.model_.predict(self._design(x))


class GainGate(Regression):
    """Standard conditional empirical loss minimization over three q gates."""
    def fit(self, x: np.ndarray, losses: np.ndarray, regions: np.ndarray) -> GainGate:
        """Regress per-day normalized deterministic-center squared losses."""
        losses = np.asarray(losses, float)
        if losses.shape != (len(x), 3) or np.any(losses < 0):
            raise ValueError("Three nonnegative gate losses per day required")
        super().fit(x, losses/(losses.mean(1, keepdims=True)+25.), regions)
        return self

    def gates(self, x: np.ndarray) -> np.ndarray:
        """Choose smallest gate on a predicted-loss tie; no target feedback."""
        return GATES[np.argmin(self.predict(x), axis=1)]


def couple_indices(indices: np.ndarray, coarse_draws: np.ndarray, calibration_z: np.ndarray,
                   days: pd.DatetimeIndex) -> np.ndarray:
    """Schaake/ECC-style reorder, preserving each sampled-day multiset exactly."""
    indices, coarse_draws, calibration_z = np.asarray(indices), np.asarray(coarse_draws), np.asarray(calibration_z)
    if (indices.ndim != 2 or coarse_draws.shape != (*indices.shape, 8)
            or not np.issubdtype(indices.dtype, np.integer) or indices.min() < 0
            or indices.max() >= len(calibration_z) or len(days) != len(indices)
            or not np.isfinite(coarse_draws).all() or not np.isfinite(calibration_z).all()):
        raise ValueError("Invalid coupling arrays")
    result = np.empty_like(indices)
    for row, day in enumerate(days):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 13013, day.year, day.month, day.day]))
        source_order = np.lexsort((rng.random(indices.shape[1]), calibration_z[indices[row]]))
        target_order = np.lexsort((rng.random(indices.shape[1]), coarse_draws[row] @ EF))
        result[row, target_order] = indices[row, source_order]
    return result


def cdf_indices(weights: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    """Invert day distributions without drawing any zero-mass trailing row."""
    weights, uniforms = np.asarray(weights, float), np.asarray(uniforms, float)
    if (weights.ndim != 2 or not weights.shape[1] or uniforms.ndim != 2
            or len(weights) != len(uniforms) or not np.isfinite(weights).all()
            or not np.isfinite(uniforms).all() or np.any(weights < 0)
            or not np.allclose(weights.sum(1), 1.) or np.any((uniforms < 0) | (uniforms > 1))):
        raise ValueError("Normalized probability rows and uniforms in [0,1] required")
    result = []
    for w, u in zip(weights, uniforms):
        cdf = np.cumsum(w)
        last_positive = np.flatnonzero(w > 0)[-1]
        cdf[last_positive:] = 1.
        result.append(np.minimum(np.searchsorted(cdf, u, side="right"), last_positive))
    return np.stack(result)


def factor_paths(mix: np.ndarray, draws: np.ndarray, qbar: np.ndarray, errors: np.ndarray,
                 fine: np.ndarray, z: np.ndarray, indices: np.ndarray, *, physical: bool) -> np.ndarray:
    """Replace, rather than add to, the coarse component of total error."""
    mix, draws, qbar = np.asarray(mix), np.asarray(draws), np.asarray(qbar)
    if draws.shape != (len(mix), indices.shape[1], 8) or qbar.shape != (len(mix), 24, 8):
        raise ValueError("Invalid factor-prediction dimensions")
    if physical:
        structural = np.stack([recompose_paths(d, np.broadcast_to(q, (len(d), 24, 8)))
                               for d, q in zip(draws, qbar)])
        return structural+fine[indices]
    center = np.stack([recompose_paths(m, q) for m, q in zip(mix, qbar)])
    target_z = (draws-mix[:, None]) @ EF
    return center[:, None]+errors[indices]-z[indices, None]+target_z[..., None]


class Context:
    """Cache nested source teachers, gates and same-cohort risk models."""
    def __init__(self, frozen: dict[str, Any], frames: dict, records: dict, *,
                 stage: str, round_number: int, weather_policy: str | None) -> None:
        self.frozen, self.frames, self.records = frozen, frames, records
        self.round_number, self.weather_policy = round_number, weather_policy
        self.monthly = pd.read_csv(Path(frozen["source"])/"monthly.csv")
        for key in ("period_start", "period_end", "available_at"):
            self.monthly[key] = pd.to_datetime(self.monthly[key], utc=True)
        queries = ({tuple(sorted(v12.allowed_sources(r))): [r] for r in v12.SOURCES}
                   if stage == "source" else {tuple(sorted(v12.SOURCES)): list(v12.TARGETS)})
        self.shapes = v12.ShapeCache(records, list(queries), queries)
        fit_subsets = set(queries)
        if round_number == 2:
            fit_subsets |= {tuple(s for s in subset if s != r) for subset in queries for r in subset}
        # Complete all shape requests before the first fit (including third nesting).
        for fit_subset in sorted(fit_subsets):
            for region in fit_subset:
                inner = tuple(s for s in fit_subset if s != region)
                self._register_profiles(inner, region, "proper")
        if round_number == 2:
            for subset in queries:
                for region in subset:
                    inner = tuple(s for s in subset if s != region)
                    self._register_profiles(inner, region, "calibration")
        self.banks: dict[tuple, Any] = {}
        self.profile_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        self.gates_cache: dict[tuple, GainGate] = {}
        self.laws: dict[tuple, dict] = {}
        self.risks: dict[tuple, dict] = {}
        self.gate_fit_audits: dict[str, dict] = {}
        self.shape_fit_audits = self.shapes.audit
        for region in v12.SOURCES:
            days, truth = daily_profiles(frames[region])
            ids = pd.Index(days).get_indexer(records[region, "proper"]["days"])
            if (ids < 0).any():
                raise ValueError("Proper truth/day mismatch")
            records[region, "proper"]["truth"] = truth[ids]

    def _register_profiles(self, subset: tuple[str, ...], region: str, split: str) -> None:
        if len(subset) < 2:
            raise ValueError("Insufficient nested source teachers")
        for budget in ("B", "W"):
            self.shapes.request(subset, region, split)
            for donor in subset:
                self.shapes.request(tuple(s for s in subset if s != donor), donor, "proper")

    def _bank(self, subset: tuple[str, ...]) -> Any:
        if subset not in self.banks:
            self.banks[subset] = v12._bank(self.frames, subset)
        return self.banks[subset]

    def _sampled(self, subset: tuple[str, ...], region: str, split: str,
                 diagnostics: dict[str, Any] | None = None) -> tuple:
        bank, record = self._bank(subset), self.records[region, split]
        ids, _ = v12._indices(bank, record["features"]["B"][:, :8], record["days"], np.array(subset))
        qb, qw = [self.shapes.get(subset, b, region, split) for b in ("B", "W")]
        db, dw = [v12._donor_predictions(bank, self.shapes, b) for b in ("B", "W")]
        native = bank.q_[ids]
        weather = delta_transport(native, qw[:, None], qb[:, None], dw[ids], db[ids],
                                  bank.fuel_support_[ids], strength=.5)
        if diagnostics is not None:
            amplitude, shape = v12.split_profiles(native)
            before = shape+.5*((qw[:, None]-qb[:, None])-(dw[ids]-db[ids]))
            after = np.divide(weather, amplitude[..., None, :], out=np.ones_like(weather),
                              where=amplitude[..., None, :] > 0)
            valid = np.broadcast_to((bank.fuel_support_[ids] & (amplitude > 0))[..., None, :], native.shape)
            diagnostics.update(mean_absolute_projection_change=float(np.abs(after-before)[valid].mean()) if valid.any() else 0.,
                fraction_negative_before_projection=float((before[valid] < 0).mean()) if valid.any() else 0.,
                sampled_supported_active_cells=int(valid.sum()))
        return native, weather, ids

    def _profiles(self, subset: tuple[str, ...], region: str, split: str) -> tuple[np.ndarray, np.ndarray]:
        key = subset, region, split
        if key not in self.profile_cache:
            native, weather, _ = self._sampled(subset, region, split)
            self.profile_cache[key] = native.mean(1), weather.mean(1)
        return self.profile_cache[key]

    def _gate(self, subset: tuple[str, ...], kind: str) -> GainGate:
        key = subset, kind
        if key not in self.gates_cache:
            xs, losses, labels = [], [], []
            for region in subset:
                inner = tuple(s for s in subset if s != region)
                a, b = self._profiles(inner, region, "proper")
                rec = self.records[region, "proper"]
                centers = np.stack([np.stack([recompose_paths(m, q) for m, q in
                    zip(rec["features"]["B"][:, :8], blend_profiles(a, b, np.full(len(a), g)))])
                    for g in GATES], axis=1)
                losses.append(((centers-rec["truth"][:, None])**2).mean(2))
                xs.append(rec["features"]["W"]); labels.extend([region]*len(a))
            model = GainGate(kind).fit(np.concatenate(xs), np.concatenate(losses), np.array(labels))
            self.gates_cache[key] = model
            audit = dict(**model.audit_, teacher_period="2024H1", teacher_geographic_oof=True,
                         teacher_chronological_hindcast=False, proper_only_gate_fit=True,
                         teacher_regions_excluded_from_own_shapes_and_donors=True,
                         teacher_nested_donor_oof=True, teacher_qbar_samples=SAMPLES,
                         gate_levels=GATES.tolist(), loss="normalized deterministic-center MSE")
            self.gate_fit_audits["|".join(subset)+":"+kind] = audit
            print("V13 gate", kind, ",".join(subset), flush=True)
        return self.gates_cache[key]

    def _policy_gates(self, subset: tuple[str, ...], region: str, split: str, policy: str) -> np.ndarray:
        x = self.records[region, split]["features"]["W"]
        if policy == "fixed_donor":
            return np.ones(len(x))
        if policy == "constant_half":
            return np.full(len(x), .5)
        if policy not in ("gate_ridge", "gate_forest"):
            raise ValueError("Unregistered weather policy")
        return self._gate(subset, policy.removeprefix("gate_")).gates(x)

    def _law(self, region: str, split: str) -> dict:
        key = region, split
        if key not in self.laws:
            rec = self.records[key]
            law = legal_coarse_law(self.monthly, region, rec["days"])
            if not np.allclose(law["mix"], rec["features"]["B"][:, :8], atol=1e-13, rtol=0):
                raise ValueError("Coarse point differs from lawful feature center")
            self.laws[key] = law
        return self.laws[key]

    def _response_rows(self, region: str, days: pd.DatetimeIndex) -> tuple:
        part = self.monthly.loc[self.monthly.region.eq(region)].set_index("period_start")
        months = days.normalize().map(lambda d: d.replace(day=1))
        rows = part.reindex(months)
        valid = (rows.available_at.le(CUTOFF) & rows.period_end.le(CUTOFF)
                 & rows[list(COLUMNS)].notna().all(axis=1)).to_numpy()
        return valid, rows

    def _calibration(self, subset: tuple[str, ...], budget: str, policy: str) -> dict:
        key = subset, budget, policy
        if key in self.risks:
            return self.risks[key]
        arrays: dict[str, list] = {k: [] for k in ("errors", "fine", "z", "features", "regions", "days",
            "response_available", "response_end", "true_mix", "forecast_mix", "qbar")}
        for region in subset:
            inner = tuple(s for s in subset if s != region)
            rec = self.records[region, "calibration"]
            a, b = self._profiles(inner, region, "calibration")
            gates = (np.zeros(len(a)) if budget == "B" else
                     self._policy_gates(inner, region, "calibration", policy))
            qbar = blend_profiles(a, b, gates)
            valid, rows = self._response_rows(region, rec["days"])
            law = self._law(region, "calibration")
            mix = rec["features"]["B"][:, :8]
            true_mix = rows[list(COLUMNS)].to_numpy(float)
            center = np.stack([recompose_paths(m, q) for m, q in zip(mix[valid], qbar[valid])])
            oracle = np.stack([recompose_paths(m, q) for m, q in zip(true_mix[valid], qbar[valid])])
            errors, fine = rec["truth"][valid]-center, rec["truth"][valid]-oracle
            values = dict(errors=errors, fine=fine, z=(true_mix[valid]-mix[valid]) @ EF,
                features=np.column_stack((rec["features"][budget], coarse_features(
                    mix, law["draws"], law["metadata"])))[valid],
                regions=np.repeat(region, valid.sum()), days=rec["days"][valid].as_unit("ns").asi8,
                response_available=pd.DatetimeIndex(rows.available_at[valid]).as_unit("ns").asi8,
                response_end=pd.DatetimeIndex(rows.period_end[valid]).as_unit("ns").asi8,
                true_mix=true_mix[valid], forecast_mix=mix[valid], qbar=qbar[valid])
            for name, value in values.items():
                arrays[name].append(value)
        cal = {name: np.concatenate(values) for name, values in arrays.items()}
        dates = pd.to_datetime(cal["days"], utc=True)
        cal["qrf"] = JointQuantileForest().fit(cal["features"], cal["errors"], cal["regions"], days=dates)
        scales, locations, location_scales = np.empty((len(dates), 2)), np.empty_like(cal["errors"]), np.empty((len(dates), 2))
        for region in subset:
            keep = cal["regions"] != region
            x, y, r = cal["features"][keep], cal["errors"][keep], cal["regions"][keep]
            scales[~keep] = ScaleModel().fit(x, y, r, days=dates[keep]).predict(cal["features"][~keep])
            locations[~keep] = Regression().fit(x, y, r).predict(cal["features"][~keep])
        adjusted = cal["errors"]-locations
        for region in subset:
            keep = cal["regions"] != region
            # The scale training responses must not depend indirectly on the
            # held region's H2 labels through other regions' location fits.
            nested_adjusted = np.empty_like(cal["errors"][keep])
            inner_regions = cal["regions"][keep]
            for inner_region in sorted(set(inner_regions)):
                fit = keep & (cal["regions"] != inner_region)
                model = Regression().fit(cal["features"][fit], cal["errors"][fit], cal["regions"][fit])
                rows = inner_regions == inner_region
                nested_adjusted[rows] = (cal["errors"][keep][rows]
                    - model.predict(cal["features"][keep][rows]))
            location_scales[~keep] = ScaleModel().fit(cal["features"][keep], nested_adjusted,
                cal["regions"][keep], days=dates[keep]).predict(cal["features"][~keep])
        cal.update(scales=scales, location_errors=adjusted, location_scales=location_scales,
            scale_model=ScaleModel().fit(cal["features"], cal["errors"], cal["regions"], days=dates),
            location_model=Regression().fit(cal["features"], cal["errors"], cal["regions"]),
            location_scale_model=ScaleModel().fit(cal["features"], adjusted, cal["regions"], days=dates))
        self.risks[key] = cal
        print("V13 risk", budget, policy, ",".join(subset), len(dates), flush=True)
        return cal

    def predict_region(self, subset: tuple[str, ...], region: str, mixes: np.ndarray, draws: np.ndarray, *,
                       stage: str, round_number: int, weather_policy: str | None) -> tuple:
        """Predict only declared known evaluation regions; never read their labels."""
        if region not in (v12.SOURCES if stage == "source" else v12.TARGETS) or region in subset:
            raise ValueError("Evaluation region must be known and excluded from training")
        rec, bank = self.records[region, "evaluation"], self._bank(subset)
        projection: dict[str, Any] = {}
        native, weather, donors = self._sampled(subset, region, "evaluation", diagnostics=projection)
        days, result = rec["days"], {}
        native_raw = np.stack([recompose_paths(m, q) for m, q in zip(draws, native)])
        fixed_raw = np.stack([recompose_paths(m, q) for m, q in zip(draws, weather)])
        simple = bank.raw_cif_[donors]+((draws-bank.donor_mix_[donors]) @ EF)[..., None]
        result["B", "native_recompose"] = dict(raw=native_raw, center=native_raw.mean(1),
            **profile_audit(native, native, bank.fuel_support_[donors]))
        result["B", "simple_anchor"] = dict(raw=simple, center=simple.mean(1))
        result["W", "fixed_donor"] = dict(raw=fixed_raw, center=fixed_raw.mean(1),
            **profile_audit(native, weather, bank.fuel_support_[donors]))
        policies = ("constant_half", "gate_ridge", "gate_forest") if round_number == 1 else (weather_policy,)
        selected_gates = None
        for policy in policies:
            gates = self._policy_gates(subset, region, "evaluation", policy)
            q = blend_profiles(native, weather, gates)
            raw = np.stack([recompose_paths(m, profile) for m, profile in zip(draws, q)])
            mode = policy if round_number == 1 else "selected_donor"
            result["W", mode] = dict(raw=raw, center=raw.mean(1), archive_arrays=dict(weather_gates=gates),
                **profile_audit(native, q, bank.fuel_support_[donors]))
            selected_gates = gates
        # Exact native center is an archived diagnostic, distinct from risk centers.
        old = Path(self.frozen["native_paths"][f"{stage}/{region}/native_recompose"])
        with np.load(old, allow_pickle=False) as archive:
            result["B", "native_recompose"]["center"] = archive["center"].copy()
        old = Path(self.frozen["native_paths"][f"{stage}/{region}/simple_anchor"])
        with np.load(old, allow_pickle=False) as archive:
            result["B", "simple_anchor"]["center"] = archive["center"].copy()
        audit = dict(training_regions=list(subset), donor_regions=list(bank.source_regions_),
            excluded_evaluation_region=region if region in v12.SOURCES else None,
            target_hourly_used=False, source_calibration_regions=list(subset) if round_number == 2 else [],
            weather_strength=.5, gate_levels=GATES.tolist(), gate_proper_only=True,
            gate_teacher_chronological_hindcast=False, risk_center_qbar_samples=SAMPLES,
            risk_center_full_bank_expectation=False, native_v12_exact_identity_required=True,
            joint_donor_identity_shared=True, weather_policy=weather_policy,
            coarse_summary_feature_names=list(COARSE_FEATURE_NAMES))
        audit["sampled_fixed_weather_projection"] = projection
        residuals = None
        if round_number == 2:
            uniforms = v12._uniforms(bank, days)
            # Target forecast metadata is lawful monthly-only, not target CIF.
            monthly = self.monthly if stage == "source" else pd.read_csv(Path(self.frozen["target"])/"monthly.csv")
            law = legal_coarse_law(monthly, region, days)
            if not np.array_equal(law["draws"], draws) or not np.allclose(law["mix"], mixes, atol=1e-13, rtol=0):
                raise ValueError("Legal V9 law reconstruction mismatch")
            cfeatures = coarse_features(mixes, draws, law["metadata"])
            for budget in ("B", "W"):
                cal = self._calibration(subset, budget, weather_policy)
                qbar = blend_profiles(native.mean(1), weather.mean(1),
                    np.zeros(len(days)) if budget == "B" else selected_gates)
                center = np.stack([recompose_paths(m, q) for m, q in zip(mixes, qbar)])
                x = np.column_stack((rec["features"][budget], cfeatures))
                cumulative = np.cumsum(region_weights(cal["regions"])); cumulative[-1] = 1.
                ids = np.searchsorted(cumulative, uniforms, side="right")
                residuals = ids
                coupled = couple_indices(ids, draws, cal["z"], days)
                common = dict(calibration_regions=np.asarray(cal["regions"], dtype="U"),
                    calibration_origins=cal["days"], calibration_response_available_at=cal["response_available"],
                    calibration_response_period_end=cal["response_end"], calibration_errors=cal["errors"],
                    calibration_fine_errors=cal["fine"], calibration_z=cal["z"],
                    calibration_true_mix=cal["true_mix"], calibration_forecast_mix=cal["forecast_mix"],
                    calibration_qbar=cal["qbar"], residual_uniforms=uniforms, query_qbar=qbar,
                    query_coarse_mix=mixes, query_coarse_draws=draws, query_coarse_features=cfeatures,
                    structural_center=center)
                for mode in RISK_MODELS:
                    extra: dict[str, Any] = {}
                    method_center = center
                    if mode == "global_total":
                        raw = center[:, None]+cal["errors"][ids]
                    elif mode == "hetero_total":
                        raw = residual_paths(center, cal["errors"], ids,
                            scales=cal["scale_model"].predict(x), calibration_scales=cal["scales"])
                    elif mode == "location_scale":
                        location = cal["location_model"].predict(x)
                        method_center = center+location
                        raw = residual_paths(method_center, cal["location_errors"], ids,
                            scales=cal["location_scale_model"].predict(x), calibration_scales=cal["location_scales"])
                        extra.update(location_prediction=location)
                    elif mode.startswith("qrf_"):
                        fraction = 0. if mode == "qrf_0" else .5
                        weights = cal["qrf"].predict_weights(x, global_fraction=fraction)
                        conditional = cdf_indices(weights, uniforms)
                        raw = center[:, None]+cal["errors"][conditional]
                        extra.update(conditional_weights=weights, conditional_residual_indices=conditional,
                            raw_conditional_quantiles=center[:, None]+cal["qrf"].predict_quantiles(
                                x, [.05, .5, .95], global_fraction=fraction), quantile_levels=np.array([.05, .5, .95]))
                    else:
                        chosen = coupled if mode.endswith("rank") else ids
                        raw = factor_paths(mixes, draws, qbar, cal["errors"], cal["fine"], cal["z"],
                                           chosen, physical=mode.startswith("physical"))
                        extra.update(fine_residual_indices=chosen, independent_residual_indices=ids,
                                     calibration_rank_indices=coupled)
                    result[budget, mode] = dict(raw=raw, center=method_center, archive_arrays={**common, **extra})
            audit.update(calibration_response_cutoff=CUTOFF.isoformat(),
                calibration_response_available_before_cutoff=True, risk_cohort_shared=True,
                physical_risk_replaces_total_error_coarse_component=True,
                total_risk_reference_adds_coarse_particles=False, exact_factor_decomposition=True,
                rank_coupling_changes_component_marginals=False, coupling_ties_seeded=True)
        return result, donors, residuals, audit


def prepare(frozen: dict[str, Any], frames: dict, records: dict, *, stage: str,
            round_number: int, weather_policy: str | None) -> Context:
    """Plan all nested fits before materializing any model predictions."""
    return Context(frozen, frames, records, stage=stage, round_number=round_number, weather_policy=weather_policy)
