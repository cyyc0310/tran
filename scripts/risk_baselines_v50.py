"""Ordinary source-trained risk baselines; external calibration never fits a head."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from fractions import Fraction
import math
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from scripts import fleet_information_v34 as fleet


SOURCES = ('BPAT', 'CISO', 'ERCO', 'FPL', 'ISNE', 'MISO', 'NYIS')
DONOR_SOURCES = ('SWPP', 'AVA', 'BANC', 'JEA', 'SC', 'NWMT', 'SCL', 'WACM', 'SEC', 'SCEG')
METHODS = ('GlobalABS', 'MCStd', 'Component', 'HETScale', 'HETLocationScale', 'CQRplus')
EXTRA_NAMES = ('point_hour', 'std_hour', 'point_day_mean', 'point_day_min', 'point_day_max',
               'mc_day_mean_std', 'mc_centered_shape_rms', 'hour_sin', 'hour_cos')
FEATURE_NAMES = {'B0': fleet.FEATURE_NAMES['Size']+EXTRA_NAMES,
                 'B1': fleet.FEATURE_NAMES['SizeCap']+EXTRA_NAMES}
PARAMETERS = dict(max_iter=128, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.,
                  learning_rate=.1, max_features=1., early_stopping=False, random_state=17023)
LOG_SCALE_MAX = float(np.log(2001.))


def _real(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError('Real '+name+' required')
    array = np.asarray(value, float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError('Finite matching '+name+' required')
    return array


def _metadata(regions: Sequence[str], keys: Sequence[str], outer: str, *, training: bool) -> np.ndarray:
    r = np.asarray(regions, dtype=str)
    if outer not in SOURCES or r.ndim != 1 or not len(r) or len(keys) != len(r) or len(set(keys)) != len(keys):
        raise ValueError('Registered outer and unique nonempty source/day keys required')
    if training and set(r) != set(SOURCES)-{outer}:
        raise ValueError('Exactly the original six proper sources excluding outer required')
    for source, key in zip(r, keys):
        try:
            label, day_string = key.split('/')
            day = date.fromisoformat(day_string)
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError('Canonical source/YYYY-MM-DD keys required') from exc
        if label != source or day_string != day.isoformat():
            raise ValueError('Source/key identity required')
        if training:
            legal = date(2024, 1, 25) <= day < date(2024, 7, 1)
        else:
            legal = ((source in DONOR_SOURCES and date(2024, 7, 1) <= day < date(2024, 12, 25))
                     or (source == outer and date(2025, 1, 1) <= day < date(2025, 7, 1)))
        if not legal:
            raise ValueError('Wrong proper/calibration/query source role or date')
    return r


def _data(X: np.ndarray, point: np.ndarray, std_hour: np.ndarray, std_level: np.ndarray,
          std_shape: np.ndarray, n: int, names: tuple[str, ...]) -> tuple[np.ndarray, ...]:
    x = _real(X, (n, 24, len(names)), 'features')
    p = _real(point, (n, 24), 'point')
    h = _real(std_hour, (n, 24), 'hour std')
    l, s = _real(std_level, (n,), 'level std'), _real(std_shape, (n,), 'shape std')
    if np.any(h < 0) or np.any(l < 0) or np.any(s < 0):
        raise ValueError('Nonnegative postclip scales required; zeros receive floor1')
    spread = lambda a: np.broadcast_to(a[:, None], (n, 24))
    expected = np.stack((p, h, spread(p.mean(1)), spread(p.min(1)), spread(p.max(1)), spread(l), spread(s),
        np.broadcast_to(np.sin(2*np.pi*np.arange(24)/24), p.shape),
        np.broadcast_to(np.cos(2*np.pi*np.arange(24)/24), p.shape)), axis=-1)
    if not np.allclose(x[..., -9:], expected, atol=1e-9, rtol=0):
        raise ValueError('Nine fixed risk features must match supplied postclip point/statistics and UTC hour')
    return x, p, h, l, s


def _balanced(regions: np.ndarray) -> np.ndarray:
    return np.array([len(regions)/(len(set(regions))*np.count_nonzero(regions == s)) for s in regions])


def _units(regions: np.ndarray, hours: int) -> tuple[np.ndarray, int]:
    """Integer common-denominator masses avoid floating empirical-CDF boundaries."""
    labels, counts = np.unique(regions, return_counts=True)
    denominator = math.lcm(*(int(n)*hours for n in counts))
    total = len(labels)*denominator
    lookup = {source: denominator//(int(n)*hours) for source, n in zip(labels, counts)}
    dtype = np.int64 if total <= np.iinfo(np.int64).max else object
    return np.repeat(np.array([lookup[source] for source in regions], dtype=dtype), hours), total


def component_bounds(level_standardized: np.ndarray, shape_standardized: np.ndarray,
                     regions: np.ndarray, point: np.ndarray, std_level: np.ndarray,
                     std_shape: np.ndarray, *, batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Exact source/day-weighted .05/.95 transport of jointly paired proper paths."""
    r = np.asarray(regions, str)
    if r.ndim != 1 or not len(r) or not set(r) <= set(SOURCES) or type(batch_size) is not int or batch_size < 1:
        raise ValueError('Nonempty proper source labels and positive integer batch size required')
    level = _real(level_standardized, (len(r),), 'standardized level')
    shape = _real(shape_standardized, (len(r), 24), 'standardized shape')
    if not np.allclose(shape.mean(1), 0, atol=1e-9, rtol=0):
        raise ValueError('Centered proper shape paths required')
    n = len(point)
    if not n:
        raise ValueError('Nonempty query required')
    p = _real(point, (n, 24), 'point')
    l, s = _real(std_level, (n,), 'level std'), _real(std_shape, (n,), 'shape std')
    if np.any(l < 0) or np.any(s < 0):
        raise ValueError('Nonnegative query scales required')
    unit, total = _units(r, 1)
    lo, hi = np.empty_like(p), np.empty_like(p)
    for start in range(0, n, batch_size):
        take = slice(start, min(start+batch_size, n))
        paths = (p[take, None]+np.maximum(l[take], 1)[:, None, None]*level[None, :, None]
                 + np.maximum(s[take], 1)[:, None, None]*shape[None])
        order = np.argsort(paths, axis=1, kind='stable')
        ordered = np.take_along_axis(paths, order, axis=1)
        cdf = np.cumsum(unit[order], axis=1)
        for numerator, target in ((1, lo), (19, hi)):
            threshold = (numerator*total+19)//20
            indices = np.sum(cdf < threshold, axis=1)
            target[take] = np.take_along_axis(ordered, indices[:, None], axis=1)[:, 0]
    if not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError('Finite transported component bounds required')
    return lo, hi


@dataclass
class RiskBaselines:
    outer: str
    budget: str
    feature_names: tuple[str, ...]
    regions: np.ndarray
    level_standardized: np.ndarray
    shape_standardized: np.ndarray
    models: dict[str, Any]
    audit: dict[str, Any]

    def predict(self, X: np.ndarray, point: np.ndarray, std_hour: np.ndarray, std_level: np.ndarray,
                std_shape: np.ndarray, regions: Sequence[str], keys: Sequence[str]) -> dict[str, Any]:
        r = _metadata(regions, keys, self.outer, training=False)
        if self.budget not in FEATURE_NAMES or self.feature_names != FEATURE_NAMES[self.budget]:
            raise ValueError('Frozen risk budget and feature order required')
        x, p, h, l, s = _data(X, point, std_hour, std_level, std_shape, len(r), self.feature_names)
        n = len(r); flat = x.reshape(-1, x.shape[-1])
        raw = {name: _real(fitted.predict(flat), (n*24,), name+' prediction').reshape(n, 24)
               for name, fitted in self.models.items()}
        lo = np.broadcast_to(p, (6, n, 24)).copy(); hi = lo.copy()
        scales, centers = np.ones_like(lo), lo.copy()
        scales[1] = np.maximum(h, 1)
        lo[2], hi[2] = component_bounds(self.level_standardized, self.shape_standardized, self.regions, p, l, s)
        for index, name in ((3, 'scale'), (4, 'location_scale')):
            scales[index] = np.maximum(np.expm1(np.clip(raw[name], 0, LOG_SCALE_MAX)), 1)
        bias = np.clip(raw['location'], -1000, 1000)
        lo[4] = hi[4] = centers[4] = p+bias
        crossing = raw['q05'] > raw['q95']
        lo[5], hi[5] = p+np.minimum(raw['q05'], raw['q95']), p+np.maximum(raw['q05'], raw['q95'])
        return dict(lo0=lo, hi0=hi, scales=scales, centers=centers,
            diagnostics=dict(cqr_crossing_count=int(crossing.sum()), cqr_crossing_fraction=float(crossing.mean()),
                cqr_crossing_mask=crossing, location_bias_clip_fraction=float((bias != raw['location']).mean()),
                scale_log_clip_fraction={name: float(((raw[name] < 0) | (raw[name] > LOG_SCALE_MAX)).mean())
                                         for name in ('scale', 'location_scale')}),
            audit=dict(new_fits=0, budget=self.budget, feature_names=self.feature_names,
                center_definition='HETLocationScale: point+clipped bias; other five: original point, not interval midpoint',
                initial_bounds_clipped=False, component_quantiles_exact=True, ensemble_crps_fabricated=False))


def fit_baselines(X: np.ndarray, point: np.ndarray, truth: np.ndarray, std_hour: np.ndarray,
                  std_level: np.ndarray, std_shape: np.ndarray, regions: Sequence[str], keys: Sequence[str],
                  *, outer: str, budget: str, feature_names: Sequence[str]) -> RiskBaselines:
    """Formal API fixes every HGB parameter; no calibration or target source fits."""
    return _fit_baselines(X, point, truth, std_hour, std_level, std_shape, regions, keys,
                         outer=outer, budget=budget, feature_names=feature_names, parameters=PARAMETERS.copy())


def _fit_baselines(X: np.ndarray, point: np.ndarray, truth: np.ndarray, std_hour: np.ndarray,
                   std_level: np.ndarray, std_shape: np.ndarray, regions: Sequence[str], keys: Sequence[str],
                   *, outer: str, budget: str, feature_names: Sequence[str],
                   parameters: Mapping[str, Any]) -> RiskBaselines:
    """Private synthetic speed profile is audited; formal callers use fit_baselines."""
    if budget not in FEATURE_NAMES or tuple(feature_names) != FEATURE_NAMES[budget]:
        raise ValueError('Exact B0/B1 feature-name whitelist and order required')
    r = _metadata(regions, keys, outer, training=True)
    if set(parameters) != set(PARAMETERS) or type(parameters['max_features']) is not float:
        raise ValueError('Complete fixed-parameter schema required')
    x, p, _, l, s = _data(X, point, std_hour, std_level, std_shape, len(r), FEATURE_NAMES[budget])
    y = _real(truth, p.shape, 'proper truth')
    error = y-p
    if not np.isfinite(error).all():
        raise ValueError('Finite proper signed errors required')
    level = error.mean(1)
    centered = error-level[:, None]
    weights = _balanced(r)
    fit_records = []

    def fitted(target: np.ndarray, mask: np.ndarray, name: str, *, quantile: float | None = None) -> Any:
        labels = r[mask]
        day_weights = _balanced(labels)  # recomputed after every source exclusion
        params = dict(parameters, loss='squared_error' if quantile is None else 'quantile')
        if quantile is not None: params['quantile'] = quantile
        head = HistGradientBoostingRegressor(**params)
        head.fit(x[mask].reshape(-1, x.shape[-1]), target[mask].reshape(-1),
                 sample_weight=np.repeat(day_weights, 24))
        fit_records.append(dict(head=name, training_keys=[key for key, keep in zip(keys, mask) if keep],
            training_sources=sorted(set(labels)), training_days=int(mask.sum()), training_hours=int(mask.sum()*24),
            sample_weight_sum=float(day_weights.sum()*24), parameters=head.get_params(deep=False)))
        return head

    full = np.ones(len(r), bool)
    models = dict(scale=fitted(np.log1p(abs(error)), full, 'HETScale'))
    oof = np.empty_like(error); folds = []
    for source in SOURCES:
        if source == outer: continue
        train, held = r != source, r == source
        mean = fitted(error, train, 'location_OOF_'+source)
        prediction = _real(mean.predict(x[held].reshape(-1, x.shape[-1])), (int(held.sum())*24,), 'OOF bias')
        oof[held] = np.clip(prediction.reshape(-1, 24), -1000, 1000)
        folds.append(dict(held_source=source, training_sources=sorted(set(r[train])),
                          held_keys=[key for key, keep in zip(keys, held) if keep]))
    models['location'] = fitted(error, full, 'location_final')
    target = np.log1p(abs(error-oof))
    models['location_scale'] = fitted(target, full, 'location_scale')
    models['q05'] = fitted(error, full, 'CQR05', quantile=.05)
    models['q95'] = fitted(error, full, 'CQR95', quantile=.95)
    return RiskBaselines(outer, budget, FEATURE_NAMES[budget], r.copy(), level/np.maximum(l, 1),
        centered/np.maximum(s, 1)[:, None], models,
        dict(outer=outer, budget=budget, feature_names=FEATURE_NAMES[budget], fit_count=len(fit_records),
             parameters=dict(parameters), synthetic_parameters_override=dict(parameters) != PARAMETERS,
             fits=fit_records, mean_oof=folds, training_keys=list(keys), training_regions=r.copy(),
             sample_weight_days=weights, hourly_weight_definition='repeat N/(K*n_source)24 times; mass24N',
             oof_bias_clipped=oof, location_scale_log_target=target,
             risk_mean_oof_is_base_teacher_third_source_exclusion=False,
             oof_scores_are_external_validation=False, postclip_statistics_are_caller_supplied=True,
             calibration_data_used=False, method_innovation_claim=False))


def _bounds(lo0: np.ndarray, hi0: np.ndarray, scales: np.ndarray) -> tuple[np.ndarray, ...]:
    lo = np.asarray(lo0)
    if lo.ndim not in (2, 3) or lo.shape[-1] != 24 or not lo.size:
        raise ValueError('Nonempty hourly bounds required')
    lo = _real(lo, lo.shape, 'initial lower')
    hi, s = _real(hi0, lo.shape, 'initial upper'), _real(scales, lo.shape, 'scale')
    if np.any(lo > hi) or np.any(s <= 0):
        raise ValueError('Ordered initial bounds and strictly positive scales required')
    return lo, hi, s


def scores(lo0: np.ndarray, hi0: np.ndarray, scales: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Hourly nonnegative interval scores, never max-over-hour path scores."""
    lo, hi, s = _bounds(lo0, hi0, scales)
    y = _real(truth, lo.shape[-2:], 'calibration truth')
    result = np.maximum(np.maximum((lo-y)/s, (y-hi)/s), 0)
    if not np.isfinite(result).all():
        raise ValueError('Finite calibration scores required')
    return result


def calibrate(score: np.ndarray, regions: Sequence[str], rule: str) -> dict[str, Any]:
    """Exact equal-source/hour empirical quantile with optional HCP infinity mass."""
    r = np.asarray(regions, str)
    if r.ndim != 1 or not len(r) or not set(r) <= set(DONOR_SOURCES) or rule not in ('equal_region', 'HCP'):
        raise ValueError('Only external calibration sources and a fixed rule allowed')
    a = _real(score, (len(r), 24), 'hourly calibration scores')
    if np.any(a < 0):
        raise ValueError('Nonnegative calibration scores required')
    k = len(set(r))
    probability = Fraction(9, 10) if rule == 'equal_region' else Fraction(9*(k+1), 10*k)
    if probability > 1:
        q = np.inf
    else:
        unit, total = _units(r, 24)
        values = a.reshape(-1)
        order = np.argsort(values, kind='stable')
        cdf = np.cumsum(unit[order])
        threshold = (probability.numerator*total+probability.denominator-1)//probability.denominator
        q = float(values[order[int(np.sum(cdf < threshold))]])
    return dict(q=q, K=k, finite_level=float(probability), rule=rule,
        infinite_mass=0. if rule == 'equal_region' else 1/(k+1), q_is_infinite=bool(np.isinf(q)),
        source_counts={s: int(np.count_nonzero(r == s)) for s in sorted(set(r))},
        finite_weight_definition='source equal mass; days and24hours equal within source',
        coverage_guarantee_claim=False, path_coverage_guarantee_claim=False, refitted=False)


def final_bounds(lo0: np.ndarray, hi0: np.ndarray, scales: np.ndarray, q: float) -> dict[str, np.ndarray]:
    """Clip expanded marginal bounds; infinite q explicitly returns physical support."""
    lo, hi, s = _bounds(lo0, hi0, scales)
    if np.iscomplexobj(q) or np.asarray(q).shape != ():
        raise ValueError('Nonnegative scalar quantile required')
    q = float(q)
    if np.isnan(q) or q < 0:
        raise ValueError('Nonnegative scalar quantile required')
    if np.isinf(q):
        return dict(lo=np.zeros_like(lo), hi=np.full_like(hi, 1000.))
    with np.errstate(over='ignore'):
        return dict(lo=np.clip(lo-q*s, 0, 1000), hi=np.clip(hi+q*s, 0, 1000))
