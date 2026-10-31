"""Ordinary visible-group regression of historical source monthly fuel splits.

Pure arrays only: no annual fields, query hidden composition, or file I/O.
Source region-month responses come from the inherited proper bank. Regression
plus geographic OOF residual transfer is not a new probability mechanism.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

from scripts.monthly_transfer_v25 import SOURCES, POS, ZERO, EF
from transcif.research.fuel_transport_v7 import recompose_paths

ARMS = ('MG-recompose_conditional', 'Ridge-residual_anchor', 'Ridge-residual_recompose',
        'ET-residual_anchor', 'ET-residual_recompose', 'Ridge-plugin_anchor', 'Ridge-plugin_recompose')
FEATURES = ('POS', 'nuclear', 'hydro', 'wind', 'solar')
SEED, SAMPLES = 17023, 128
SUPPORT_ATOL = 1e-10


def _simplex(x: np.ndarray, k: int, label: str) -> None:
    if (x.ndim < 1 or x.shape[-1] != k or not np.isfinite(x).all() or np.any(x < 0)
            or not np.allclose(x.sum(-1), 1., atol=1e-12, rtol=0)):
        raise ValueError('Finite simplex required: '+label)


def project(values: np.ndarray) -> np.ndarray:
    """Euclidean four-coordinate simplex projection, only after raw regression."""
    x = np.asarray(values, float)
    if x.ndim < 1 or x.shape[-1] != 4 or not np.isfinite(x).all():
        raise ValueError('Finite four-coordinate regression output required')
    flat = x.reshape(-1, 4)
    ordered = np.sort(flat, axis=1)[:, ::-1]
    candidates = (np.cumsum(ordered, axis=1)-1)/np.arange(1, 5)
    count = (ordered > candidates).sum(1)-1
    shift = candidates[np.arange(len(flat)), count]
    result = np.maximum(flat-shift[:, None], 0)
    result /= result.sum(1)[:, None]
    return result.reshape(x.shape)


def _bank(bank: Mapping[str, np.ndarray], outer: str) -> None:
    n = len(bank['days'])
    shapes = dict(q=(n, 24, 8), donor_mix=(n, 8), raw_cif=(n, 24), support=(n, 8),
                  regions=(n,), days=(n,))
    if (outer not in SOURCES or not n or set(bank['regions']) != set(SOURCES)-{outer}
            or any(np.asarray(bank[k]).shape != shape for k, shape in shapes.items())):
        raise ValueError('Six-source proper bank excluding outer region required')
    if not np.issubdtype(bank['days'].dtype, np.integer):
        raise ValueError('Integer-nanosecond proper bank dates required')
    dates = pd.DatetimeIndex(pd.to_datetime(bank['days'], utc=True))
    if (dates.hasnans or not dates.equals(dates.normalize())
            or np.any(dates < pd.Timestamp('2024-01-25', tz='UTC'))
            or np.any(dates >= pd.Timestamp('2024-07-01', tz='UTC'))
            or pd.MultiIndex.from_arrays([bank['regions'], bank['days']]).has_duplicates):
        raise ValueError('Unique region-day rows in fixed H1 proper window required')
    _simplex(bank['donor_mix'], 8, 'source monthly mix')
    if (bank['support'].dtype != np.bool_ or not np.array_equal(bank['support'], bank['donor_mix'] > 0)
            or not np.isfinite(bank['q']).all() or np.any(bank['q'] < 0)
            or np.any(bank['q'][~np.broadcast_to(bank['support'][:, None], bank['q'].shape)] != 1)
            or not np.isfinite(bank['raw_cif']).all() or np.any(bank['raw_cif'] < 0)
            or np.any(bank['raw_cif'] > 1000)):
        raise ValueError('Valid source q/support/raw CIF required')


def region_months(bank: Mapping[str, np.ndarray], *, outer: str) -> dict[str, Any]:
    """One response per region-month; donor-day multiplicity never reweights fit."""
    _bank(bank, outer)
    months = pd.DatetimeIndex(pd.to_datetime(bank['days'], utc=True)).month.to_numpy()
    regions, period, mixes, counts, keys = [], [], [], [], []
    mapping = np.full(len(months), -1, dtype=np.int64)
    for region in SOURCES:
        if region == outer:
            continue
        for month in sorted(set(months[bank['regions'] == region])):
            take = np.flatnonzero((bank['regions'] == region) & (months == month))
            values = bank['donor_mix'][take]
            if not np.array_equal(values, np.broadcast_to(values[0], values.shape)):
                raise ValueError('Region-month donor composition must be exact across all day rows')
            mapping[take] = len(mixes)
            mixes.append(values[0].copy()); regions.append(region); period.append(month)
            counts.append(len(take)); keys.append(f'{region}/2024-{month:02d}')
    mix = np.stack(mixes)
    positive = mix[:, POS].sum(1)
    ratios = np.divide(mix[:, POS], positive[:, None], out=np.zeros((len(mix), 4)), where=positive[:, None] > 0)
    return dict(X=np.column_stack([positive, mix[:, ZERO]]), ratios=ratios,
        regions=np.asarray(regions), months=np.asarray(period, dtype=np.int64),
        bank_month_ids=mapping, row_counts=np.asarray(counts, dtype=np.int64), keys=keys,
        zero_pos_mask=positive == 0, audit=dict(outer_excluded=outer, month_mix_exact=True,
            region_months=len(mix), bank_days=len(months), positive_months=int(np.count_nonzero(positive)),
            zero_pos_keys=[key for key, value in zip(keys, positive) if value == 0],
            features=list(FEATURES), query_hidden_fuel_features=False))


def _weights(labels: np.ndarray) -> np.ndarray:
    if not len(labels):
        raise ValueError('No positive-POS training months')
    return np.array([len(labels)/(len(set(labels))*np.count_nonzero(labels == r)) for r in labels])


def _estimator(name: str) -> Any:
    if name == 'Ridge':
        return Ridge(alpha=1., fit_intercept=True)
    return ExtraTreesRegressor(n_estimators=128, min_samples_leaf=2, max_features=1.,
                               random_state=SEED, n_jobs=1)


def _fit(name: str, table: dict[str, Any], mask: np.ndarray, outer: str,
         held: str | None) -> tuple[Any, dict[str, Any]]:
    indices = np.flatnonzero(mask)
    weight = _weights(table['regions'][indices])
    model = _estimator(name)
    model.fit(table['X'][indices], table['ratios'][indices], sample_weight=weight)
    meta = dict(model=name, excluded_outer=outer, excluded_oof_region=held,
        training_regions=sorted(set(table['regions'][indices])),
        training_keys=[table['keys'][i] for i in indices], training_indices=indices.tolist(),
        sample_weights=weight.tolist(), weight_mean=float(weight.mean()),
        weight_definition='N / (number of represented regions * region month count)',
        parameters=dict(alpha=1., fit_intercept=True) if name == 'Ridge' else
                   dict(n_estimators=128, min_samples_leaf=2, max_features=1., random_state=SEED, n_jobs=1))
    if name == 'Ridge':
        meta.update(coefficients=model.coef_.tolist(), intercept=model.intercept_.tolist())
    else:
        meta.update(feature_importances=model.feature_importances_.tolist())
    return model, meta


def _effective(bank: Mapping[str, np.ndarray], ids: np.ndarray, fallback: np.ndarray,
               positive: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    missing = (bank['donor_mix'][ids][:, POS].sum(1) == 0) & (positive > 0)
    if not np.array_equal(fallback >= 0, missing):
        raise ValueError('Shared fallback IDs inconsistent with positive query POS')
    selected = np.where(fallback >= 0, fallback, ids)
    mass = bank['donor_mix'][selected][:, POS].sum(1)
    if np.any((positive > 0) & (mass <= 0)):
        raise ValueError('Positive POS needs supported effective donor')
    ratio = np.divide(bank['donor_mix'][selected][:, POS], mass[:, None],
                      out=np.zeros((SAMPLES, 4)), where=mass[:, None] > 0)
    q = bank['q'][ids].copy(order='K'); support = bank['support'][ids].copy()
    for particle in np.flatnonzero(missing):
        q[particle][:, POS] = bank['q'][selected[particle]][:, POS]
        support[particle, POS] = bank['support'][selected[particle], POS]
    return q, support, selected, ratio


def _unsupported(fine: np.ndarray, q: np.ndarray, support: np.ndarray) -> np.ndarray:
    energy = (fine[:, None, POS]*q[:, :, POS]).sum(1)
    numerator = (energy*~support[:, POS]).sum(1); denominator = energy.sum(1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


@dataclass
class RegressionContext:
    bank: Mapping[str, np.ndarray]
    outer: str
    table: dict[str, Any]
    models: dict[str, Any]
    residuals: dict[str, np.ndarray]
    audit: dict[str, Any]

    def predict(self, baseline: Mapping[str, np.ndarray], mg_draws: np.ndarray,
                days: pd.DatetimeIndex) -> dict[str, Any]:
        """Matched fixed stream; raw regression and OOF deltas precede projection."""
        dates = pd.DatetimeIndex(pd.to_datetime(days, utc=True)); t = len(dates)
        draws = np.asarray(mg_draws, float)
        if (not t or dates.hasnans or dates.has_duplicates or not dates.equals(dates.normalize())
                or np.any(dates < pd.Timestamp('2024-07-01', tz='UTC')) or draws.shape != (t, SAMPLES, 5)):
            raise ValueError('Deployed UTC days with only five visible draw coordinates required')
        _simplex(draws, 5, 'query visible draws')
        for key in ('donor_ids', 'fallback_ids'):
            ids = baseline[key]
            if (ids.shape != (t, SAMPLES) or not np.issubdtype(ids.dtype, np.integer)
                    or np.any(ids < (-1 if key == 'fallback_ids' else 0)) or np.any(ids >= len(self.bank['days']))):
                raise ValueError('Integer shared stream IDs in bank range required')
        if baseline['raw'].shape != (t, SAMPLES, 24) or baseline['fine_draws'].shape != (t, SAMPLES, 8):
            raise ValueError('Complete inherited MG stream required')
        _simplex(baseline['fine_draws'], 8, 'baseline modeled composition')
        if not np.isfinite(baseline['raw']).all():
            raise ValueError('Finite inherited MG paths required')
        points = {name: model.predict(draws.reshape(-1, 5)).reshape(t, SAMPLES, 4)
                  for name, model in self.models.items()}
        output = {a: {k: [] for k in ('raw', 'fine_draws', 'ratios', 'unprojected_ratios',
            'projection_l1', 'unsupported_positive_energy_fraction', 'source_EF_outside', 'zero_total_mask')} for a in ARMS}
        effective_ids = np.empty((t, SAMPLES), np.int64)
        month_ids = np.empty_like(effective_ids)
        source_ef = self.table['ratios'][~self.table['zero_pos_mask']]@EF[POS]
        lo, hi = float(source_ef.min()), float(source_ef.max())
        visible_distances = np.linalg.norm(draws[:, :, None, :]-self.table['X'][~self.table['zero_pos_mask']][None, None], axis=-1).min(-1)
        for day in range(t):
            ids, fallback = baseline['donor_ids'][day], baseline['fallback_ids'][day]
            q, support, selected, ratio = _effective(self.bank, ids, fallback, draws[day, :, 0])
            effective_ids[day] = selected
            months = self.table['bank_month_ids'][selected]; month_ids[day] = months
            fine = np.zeros((SAMPLES, 8)); fine[:, ZERO] = draws[day, :, 1:]; fine[:, POS] = draws[day, :, :1]*ratio
            if not np.array_equal(fine, baseline['fine_draws'][day]):
                raise ValueError('Exact shared MG group draws/effective split required')
            original_raw, _ = recompose_paths(fine, q, return_zero_total=True)
            if not np.array_equal(original_raw, baseline['raw'][day]):
                raise ValueError('Exact original MG raw recomposition required')
            variants = {ARMS[0]: ratio}
            for name in ('Ridge', 'ET'):
                raw = points[name][day]+self.residuals[name][months]
                variants[name+'-residual_anchor'] = raw
                variants[name+'-residual_recompose'] = raw
            variants['Ridge-plugin_anchor'] = points['Ridge'][day]
            variants['Ridge-plugin_recompose'] = points['Ridge'][day]
            for arm in ARMS:
                before = variants[arm]
                after = ratio.copy() if arm == ARMS[0] else project(before)
                fd = baseline['fine_draws'][day].copy(); fd[:, POS] = draws[day, :, :1]*after
                physical, zero = recompose_paths(fd, q, return_zero_total=True)
                if arm == ARMS[0]:
                    raw = baseline['raw'][day].copy()
                elif arm.endswith('_anchor'):
                    raw = self.bank['raw_cif'][ids]+((fd-self.bank['donor_mix'][ids])@EF)[:, None]
                else:
                    raw = physical
                value = after@EF[POS]
                values = dict(raw=raw, fine_draws=fd, ratios=after, unprojected_ratios=before,
                    projection_l1=abs(after-before).sum(1),
                    unsupported_positive_energy_fraction=_unsupported(fd, q, support),
                    source_EF_outside=(value < lo-SUPPORT_ATOL) | (value > hi+SUPPORT_ATOL),
                    zero_total_mask=zero)
                for key, val in values.items():
                    output[arm][key].append(val)
        arrays = {a: {k: np.stack(v) for k, v in fields.items()} for a, fields in output.items()}
        return dict(arms=arrays, effective_pos_ids=effective_ids, effective_month_ids=month_ids,
            donor_ids=baseline['donor_ids'].copy(), fallback_ids=baseline['fallback_ids'].copy(),
            point_ridge_raw=points['Ridge'], point_et_raw=points['ET'],
            ridge_oof_residuals=self.residuals['Ridge'].copy(), et_oof_residuals=self.residuals['ET'].copy(),
            source_month_X=self.table['X'].copy(), source_month_ratios=self.table['ratios'].copy(),
            source_month_regions=self.table['regions'].copy(), source_month_numbers=self.table['months'].copy(),
            source_month_zero_pos=self.table['zero_pos_mask'].copy(),
            source_bank_month_ids=self.table['bank_month_ids'].copy(),
            query_nearest_visible_distance=visible_distances, audit=dict(self.audit,
                source_EF_support=[lo, hi], source_EF_support_atol=SUPPORT_ATOL,
                source_EF_support_definition='scalar ratio dot EF_POS, not four-dimensional hull or path CIF support',
                visible_distance_definition='Euclidean distance to nearest positive-POS training row, not convex-hull distance; no routing',
                source_EF_outside_fractions={a: float(v['source_EF_outside'].mean()) for a, v in arrays.items()},
                original_mg_raw_preserved_exact=True, original_mg_fine_preserved_exact=True,
                unsupported_definition='unsupported modeled POS energy / all modeled POS energy; zero if no POS energy',
                no_file_io=True, new_fine_parsed=False, annual_fields_used=False,
                target_values_opened=False, unseen_values_opened=False, custom_method=False,
                probability_control_interpretation='ordinary projected regression plus geographic source residuals; dependence and center may change',
                ET_point_interpretation='weighted leaf interpolation; OOF residual addition may extend source ratio support',
                future_realized_energy_conserved=False))


def fit_regressions(bank: Mapping[str, np.ndarray], *, outer: str) -> RegressionContext:
    """Final six-source fits and strictly source-excluded OOF ratio responses."""
    table = region_months(bank, outer=outer)
    positive = ~table['zero_pos_mask']; labels = table['regions']
    if len(set(labels[positive])) < 2:
        raise ValueError('At least two positive-response source regions needed for source OOF')
    models, residuals, fits = {}, {}, {}
    for name in ('Ridge', 'ET'):
        model, full = _fit(name, table, positive, outer, None)
        models[name] = model
        prediction = np.zeros_like(table['ratios']); folds = []
        for source in [r for r in SOURCES if r != outer]:
            held = (labels == source) & positive
            if not held.any():
                folds.append(dict(excluded_oof_region=source, no_positive_response=True))
                continue
            train = positive & (labels != source)
            fold, meta = _fit(name, table, train, outer, source)
            prediction[held] = fold.predict(table['X'][held])
            meta.update(predicted_keys=[key for key, take in zip(table['keys'], held) if take],
                        held_prediction_raw=prediction[held].tolist())
            folds.append(meta)
        delta = np.zeros_like(prediction); delta[positive] = table['ratios'][positive]-prediction[positive]
        residuals[name] = delta
        fits[name] = dict(final=full, folds=folds, final_month_predictions_raw=model.predict(table['X']).tolist(),
            oof_predictions_raw=[row.tolist() if keep else None for row, keep in zip(prediction, positive)],
            oof_residuals=delta.tolist())
    audit = dict(arms=list(ARMS), seed=SEED, samples=SAMPLES, outer=outer,
        cohort=table['audit'], source_keys=table['keys'], source_row_counts=table['row_counts'].tolist(),
        source_month_regions=table['regions'].tolist(), source_month_numbers=table['months'].tolist(),
        source_month_visible_X=table['X'].tolist(), source_month_ratio_truth=table['ratios'].tolist(),
        source_month_zero_pos=table['zero_pos_mask'].tolist(), fits=fits,
        final_and_fold_weights='mean one; recomputed from each training cohort after zero-POS exclusion',
        projection_order='raw sklearn outputs; delta=y-raw OOF prediction; final-only simplex projection')
    return RegressionContext(bank, outer, table, models, residuals, audit)
