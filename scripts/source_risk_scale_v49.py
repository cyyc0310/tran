"""Pair-excluded H1 banks and the fixed V29 Scale teacher for H1/H2 retrospection.

Pure rows/arrays. No I/O, new risk calibration, background fabrication, mutable
old-module configuration, or claim of a historical first-vintage backtest.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts import conditional_law_v29 as old
from scripts import coarse_supervision_v28 as coarse
from scripts import group_regression_v27 as prior
from scripts import monthly_transfer_v25 as sampling
from transcif.research.fuel_transport_v7 import FuelTrajectoryBank, recompose_paths


SOURCES, POS, ZERO, EF = prior.SOURCES, prior.POS, prior.ZERO, prior.EF
SEED, SAMPLES, TREES = 17023, 128, 128
BANK_START = pd.Timestamp('2024-01-25', tz='UTC')
BANK_CUTOFF = pd.Timestamp('2024-07-01', tz='UTC')
DAYS = pd.date_range('2024-01-25', '2024-12-31', tz='UTC').as_unit('ns')


def _excluded(value: tuple[str, ...], lengths: tuple[int, ...] = (1, 2)) -> tuple[str, ...]:
    if (type(value) is not tuple or len(value) not in lengths or len(set(value)) != len(value)
            or value != tuple(sorted(value)) or not set(value) <= set(SOURCES)):
        raise ValueError('Canonical sorted registered exclusion tuple required')
    return value


def _real(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError('Real '+name+' required')
    try:
        result = np.asarray(raw, float)
    except (TypeError, ValueError) as exc:
        raise ValueError('Real numeric '+name+' required') from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError('Finite matching '+name+' required')
    return result


def validate_pair_bank(bank: Mapping[str, Any], pair: tuple[str, str]) -> None:
    """The formal entry accepts exactly two distinct canonical exclusions."""
    validate_bank(bank, _excluded(pair, (2,)))


def validate_bank(bank: Mapping[str, Any], allowed_excluded_tuple: tuple[str, ...]) -> None:
    """Validate pair banks; a single exclusion exists only for synthetic parity."""
    excluded = _excluded(allowed_excluded_tuple)
    expected = tuple(s for s in SOURCES if s not in excluded)
    if (not np.array_equal(bank['excluded'], excluded)
            or not np.array_equal(bank['source_regions'], expected)):
        raise ValueError('Exact excluded and ordered source registry required')
    regions, days = np.asarray(bank['regions']), np.asarray(bank['days'])
    n = len(regions)
    row_ids, cutoff = np.asarray(bank['row_ids']), np.asarray(bank['cutoff'])
    if (not n or regions.shape != (n,) or set(regions) != set(expected) or days.shape != (n,)
            or not np.issubdtype(days.dtype, np.integer) or not np.issubdtype(row_ids.dtype, np.integer)
            or not np.array_equal(row_ids, np.arange(n)) or cutoff.shape != ()
            or not np.issubdtype(cutoff.dtype, np.integer) or cutoff.item() != BANK_CUTOFF.value):
        raise ValueError('Integer fixed-H1 row/date/cutoff provenance required')
    dates = pd.DatetimeIndex(pd.to_datetime(days, unit='ns', utc=True)).as_unit('ns')
    months = np.asarray(bank['months'])
    if (dates.hasnans or not dates.equals(dates.normalize()) or np.any(dates < BANK_START)
            or np.any(dates >= BANK_CUTOFF) or not np.array_equal(months, dates.month)
            or not np.issubdtype(months.dtype, np.integer)
            or pd.MultiIndex.from_arrays([regions, days]).has_duplicates):
        raise ValueError('Unique source-day rows and exact H1 months required')
    q = _real(bank['q'], (n, 24, 8), 'q')
    raw = _real(bank['raw_cif'], (n, 24), 'raw_cif')
    mix = _real(bank['donor_mix'], (n, 8), 'donor_mix')
    monthly = _real(bank['monthmean'], (n, 8), 'monthmean')
    support = np.asarray(bank['support'])
    prior._simplex(mix, 8, 'bank mix')
    if (support.shape != (n, 8) or support.dtype != np.bool_ or np.any(q < 0)
            or np.any(raw < 0) or np.any(raw > 1000) or np.any(monthly < 0)
            or np.any(monthly.sum(1) <= 0) or not np.array_equal(support, monthly > 0)
            or not np.array_equal(support, mix > 0)
            or not np.allclose(mix, monthly/monthly.sum(1)[:, None], atol=1e-12, rtol=0)
            or np.any(q[~np.broadcast_to(support[:, None], q.shape)] != 1)):
        raise ValueError('Valid q/support/raw and observed-support monthly normalization required')
    for source in expected:
        for month in np.unique(months[regions == source]):
            take = (regions == source) & (months == month)
            if (not np.array_equal(monthly[take], np.broadcast_to(monthly[take][0], (take.sum(), 8)))
                    or not np.array_equal(mix[take], np.broadcast_to(mix[take][0], (take.sum(), 8)))
                    or not np.allclose(q[take].mean((0, 1)), 1., atol=1e-10, rtol=0)):
                raise ValueError('Exact region-month mix and mean-one observed-support q required')
    if not np.allclose(recompose_paths(mix, q), raw, atol=1e-9, rtol=0):
        raise ValueError('Bank own-month raw reconstruction failed')


def _region_months(bank: Mapping[str, Any], excluded: tuple[str, ...]) -> dict[str, Any]:
    """Original key order and source-month mapping, without the old six-source gate."""
    mixes, regions, months, counts, keys = [], [], [], [], []
    mapping = np.full(len(bank['days']), -1, np.int64)
    for source in SOURCES:
        if source in excluded:
            continue
        for month in sorted(set(bank['months'][bank['regions'] == source])):
            take = np.flatnonzero((bank['regions'] == source) & (bank['months'] == month))
            mapping[take] = len(mixes)
            mixes.append(bank['donor_mix'][take[0]].copy())
            regions.append(source); months.append(month); counts.append(len(take))
            keys.append(f'{source}/2024-{month:02d}')
    mix = np.stack(mixes)
    positive = mix[:, POS].sum(1)
    ratios = np.divide(mix[:, POS], positive[:, None], out=np.zeros((len(mix), 4)), where=positive[:, None] > 0)
    return dict(X=np.column_stack([positive, mix[:, ZERO]]), ratios=ratios, regions=np.array(regions),
                months=np.array(months, np.int64), bank_month_ids=mapping,
                row_counts=np.array(counts, np.int64), keys=keys, zero_pos_mask=positive == 0)


@dataclass
class ScaleContext:
    bank: Mapping[str, Any]
    excluded: tuple[str, ...]
    table: dict[str, Any]
    history: dict[str, Any]
    final: old.LeafLaw
    folds: dict[str, old.LeafLaw]
    oof_mean: np.ndarray
    oof_scale: np.ndarray
    oof_residual: np.ndarray
    oof_covariance3: np.ndarray
    oof_leaf_ids: np.ndarray
    oof_weights: np.ndarray
    audit: dict[str, Any]


def fit_scale(bank: Mapping[str, Any], monthly_rows: Sequence[Mapping[str, str]],
              pair: tuple[str, str]) -> ScaleContext:
    """Six fixed ET fits for a pair: one five-source head and five four-source OOFs."""
    return _fit_scale(bank, monthly_rows, _excluded(pair, (2,)))


def _fit_scale(bank: Mapping[str, Any], monthly_rows: Sequence[Mapping[str, str]],
               excluded: tuple[str, ...]) -> ScaleContext:
    """Single-exclusion mode is solely for synthetic comparison to sealed V29."""
    validate_bank(bank, excluded)
    table = _region_months(bank, excluded)
    # No metadata or fine response of either excluded region is passed downstream.
    rows = [row for row in monthly_rows if row['region'] not in excluded]
    history = coarse.select_coarse_rows(rows, outer=excluded[0], window='L24')
    positive, labels = history['positive_mask'], history['regions']
    sources = [s for s in SOURCES if s not in excluded]
    if set(labels[positive]) != set(sources) or not np.any(~table['zero_pos_mask']):
        raise ValueError('Positive supervision from every allowed source and a positive bank month required')
    final = old._fit_leaf(history, positive, excluded[0], None)
    final.metadata['excluded_pair'] = list(excluded)
    n = len(table['keys'])
    means, scales = np.zeros((n, 4)), np.empty(n)
    covariance, leaves = np.empty((n, 3, 3)), np.empty((n, TREES), np.int64)
    weights = np.zeros((n, len(final.y)))
    global_indices = np.flatnonzero(positive)
    folds, metadata = {}, []
    for teacher in sources:
        train = positive & (labels != teacher)
        fold = old._fit_leaf(history, train, excluded[0], teacher)
        fold.metadata.update(excluded_pair=list(excluded), excluded_regions=sorted((*excluded, teacher)))
        folds[teacher] = fold
        held = table['regions'] == teacher
        prediction = fold.conditional(table['X'][held])
        means[held], scales[held] = prediction['mean'], prediction['scale']
        covariance[held], leaves[held] = prediction['covariance3'], prediction['leaf_ids']
        columns = np.searchsorted(global_indices, np.flatnonzero(train))
        weights[np.ix_(np.flatnonzero(held), columns)] = prediction['weights']
        metadata.append(dict(fold.metadata,
            bank_response_keys=[key for key, keep in zip(table['keys'], held) if keep],
            bank_response_X=table['X'][held].tolist(), bank_prediction_raw=means[held].tolist(),
            bank_scale=scales[held].tolist()))
    residual = np.zeros_like(means)
    nonzero = ~table['zero_pos_mask']
    residual[nonzero] = table['ratios'][nonzero]-means[nonzero]
    audit = dict(excluded_pair=list(excluded), fit_count=1+len(folds), fitted_head_models=1+len(folds),
        final_source_count=len(sources), oof_source_count=len(sources)-1, window='L24',
        bank_start=BANK_START.isoformat(), bank_cutoff_exclusive=BANK_CUTOFF.isoformat(),
        bank_source_regions=sources, bank_row_count=len(bank['days']),
        training_snapshot=coarse.CUTOFF.isoformat(), history=history['audit'], history_keys=history['keys'],
        history_zero_pos_mask=(~positive).tolist(), fits=dict(final=final.metadata, folds=metadata),
        source_bank_keys=table['keys'], source_bank_row_counts=table['row_counts'].tolist(),
        source_month_zero_pos=table['zero_pos_mask'].tolist(), source_bank_month_ids=table['bank_month_ids'].tolist(),
        scale_floor_squared=old.SCALE_FLOOR_SQUARED, helmert_basis=old.HELMERT.tolist(),
        source_geographic_oof=True, chronological_online_backtest=False,
        head_and_bank_label_definitions_differ=True, q_bank_expanded=False,
        residual_definition='observed-support bank month ratio minus source-excluded full-calendar-label head mean',
        scale_definition='sqrt(trace full-four-coordinate conditional covariance + 1e-12)',
        projection_order='raw conditional mean plus std-ratio times bank OOF residual, then one simplex projection',
        seed=SEED, samples=SAMPLES, no_query_hidden_labels=True, method_innovation_claim=False)
    return ScaleContext(bank, excluded, table, history, final, folds, means, scales, residual,
                        covariance, leaves, weights, audit)


def predict(context: ScaleContext, MGmix: np.ndarray, MGdraw: np.ndarray,
            days: Sequence[Any], region: str) -> dict[str, Any]:
    """Replay both panels from the Jan25 snapshot, not an online temporal backtest.

    H1 can use later other-source H1 bank rows; pair exclusions remain geographic.
    Only H2 origins are all at or after the frozen bank's exclusive cutoff.
    """
    dates = pd.DatetimeIndex(days)
    if dates.tz is None or region not in context.excluded:
        raise ValueError('Timezone-aware H1/H2 days and an excluded held query region required')
    dates = dates.tz_convert('UTC').as_unit('ns')
    if not dates.equals(DAYS):
        raise ValueError('All 342 ordered UTC Jan25 through Dec31 midnight origins required')
    n = len(DAYS)
    mix = _real(MGmix, (n, 5), 'MG point')
    draws = _real(MGdraw, (n, SAMPLES, 5), 'MG draws')
    prior._simplex(mix, 5, 'MG point'); prior._simplex(draws, 5, 'MG draws')
    count = len(context.table['keys'])
    scale = _real(context.oof_scale, (count,), 'source OOF scale')
    _real(context.oof_residual, (count, 4), 'source OOF residual')
    if np.any(scale <= 0):
        raise ValueError('Strictly positive source OOF standard deviations required')
    query = context.final.conditional(draws.reshape(-1, 5))
    query = {name: value.reshape((n, SAMPLES)+value.shape[1:]) for name, value in query.items()
             if name in ('mean', 'scale', 'covariance3')}
    output = {name: [] for name in ('A', 'R', 'fine', 'ratios', 'unprojected_ratios', 'projection_l1',
        'donor_ids', 'fallback_ids', 'effective_pos_ids', 'effective_month_ids', 'source_scale', 'scale_ratio',
        'zero_total_mask', 'unsupported_positive_energy_fraction')}
    for t, day in enumerate(dates):
        uniforms = FuelTrajectoryBank._rng(SEED, day).random((SAMPLES, 8))
        weight = sampling.donor_weights(context.bank, mix[t], day, budget='MG', conditional=True)
        ids = sampling._sample(weight, uniforms[:, 0])
        original, original_q, fallback = sampling._fine_draws(context.bank, draws[t], ids, weight, uniforms, 'MG')
        q, support, selected, donor_ratio = prior._effective(context.bank, ids, fallback, draws[t, :, 0])
        if (not np.array_equal(q, original_q)
                or not np.array_equal(original[:, POS], draws[t, :, :1]*donor_ratio)):
            raise ValueError('Original sampler and effective donor stream differ')
        mapping = context.table['bank_month_ids'][selected]
        source_scale = scale[mapping]
        multiplier = query['scale'][t]/source_scale
        before = query['mean'][t]+multiplier[:, None]*context.oof_residual[mapping]
        ratios = prior.project(before)
        fine = original.copy(); fine[:, POS] = draws[t, :, :1]*ratios
        physical, zero = recompose_paths(fine, q, return_zero_total=True)
        anchor = context.bank['raw_cif'][ids]+((fine-context.bank['donor_mix'][ids])@EF)[:, None]
        values = dict(A=anchor, R=physical, fine=fine, ratios=ratios, unprojected_ratios=before,
            projection_l1=abs(ratios-before).sum(1), donor_ids=ids, fallback_ids=fallback,
            effective_pos_ids=selected, effective_month_ids=mapping, source_scale=source_scale, scale_ratio=multiplier,
            zero_total_mask=zero, unsupported_positive_energy_fraction=prior._unsupported(fine, q, support))
        for name, value in values.items():
            if not np.isfinite(value).all():
                raise ValueError('Finite Scale law outputs required')
            output[name].append(value)
    result = {name: np.stack(value) for name, value in output.items()}
    prior._simplex(result['fine'], 8, 'predicted fine mix')
    result.update(query_mean=query['mean'], query_scale=query['scale'], query_covariance3=query['covariance3'],
        U=result['fine']@EF, P=draws[..., 0]*(query['mean']@EF[POS]),
        audit=dict(excluded_pair=list(context.excluded), held_region=region, days=n,
            start=DAYS[0].isoformat(), end_exclusive='2025-01-01T00:00:00+00:00',
            H1_future_other_source_bank_allowed=True, H1_temporal_bank_holdout=False,
            H1_query_days=int(np.count_nonzero(DAYS < BANK_CUTOFF)),
            H2_query_days=int(np.count_nonzero(DAYS >= BANK_CUTOFF)), H2_origins_after_bank_cutoff=True,
            seed=SEED, samples=SAMPLES, donor_sampling='V25 MG conditional with original RNG[:,0] and fallback[:,2]',
            source_fits_reused=True, new_fits=0, hourly_truth_read=False, labels_used_as_query_features=False,
            source_scale_is_standard_deviation=True, chronological_online_backtest=False))
    return result
