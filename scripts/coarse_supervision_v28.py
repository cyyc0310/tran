"""Fixed ordinary head procedures over alternative source coarse-label windows.

Pure string-row/array interfaces; query information and the H1 q/residual pool
are unchanged. Longer supervised history is not additional local information.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts import group_regression_v27 as prior
from transcif.research.fuel_transport_v7 import recompose_paths

SOURCES, POS, ZERO, EF = prior.SOURCES, prior.POS, prior.ZERO, prior.EF
WINDOWS = ('S24', 'L24', 'LR24')
CUTOFF = pd.Timestamp('2025-01-01', tz='UTC')
FUELS = ('coal', 'gas', 'oil', 'nuclear', 'hydro', 'wind', 'solar', 'other')
SHARES = tuple('fuelshare_'+f for f in FUELS)
ARMS = ('MG-recompose_conditional', 'V27-Ridge-residual_recompose', 'V27-ET-residual_recompose') + tuple(
    f'{window}-{name}-{mode}' for window in WINDOWS for name in ('Ridge', 'ET')
    for mode in ('plugin_recompose', 'residual_anchor', 'residual_recompose'))
FIELDS = ('raw', 'fine_draws', 'ratios', 'unprojected_ratios', 'projection_l1',
          'unsupported_positive_energy_fraction', 'source_EF_outside', 'zero_total_mask')


def _utc(value: Any) -> pd.Timestamp:
    at = pd.Timestamp(value)
    if pd.isna(at) or at.tzinfo is None:
        raise ValueError('Explicit finite UTC metadata required')
    return at.tz_convert('UTC')


def select_coarse_rows(rows: Sequence[Mapping[str, str]], *, outer: str, window: str) -> dict[str, Any]:
    """Exclude outer and out-of-window/unreleased rows before share conversion."""
    if outer not in SOURCES or window not in WINDOWS:
        raise ValueError('Registered outer source and fixed supervision window required')
    lower = pd.Timestamp('2024-01-01' if window == 'S24' else '2022-01-01', tz='UTC')
    upper = CUTOFF if window == 'LR24' else pd.Timestamp('2024-07-01', tz='UTC')
    selected: dict[tuple[str, pd.Timestamp], tuple] = {}
    for row in rows:
        region = row['region']
        if region == outer:
            continue
        if region not in SOURCES:
            raise ValueError('Only registered source monthly rows accepted')
        start = _utc(row['period_start'])
        if not lower <= start < upper:
            continue
        end, released = _utc(row['period_end']), _utc(row['available_at'])
        if end > CUTOFF or released > CUTOFF:
            continue
        if (start != start.normalize() or start.day != 1 or end != start+pd.offsets.MonthBegin(1)
                or released < end):
            raise ValueError('Whole monthly row with valid release metadata required')
        key = (region, start)
        if key in selected:
            raise ValueError('Duplicate source region-month/vintage')
        value = np.array([float(row[c]) for c in SHARES])
        prior._simplex(value, 8, 'released source coarse response')
        selected[key] = (value, end, released)
    if not selected:
        raise ValueError('No eligible source coarse labels')
    keys = sorted(selected, key=lambda key: (SOURCES.index(key[0]), key[1]))
    mixes = np.stack([selected[k][0] for k in keys]); mass = mixes[:, POS].sum(1)
    ratio = np.divide(mixes[:, POS], mass[:, None], out=np.zeros((len(keys), 4)), where=mass[:, None] > 0)
    labels = np.asarray([k[0] for k in keys])
    return dict(X=np.column_stack([mass, mixes[:, ZERO]]), y=ratio, regions=labels,
        keys=[f'{r}/{at.year:04d}-{at.month:02d}' for r, at in keys], positive_mask=mass > 0,
        period_start=np.array([k[1].value for k in keys], np.int64),
        period_end=np.array([selected[k][1].value for k in keys], np.int64),
        available_at=np.array([selected[k][2].value for k in keys], np.int64),
        audit=dict(window=window, outer_excluded=outer, deployment_cutoff=CUTOFF.isoformat(),
            start_inclusive=lower.isoformat(), start_end_exclusive=upper.isoformat(),
            regions=sorted(set(labels)), absent_regions=[r for r in SOURCES if r != outer and r not in labels],
            eligible_months=len(keys), positive_months=int(np.count_nonzero(mass)),
            zero_pos_keys=[f'{r}/{at.year:04d}-{at.month:02d}' for (r, at), p in zip(keys, mass) if p == 0],
            source_month_numbers={r: int(np.count_nonzero(labels == r)) for r in SOURCES if r != outer},
            label_definition='source released full-calendar monthly proxy, not proper-bank observed-support month'))


def _fit(name: str, history: dict[str, Any], mask: np.ndarray, outer: str,
         held: str | None) -> tuple[Any, dict[str, Any]]:
    indices = np.flatnonzero(mask)
    weight = prior._weights(history['regions'][indices])
    model = prior._estimator(name)
    model.fit(history['X'][indices], history['y'][indices], sample_weight=weight)
    meta = dict(model=name, excluded_outer=outer, excluded_oof_region=held,
        training_keys=[history['keys'][i] for i in indices], training_indices=indices.tolist(),
        training_regions=sorted(set(history['regions'][indices])), sample_weights=weight.tolist(),
        training_X=history['X'][indices].tolist(), training_y=history['y'][indices].tolist(),
        period_start=history['period_start'][indices].tolist(), period_end=history['period_end'][indices].tolist(),
        available_at=history['available_at'][indices].tolist(), weight_mean=float(weight.mean()),
        parameters=dict(alpha=1., fit_intercept=True) if name == 'Ridge' else
        dict(n_estimators=128, min_samples_leaf=2, max_features=1., random_state=17023, n_jobs=1))
    if name == 'Ridge':
        meta.update(coefficients=model.coef_.tolist(), intercept=model.intercept_.tolist())
    else:
        meta['feature_importances'] = model.feature_importances_.tolist()
    return model, meta


def _plugin(bank: Mapping[str, np.ndarray], baseline: Mapping[str, np.ndarray], draws: np.ndarray,
            points: np.ndarray, bank_bounds: tuple[float, float]) -> dict[str, np.ndarray]:
    output: dict[str, list] = {k: [] for k in FIELDS}
    for t in range(len(draws)):
        q, support, _, _ = prior._effective(bank, baseline['donor_ids'][t], baseline['fallback_ids'][t], draws[t, :, 0])
        ratios = prior.project(points[t]); fine = np.zeros((128, 8))
        fine[:, ZERO] = draws[t, :, 1:]; fine[:, POS] = draws[t, :, :1]*ratios
        raw, zero = recompose_paths(fine, q, return_zero_total=True)
        lo, hi = bank_bounds; value = ratios@EF[POS]
        data = dict(raw=raw, fine_draws=fine, ratios=ratios, unprojected_ratios=points[t],
            projection_l1=abs(ratios-points[t]).sum(1),
            unsupported_positive_energy_fraction=prior._unsupported(fine, q, support),
            source_EF_outside=(value < lo-prior.SUPPORT_ATOL) | (value > hi+prior.SUPPORT_ATOL), zero_total_mask=zero)
        for k, v in data.items():
            output[k].append(v)
    return {k: np.stack(v) for k, v in output.items()}


@dataclass
class CoarseContext:
    bank: Mapping[str, np.ndarray]
    outer: str
    bank_table: dict[str, Any]
    histories: dict[str, dict[str, Any]]
    contexts: dict[str, prior.RegressionContext]
    audit: dict[str, Any]

    def predict(self, baseline: Mapping[str, np.ndarray], mg_draws: np.ndarray, days: pd.DatetimeIndex,
                background_v27: Mapping[str, np.ndarray]) -> dict[str, Any]:
        dates = pd.DatetimeIndex(pd.to_datetime(days, utc=True))
        if dates.empty or dates.hasnans or np.any(dates < CUTOFF):
            raise ValueError('Query origins must be on or after fixed deployment cutoff 2025-01-01')
        if (background_v27['arms'].tolist() != list(prior.ARMS)
                or not np.array_equal(background_v27['origins'], dates.asi8)
                or not np.array_equal(background_v27['donor_ids'], baseline['donor_ids'])
                or not np.array_equal(background_v27['fallback_ids'], baseline['fallback_ids'])):
            raise ValueError('Exact fixed V27 archive and shared sampler registry required')
        if (not np.array_equal(background_v27['raw'][0], baseline['raw'])
                or not np.array_equal(background_v27['fine_draws'][0], baseline['fine_draws'])):
            raise ValueError('Exact original MG background required')
        arms = {name: {k: background_v27[k][i].copy() for k in FIELDS}
                for name, i in zip(ARMS[:3], (0, 2, 4))}
        arrays: dict[str, np.ndarray] = {}
        audits = {}; head_bounds = []; head_distance = []
        head_outside = np.zeros((len(ARMS), len(dates), 128), bool)
        for window in WINDOWS:
            result = self.contexts[window].predict(baseline, mg_draws, dates)
            audits[window] = result['audit']
            history = self.histories[window]
            source = history['y'][history['positive_mask']]@EF[POS]
            lo, hi = float(source.min()), float(source.max()); head_bounds.append([lo, hi])
            x = history['X'][history['positive_mask']]
            head_distance.append(np.linalg.norm(mg_draws[:, :, None]-x[None, None], axis=-1).min(-1))
            for name in ('Ridge', 'ET'):
                point_key = 'point_ridge_raw' if name == 'Ridge' else 'point_et_raw'
                arrays[f'{window}_{name}_query_raw'] = result[point_key]
                arrays[f'{window}_{name}_oof_bank_residuals'] = self.contexts[window].residuals[name].copy()
                for mode in ('plugin_recompose', 'residual_anchor', 'residual_recompose'):
                    arm = f'{window}-{name}-{mode}'
                    if name == 'ET' and mode == 'plugin_recompose':
                        value = _plugin(self.bank, baseline, mg_draws, result[point_key], tuple(result['audit']['source_EF_support']))
                    else:
                        value = result['arms'][f'{name}-{mode}']
                    arms[arm] = value
                    scalar = value['ratios']@EF[POS]
                    head_outside[ARMS.index(arm)] = (scalar < lo-prior.SUPPORT_ATOL) | (scalar > hi+prior.SUPPORT_ATOL)
            if window == WINDOWS[0]:
                arrays.update(effective_pos_ids=result['effective_pos_ids'], effective_month_ids=result['effective_month_ids'],
                    source_bank_month_ids=result['source_bank_month_ids'], source_month_X=result['source_month_X'],
                    source_month_ratios=result['source_month_ratios'], source_month_regions=result['source_month_regions'],
                    source_month_numbers=result['source_month_numbers'], source_month_zero_pos=result['source_month_zero_pos'],
                    bank_query_nearest_visible_distance=result['query_nearest_visible_distance'])
        return dict(arms={a: arms[a] for a in ARMS}, **arrays,
            head_source_EF_bounds=np.asarray(head_bounds), head_source_EF_outside=head_outside,
            head_query_nearest_visible_distance=np.stack(head_distance), audit={**self.audit,
                'windows': audits, 'background_exact': True, 'no_file_io': True,
                'annual_values_used': False, 'new_fine_parsed': False, 'custom_method': False,
                'target_values_opened': False, 'unseen_values_opened': False,
                'source_EF_outside_definition': 'arm field uses fixed H1 bank ratio EF range; head_source_EF_outside uses window coarse-label range, backgrounds not applicable',
                'head_source_EF_outside_applicable': [False]*3+[True]*18,
                'head_source_EF_bounds': head_bounds,
                'visible_distance_definition': 'nearest positive coarse-row Euclidean distance, no routing or convex-hull claim',
                'probability_interpretation': 'fixed ordinary head procedures; residual source/q pairing and nonlinear centers may differ'})


def fit_heads(bank: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, str]], *, outer: str) -> CoarseContext:
    """Coarse-label heads; OOF residual responses always stay on fixed H1 bank months."""
    table = prior.region_months(bank, outer=outer)
    histories = {w: select_coarse_rows(rows, outer=outer, window=w) for w in WINDOWS}
    sets = [set(histories[w]['keys']) for w in WINDOWS]
    if not sets[0] <= sets[1] <= sets[2]:
        raise ValueError('Fixed supervision windows must be nested')
    contexts = {}; fit_audits = {}
    positive_bank = ~table['zero_pos_mask']
    for window in WINDOWS:
        history = histories[window]; positive = history['positive_mask']; labels = history['regions']
        models, residuals, fits = {}, {}, {}
        for name in ('Ridge', 'ET'):
            model, full = _fit(name, history, positive, outer, None); models[name] = model
            oof = np.zeros_like(table['ratios']); folds = []
            for teacher in [r for r in SOURCES if r != outer]:
                held = (table['regions'] == teacher) & positive_bank
                if not held.any():
                    folds.append(dict(excluded_oof_region=teacher, no_positive_bank_response=True)); continue
                train = positive & (labels != teacher)
                fold, meta = _fit(name, history, train, outer, teacher)
                oof[held] = fold.predict(table['X'][held])
                meta.update(residual_response_keys=[k for k, keep in zip(table['keys'], held) if keep],
                    bank_response_X=table['X'][held].tolist(), bank_response_y=table['ratios'][held].tolist(),
                    bank_prediction_raw=oof[held].tolist())
                folds.append(meta)
            delta = np.zeros_like(oof); delta[positive_bank] = table['ratios'][positive_bank]-oof[positive_bank]
            residuals[name] = delta
            fits[name] = dict(final=full, folds=folds, bank_oof_prediction_raw=oof.tolist(), bank_oof_residuals=delta.tolist())
        fit_audits[window] = dict(history=history['audit'], history_keys=history['keys'],
            zero_pos_mask=(~history['positive_mask']).tolist(), fits=fits)
        context_audit = dict(arms=list(prior.ARMS), outer=outer, fits=fits, source_keys=table['keys'],
            fit_dataset=window+' released coarse history', residual_dataset='fixed proper bank region-months',
            chronological_bank_forecast=False, source_geographic_oof=True)
        contexts[window] = prior.RegressionContext(bank, outer, table, models, residuals, context_audit)
    audit = dict(arms=list(ARMS), windows_order=list(WINDOWS), outer=outer, deployment_cutoff=CUTOFF.isoformat(),
        fit_windows=fit_audits, source_bank_keys=table['keys'], source_bank_row_counts=table['row_counts'].tolist(),
        fitted_head_models=sum(1+sum('no_positive_bank_response' not in f for f in v['folds'])
                               for w in fit_audits.values() for v in w['fits'].values()),
        fixed_residual_region_months=len(table['keys']), fixed_positive_residual_months=int(positive_bank.sum()),
        training_window_keys_nested=True, head_and_bank_label_definitions_differ=True,
        supervised_label_definition='released source full-calendar proxy monthly composition',
        residual_label_definition='proper-bank source observed-support H1 monthly composition',
        procedure_caveat='mean-one weights with fixed Ridge alpha1 change relative average-loss penalty with N; fixed ET leaf2 changes smoothing with N; no retuning',
        no_query_hidden_labels=True, training_features=list(prior.FEATURES), annual_fields_used=False, residual_pool_expanded=False,
        q_bank_expanded=False, seed=17023, samples=128)
    return CoarseContext(bank, outer, table, histories, contexts, audit)
