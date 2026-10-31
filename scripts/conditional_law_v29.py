"""Ordinary weighted ET leaf, Gaussian and paired location-scale controls.

Pure source arrays and released monthly string rows only. No file I/O, new
fine observations, tuning or claim of a new conditional-distribution method.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts import coarse_supervision_v28 as coarse
from scripts import group_regression_v27 as prior
from transcif.research.fuel_transport_v7 import recompose_paths

SOURCES, POS, ZERO, EF = prior.SOURCES, prior.POS, prior.ZERO, prior.EF
SEED, SAMPLES, TREES = 17023, 128, 128
CUTOFF, FIELDS = coarse.CUTOFF, coarse.FIELDS
ARMS = ('L24-ET-plugin_recompose', 'L24-ET-residual_anchor', 'L24-ET-residual_recompose',
        'Leaf-anchor', 'Leaf-recompose', 'Gaussian-anchor', 'Gaussian-recompose',
        'Scale-anchor', 'Scale-recompose', 'Scale-shuffle_recompose')
SCALE_FLOOR_SQUARED = 1e-12
STREAM_SALTS = dict(leaf=2901, gaussian=2902, shuffle=2903)
HELMERT = np.array([[1/np.sqrt(2), 1/np.sqrt(6), 1/np.sqrt(12)],
                    [-1/np.sqrt(2), 1/np.sqrt(6), 1/np.sqrt(12)],
                    [0., -2/np.sqrt(6), 1/np.sqrt(12)], [0., 0., -3/np.sqrt(12)]])


def _rng(day: pd.Timestamp, salt: int) -> np.random.Generator:
    stamp = int(day.value)
    return np.random.default_rng(np.random.SeedSequence([SEED, salt, stamp & 0xffffffff, stamp >> 32]))


def sample_rows(weights: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    """Whole-response inverse CDF with the final positive support capped at one."""
    w, u = np.asarray(weights, float), np.asarray(uniforms, float)
    if (w.ndim != 2 or u.shape != w.shape[:1] or not np.isfinite(w).all()
            or np.any(w < 0) or not np.allclose(w.sum(1), 1., atol=1e-12, rtol=0)
            or not np.isfinite(u).all() or np.any(u < 0) or np.any(u >= 1)):
        raise ValueError('Normalized nonnegative row weights and uniforms in [0,1) required')
    cdf = np.cumsum(w, axis=1)
    last = np.max(np.where(w > 0, np.arange(w.shape[1]), -1), axis=1)
    cdf[np.arange(w.shape[1])[None] >= last[:, None]] = 1.
    selected = np.sum(u[:, None] >= cdf, axis=1)
    if np.any(w[np.arange(len(w)), selected] <= 0):
        raise ValueError('Conditional CDF selected zero support')
    return selected.astype(np.int64)


def gaussian_ratios(mean: np.ndarray, covariance3: np.ndarray, normal3: np.ndarray) -> np.ndarray:
    """Unprojected full-covariance perturbation in a fixed orthonormal tangent basis."""
    mu, cov, z = np.asarray(mean, float), np.asarray(covariance3, float), np.asarray(normal3, float)
    if (mu.shape[-1:] != (4,) or cov.shape != mu.shape[:-1]+(3, 3)
            or z.shape != mu.shape[:-1]+(3,) or not all(np.isfinite(v).all() for v in (mu, cov, z))
            or not np.allclose(cov, cov.swapaxes(-1, -2), atol=1e-12, rtol=0)):
        raise ValueError('Finite matching mean, symmetric tangent covariance and normal arrays required')
    values, vectors = np.linalg.eigh(cov)
    if np.any(values < -1e-12):
        raise ValueError('Tangent covariance must be positive semidefinite')
    root = (vectors*np.sqrt(np.maximum(values, 0))[..., None, :])@vectors.swapaxes(-1, -2)
    displacement = np.einsum('...ij,...j->...i', root, z)
    return mu+displacement@HELMERT.T


@dataclass
class LeafLaw:
    model: Any
    X: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    leaf_ids: np.ndarray
    pools: list[dict[int, tuple[np.ndarray, np.ndarray]]]
    metadata: dict[str, Any]

    def conditional(self, features: np.ndarray) -> dict[str, np.ndarray]:
        x = np.asarray(features, float)
        if x.ndim != 2 or x.shape[1] != 5 or not len(x) or not np.isfinite(x).all():
            raise ValueError('Nonempty finite visible five-coordinate features required')
        leaves = self.model.apply(x)
        weight = np.zeros((len(x), len(self.y)))
        for tree, pool in enumerate(self.pools):
            for leaf in np.unique(leaves[:, tree]):
                queries = np.flatnonzero(leaves[:, tree] == leaf)
                rows, probabilities = pool[int(leaf)]
                weight[np.ix_(queries, rows)] += probabilities[None]/TREES
        if np.any(weight < 0) or not np.allclose(weight.sum(1), 1, atol=1e-12, rtol=0):
            raise ValueError('ET leaf weights lost normalization')
        mean = weight@self.y
        if np.max(abs(mean-self.model.predict(x)), initial=0) > 1e-12:
            raise ValueError('Weighted leaf mean differs from fixed ET prediction')
        covariance = np.empty((len(x), 3, 3)); variance = np.empty(len(x))
        for start in range(0, len(x), 512):
            take = slice(start, start+512)
            difference = self.y[None]-mean[take, None]
            tangent = difference@HELMERT
            covariance[take] = np.einsum('qn,qni,qnj->qij', weight[take], tangent, tangent)
            variance[take] = np.einsum('qn,qni,qni->q', weight[take], difference, difference)
        return dict(weights=weight, leaf_ids=leaves, mean=mean, covariance3=covariance,
                    scale=np.sqrt(np.maximum(variance, 0)+SCALE_FLOOR_SQUARED))


def _fit_leaf(history: dict[str, Any], mask: np.ndarray, outer: str, held: str | None) -> LeafLaw:
    model, metadata = coarse._fit('ET', history, mask, outer, held)
    x, y = history['X'][mask].copy(), history['y'][mask].copy()
    weights = np.asarray(metadata['sample_weights']); leaves = model.apply(x)
    pools = []
    for tree in range(TREES):
        pool = {}
        for leaf in np.unique(leaves[:, tree]):
            rows = np.flatnonzero(leaves[:, tree] == leaf)
            probability = weights[rows]/weights[rows].sum()
            pool[int(leaf)] = (rows, probability)
        pools.append(pool)
    metadata.update(training_leaf_ids=leaves.tolist(), leaf_response_definition='complete four-coordinate POS ratios',
                    bootstrap=False, leaf_weight_definition='within-tree leaf sample_weight normalization, then tree average')
    return LeafLaw(model, x, y, weights, leaves, pools, metadata)


def _validate_stream(bank: Mapping[str, np.ndarray], baseline: Mapping[str, np.ndarray], draws: np.ndarray,
                     dates: pd.DatetimeIndex, background: Mapping[str, np.ndarray]) -> None:
    n = len(dates)
    if (dates.has_duplicates or not dates.equals(dates.normalize()) or draws.shape != (n, SAMPLES, 5)):
        raise ValueError('Distinct deployed UTC days and visible five-coordinate draws required')
    prior._simplex(draws, 5, 'visible MG draws')
    for key in ('donor_ids', 'fallback_ids'):
        ids = baseline[key]
        if (ids.shape != (n, SAMPLES) or not np.issubdtype(ids.dtype, np.integer)
                or np.any(ids < (-1 if key == 'fallback_ids' else 0)) or np.any(ids >= len(bank['days']))):
            raise ValueError('Shared integer sampler IDs must be in the fixed bank')
        if not np.array_equal(ids, background[key]):
            raise ValueError('Exact V28 shared sampler registry required')
    if (background['arms'].tolist() != list(coarse.ARMS) or not np.array_equal(background['origins'], dates.asi8)
            or baseline['raw'].shape != (n, SAMPLES, 24) or not np.isfinite(baseline['raw']).all()
            or baseline['fine_draws'].shape != (n, SAMPLES, 8)):
        raise ValueError('Full inherited V28 grid and finite MG stream required')
    prior._simplex(baseline['fine_draws'], 8, 'inherited MG composition')
    for field in ('raw', 'fine_draws'):
        if not np.array_equal(background[field][0], baseline[field]):
            raise ValueError('Exact original MG background required')


@dataclass
class LeafContext:
    bank: Mapping[str, np.ndarray]
    outer: str
    table: dict[str, Any]
    history: dict[str, Any]
    final: LeafLaw
    folds: dict[str, LeafLaw]
    oof_mean: np.ndarray
    oof_scale: np.ndarray
    oof_residual: np.ndarray
    oof_covariance3: np.ndarray
    oof_leaf_ids: np.ndarray
    oof_weights: np.ndarray
    audit: dict[str, Any]

    def predict(self, baseline: Mapping[str, np.ndarray], mg_draws: np.ndarray, days: pd.DatetimeIndex,
                background_v28: Mapping[str, np.ndarray]) -> dict[str, Any]:
        dates = pd.DatetimeIndex(pd.to_datetime(days, utc=True))
        if dates.empty or dates.hasnans or np.any(dates < CUTOFF):
            raise ValueError('Query origins must be on or after fixed deployment cutoff 2025-01-01')
        draws = np.asarray(mg_draws, float)
        _validate_stream(self.bank, baseline, draws, dates, background_v28)
        n = len(dates); query = self.final.conditional(draws.reshape(-1, 5))
        query = {k: v.reshape((n, SAMPLES)+v.shape[1:]) for k, v in query.items()}
        point = np.asarray(background_v28['L24_ET_query_raw'])
        if (point.shape != query['mean'].shape or not np.isfinite(point).all()
                or np.max(abs(point-query['mean']), initial=0) > 1e-12):
            raise ValueError('Weighted leaf mean must reproduce V28 L24 ET query prediction within 1e-12')
        arms = {name: {k: background_v28[k][coarse.ARMS.index(name)].copy() for k in FIELDS} for name in ARMS[:3]}
        outputs = {a: {k: [] for k in FIELDS} for a in ARMS[3:]}
        uniforms, normals, permutations, samples, effective, months = [], [], [], [], [], []
        paired_scales, scale_ratios, shuffled_ids, shuffled_effective = [], [], [], []
        bank_ef = self.table['ratios'][~self.table['zero_pos_mask']]@EF[POS]
        lo, hi = float(bank_ef.min()), float(bank_ef.max())
        for t, day in enumerate(dates):
            ids = baseline['donor_ids'][t]
            q, support, selected, original_ratio = prior._effective(self.bank, ids, baseline['fallback_ids'][t], draws[t, :, 0])
            mapping = self.table['bank_month_ids'][selected]; effective.append(selected); months.append(mapping)
            original = np.zeros((SAMPLES, 8)); original[:, ZERO] = draws[t, :, 1:]; original[:, POS] = draws[t, :, :1]*original_ratio
            if not np.array_equal(original, baseline['fine_draws'][t]):
                raise ValueError('Exact shared MG effective ratio and visible draw identity required')
            if not np.array_equal(recompose_paths(original, q), baseline['raw'][t]):
                raise ValueError('Exact original MG raw recomposition required')
            u = _rng(day, STREAM_SALTS['leaf']).random(SAMPLES)
            z = _rng(day, STREAM_SALTS['gaussian']).standard_normal((SAMPLES, 3))
            permutation = _rng(day, STREAM_SALTS['shuffle']).permutation(SAMPLES)
            sampled = sample_rows(query['weights'][t], u)
            uniforms.append(u); normals.append(z); permutations.append(permutation); samples.append(sampled)
            source_scale = self.oof_scale[mapping]
            multiplier = query['scale'][t]/source_scale
            paired_scales.append(source_scale); scale_ratios.append(multiplier)
            shuffled_ids.append(ids[permutation]); shuffled_effective.append(selected[permutation])
            before = dict(Leaf=self.final.y[sampled].copy(),
                Gaussian=gaussian_ratios(query['mean'][t], query['covariance3'][t], z),
                Scale=query['mean'][t]+multiplier[:, None]*self.oof_residual[mapping])
            ratios = {name: value.copy() if name == 'Leaf' else prior.project(value) for name, value in before.items()}
            for arm in ARMS[3:]:
                family = arm.split('-')[0]; ratio = ratios[family]
                fine = original.copy(); fine[:, POS] = draws[t, :, :1]*ratio
                used_q, used_support = (q[permutation], support[permutation]) if arm == ARMS[-1] else (q, support)
                physical, zero = recompose_paths(fine, used_q, return_zero_total=True)
                raw = (self.bank['raw_cif'][ids]+((fine-self.bank['donor_mix'][ids])@EF)[:, None]
                       if arm.endswith('-anchor') else physical)
                value = ratio@EF[POS]
                fields = dict(raw=raw, fine_draws=fine, ratios=ratio, unprojected_ratios=before[family],
                    projection_l1=abs(ratio-before[family]).sum(1),
                    unsupported_positive_energy_fraction=prior._unsupported(fine, used_q, used_support),
                    source_EF_outside=(value < lo-prior.SUPPORT_ATOL) | (value > hi+prior.SUPPORT_ATOL), zero_total_mask=zero)
                for key, value in fields.items():
                    if not np.isfinite(value).all():
                        raise ValueError('Nonfinite conditional law path or diagnostic')
                    outputs[arm][key].append(value)
        arms.update({a: {k: np.stack(v) for k, v in fields.items()} for a, fields in outputs.items()})
        return dict(arms=arms, donor_ids=baseline['donor_ids'].copy(), fallback_ids=baseline['fallback_ids'].copy(),
            effective_pos_ids=np.stack(effective), effective_month_ids=np.stack(months),
            uniforms=np.stack(uniforms), normal3=np.stack(normals), permutation=np.stack(permutations),
            shuffled_original_ids=np.stack(shuffled_ids), shuffled_effective_pos_ids=np.stack(shuffled_effective),
            source_scale=np.stack(paired_scales), scale_ratio=np.stack(scale_ratios),
            query_floor_fraction=SCALE_FLOOR_SQUARED/np.square(query['scale']),
            source_floor_fraction=SCALE_FLOOR_SQUARED/np.square(np.stack(paired_scales)),
            query_near_floor=np.square(query['scale']) <= 2*SCALE_FLOOR_SQUARED,
            source_near_floor=np.square(np.stack(paired_scales)) <= 2*SCALE_FLOOR_SQUARED,
            leaf_sample_ids=np.stack(samples), query_weights=query['weights'], query_leaf_ids=query['leaf_ids'],
            query_mean=query['mean'], query_covariance3=query['covariance3'], query_scale=query['scale'],
            training_X=self.final.X.copy(), training_ratios=self.final.y.copy(), training_weights=self.final.weights.copy(),
            training_leaf_ids=self.final.leaf_ids.copy(), training_keys=np.asarray(self.final.metadata['training_keys']),
            bank_oof_mean=self.oof_mean.copy(), bank_oof_scale=self.oof_scale.copy(), bank_oof_residuals=self.oof_residual.copy(),
            bank_oof_near_floor=np.square(self.oof_scale) <= 2*SCALE_FLOOR_SQUARED,
            bank_oof_covariance3=self.oof_covariance3.copy(), bank_oof_leaf_ids=self.oof_leaf_ids.copy(),
            bank_oof_weights=self.oof_weights.copy(), source_month_X=self.table['X'].copy(),
            source_month_ratios=self.table['ratios'].copy(), source_month_regions=self.table['regions'].copy(),
            source_month_numbers=self.table['months'].copy(), source_month_zero_pos=self.table['zero_pos_mask'].copy(),
            source_bank_month_ids=self.table['bank_month_ids'].copy(),
            audit=dict(self.audit, original_mg_raw_preserved_exact=True, original_mg_fine_preserved_exact=True,
                background_exact=True, query_v28_mean_max_abs=float(abs(point-query['mean']).max()),
                source_EF_support=[lo, hi], source_EF_support_atol=prior.SUPPORT_ATOL,
                source_EF_support_definition='fixed H1 bank scalar ratio dot EF_POS, not hull or CIF path support',
                no_file_io=True, new_fine_parsed=False, annual_fields_used=False, target_values_opened=False,
                unseen_values_opened=False, custom_method=False, future_realized_energy_conserved=False,
                shuffle_definition='per-day complete effective q/support permutation; fine unchanged, CIF marginals may change',
                probability_control_interpretation='ordinary conditional distribution controls; centers and dependence may change'))


def fit_leaf_controls(bank: Mapping[str, np.ndarray], raw_month_rows: Sequence[Mapping[str, str]], *, outer: str) -> LeafContext:
    """Seven fixed ETs per outer; every teacher excluded from all coarse-history years."""
    table = prior.region_months(bank, outer=outer)
    history = coarse.select_coarse_rows(raw_month_rows, outer=outer, window='L24')
    positive, labels = history['positive_mask'], history['regions']
    if len(set(labels[positive])) < 2 or not np.any(~table['zero_pos_mask']):
        raise ValueError('Positive source supervision and bank responses required for geographic OOF')
    final = _fit_leaf(history, positive, outer, None)
    count = len(table['keys']); oof_mean = np.zeros((count, 4)); oof_scale = np.empty(count)
    oof_covariance = np.empty((count, 3, 3)); oof_leaves = np.empty((count, TREES), np.int64)
    oof_weights = np.zeros((count, len(final.y))); folds = {}; metas = []
    global_indices = np.flatnonzero(positive)
    for teacher in [r for r in SOURCES if r != outer]:
        train = positive & (labels != teacher)
        fold = _fit_leaf(history, train, outer, teacher); folds[teacher] = fold
        held = table['regions'] == teacher
        metadata = dict(fold.metadata)
        if held.any():
            conditional = fold.conditional(table['X'][held])
            oof_mean[held], oof_scale[held] = conditional['mean'], conditional['scale']
            oof_covariance[held], oof_leaves[held] = conditional['covariance3'], conditional['leaf_ids']
            columns = np.searchsorted(global_indices, np.flatnonzero(train))
            oof_weights[np.ix_(np.flatnonzero(held), columns)] = conditional['weights']
            metadata.update(bank_response_keys=[k for k, keep in zip(table['keys'], held) if keep],
                bank_response_X=table['X'][held].tolist(), bank_prediction_raw=conditional['mean'].tolist(),
                bank_scale=conditional['scale'].tolist())
        else:
            metadata['no_bank_response'] = True
        metas.append(metadata)
    residual = np.zeros_like(oof_mean); positive_bank = ~table['zero_pos_mask']
    residual[positive_bank] = table['ratios'][positive_bank]-oof_mean[positive_bank]
    audit = dict(arms=list(ARMS), outer=outer, window='L24', deployment_cutoff=CUTOFF.isoformat(),
        history=history['audit'], history_keys=history['keys'], history_zero_pos_mask=(~positive).tolist(),
        fits=dict(final=final.metadata, folds=metas), fitted_head_models=1+len(folds),
        fixed_residual_region_months=count, source_bank_keys=table['keys'], source_bank_row_counts=table['row_counts'].tolist(),
        scale_floor_squared=SCALE_FLOOR_SQUARED, helmert_basis=HELMERT.tolist(), stream_salts=STREAM_SALTS.copy(),
        near_floor_definition='conditional scale squared <= 2e-12; floor fraction is1e-12 / scale squared',
        rng_definition='default_rng(SeedSequence([17023,salt,UTC-day-ns-low32,UTC-day-ns-high32]))',
        seed=SEED, samples=SAMPLES, trees=TREES, source_geographic_oof=True, chronological_bank_forecast=False,
        head_and_bank_label_definitions_differ=True, residual_pool_expanded=False, q_bank_expanded=False,
        supervised_label_definition='released source full-calendar proxy monthly composition',
        residual_label_definition='proper bank observed-support month ratio minus strictly source-excluded conditional mean',
        projection_order='empirical Leaf rows unchanged; Gaussian and Scale raw ratios projected once at final query',
        covariance_definition='total-weight centered covariance in fixed orthonormal Helmert3 tangent space; symmetric PSD square root V sqrt(Lambda) V.T; scale uses trace full4 covariance plus1e-12',
        training_features=list(prior.FEATURES), no_query_hidden_labels=True)
    return LeafContext(bank, outer, table, history, final, folds, oof_mean, oof_scale, residual,
                       oof_covariance, oof_leaves, oof_weights, audit)
