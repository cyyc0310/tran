"""Ordinary source-prior monthly transport; pure arrays, no file or fine-data I/O.

MG observes a five-coordinate monthly simplex, not a merged M8 forecast. These
samplers do not enforce future realized monthly energy or establish novelty.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from transcif.research.fuel_transport_v7 import FuelTrajectoryBank, recompose_paths

SOURCES = ('BPAT', 'CISO', 'ERCO', 'FPL', 'ISNE', 'MISO', 'NYIS')
FUELS = ('coal', 'gas', 'oil', 'nuclear', 'hydro', 'wind', 'solar', 'other')
M8_COLUMNS = tuple('fuelshare_'+f for f in FUELS)
MG_COLUMNS = ('fuelshare_POS', 'fuelshare_nuclear', 'fuelshare_hydro',
              'fuelshare_wind', 'fuelshare_solar')
POS = np.array([0, 1, 2, 7])
ZERO = np.array([3, 4, 5, 6])
EF = np.array([1000., 400., 800., 0., 0., 0., 0., 400.])
START = pd.Timestamp('2024-01-25', tz='UTC')
CUTOFF = pd.Timestamp('2024-07-01', tz='UTC')
ARMS = tuple(f'{b}-{a}' for b in ('M8', 'MG') for a in
             ('anchor_conditional', 'recompose_global', 'recompose_conditional',
              'conditional_shuffled')) + ('MG-conditional_plugin_split',)


def columns(budget: str) -> tuple[str, ...]:
    if budget not in ('M8', 'MG'):
        raise ValueError('Only M8/MG monthly budgets are registered')
    return M8_COLUMNS if budget == 'M8' else MG_COLUMNS


def _simplex_rows(values: np.ndarray, active: np.ndarray) -> np.ndarray:
    x = np.asarray(values, float)
    if not active.any():
        active = np.ones(x.shape[1], bool)
    y = x[:, active]
    ordered = np.sort(y, axis=1)[:, ::-1]
    candidates = (np.cumsum(ordered, axis=1)-1)/np.arange(1, y.shape[1]+1)
    rho = np.sum(ordered > candidates, axis=1)-1
    threshold = candidates[np.arange(len(y)), rho]
    result = np.zeros_like(x)
    result[:, active] = np.maximum(y-threshold[:, None], 0)
    return result/result.sum(axis=1, keepdims=True)


def _simplex(value: np.ndarray, active: np.ndarray) -> np.ndarray:
    # Scalar arithmetic order deliberately follows the frozen V9 implementation.
    if not active.any():
        active = np.ones(len(value), bool)
    y = value[active]
    ordered = np.sort(y)[::-1]
    rho = np.flatnonzero(ordered-(np.cumsum(ordered)-1)/np.arange(1, len(y)+1) > 0)[-1]
    threshold = (ordered[:rho+1].sum()-1)/(rho+1)
    result = np.zeros(len(value))
    result[active] = np.maximum(y-threshold, 0)
    return result/result.sum()


def _valid_simplex(x: np.ndarray, dimension: int) -> None:
    if (x.shape[-1] != dimension or not np.isfinite(x).all() or np.any(x < 0)
            or not np.allclose(x.sum(-1), 1., atol=1e-12, rtol=0)):
        raise ValueError('Finite nonnegative visible simplex required')


def group_monthly(frame: pd.DataFrame, budget: str) -> pd.DataFrame:
    """Publish visible fields from caller-filtered rows, dropping hidden POS fields."""
    target = columns(budget)
    metadata = ['region', 'period_start', 'period_end', 'available_at']
    if not set(frame.region) <= set(SOURCES):
        raise ValueError('Only registered source monthly rows may be published')
    x = frame[list(M8_COLUMNS)].to_numpy(float)
    _valid_simplex(x, 8)
    result = frame[metadata].copy()
    if budget == 'M8':
        result[list(target)] = x
    else:
        result[list(target)] = np.column_stack([x[:, POS].sum(1), x[:, ZERO]])
    return result


def _ordinal(t: pd.Timestamp) -> int:
    return t.year*12+t.month-1


def _point(history: pd.DataFrame, at: pd.Timestamp, cols: tuple[str, ...],
           prior: np.ndarray) -> np.ndarray:
    if len(history) < 12:
        if history.empty:
            return prior.copy()
        same = history.loc[history.period_start.dt.month.eq(at.month)]
        return (same if len(same) else history)[list(cols)].iloc[-1].to_numpy(float)
    rows = history.tail(60)
    k = len(cols)
    basis = np.zeros((k, k-1))
    for j in range(1, k):
        basis[:j, j-1] = 1/np.sqrt(j*(j+1))
        basis[j, j-1] = -j/np.sqrt(j*(j+1))
    months = np.array([_ordinal(t) for t in rows.period_start], float)
    month = rows.period_start.dt.month.to_numpy(float)
    phase = 2*np.pi*(month-1)/12
    design = np.column_stack([np.ones(len(rows)), (months-months[-1])/12,
                              np.sin(phase), np.cos(phase), np.sin(2*phase), np.cos(2*phase)])
    response = (rows[list(cols)].to_numpy(float)-1/k) @ basis
    penalty = np.diag([0, 6, 2, 2, 4, 4])
    coefficient = np.linalg.solve(design.T@design+penalty, design.T@response)
    phase = 2*np.pi*(at.month-1)/12
    query = np.array([1, (_ordinal(at)-months[-1])/12, np.sin(phase), np.cos(phase),
                      np.sin(2*phase), np.cos(2*phase)])
    active = history[list(cols)].to_numpy(float).max(axis=0) > 0
    return _simplex(1/k+(query@coefficient)@basis.T, active)


def coarse_law(monthly: pd.DataFrame, region: str, days: pd.DatetimeIndex, *,
               budget: str, samples: int = 128, seed: int = 17023) -> dict[str, Any]:
    """Generic visible-space V14 harmonic/prequential joint-residual law."""
    cols = columns(budget)
    if (region not in SOURCES or not isinstance(samples, (int, np.integer)) or samples < 1
            or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError('Registered source, positive draw count and nonnegative seed required')
    if budget == 'MG' and any(c in monthly for c in (M8_COLUMNS[i] for i in POS)):
        raise ValueError('MG law accepts only grouped visible fields')
    dates = pd.DatetimeIndex(pd.to_datetime(days, utc=True))
    if dates.empty or dates.hasnans:
        raise ValueError('Finite nonempty query dates required')
    part = monthly.loc[monthly.region.eq(region), ['region', 'period_start', 'period_end',
                                                 'available_at', *cols]].copy()
    for c in ('period_start', 'period_end', 'available_at'):
        part[c] = pd.to_datetime(part[c], utc=True)
        if part[c].isna().any():
            raise ValueError('Finite monthly timestamps required')
    starts = part.period_start
    if (not starts.eq(starts.dt.normalize()).all() or not starts.dt.day.eq(1).all()
            or not part.period_end.eq(starts+pd.offsets.MonthBegin(1)).all()
            or not part.available_at.ge(part.period_end).all()):
        raise ValueError('Calendar monthly rows with valid release times required')
    part = part.loc[part.period_end.le(dates.max()) & part.available_at.le(dates.max())]
    part = part.sort_values(['period_end', 'available_at']).reset_index(drop=True)
    if part.period_start.duplicated().any():
        raise ValueError('One frozen vintage per region-month required')
    _valid_simplex(part[list(cols)].to_numpy(float), len(cols))
    prior = np.full(8, 1/8) if budget == 'M8' else np.array([.5, .125, .125, .125, .125])
    cache: dict[tuple, np.ndarray] = {}

    def point_at(history: pd.DataFrame, at: pd.Timestamp) -> np.ndarray:
        key = (tuple(history.index), at.year, at.month)
        if key not in cache:
            cache[key] = _point(history, at, cols, prior)
        return cache[key]

    points, particles, meta = [], [], []
    for origin in dates:
        prefix = part.loc[part.period_end.le(origin) & part.available_at.le(origin)]
        records = []
        for _, label in prefix.tail(60).iterrows():
            pseudo = label.period_start + pd.Timedelta(days=min(origin.day, (label.period_end-label.period_start).days)-1)
            history = prefix.loc[prefix.period_end.le(pseudo) & prefix.available_at.le(pseudo)]
            horizon = _ordinal(pseudo)-_ordinal(history.period_start.iloc[-1]) if len(history) else 99
            records.append((label[list(cols)].to_numpy(float)-point_at(history, pseudo), horizon))
        records = records[-24:]
        point = point_at(prefix, origin)
        horizon = _ordinal(origin)-_ordinal(prefix.period_start.iloc[-1]) if len(prefix) else 99
        if records:
            age = np.arange(len(records)-1, -1, -1, dtype=float)
            gaps = np.array([abs(h-horizon) for _, h in records], float)
            weight = np.exp(-age/12-gaps/1.5); weight /= weight.sum()
            key = [seed & 0xffffffff, (seed >> 32) & 0xffffffff, origin.year, origin.month, origin.day]
            rng = np.random.default_rng(np.random.SeedSequence(key))
            choice = rng.choice(len(records), size=samples, p=weight)
            residual = np.stack([r for r, _ in records])
            active = prefix[list(cols)].to_numpy(float).max(axis=0) > 0
            draw = _simplex_rows(point[None, :]+residual[choice], active)
        else:
            draw = np.repeat(point[None, :], samples, axis=0)
        age = float((origin-prefix.available_at.max())/pd.Timedelta(days=1)) if len(prefix) else 999.
        points.append(point); particles.append(draw); meta.append([len(prefix), len(records), horizon, age])
    return dict(mix=np.stack(points), draws=np.stack(particles), metadata=np.asarray(meta, float),
                audit=dict(budget=budget, input_dimension=len(cols), region=region,
                           released_prefix_only=True, joint_prequential_residuals=True,
                           current_month_truth_used=False))


def union_bank(first: dict[str, np.ndarray], second: dict[str, np.ndarray],
               query: dict[str, np.ndarray], *, outer: str) -> dict[str, Any]:
    """Two pair-bank aliases make a six-source view; duplicate values must be exact."""
    if outer not in SOURCES:
        raise ValueError('Registered outer source required')
    fields = ('q', 'raw_cif', 'donor_mix', 'monthmean', 'support', 'months')
    rows: dict[tuple[str, int], tuple[int, int]] = {}
    banks = (first, second)
    exclusions = []
    for b, bank in enumerate(banks):
        excluded = tuple(bank['excluded'].tolist()); exclusions.append(excluded)
        expected = tuple(r for r in SOURCES if r not in excluded)
        n = len(bank['days'])
        if (len(excluded) != 2 or outer not in excluded or len(set(excluded)) != 2
                or not set(excluded) <= set(SOURCES) or tuple(bank['source_regions']) != expected
                or int(bank['cutoff']) != CUTOFF.value or not n
                or not np.array_equal(bank['row_ids'], np.arange(n))):
            raise ValueError('Pair-excluded fixed H1 bank provenance required')
        shapes = dict(q=(n, 24, 8), raw_cif=(n, 24), donor_mix=(n, 8), monthmean=(n, 8),
                      support=(n, 8), months=(n,), regions=(n,), days=(n,))
        if any(bank[k].shape != shape for k, shape in shapes.items()):
            raise ValueError('Bank field shapes differ')
        if bank['support'].dtype != np.bool_ or set(bank['regions']) != set(expected):
            raise ValueError('Bank support/source registry invalid')
        _valid_simplex(bank['donor_mix'], 8)
        if any(not np.isfinite(bank[k]).all() or np.any(bank[k] < 0)
               for k in ('q', 'raw_cif', 'monthmean')):
            raise ValueError('Finite nonnegative donor fields required')
        if (np.any(bank['raw_cif'] > 1000) or np.any(bank['monthmean'].sum(1) <= 0)
                or not np.array_equal(bank['support'], bank['monthmean'] > 0)
                or not np.allclose(bank['donor_mix'], bank['monthmean']/bank['monthmean'].sum(1)[:, None],
                                   atol=1e-12, rtol=0)
                or np.any(bank['q'][~np.broadcast_to(bank['support'][:, None], bank['q'].shape)] != 1)):
            raise ValueError('Bank support or monthly normalization inconsistent')
        dates = pd.DatetimeIndex(pd.to_datetime(bank['days'], utc=True))
        if (not dates.equals(dates.normalize()) or np.any(dates < START) or np.any(dates >= CUTOFF)
                or not np.array_equal(bank['months'], dates.month)):
            raise ValueError('Only fixed-H1 complete donor dates allowed')
        seen = set()
        for i, key in enumerate(zip(bank['regions'].tolist(), bank['days'].tolist())):
            if key in seen:
                raise ValueError('Duplicate row inside pair bank')
            seen.add(key)
            if key in rows:
                prior_b, prior_i = rows[key]
                if any(not np.array_equal(bank[k][i], banks[prior_b][k][prior_i]) for k in fields):
                    raise ValueError('Overlapping pair-bank values must be exact')
            else:
                rows[key] = (b, i)
    if exclusions[0] == exclusions[1]:
        raise ValueError('Two different teacher exclusions required')
    keys = list(zip(query['bank_regions'].tolist(), query['bank_days'].tolist()))
    if (len(keys) != len(set(keys)) or set(keys) != set(rows)
            or set(query['bank_regions']) != set(SOURCES)-{outer}
            or not np.array_equal(query['bank_row_ids'], np.arange(len(keys)))
            or int(query['bank_cutoff']) != CUTOFF.value):
        raise ValueError('Exact six-source query row provenance required')
    mapping = np.array([rows[k] for k in keys], dtype=np.int64)
    result = {k: np.stack([banks[b][k][i] for b, i in mapping]) for k in fields}
    result['q'] = np.ascontiguousarray(result['q'].transpose(0, 2, 1)).transpose(0, 2, 1)
    result.update(regions=query['bank_regions'].copy(), days=query['bank_days'].copy(),
                  row_ids=query['bank_row_ids'].copy(), row_mapping=mapping,
                  source_regions=np.asarray([r for r in SOURCES if r != outer]),
                  cutoff=np.array(CUTOFF.value), outer=outer)
    return result


def donor_weights(bank: dict[str, Any], mix: np.ndarray, day: pd.Timestamp, *,
                  budget: str, conditional: bool) -> np.ndarray:
    """Equal-region seasonal prior; conditional similarity uses visible fields only."""
    cols = columns(budget)
    _valid_simplex(np.asarray(mix), len(cols))
    regions = bank['regions']
    prior = np.array([1/(len(set(regions))*np.count_nonzero(regions == r)) for r in regions])
    difference = np.abs(day.month-bank['months'])
    circular = (np.minimum(difference, 12-difference)/6.)**2
    log_kernel = -circular/.5**2
    if conditional:
        donor = bank['donor_mix']
        visible = donor if budget == 'M8' else np.column_stack([donor[:, POS].sum(1), donor[:, ZERO]])
        hellinger = .5*np.sum((np.sqrt(mix)-np.sqrt(visible))**2, axis=1)
        log_kernel -= hellinger/.3**2
    shifted = log_kernel-np.max(log_kernel)
    weight = prior*np.exp(shifted)
    return weight/weight.sum()


def _sample(weight: np.ndarray, uniform: np.ndarray) -> np.ndarray:
    positive = np.flatnonzero(weight > 0)
    if not len(positive):
        raise ValueError('No positive source-prior mass')
    cdf = np.cumsum(weight)
    cdf[positive[-1]:] = 1.
    indices = np.searchsorted(cdf, uniform, side='right')
    if np.any(weight[indices] <= 0):
        raise ValueError('CDF selected a zero-mass donor')
    return indices


def _fine_draws(bank: dict[str, Any], draw: np.ndarray, ids: np.ndarray, weight: np.ndarray,
                uniforms: np.ndarray, budget: str, plugin: bool = False
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = bank['q'][ids].copy(order='K')
    fallback = np.full(len(ids), -1, dtype=np.int64)
    if budget == 'M8':
        return draw.copy(), q, fallback
    positive_mass = bank['donor_mix'][:, POS].sum(1)
    supported = positive_mass > 0
    supported_weight = np.where(supported, weight, 0.)
    needed = draw[:, 0] > 0
    if supported_weight.sum() <= 0:
        if needed.any():
            raise ValueError('MG prior has no positive-group-supported donor mass')
    else:
        supported_weight /= supported_weight.sum()
    ratio = np.zeros((len(weight), 4))
    ratio[supported] = bank['donor_mix'][supported][:, POS]/positive_mass[supported, None]
    chosen = ids.copy()
    missing = ~supported[ids] & needed
    if missing.any():
        borrowed = _sample(supported_weight, uniforms[:, 2])
        chosen[missing] = borrowed[missing]; fallback[missing] = borrowed[missing]
        for particle in np.flatnonzero(missing):
            # Own monthly POS mix is zero: its raw CIF is invariant to this q replacement.
            q[particle][:, POS] = bank['q'][chosen[particle]][:, POS]
    r = np.broadcast_to(supported_weight@ratio, (len(ids), 4)) if plugin else ratio[chosen]
    fine = np.zeros((len(ids), 8))
    fine[:, POS] = draw[:, :1]*r
    fine[:, ZERO] = draw[:, 1:]
    return fine, q, fallback


def predict(bank: dict[str, Any], laws: dict[str, dict[str, Any]], days: pd.DatetimeIndex, *,
            samples: int = 128, seed: int = 17023) -> dict[str, Any]:
    """Nine registered ordinary arms; shared uniforms and explicit joint pairing."""
    dates = pd.DatetimeIndex(pd.to_datetime(days, utc=True))
    if dates.empty or dates.hasnans or np.any(dates < CUTOFF) or samples != 128 or seed != 17023:
        raise ValueError('Nonempty deployed origins and fixed128/17023 experiment required')
    results: dict[str, dict[str, list]] = {a: {k: [] for k in
        ('raw', 'donor_ids', 'fallback_ids', 'permutation', 'fine_draws', 'sampled_q', 'weights')} for a in ARMS}
    for budget in ('M8', 'MG'):
        law = laws[budget]
        if law['mix'].shape != (len(dates), len(columns(budget))) or law['draws'].shape != (len(dates), samples, len(columns(budget))):
            raise ValueError('Visible coarse law shape mismatch')
        _valid_simplex(law['mix'], len(columns(budget))); _valid_simplex(law['draws'], len(columns(budget)))
        for t, day in enumerate(dates):
            uniforms = FuelTrajectoryBank._rng(seed, day).random((samples, 8))
            permutation = FuelTrajectoryBank._rng(seed ^ 0x53485546, day).permutation(samples)
            for conditional in (False, True):
                weight = donor_weights(bank, law['mix'][t], day, budget=budget, conditional=conditional)
                ids = _sample(weight, uniforms[:, 0])
                fine, q, fallback = _fine_draws(bank, law['draws'][t], ids, weight, uniforms, budget)
                variants = [('recompose_conditional' if conditional else 'recompose_global', fine, q, False, None)]
                if conditional:
                    variants += [('anchor_conditional', fine, q, True, None),
                                 ('conditional_shuffled', fine, q[permutation], False, permutation)]
                    if budget == 'MG':
                        pf, pq, pfb = _fine_draws(bank, law['draws'][t], ids, weight, uniforms, budget, True)
                        if not np.array_equal(pfb, fallback):
                            raise AssertionError('Plugin and sampled arms must share borrowed q')
                        variants.append(('conditional_plugin_split', pf, pq, False, None))
                for name, fd, sq, anchor, perm in variants:
                    raw = (bank['raw_cif'][ids]+((fd-bank['donor_mix'][ids])@EF)[:, None]
                           if anchor else recompose_paths(fd, sq))
                    if not np.isfinite(raw).all():
                        raise ValueError('Nonfinite transport paths')
                    arm = results[f'{budget}-{name}']
                    for key, value in dict(raw=raw, donor_ids=ids, fallback_ids=fallback,
                        permutation=np.arange(samples) if perm is None else perm,
                        fine_draws=fd, sampled_q=sq, weights=weight).items():
                        arm[key].append(value)
    arrays = {a: {k: np.stack(v) for k, v in fields.items()} for a, fields in results.items()}
    return dict(arms=arrays, laws=laws, audit=dict(arms=list(ARMS), samples=samples, seed=seed,
        no_hourly_file_io=True, future_month_energy_conserved=False,
        global_prior='equal-region times common calendar kernel',
        fallback_counts={a: int(np.count_nonzero(x['fallback_ids'] >= 0)) for a, x in arrays.items()},
        plugin_interpretation='changes split variability, dependence and possibly nonlinear center',
        shuffle_interpretation='preserves input multisets, not CIF marginals; removes full composition/q pairing'))
