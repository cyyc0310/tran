"""V7-style analog references under a six-category delayed public-input contract."""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd

from transcif.research.multicountry import FACTORS, FINE, GROUPS, coarse_values, weights

MODES = ('flat', 'mean_profile', 'joint', 'independent', 'direct_coarse', 'direct_cif')
RESPONSE_MODES = ('response_coupled', 'response_shuffled', 'transition_only')


def permuted_indices(indices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Change cross-category pairing while preserving every empirical marginal."""
    return np.column_stack([indices, *(rng.permutation(indices) for _ in range(len(GROUPS)-1))])


def thermal_fraction(mix: np.ndarray, paths: np.ndarray) -> np.ndarray:
    generation = mix * paths
    total = generation.sum(axis=-1)
    return np.divide(generation[..., 0], total, out=np.full(total.shape, mix[0]), where=total > 0)


def fit(frame: pd.DataFrame, fuel: np.ndarray, *, include_response: bool = False) -> dict:
    """Accept only source 2022 labels already eligible for the public-month models."""
    if len(frame) != len(fuel) or not frame.clock.dt.year.eq(2022).all():
        raise ValueError('aligned source-2022 rows required')
    if not np.isfinite(fuel).all() or (fuel < 0).any() or (fuel.sum(axis=1) <= 0).any():
        raise ValueError('invalid source generation')
    f = frame.reset_index(drop=True).copy()
    f['position'] = np.arange(len(f))
    f['day'] = f.clock.dt.floor('D')
    coarse = coarse_values(pd.DataFrame(fuel, columns=FINE))
    days, visible, countries, regions, raw, fine_days = [], [], [], [], [], []
    for (country, region, day), group in f.groupby(['country', 'region', 'day'], sort=True):
        group = group.sort_values('clock')
        if len(group) != 24 or group.clock.dt.hour.tolist() != list(range(24)):
            continue
        values = group[list(GROUPS)].to_numpy(float)
        if not np.allclose(values, values[0], atol=1e-12, rtol=0.):
            raise ValueError('public monthly shares change within a day')
        ix = group.position.to_numpy()
        days.append(day); visible.append(values[0]); countries.append(country); regions.append(region)
        raw.append(coarse[ix]); fine_days.append(fuel[ix])
    if not days or set(regions) != set(f.region):
        raise ValueError('every source region needs complete 24-hour source days')
    raw = np.asarray(raw); fine_days = np.asarray(fine_days)
    meta = pd.DataFrame({'country': countries, 'region': regions, 'day': days})
    meta['period'] = meta.day.dt.to_period('M')
    q, support = np.ones_like(raw), np.zeros((len(raw), len(GROUPS)), dtype=bool)
    transition = np.ones((len(raw), len(GROUPS)))
    newly_active = np.zeros_like(transition)
    normalization = []
    for (region, period), group in meta.groupby(['region', 'period']):
        ix = group.index.to_numpy(); mean = raw[ix].mean(axis=(0, 1)); active = mean > 0
        q[ix] = np.divide(raw[ix], mean, out=np.ones_like(raw[ix]), where=active)
        support[ix] = active
        current_mix = mean / mean.sum()
        old_mix = np.asarray(visible)[ix]
        transition[ix] = np.divide(current_mix, old_mix, out=np.ones_like(old_mix), where=old_mix > 0)
        newly_active[ix] = (old_mix <= 0) * current_mix
        normalization.append({'region': region, 'period': str(period), 'days': len(ix),
                              'mean_power': mean.tolist(), 'supported': active.tolist()})
    regional_ef = []
    for (country, region), group in meta.groupby(['country', 'region']):
        ix = group.index.to_numpy()
        denominator = raw[ix, :, 0].sum()
        if denominator <= 0:
            raise ValueError('effective thermal factor undefined in a source region')
        regional_ef.append({'country': country, 'ef': float((fine_days[ix] @ FACTORS).sum() / denominator)})
    macro_ef = float(pd.DataFrame(regional_ef).groupby('country').ef.mean().mean())
    bank = {'q': q, 'support': support, 'visible': np.asarray(visible), 'prior': weights(meta),
            'months': meta.day.dt.month.to_numpy(), 'raw_fraction': raw[:, :, 0] / raw.sum(axis=2),
            'raw_cif': (fine_days @ FACTORS) / fine_days.sum(axis=2), 'macro_ef': macro_ef,
            'audit': {'eligible_hours': len(frame), 'retained_hours': len(days) * 24,
                      'retained_days': len(days), 'countries': sorted(set(countries)),
                      'region_days': meta.groupby('region').size().to_dict(),
                      'normalization': normalization,
                      'missing_category_policy': 'q=1 prior; unsupported, not certified physical zero',
                      'target_monthly_energy_conserved': False}}
    if include_response:
        bank['transition'] = transition
        bank['audit']['newly_active_source_mass_mean'] = float(newly_active.sum(axis=1).mean())
        bank['audit']['response'] = 'current observed-support monthly share / source own lag2 share; zero lag -> ratio1 unsupported'
    return bank


def predict(bank: dict, public: pd.DataFrame, n_members: int = 128) -> tuple[dict, pd.DataFrame]:
    """No target hourly generation accepted. Values are thermal fractions except direct_cif."""
    if n_members < 1:
        raise ValueError('positive particle count required')
    if public.clock.dt.year.min() < 2023:
        raise ValueError('queries must follow source training year')
    f = public.reset_index(drop=True).copy()
    f['day'] = f.clock.dt.floor('D'); f['position'] = np.arange(len(f))
    modes = (*MODES, *RESPONSE_MODES) if 'transition' in bank else MODES
    out = {mode: np.empty((len(f), n_members)) for mode in modes}
    diagnostics = []
    for (region, day), group in f.groupby(['region', 'day'], sort=True):
        mix = group.iloc[0][list(GROUPS)].to_numpy(float)
        if not np.isfinite(mix).all() or (mix < 0).any() or not np.isclose(mix.sum(), 1.):
            raise ValueError('valid public composition required')
        if not np.allclose(group[list(GROUPS)], mix, atol=1e-12, rtol=0.):
            raise ValueError('public shares change within query day')
        distance = np.minimum(abs(bank['months'] - day.month), 12 - abs(bank['months'] - day.month)) / 6.
        hellinger = .5 * ((np.sqrt(bank['visible']) - np.sqrt(mix)) ** 2).sum(axis=1)
        log_kernel = -distance ** 2 / .5 ** 2 - hellinger / .3 ** 2
        w = bank['prior'] * np.exp(log_kernel - log_kernel.max()); w /= w.sum()
        cdf = np.cumsum(w); cdf[-1] = 1.
        key = hashlib.sha256(f'416|{region}|{day.isoformat()}'.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(key[:8], 'little'))
        uniforms = rng.random((n_members, len(GROUPS)))
        joint_ix = np.searchsorted(cdf, uniforms[:, 0], side='right')
        independent_ix = permuted_indices(joint_ix, rng)
        joint = thermal_fraction(mix, bank['q'][joint_ix])
        independent_q = bank['q'][independent_ix[:, None, :], np.arange(24)[None, :, None],
                                    np.arange(len(GROUPS))[None, None, :]]
        independent = thermal_fraction(mix, independent_q)
        mean = thermal_fraction(mix, np.einsum('d,dhf->hf', w, bank['q']))
        ix, hours = group.position.to_numpy(), group.clock.dt.hour.to_numpy()
        out['flat'][ix] = mix[0]
        out['mean_profile'][ix] = mean[hours, None]
        out['joint'][ix] = joint[:, hours].T
        out['independent'][ix] = independent[:, hours].T
        out['direct_coarse'][ix] = bank['raw_fraction'][joint_ix][:, hours].T
        out['direct_cif'][ix] = bank['raw_cif'][joint_ix][:, hours].T
        if 'transition' in bank:
            change = bank['transition'][joint_ix]
            shuffled = rng.permutation(change, axis=0)
            coupled_path = thermal_fraction(mix, bank['q'][joint_ix] * change[:, None, :])
            shuffled_path = thermal_fraction(mix, bank['q'][joint_ix] * shuffled[:, None, :])
            level = thermal_fraction(mix, change)
            out['response_coupled'][ix] = coupled_path[:, hours].T
            out['response_shuffled'][ix] = shuffled_path[:, hours].T
            out['transition_only'][ix] = level[None, :]
        diagnostics.append({'region': region, 'day': str(day.date()),
                            'effective_donors': float(1. / (w @ w)),
                            'unsupported_target_mass': float(mix @ (1. - w @ bank['support'])),
                            'same_month_mass': float(w @ (bank['months'] == day.month))})
    return out, pd.DataFrame(diagnostics)


def members(values: dict, factor: float) -> dict:
    return {k: v if k == 'direct_cif' else factor * v for k, v in values.items()}


def crps(ensemble: np.ndarray, y: np.ndarray) -> np.ndarray:
    ordered = np.sort(ensemble, axis=1)
    n = ensemble.shape[1]
    coefficient = 2 * np.arange(1, n + 1) - n - 1
    return np.abs(ensemble - y[:, None]).mean(axis=1) - (ordered * coefficient).sum(axis=1) / n ** 2
