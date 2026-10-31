"""Source-only fuel trajectory analogs followed by the generation-ratio operator.

This combines existing analog sampling with fuel-wise trajectory recomposition;
it is a mechanism prototype, not a new statistical theory or a target coverage
guarantee. Forecast inputs are lawful fuel shares and calendar timestamps only.
The caller is responsible for release-time forecasting of those shares.

Normalization is by source region-month *observed-support* mean power, never a
daily mean. A donor's own observed monthly shares reconstruct its hourly CIF
exactly. Kernel reweighting and finite draws do not generally conserve target
monthly energy: weighted mean q need not equal one. January-June source support
also leaves July-December calendar extrapolation intrinsically weak.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from .dynamic_v2 import EF
from .eia_proxy import FUELS


GENERATION_COLUMNS = tuple(f"generation_{fuel}" for fuel in FUELS)
MODES = ("recompose_joint", "recompose_independent", "direct_analog", "pooled_recompose", "mean_profile")
BANDWIDTHS = (.1, .3, 1.)
FIT_START = pd.Timestamp("2024-01-01", tz="UTC")
FIT_CUTOFF = pd.Timestamp("2024-07-01", tz="UTC")
SEASON_BANDWIDTH = .5


def _mix_array(mix):
    mix = np.asarray(mix, dtype=float)
    if mix.ndim == 1:
        mix = mix[None, :]
    if (mix.ndim != 2 or mix.shape[1] != len(FUELS) or not len(mix)
            or not np.isfinite(mix).all() or np.any(mix < 0)
            or not np.allclose(mix.sum(axis=1), 1., atol=1e-8, rtol=1e-8)):
        raise ValueError("forecast_mix must contain finite nonnegative eight-fuel simplex rows.")
    return mix


def recompose_paths(mix, trajectories, *, return_zero_total=False):
    """Apply CIF=sum_f(EF_f*m_f*q_hf)/sum_f(m_f*q_hf).

    ``mix`` is (8,) with trajectories (...,24,8), or (N,8) with
    trajectories (N,...,24,8). Zero generation has undefined CIF; its explicit
    fixed fallback is m@EF, with a returned mask when requested. Unsupported
    fuel trajectories q=1 are supplied by the fitted bank, not inferred here.
    """
    original = np.asarray(mix, dtype=float)
    shares = _mix_array(original)
    q = np.asarray(trajectories, dtype=float)
    if q.ndim < 2 or q.shape[-2:] != (24, len(FUELS)) or not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError("trajectories must be finite nonnegative (...,24,8).")
    if original.ndim == 1:
        scaled_mix = shares[0]
        fallback = float(shares[0] @ EF)
    else:
        if q.ndim < 3 or q.shape[0] != len(shares):
            raise ValueError("Batched mixes and trajectories must have the same first dimension.")
        scaled_mix = shares.reshape((len(shares),) + (1,) * (q.ndim - 2) + (len(FUELS),))
        fallback = (shares @ EF).reshape((len(shares),) + (1,) * (q.ndim - 2))
    generation = q * scaled_mix
    total = generation.sum(axis=-1)
    zero = total <= 0
    proportions = np.divide(generation, total[..., None], out=np.zeros_like(generation), where=~zero[..., None])
    result = proportions @ EF
    result = np.where(zero, np.broadcast_to(fallback, total.shape), result)
    if not np.isfinite(result).all():
        raise ValueError("Recomposed CIF overflowed; check trajectory magnitudes.")
    result = np.clip(result, EF.min(), EF.max())
    return (result, zero) if return_zero_total else result


class FuelTrajectoryBank:
    """Store complete joint source days normalized by observed monthly power.

    ``fit`` accepts only declared source frames. It selects 2024H1 before
    inspecting generation values; rows dated 2025 or later are rejected outright
    so they cannot accidentally become training labels. A supplied ``monthly``
    table is optional provenance metadata, restricted to source 2024H1 periods.
    Its shares are never used: donor true energy shares come from the same
    observed source hours as the normalization operator.

    The kernel has equal-region PRIOR 1/(R*n_region), multiplied by
    exp(-Hellinger_squared/bandwidth^2 - circular_month_distance/0.5^2),
    then normalized globally. Hellinger_squared = .5*sum((sqrt(m)-sqrt(p))^2);
    circular_month_distance = (min(|month-month_j|,12-|month-month_j|)/6)^2.
    Posterior region mass can therefore favor composition-relevant regions.
    ``pooled_recompose`` removes only the Hellinger term.
    """

    def __init__(self, source_regions=None):
        if source_regions is not None:
            source_regions = tuple(str(region) for region in source_regions)
            if (not source_regions or len(set(source_regions)) != len(source_regions)
                    or any(not region.strip() for region in source_regions)):
                raise ValueError("Declare nonempty unique source regions.")
        self.declared_source_regions = source_regions
        self.fitted = False

    def fit(self, frames, monthly=None, *, cutoff="2024-07-01"):
        self.fitted = False
        cutoff = pd.Timestamp(cutoff)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        if cutoff != FIT_CUTOFF:
            raise ValueError("This prototype freezes source fitting to 2024H1, cutoff 2024-07-01 UTC.")
        if not isinstance(frames, dict) or not frames or any(not isinstance(key, str) or not key.strip() for key in frames):
            raise ValueError("frames must be a nonempty source-region-to-DataFrame dictionary.")
        regions = tuple(sorted(frames))
        if self.declared_source_regions is not None and set(regions) != set(self.declared_source_regions):
            raise ValueError("frames must contain exactly the declared source regions.")
        monthly_rows = 0
        if monthly is not None:
            required = {"region", "period_start", "period_end"}
            if not required.issubset(monthly):
                raise ValueError("Optional monthly provenance needs region/period_start/period_end.")
            starts = pd.DatetimeIndex(pd.to_datetime(monthly.period_start, utc=True))
            ends = pd.DatetimeIndex(pd.to_datetime(monthly.period_end, utc=True))
            if (starts.hasnans or ends.hasnans or not set(monthly.region).issubset(regions)
                    or np.any(starts < FIT_START) or np.any(ends > cutoff) or np.any(starts >= ends)):
                raise ValueError("Optional monthly provenance may contain only declared-source 2024H1 periods.")
            monthly_rows = len(monthly)
        q_parts, raw_parts, mix_parts, mean_parts, support_parts, day_parts, region_parts = [], [], [], [], [], [], []
        month_audit, row_audit = [], {}
        for region in regions:
            frame = frames[region]
            required = {"time", "valid", *GENERATION_COLUMNS}
            if not required.issubset(frame):
                raise ValueError(f"Missing source columns for {region}: {sorted(required - set(frame))}")
            times = pd.DatetimeIndex(pd.to_datetime(frame.time, utc=True))
            if times.hasnans or np.any(times.year >= 2025):
                raise ValueError("Source frames must not contain missing times or 2025/later labels.")
            selected = (times >= FIT_START) & (times < cutoff)
            part = frame.loc[selected, ["valid", *GENERATION_COLUMNS]].copy()
            part["_time"] = times[selected]
            part = part.sort_values("_time").reset_index(drop=True)
            selected_times = pd.DatetimeIndex(part._time)
            if selected_times.has_duplicates or not selected_times.equals(selected_times.floor("h")):
                raise ValueError("Source fitting hours must be unique exact UTC hour starts.")
            validity = part.valid.to_numpy()
            if not np.isin(validity, [True, False, 0, 1]).all():
                raise ValueError("Source valid flags must be boolean or zero/one.")
            part["_valid"] = validity.astype(bool)
            part["_day"] = selected_times.floor("D")
            observed, donor_days = [], []
            for day, group in part.groupby("_day", sort=True):
                if len(group) != 24 or not group._valid.all():
                    continue
                values = group[list(GENERATION_COLUMNS)].to_numpy(dtype=float)
                totals = values.sum(axis=1)
                if (not np.isfinite(values).all() or np.any(values < 0)
                        or not np.isfinite(totals).all() or np.any(totals <= 0)):
                    raise ValueError("Valid source days need finite nonnegative generation and positive hourly total.")
                observed.append(values)
                donor_days.append(day)
            if not observed:
                raise ValueError(f"No complete valid 2024H1 source days in {region}.")
            generation = np.stack(observed)
            days = pd.DatetimeIndex(donor_days)
            row_audit[region] = {"supplied_rows": len(frame), "fitting_period_rows": int(selected.sum()),
                                 "retained_days": len(days), "ignored_rows_outside_fit": int((~selected).sum())}
            q, mixes, means = np.ones_like(generation), np.empty((len(days), 8)), np.empty((len(days), 8))
            support = np.empty((len(days), 8), dtype=bool)
            for month in sorted(set(days.month)):
                mask = days.month == month
                values = generation[mask]
                mean = values.mean(axis=(0, 1))
                if not np.isfinite(mean).all() or mean.sum() <= 0:
                    raise ValueError("Observed-support source monthly means overflowed.")
                supported = mean > 0
                q[mask] = np.divide(values, mean[None, None, :], out=np.ones_like(values),
                                    where=supported[None, None, :])
                means[mask], mixes[mask], support[mask] = mean, mean / mean.sum(), supported
                month_audit.append({"region": region, "month": f"2024-{month:02d}",
                    "observed_days": int(mask.sum()), "observed_hours": int(mask.sum() * 24),
                    "mean_generation": mean.tolist(), "supported_fuels": supported.tolist()})
            raw = (generation / generation.sum(axis=2, keepdims=True)) @ EF
            if not np.isfinite(q).all() or not np.isfinite(raw).all():
                raise ValueError("Source trajectories overflowed.")
            q_parts.append(q); raw_parts.append(raw); mix_parts.append(mixes); mean_parts.append(means)
            support_parts.append(support); day_parts.append(days); region_parts.append(np.repeat(region, len(days)))
        self.q_, self.raw_cif_, self.donor_mix_ = np.concatenate(q_parts), np.concatenate(raw_parts), np.concatenate(mix_parts)
        self.month_mean_generation_, self.fuel_support_ = np.concatenate(mean_parts), np.concatenate(support_parts)
        self.days_ = pd.DatetimeIndex(np.concatenate([days.to_numpy() for days in day_parts]))
        self.regions_ = np.concatenate(region_parts)
        self.months_ = self.days_.month.to_numpy()
        self.source_regions_, self.cutoff_ = regions, cutoff
        self.audit_ = {"source_regions": list(regions), "fit_start": str(FIT_START), "cutoff_exclusive": str(cutoff),
            "first_donor_day": str(self.days_.min()), "last_donor_day": str(self.days_.max()),
            "donor_days": len(self.days_), "region_rows": row_audit, "source_months": month_audit,
            "normalization": "generation / region-month observed-support mean power; never daily normalization",
            "missing_fuel_policy": "q=1 flat prior, support=False; no shape identified for zero-mean fuels",
            "monthly_provenance_rows": monthly_rows, "monthly_shares_used": False,
            "target_hourly_labels_used": False, "fuels": list(FUELS), "emission_factors": EF.tolist(),
            "calendar_support": sorted(set(self.months_.tolist())),
            "calendar_limit": "Only January-June source months; July-December phase extrapolation is weak.",
            "energy_limit": "Own observed monthly mix reconstructs donors; weighted or sampled target monthly energy is not constrained."}
        self.fitted = True
        return self

    def _weights(self, mix, origin, mode, bandwidth, eligible):
        available_regions = np.unique(self.regions_[eligible])
        prior = np.zeros(len(self.days_))
        for region in available_regions:
            mask = eligible & (self.regions_ == region)
            prior[mask] = 1 / (len(available_regions) * mask.sum())
        difference = np.abs(origin.month - self.months_)
        circular = (np.minimum(difference, 12 - difference) / 6.)**2
        hellinger = .5 * np.sum((np.sqrt(mix) - np.sqrt(self.donor_mix_))**2, axis=1)
        log_kernel = -circular / SEASON_BANDWIDTH**2
        if mode != "pooled_recompose":
            log_kernel -= hellinger / bandwidth**2
        weights = np.zeros(len(prior))
        shifted = log_kernel[eligible] - np.max(log_kernel[eligible])
        weights[eligible] = prior[eligible] * np.exp(shifted)
        weights /= weights.sum()
        return weights, prior, circular, hellinger

    @staticmethod
    def _rng(seed, origin):
        # Timestamp.value is UTC nanoseconds even when input index uses us/ms.
        stamp = int(origin.value)
        sequence = np.random.SeedSequence([int(seed) & 0xffffffff, int(seed) >> 32,
                                          stamp & 0xffffffff, (stamp >> 32) & 0xffffffff])
        return np.random.default_rng(sequence)

    def predict(self, forecast_mix, origins, mode="recompose_joint", n_scenarios=128,
                seed=17023, bandwidth=.3, exclude_region=None):
        """Return finite particle paths, their exact mean, and support diagnostics.

        Joint/direct modes share weights and inverse-CDF donor draws. Independent
        mode gives each fuel those same donor marginals with separate uniforms,
        changing the copula. All random modes use the same particle budget and
        point=ensemble.mean. mean_profile instead recomposes full-bank E[q] and
        repeats that plug-in path; ratio(E[q]) is not E[ratio(q)]. Randomness is
        keyed by UTC origin and seed, invariant to batch ordering/splitting.
        """
        if not self.fitted:
            raise RuntimeError("Fit FuelTrajectoryBank before prediction.")
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}; expected {MODES}.")
        mix = _mix_array(forecast_mix)
        origins = pd.DatetimeIndex(pd.to_datetime(origins, utc=True))
        if len(origins) != len(mix) or origins.hasnans or np.any(origins < self.cutoff_):
            raise ValueError("origins must match forecast_mix and follow the source fitting cutoff.")
        if not isinstance(n_scenarios, (int, np.integer)) or isinstance(n_scenarios, bool) or n_scenarios < 1:
            raise ValueError("n_scenarios must be a positive integer.")
        if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or not 0 <= seed < 2**64:
            raise ValueError("seed must be a nonnegative integer less than 2**64.")
        if not np.isfinite(bandwidth) or not 1e-6 <= bandwidth <= 1e6:
            raise ValueError("bandwidth must be finite and lie in [1e-6,1e6].")
        if exclude_region is not None and not isinstance(exclude_region, str):
            raise ValueError("exclude_region must be a source region string or None.")
        eligible = np.ones(len(self.days_), dtype=bool) if exclude_region is None else self.regions_ != exclude_region
        if not eligible.any():
            raise ValueError("Region exclusion leaves no source donors.")
        n, d = len(mix), len(self.days_)
        ensemble = np.empty((n, n_scenarios, 24))
        weights_all, prior_all = np.empty((n, d)), np.empty((n, d))
        support = np.empty((n, 8))
        sampled_mass = np.empty((n, n_scenarios))
        indices_all = np.empty((n, n_scenarios, 8), dtype=int)
        zero_fraction, season_distance, mix_distance, same_month_mass = (np.empty(n) for _ in range(4))
        for i, origin in enumerate(origins):
            weights, prior, circular, hellinger = self._weights(mix[i], origin, mode, bandwidth, eligible)
            weights_all[i], prior_all[i] = weights, prior
            support[i] = np.clip(weights @ self.fuel_support_, 0., 1.)
            season_distance[i], mix_distance[i] = weights @ circular, weights @ hellinger
            same_month_mass[i] = weights @ (self.months_ == origin.month)
            zero_fraction[i] = 0.
            if mode == "mean_profile":
                profile = np.einsum("d,dhf->hf", weights, self.q_)
                path, zero = recompose_paths(mix[i], profile, return_zero_total=True)
                ensemble[i] = path
                indices_all[i] = -1
                sampled_mass[i] = (1 - support[i]) @ mix[i]
                zero_fraction[i] = zero.mean()
                continue
            uniforms = self._rng(seed, origin).random((n_scenarios, 8))
            cdf = np.cumsum(weights)
            cdf[-1] = 1.
            if mode == "recompose_independent":
                indices = np.searchsorted(cdf, uniforms, side="right")
                q = self.q_[indices[:, None, :], np.arange(24)[None, :, None], np.arange(8)[None, None, :]]
                selected_support = self.fuel_support_[indices, np.arange(8)[None, :]]
            else:
                indices = np.repeat(np.searchsorted(cdf, uniforms[:, 0], side="right")[:, None], 8, axis=1)
                q = self.q_[indices[:, 0]]
                selected_support = self.fuel_support_[indices[:, 0]]
            indices_all[i] = indices
            sampled_mass[i] = (1 - selected_support) @ mix[i]
            if mode == "direct_analog":
                ensemble[i] = self.raw_cif_[indices[:, 0]]
            else:
                ensemble[i], zero = recompose_paths(mix[i], q, return_zero_total=True)
                zero_fraction[i] = zero.mean()
        posterior_region = np.column_stack([weights_all[:, self.regions_ == region].sum(axis=1)
                                             for region in self.source_regions_])
        prior_region = np.column_stack([prior_all[:, self.regions_ == region].sum(axis=1)
                                         for region in self.source_regions_])
        return {"point": ensemble.mean(axis=1), "ensemble": ensemble,
            "donor_weights": weights_all, "donor_prior": prior_all,
            "region_weights": posterior_region, "region_prior": prior_region,
            "weighted_fuel_support": support, "unsupported_target_mass": np.sum(mix * (1 - support), axis=1),
            "sampled_unsupported_target_mass": sampled_mass, "sampled_donor_indices": indices_all,
            "effective_donors": 1 / np.sum(weights_all**2, axis=1),
            "zero_generation_fallback_fraction": zero_fraction,
            "weighted_calendar_distance": season_distance, "weighted_hellinger_distance": mix_distance,
            "same_calendar_month_mass": same_month_mass,
            "calendar_month_unseen": ~np.isin(origins.month, self.months_),
            "source_regions": self.source_regions_, "mode": mode, "bandwidth": float(bandwidth),
            "seed": int(seed), "exclude_region": exclude_region,
            "point_estimator": "ratio_of_weighted_mean_q_plugin" if mode == "mean_profile" else "uniform_finite_ensemble_mean",
            "seed_policy": "UTC origin nanoseconds plus seed; batch-order invariant; no target labels",
            "source_fit": deepcopy(self.audit_)}
