"""Pure annual POS-composition context controls over a sealed V25 MG stream.

No file I/O or fitting occurs here. Annual rows must be the hash-bound V24
complete twelve-record proxy years, supplied as CSV strings by the caller.
These controls do not conserve future realized annual or monthly energy.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from transcif.research.fuel_transport_v7 import recompose_paths

SOURCES = ('BPAT', 'CISO', 'ERCO', 'FPL', 'ISNE', 'MISO', 'NYIS')
FUELS = ('coal', 'gas', 'oil', 'nuclear', 'hydro', 'wind', 'solar', 'other')
POS = np.array([0, 1, 2, 7])
ZERO = np.array([3, 4, 5, 6])
EF = np.array([1000., 400., 800., 0., 0., 0., 0., 400.])
ARMS = ('MG-recompose_conditional', 'AY-fixed_anchor', 'AY-fixed_recompose',
        'AY-residual_anchor', 'AY-residual_recompose')
SAMPLES = 128


@dataclass(frozen=True)
class AnnualContext:
    """Latest legally released year; zero POS is distinct from an absent year."""

    region: str
    origin: str
    status: str
    ratio: tuple[float, float, float, float] | None = None
    year: int | None = None
    period_end: str | None = None
    available_at: str | None = None
    age_days: float | None = None
    release_age_days: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe provenance, without exposing any unused annual fuel fields."""
        return dict(region=self.region, origin=self.origin, status=self.status,
            ratio=None if self.ratio is None else list(self.ratio), year=self.year,
            period_end=self.period_end, available_at=self.available_at,
            age_days=self.age_days, release_age_days=self.release_age_days)


def _utc(value: Any, label: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError('Explicit finite timezone required: '+label)
    return stamp.tz_convert('UTC')


def select_annual_context(rows: Sequence[Mapping[str, str]], region: str,
                          origin: pd.Timestamp) -> AnnualContext:
    """Select the latest unique released complete year before parsing its masses.

    Non-eight schemas and other regions are ignored before timestamp/numeric
    parsing. Future period ends or releases never have fuel values converted.
    The latest zero-POS year is retained as ``zero_pos``; an older positive year
    is not substituted. Complete-year status is inherited from pinned V24
    provenance and its ``unit=proxy_year`` record, not inferred from 8760 hours.
    """
    if region not in SOURCES:
        raise ValueError('Registered source region required')
    at = _utc(origin, 'context origin')
    eligible: dict[int, Mapping[str, str]] = {}
    times: dict[int, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for row in rows:
        if row.get('region') != region or row.get('schema') != 'eight':
            continue
        end = _utc(row['period_end'], 'annual end')
        release = _utc(row['available_at'], 'annual release')
        if end > at or release > at:
            continue
        start = _utc(row['period_start'], 'annual start')
        year = int(row['year'])
        if (row.get('unit') != 'proxy_year' or start != pd.Timestamp(year=year, month=1, day=1, tz='UTC')
                or end != pd.Timestamp(year=year+1, month=1, day=1, tz='UTC') or release < end):
            raise ValueError('Complete calendar-year proxy with legal release required')
        if year in eligible:
            raise ValueError('Duplicate eligible source annual year/vintage')
        eligible[year] = row
        times[year] = (end, release)
    if not eligible:
        return AnnualContext(region, at.isoformat(), 'missing')
    year = max(eligible); row = eligible[year]; end, release = times[year]
    masses = np.array([float(row['lower_mass_'+fuel]) for fuel in FUELS])
    if (not np.isfinite(masses).all() or np.any(masses < 0)
            or not np.isclose(masses.sum(), 1., atol=1e-12, rtol=0)):
        raise ValueError('Finite nonnegative annual eight-fuel simplex required')
    positive = float(masses[POS].sum())
    ratio = None if positive == 0 else tuple(float(x) for x in masses[POS]/positive)
    return AnnualContext(region, at.isoformat(), 'zero_pos' if positive == 0 else 'available', ratio,
        year, end.isoformat(), release.isoformat(), float((at-end)/pd.Timedelta(days=1)),
        float((at-release)/pd.Timedelta(days=1)))


def project_simplex(values: np.ndarray) -> np.ndarray:
    """Euclidean projection of finite (...,4) residual-adjusted POS ratios."""
    x = np.asarray(values, float)
    if x.ndim < 1 or x.shape[-1] != 4 or not np.isfinite(x).all():
        raise ValueError('Finite four-coordinate values required')
    flat = x.reshape(-1, 4)
    ordered = np.sort(flat, axis=1)[:, ::-1]
    candidates = (np.cumsum(ordered, axis=1)-1)/np.arange(1, 5)
    count = (ordered > candidates).sum(1)-1
    threshold = candidates[np.arange(len(flat)), count]
    result = np.maximum(flat-threshold[:, None], 0)
    result /= result.sum(1)[:, None]
    return result.reshape(x.shape)


def _simplex(x: np.ndarray, dimension: int, label: str) -> None:
    if (x.shape[-1] != dimension or not np.isfinite(x).all() or np.any(x < 0)
            or not np.allclose(x.sum(-1), 1., atol=1e-12, rtol=0)):
        raise ValueError('Finite simplex required: '+label)


def _inputs(bank: Mapping[str, np.ndarray], baseline: Mapping[str, np.ndarray],
            mg_draws: np.ndarray, dates: pd.DatetimeIndex, region: str) -> None:
    n = len(bank['days']); t = len(dates)
    if (region not in SOURCES or not t or dates.hasnans or not dates.equals(dates.normalize())
            or dates.has_duplicates or np.any(dates < pd.Timestamp('2024-07-01', tz='UTC'))):
        raise ValueError('Unique complete deployed source days required')
    expected_regions = set(SOURCES)-{region}
    if set(bank['regions']) != expected_regions:
        raise ValueError('Exactly six proper sources excluding query region required')
    shapes = dict(q=(n, 24, 8), donor_mix=(n, 8), raw_cif=(n, 24), support=(n, 8), regions=(n,), days=(n,))
    if any(np.asarray(bank[k]).shape != shape for k, shape in shapes.items()):
        raise ValueError('Invalid proper bank shape')
    if not np.issubdtype(bank['days'].dtype, np.integer):
        raise ValueError('Bank day timestamps must be UTC integer nanoseconds')
    source_dates = pd.DatetimeIndex(pd.to_datetime(bank['days'], utc=True))
    if (source_dates.hasnans or not source_dates.equals(source_dates.normalize())
            or np.any(source_dates < pd.Timestamp('2024-01-25', tz='UTC'))
            or np.any(source_dates >= pd.Timestamp('2024-07-01', tz='UTC'))):
        raise ValueError('Fixed proper H1 source dates required')
    if bank['support'].dtype != np.bool_:
        raise ValueError('Boolean observed fuel support required')
    if (not np.isfinite(bank['q']).all() or np.any(bank['q'] < 0)
            or np.any(bank['q'][~np.broadcast_to(bank['support'][:, None], bank['q'].shape)] != 1)
            or not np.array_equal(bank['donor_mix'] > 0, bank['support'])
            or not np.isfinite(bank['raw_cif']).all() or np.any(bank['raw_cif'] < 0) or np.any(bank['raw_cif'] > 1000)):
        raise ValueError('Valid bank trajectories/support/raw CIF required')
    _simplex(bank['donor_mix'], 8, 'bank monthly composition')
    if mg_draws.shape != (t, SAMPLES, 5):
        raise ValueError('Fixed128 MG draw shape required')
    _simplex(mg_draws, 5, 'MG draws')
    for field in ('donor_ids', 'fallback_ids'):
        value = baseline[field]
        if value.shape != (t, SAMPLES) or not np.issubdtype(value.dtype, np.integer):
            raise ValueError('Integer shared sampler IDs required')
        lower = -1 if field == 'fallback_ids' else 0
        if np.any(value < lower) or np.any(value >= n):
            raise ValueError('Shared sampler ID out of range')
    if baseline['fine_draws'].shape != (t, SAMPLES, 8) or baseline['raw'].shape != (t, SAMPLES, 24):
        raise ValueError('Complete MG baseline fine/raw shape required')
    _simplex(baseline['fine_draws'], 8, 'baseline fine composition')
    if not np.isfinite(baseline['raw']).all():
        raise ValueError('Finite MG baseline paths required')


def _effective(bank: Mapping[str, np.ndarray], ids: np.ndarray, fallback: np.ndarray,
               positive_draw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    original_mass = bank['donor_mix'][ids][:, POS].sum(1)
    missing = (original_mass == 0) & (positive_draw > 0)
    if not np.array_equal(fallback >= 0, missing):
        raise ValueError('Baseline fallback IDs do not match unsupported positive POS draws')
    selected = np.where(fallback >= 0, fallback, ids)
    selected_mass = bank['donor_mix'][selected][:, POS].sum(1)
    if np.any((positive_draw > 0) & (selected_mass <= 0)):
        raise ValueError('Positive query POS requires a supported effective source row')
    ratio = np.zeros((SAMPLES, 4))
    supported = selected_mass > 0
    ratio[supported] = bank['donor_mix'][selected[supported]][:, POS]/selected_mass[supported, None]
    q = bank['q'][ids].copy(order='K'); support = bank['support'][ids].copy()
    for particle in np.flatnonzero(missing):
        q[particle][:, POS] = bank['q'][selected[particle]][:, POS]
        support[particle, POS] = bank['support'][selected[particle], POS]
    return q, support, selected, ratio


def _unsupported(fine: np.ndarray, q: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Fraction of modeled POS energy using unsupported (flat-q=1) fuel channels."""
    energy = (fine[:, None, POS]*q[:, :, POS]).sum(1)
    unsupported = (energy*~support[:, POS]).sum(1)
    total = energy.sum(1)
    return np.divide(unsupported, total, out=np.zeros_like(total), where=total > 0)


def predict(bank: Mapping[str, np.ndarray], baseline: Mapping[str, np.ndarray],
            mg_draws: np.ndarray, days: pd.DatetimeIndex,
            annual_rows: Sequence[Mapping[str, str]], *, region: str) -> dict[str, Any]:
    """Five matched context controls, preserving the original MG baseline exactly.

    For a borrowed POS trajectory, its donor-month ratio, source/date/year
    context and observed support all follow j; non-POS trajectories remain i.
    Missing/zero-POS target context preserves each original MG ratio in both
    AY streams. An unavailable source year gives zero residual and therefore
    the fixed target ratio, explicitly not the original-MG fallback. Undefined
    donor-month POS ratios only occur for zero query POS and likewise use zero
    residual; they are separately flagged and cannot affect any physical path.
    """
    dates = pd.DatetimeIndex([_utc(day, 'query day') for day in days])
    draws = np.asarray(mg_draws, float)
    _inputs(bank, baseline, draws, dates, region)
    contexts = [select_annual_context(annual_rows, region, day) for day in dates]
    donor_contexts = [select_annual_context(annual_rows, str(r), pd.Timestamp(int(d), tz='UTC'))
                      for r, d in zip(bank['regions'], bank['days'])]
    output = {arm: {key: [] for key in ('raw', 'fine_draws', 'ratios', 'unsupported_positive_energy_fraction', 'zero_total_mask')} for arm in ARMS}
    target_unavailable = np.zeros(len(dates), bool)
    donor_unavailable = np.zeros((len(dates), SAMPLES), bool)
    donor_month_zero = np.zeros_like(donor_unavailable)
    effective_ids = np.empty((len(dates), SAMPLES), np.int64)
    unprojected = np.empty((len(dates), SAMPLES, 4))
    projection_l1 = np.zeros((len(dates), SAMPLES))
    projection_boundary_count = np.zeros((len(dates), SAMPLES), np.int64)
    projection_applied = np.zeros((len(dates), SAMPLES), bool)
    baseline_error = 0.
    for t, (day, context) in enumerate(zip(dates, contexts)):
        ids, borrowed = baseline['donor_ids'][t], baseline['fallback_ids'][t]
        q, support, selected, original = _effective(bank, ids, borrowed, draws[t, :, 0])
        effective_ids[t] = selected
        original_fine = np.zeros((SAMPLES, 8)); original_fine[:, ZERO] = draws[t, :, 1:]
        original_fine[:, POS] = draws[t, :, :1]*original
        if not np.array_equal(baseline['fine_draws'][t], original_fine):
            raise ValueError('Baseline must use identical MG draws and effective donor-month ratios')
        original_raw, original_zero = recompose_paths(baseline['fine_draws'][t], q, return_zero_total=True)
        error = float(abs(original_raw-baseline['raw'][t]).max())
        baseline_error = max(baseline_error, error)
        # This paused draft requires the identical sealed stream, including its
        # reduction layout; a numerical tolerance would admit altered inputs.
        if not np.array_equal(original_raw, baseline['raw'][t]):
            raise ValueError('Original sealed MG baseline recomposition identity failed')
        fixed, residual = original.copy(), original.copy()
        unprojected[t] = original
        donor_month_zero[t] = bank['donor_mix'][selected][:, POS].sum(1) == 0
        donor_unavailable[t] = [donor_contexts[i].status != 'available' for i in selected]
        target_unavailable[t] = context.status != 'available'
        if not target_unavailable[t]:
            target = np.asarray(context.ratio)
            fixed[:] = target
            residual[:] = target
            unprojected[t] = target
            usable = ~donor_unavailable[t] & ~donor_month_zero[t]
            if usable.any():
                source_year = np.array([donor_contexts[i].ratio for i in selected[usable]], float)
                unprojected[t, usable] = target+original[usable]-source_year
                residual[usable] = project_simplex(unprojected[t, usable])
                projection_applied[t, usable] = True
                projection_l1[t, usable] = abs(residual[usable]-unprojected[t, usable]).sum(1)
                projection_boundary_count[t, usable] = (residual[usable] == 0).sum(1)
        variants = {ARMS[0]: (original, False), ARMS[1]: (fixed, True), ARMS[2]: (fixed, False),
                    ARMS[3]: (residual, True), ARMS[4]: (residual, False)}
        for arm, (ratio, anchor) in variants.items():
            fine = baseline['fine_draws'][t].copy()
            fine[:, POS] = draws[t, :, :1]*ratio
            if target_unavailable[t] or arm == ARMS[0]:
                # Preserve baseline inputs exactly in every original-ratio fallback.
                fine = baseline['fine_draws'][t].copy()
            physical, zero = recompose_paths(fine, q, return_zero_total=True)
            if arm == ARMS[0]:
                raw, zero = baseline['raw'][t].copy(), original_zero
            else:
                raw = bank['raw_cif'][ids]+((fine-bank['donor_mix'][ids])@EF)[:, None] if anchor else physical
            values = dict(raw=raw, fine_draws=fine, ratios=ratio.copy(),
                unsupported_positive_energy_fraction=_unsupported(fine, q, support), zero_total_mask=zero)
            for key, value in values.items():
                output[arm][key].append(value)
    arrays = {arm: {key: np.stack(value) for key, value in fields.items()} for arm, fields in output.items()}
    return dict(arms=arrays, donor_ids=baseline['donor_ids'].copy(), fallback_ids=baseline['fallback_ids'].copy(),
        effective_pos_ids=effective_ids, target_year=np.array([c.year if c.year is not None else -1 for c in contexts], np.int64),
        residual_unprojected_ratios=unprojected, residual_projection_l1=projection_l1,
        residual_projection_boundary_count=projection_boundary_count, residual_projection_applied=projection_applied,
        donor_year=np.array([c.year if c.year is not None else -1 for c in donor_contexts], np.int64),
        target_context_unavailable=target_unavailable, source_year_unavailable=donor_unavailable,
        donor_month_zero=donor_month_zero, audit=dict(arms=list(ARMS), region=region, samples=SAMPLES,
            target_contexts=[c.as_dict() for c in contexts], donor_contexts=[c.as_dict() for c in donor_contexts],
            original_mg_raw_preserved_exact=True, original_mg_fine_preserved_exact=True,
            baseline_recomposition_max_abs_error=baseline_error,
            residual_projection_max_l1=float(projection_l1.max()), residual_projection_mean_l1=float(projection_l1.mean()),
            residual_projected_particles=int(np.count_nonzero(projection_applied)),
            residual_boundary_coordinates=int(projection_boundary_count.sum()),
            target_missing_days=int(sum(c.status == 'missing' for c in contexts)),
            target_zero_pos_days=int(sum(c.status == 'zero_pos' for c in contexts)),
            residual_zero_delta_particles=int(np.count_nonzero(~target_unavailable[:, None] & (donor_unavailable | donor_month_zero))),
            target_original_ratio_particles=int(np.count_nonzero(target_unavailable)*SAMPLES),
            unsupported_fraction_definition='unsupported POS modeled energy / all POS modeled energy; zero if POS energy is zero',
            unsupported_q_assumption='flat one on observed-unsupported fuel channels, inherited source prior',
            source_missing_rule='zero residual -> target fixed ratio; distinct from target-missing original ratio',
            donor_month_zero_rule='zero residual; only legal when query POS draw is zero',
            zero_total_mask_definition='physical recomposition zero denominator; same diagnostic also reported for anchors',
            target_values_opened=False, unseen_values_opened=False, no_file_io=True,
            annual_or_monthly_realized_energy_conserved=False, custom_method=False))
