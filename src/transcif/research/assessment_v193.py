"""Typed, same-cohort CIF artifact assessment; no new coverage guarantees."""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np
import pandas as pd

from .weighted_metrics import PreparedDistribution


class ArtifactKind(str, Enum):
    DISTRIBUTION = 'predictive_distribution'
    INTERVAL = 'probability_interval'
    IDENTIFIED = 'identified_set'


def timestamp(value: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError('explicit timezone required')
    return result.tz_convert('UTC')


@dataclass(frozen=True)
class InformationContract:
    region: str
    origin: str
    available_at: str
    period_end: str
    sources: tuple[str, ...]
    calibration_regions: tuple[str, ...] = ()
    calibration_kind: str = 'none'
    fit_end: str | None = None
    calibration_start: str | None = None
    calibration_end: str | None = None
    horizon: int = 24
    track: str = 'controlled_month_only'

    def validate(self) -> None:
        origin = timestamp(self.origin)
        if timestamp(self.available_at) > origin or timestamp(self.period_end) > origin:
            raise ValueError('future statistic input')
        if self.horizon < 1 or not self.region or self.track != 'controlled_month_only':
            raise ValueError('explicit supported information track/horizon required')
        if len(set(self.sources)) != len(self.sources) or len(set(self.calibration_regions)) != len(self.calibration_regions):
            raise ValueError('duplicate role regions')
        if self.region in self.sources or self.region in self.calibration_regions:
            raise ValueError('target/source or target/calibration overlap')
        if self.fit_end is not None and timestamp(self.fit_end) > origin:
            raise ValueError('future fitted labels')
        allowed = ('none', 'source_residual_transfer', 'region_split', 'temporal_split')
        if self.calibration_kind not in allowed:
            raise ValueError('unsupported calibration claim')
        if self.calibration_kind == 'none':
            if self.calibration_regions or self.calibration_start or self.calibration_end:
                raise ValueError('none calibration has populated roles')
        else:
            if not self.calibration_regions or self.calibration_start is None or self.calibration_end is None:
                raise ValueError('incomplete calibration provenance')
            start, end = timestamp(self.calibration_start), timestamp(self.calibration_end)
            if start >= end or end > origin:
                raise ValueError('invalid/future calibration interval')
            if self.calibration_kind == 'region_split' and set(self.sources) & set(self.calibration_regions):
                raise ValueError('region-split calibration overlaps fitted regions')
            if self.calibration_kind == 'temporal_split' and (self.fit_end is None or timestamp(self.fit_end) > start):
                raise ValueError('temporal-split calibration overlaps fit')

    @property
    def key(self) -> tuple[str, str, int]:
        return self.region, timestamp(self.origin).isoformat(), self.horizon


@dataclass
class ForecastArtifact:
    kind: ArtifactKind
    contract: InformationContract
    nominal_alpha: float | None
    predictions: np.ndarray | None = None
    weights: np.ndarray | None = None
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.contract.validate()
        if not isinstance(self.kind, ArtifactKind):
            raise ValueError('explicit artifact enum required')
        if self.kind == ArtifactKind.IDENTIFIED:
            if self.nominal_alpha is not None or not self.assumptions:
                raise ValueError('identified set has assumptions, never nominal coverage')
        elif self.nominal_alpha != .1:
            raise ValueError('this assessor explicitly implements central nominal90 only')
        if self.kind == ArtifactKind.DISTRIBUTION:
            if self.lower is not None or self.upper is not None or self.predictions is None or self.weights is None:
                raise ValueError('distribution requires only finite weighted particles')
            self.prepared = PreparedDistribution(self.predictions, self.weights)
            if self.prepared.predictions.shape[1] != self.contract.horizon:
                raise ValueError('distribution horizon mismatch')
            self.lower, self.upper = self.prepared.lower.copy(), self.prepared.upper.copy()
        else:
            if self.predictions is not None or self.weights is not None or self.lower is None or self.upper is None:
                raise ValueError('bounds do not define a predictive distribution')
            self.lower, self.upper = np.array(self.lower, float, copy=True), np.array(self.upper, float, copy=True)
        if (self.lower.shape != (self.contract.horizon,) or self.upper.shape != self.lower.shape
                or not np.isfinite(self.lower).all() or not np.isfinite(self.upper).all()
                or np.any(self.lower > self.upper)):
            raise ValueError('invalid bounds')
        self.lower.setflags(write=False);self.upper.setflags(write=False)

    @property
    def width(self) -> float:
        return float(np.mean(self.upper - self.lower))

    def score(self, truth: np.ndarray) -> dict:
        truth = np.asarray(truth, float)
        if truth.shape != self.lower.shape or not np.isfinite(truth).all():
            raise ValueError('truth shape/nonfinite')
        contains = (truth >= self.lower) & (truth <= self.upper)
        if self.kind == ArtifactKind.IDENTIFIED:
            return {'containment': float(contains.mean()), 'set_width': self.width,
                    'coverage90': None, 'width90': None, 'winkler90': None, 'crps': None, 'mae': None}
        winkler = self.upper - self.lower + 20 * np.maximum(self.lower - truth, 0) + 20 * np.maximum(truth - self.upper, 0)
        result = {'coverage90': float(contains.mean()), 'width90': self.width,
                  'winkler90': float(winkler.mean()), 'crps': None, 'mae': None}
        if self.kind == ArtifactKind.DISTRIBUTION:
            result.update(self.prepared.score(truth))
        return result


def validate_cohorts(cohorts: dict[str, list[InformationContract]],
                     calibration_policies: dict[str, tuple] | None = None) -> None:
    """Same keys AND full contracts; explicit calibration-only frozen exceptions.

    A policy tuple is (kind, regions_rule, start, end). The only regions rules
    here are empty or the SAME already-validated fitted-source list. No input
    availability, source, fit, track, or horizon field can be excepted.
    """
    if calibration_policies is not None and set(calibration_policies) != set(cohorts):
        raise ValueError('method policy key mismatch')
    expected = None
    for method, contracts in cohorts.items():
        keyed = {}
        for contract in contracts:
            contract.validate()
            key = contract.key
            if key in keyed:
                raise ValueError('duplicate forecast cohort')
            compare = contract
            if calibration_policies is not None:
                kind, regions_rule, start, end = calibration_policies[method]
                if regions_rule not in ('empty', 'sources'):
                    raise ValueError('unsupported calibration region exception')
                regions = () if regions_rule == 'empty' else contract.sources
                actual = (contract.calibration_kind, contract.calibration_regions,
                          contract.calibration_start, contract.calibration_end)
                if actual != (kind, regions, start, end):
                    raise ValueError('undeclared method calibration difference')
                compare = replace(contract, calibration_kind='none', calibration_regions=(),
                                  calibration_start=None, calibration_end=None)
            keyed[key] = compare
        if not keyed:
            raise ValueError('empty forecast cohort')
        if expected is not None and keyed != expected:
            raise ValueError('different forecast cohort or information contract')
        expected = keyed
    if expected is None:
        raise ValueError('no cohorts')


def validate_scoring_masks(frame: pd.DataFrame) -> None:
    """Every method must retain the same windows, availability and label hours."""
    expected = None
    for _, group in frame.groupby('method'):
        if group.duplicated(['region', 'origin']).any():
            raise ValueError('duplicate scoring cohort')
        mask = group[['region', 'origin', 'status', 'hours']].sort_values(['region', 'origin']).reset_index(drop=True)
        if ((mask.status.eq('scored') & mask.hours.le(0))
                | (~mask.status.eq('scored') & mask.hours.ne(0))).any():
            raise ValueError('inconsistent label hours/status')
        if expected is not None and not mask.equals(expected):
            raise ValueError('different scoring cohort/mask')
        expected = mask
    if expected is None:
        raise ValueError('empty scoring cohort')


def summarize_selection(frame: pd.DataFrame, width_limit: float) -> dict:
    """Fixed-region weights conditioned on acceptance, with explicit denominators."""
    if len(frame) == 0 or not np.isfinite(width_limit) or width_limit < 0:
        raise ValueError('nonempty population and fixed finite width limit required')
    if frame.duplicated(['region', 'origin']).any():
        raise ValueError('duplicate evaluation keys')
    metrics = ('mae', 'crps', 'coverage90', 'width90', 'winkler90')
    full, accepted_mass, selected_sums, conditional = [], [], [], []
    total_hours, chosen_hours, chosen_windows, evaluated = 0, 0, 0, 0
    for _, region in frame.groupby('region', sort=True):
        valid = region.loc[region.status.eq('scored')]
        n = float(valid.hours.sum())
        if n == 0:
            raise ValueError('entire region without labels: do not silently change macro cohort')
        if not np.isfinite(valid[list(metrics)]).all().all():
            raise ValueError('nonfinite scored metrics')
        select = valid.loc[valid.predictive_width.le(width_limit)]
        m = float(select.hours.sum())
        full.append((valid[list(metrics)].mul(valid.hours, axis=0).sum() / n).to_numpy())
        accepted_mass.append(m / n)
        selected_sums.append((select[list(metrics)].mul(select.hours, axis=0).sum() / n).to_numpy())
        if m > 0:
            conditional.append((select[list(metrics)].mul(select.hours, axis=0).sum() / m).to_numpy())
        total_hours += int(n);chosen_hours += int(m);chosen_windows += len(select);evaluated += len(valid)
    mass = float(np.mean(accepted_mass))
    selected = np.mean(selected_sums, axis=0) / mass if mass > 0 else None
    conditional_mean = np.mean(conditional, axis=0) if conditional else None
    return {'width_limit': float(width_limit), 'forecast_windows': len(frame), 'scored_windows': evaluated,
            'missing_windows': len(frame) - evaluated, 'scored_hours': total_hours,
            'accepted_scored_windows': chosen_windows, 'accepted_scored_hours': chosen_hours,
            'regions': len(full), 'regions_with_accepted_labels': len(conditional),
            'fixed_region_weight_acceptance_rate': mass,
            'forecast_acceptance_rate': float(frame.predictive_width.le(width_limit).mean()),
            'all': dict(zip(metrics, np.mean(full, axis=0).tolist())),
            'accepted_fixed_region_conditioned': dict(zip(metrics, selected.tolist())) if selected is not None else None,
            'accepted_nonempty_region_macro': dict(zip(metrics, conditional_mean.tolist())) if conditional_mean is not None else None,
            'status': 'scored' if mass > 0 else 'no_accepted_labels'}
