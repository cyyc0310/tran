"""Analytic/type/cohort negative tests for the V193 assessor; no data or models."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from transcif.research.assessment_v193 import (
    ArtifactKind, ForecastArtifact, InformationContract, summarize_selection, validate_cohorts, validate_scoring_masks)


def run_checks() -> dict:
    c = InformationContract('T', '2024-01-01T00:00Z', '2023-12-31T00:00Z',
                            '2023-12-01T00:00Z', ('S',), horizon=1)
    negative = []

    def rejects(name, call):
        try:
            call()
        except ValueError:
            negative.append(name)
        else:
            raise AssertionError('negative test not rejected: ' + name)

    for name, changes in [
        ('future_release', {'available_at': '2024-01-02T00:00Z'}),
        ('future_period', {'period_end': '2024-01-02T00:00Z'}),
        ('target_in_sources', {'sources': ('S', 'T')}),
        ('target_in_calibration', {'calibration_regions': ('T',), 'calibration_kind': 'source_residual_transfer',
                                   'calibration_start': '2023-01-01T00:00Z', 'calibration_end': '2024-01-01T00:00Z'}),
        ('region_split_overlap', {'calibration_regions': ('S',), 'calibration_kind': 'region_split',
                                 'calibration_start': '2023-01-01T00:00Z', 'calibration_end': '2024-01-01T00:00Z'}),
        ('temporal_split_overlap', {'calibration_regions': ('S',), 'calibration_kind': 'temporal_split',
                                   'fit_end': '2023-06-01T00:00Z', 'calibration_start': '2023-01-01T00:00Z',
                                   'calibration_end': '2024-01-01T00:00Z'}),
    ]:
        rejects(name, lambda changes=changes: replace(c, **changes).validate())
    rejects('identified_as_90pct', lambda: ForecastArtifact(ArtifactKind.IDENTIFIED, c, .1, lower=np.array([0]), upper=np.array([2]), assumptions=('toy',)))
    rejects('interval_as_CDF', lambda: ForecastArtifact(ArtifactKind.INTERVAL, c, .1, predictions=np.array([[1]]), weights=np.array([1]), lower=np.array([0]), upper=np.array([2])))
    rejects('negative_weight', lambda: ForecastArtifact(ArtifactKind.DISTRIBUTION, c, .1, predictions=np.array([[0], [2]]), weights=np.array([-1, 2])))
    rejects('nonfinite_prediction', lambda: ForecastArtifact(ArtifactKind.DISTRIBUTION, c, .1, predictions=np.array([[np.nan]]), weights=np.array([1])))
    rejects('different_cohort', lambda: validate_cohorts({'a': [c], 'b': [replace(c, region='U')]}))
    rejects('duplicate_cohort', lambda: validate_cohorts({'a': [c, c]}))
    for field, value in [('available_at', '2023-12-30T00:00Z'), ('period_end', '2023-11-01T00:00Z'),
                         ('sources', ('OTHER',)), ('fit_end', '2023-01-01T00:00Z')]:
        rejects('same_key_different_' + field,
                lambda field=field, value=value: validate_cohorts({'a': [c], 'b': [replace(c, **{field: value})]}))
    policy = {'a': ('none', 'empty', None, None), 'b': ('none', 'empty', None, None)}
    rejects('policy_cannot_except_available_at', lambda: validate_cohorts(
        {'a': [c], 'b': [replace(c, available_at='2023-12-30T00:00Z')]}, policy))
    calibrated = replace(c, calibration_kind='source_residual_transfer', calibration_regions=('S',),
                         calibration_start='2023-01-01T00:00Z', calibration_end='2024-01-01T00:00Z')
    rejects('undeclared_calibration_change', lambda: validate_cohorts({'a': [c], 'b': [calibrated]}, policy))
    allowed = {'a': policy['a'], 'b': ('source_residual_transfer', 'sources',
                                    '2023-01-01T00:00Z', '2024-01-01T00:00Z')}
    validate_cohorts({'a': [c], 'b': [calibrated]}, allowed)
    rejects('calibration_region_not_in_policy', lambda: validate_cohorts(
        {'a': [c], 'b': [replace(calibrated, calibration_regions=('OTHER',))]}, allowed))

    # Interval-only artifacts cannot supply a CRPS; identified sets have no nominal scores.
    interval = ForecastArtifact(ArtifactKind.INTERVAL, c, .1, lower=np.array([0]), upper=np.array([2]))
    score = interval.score(np.array([3]))
    assert score['winkler90'] == 22 and score['crps'] is None
    identified = ForecastArtifact(ArtifactKind.IDENTIFIED, c, None, lower=np.array([0]), upper=np.array([2]), assumptions=('toy',))
    assert identified.score(np.array([1]))['coverage90'] is None
    rng = np.random.default_rng(193);max_error = 0.
    for _ in range(40):
        x = np.round(rng.normal(size=11), 1);w = rng.random(11);w[0] = 0;w /= w.sum();y = float(rng.normal())
        dist = ForecastArtifact(ArtifactKind.DISTRIBUTION, c, .1, predictions=x[:, None], weights=w)
        z = np.unique(np.r_[x, y]);mid = (z[:-1] + z[1:]) / 2
        F = np.asarray([w[x <= t].sum() for t in mid]);indicator = mid >= y
        integral = float(np.sum(np.diff(z) * (F - indicator) ** 2))
        max_error = max(max_error, abs(dist.score(np.array([y]))['crps'] - integral))
    if max_error > 1e-12:
        raise ValueError('independent toy CDF integral')
    empty = pd.DataFrame([{'region': 'T', 'origin': c.origin, 'status': 'scored', 'hours': 1,
                           'predictive_width': 100., 'mae': 1., 'crps': 1., 'coverage90': 0.,
                           'width90': 100., 'winkler90': 110.}])
    masks = pd.DataFrame([{'method': 'a', 'region': 'T', 'origin': c.origin, 'status': 'scored', 'hours': 24},
                          {'method': 'b', 'region': 'T', 'origin': c.origin, 'status': 'missing_label', 'hours': 0}])
    rejects('different_scoring_mask', lambda: validate_scoring_masks(masks))
    no_accept = summarize_selection(empty, 0)
    assert no_accept['accepted_fixed_region_conditioned'] is None and no_accept['status'] == 'no_accepted_labels'
    return {'negative_cases_rejected': negative, 'analytic_winkler_passed': True,
            'interval_CRPS_None': True, 'identified_nominal_metrics_None': True,
            'empty_selection_None': True, 'toy_CDF_cases': 40, 'max_toy_CDF_error': max_error}
