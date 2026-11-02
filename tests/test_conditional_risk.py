import numpy as np
import pandas as pd
import pytest

from transcif.research.capacity_transfer import COLUMNS
from transcif.research.conditional_risk import KINDS, bounds, fit, interval_metrics, risk_inputs


def test_conditional_risk_accepts_no_hourly_target_labels_and_survives_round_trip(tmp_path):
    import joblib

    rng = np.random.default_rng(930)
    train = pd.DataFrame(rng.normal(size=(240, len(COLUMNS))), columns=COLUMNS)
    train['country'], train['region'] = 'SOURCE', 'A'
    cal = train.copy()
    point = np.repeat(300., 240)
    fitted = fit(train, rng.normal(size=240), point, cal, rng.normal(size=240), point)
    path = tmp_path / 'trusted-local-reference.joblib'
    joblib.dump(fitted, path)
    restored = joblib.load(path)
    public = train.drop(columns=['country', 'region'])
    before, after = bounds(fitted, public, point), bounds(restored, public, point)
    assert set(before) == set(KINDS)
    for kind in KINDS:
        for a, b in zip(before[kind], after[kind]):
            np.testing.assert_array_equal(a, b)
        assert (before[kind][0] <= before[kind][1]).all()
    assert risk_inputs(public, point).shape == (240, len(COLUMNS) + 1)


def test_common_support_clipping_does_not_improve_coverage_by_widening():
    y = np.array([0., 100., 900., 1000.])
    lo, hi = np.array([-20., -10., 850., 800.]), np.array([20., 800., 1100., 1200.])
    raw = interval_metrics(y, lo, hi)
    clipped = interval_metrics(y, np.clip(lo, 0., 1000.), np.clip(hi, 0., 1000.))
    assert clipped['coverage90'] == raw['coverage90']
    assert clipped['width90'] < raw['width90']
    with pytest.raises(ValueError, match='invalid'):
        interval_metrics(y, hi, lo)


def test_endpoint_projection_can_add_coverage_at_support_boundaries():
    # Endpoint projection is not set intersection: a wholly out-of-support
    # interval becomes a degenerate boundary interval, rather than empty.
    y = np.array([0., 1000.])
    lo, hi = np.array([-20., 1010.]), np.array([-10., 1020.])
    raw = interval_metrics(y, lo, hi)
    projected = interval_metrics(y, np.clip(lo, 0., 1000.), np.clip(hi, 0., 1000.))
    assert raw['coverage90'] == 0. and projected['coverage90'] == 1.
    assert raw['width90'] == 10. and projected['width90'] == 0.
