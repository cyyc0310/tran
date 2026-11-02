"""Capacity inputs obey availability and cannot require target generation labels."""
import json

import numpy as np
import pandas as pd
import pytest

from transcif.research import capacity_transfer as capacity
from transcif.research.multicountry import FINE, GROUPS, fit, public_features


def test_zero_stock_exclusion_preserves_country_balance_for_each_fuel():
    frame = pd.DataFrame({'country': ['A', 'A', 'A', 'B'], 'region': ['A1', 'A2', 'A3', 'B1'],
                          'capacity_coal_mw': [0., 1., 1., 1.], 'capacity_gas_mw': [1.] * 4})
    # Country A's two eligible coal regions average 2; B averages 6.
    # A's excluded stock entry must not change country A's overall mass.
    fuel = np.array([[0., 2.], [1., 2.], [3., 2.], [6., 6.]])
    np.testing.assert_allclose(capacity.nominal_rates(frame, fuel), [4., 4.])


def test_capacity_availability_gate_rejects_future_or_undated_metadata(tmp_path):
    path = tmp_path / 'capacity.json'
    for date in ['2022-02-01T00:00:00Z', '2021-06-03']:
        path.write_text(json.dumps({'regions': {'R': {'coal_mw': 1., 'gas_mw': 2., 'available_by_utc': date}}}))
        with pytest.raises(ValueError, match='unavailable'):
            capacity.load_capacity(path)


def test_public_capacity_prediction_without_target_hourly_labels():
    clock = pd.Series(pd.date_range('2022-04-01', periods=240, freq='h'))
    months = pd.DataFrame([[60, 10, 10, 10, 10, 0]] * 3, columns=GROUPS,
                          index=pd.period_range('2022-01', periods=3, freq='M'))
    public = public_features(months, clock)
    public['country'], public['region'] = 'S', 'R'
    table = {'regions': {'R': {'coal_mw': 25., 'gas_mw': 75.}}}
    training = capacity.add_capacity(public, table)
    fuel = np.tile([15., 45., 0., 0., 10., 10., 10., 10., 0.], (len(clock), 1))
    assert fuel.shape[1] == len(FINE)
    coarse = np.tile([60., 10., 10., 10., 10., 0.], (len(clock), 1))
    reference = fit(training, fuel, coarse)
    model = capacity.fit(training, fuel, reference)
    predictions = capacity.predict(model, training)
    assert set(predictions) == set(capacity.METHODS)
    assert all(np.isfinite(v).all() for v in predictions.values())
    changed = capacity.add_capacity(public, {'regions': {'R': {'coal_mw': 75., 'gas_mw': 25.}}})
    new = capacity.predict(model, changed)
    assert np.all(new['capacity_anchor_flat'] > predictions['capacity_anchor_flat'])
