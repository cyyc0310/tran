import csv
import numpy as np
import pandas as pd
import pytest

from scripts import disjoint_calibration_v50 as m


def test_label_release_uses_end_of_day_and_fixed_snapshot():
    days = pd.DatetimeIndex(['2024-10-02', '2024-10-03', '2024-12-01', '2024-12-02',
                             '2024-12-24'], tz='UTC')
    assert m.eligible(days, 90).tolist() == [True, False, False, False, False]
    assert m.eligible(days, 30).tolist() == [True, True, True, False, False]
    assert m.eligible(days, 7).all()
    with pytest.raises(ValueError):
        m.eligible(pd.DatetimeIndex(['2024-12-25'], tz='UTC'), 7)
    with pytest.raises(ValueError):
        m.eligible(days, 0)


def _csv(path, days):
    fields = ['time', 'region', 'valid']+['generation_'+f for f in m.FUELS]
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fields); writer.writeheader()
        for day, valid, value in days:
            for at in pd.date_range(day, periods=24, freq='h', tz='UTC'):
                writer.writerow(dict(time=at.isoformat(), region='SWPP', valid=valid,
                    **{name: ('1' if fuel == 'gas' else '0') if value == 'gas' else value
                       for name, fuel in zip(fields[3:], m.FUELS)}))


def test_metadata_and_truth_never_convert_future_or_invalid_generation(tmp_path):
    path = tmp_path/'SWPP_hourly.csv'
    _csv(path, [('2024-07-01', True, 'gas'), ('2024-07-02', False, 'POISON'),
                ('2025-01-01', True, 'POISON')])
    keys = m.metadata(path, 'SWPP')
    assert keys == ['SWPP/2024-07-01']
    positions, truth = m.external_truth(path, 'SWPP', keys)
    assert positions.tolist() == [0]
    np.testing.assert_array_equal(truth, np.full((1, 24), 400.))
    with pytest.raises(ValueError):
        m.external_truth(path, 'SWPP', [])
    with pytest.raises(ValueError):
        m.metadata(path, 'BPAT')


def test_metadata_can_contain_only_unconvertible_generation(tmp_path):
    path = tmp_path/'SWPP_hourly.csv'
    _csv(path, [('2024-07-01', True, 'POISON')])
    assert m.metadata(path, 'SWPP') == ['SWPP/2024-07-01']
    with pytest.raises(ValueError):
        m.external_truth(path, 'SWPP', ['SWPP/2024-07-01'])


def test_feature_budget_and_postclip_statistics():
    raw = np.zeros((5, 2, 128, 24)); raw[:, :, 0] = 2000
    features = {'Size': np.zeros((2, 128, 31)),
                'SizeCap': np.concatenate((np.zeros((2, 128, 31)), np.ones((2, 128, 5))), -1)}
    result = m.feature_packet(raw, features)
    np.testing.assert_array_equal(result['point'], np.full((5, 2, 24), 1000/128))
    np.testing.assert_array_equal(result['std_shape'], np.zeros((5, 2)))
    assert result['X'][0].shape == (2, 24, 40)
    assert result['X'][4].shape == (2, 24, 45)
    b0 = m.feature_packet(raw[:3], {'Size': features['Size']})
    for i in range(3):
        np.testing.assert_array_equal(result['X'][i], b0['X'][i])
    features['SizeCap'][..., 31:] *= 123
    changed = m.feature_packet(raw, features)
    for i in range(3):
        np.testing.assert_array_equal(result['X'][i], changed['X'][i])
    assert not np.array_equal(result['X'][4], changed['X'][4])
    with pytest.raises(ValueError):
        m.feature_packet(raw.astype(complex)+1j, features)


def test_interval_metrics_are_hourly_and_separately_pathwise():
    lo = np.full((2, 3, 24), 100.); hi = lo+200
    y = np.full((3, 24), 200.); y[0, 0] = 400
    result = m.interval_metrics(lo, hi, y)
    assert result['coverage90'][0, 0] == 23/24
    assert result['path_coverage90'][0, 0] == 0
    assert result['interval_score90'][0, 0] == 200+2000/24
    assert result['upper_miss'][0, 0] == 1/24
    with pytest.raises(ValueError):
        m.interval_metrics(hi, lo, y)


def test_reference_intervals_replay_all_five_frozen_spread_conventions():
    from scripts.native_center_v18 import scaled_paths
    raw = np.linspace(-200, 1200, 128)[None, None, :, None]*np.ones((3, 1, 1, 24))
    for factor in m.FACTORS:
        lo, hi = m.raw_intervals(raw, factor)
        reference = np.stack([np.sort(scaled_paths(base, factor)[1], axis=1) for base in raw])
        np.testing.assert_array_equal(lo, reference[:, :, 6])
        np.testing.assert_array_equal(hi, reference[:, :, 121])
    with pytest.raises(ValueError):
        m.raw_intervals(raw, 1.1)
